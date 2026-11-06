import os
import shutil
import tempfile
import time
import wave
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from media.ffmpeg import FFmpegExecutionError, get_ffmpeg_path, run_ffmpeg_command
from media.service import MediaService
from provenance.service import ProvenanceService
from watermark.audio.base import BaseAudioWatermarkEngine, AudioDetectionResult, AudioEmbeddingResult
from watermark.audio.config import AUDIO_SAMPLE_RATE


def compute_watermark_correlation(
    samples: np.ndarray,
    registered_tokens: List[str],
) -> Tuple[Optional[str], float]:
    """Computes normalized spread-spectrum watermark correlation across candidate tokens.

    Uses vectorized PRNG sequence mapping and SIMD dot products while preserving exact
    random sequence alignment and threshold semantics.
    """
    if len(samples) < 100 or not np.all(np.isfinite(samples)):
        return None, 0.0

    # Deduplicate and pre-filter valid tokens
    seen = set()
    token_pairs: List[Tuple[str, int]] = []
    for tok_str in registered_tokens:
        if tok_str not in seen:
            seen.add(tok_str)
            try:
                token_pairs.append((tok_str, int(tok_str, 16)))
            except ValueError:
                continue

    if not token_pairs:
        return None, 0.0

    n_samples = len(samples)
    samples_f64 = samples if samples.dtype == np.float64 else samples.astype(np.float64)

    best_token: Optional[str] = None
    best_score: float = 0.0

    for tok_str, tok_int in token_pairs:
        rng = np.random.RandomState(seed=tok_int)
        raw_bits = rng.randint(0, 2, size=n_samples)
        ref_seq = np.where(raw_bits, 1.0, -1.0)

        score = float(np.dot(samples_f64, ref_seq) / n_samples)
        if score > best_score:
            best_score = score
            best_token = tok_str

    return best_token, best_score


class CustomAudioWatermarkEngine(BaseAudioWatermarkEngine):
    """MediaProof Custom Deterministic Spread-Spectrum Audio Watermark Baseline Engine."""

    @property
    def engine_name(self) -> str:
        return "custom"

    @property
    def display_name(self) -> str:
        return "MediaProof Custom Spread-Spectrum Audio Watermark Baseline"

    def embed(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        asset_id: str,
        timeout: float = 60.0,
    ) -> AudioEmbeddingResult:
        start_time = time.time()
        in_p = Path(input_path).resolve()
        out_p = Path(output_path).resolve()

        if not in_p.exists() or not in_p.is_file():
            raise FileNotFoundError(f"Input audio file not found: {in_p}")

        ffmpeg_exe = get_ffmpeg_path()
        if not ffmpeg_exe or not ffmpeg_exe.exists():
            raise RuntimeError("FFmpeg executable is not available for audio watermarking.")

        probe_res = MediaService.probe_file(in_p)
        if not probe_res.has_audio:
            raise ValueError(f"File '{in_p.name}' does not contain an audio stream.")

        watermark_tok = ProvenanceService.ensure_watermark_token(asset_id)
        token_int = int(watermark_tok, 16)

        out_p.parent.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(prefix="mp_audio_custom_embed_"))

        try:
            pcm_in_path = temp_dir / "input_16k.wav"
            pcm_out_path = temp_dir / "watermarked_16k.wav"

            # 1. Convert input audio to 16 kHz Mono 16-bit PCM WAV using FFmpeg
            cmd_conv = [
                str(ffmpeg_exe), "-y",
                "-i", str(in_p),
                "-ar", str(AUDIO_SAMPLE_RATE),
                "-ac", "1",
                "-c:a", "pcm_s16le",
                str(pcm_in_path)
            ]
            res_conv = run_ffmpeg_command(cmd_conv, timeout=timeout)
            if res_conv.returncode != 0 or not pcm_in_path.exists():
                raise FFmpegExecutionError(
                    message=f"Failed to convert audio to 16kHz PCM: {res_conv.stderr}",
                    returncode=res_conv.returncode,
                    stdout=res_conv.stdout,
                    stderr=res_conv.stderr
                )

            # 2. Read PCM audio samples
            with wave.open(str(pcm_in_path), "rb") as wf:
                n_frames = wf.getnframes()
                raw_bytes = wf.readframes(n_frames)
                samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)

            # 3. Generate deterministic low-amplitude spread spectrum noise sequence
            rng = np.random.RandomState(seed=token_int)
            spread_noise = rng.choice([-1.0, 1.0], size=len(samples)) * 50.0  # ~ -36 dB amplitude perturbation

            # Embed spread spectrum watermark
            wm_samples = np.clip(samples + spread_noise, -32768, 32767).astype(np.int16)

            # 4. Save watermarked 16kHz PCM WAV
            with wave.open(str(pcm_out_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(AUDIO_SAMPLE_RATE)
                wf.writeframes(wm_samples.tobytes())

            # 5. Export to target format using FFmpeg
            cmd_export = [
                str(ffmpeg_exe), "-y",
                "-i", str(pcm_out_path),
                str(out_p)
            ]
            res_exp = run_ffmpeg_command(cmd_export, timeout=timeout)
            if res_exp.returncode != 0 or not out_p.exists():
                raise FFmpegExecutionError(
                    message=f"Failed to export watermarked audio: {res_exp.stderr}",
                    returncode=res_exp.returncode,
                    stdout=res_exp.stdout,
                    stderr=res_exp.stderr
                )

            proc_time = time.time() - start_time
            return AudioEmbeddingResult(
                engine=self.engine_name,
                output_path=str(out_p),
                asset_id=asset_id,
                size_bytes=out_p.stat().st_size,
                has_watermark=True,
                processing_time_seconds=proc_time,
                details={
                    "engine_display_name": self.display_name,
                    "watermark_token": watermark_tok,
                },
            )

        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

    def detect(
        self,
        input_path: Union[str, Path],
        timeout: float = 30.0,
    ) -> AudioDetectionResult:
        start_time = time.time()
        in_p = Path(input_path).resolve()
        if not in_p.exists() or not in_p.is_file():
            raise FileNotFoundError(f"Input audio file not found: {in_p}")

        ffmpeg_exe = get_ffmpeg_path()
        if not ffmpeg_exe or not ffmpeg_exe.exists():
            return AudioDetectionResult(
                engine=self.engine_name,
                watermark_detected=False,
                confidence=0.0,
                recovered_asset_id=None,
                provenance_record=None,
                processing_time_seconds=time.time() - start_time,
                message="FFmpeg binary unavailable for custom audio watermark detection.",
            )

        probe_res = MediaService.probe_file(in_p)
        if not probe_res.has_audio:
            return AudioDetectionResult(
                engine=self.engine_name,
                watermark_detected=False,
                confidence=0.0,
                recovered_asset_id=None,
                provenance_record=None,
                processing_time_seconds=time.time() - start_time,
                message="Media file contains no audio streams.",
            )

        temp_dir = Path(tempfile.mkdtemp(prefix="mp_audio_custom_detect_"))

        try:
            pcm_in_path = temp_dir / "detect_16k.wav"

            cmd_conv = [
                str(ffmpeg_exe), "-y",
                "-i", str(in_p),
                "-ar", str(AUDIO_SAMPLE_RATE),
                "-ac", "1",
                "-c:a", "pcm_s16le",
                str(pcm_in_path)
            ]
            res_conv = run_ffmpeg_command(cmd_conv, timeout=timeout)
            if res_conv.returncode != 0 or not pcm_in_path.exists():
                return AudioDetectionResult(
                    engine=self.engine_name,
                    watermark_detected=False,
                    confidence=0.0,
                    recovered_asset_id=None,
                    provenance_record=None,
                    processing_time_seconds=time.time() - start_time,
                    message=f"FFmpeg audio extraction failed: {res_conv.stderr}",
                )

            with wave.open(str(pcm_in_path), "rb") as wf:
                n_frames = wf.getnframes()
                raw_bytes = wf.readframes(n_frames)
                samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)

            if len(samples) < 100:
                return AudioDetectionResult(
                    engine=self.engine_name,
                    watermark_detected=False,
                    confidence=0.0,
                    recovered_asset_id=None,
                    provenance_record=None,
                    processing_time_seconds=time.time() - start_time,
                    message="Insufficient audio sample length for watermark detection.",
                )

            if not np.all(np.isfinite(samples)):
                return AudioDetectionResult(
                    engine=self.engine_name,
                    watermark_detected=False,
                    confidence=0.0,
                    recovered_asset_id=None,
                    provenance_record=None,
                    processing_time_seconds=time.time() - start_time,
                    message="Audio samples contain non-finite values (NaN or Inf).",
                )

            # Query all registered SQLite watermark tokens and compute correlation score
            from provenance.repository import get_db_connection
            conn = get_db_connection()
            try:
                cursor = conn.execute("SELECT DISTINCT watermark_token FROM assets WHERE watermark_token IS NOT NULL")
                registered_tokens = [row["watermark_token"] for row in cursor.fetchall()]
            finally:
                conn.close()

            best_token, best_score = compute_watermark_correlation(samples, registered_tokens)

            # Threshold check for custom spread spectrum correlation (50.0 expected amplitude)
            if best_token and best_score > 25.0:
                provenance_rec = ProvenanceService.find_asset_by_watermark_token(best_token)
                if provenance_rec:
                    return AudioDetectionResult(
                        engine=self.engine_name,
                        watermark_detected=True,
                        confidence=1.0,
                        recovered_asset_id=provenance_rec.asset_id,
                        provenance_record=provenance_rec.to_dict(),
                        processing_time_seconds=time.time() - start_time,
                        message="Custom spread-spectrum audio watermark payload successfully detected and decoded.",
                        details={
                            "watermark_token": best_token,
                            "engine_display_name": self.display_name,
                            "correlation_score": round(best_score, 4),
                        },
                    )

            return AudioDetectionResult(
                engine=self.engine_name,
                watermark_detected=False,
                confidence=round(best_score, 4),
                recovered_asset_id=None,
                provenance_record=None,
                processing_time_seconds=time.time() - start_time,
                message="No registered custom spread-spectrum audio watermark detected.",
            )

        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
