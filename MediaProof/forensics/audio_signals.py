import shutil
import tempfile
import wave
from pathlib import Path
from typing import List, Union

import numpy as np

from media.ffmpeg import get_ffmpeg_path, run_ffmpeg_command
from media.service import MediaService
from forensics.config import (
    AUDIO_TARGET_SAMPLE_RATE,
    CATEGORY_GENERIC_TECHNICAL,
    HEURISTIC_AUDIO_CLIPPING_RATIO_THRESHOLD,
    HEURISTIC_AUDIO_HIGH_FREQ_PHASE_ENTROPY_THRESHOLD,
    HEURISTIC_AUDIO_SILENCE_RATIO_THRESHOLD,
    HEURISTIC_AUDIO_SPECTRAL_ROLLOFF_THRESHOLD,
    RISK_LEVEL_HIGH,
    RISK_LEVEL_LOW,
    RISK_LEVEL_MEDIUM,
)
from forensics.models import ForensicSignal


def extract_audio_signals(file_path: Union[str, Path], timeout: float = 30.0) -> List[ForensicSignal]:
    """Extracts deterministic acoustic and spectral forensic features from audio streams."""
    signals: List[ForensicSignal] = []
    p = Path(file_path).resolve()

    ffmpeg_exe = get_ffmpeg_path()
    if not ffmpeg_exe or not ffmpeg_exe.exists():
        signals.append(ForensicSignal(
            signal_name="audio_forensics_status",
            risk_level=RISK_LEVEL_MEDIUM,
            score=0.5,
            details="FFmpeg binary not available for audio extraction.",
            evidence_strength="low",
            type="heuristic",
            category=CATEGORY_GENERIC_TECHNICAL,
        ))
        return signals

    probe_res = MediaService.probe_file(p)
    if not probe_res.has_audio:
        return signals

    temp_dir = Path(tempfile.mkdtemp(prefix="mp_forensics_audio_"))

    try:
        pcm_16k_path = temp_dir / "audio_16k.wav"
        cmd_conv = [
            str(ffmpeg_exe), "-y",
            "-i", str(p),
            "-ar", str(AUDIO_TARGET_SAMPLE_RATE),
            "-ac", "1",
            "-c:a", "pcm_s16le",
            str(pcm_16k_path)
        ]
        res_conv = run_ffmpeg_command(cmd_conv, timeout=timeout)
        if res_conv.returncode != 0 or not pcm_16k_path.exists():
            signals.append(ForensicSignal(
                signal_name="audio_extraction_status",
                risk_level=RISK_LEVEL_MEDIUM,
                score=0.5,
                details=f"FFmpeg audio extraction failed: {res_conv.stderr}",
                evidence_strength="low",
                type="heuristic"
            ))
            return signals

        with wave.open(str(pcm_16k_path), "rb") as wf:
            n_frames = wf.getnframes()
            raw_bytes = wf.readframes(n_frames)
            int16_samples = np.frombuffer(raw_bytes, dtype=np.int16)
            samples = int16_samples.astype(np.float32)

        if len(samples) == 0:
            signals.append(ForensicSignal(
                signal_name="audio_sample_count",
                risk_level=RISK_LEVEL_LOW,
                score=0.0,
                details="Audio stream contains 0 PCM samples.",
                evidence_strength="low",
                type="heuristic"
            ))
            return signals

        # 1. Amplitude Statistics: RMS, Peak, Clipping Ratio, Silence Ratio
        peak_val = float(np.max(np.abs(samples)))
        rms_val = float(np.sqrt(np.mean(samples ** 2)))
        clipping_ratio = float(np.sum(np.abs(samples) >= 32700) / len(samples))
        silence_ratio = float(np.sum(np.abs(samples) < 100) / len(samples))

        # Signal A: Amplitude & Clipping
        if clipping_ratio > HEURISTIC_AUDIO_CLIPPING_RATIO_THRESHOLD:
            signals.append(ForensicSignal(
                signal_name="audio_amplitude_clipping",
                risk_level=RISK_LEVEL_MEDIUM,
                score=clipping_ratio,
                details=f"High clipping ratio ({clipping_ratio * 100:.2f}%); peak={peak_val:.0f}, rms={rms_val:.1f}.",
                evidence_strength="low",
                type="heuristic"
            ))
        else:
            signals.append(ForensicSignal(
                signal_name="audio_amplitude_clipping",
                risk_level=RISK_LEVEL_LOW,
                score=clipping_ratio,
                details=f"Normal audio amplitude bounds; peak={peak_val:.0f}, rms={rms_val:.1f}, clipping={clipping_ratio * 100:.2f}%.",
                evidence_strength="low",
                type="heuristic"
            ))

        # Signal B: Silence Ratio
        if silence_ratio > HEURISTIC_AUDIO_SILENCE_RATIO_THRESHOLD:
            signals.append(ForensicSignal(
                signal_name="audio_silence_ratio",
                risk_level=RISK_LEVEL_MEDIUM,
                score=silence_ratio,
                details=f"High silence proportion ({silence_ratio * 100:.2f}% of audio clip).",
                evidence_strength="low",
                type="heuristic"
            ))
        else:
            signals.append(ForensicSignal(
                signal_name="audio_silence_ratio",
                risk_level=RISK_LEVEL_LOW,
                score=silence_ratio,
                details=f"Normal speech energy continuity ({silence_ratio * 100:.2f}% silence).",
                evidence_strength="low",
                type="heuristic"
            ))

        # 2. Spectral Analysis (STFT, Spectral Centroid, Rolloff, Phase Entropy)
        # Compute Short-Time Fourier Transform (STFT)
        n_fft = 512
        hop_length = 256
        if len(samples) >= n_fft:
            # Frame the signal
            n_frames_stft = 1 + (len(samples) - n_fft) // hop_length
            shape = (n_frames_stft, n_fft)
            strides = (samples.strides[0] * hop_length, samples.strides[0])
            frames_stft = np.lib.stride_tricks.as_strided(samples, shape=shape, strides=strides)

            # Apply Hanning window
            window = np.hanning(n_fft)
            windowed_frames = frames_stft * window

            # FFT
            fft_mag = np.abs(np.fft.rfft(windowed_frames, n=n_fft)) # shape: (n_frames, n_fft//2 + 1)
            fft_phase = np.angle(np.fft.rfft(windowed_frames, n=n_fft))

            freqs = np.fft.rfftfreq(n_fft, d=1.0 / AUDIO_TARGET_SAMPLE_RATE)

            # Spectral Centroid
            sum_mag = np.sum(fft_mag, axis=1) + 1e-8
            centroids = np.sum(fft_mag * freqs, axis=1) / sum_mag
            avg_centroid = float(np.mean(centroids))

            # Spectral Rolloff (85% energy point)
            cum_energy = np.cumsum(fft_mag, axis=1)
            total_energy_per_frame = cum_energy[:, -1:] + 1e-8
            rolloff_idx = np.argmax(cum_energy >= 0.85 * total_energy_per_frame, axis=1)
            rolloff_freqs = freqs[rolloff_idx]
            avg_rolloff = float(np.mean(rolloff_freqs))

            # Phase Entropy / Variance across frames
            phase_diffs = np.diff(fft_phase, axis=0)
            phase_var = float(np.mean(np.var(phase_diffs, axis=0)))

            # Signal C: Spectral Centroid & Rolloff
            if avg_rolloff > HEURISTIC_AUDIO_SPECTRAL_ROLLOFF_THRESHOLD:
                signals.append(ForensicSignal(
                    signal_name="spectral_rolloff_bandwidth",
                    risk_level=RISK_LEVEL_MEDIUM,
                    score=avg_rolloff,
                    details=f"High spectral rolloff frequency ({avg_rolloff:.1f} Hz); centroid={avg_centroid:.1f} Hz.",
                    evidence_strength="low",
                    type="heuristic"
                ))
            else:
                signals.append(ForensicSignal(
                    signal_name="spectral_rolloff_bandwidth",
                    risk_level=RISK_LEVEL_LOW,
                    score=avg_rolloff,
                    details=f"Standard audio spectral bandwidth; rolloff={avg_rolloff:.1f} Hz, centroid={avg_centroid:.1f} Hz.",
                    evidence_strength="low",
                    type="heuristic"
                ))

            # Signal D: Phase Entropy & Continuity
            signals.append(ForensicSignal(
                signal_name="audio_phase_continuity",
                risk_level=RISK_LEVEL_LOW,
                score=phase_var,
                details=f"Acoustic phase variance score: {phase_var:.4f}.",
                evidence_strength="low",
                type="heuristic"
            ))

        return signals

    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
