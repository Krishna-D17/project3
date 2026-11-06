from pathlib import Path
from typing import List, Union

from media.service import MediaService
from forensics.config import (
    CATEGORY_METADATA,
    RISK_LEVEL_HIGH,
    RISK_LEVEL_LOW,
    RISK_LEVEL_MEDIUM,
)
from forensics.models import ForensicSignal


def extract_metadata_signals(file_path: Union[str, Path]) -> List[ForensicSignal]:
    """Extracts container and codec metadata signals via FFprobe and evaluates structural anomalies."""
    signals: List[ForensicSignal] = []
    p = Path(file_path).resolve()

    try:
        probe_res = MediaService.probe_file(p)
    except Exception as e:
        signals.append(ForensicSignal(
            signal_name="container_probe_error",
            risk_level=RISK_LEVEL_HIGH,
            score=1.0,
            details=f"Failed to probe media metadata: {str(e)}",
            evidence_strength="medium",
            type="heuristic",
            category=CATEGORY_METADATA,
        ))
        return signals

    # 1. Audio / Video Stream Presence Signal
    has_v = probe_res.has_video
    has_a = probe_res.has_audio
    signals.append(ForensicSignal(
        signal_name="stream_presence",
        risk_level=RISK_LEVEL_LOW,
        score=0.0,
        details=f"Container streams: video={has_v}, audio={has_a}, format={probe_res.container_format}",
        evidence_strength="low",
        type="heuristic",
        category=CATEGORY_METADATA,
    ))

    # 2. Audio/Video Duration Alignment Signal
    if has_v and has_a and probe_res.duration_seconds is not None:
        signals.append(ForensicSignal(
            signal_name="stream_duration_alignment",
            risk_level=RISK_LEVEL_LOW,
            score=0.0,
            details=f"Container duration: {probe_res.duration_seconds:.2f}s",
            evidence_strength="low",
            type="heuristic",
            category=CATEGORY_METADATA,
        ))

    # 3. Codec & Encoder Metadata Consistency
    if has_v and probe_res.video_streams:
        v_info = probe_res.video_streams[0]
        v_codec = v_info.codec or "unknown"
        if v_codec.lower() in ["h264", "hevc", "vp9", "av1", "mpeg4", "rawvideo"]:
            signals.append(ForensicSignal(
                signal_name="video_codec_standard",
                risk_level=RISK_LEVEL_LOW,
                score=0.0,
                details=f"Standard video codec detected: {v_codec} ({v_info.width}x{v_info.height} @ {v_info.fps:.2f}fps)",
                evidence_strength="low",
                type="heuristic",
                category=CATEGORY_METADATA,
            ))
        else:
            signals.append(ForensicSignal(
                signal_name="video_codec_unusual",
                risk_level=RISK_LEVEL_MEDIUM,
                score=0.5,
                details=f"Unusual or legacy video codec tag: {v_codec}",
                evidence_strength="medium",
                type="heuristic",
                category=CATEGORY_METADATA,
            ))

    if has_a and probe_res.audio_streams:
        a_info = probe_res.audio_streams[0]
        a_codec = a_info.codec or "unknown"
        s_rate = a_info.sample_rate or 0
        if a_codec.lower() in ["aac", "mp3", "pcm_s16le", "flac", "opus", "vorbis"]:
            signals.append(ForensicSignal(
                signal_name="audio_codec_standard",
                risk_level=RISK_LEVEL_LOW,
                score=0.0,
                details=f"Standard audio codec detected: {a_codec} ({s_rate}Hz, {a_info.channels}ch)",
                evidence_strength="low",
                type="heuristic",
                category=CATEGORY_METADATA,
            ))
        else:
            signals.append(ForensicSignal(
                signal_name="audio_codec_unusual",
                risk_level=RISK_LEVEL_MEDIUM,
                score=0.4,
                details=f"Unusual audio codec tag: {a_codec}",
                evidence_strength="low",
                type="heuristic",
                category=CATEGORY_METADATA,
            ))

    return signals
