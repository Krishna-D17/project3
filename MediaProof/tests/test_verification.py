import os
import shutil
import struct
import random
import wave
import pytest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient

from backend.main import app
from media.ffmpeg import get_ffmpeg_status
from provenance.models import AssetRecord
from provenance.service import ProvenanceService
from provenance.verification import (
    VerificationService,
    STATE_VERIFIED_ORIGINAL,
    STATE_VERIFIED_DERIVATIVE,
    STATE_POTENTIALLY_MANIPULATED,
    STATE_UNKNOWN_PROVENANCE,
    STATE_INSUFFICIENT_EVIDENCE,
)
from watermark.audio.embed import embed_audio_watermark
from watermark.video.embed import embed_video_watermark
from forensics.models import UnifiedForensicResult, ForensicSignal

client = TestClient(app)


def is_ffmpeg_available() -> bool:
    status = get_ffmpeg_status()
    return bool(status["ffmpeg_available"] and status["ffprobe_available"])


def create_fresh_audio_file(path: Path) -> Path:
    """Generates a fresh 1-second 16kHz PCM WAV file with unique random samples."""
    sample_rate = 16000
    num_samples = sample_rate * 1
    pcm_bytes = bytearray()
    for _ in range(num_samples):
        sample = random.randint(-1000, 1000)
        pcm_bytes.extend(struct.pack("<h", sample))

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return path


# ==============================================================================
# 1. Exact Original: Hash Match + Watermark -> VERIFIED ORIGINAL
# ==============================================================================

def test_verify_exact_original_with_watermark(tmp_path):
    if not is_ffmpeg_available():
        pytest.skip("FFmpeg/FFprobe binaries not available")

    audio_file = create_fresh_audio_file(tmp_path / "orig_source.wav")
    record = ProvenanceService.register_media(audio_file)
    asset_id = record.asset_id

    wm_output_path = tmp_path / "original_wm.wav"
    embed_audio_watermark(audio_file, wm_output_path, asset_id, engine_type="custom")

    # Register watermarked derivative record linking parent_asset_id = asset_id
    reg_wm = ProvenanceService.register_media(wm_output_path, parent_asset_id=asset_id, relationship_type="WATERMARKED")

    res = VerificationService.verify_media(wm_output_path, audio_engine="custom")
    assert res.status == "ok"
    assert res.classification == STATE_VERIFIED_ORIGINAL
    assert res.confidence == "HIGH"
    assert res.hash_evidence.status == "MATCH"
    assert res.watermark_evidence.status == "PRESENT_AND_RESOLVED"


# ==============================================================================
# 2. Exact Original: Hash Match + No Watermark -> VERIFIED ORIGINAL
# ==============================================================================

def test_verify_exact_original_without_watermark(tmp_path):
    if not is_ffmpeg_available():
        pytest.skip("FFmpeg/FFprobe binaries not available")

    audio_file = create_fresh_audio_file(tmp_path / "unwatermarked_orig.wav")
    record = ProvenanceService.register_media(audio_file)

    res = VerificationService.verify_media(audio_file, audio_engine="custom")
    assert res.status == "ok"
    assert res.classification == STATE_VERIFIED_ORIGINAL
    assert res.confidence == "HIGH"
    assert res.hash_evidence.status == "MATCH"
    assert any("watermark evidence unavailable or not detected" in lim.lower() for lim in res.limitations)


# ==============================================================================
# 3. Derivative: Hash Mismatch + Valid Watermark + Low Forensic -> VERIFIED DERIVATIVE
# ==============================================================================

# 3. Derivative: Hash Mismatch + Valid Watermark + Low Forensic -> VERIFIED DERIVATIVE
# ==============================================================================

# 3. Derivative: Hash Mismatch + Valid Watermark + Low Forensic -> VERIFIED DERIVATIVE (MEDIUM CONFIDENCE)
# ==============================================================================

def test_verify_derivative_with_watermark(tmp_path):
    if not is_ffmpeg_available():
        pytest.skip("FFmpeg/FFprobe binaries not available")

    audio_file = create_fresh_audio_file(tmp_path / "parent_source.wav")
    record = ProvenanceService.register_media(audio_file)
    asset_id = record.asset_id

    wm_output_path = tmp_path / "derivative_wm.wav"
    embed_audio_watermark(audio_file, wm_output_path, asset_id, engine_type="custom")

    mock_low_forensic = UnifiedForensicResult(
        forensic_status="LOW_RISK",
        risk_score=0.05,
        risk_score_type="heuristic",
        generic_forensic_status="LOW_GENERIC_ANOMALY",
        manipulation_forensic_status="NO_DETECTOR_AVAILABLE",
        video_signals=[],
        audio_signals=[],
        metadata_signals=[],
        ml_signals=[],
        limitations=[],
        processing_time_seconds=0.1
    )

    with patch("forensics.service.ForensicService.analyze_media", return_value=mock_low_forensic):
        res = VerificationService.verify_media(wm_output_path, audio_engine="custom")
        assert res.status == "ok"
        assert res.classification == STATE_VERIFIED_DERIVATIVE
        assert res.confidence == "MEDIUM"  # MUST NOT BE HIGH!
        assert res.provenance_confidence == "HIGH"
        assert res.authenticity_confidence == "UNKNOWN"
        assert res.watermark_evidence.status == "PRESENT_AND_RESOLVED"
        assert res.hash_evidence.status == "MISMATCH"
        assert any("watermark survival does not prove the derivative is unmodified" in cb for cb in res.confidence_basis)


# ==============================================================================
# 3b. Adversarial Derivative: Watermark Resolved + No ML Detector -> MUST NOT BE HIGH CONFIDENCE
# ==============================================================================

def test_verify_adversarial_deepfake_derivative_without_ml_detector(tmp_path):
    """Adversarial Regression Test: Watermark survived deepfake creation (wm_000_virat_7s_from_protected.mp4 scenario).
    Without ML detector, system must NOT emit HIGH confidence authenticity or claim derivative is unmodified."""
    if not is_ffmpeg_available():
        pytest.skip("FFmpeg/FFprobe binaries not available")

    audio_file = create_fresh_audio_file(tmp_path / "deepfake_source.wav")
    record = ProvenanceService.register_media(audio_file)
    asset_id = record.asset_id

    wm_output_path = tmp_path / "deepfake_wm.wav"
    embed_audio_watermark(audio_file, wm_output_path, asset_id, engine_type="custom")

    mock_medium_forensic = UnifiedForensicResult(
        forensic_status="ELEVATED_RISK",
        risk_score=0.30,
        risk_score_type="heuristic",
        generic_forensic_status="MEDIUM_GENERIC_ANOMALY",
        manipulation_forensic_status="NO_DETECTOR_AVAILABLE",
        video_signals=[],
        audio_signals=[ForensicSignal("phase_entropy_anomaly", "MEDIUM", 0.30, "Phase entropy anomaly")],
        metadata_signals=[],
        ml_signals=[],
        limitations=[],
        processing_time_seconds=0.1
    )

    with patch("forensics.service.ForensicService.analyze_media", return_value=mock_medium_forensic):
        res = VerificationService.verify_media(wm_output_path, audio_engine="custom")
        assert res.status == "ok"
        assert res.classification == STATE_VERIFIED_DERIVATIVE
        assert res.confidence == "MEDIUM"  # MUST NOT BE HIGH!
        assert res.provenance_confidence == "HIGH"
        assert res.authenticity_confidence == "UNKNOWN"
        assert res.watermark_evidence.status == "PRESENT_AND_RESOLVED"
        assert any("watermark survival does not prove the derivative is unmodified" in cb for cb in res.confidence_basis)


# ==============================================================================
# 3c. Validated Manipulation Evidence: Watermark Resolved + ML Suspected -> POTENTIALLY MANIPULATED
# ==============================================================================

def test_verify_watermark_with_suspected_manipulation(tmp_path):
    if not is_ffmpeg_available():
        pytest.skip("FFmpeg/FFprobe binaries not available")

    audio_file = create_fresh_audio_file(tmp_path / "deepfake_suspect_source.wav")
    record = ProvenanceService.register_media(audio_file)
    asset_id = record.asset_id

    wm_output_path = tmp_path / "deepfake_suspect_wm.wav"
    embed_audio_watermark(audio_file, wm_output_path, asset_id, engine_type="custom")

    mock_suspect_forensic = UnifiedForensicResult(
        forensic_status="HIGHER_RISK",
        risk_score=0.85,
        risk_score_type="heuristic",
        generic_forensic_status="MEDIUM_GENERIC_ANOMALY",
        manipulation_forensic_status="MANIPULATION_SUSPECTED",
        video_signals=[],
        audio_signals=[],
        metadata_signals=[],
        ml_signals=[],
        limitations=[],
        processing_time_seconds=0.1
    )

    with patch("forensics.service.ForensicService.analyze_media", return_value=mock_suspect_forensic):
        res = VerificationService.verify_media(wm_output_path, audio_engine="custom")
        assert res.status == "ok"
        assert res.classification == STATE_POTENTIALLY_MANIPULATED
        assert res.confidence == "HIGH"
        assert res.provenance_confidence == "HIGH"
        assert res.authenticity_confidence == "LOW"
        assert res.watermark_evidence.status == "PRESENT_AND_RESOLVED"
        assert any("Watermark provenance does not override" in cb for cb in res.confidence_basis)


# ==============================================================================
# 4. Conflicting Cryptographic Hash Match + Manipulation Evidence -> INSUFFICIENT EVIDENCE
# ==============================================================================

def test_verify_hash_match_with_high_forensic_risk(tmp_path):
    if not is_ffmpeg_available():
        pytest.skip("FFmpeg/FFprobe binaries not available")

    audio_file = create_fresh_audio_file(tmp_path / "hash_match_high_forensics.wav")
    record = ProvenanceService.register_media(audio_file)

    mock_high_forensic = UnifiedForensicResult(
        forensic_status="HIGHER_RISK",
        risk_score=0.80,
        risk_score_type="heuristic",
        generic_forensic_status="HIGH_GENERIC_ANOMALY",
        manipulation_forensic_status="MANIPULATION_SUSPECTED",
        video_signals=[],
        audio_signals=[ForensicSignal("audio_clipping", "HIGH", 0.80, "Severe clipping")],
        metadata_signals=[],
        ml_signals=[],
        limitations=[],
        processing_time_seconds=0.1
    )

    with patch("forensics.service.ForensicService.analyze_media", return_value=mock_high_forensic):
        res = VerificationService.verify_media(audio_file, audio_engine="custom")
        assert res.status == "ok"
        assert res.classification == STATE_INSUFFICIENT_EVIDENCE
        assert res.confidence == "LOW"
        assert len(res.conflicts) > 0
        assert "manipulation-specific evidence" in res.conflicts[0].lower()


# ==============================================================================
# 5. Watermark Absent: Must NOT Imply Manipulation -> UNKNOWN PROVENANCE
# ==============================================================================

def test_verify_unknown_provenance(tmp_path):
    if not is_ffmpeg_available():
        pytest.skip("FFmpeg/FFprobe binaries not available")

    unreg_file = create_fresh_audio_file(tmp_path / "unregistered_new_bytes.wav")

    res = VerificationService.verify_media(unreg_file, audio_engine="custom")
    assert res.status == "ok"
    assert res.classification == STATE_UNKNOWN_PROVENANCE
    assert res.provenance_confidence == "NONE"
    assert res.authenticity_confidence == "UNKNOWN"


# ==============================================================================
# 6. Unregistered File with High Generic Anomaly -> POTENTIALLY MANIPULATED
# ==============================================================================

def test_verify_suspicious_unregistered_high_forensics(tmp_path):
    if not is_ffmpeg_available():
        pytest.skip("FFmpeg/FFprobe binaries not available")

    unreg_file = create_fresh_audio_file(tmp_path / "suspicious_unregistered.wav")

    mock_high_forensic = UnifiedForensicResult(
        forensic_status="HIGHER_RISK",
        risk_score=0.88,
        risk_score_type="heuristic",
        generic_forensic_status="HIGH_GENERIC_ANOMALY",
        manipulation_forensic_status="NO_DETECTOR_AVAILABLE",
        video_signals=[],
        audio_signals=[ForensicSignal("audio_clipping", "HIGH", 0.95, "Severe clipping")],
        metadata_signals=[],
        ml_signals=[],
        limitations=[],
        processing_time_seconds=0.1
    )

    with patch("provenance.repository.get_asset_by_sha256", return_value=None):
        with patch("forensics.service.ForensicService.analyze_media", return_value=mock_high_forensic):
            res = VerificationService.verify_media(unreg_file, audio_engine="custom")
            assert res.status == "ok"
            assert res.classification == STATE_POTENTIALLY_MANIPULATED


# ==============================================================================
# 7. Conflicting Provenance: Watermark Asset A + Hash Asset B -> INSUFFICIENT EVIDENCE
# ==============================================================================

def test_verify_conflicting_provenance(tmp_path):
    if not is_ffmpeg_available():
        pytest.skip("FFmpeg/FFprobe binaries not available")

    file_a = create_fresh_audio_file(tmp_path / "file_a.wav")
    record_a = ProvenanceService.register_media(file_a)

    dummy_record_b = AssetRecord(
        asset_id="MP-11111111-2222-3333-4444-555555555555",
        sha256="0000000000000000000000000000000000000000000000000000000000000000",
        original_filename="dummy_b.wav",
        original_path="dummy_b.wav",
        file_size_bytes=1000,
        media_type="audio",
        container_format="wav",
        duration_seconds=1.0,
        video_codec=None,
        width=None,
        height=None,
        fps=None,
        audio_codec="pcm_s16le",
        sample_rate=16000,
        channels=1,
        watermark_token="B999",
        parent_asset_id=None,
    )

    # Patch service get_asset_by_sha256 to return Asset B while watermark resolves to Asset A
    with patch("provenance.service.get_asset_by_sha256", return_value=dummy_record_b):
        with patch("provenance.verification.get_asset_by_watermark_token", return_value=record_a):
            with patch("provenance.verification.detect_audio_watermark", return_value={
                "status": "ok",
                "watermark_detected": True,
                "recovered_asset_id": record_a.asset_id,
                "watermark_token": record_a.watermark_token,
                "details": {"watermark_token": record_a.watermark_token}
            }):
                res = VerificationService.verify_media(file_a, audio_engine="custom")
                assert res.classification == STATE_INSUFFICIENT_EVIDENCE
                assert len(res.conflicts) > 0
                assert "Provenance Conflict" in res.conflicts[0]


# ==============================================================================
# 8. Unresolved Watermark Handling
# ==============================================================================

def test_verify_unresolved_watermark(tmp_path):
    if not is_ffmpeg_available():
        pytest.skip("FFmpeg/FFprobe binaries not available")

    audio_file = create_fresh_audio_file(tmp_path / "unresolved_src.wav")

    with patch("provenance.repository.get_asset_by_sha256", return_value=None):
        with patch("provenance.verification.get_asset_by_watermark_token", return_value=None):
            with patch("provenance.verification.detect_audio_watermark", return_value={
                "status": "ok",
                "watermark_detected": True,
                "recovered_asset_id": None,
                "watermark_token": "9999",
                "details": {"watermark_token": "9999"}
            }):
                res = VerificationService.verify_media(audio_file, audio_engine="custom")
                assert res.watermark_evidence.status == "PRESENT_BUT_UNRESOLVED"
                assert any("not present in database registry" in lim for lim in res.limitations)


# ==============================================================================
# 9. Detector Failure Graceful Partial Evidence
# ==============================================================================

def test_verify_forensic_detector_failure_graceful_partial(tmp_path):
    if not is_ffmpeg_available():
        pytest.skip("FFmpeg/FFprobe binaries not available")

    audio_file = create_fresh_audio_file(tmp_path / "detector_fail.wav")

    with patch("forensics.service.ForensicService.analyze_media", side_effect=RuntimeError("Forensic crash")):
        res = VerificationService.verify_media(audio_file, audio_engine="custom")
        assert res.status == "ok"
        assert res.hash_evidence.status in ["MATCH", "MISMATCH"]
        assert any("Forensic evaluation exception" in lim for lim in res.limitations)


# ==============================================================================
# 10. Malformed Media File Handling
# ==============================================================================

def test_verify_malformed_media_file(tmp_path):
    bad_file = tmp_path / "corrupt.mp4"
    bad_file.write_bytes(b"CORRUPT_BYTES_DATA")

    res = VerificationService.verify_media(bad_file)
    assert res.status == "error"
    assert res.classification == STATE_INSUFFICIENT_EVIDENCE
    assert res.confidence == "LOW"


# ==============================================================================
# 11. Live HTTP API Endpoint (`POST /api/verify`)
# ==============================================================================

def test_api_verify_endpoint(tmp_path):
    if not is_ffmpeg_available():
        pytest.skip("FFmpeg/FFprobe binaries not available")

    audio_file = create_fresh_audio_file(tmp_path / "api_src.wav")
    record = ProvenanceService.register_media(audio_file)
    asset_id = record.asset_id

    wm_output_path = tmp_path / "api_verify_wm.wav"
    embed_audio_watermark(audio_file, wm_output_path, asset_id, engine_type="custom")

    with open(wm_output_path, "rb") as f:
        res = client.post(
            "/api/verify",
            files={"file": ("api_wm_verify.wav", f, "audio/wav")},
            data={"audio_engine": "custom", "video_engine": "custom"}
        )

    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "classification" in data
    assert "confidence" in data
    assert "hash_evidence" in data
    assert "watermark_evidence" in data
    assert "provenance_evidence" in data
    assert "forensic_evidence" in data
    assert "metadata_evidence" in data


# ==============================================================================
# 12. Hardened API Status Code Test for Corrupt Media File
# ==============================================================================

def test_api_verify_endpoint_corrupt_file_returns_400(tmp_path):
    bad_file = tmp_path / "corrupt_test.mp4"
    bad_file.write_bytes(b"CORRUPT_BYTES_DATA_12345")

    with open(bad_file, "rb") as f:
        res = client.post(
            "/api/verify",
            files={"file": ("corrupt_test.mp4", f, "video/mp4")},
            data={"video_engine": "custom", "audio_engine": "custom"}
        )

    assert res.status_code == 400
    data = res.json()
    assert data["status"] == "error"
    assert data["classification"] == STATE_INSUFFICIENT_EVIDENCE


# ==============================================================================
# 13. Frontend UI Static Assets Loading Test
# ==============================================================================

def test_frontend_ui_assets_load():
    res_root = client.get("/")
    assert res_root.status_code == 200
    assert "text/html" in res_root.headers.get("content-type", "")
    assert "MediaProof" in res_root.text

    res_js = client.get("/static/app.js")
    assert res_js.status_code == 200

    res_css = client.get("/static/style.css")
    assert res_css.status_code == 200


# ==============================================================================
# 14. Regression Tests: NO_VALIDATED_DETECTOR & Clean Detector Explanations
# ==============================================================================

def test_verify_derivative_with_no_validated_detector_constant(tmp_path):
    """Regression Test 1: Verify that canonical NO_VALIDATED_DETECTOR results in VERIFIED DERIVATIVE
    with authenticity_confidence='UNKNOWN' and never claims ML detector reported clean."""
    if not is_ffmpeg_available():
        pytest.skip("FFmpeg/FFprobe binaries not available")

    audio_file = create_fresh_audio_file(tmp_path / "noval_src.wav")
    record = ProvenanceService.register_media(audio_file)
    asset_id = record.asset_id

    wm_output_path = tmp_path / "noval_wm.wav"
    embed_audio_watermark(audio_file, wm_output_path, asset_id, engine_type="custom")

    from forensics.config import STATUS_MANIPULATION_NO_DETECTOR
    assert STATUS_MANIPULATION_NO_DETECTOR == "NO_VALIDATED_DETECTOR"

    mock_forensic = UnifiedForensicResult(
        forensic_status="LOW_RISK",
        risk_score=0.05,
        risk_score_type="heuristic",
        generic_forensic_status="LOW_GENERIC_ANOMALY",
        manipulation_forensic_status=STATUS_MANIPULATION_NO_DETECTOR,
        video_signals=[],
        audio_signals=[],
        metadata_signals=[],
        ml_signals=[],
        limitations=[],
        processing_time_seconds=0.05
    )

    with patch("provenance.verification.ForensicService.analyze_media", return_value=mock_forensic):
        res = VerificationService.verify_media(wm_output_path, audio_engine="custom")
        assert res.status == "ok"
        assert res.classification == STATE_VERIFIED_DERIVATIVE
        assert res.confidence == "MEDIUM"
        assert res.authenticity_confidence == "UNKNOWN"
        assert res.manipulation_evidence.status == "NOT_AVAILABLE"
        assert not any("Installed ML detector reported no manipulation anomaly" in cb for cb in res.confidence_basis)
        assert any("No validated manipulation-specific detector is available" in cb for cb in res.confidence_basis)


def test_verify_derivative_with_no_detector_available_string(tmp_path):
    """Regression Test 2: Verify that NO_DETECTOR_AVAILABLE string remains compatible."""
    if not is_ffmpeg_available():
        pytest.skip("FFmpeg/FFprobe binaries not available")

    audio_file = create_fresh_audio_file(tmp_path / "nodet_src.wav")
    record = ProvenanceService.register_media(audio_file)
    asset_id = record.asset_id

    wm_output_path = tmp_path / "nodet_wm.wav"
    embed_audio_watermark(audio_file, wm_output_path, asset_id, engine_type="custom")

    mock_forensic = UnifiedForensicResult(
        forensic_status="LOW_RISK",
        risk_score=0.05,
        risk_score_type="heuristic",
        generic_forensic_status="LOW_GENERIC_ANOMALY",
        manipulation_forensic_status="NO_DETECTOR_AVAILABLE",
        video_signals=[],
        audio_signals=[],
        metadata_signals=[],
        ml_signals=[],
        limitations=[],
        processing_time_seconds=0.05
    )

    with patch("provenance.verification.ForensicService.analyze_media", return_value=mock_forensic):
        res = VerificationService.verify_media(wm_output_path, audio_engine="custom")
        assert res.status == "ok"
        assert res.classification == STATE_VERIFIED_DERIVATIVE
        assert res.confidence == "MEDIUM"
        assert res.authenticity_confidence == "UNKNOWN"
        assert res.manipulation_evidence.status == "NOT_AVAILABLE"
        assert any("No validated manipulation-specific detector is available" in cb for cb in res.confidence_basis)
        assert not any("Installed ML detector reported no manipulation anomaly" in cb for cb in res.confidence_basis)


def test_verify_derivative_with_installed_clean_detector(tmp_path):
    """Regression Test 3: Verify that an installed clean detector sets authenticity_confidence='MEDIUM'
    and explicitly references clean detector in confidence basis."""
    if not is_ffmpeg_available():
        pytest.skip("FFmpeg/FFprobe binaries not available")

    audio_file = create_fresh_audio_file(tmp_path / "clean_src.wav")
    record = ProvenanceService.register_media(audio_file)
    asset_id = record.asset_id

    wm_output_path = tmp_path / "clean_wm.wav"
    embed_audio_watermark(audio_file, wm_output_path, asset_id, engine_type="custom")

    from forensics.config import STATUS_MANIPULATION_NO_ANOMALY

    mock_forensic = UnifiedForensicResult(
        forensic_status="LOW_RISK",
        risk_score=0.05,
        risk_score_type="heuristic",
        generic_forensic_status="LOW_GENERIC_ANOMALY",
        manipulation_forensic_status=STATUS_MANIPULATION_NO_ANOMALY,
        video_signals=[],
        audio_signals=[],
        metadata_signals=[],
        ml_signals=[],
        limitations=[],
        processing_time_seconds=0.05
    )

    with patch("provenance.verification.ForensicService.analyze_media", return_value=mock_forensic):
        res = VerificationService.verify_media(wm_output_path, audio_engine="custom")
        assert res.status == "ok"
        assert res.classification == STATE_VERIFIED_DERIVATIVE
        assert res.confidence == "MEDIUM"
        assert res.authenticity_confidence == "MEDIUM"
        assert res.manipulation_evidence.status == "NO_ANOMALY_DETECTED"
        assert any("Installed ML detector reported no manipulation anomaly" in cb for cb in res.confidence_basis)


