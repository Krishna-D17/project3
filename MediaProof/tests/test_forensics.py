import os
import shutil
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from forensics.audio_signals import extract_audio_signals
from forensics.metadata_signals import extract_metadata_signals
from forensics.models import ForensicSignal, UnifiedForensicResult
from forensics.service import ForensicService
from forensics.video_signals import extract_video_signals
from media.ffmpeg import get_ffmpeg_status

client = TestClient(app)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_MEDIA_DIR = PROJECT_ROOT / "data" / "temp" / "test_media"


def is_ffmpeg_available() -> bool:
    status = get_ffmpeg_status()
    return bool(status["ffmpeg_available"] and status["ffprobe_available"])


# ==============================================================================
# 1. Metadata Extraction Tests
# ==============================================================================

def test_metadata_signal_extraction():
    if not is_ffmpeg_available():
        pytest.skip("FFmpeg/FFprobe binaries not available")

    audio_file = TEST_MEDIA_DIR / "test_audio.wav"
    if not audio_file.exists():
        pytest.skip("Test audio file not generated")

    signals = extract_metadata_signals(audio_file)
    assert isinstance(signals, list)
    assert len(signals) > 0
    assert any(s.signal_name == "stream_presence" for s in signals)


# ==============================================================================
# 2. Audio Signal Extraction Tests
# ==============================================================================

def test_audio_signal_extraction():
    if not is_ffmpeg_available():
        pytest.skip("FFmpeg/FFprobe binaries not available")

    audio_file = TEST_MEDIA_DIR / "test_audio.wav"
    if not audio_file.exists():
        pytest.skip("Test audio file not generated")

    signals = extract_audio_signals(audio_file)
    assert isinstance(signals, list)
    assert len(signals) > 0
    assert any(s.signal_name == "audio_amplitude_clipping" for s in signals)
    assert any(s.signal_name == "spectral_rolloff_bandwidth" for s in signals)


# ==============================================================================
# 3. Video Signal Extraction Tests
# ==============================================================================

def test_video_signal_extraction():
    if not is_ffmpeg_available():
        pytest.skip("FFmpeg/FFprobe binaries not available")

    video_file = TEST_MEDIA_DIR / "test_video.mp4"
    if not video_file.exists():
        pytest.skip("Test video file not generated")

    signals = extract_video_signals(video_file)
    assert isinstance(signals, list)
    assert len(signals) > 0
    assert any(s.signal_name == "video_sampled_frame_count" for s in signals)
    assert any(s.signal_name == "high_frequency_energy_ratio" for s in signals)


# ==============================================================================
# 4. Combined Audio / Video Analysis (MP4)
# ==============================================================================

def test_combined_mp4_forensic_analysis():
    if not is_ffmpeg_available():
        pytest.skip("FFmpeg/FFprobe binaries not available")

    comb_file = TEST_MEDIA_DIR / "test_combined.mp4"
    if not comb_file.exists():
        pytest.skip("Test combined video/audio file not generated")

    res = ForensicService.analyze_media(comb_file)
    assert isinstance(res, UnifiedForensicResult)
    assert res.forensic_status in ["LOW_RISK", "ELEVATED_RISK", "HIGHER_RISK", "INSUFFICIENT_EVIDENCE"]
    assert len(res.video_signals) > 0
    assert len(res.audio_signals) > 0
    assert len(res.metadata_signals) > 0


# ==============================================================================
# 5. WAV Audio Analysis
# ==============================================================================

def test_wav_forensic_analysis():
    if not is_ffmpeg_available():
        pytest.skip("FFmpeg/FFprobe binaries not available")

    audio_file = TEST_MEDIA_DIR / "test_audio.wav"
    if not audio_file.exists():
        pytest.skip("Test audio file not generated")

    res = ForensicService.analyze_media(audio_file)
    assert isinstance(res, UnifiedForensicResult)
    assert len(res.audio_signals) > 0
    assert len(res.video_signals) == 0


# ==============================================================================
# 6. Deterministic Repeated Analysis
# ==============================================================================

def test_deterministic_repeated_forensic_analysis():
    if not is_ffmpeg_available():
        pytest.skip("FFmpeg/FFprobe binaries not available")

    audio_file = TEST_MEDIA_DIR / "test_audio.wav"
    if not audio_file.exists():
        pytest.skip("Test audio file not generated")

    res1 = ForensicService.analyze_media(audio_file)
    res2 = ForensicService.analyze_media(audio_file)

    assert res1.forensic_status == res2.forensic_status
    assert res1.risk_score == res2.risk_score
    assert len(res1.audio_signals) == len(res2.audio_signals)


# ==============================================================================
# 7. Malformed / Unsupported Media Handling
# ==============================================================================

def test_malformed_media_forensic_handling(tmp_path):
    bad_file = tmp_path / "bad.mp4"
    bad_file.write_bytes(b"NOT_A_MEDIA_FILE_HEADER")

    res = ForensicService.analyze_media(bad_file)
    assert res.forensic_status == "INSUFFICIENT_EVIDENCE"
    assert len(res.limitations) > 0


# ==============================================================================
# 8. ML Unavailable Behavior
# ==============================================================================

def test_ml_unavailable_behavior():
    if not is_ffmpeg_available():
        pytest.skip("FFmpeg/FFprobe binaries not available")

    audio_file = TEST_MEDIA_DIR / "test_audio.wav"
    if not audio_file.exists():
        pytest.skip("Test audio file not generated")

    res = ForensicService.analyze_media(audio_file)
    # ML signals should report available=False when checkpoint files are absent
    for ml in res.ml_signals:
        assert isinstance(ml.available, bool)
        if not ml.available:
            assert "UNAVAILABLE" in ml.status


# ==============================================================================
# 9. API Endpoint Test (`POST /api/forensics/analyze`)
# ==============================================================================

def test_api_forensics_analyze_endpoint():
    if not is_ffmpeg_available():
        pytest.skip("FFmpeg/FFprobe binaries not available")

    audio_file = TEST_MEDIA_DIR / "test_audio.wav"
    if not audio_file.exists():
        pytest.skip("Test audio file not generated")

    with open(audio_file, "rb") as f:
        res = client.post(
            "/api/forensics/analyze",
            files={"file": ("api_test_audio.wav", f, "audio/wav")}
        )

    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["media_type"] == "audio"
    assert "forensic" in data
    forensic = data["forensic"]
    assert "forensic_status" in forensic
    assert "risk_score" in forensic
    assert forensic["risk_score_type"] == "heuristic"


# ==============================================================================
# 10. Context-Aware Parent Baseline Comparison Test
# ==============================================================================

def test_context_aware_parent_baseline_comparison():
    if not is_ffmpeg_available():
        pytest.skip("FFmpeg/FFprobe binaries not available")

    audio_file = TEST_MEDIA_DIR / "test_audio.wav"
    if not audio_file.exists():
        pytest.skip("Test audio file not generated")

    res = ForensicService.analyze_media(audio_file, parent_file_path=audio_file)
    assert isinstance(res, UnifiedForensicResult)
    assert hasattr(res, "generic_forensic_status")
    assert hasattr(res, "manipulation_forensic_status")
    assert hasattr(res, "transformation_consistency")
    assert res.transformation_consistency.status in ["CONSISTENT", "DEVIATION", "NOT_AVAILABLE"]


# ==============================================================================
# 11. Video ML Detector Interface & Metadata Architecture Contract Test
# ==============================================================================

def test_video_ml_detector_contract_and_metadata(tmp_path):
    from forensics.video_ml import (
        BaseDeepfakeDetector,
        DeepfakeModelMetadata,
        EfficientNetB4Adapter,
        XceptionDeepfakeDetector,
        get_video_deepfake_detector,
    )

    # 1. Base interface subclass check
    eff_adapter = EfficientNetB4Adapter()
    xc_detector = XceptionDeepfakeDetector()
    assert isinstance(eff_adapter, BaseDeepfakeDetector)
    assert isinstance(xc_detector, BaseDeepfakeDetector)

    # 2. Metadata schema verification
    meta_dict = xc_detector.metadata.to_dict()
    assert meta_dict["model_name"] == "DeepfakeBench Xception"
    assert meta_dict["architecture"] == "Xception"
    assert meta_dict["input_size"] == [299, 299]
    assert meta_dict["available"] is True

    # 3. Missing checkpoint handling returns UNAVAILABLE_CHECKPOINT_MISSING
    fake_video = tmp_path / "dummy.mp4"
    fake_video.write_bytes(b"DUMMY_VIDEO_DATA")
    missing_xc = XceptionDeepfakeDetector(checkpoint_path=tmp_path / "missing.pth")
    res_eff = eff_adapter.predict_video(fake_video)
    res_xc = missing_xc.predict_video(fake_video)
    assert res_eff.available is False
    assert "UNAVAILABLE" in res_eff.status
    assert res_xc.available is False
    assert "UNAVAILABLE" in res_xc.status

    # 4. Factory default selection
    default_det = get_video_deepfake_detector("auto")
    assert isinstance(default_det, BaseDeepfakeDetector)


# ==============================================================================
# 12. Frame-Extraction Failure Safe Propagation Test (Phase 0.1.4 Test B)
# ==============================================================================

def test_frame_extraction_failure_does_not_produce_no_anomaly(monkeypatch, tmp_path):
    from forensics.models import MLInferenceResult
    from forensics.service import (
        ForensicService,
        STATUS_MANIPULATION_NO_ANOMALY,
        STATUS_MANIPULATION_NO_DETECTOR,
        STATUS_MANIPULATION_SUSPECTED,
    )

    dummy_video = tmp_path / "failed_extraction.mp4"
    dummy_video.write_bytes(b"DUMMY_MP4_CONTENT")

    # Mock probe to indicate video stream present
    class MockProbe:
        has_video = True
        has_audio = False
        duration = 5.0
        width = 1280
        height = 720
        format_name = "mov,mp4,m4a,3gp,3g2,mj2"
        streams = []

    monkeypatch.setattr("forensics.service.MediaService.probe_file", lambda p: MockProbe())
    monkeypatch.setattr("forensics.service.extract_metadata_signals", lambda p: [])
    monkeypatch.setattr("forensics.service.extract_video_signals", lambda p, timeout=15.0: [])

    # Case 1: Video ML returns FRAME_EXTRACTION_FAILED
    failed_result = MLInferenceResult(
        model_name="DeepfakeBench Xception",
        available=False,
        status="FRAME_EXTRACTION_FAILED",
        prediction_label="NO_VALIDATED_DETECTOR",
        confidence=None,
        processing_time_seconds=0.05,
        details={
            "message": "Failed to extract or sample usable frames from input video.",
            "error": "frame_extraction_failed",
        }
    )

    monkeypatch.setattr("forensics.service.run_video_ml_inference", lambda p: failed_result)

    res = ForensicService.analyze_media(dummy_video)

    # Verify that:
    # 1. Failure is NOT treated as a valid negative inference (NO_ANOMALY_DETECTED)
    assert res.manipulation_forensic_status != STATUS_MANIPULATION_NO_ANOMALY
    # 2. Status defaults safely to NO_VALIDATED_DETECTOR
    assert res.manipulation_forensic_status == STATUS_MANIPULATION_NO_DETECTOR
    # 3. Failure does NOT falsely cause media to be classified as MANIPULATION_SUSPECTED
    assert res.manipulation_forensic_status != STATUS_MANIPULATION_SUSPECTED
    # 4. Service records an appropriate limitation
    assert any("Failed to extract or sample usable frames" in lim for lim in res.limitations)

    # Case 2: Genuine successful negative inference produces NO_ANOMALY_DETECTED
    genuine_clean_result = MLInferenceResult(
        model_name="DeepfakeBench Xception",
        available=True,
        status="NO_MANIPULATION_DETECTED",
        prediction_label="NO_MANIPULATION_DETECTED",
        confidence=0.04,
        processing_time_seconds=0.10,
        details={"fake_probability_mean": 0.04}
    )
    monkeypatch.setattr("forensics.service.run_video_ml_inference", lambda p: genuine_clean_result)

    res_clean = ForensicService.analyze_media(dummy_video)
    assert res_clean.manipulation_forensic_status == STATUS_MANIPULATION_NO_ANOMALY

    # Case 3: Genuine positive manipulation inference produces MANIPULATION_SUSPECTED
    genuine_fake_result = MLInferenceResult(
        model_name="DeepfakeBench Xception",
        available=True,
        status="MANIPULATION_SUSPECTED",
        prediction_label="deepfake",
        confidence=0.96,
        processing_time_seconds=0.10,
        details={"fake_probability_mean": 0.96}
    )
    monkeypatch.setattr("forensics.service.run_video_ml_inference", lambda p: genuine_fake_result)

    res_fake = ForensicService.analyze_media(dummy_video)
    assert res_fake.manipulation_forensic_status == STATUS_MANIPULATION_SUSPECTED


# ==============================================================================
# 13. Non-Finite & Malformed ML Output Safe Propagation Test (Phase 0.1.7)
# ==============================================================================

def test_non_finite_and_malformed_ml_output_safe_propagation(monkeypatch, tmp_path):
    from forensics.models import MLInferenceResult
    from forensics.service import (
        ForensicService,
        STATUS_MANIPULATION_NO_ANOMALY,
        STATUS_MANIPULATION_NO_DETECTOR,
        STATUS_MANIPULATION_SUSPECTED,
    )

    dummy_video = tmp_path / "numerical_fail.mp4"
    dummy_video.write_bytes(b"DUMMY_MP4_CONTENT")

    class MockProbe:
        has_video = True
        has_audio = False
        duration = 5.0
        width = 1280
        height = 720
        format_name = "mov,mp4,m4a,3gp,3g2,mj2"
        streams = []

    monkeypatch.setattr("forensics.service.MediaService.probe_file", lambda p: MockProbe())
    monkeypatch.setattr("forensics.service.extract_metadata_signals", lambda p: [])
    monkeypatch.setattr("forensics.service.extract_video_signals", lambda p, timeout=15.0: [])

    # Case 1: Video ML returns NON_FINITE_MODEL_OUTPUT
    non_finite_res = MLInferenceResult(
        model_name="DeepfakeBench Xception",
        available=False,
        status="NON_FINITE_MODEL_OUTPUT",
        prediction_label="NO_VALIDATED_DETECTOR",
        confidence=None,
        processing_time_seconds=0.05,
        details={
            "message": "Model inference produced non-finite logits (NaN or Inf).",
            "error": "non_finite_logits",
        }
    )
    monkeypatch.setattr("forensics.service.run_video_ml_inference", lambda p: non_finite_res)

    res_nf = ForensicService.analyze_media(dummy_video)
    assert res_nf.manipulation_forensic_status != STATUS_MANIPULATION_NO_ANOMALY
    assert res_nf.manipulation_forensic_status == STATUS_MANIPULATION_NO_DETECTOR
    assert res_nf.manipulation_forensic_status != STATUS_MANIPULATION_SUSPECTED
    assert any("non-finite" in lim.lower() for lim in res_nf.limitations)

    # Case 2: Video ML returns MALFORMED_MODEL_OUTPUT
    malformed_res = MLInferenceResult(
        model_name="DeepfakeBench Xception",
        available=False,
        status="MALFORMED_MODEL_OUTPUT",
        prediction_label="NO_VALIDATED_DETECTOR",
        confidence=None,
        processing_time_seconds=0.05,
        details={
            "message": "Model output shape torch.Size([0, 2]) is invalid (expected [N, 2]).",
            "error": "malformed_output_shape",
        }
    )
    monkeypatch.setattr("forensics.service.run_video_ml_inference", lambda p: malformed_res)

    res_mal = ForensicService.analyze_media(dummy_video)
    assert res_mal.manipulation_forensic_status != STATUS_MANIPULATION_NO_ANOMALY
    assert res_mal.manipulation_forensic_status == STATUS_MANIPULATION_NO_DETECTOR
    assert res_mal.manipulation_forensic_status != STATUS_MANIPULATION_SUSPECTED
    assert any("invalid" in lim.lower() for lim in res_mal.limitations)




