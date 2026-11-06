import os
import shutil
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from backend.main import app
from media.ffmpeg import get_ffmpeg_status
from provenance.models import generate_asset_id
from provenance.service import ProvenanceService
from watermark.video.custom_engine import CustomVideoWatermarkEngine
from watermark.video.detect import detect_video_watermark
from watermark.video.embed import embed_video_watermark
from watermark.video.factory import get_video_watermark_engine
from watermark.video.payload import decode_watermark_token_payload, encode_watermark_token_payload
from watermark.video.videoseal_engine import MetaVideoSealEngine

client = TestClient(app)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_MEDIA_DIR = PROJECT_ROOT / "data" / "temp" / "test_media"
PROTECTED_DIR = PROJECT_ROOT / "data" / "protected"


def is_ffmpeg_available() -> bool:
    status = get_ffmpeg_status()
    return bool(status["ffmpeg_available"] and status["ffprobe_available"])


# ==============================================================================
# Suite A: Payload & Repetition Token Encoding Tests
# ==============================================================================

def test_token_repetition_payload_encoding_decoding():
    token_hex = "A3F8"
    bits = encode_watermark_token_payload(token_hex)
    assert len(bits) == 256

    decoded_token = decode_watermark_token_payload(bits, token_bits=16)
    assert decoded_token == token_hex


def test_token_repetition_majority_voting_resilience():
    token_hex = "2D07"
    bits = encode_watermark_token_payload(token_hex)
    
    # Inject 30% bit noise into the 256-bit repetition array
    noisy_bits = [1 - b if (i % 3 == 0) else b for i, b in enumerate(bits)]
    decoded_token = decode_watermark_token_payload(noisy_bits, token_bits=16)
    assert decoded_token == token_hex


# ==============================================================================
# Suite B: Custom Engine Spatial Baseline Tests
# ==============================================================================

def test_custom_engine_watermark_embed_and_detect(tmp_path):
    if not is_ffmpeg_available():
        pytest.skip("FFmpeg/FFprobe binaries not available")

    mp4_file = TEST_MEDIA_DIR / "test_combined.mp4"
    if not mp4_file.exists():
        pytest.skip("Test combined MP4 file not generated")

    record = ProvenanceService.register_media(mp4_file)
    asset_id = record.asset_id

    output_wm_path = tmp_path / "custom_wm_output.mp4"
    embed_res = embed_video_watermark(mp4_file, output_wm_path, asset_id, engine_type="custom")
    assert embed_res["status"] == "ok"
    assert embed_res["engine"] == "custom"
    assert embed_res["has_watermark"] is True
    assert output_wm_path.exists()

    detect_res = detect_video_watermark(output_wm_path, engine_type="custom")
    assert detect_res["status"] == "ok"
    assert detect_res["engine"] == "custom"
    assert detect_res["watermark_detected"] is True
    assert detect_res["recovered_asset_id"] == asset_id
    assert detect_res["provenance_record"] is not None


# ==============================================================================
# Suite C: Engine Factory Tests
# ==============================================================================

def test_engine_factory_resolution():
    custom_eng = get_video_watermark_engine("custom")
    assert isinstance(custom_eng, CustomVideoWatermarkEngine)
    assert custom_eng.engine_name == "custom"

    vseal_eng = get_video_watermark_engine("videoseal")
    assert isinstance(vseal_eng, MetaVideoSealEngine)
    assert vseal_eng.engine_name == "videoseal"

    with pytest.raises(ValueError):
        get_video_watermark_engine("unknown_engine_type")


# ==============================================================================
# Suite D: Meta VideoSeal Engine Integration & Recovery Tests
# ==============================================================================

def test_meta_videoseal_engine_execution(tmp_path):
    if not is_ffmpeg_available():
        pytest.skip("FFmpeg/FFprobe binaries not available")

    mp4_src = TEST_MEDIA_DIR / "test_combined.mp4"
    if not mp4_src.exists():
        pytest.skip("Test combined MP4 file not generated")

    mp4_file = tmp_path / "vseal_exec_input.mp4"
    shutil.copy(mp4_src, mp4_file)

    record = ProvenanceService.register_media(mp4_file)
    asset_id = record.asset_id

    output_wm_path = tmp_path / "vseal_wm_output.mp4"

    embed_res = embed_video_watermark(mp4_file, output_wm_path, asset_id, engine_type="videoseal")
    assert embed_res["status"] == "ok"
    assert embed_res["engine"] == "videoseal"
    assert embed_res["has_watermark"] is True
    assert output_wm_path.exists()

    detect_res = detect_video_watermark(output_wm_path, engine_type="videoseal")
    assert detect_res["status"] == "ok"
    assert detect_res["engine"] == "videoseal"
    assert detect_res["watermark_detected"] is True
    assert detect_res["recovered_asset_id"] == asset_id
    assert detect_res["provenance_record"] is not None


# ==============================================================================
# Suite E: Dual Engine API Endpoint Tests
# ==============================================================================

def test_api_video_watermark_custom_engine():
    if not is_ffmpeg_available():
        pytest.skip("FFmpeg/FFprobe binaries not available")

    mp4_file = TEST_MEDIA_DIR / "test_combined.mp4"
    if not mp4_file.exists():
        pytest.skip("Test combined MP4 file not generated")

    record = ProvenanceService.register_media(mp4_file)
    asset_id = record.asset_id

    # 1. API Embed with engine=custom
    with open(mp4_file, "rb") as f:
        res_embed = client.post(
            "/api/watermark/video/embed",
            files={"file": ("api_custom_input.mp4", f, "video/mp4")},
            data={"asset_id": asset_id, "engine": "custom"}
        )

    assert res_embed.status_code == 200
    embed_data = res_embed.json()
    assert embed_data["status"] == "ok"
    assert embed_data["engine"] == "custom"
    wm_output_path = Path(embed_data["output_path"])
    assert wm_output_path.exists()

    # 2. API Detect with engine=custom
    with open(wm_output_path, "rb") as f:
        res_detect = client.post(
            "/api/watermark/video/detect",
            files={"file": ("custom_wm_input.mp4", f, "video/mp4")},
            data={"engine": "custom"}
        )

    assert res_detect.status_code == 200
    det_data = res_detect.json()
    assert det_data["status"] == "ok"
    assert det_data["engine"] == "custom"
    assert det_data["watermark_detected"] is True
    assert det_data["recovered_asset_id"] == asset_id


def test_api_video_watermark_videoseal_engine(tmp_path):
    if not is_ffmpeg_available():
        pytest.skip("FFmpeg/FFprobe binaries not available")

    mp4_src = TEST_MEDIA_DIR / "test_combined.mp4"
    if not mp4_src.exists():
        pytest.skip("Test combined MP4 file not generated")

    mp4_file = tmp_path / "vseal_api_input.mp4"
    shutil.copy(mp4_src, mp4_file)

    record = ProvenanceService.register_media(mp4_file)
    asset_id = record.asset_id

    # 1. API Embed with engine=videoseal
    with open(mp4_file, "rb") as f:
        res_embed = client.post(
            "/api/watermark/video/embed",
            files={"file": ("api_vseal_input.mp4", f, "video/mp4")},
            data={"asset_id": asset_id, "engine": "videoseal"}
        )

    assert res_embed.status_code == 200
    embed_data = res_embed.json()
    assert embed_data["status"] == "ok"
    assert embed_data["engine"] == "videoseal"
    wm_output_path = Path(embed_data["output_path"])
    assert wm_output_path.exists()

    # 2. API Detect with engine=videoseal
    with open(wm_output_path, "rb") as f:
        res_detect = client.post(
            "/api/watermark/video/detect",
            files={"file": ("vseal_wm_input.mp4", f, "video/mp4")},
            data={"engine": "videoseal"}
        )

    assert res_detect.status_code == 200
    det_data = res_detect.json()
    assert det_data["status"] == "ok"
    assert det_data["engine"] == "videoseal"
    assert det_data["watermark_detected"] is True
    assert det_data["recovered_asset_id"] == asset_id


# ==============================================================================
# Suite F: VideoSeal Confidence Floor & Hamming Fallback Regression Tests
# ==============================================================================

def test_videoseal_confidence_not_clamped_to_098(tmp_path):
    """Regression Test: Verify VideoSeal confidence accurately reports genuine detector
    probability (e.g. 0.70) without being artificially clamped to 0.98."""
    if not is_ffmpeg_available():
        pytest.skip("FFmpeg/FFprobe binaries not available")

    mp4_src = TEST_MEDIA_DIR / "test_combined.mp4"
    if not mp4_src.exists():
        pytest.skip("Test combined MP4 file not generated")

    mp4_file = tmp_path / "vseal_conf_test.mp4"
    shutil.copy(mp4_src, mp4_file)

    rec = ProvenanceService.register_media(mp4_file)
    token = ProvenanceService.ensure_watermark_token(rec.asset_id)

    import torch
    from unittest.mock import MagicMock
    from watermark.video.videoseal_engine import MetaVideoSealEngine
    from watermark.video.payload import encode_watermark_token_payload

    engine = MetaVideoSealEngine()
    mock_model = MagicMock()
    # Logit 0.8473 gives sigmoid ~= 0.70 (below 0.98)
    token_bits = encode_watermark_token_payload(token)
    bit_logits = [3.0 if b == 1 else -3.0 for b in token_bits]
    preds = torch.tensor([[0.8473] + bit_logits])
    mock_model.detect.return_value = {"preds": preds}
    engine._load_model = MagicMock(return_value=mock_model)

    res = engine.detect(mp4_file)
    assert res.watermark_detected is True
    assert res.recovered_asset_id == rec.asset_id
    assert res.confidence == 0.70
    assert res.confidence < 0.98, "VideoSeal confidence must not be artificially clamped to 0.98"
    assert res.details["matching_method"] == "exact"
    assert res.details["hamming_distance"] == 0


def test_videoseal_hamming_fallback_metadata_separated(tmp_path):
    """Regression Test: Verify that when exact token lookup fails but Hamming fallback matches,
    it records matching_method='hamming_fallback', distance > 0, and explicit message."""
    if not is_ffmpeg_available():
        pytest.skip("FFmpeg/FFprobe binaries not available")

    mp4_src = TEST_MEDIA_DIR / "test_combined.mp4"
    if not mp4_src.exists():
        pytest.skip("Test combined MP4 file not generated")

    mp4_file = tmp_path / "vseal_hfall_test.mp4"
    shutil.copy(mp4_src, mp4_file)

    rec = ProvenanceService.register_media(mp4_file)
    token = ProvenanceService.ensure_watermark_token(rec.asset_id)
    # Find a 1-bit Hamming distance discrepancy that uniquely resolves to rec via Hamming fallback
    val = int(token, 16)
    from provenance.repository import get_asset_by_watermark_token
    flipped_token = None
    for bit_idx in range(16):
        cand_val = val ^ (1 << bit_idx)
        cand_hex = f"{cand_val:04X}"
        if get_asset_by_watermark_token(cand_hex, allow_hamming=False) is None:
            matched = get_asset_by_watermark_token(cand_hex, allow_hamming=True)
            if matched and matched.asset_id == rec.asset_id:
                flipped_token = cand_hex
                break

    assert flipped_token is not None, "Could not find a 1-bit flip that uniquely matches the test asset via Hamming"

    import torch
    from unittest.mock import MagicMock
    from watermark.video.videoseal_engine import MetaVideoSealEngine
    from watermark.video.payload import encode_watermark_token_payload

    engine = MetaVideoSealEngine()
    mock_model = MagicMock()
    flipped_bits = encode_watermark_token_payload(flipped_token)
    bit_logits = [3.0 if b == 1 else -3.0 for b in flipped_bits]
    preds = torch.tensor([[1.2] + bit_logits])
    mock_model.detect.return_value = {"preds": preds}
    engine._load_model = MagicMock(return_value=mock_model)

    res = engine.detect(mp4_file)
    assert res.watermark_detected is True
    assert res.recovered_asset_id == rec.asset_id
    assert res.details["matching_method"] == "hamming_fallback"
    assert res.details["hamming_distance"] == 1
    assert res.details["watermark_token"] == token
    assert res.details["recovered_token"] == flipped_token
    assert "Hamming distance fallback" in res.message


# ==============================================================================
# Suite G: Tier 2A VideoSeal Neural Presence Gating & Robustness Tests
# ==============================================================================

def test_videoseal_near_zero_presence_with_exact_token_match_rejected(tmp_path):
    """Regression Test 3: Near-zero neural presence (presence=0.0474 < 0.70) must NOT establish
    watermark detection, even if decoded token exactly matches a registered asset."""
    if not is_ffmpeg_available():
        pytest.skip("FFmpeg/FFprobe binaries not available")

    mp4_src = TEST_MEDIA_DIR / "test_combined.mp4"
    if not mp4_src.exists():
        pytest.skip("Test combined MP4 file not generated")

    mp4_file = tmp_path / "vseal_low_exact.mp4"
    shutil.copy(mp4_src, mp4_file)

    rec = ProvenanceService.register_media(mp4_file)
    token = ProvenanceService.ensure_watermark_token(rec.asset_id)

    import torch
    from unittest.mock import MagicMock
    from watermark.video.videoseal_engine import MetaVideoSealEngine
    from watermark.video.payload import encode_watermark_token_payload

    engine = MetaVideoSealEngine()
    mock_model = MagicMock()
    # Logit -3.0 -> Sigmoid(-3.0) ~= 0.0474 (far below 0.70 threshold)
    token_bits = encode_watermark_token_payload(token)
    bit_logits = [3.0 if b == 1 else -3.0 for b in token_bits]
    preds = torch.tensor([[-3.0] + bit_logits])
    mock_model.detect.return_value = {"preds": preds}
    engine._load_model = MagicMock(return_value=mock_model)

    res = engine.detect(mp4_file)
    assert res.watermark_detected is False
    assert res.recovered_asset_id is None
    assert res.provenance_record is None
    assert res.confidence == 0.0474
    assert res.details["threshold_passed"] is False
    assert res.details["rejected_reason"] == "presence_below_threshold"
    assert "below detection threshold" in res.message


def test_videoseal_near_zero_presence_with_hamming_fallback_rejected(tmp_path):
    """Regression Test 4: Near-zero neural presence (presence=0.0759 < 0.70) must NOT establish
    watermark detection when candidate token matches via Hamming fallback."""
    if not is_ffmpeg_available():
        pytest.skip("FFmpeg/FFprobe binaries not available")

    mp4_src = TEST_MEDIA_DIR / "test_combined.mp4"
    if not mp4_src.exists():
        pytest.skip("Test combined MP4 file not generated")

    mp4_file = tmp_path / "vseal_low_hfall.mp4"
    shutil.copy(mp4_src, mp4_file)

    rec = ProvenanceService.register_media(mp4_file)
    token = ProvenanceService.ensure_watermark_token(rec.asset_id)

    val = int(token, 16)
    from provenance.repository import get_asset_by_watermark_token
    flipped_token = None
    for bit_idx in range(16):
        cand_val = val ^ (1 << bit_idx)
        cand_hex = f"{cand_val:04X}"
        if get_asset_by_watermark_token(cand_hex, allow_hamming=False) is None:
            matched = get_asset_by_watermark_token(cand_hex, allow_hamming=True)
            if matched and matched.asset_id == rec.asset_id:
                flipped_token = cand_hex
                break

    assert flipped_token is not None

    import torch
    from unittest.mock import MagicMock
    from watermark.video.videoseal_engine import MetaVideoSealEngine
    from watermark.video.payload import encode_watermark_token_payload

    engine = MetaVideoSealEngine()
    mock_model = MagicMock()
    # Logit -2.5 -> Sigmoid(-2.5) ~= 0.0759
    flipped_bits = encode_watermark_token_payload(flipped_token)
    bit_logits = [3.0 if b == 1 else -3.0 for b in flipped_bits]
    preds = torch.tensor([[-2.5] + bit_logits])
    mock_model.detect.return_value = {"preds": preds}
    engine._load_model = MagicMock(return_value=mock_model)

    res = engine.detect(mp4_file)
    assert res.watermark_detected is False
    assert res.recovered_asset_id is None
    assert res.provenance_record is None
    assert res.confidence == 0.0759
    assert res.details["threshold_passed"] is False
    assert res.details["rejected_reason"] == "presence_below_threshold"


def test_videoseal_non_finite_detector_output(tmp_path):
    """Regression Test 5: Detector returning NaN or Inf is handled gracefully without crashing,
    reporting watermark_detected=False and confidence=0.0."""
    if not is_ffmpeg_available():
        pytest.skip("FFmpeg/FFprobe binaries not available")

    mp4_src = TEST_MEDIA_DIR / "test_combined.mp4"
    if not mp4_src.exists():
        pytest.skip("Test combined MP4 file not generated")

    mp4_file = tmp_path / "vseal_nan.mp4"
    shutil.copy(mp4_src, mp4_file)

    import torch
    from unittest.mock import MagicMock
    from watermark.video.videoseal_engine import MetaVideoSealEngine

    engine = MetaVideoSealEngine()
    mock_model = MagicMock()
    # NaN in presence logit
    preds = torch.tensor([[float("nan")] + [1.0] * 256])
    mock_model.detect.return_value = {"preds": preds}
    engine._load_model = MagicMock(return_value=mock_model)

    res = engine.detect(mp4_file)
    assert res.watermark_detected is False
    assert res.confidence == 0.0
    assert res.details["error"] == "non_finite_presence"


def test_videoseal_detector_failure_or_missing(tmp_path):
    """Regression Test 6: Detector execution exception does not crash and reports detection failure."""
    if not is_ffmpeg_available():
        pytest.skip("FFmpeg/FFprobe binaries not available")

    mp4_src = TEST_MEDIA_DIR / "test_combined.mp4"
    if not mp4_src.exists():
        pytest.skip("Test combined MP4 file not generated")

    mp4_file = tmp_path / "vseal_crash.mp4"
    shutil.copy(mp4_src, mp4_file)

    from unittest.mock import MagicMock
    from watermark.video.videoseal_engine import MetaVideoSealEngine

    engine = MetaVideoSealEngine()
    mock_model = MagicMock()
    mock_model.detect.side_effect = RuntimeError("VideoSeal forward execution error")
    engine._load_model = MagicMock(return_value=mock_model)

    res = engine.detect(mp4_file)
    assert res.watermark_detected is False
    assert res.confidence == 0.0
    assert "failed" in res.message.lower()


def test_videoseal_strong_presence_unregistered_token(tmp_path):
    """Regression Test 7: Strong presence (presence=0.90 >= 0.70) with an unmapped token
    returns watermark_detected=False and recovered_asset_id=None."""
    if not is_ffmpeg_available():
        pytest.skip("FFmpeg/FFprobe binaries not available")

    mp4_src = TEST_MEDIA_DIR / "test_combined.mp4"
    if not mp4_src.exists():
        pytest.skip("Test combined MP4 file not generated")

    mp4_file = tmp_path / "vseal_unreg.mp4"
    shutil.copy(mp4_src, mp4_file)

    import torch
    from unittest.mock import MagicMock, patch
    from watermark.video.videoseal_engine import MetaVideoSealEngine
    from watermark.video.payload import encode_watermark_token_payload

    engine = MetaVideoSealEngine()
    unregistered_token = "F0F0"

    mock_model = MagicMock()
    # Logit 2.2 -> Sigmoid(2.2) ~= 0.9002
    token_bits = encode_watermark_token_payload(unregistered_token)
    bit_logits = [3.0 if b == 1 else -3.0 for b in token_bits]
    preds = torch.tensor([[2.2] + bit_logits])
    mock_model.detect.return_value = {"preds": preds}
    engine._load_model = MagicMock(return_value=mock_model)

    with patch("watermark.video.videoseal_engine.get_asset_by_watermark_token", return_value=None), \
         patch("provenance.service.ProvenanceService.find_asset_by_watermark_token", return_value=None):
        res = engine.detect(mp4_file)

    assert res.watermark_detected is False
    assert res.confidence == 0.9002
    assert res.recovered_asset_id is None
    assert res.details["threshold_passed"] is True
    assert "no registered" in res.message.lower()


def test_videoseal_rejected_detection_prevents_verified_derivative_link(tmp_path):
    """Regression Test 8: Sub-threshold VideoSeal detection on a file with registered token match
    must NOT establish a verified derivative link downstream in VerificationService."""
    if not is_ffmpeg_available():
        pytest.skip("FFmpeg/FFprobe binaries not available")

    mp4_src = TEST_MEDIA_DIR / "test_combined.mp4"
    if not mp4_src.exists():
        pytest.skip("Test combined MP4 file not generated")

    mp4_file = tmp_path / "vseal_downstream_orig.mp4"
    shutil.copy(mp4_src, mp4_file)
    rec = ProvenanceService.register_media(mp4_file)
    token = ProvenanceService.ensure_watermark_token(rec.asset_id)

    # Create a derivative file with differing hash from test_video.mp4 or copy
    diff_src = TEST_MEDIA_DIR / "test_video.mp4"
    diff_file = tmp_path / "vseal_downstream_diff.mp4"
    if diff_src.exists():
        shutil.copy(diff_src, diff_file)
    else:
        shutil.copy(mp4_src, diff_file)

    import torch
    from unittest.mock import patch, MagicMock
    from provenance.verification import VerificationService, STATE_VERIFIED_DERIVATIVE, STATE_UNKNOWN_PROVENANCE
    from watermark.video.payload import encode_watermark_token_payload
    from forensics.models import UnifiedForensicResult

    mock_model = MagicMock()
    # Logit -3.0 -> Sigmoid(-3.0) ~= 0.0474 (rejected)
    token_bits = encode_watermark_token_payload(token)
    bit_logits = [3.0 if b == 1 else -3.0 for b in token_bits]
    preds = torch.tensor([[-3.0] + bit_logits])
    mock_model.detect.return_value = {"preds": preds}

    mock_clean_forensic = UnifiedForensicResult(
        forensic_status="LOW_RISK",
        risk_score=0.05,
        risk_score_type="heuristic",
        generic_forensic_status="LOW_GENERIC_ANOMALY",
        manipulation_forensic_status="NO_VALIDATED_DETECTOR",
        video_signals=[],
        audio_signals=[],
        metadata_signals=[],
        ml_signals=[],
        limitations=[],
        processing_time_seconds=0.05,
    )

    with patch("watermark.video.videoseal_engine.MetaVideoSealEngine._load_model", return_value=mock_model), \
         patch("provenance.verification.ForensicService.analyze_media", return_value=mock_clean_forensic):
        ver_res = VerificationService.verify_media(diff_file, video_engine="videoseal")
        # Must NOT classify as VERIFIED DERIVATIVE because presence check failed!
        assert ver_res.classification != STATE_VERIFIED_DERIVATIVE
        assert ver_res.classification == STATE_UNKNOWN_PROVENANCE
        assert ver_res.watermark_evidence.status == "NOT_DETECTED"


def test_videoseal_threshold_validation_constructor():
    """Verify MetaVideoSealEngine constructor rejects invalid confidence thresholds."""
    from watermark.video.videoseal_engine import MetaVideoSealEngine

    # Out of range [-1.0, 1.5]
    with pytest.raises(ValueError, match="between 0.0 and 1.0 inclusive"):
        MetaVideoSealEngine(confidence_threshold=-1.0)
    with pytest.raises(ValueError, match="between 0.0 and 1.0 inclusive"):
        MetaVideoSealEngine(confidence_threshold=1.5)

    # Non-finite (NaN, +inf, -inf)
    with pytest.raises(ValueError, match="finite number"):
        MetaVideoSealEngine(confidence_threshold=float("nan"))
    with pytest.raises(ValueError, match="finite number"):
        MetaVideoSealEngine(confidence_threshold=float("inf"))
    with pytest.raises(ValueError, match="finite number"):
        MetaVideoSealEngine(confidence_threshold=float("-inf"))

    # Booleans and strings rejected
    with pytest.raises(ValueError, match="numeric float or int"):
        MetaVideoSealEngine(confidence_threshold=True)
    with pytest.raises(ValueError, match="numeric float or int"):
        MetaVideoSealEngine(confidence_threshold="0.5")

    # Valid thresholds accepted
    e1 = MetaVideoSealEngine(confidence_threshold=0.0)
    assert e1.confidence_threshold == 0.0
    e2 = MetaVideoSealEngine(confidence_threshold=1.0)
    assert e2.confidence_threshold == 1.0
    e3 = MetaVideoSealEngine(confidence_threshold=0.65)
    assert e3.confidence_threshold == 0.65
    e4 = MetaVideoSealEngine()
    assert e4.confidence_threshold is None


def test_videoseal_threshold_validation_per_call(tmp_path):
    """Verify MetaVideoSealEngine.detect rejects invalid per-call confidence thresholds."""
    from watermark.video.videoseal_engine import MetaVideoSealEngine

    dummy_video = tmp_path / "dummy.mp4"
    engine = MetaVideoSealEngine()

    # Out of range [-1.0, 1.5]
    with pytest.raises(ValueError, match="between 0.0 and 1.0 inclusive"):
        engine.detect(dummy_video, confidence_threshold=-1.0)
    with pytest.raises(ValueError, match="between 0.0 and 1.0 inclusive"):
        engine.detect(dummy_video, confidence_threshold=1.5)

    # Non-finite (NaN, +inf, -inf)
    with pytest.raises(ValueError, match="finite number"):
        engine.detect(dummy_video, confidence_threshold=float("nan"))
    with pytest.raises(ValueError, match="finite number"):
        engine.detect(dummy_video, confidence_threshold=float("inf"))
    with pytest.raises(ValueError, match="finite number"):
        engine.detect(dummy_video, confidence_threshold=float("-inf"))

    # Booleans and strings rejected
    with pytest.raises(ValueError, match="numeric float or int"):
        engine.detect(dummy_video, confidence_threshold=True)
    with pytest.raises(ValueError, match="numeric float or int"):
        engine.detect(dummy_video, confidence_threshold="0.5")


def test_videoseal_per_call_threshold_overrides_constructor(tmp_path):
    """Verify that a valid per-call threshold overrides the constructor-configured threshold."""
    if not is_ffmpeg_available():
        pytest.skip("FFmpeg/FFprobe binaries not available")

    mp4_src = TEST_MEDIA_DIR / "test_combined.mp4"
    if not mp4_src.exists():
        pytest.skip("Test combined MP4 file not generated")

    mp4_file = tmp_path / "vseal_override.mp4"
    shutil.copy(mp4_src, mp4_file)

    rec = ProvenanceService.register_media(mp4_file)
    token = ProvenanceService.ensure_watermark_token(rec.asset_id)

    import torch
    from unittest.mock import MagicMock
    from watermark.video.videoseal_engine import MetaVideoSealEngine
    from watermark.video.payload import encode_watermark_token_payload

    # Constructor threshold is set high (0.85)
    engine = MetaVideoSealEngine(confidence_threshold=0.85)

    # Mock model returning presence = 0.70 (logit ~0.8473)
    mock_model = MagicMock()
    token_bits = encode_watermark_token_payload(token)
    bit_logits = [3.0 if b == 1 else -3.0 for b in token_bits]
    preds = torch.tensor([[0.8473] + bit_logits])
    mock_model.detect.return_value = {"preds": preds}
    engine._load_model = MagicMock(return_value=mock_model)

    # 1. Without per-call override, effective threshold is 0.85 -> presence 0.70 fails!
    res_default = engine.detect(mp4_file)
    assert res_default.watermark_detected is False
    assert res_default.details["detection_threshold"] == 0.85
    assert res_default.details["threshold_passed"] is False

    # 2. With per-call override = 0.60, effective threshold becomes 0.60 -> presence 0.70 passes!
    res_override = engine.detect(mp4_file, confidence_threshold=0.60)
    assert res_override.watermark_detected is True
    assert res_override.details["detection_threshold"] == 0.60
    assert res_override.details["threshold_passed"] is True
    assert res_override.recovered_asset_id == rec.asset_id

