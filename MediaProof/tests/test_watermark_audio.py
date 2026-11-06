import os
import shutil
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from backend.main import app
from media.ffmpeg import get_ffmpeg_status
from provenance.service import ProvenanceService
from watermark.audio.factory import get_audio_watermark_engine
from watermark.audio.custom_engine import CustomAudioWatermarkEngine
from watermark.audio.audioseal_engine import MetaAudioSealEngine
from watermark.audio.embed import embed_audio_watermark
from watermark.audio.detect import detect_audio_watermark

client = TestClient(app)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_MEDIA_DIR = PROJECT_ROOT / "data" / "temp" / "test_media"


def is_ffmpeg_available() -> bool:
    status = get_ffmpeg_status()
    return bool(status["ffmpeg_available"] and status["ffprobe_available"])


# ==============================================================================
# Suite A: Custom Audio Engine Unit Tests
# ==============================================================================

def create_test_wav(path: Path) -> Path:
    import struct, random, wave
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


def test_custom_audio_engine_embed_and_detect(tmp_path):
    if not is_ffmpeg_available():
        pytest.skip("FFmpeg/FFprobe binaries not available")

    audio_file = create_test_wav(tmp_path / "custom_src.wav")

    record = ProvenanceService.register_media(audio_file)
    asset_id = record.asset_id

    output_wm_path = tmp_path / "custom_audio_output.wav"
    embed_res = embed_audio_watermark(audio_file, output_wm_path, asset_id, engine_type="custom")
    assert embed_res["status"] == "ok"
    assert embed_res["engine"] == "custom"
    assert embed_res["has_watermark"] is True
    assert output_wm_path.exists()

    detect_res = detect_audio_watermark(output_wm_path, engine_type="custom")
    assert detect_res["status"] == "ok"
    assert detect_res["engine"] == "custom"
    assert detect_res["watermark_detected"] is True
    assert detect_res["recovered_asset_id"] == asset_id
    assert detect_res["provenance_record"] is not None


# ==============================================================================
# Suite B: Engine Factory Tests
# ==============================================================================

def test_audio_engine_factory_resolution():
    custom_eng = get_audio_watermark_engine("custom")
    assert isinstance(custom_eng, CustomAudioWatermarkEngine)
    assert custom_eng.engine_name == "custom"

    aseal_eng = get_audio_watermark_engine("audioseal")
    assert isinstance(aseal_eng, MetaAudioSealEngine)
    assert aseal_eng.engine_name == "audioseal"

    with pytest.raises(ValueError):
        get_audio_watermark_engine("unknown_engine_type")


# ==============================================================================
# Suite C: Meta AudioSeal Neural Engine Execution Tests
# ==============================================================================

def test_meta_audioseal_engine_execution(tmp_path):
    if not is_ffmpeg_available():
        pytest.skip("FFmpeg/FFprobe binaries not available")

    audio_file = create_test_wav(tmp_path / "audioseal_src.wav")

    record = ProvenanceService.register_media(audio_file)
    asset_id = record.asset_id

    output_wm_path = tmp_path / "audioseal_output.wav"

    embed_res = embed_audio_watermark(audio_file, output_wm_path, asset_id, engine_type="audioseal")
    assert embed_res["status"] == "ok"
    assert embed_res["engine"] == "audioseal"
    assert embed_res["has_watermark"] is True
    assert output_wm_path.exists()

    detect_res = detect_audio_watermark(output_wm_path, engine_type="audioseal")
    assert detect_res["status"] == "ok"
    assert detect_res["engine"] == "audioseal"
    assert detect_res["watermark_detected"] is True
    assert detect_res["recovered_asset_id"] == asset_id
    assert detect_res["provenance_record"] is not None


# ==============================================================================
# Suite D: Dual Engine Audio REST API Endpoint Tests
# ==============================================================================

def test_api_audio_watermark_custom_engine(tmp_path):
    if not is_ffmpeg_available():
        pytest.skip("FFmpeg/FFprobe binaries not available")

    audio_file = create_test_wav(tmp_path / "api_custom_src.wav")

    record = ProvenanceService.register_media(audio_file)
    asset_id = record.asset_id

    # 1. API Embed
    with open(audio_file, "rb") as f:
        res_embed = client.post(
            "/api/watermark/audio/embed",
            files={"file": ("api_custom_audio.wav", f, "audio/wav")},
            data={"asset_id": asset_id, "engine": "custom"}
        )

    assert res_embed.status_code == 200
    embed_data = res_embed.json()
    assert embed_data["status"] == "ok"
    assert embed_data["engine"] == "custom"
    wm_output_path = Path(embed_data["output_path"])
    assert wm_output_path.exists()

    # 2. API Detect
    with open(wm_output_path, "rb") as f:
        res_detect = client.post(
            "/api/watermark/audio/detect",
            files={"file": ("custom_wm_audio.wav", f, "audio/wav")},
            data={"engine": "custom"}
        )

    assert res_detect.status_code == 200
    det_data = res_detect.json()
    assert det_data["status"] == "ok"
    assert det_data["engine"] == "custom"
    assert det_data["watermark_detected"] is True
    assert det_data["recovered_asset_id"] == asset_id


def test_api_audio_watermark_audioseal_engine(tmp_path):
    if not is_ffmpeg_available():
        pytest.skip("FFmpeg/FFprobe binaries not available")

    audio_file = create_test_wav(tmp_path / "api_aseal_src.wav")

    record = ProvenanceService.register_media(audio_file)
    asset_id = record.asset_id

    # 1. API Embed with engine=audioseal
    with open(audio_file, "rb") as f:
        res_embed = client.post(
            "/api/watermark/audio/embed",
            files={"file": ("api_aseal_audio.wav", f, "audio/wav")},
            data={"asset_id": asset_id, "engine": "audioseal"}
        )

    assert res_embed.status_code == 200
    embed_data = res_embed.json()
    assert embed_data["status"] == "ok"
    assert embed_data["engine"] == "audioseal"
    wm_output_path = Path(embed_data["output_path"])
    assert wm_output_path.exists()

    # 2. API Detect with engine=audioseal
    with open(wm_output_path, "rb") as f:
        res_detect = client.post(
            "/api/watermark/audio/detect",
            files={"file": ("aseal_wm_audio.wav", f, "audio/wav")},
            data={"engine": "audioseal"}
        )

    assert res_detect.status_code == 200
    det_data = res_detect.json()
    assert det_data["status"] == "ok"
    assert det_data["engine"] == "audioseal"
    assert det_data["watermark_detected"] is True
    assert det_data["recovered_asset_id"] == asset_id


# ==============================================================================
# Suite E: AudioSeal Confidence Floor & Hamming Fallback Regression Tests
# ==============================================================================

def test_audioseal_confidence_not_clamped_to_098(tmp_path):
    """Regression Test: Verify AudioSeal confidence accurately reports genuine detector
    probability (e.g. 0.7311) without being artificially clamped to 0.98."""
    if not is_ffmpeg_available():
        pytest.skip("FFmpeg/FFprobe binaries not available")

    import torch
    from unittest.mock import MagicMock
    from watermark.audio.audioseal_engine import MetaAudioSealEngine, hex_token_to_bits_tensor

    audio_file = create_test_wav(tmp_path / "test_aseal_conf.wav")
    rec = ProvenanceService.register_media(audio_file)
    token = ProvenanceService.ensure_watermark_token(rec.asset_id)

    engine = MetaAudioSealEngine()
    mock_det = MagicMock()
    # Sigmoid(1.0) ~= 0.7311 (strictly below 0.98)
    token_bits = hex_token_to_bits_tensor(token).float()
    mock_det.return_value = (torch.tensor([[1.0]]), token_bits)
    engine._load_models = MagicMock(return_value=(None, mock_det))

    res = engine.detect(audio_file)
    assert res.watermark_detected is True
    assert res.recovered_asset_id == rec.asset_id
    assert res.confidence == 0.7311
    assert res.confidence < 0.98, "AudioSeal confidence must not be artificially clamped to 0.98"
    assert res.details["matching_method"] == "exact"
    assert res.details["hamming_distance"] == 0


def test_audioseal_hamming_fallback_metadata_separated(tmp_path):
    """Regression Test: Verify that when exact token lookup fails but Hamming fallback matches,
    it records matching_method='hamming_fallback', distance > 0, and explicit message."""
    if not is_ffmpeg_available():
        pytest.skip("FFmpeg/FFprobe binaries not available")

    import torch
    from unittest.mock import MagicMock
    from watermark.audio.audioseal_engine import (
        MetaAudioSealEngine,
        hex_token_to_bits_tensor,
        bits_array_to_hex_token,
    )

    audio_file = create_test_wav(tmp_path / "test_aseal_hfall.wav")
    rec = ProvenanceService.register_media(audio_file)
    token = ProvenanceService.ensure_watermark_token(rec.asset_id)

    from provenance.repository import get_asset_by_watermark_token
    token_bits = hex_token_to_bits_tensor(token).float()
    flipped_bits = None
    raw_token = None
    for bit_idx in range(16):
        cand = token_bits.clone()
        cand[0, bit_idx] = 1.0 - cand[0, bit_idx]
        cand_hex = bits_array_to_hex_token(cand.squeeze(0).int().tolist())
        if get_asset_by_watermark_token(cand_hex, allow_hamming=False) is None:
            matched = get_asset_by_watermark_token(cand_hex, allow_hamming=True)
            if matched and matched.asset_id == rec.asset_id:
                flipped_bits = cand
                raw_token = cand_hex
                break

    assert flipped_bits is not None, "Could not find a 1-bit flip that uniquely matches the test asset via Hamming"

    engine = MetaAudioSealEngine()
    mock_det = MagicMock()
    mock_det.return_value = (torch.tensor([[1.5]]), flipped_bits)
    engine._load_models = MagicMock(return_value=(None, mock_det))

    res = engine.detect(audio_file)
    assert res.watermark_detected is True
    assert res.recovered_asset_id == rec.asset_id
    assert res.details["matching_method"] == "hamming_fallback"
    assert res.details["hamming_distance"] == 1
    assert res.details["watermark_token"] == token
    assert res.details["raw_decoded_token"] == raw_token
    assert "Hamming distance fallback" in res.message


# ==============================================================================
# Suite F: Tier 2A AudioSeal Neural Presence Gating & Robustness Tests
# ==============================================================================

def test_audioseal_near_zero_presence_with_exact_token_match_rejected(tmp_path):
    """Regression Test 3: Near-zero neural presence (presence=0.0474 < 0.50) must NOT establish
    watermark detection, even if decoded token exactly matches a registered asset."""
    if not is_ffmpeg_available():
        pytest.skip("FFmpeg/FFprobe binaries not available")

    import torch
    from unittest.mock import MagicMock
    from watermark.audio.audioseal_engine import MetaAudioSealEngine, hex_token_to_bits_tensor

    audio_file = create_test_wav(tmp_path / "test_low_exact.wav")
    rec = ProvenanceService.register_media(audio_file)
    token = ProvenanceService.ensure_watermark_token(rec.asset_id)

    engine = MetaAudioSealEngine()
    mock_det = MagicMock()
    # Logit -3.0 -> Sigmoid(-3.0) ~= 0.0474 (near zero presence)
    token_bits = hex_token_to_bits_tensor(token).float()
    mock_det.return_value = (torch.tensor([[-3.0]]), token_bits)
    engine._load_models = MagicMock(return_value=(None, mock_det))

    res = engine.detect(audio_file)
    assert res.watermark_detected is False
    assert res.recovered_asset_id is None
    assert res.provenance_record is None
    assert res.confidence == 0.0474
    assert res.details["threshold_passed"] is False
    assert res.details["rejected_reason"] == "presence_below_threshold"
    assert "below detection threshold" in res.message


def test_audioseal_near_zero_presence_with_hamming_fallback_rejected(tmp_path):
    """Regression Test 4: Near-zero neural presence (presence=0.0759 < 0.50) must NOT establish
    watermark detection when candidate token matches via Hamming fallback."""
    if not is_ffmpeg_available():
        pytest.skip("FFmpeg/FFprobe binaries not available")

    import torch
    from unittest.mock import MagicMock
    from watermark.audio.audioseal_engine import (
        MetaAudioSealEngine,
        hex_token_to_bits_tensor,
        bits_array_to_hex_token,
    )

    audio_file = create_test_wav(tmp_path / "test_low_hfall.wav")
    rec = ProvenanceService.register_media(audio_file)
    token = ProvenanceService.ensure_watermark_token(rec.asset_id)

    from provenance.repository import get_asset_by_watermark_token
    token_bits = hex_token_to_bits_tensor(token).float()
    flipped_bits = None
    for bit_idx in range(16):
        cand = token_bits.clone()
        cand[0, bit_idx] = 1.0 - cand[0, bit_idx]
        cand_hex = bits_array_to_hex_token(cand.squeeze(0).int().tolist())
        if get_asset_by_watermark_token(cand_hex, allow_hamming=False) is None:
            matched = get_asset_by_watermark_token(cand_hex, allow_hamming=True)
            if matched and matched.asset_id == rec.asset_id:
                flipped_bits = cand
                break

    assert flipped_bits is not None

    engine = MetaAudioSealEngine()
    mock_det = MagicMock()
    # Logit -2.5 -> Sigmoid(-2.5) ~= 0.0759
    mock_det.return_value = (torch.tensor([[-2.5]]), flipped_bits)
    engine._load_models = MagicMock(return_value=(None, mock_det))

    res = engine.detect(audio_file)
    assert res.watermark_detected is False
    assert res.recovered_asset_id is None
    assert res.provenance_record is None
    assert res.confidence == 0.0759
    assert res.details["threshold_passed"] is False
    assert res.details["rejected_reason"] == "presence_below_threshold"


def test_audioseal_non_finite_detector_output(tmp_path):
    """Regression Test 5: Detector returning NaN or Inf is handled gracefully without crashing,
    reporting watermark_detected=False and confidence=0.0."""
    if not is_ffmpeg_available():
        pytest.skip("FFmpeg/FFprobe binaries not available")

    import torch
    from unittest.mock import MagicMock
    from watermark.audio.audioseal_engine import MetaAudioSealEngine, hex_token_to_bits_tensor

    audio_file = create_test_wav(tmp_path / "test_nan.wav")
    engine = MetaAudioSealEngine()
    mock_det = MagicMock()
    mock_det.return_value = (torch.tensor([[float("nan")]]), hex_token_to_bits_tensor("ABCD").float())
    engine._load_models = MagicMock(return_value=(None, mock_det))

    res = engine.detect(audio_file)
    assert res.watermark_detected is False
    assert res.confidence == 0.0
    assert res.details["error"] == "non_finite_presence"


def test_audioseal_detector_failure_or_missing(tmp_path):
    """Regression Test 6: Detector execution exception does not crash and reports detection failure."""
    if not is_ffmpeg_available():
        pytest.skip("FFmpeg/FFprobe binaries not available")

    from unittest.mock import MagicMock
    from watermark.audio.audioseal_engine import MetaAudioSealEngine

    audio_file = create_test_wav(tmp_path / "test_crash.wav")
    engine = MetaAudioSealEngine()
    mock_det = MagicMock(side_effect=RuntimeError("AudioSeal neural forward pass crashed"))
    engine._load_models = MagicMock(return_value=(None, mock_det))

    res = engine.detect(audio_file)
    assert res.watermark_detected is False
    assert res.confidence == 0.0
    assert "failed" in res.message.lower()


def test_audioseal_strong_presence_unregistered_token(tmp_path):
    """Regression Test 7: Strong presence (presence=0.90 >= 0.50) with an unmapped token
    returns watermark_detected=False and recovered_asset_id=None."""
    if not is_ffmpeg_available():
        pytest.skip("FFmpeg/FFprobe binaries not available")

    import torch
    from unittest.mock import MagicMock, patch
    from watermark.audio.audioseal_engine import MetaAudioSealEngine, hex_token_to_bits_tensor

    audio_file = create_test_wav(tmp_path / "test_unreg.wav")
    engine = MetaAudioSealEngine()
    unregistered_token = "EEEE"

    mock_det = MagicMock()
    # Logit 2.2 -> Sigmoid(2.2) ~= 0.9002
    mock_det.return_value = (torch.tensor([[2.2]]), hex_token_to_bits_tensor(unregistered_token).float())
    engine._load_models = MagicMock(return_value=(None, mock_det))

    with patch("watermark.audio.audioseal_engine.get_asset_by_watermark_token", return_value=None), \
         patch("provenance.service.ProvenanceService.find_asset_by_watermark_token", return_value=None):
        res = engine.detect(audio_file)

    assert res.watermark_detected is False
    assert res.confidence == 0.9002
    assert res.recovered_asset_id is None
    assert res.details["threshold_passed"] is True
    assert "no registered" in res.message.lower()


def test_audioseal_rejected_detection_prevents_verified_derivative_link(tmp_path):
    """Regression Test 8: Sub-threshold AudioSeal detection on a file with registered token match
    must NOT establish a verified derivative link downstream in VerificationService."""
    if not is_ffmpeg_available():
        pytest.skip("FFmpeg/FFprobe binaries not available")

    import torch
    from unittest.mock import patch, MagicMock
    from provenance.verification import VerificationService, STATE_VERIFIED_DERIVATIVE, STATE_UNKNOWN_PROVENANCE
    from watermark.audio.audioseal_engine import hex_token_to_bits_tensor

    audio_file = create_test_wav(tmp_path / "test_downstream.wav")
    rec = ProvenanceService.register_media(audio_file)
    token = ProvenanceService.ensure_watermark_token(rec.asset_id)

    # Create a derivative audio file whose hash differs from original
    diff_file = create_test_wav(tmp_path / "test_downstream_diff.wav")

    mock_det = MagicMock()
    # Near-zero presence (logit -3.0 -> ~0.0474) with exact token match
    mock_det.return_value = (torch.tensor([[-3.0]]), hex_token_to_bits_tensor(token).float())

    with patch("watermark.audio.audioseal_engine.MetaAudioSealEngine._load_models", return_value=(None, mock_det)):
        ver_res = VerificationService.verify_media(diff_file, audio_engine="audioseal")
        # Must NOT classify as VERIFIED DERIVATIVE because presence check failed!
        assert ver_res.classification != STATE_VERIFIED_DERIVATIVE
        assert ver_res.classification == STATE_UNKNOWN_PROVENANCE
        assert ver_res.watermark_evidence.status == "NOT_DETECTED"


def test_audioseal_threshold_validation_constructor():
    """Verify MetaAudioSealEngine constructor rejects invalid confidence thresholds."""
    from watermark.audio.audioseal_engine import MetaAudioSealEngine

    # Out of range [-1.0, 1.5]
    with pytest.raises(ValueError, match="between 0.0 and 1.0 inclusive"):
        MetaAudioSealEngine(confidence_threshold=-1.0)
    with pytest.raises(ValueError, match="between 0.0 and 1.0 inclusive"):
        MetaAudioSealEngine(confidence_threshold=1.5)

    # Non-finite (NaN, +inf, -inf)
    with pytest.raises(ValueError, match="finite number"):
        MetaAudioSealEngine(confidence_threshold=float("nan"))
    with pytest.raises(ValueError, match="finite number"):
        MetaAudioSealEngine(confidence_threshold=float("inf"))
    with pytest.raises(ValueError, match="finite number"):
        MetaAudioSealEngine(confidence_threshold=float("-inf"))

    # Booleans and strings rejected
    with pytest.raises(ValueError, match="numeric float or int"):
        MetaAudioSealEngine(confidence_threshold=True)
    with pytest.raises(ValueError, match="numeric float or int"):
        MetaAudioSealEngine(confidence_threshold="0.5")

    # Valid thresholds accepted
    e1 = MetaAudioSealEngine(confidence_threshold=0.0)
    assert e1.confidence_threshold == 0.0
    e2 = MetaAudioSealEngine(confidence_threshold=1.0)
    assert e2.confidence_threshold == 1.0
    e3 = MetaAudioSealEngine(confidence_threshold=0.65)
    assert e3.confidence_threshold == 0.65
    e4 = MetaAudioSealEngine()
    assert e4.confidence_threshold is None


def test_audioseal_threshold_validation_per_call(tmp_path):
    """Verify MetaAudioSealEngine.detect rejects invalid per-call confidence thresholds."""
    from watermark.audio.audioseal_engine import MetaAudioSealEngine

    dummy_audio = tmp_path / "dummy.wav"
    engine = MetaAudioSealEngine()

    # Out of range [-1.0, 1.5]
    with pytest.raises(ValueError, match="between 0.0 and 1.0 inclusive"):
        engine.detect(dummy_audio, confidence_threshold=-1.0)
    with pytest.raises(ValueError, match="between 0.0 and 1.0 inclusive"):
        engine.detect(dummy_audio, confidence_threshold=1.5)

    # Non-finite (NaN, +inf, -inf)
    with pytest.raises(ValueError, match="finite number"):
        engine.detect(dummy_audio, confidence_threshold=float("nan"))
    with pytest.raises(ValueError, match="finite number"):
        engine.detect(dummy_audio, confidence_threshold=float("inf"))
    with pytest.raises(ValueError, match="finite number"):
        engine.detect(dummy_audio, confidence_threshold=float("-inf"))

    # Booleans and strings rejected
    with pytest.raises(ValueError, match="numeric float or int"):
        engine.detect(dummy_audio, confidence_threshold=True)
    with pytest.raises(ValueError, match="numeric float or int"):
        engine.detect(dummy_audio, confidence_threshold="0.5")


def test_audioseal_per_call_threshold_overrides_constructor(tmp_path):
    """Verify that a valid per-call threshold overrides the constructor-configured threshold."""
    if not is_ffmpeg_available():
        pytest.skip("FFmpeg/FFprobe binaries not available")

    import torch
    from unittest.mock import MagicMock
    from watermark.audio.audioseal_engine import MetaAudioSealEngine, hex_token_to_bits_tensor

    audio_file = create_test_wav(tmp_path / "test_override.wav")
    rec = ProvenanceService.register_media(audio_file)
    token = ProvenanceService.ensure_watermark_token(rec.asset_id)

    # Constructor threshold is set high (0.85)
    engine = MetaAudioSealEngine(confidence_threshold=0.85)

    # Mock detector returning presence = 0.70 (logit ~0.8473)
    mock_det = MagicMock()
    token_bits = hex_token_to_bits_tensor(token).float()
    mock_det.return_value = (torch.tensor([[0.8473]]), token_bits)
    engine._load_models = MagicMock(return_value=(None, mock_det))

    # 1. Without per-call override, effective threshold is 0.85 -> presence 0.70 fails!
    res_default = engine.detect(audio_file)
    assert res_default.watermark_detected is False
    assert res_default.details["detection_threshold"] == 0.85
    assert res_default.details["threshold_passed"] is False

    # 2. With per-call override = 0.60, effective threshold becomes 0.60 -> presence 0.70 passes!
    res_override = engine.detect(audio_file, confidence_threshold=0.60)
    assert res_override.watermark_detected is True
    assert res_override.details["detection_threshold"] == 0.60
    assert res_override.details["threshold_passed"] is True
    assert res_override.recovered_asset_id == rec.asset_id


# ==============================================================================
# Suite F: Custom Audio Watermark Correlation Search Optimization Tests (Tier 2B)
# ==============================================================================

def test_custom_audio_correlation_reference_agreement():
    """Verify exact numerical agreement between optimized correlation search and an independent reference."""
    import numpy as np
    from watermark.audio.custom_engine import compute_watermark_correlation

    def independent_reference_search(samples, tokens):
        best_tok = None
        best_sc = 0.0
        for t in tokens:
            try:
                t_int = int(t, 16)
            except ValueError:
                continue
            rng = np.random.RandomState(t_int)
            seq = rng.choice([-1.0, 1.0], size=len(samples))
            sc = float(np.dot(samples, seq) / len(samples))
            if sc > best_sc:
                best_sc = sc
                best_tok = t
        return best_tok, best_sc

    tokens = ["A1B2", "C3D4", "E5F6", "0042", "FFFF"]
    np.random.seed(1234)

    for n_samples in [500, 16000, 32000]:
        samples = np.random.uniform(-500.0, 500.0, size=n_samples).astype(np.float32)

        # 1. Unwatermarked
        ref_tok, ref_sc = independent_reference_search(samples, tokens)
        opt_tok, opt_sc = compute_watermark_correlation(samples, tokens)
        assert ref_tok == opt_tok
        assert abs(ref_sc - opt_sc) < 1e-4

        # 2. Watermarked with candidate 'C3D4'
        rng_wm = np.random.RandomState(int("C3D4", 16))
        wm_samples = samples + rng_wm.choice([-1.0, 1.0], size=n_samples) * 50.0
        ref_tok_wm, ref_sc_wm = independent_reference_search(wm_samples, tokens)
        opt_tok_wm, opt_sc_wm = compute_watermark_correlation(wm_samples, tokens)
        assert opt_tok_wm == "C3D4"
        assert ref_tok_wm == opt_tok_wm
        assert abs(ref_sc_wm - opt_sc_wm) < 1e-4


def test_custom_audio_correlation_known_positive_and_negative():
    """Verify known watermark detection at aligned position and negative rejection on clean audio."""
    import numpy as np
    from watermark.audio.custom_engine import compute_watermark_correlation

    token = "BEEF"
    tokens = ["1234", "5678", token, "ABCD"]
    n_samples = 16000

    np.random.seed(99)
    clean_audio = np.random.normal(0, 100.0, size=n_samples).astype(np.float32)

    # Clean audio -> no match above threshold
    best_tok, best_sc = compute_watermark_correlation(clean_audio, tokens)
    assert best_sc < 25.0

    # Embedded audio -> matches target token with score ~50.0
    rng_wm = np.random.RandomState(int(token, 16))
    wm_samples = clean_audio + rng_wm.choice([-1.0, 1.0], size=n_samples) * 50.0
    best_tok_wm, best_sc_wm = compute_watermark_correlation(wm_samples, tokens)
    assert best_tok_wm == token
    assert best_sc_wm > 40.0


def test_custom_audio_correlation_empty_and_short_signals():
    """Verify empty and shorter-than-minimum signals safely return (None, 0.0)."""
    import numpy as np
    from watermark.audio.custom_engine import compute_watermark_correlation

    assert compute_watermark_correlation(np.array([], dtype=np.float32), ["1234"]) == (None, 0.0)
    assert compute_watermark_correlation(np.zeros(50, dtype=np.float32), ["1234"]) == (None, 0.0)
    assert compute_watermark_correlation(np.zeros(99, dtype=np.float32), ["1234"]) == (None, 0.0)


def test_custom_audio_correlation_non_finite_inputs():
    """Verify NaN and Inf inputs fail safely without throwing exceptions or generating false matches."""
    import numpy as np
    from watermark.audio.custom_engine import compute_watermark_correlation

    nan_arr = np.array([float("nan")] * 200, dtype=np.float32)
    assert compute_watermark_correlation(nan_arr, ["1234"]) == (None, 0.0)

    inf_arr = np.array([float("inf")] * 200, dtype=np.float32)
    assert compute_watermark_correlation(inf_arr, ["1234"]) == (None, 0.0)

    mixed_arr = np.ones(200, dtype=np.float32)
    mixed_arr[50] = float("nan")
    assert compute_watermark_correlation(mixed_arr, ["1234"]) == (None, 0.0)


def test_custom_audio_correlation_polarity_and_offsets():
    """Verify inverted polarity and lag offset shifts do not trigger false detection."""
    import numpy as np
    from watermark.audio.custom_engine import compute_watermark_correlation

    token = "CAFE"
    tokens = [token]
    n_samples = 16000
    clean_audio = np.zeros(n_samples, dtype=np.float32)

    rng_wm = np.random.RandomState(int(token, 16))
    spread = rng_wm.choice([-1.0, 1.0], size=n_samples) * 50.0

    # 1. Inverted polarity: spread multiplied by -1.0
    inverted_audio = clean_audio - spread
    best_tok_inv, best_sc_inv = compute_watermark_correlation(inverted_audio, tokens)
    # Expected correlation is -50.0, which must NOT trigger positive detection
    assert best_tok_inv is None
    assert best_sc_inv == 0.0

    # 2. Offset shift: shifted by 100 samples
    shifted_audio = np.roll(clean_audio + spread, shift=100)
    best_tok_shift, best_sc_shift = compute_watermark_correlation(shifted_audio, tokens)
    # Misaligned random sequence yields near-zero correlation
    assert best_sc_shift < 10.0


def test_custom_audio_correlation_threshold_boundary():
    """Verify behavior right around the 25.0 correlation detection threshold."""
    import numpy as np
    from watermark.audio.custom_engine import compute_watermark_correlation

    token = "FEED"
    tokens = [token]
    n_samples = 16000

    rng_wm = np.random.RandomState(int(token, 16))
    ref_seq = rng_wm.choice([-1.0, 1.0], size=n_samples)

    # Sub-threshold: amplitude 24.0 (score ~24.0 < 25.0 threshold)
    sub_audio = (ref_seq * 24.0).astype(np.float32)
    best_tok_sub, best_sc_sub = compute_watermark_correlation(sub_audio, tokens)
    assert best_tok_sub == token
    assert 23.5 < best_sc_sub < 24.5
    assert best_sc_sub < 25.0  # Does not cross detection threshold

    # Super-threshold: amplitude 26.0 (score ~26.0 > 25.0 threshold)
    super_audio = (ref_seq * 26.0).astype(np.float32)
    best_tok_sup, best_sc_sup = compute_watermark_correlation(super_audio, tokens)
    assert best_tok_sup == token
    assert 25.5 < best_sc_sup < 26.5
    assert best_sc_sup > 25.0  # Crosses detection threshold


def test_custom_audio_correlation_exact_float64_decision_parity():
    """Verify strict mathematical decision and score parity with original float64 baseline."""
    import numpy as np
    from watermark.audio.custom_engine import compute_watermark_correlation

    def original_float64_baseline(samples, tokens):
        best_token = None
        best_score = 0.0
        for tok_str in tokens:
            try:
                tok_int = int(tok_str, 16)
            except ValueError:
                continue
            rng = np.random.RandomState(seed=tok_int)
            ref_seq = rng.choice([-1.0, 1.0], size=len(samples))
            score = float(np.dot(samples, ref_seq) / len(samples))
            if score > best_score:
                best_score = score
                best_token = tok_str
        return best_token, best_score

    # 1. Test exactly at 25.0, slightly below (24.9999), and slightly above (25.0001)
    target_token = "A1B2"
    tokens = [target_token, "C3D4", "E5F6"]

    for n in [1000, 16000, 48000]:
        rng = np.random.RandomState(int(target_token, 16))
        ref = rng.choice([-1.0, 1.0], size=n)

        for amp in [24.9999, 25.0, 25.0001]:
            samp = (ref * amp).astype(np.float32)
            tok_base, sc_base = original_float64_baseline(samp, tokens)
            tok_corr, sc_corr = compute_watermark_correlation(samp, tokens)

            dec_base = (tok_base is not None) and (sc_base > 25.0)
            dec_corr = (tok_corr is not None) and (sc_corr > 25.0)

            assert abs(sc_base - sc_corr) < 1e-12, f"Score mismatch at n={n}, amp={amp}: base={sc_base}, corr={sc_corr}"
            assert tok_base == tok_corr, f"Token mismatch at n={n}, amp={amp}"
            assert dec_base == dec_corr, f"Decision mismatch at n={n}, amp={amp}: base={dec_base}, corr={dec_corr}"

            # At exactly 25.0, both MUST be rejected under strict > 25.0
            if amp == 25.0:
                assert dec_base is False
                assert dec_corr is False

    # 2. Competing registered tokens with close scores
    rng_comp = np.random.RandomState(42)
    noise = rng_comp.normal(0, 10.0, size=16000).astype(np.float32)
    comp_tokens = ["1111", "2222", "3333", "4444", "5555"]
    # Embed token "3333" at 30.0 and token "2222" at 20.0
    r_3 = np.random.RandomState(int("3333", 16)).choice([-1.0, 1.0], size=16000)
    r_2 = np.random.RandomState(int("2222", 16)).choice([-1.0, 1.0], size=16000)
    comp_samples = noise + (r_3 * 30.0).astype(np.float32) + (r_2 * 20.0).astype(np.float32)

    tok_b, sc_b = original_float64_baseline(comp_samples, comp_tokens)
    tok_c, sc_c = compute_watermark_correlation(comp_samples, comp_tokens)
    assert tok_b == "3333"
    assert tok_c == "3333"
    assert abs(sc_b - sc_c) < 1e-12
    assert (sc_b > 25.0) == (sc_c > 25.0) == True



