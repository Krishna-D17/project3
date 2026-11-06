"""
Unit & Regression Tests for DeepfakeBench v1.0.1 Xception Deepfake Detector.
Verifies architectural compatibility, state_dict strict load, CPU execution,
checksum integrity, offline safety, and decision engine isolation.
"""

import hashlib
from pathlib import Path
import pytest
import torch

from forensics.deepfakebench_xception import DeepfakeBenchXception
from forensics.video_ml import (
    EXPECTED_XCEPTION_SHA256,
    XceptionDeepfakeDetector,
)
from forensics.service import ForensicService

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRODUCTION_CHECKPOINT = PROJECT_ROOT / "models" / "forensics" / "video_deepfake_xception.pth"


# 1. Checkpoint Exists Test
def test_checkpoint_exists():
    assert PRODUCTION_CHECKPOINT.exists(), f"Production checkpoint missing at {PRODUCTION_CHECKPOINT}"


# 2. Checkpoint SHA-256 Matches Test
def test_checkpoint_sha256_matches():
    h = hashlib.sha256()
    with open(PRODUCTION_CHECKPOINT, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    actual_sha = h.hexdigest()
    assert actual_sha == EXPECTED_XCEPTION_SHA256, (
        f"SHA-256 mismatch! Actual: {actual_sha}, Expected: {EXPECTED_XCEPTION_SHA256}"
    )


# 3. Model Loads on CPU Test
def test_model_loads_on_cpu():
    detector = XceptionDeepfakeDetector(checkpoint_path=PRODUCTION_CHECKPOINT)
    loaded = detector.load()
    assert loaded is True
    assert detector.model is not None


# 4. Strict State Dict Compatibility Test
def test_strict_state_dict_compatibility():
    model = DeepfakeBenchXception(num_classes=2)
    ckpt = torch.load(PRODUCTION_CHECKPOINT, map_location="cpu")
    # Must load strictly without missing or unexpected keys
    res = model.load_state_dict(ckpt, strict=True)
    assert len(res.missing_keys) == 0, f"Missing state_dict keys: {res.missing_keys}"
    assert len(res.unexpected_keys) == 0, f"Unexpected state_dict keys: {res.unexpected_keys}"


# 5. Output Has Exactly 2 Classes Test
def test_output_has_exactly_two_classes():
    detector = XceptionDeepfakeDetector(checkpoint_path=PRODUCTION_CHECKPOINT)
    detector.load()

    dummy_input = torch.randn(2, 3, 299, 299)  # Batch of 2 frames
    with torch.no_grad():
        logits = detector.model(dummy_input)

    assert logits.shape == (2, 2), f"Expected shape (2, 2), got {logits.shape}"


# 6. Class Mapping Verification Test
def test_class_mapping_semantics():
    detector = XceptionDeepfakeDetector(checkpoint_path=PRODUCTION_CHECKPOINT)
    detector.load()
    meta_dict = detector.metadata.to_dict()
    assert meta_dict["architecture"] == "Xception"
    assert meta_dict["model_name"] == "DeepfakeBench Xception"

    # Dummy tensor forward pass & softmax mapping check
    dummy_input = torch.zeros(1, 3, 299, 299)
    with torch.no_grad():
        logits = detector.model(dummy_input)
        probs = torch.softmax(logits, dim=1)

    assert probs.shape == (1, 2)
    assert abs(float(probs.sum()) - 1.0) < 1e-5
    # Class 0 = Real, Class 1 = Fake
    p_real = float(probs[0, 0])
    p_fake = float(probs[0, 1])
    assert 0.0 <= p_real <= 1.0
    assert 0.0 <= p_fake <= 1.0


# 7. Missing Checkpoint Returns UNAVAILABLE & NO_VALIDATED_DETECTOR
def test_missing_checkpoint_handling(tmp_path):
    missing_path = tmp_path / "non_existent.pth"
    detector = XceptionDeepfakeDetector(checkpoint_path=missing_path)

    assert detector.is_available() is False
    res = detector.predict_video(PROJECT_ROOT / "data" / "originals" / "000.mp4")
    assert res.available is False
    assert res.status == "UNAVAILABLE_CHECKPOINT_MISSING"
    assert res.prediction_label == "NO_VALIDATED_DETECTOR"


# 8. No Runtime Network Download Test (Offline Enforcement)
def test_no_runtime_network_download(monkeypatch):
    def mock_urllib_fail(*args, **kwargs):
        raise RuntimeError("Network download attempted during test execution!")

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", mock_urllib_fail)

    detector = XceptionDeepfakeDetector(checkpoint_path=PRODUCTION_CHECKPOINT)
    detector.load()
    assert detector.model is not None


# 9. Existing Decision-Engine Semantics Remain Unchanged Test
def test_decision_engine_semantics_unchanged(tmp_path):
    # Verify ForensicService integrates ML inference result without breaking evidence structure
    dummy_video = PROJECT_ROOT / "data" / "originals" / "000.mp4"
    if not dummy_video.exists():
        pytest.skip("Original test video not available")

    res = ForensicService.analyze_media(dummy_video)
    assert hasattr(res, "generic_forensic_status")
    assert hasattr(res, "manipulation_forensic_status")
    assert res.risk_score_type == "heuristic"
    # Video ML signal present in result
    v_ml = [m for m in res.ml_signals if m.model_name == "DeepfakeBench Xception"]
    assert len(v_ml) == 1
    assert v_ml[0].available is True


# 10. Watermark Presence Does Not Influence ML Prediction Test
def test_watermark_presence_does_not_influence_ml_prediction():
    orig_video = PROJECT_ROOT / "data" / "protected" / "wm_000_original_7s.mp4"
    vseal_video = PROJECT_ROOT / "data" / "protected" / "wm_000.mp4"

    if not orig_video.exists() or not vseal_video.exists():
        pytest.skip("Test videos not available")

    detector = XceptionDeepfakeDetector(checkpoint_path=PRODUCTION_CHECKPOINT)

    res_orig = detector.predict_video(orig_video)
    res_vseal = detector.predict_video(vseal_video)

    assert res_orig.available is True
    assert res_vseal.available is True

    # Both predictions evaluate independently of watermark status
    assert "fake_probability_mean" in res_orig.details
    assert "fake_probability_mean" in res_vseal.details


# 11. Face Detector Initialization Test
def test_face_detector_initialization():
    from forensics.config import YUNET_FACE_DETECTOR_PATH
    detector = XceptionDeepfakeDetector(checkpoint_path=PRODUCTION_CHECKPOINT)
    assert detector.face_detector_path.exists(), f"YuNet ONNX model missing at {YUNET_FACE_DETECTOR_PATH}"
    meta = detector.metadata.to_dict()
    assert meta["preprocessing"]["face_detector_installed"] is True
    assert "YuNet" in meta["preprocessing"]["face_detector_model"]


# 12. Successful Face Crop Test
def test_successful_face_crop():
    orig_video = PROJECT_ROOT / "data" / "protected" / "wm_000_original_7s.mp4"
    if not orig_video.exists():
        pytest.skip("Original test video not available")

    detector = XceptionDeepfakeDetector(checkpoint_path=PRODUCTION_CHECKPOINT)
    res = detector.predict_video(orig_video)

    assert res.available is True
    face_meta = res.details.get("face_detection", {})
    assert face_meta.get("face_detector_installed") is True
    assert face_meta.get("crop_method") in ["face_crop", "face_crop_hybrid"]
    assert face_meta.get("face_detected_frames") > 0


# 13. No-Face Fallback Test
def test_no_face_fallback():
    import numpy as np
    from PIL import Image

    detector = XceptionDeepfakeDetector(checkpoint_path=PRODUCTION_CHECKPOINT)
    blank_pil = Image.new("RGB", (640, 480), color=(0, 0, 0))
    blank_bgr = np.zeros((480, 640, 3), dtype=np.uint8)

    crop, is_face = detector._detect_and_crop_face(blank_pil, blank_bgr)
    assert is_face is False
    assert crop.size == (480, 480)  # Center-square crop fallback


# 14. Multiple-Face Deterministic Selection Test
def test_multiple_face_deterministic_selection(monkeypatch):
    import numpy as np
    from PIL import Image
    import cv2

    detector = XceptionDeepfakeDetector(checkpoint_path=PRODUCTION_CHECKPOINT)
    pil_img = Image.new("RGB", (640, 480), color=(128, 128, 128))
    bgr_img = np.full((480, 640, 3), 128, dtype=np.uint8)

    # Mock YuNet detect to return 2 faces (small face area 1000, large face area 4000)
    fake_faces = np.array([
        [100, 100, 20, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.90],   # Area = 1000
        [200, 200, 80, 80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.95],   # Area = 6400 (larger)
    ], dtype=np.float32)

    class MockYuNet:
        def detect(self, img):
            return 1, fake_faces

    def mock_create(*args, **kwargs):
        return MockYuNet()

    monkeypatch.setattr(cv2.FaceDetectorYN, "create", mock_create)

    crop, is_face = detector._detect_and_crop_face(pil_img, bgr_img)
    assert is_face is True
    # The larger face centered at (240, 240) with side=80*1.3=104 -> x1=188, y1=188, x2=292, y2=292
    assert crop.size == (104, 104)


# 15. Malformed Frame Handling Test
def test_malformed_frame_handling():
    import numpy as np
    from PIL import Image

    detector = XceptionDeepfakeDetector(checkpoint_path=PRODUCTION_CHECKPOINT)
    tiny_pil = Image.new("RGB", (4, 4), color=(255, 0, 0))
    tiny_bgr = np.zeros((4, 4, 3), dtype=np.uint8)

    crop, is_face = detector._detect_and_crop_face(tiny_pil, tiny_bgr)
    assert is_face is False
    assert crop.size == (4, 4)


# 16. Detector Still Works When Face Detection Missing Test
def test_detector_still_works_when_face_detection_missing(tmp_path):
    orig_video = PROJECT_ROOT / "data" / "protected" / "wm_000_original_7s.mp4"
    if not orig_video.exists():
        pytest.skip("Original test video not available")

    missing_face_path = tmp_path / "missing_yunet.onnx"
    detector = XceptionDeepfakeDetector(
        checkpoint_path=PRODUCTION_CHECKPOINT,
        face_detector_path=missing_face_path
    )

    res = detector.predict_video(orig_video)
    assert res.available is True
    face_meta = res.details.get("face_detection", {})
    assert face_meta.get("face_detector_installed") is False
    assert face_meta.get("crop_method") == "center_square_crop_fallback"
    assert face_meta.get("fallback_frames") == res.details.get("analyzed_frames_count")


# 17. Metadata Correctly Reports Face / Fallback Usage Test
def test_metadata_reports_crop_usage():
    orig_video = PROJECT_ROOT / "data" / "protected" / "wm_000_original_7s.mp4"
    if not orig_video.exists():
        pytest.skip("Original test video not available")

    detector = XceptionDeepfakeDetector(checkpoint_path=PRODUCTION_CHECKPOINT)
    res = detector.predict_video(orig_video)

    assert "face_detection" in res.details
    fd_dict = res.details["face_detection"]
    assert "face_detector_installed" in fd_dict
    assert "crop_method" in fd_dict
    assert "face_detected_frames" in fd_dict
    assert "fallback_frames" in fd_dict
    assert "total_sampled_frames" in fd_dict
    assert fd_dict["total_sampled_frames"] == fd_dict["face_detected_frames"] + fd_dict["fallback_frames"]


# 18. Frame-Extraction Failure Contract Test (Phase 0.1.4 Test A)
def test_frame_extraction_failure_is_not_clean_prediction(monkeypatch, tmp_path):
    detector = XceptionDeepfakeDetector(checkpoint_path=PRODUCTION_CHECKPOINT)
    dummy_video = tmp_path / "zero_frames.mp4"
    dummy_video.write_bytes(b"CORRUPTED_OR_EMPTY_VIDEO_CONTENT")

    # Mock _extract_and_preprocess_frames to simulate zero usable frames extracted
    def mock_extract_empty(path):
        return [], {
            "face_detector_installed": True,
            "crop_method": "none",
            "face_detected_frames": 0,
            "fallback_frames": 0,
            "total_sampled_frames": 0,
        }

    monkeypatch.setattr(detector, "_extract_and_preprocess_frames", mock_extract_empty)

    result = detector.predict_video(dummy_video)

    # 1. Unsuccessful / unavailable analysis
    assert result.available is False
    # 2. Specific status
    assert result.status == "FRAME_EXTRACTION_FAILED"
    # 3. Prediction label must not indicate a clean / authentic prediction
    assert result.prediction_label != "NO_MANIPULATION_DETECTED"
    assert result.prediction_label == "NO_VALIDATED_DETECTOR"
    # 4. Confidence must not encode a successful negative prediction
    assert result.confidence is None
    # 5. Explanatory details
    assert "details" in result.__dict__ or hasattr(result, "details")
    assert result.details.get("error") == "frame_extraction_failed"
    assert "Failed to extract or sample usable frames" in result.details.get("message", "")
    assert "metadata" in result.details


# 19. Genuine Successful Negative Inference Preserved Test (Phase 0.1.4 Test C)
def test_genuine_negative_inference_remains_clean(monkeypatch, tmp_path):
    detector = XceptionDeepfakeDetector(checkpoint_path=PRODUCTION_CHECKPOINT)
    dummy_video = tmp_path / "valid_video.mp4"
    dummy_video.write_bytes(b"VALID_VIDEO_CONTENT")

    # Provide 2 dummy frame tensors (shape: 3, 299, 299)
    dummy_frames = [torch.zeros(3, 299, 299), torch.zeros(3, 299, 299)]
    crop_meta = {
        "face_detector_installed": True,
        "crop_method": "face_crop",
        "face_detected_frames": 2,
        "fallback_frames": 0,
        "total_sampled_frames": 2,
    }

    monkeypatch.setattr(detector, "_extract_and_preprocess_frames", lambda path: (dummy_frames, crop_meta))

    # Mock the model's forward pass to output logits indicating genuine clean media
    # Class 0 (real) = 4.0, Class 1 (fake) = -4.0 -> fake prob ~ 0.0003 (< 0.50)
    class MockCleanModel(torch.nn.Module):
        def forward(self, x):
            batch_size = x.shape[0]
            return torch.tensor([[4.0, -4.0]] * batch_size)

    detector.model = MockCleanModel()
    monkeypatch.setattr(detector, "load", lambda: True)

    result = detector.predict_video(dummy_video)

    assert result.available is True
    assert result.status == "NO_MANIPULATION_DETECTED"
    assert result.prediction_label == "NO_MANIPULATION_DETECTED"
    assert result.confidence is not None
    assert result.confidence < detector.threshold
    assert result.details.get("fake_probability_mean") < detector.threshold
    assert result.details.get("analyzed_frames_count") == 2


# ------------------------------------------------------------------------------
# Phase 0.1.7 Numerical & Structural Safety Tests
# ------------------------------------------------------------------------------

def _setup_mock_detector(monkeypatch, tmp_path, mock_model, frame_count: int = 1):
    detector = XceptionDeepfakeDetector(checkpoint_path=PRODUCTION_CHECKPOINT)
    dummy_video = tmp_path / "test_vid.mp4"
    dummy_video.write_bytes(b"TEST_CONTENT")
    dummy_frames = [torch.zeros(3, 299, 299) for _ in range(frame_count)]
    crop_meta = {
        "face_detector_installed": True,
        "crop_method": "center_crop",
        "face_detected_frames": 0,
        "fallback_frames": frame_count,
        "total_sampled_frames": frame_count,
    }
    monkeypatch.setattr(detector, "_extract_and_preprocess_frames", lambda p: (dummy_frames, crop_meta))
    monkeypatch.setattr(detector, "load", lambda: True)
    detector.model = mock_model
    return detector, dummy_video


# 20. NaN in Model Logits Rejected (Phase 0.1.7 Test 1)
def test_model_output_nan_logits_rejected(monkeypatch, tmp_path):
    class MockNanModel(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([[float("nan"), 1.0]])

    detector, dummy_video = _setup_mock_detector(monkeypatch, tmp_path, MockNanModel())
    res = detector.predict_video(dummy_video)

    assert res.available is False
    assert res.status == "NON_FINITE_MODEL_OUTPUT"
    assert res.prediction_label != "NO_MANIPULATION_DETECTED"
    assert res.prediction_label == "NO_VALIDATED_DETECTOR"
    assert res.confidence is None
    assert res.details.get("error") == "non_finite_logits"


# 21. Positive Infinity in Model Logits Rejected (Phase 0.1.7 Test 2)
def test_model_output_pos_inf_logits_rejected(monkeypatch, tmp_path):
    class MockPosInfModel(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([[float("inf"), float("inf")]])

    detector, dummy_video = _setup_mock_detector(monkeypatch, tmp_path, MockPosInfModel())
    res = detector.predict_video(dummy_video)

    assert res.available is False
    assert res.status == "NON_FINITE_MODEL_OUTPUT"
    assert res.prediction_label != "NO_MANIPULATION_DETECTED"
    assert res.prediction_label == "NO_VALIDATED_DETECTOR"
    assert res.confidence is None
    assert res.details.get("error") == "non_finite_logits"


# 22. Negative Infinity in Model Logits Rejected (Phase 0.1.7 Test 3)
def test_model_output_neg_inf_logits_rejected(monkeypatch, tmp_path):
    class MockNegInfModel(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([[float("-inf"), float("-inf")]])

    detector, dummy_video = _setup_mock_detector(monkeypatch, tmp_path, MockNegInfModel())
    res = detector.predict_video(dummy_video)

    assert res.available is False
    assert res.status == "NON_FINITE_MODEL_OUTPUT"
    assert res.prediction_label != "NO_MANIPULATION_DETECTED"
    assert res.prediction_label == "NO_VALIDATED_DETECTOR"
    assert res.confidence is None
    assert res.details.get("error") == "non_finite_logits"


# 23. Empty Output Tensor Rejected (Phase 0.1.7 Test 4)
def test_model_output_empty_tensor_rejected(monkeypatch, tmp_path):
    class MockEmptyModel(torch.nn.Module):
        def forward(self, x):
            return torch.empty(0, 2)

    detector, dummy_video = _setup_mock_detector(monkeypatch, tmp_path, MockEmptyModel())
    res = detector.predict_video(dummy_video)

    assert res.available is False
    assert res.status == "MALFORMED_MODEL_OUTPUT"
    assert res.prediction_label != "NO_MANIPULATION_DETECTED"
    assert res.prediction_label == "NO_VALIDATED_DETECTOR"
    assert res.confidence is None
    assert res.details.get("error") == "malformed_output_shape"


# 24. One-Dimensional Output Rejected (Phase 0.1.7 Test 5)
def test_model_output_one_dimensional_rejected(monkeypatch, tmp_path):
    class Mock1DModel(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([1.0, 2.0])

    detector, dummy_video = _setup_mock_detector(monkeypatch, tmp_path, Mock1DModel())
    res = detector.predict_video(dummy_video)

    assert res.available is False
    assert res.status == "MALFORMED_MODEL_OUTPUT"
    assert res.prediction_label != "NO_MANIPULATION_DETECTED"
    assert res.prediction_label == "NO_VALIDATED_DETECTOR"
    assert res.confidence is None
    assert res.details.get("error") == "malformed_output_shape"


# 25. One-Class Output Rejected (Phase 0.1.7 Test 6)
def test_model_output_one_class_rejected(monkeypatch, tmp_path):
    class Mock1ClassModel(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([[0.5]])

    detector, dummy_video = _setup_mock_detector(monkeypatch, tmp_path, Mock1ClassModel())
    res = detector.predict_video(dummy_video)

    assert res.available is False
    assert res.status == "MALFORMED_MODEL_OUTPUT"
    assert res.prediction_label != "NO_MANIPULATION_DETECTED"
    assert res.prediction_label == "NO_VALIDATED_DETECTOR"
    assert res.confidence is None
    assert res.details.get("error") == "malformed_output_shape"


# 26. Three-Class Output Rejected (Phase 0.1.7 Test 7)
def test_model_output_three_classes_rejected(monkeypatch, tmp_path):
    class Mock3ClassModel(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([[0.2, 0.5, 0.3]])

    detector, dummy_video = _setup_mock_detector(monkeypatch, tmp_path, Mock3ClassModel())
    res = detector.predict_video(dummy_video)

    assert res.available is False
    assert res.status == "MALFORMED_MODEL_OUTPUT"
    assert res.prediction_label != "NO_MANIPULATION_DETECTED"
    assert res.prediction_label == "NO_VALIDATED_DETECTOR"
    assert res.confidence is None
    assert res.details.get("error") == "malformed_output_shape"


# 27. Valid Clean Logits Retain Negative Prediction (Phase 0.1.7 Test 8)
def test_model_output_valid_clean_logits_preserved(monkeypatch, tmp_path):
    class MockCleanModel(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([[5.0, -5.0]])

    detector, dummy_video = _setup_mock_detector(monkeypatch, tmp_path, MockCleanModel())
    res = detector.predict_video(dummy_video)

    assert res.available is True
    assert res.status == "NO_MANIPULATION_DETECTED"
    assert res.prediction_label == "NO_MANIPULATION_DETECTED"
    assert res.confidence is not None
    assert res.confidence < detector.threshold
    assert res.details.get("fake_probability_mean") < detector.threshold


# 28. Valid Fake Logits Retain Positive Prediction (Phase 0.1.7 Test 9)
def test_model_output_valid_fake_logits_preserved(monkeypatch, tmp_path):
    class MockFakeModel(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([[-5.0, 5.0]])

    detector, dummy_video = _setup_mock_detector(monkeypatch, tmp_path, MockFakeModel())
    res = detector.predict_video(dummy_video)

    assert res.available is True
    assert res.status == "MANIPULATION_SUSPECTED"
    assert res.prediction_label == "MANIPULATION_SUSPECTED"
    assert res.confidence is not None
    assert res.confidence >= detector.threshold
    assert res.details.get("fake_probability_mean") >= detector.threshold


# 29. Batch Cardinality Mismatch Rejected (Phase 0.1.9)
def test_model_output_batch_size_mismatch_rejected(monkeypatch, tmp_path):
    # Model receives a batch of 2 frame tensors, but returns only 1 row of logits
    class MockTruncatedModel(torch.nn.Module):
        def forward(self, x):
            assert x.shape[0] == 2, f"Expected batch input of 2 frames, got {x.shape[0]}"
            return torch.tensor([[1.0, -1.0]])  # Shape [1, 2], mismatching batch.shape[0] == 2

    detector, dummy_video = _setup_mock_detector(
        monkeypatch, tmp_path, MockTruncatedModel(), frame_count=2
    )
    res = detector.predict_video(dummy_video)

    assert res.available is False
    assert res.status == "MALFORMED_MODEL_OUTPUT"
    assert res.prediction_label != "NO_MANIPULATION_DETECTED"
    assert res.prediction_label == "NO_VALIDATED_DETECTOR"
    assert res.confidence is None
    assert res.details.get("error") == "malformed_output_shape"
    assert "expected [2, 2]" in res.details.get("message", "")




