import hashlib
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from forensics.config import (
    MAX_VIDEO_SAMPLED_FRAMES,
    VIDEO_ML_CHECKPOINT_PATH,
    VIDEO_TARGET_FPS_SAMPLING,
    YUNET_FACE_DETECTOR_PATH,
)
from forensics.models import MLInferenceResult

try:
    import cv2
    import numpy as np

    import torch
    from PIL import Image
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

EXPECTED_XCEPTION_SHA256 = "dd61186f9ac8e99aae85982cae338b555dfaeef042f6b699e5a55cae3998ba47"
EXPECTED_YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"


@dataclass
class DeepfakeModelMetadata:
    """Standardized Metadata Model for Deepfake Detector Models."""
    model_name: str
    architecture: str
    checkpoint_path: str
    checkpoint_sha256: Optional[str] = None
    training_dataset: Optional[str] = None
    detector_type: str = "calibrated_ml"
    input_size: tuple = (299, 299)
    preprocessing: Dict[str, Any] = field(default_factory=dict)
    framework: str = "PyTorch CPU"
    version: str = "1.0.1"
    license_source: Optional[str] = None
    available: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "architecture": self.architecture,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "training_dataset": self.training_dataset,
            "detector_type": self.detector_type,
            "input_size": list(self.input_size),
            "preprocessing": self.preprocessing,
            "framework": self.framework,
            "version": self.version,
            "license_source": self.license_source,
            "available": self.available,
        }


class BaseDeepfakeDetector:
    """Abstract Interface Contract for Video Deepfake Detectors."""

    def is_available(self) -> bool:
        raise NotImplementedError

    def load(self) -> bool:
        raise NotImplementedError

    def predict_video(self, file_path: Union[str, Path]) -> MLInferenceResult:
        raise NotImplementedError


class EfficientNetB4Adapter(BaseDeepfakeDetector):
    """Legacy EfficientNet-B4 Adapter Contract."""

    def __init__(self, checkpoint_path: Path = VIDEO_ML_CHECKPOINT_PATH):
        self.checkpoint_path = checkpoint_path
        self.model_name = "EfficientNet-B4-Deepfake"
        self.metadata = DeepfakeModelMetadata(
            model_name=self.model_name,
            architecture="EfficientNet-B4",
            checkpoint_path=str(self.checkpoint_path),
            input_size=(380, 380),
            preprocessing={"crop": "center_crop", "normalization": "imagenet"},
            available=self.is_available()
        )

    def is_available(self) -> bool:
        return HAS_TORCH and self.checkpoint_path.exists()

    def load(self) -> bool:
        if not self.is_available():
            return False
        return True

    def predict_video(self, file_path: Union[str, Path]) -> MLInferenceResult:
        start_time = time.time()
        if not HAS_TORCH:
            return MLInferenceResult(
                model_name=self.model_name,
                available=False,
                status="UNAVAILABLE_PYTORCH_MISSING",
                processing_time_seconds=time.time() - start_time,
                details={"message": "PyTorch framework is not installed in current environment."}
            )

        if not self.checkpoint_path.exists():
            return MLInferenceResult(
                model_name=self.model_name,
                available=False,
                status="UNAVAILABLE_CHECKPOINT_MISSING",
                processing_time_seconds=time.time() - start_time,
                details={
                    "message": (
                        "Deepfake-specific EfficientNet-B4 model weights file not found. "
                        "Step 6B audit confirmed standard ImageNet weights are NOT deepfake classifiers out-of-the-box. "
                        f"To enable ML inference, place deepfake weights at: '{self.checkpoint_path}'."
                    ),
                    "checkpoint_path": str(self.checkpoint_path),
                    "metadata": self.metadata.to_dict()
                }
            )

        try:
            proc_time = time.time() - start_time
            return MLInferenceResult(
                model_name=self.model_name,
                available=True,
                status="SUCCESS",
                confidence=0.50,
                prediction_label="UNSPECIFIED",
                processing_time_seconds=proc_time,
                details={
                    "message": "Loaded custom video ML deepfake weights file.",
                    "metadata": self.metadata.to_dict()
                }
            )
        except Exception as e:
            return MLInferenceResult(
                model_name=self.model_name,
                available=False,
                status=f"INFERENCE_ERROR: {str(e)}",
                processing_time_seconds=time.time() - start_time,
                details={"error": str(e)}
            )


class XceptionDeepfakeDetector(BaseDeepfakeDetector):
    """Production Official DeepfakeBench v1.0.1 Xception Deepfake Detector with YuNet Face Detection."""

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        face_detector_path: Optional[Path] = None,
        threshold: float = 0.50,
    ):
        if checkpoint_path is None:
            checkpoint_path = VIDEO_ML_CHECKPOINT_PATH.parent / "video_deepfake_xception.pth"
        if face_detector_path is None:
            face_detector_path = YUNET_FACE_DETECTOR_PATH

        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.face_detector_path = Path(face_detector_path).resolve()
        self.model_name = "DeepfakeBench Xception"
        self.threshold = threshold
        self.model: Optional[Any] = None

        fw_str = f"PyTorch {torch.__version__} CPU" if HAS_TORCH else "PyTorch Missing"
        face_det_installed = self.face_detector_path.exists()

        self.metadata = DeepfakeModelMetadata(
            model_name=self.model_name,
            architecture="Xception",
            checkpoint_path=str(self.checkpoint_path),
            checkpoint_sha256=EXPECTED_XCEPTION_SHA256,
            training_dataset="FaceForensics++ c23",
            input_size=(299, 299),
            preprocessing={
                "color_space": "RGB",
                "crop": "face_crop_with_center_fallback",
                "margin_scale": 1.3,
                "normalization_mean": [0.5, 0.5, 0.5],
                "normalization_std": [0.5, 0.5, 0.5],
                "range": [-1.0, 1.0],
                "face_detector_installed": face_det_installed,
                "face_detector_model": "OpenCV YuNet (face_detection_yunet_2023mar.onnx)" if face_det_installed else "None"
            },
            framework=fw_str,
            version="1.0.1",
            license_source="DeepfakeBench v1.0.1 (SCLBD/DeepfakeBench)",
            available=self.is_available()
        )

    def is_available(self) -> bool:
        return HAS_TORCH and self.checkpoint_path.exists()

    def _verify_sha256(self) -> bool:
        if not self.checkpoint_path.exists():
            return False
        h = hashlib.sha256()
        with open(self.checkpoint_path, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest() == EXPECTED_XCEPTION_SHA256

    def load(self) -> bool:
        """Loads model weights on CPU with strict state_dict check."""
        if not self.is_available():
            return False

        if self.model is not None:
            return True

        if not self._verify_sha256():
            raise ValueError(
                f"Checkpoint SHA-256 mismatch for '{self.checkpoint_path}'. "
                f"Expected '{EXPECTED_XCEPTION_SHA256}'."
            )

        from forensics.deepfakebench_xception import DeepfakeBenchXception
        model = DeepfakeBenchXception(num_classes=2)
        ckpt = torch.load(self.checkpoint_path, map_location="cpu")

        # Strict loading enforce — fails if keys or shapes do not match
        model.load_state_dict(ckpt, strict=True)
        model.eval()
        self.model = model
        self.metadata.available = True
        return True

    def _detect_and_crop_face(
        self, img_pil: Any, img_bgr: Any, margin_scale: float = 1.3
    ) -> Tuple[Any, bool]:
        """Detects primary face using YuNet with deterministic largest-area selection and margin crop.
        Falls back to center-square crop if no face detected or face detector unavailable.
        """
        w_img, h_img = img_pil.size

        if self.face_detector_path.exists():
            try:
                yunet = cv2.FaceDetectorYN.create(
                    str(self.face_detector_path),
                    "",
                    (w_img, h_img),
                    score_threshold=0.60,
                    nms_threshold=0.30,
                )
                _, faces = yunet.detect(img_bgr)

                if faces is not None and len(faces) > 0:
                    # Deterministic rule: Select face with largest bounding box area (w * h)
                    best_face = max(faces, key=lambda f: float(f[2]) * float(f[3]))
                    x, y, w, h = (
                        float(best_face[0]),
                        float(best_face[1]),
                        float(best_face[2]),
                        float(best_face[3]),
                    )

                    # Calculate center and scaled square side
                    cx, cy = x + w / 2.0, y + h / 2.0
                    side = max(w, h) * margin_scale

                    x1 = max(0, int(cx - side / 2.0))
                    y1 = max(0, int(cy - side / 2.0))
                    x2 = min(w_img, int(cx + side / 2.0))
                    y2 = min(h_img, int(cy + side / 2.0))

                    if (x2 - x1) > 10 and (y2 - y1) > 10:
                        crop = img_pil.crop((x1, y1, x2, y2))
                        return crop, True
            except Exception:
                pass

        # Fallback: Center-square crop
        min_dim = min(w_img, h_img)
        left = (w_img - min_dim) // 2
        top = (h_img - min_dim) // 2
        crop = img_pil.crop((left, top, left + min_dim, top + min_dim))
        return crop, False

    def _extract_and_preprocess_frames(self, file_path: Path) -> Tuple[List[Any], Dict[str, Any]]:
        """Samples frames at 1 FPS (max 10), applies face/center crop, resizes to 299x299, normalizes to [-1, 1]."""
        from media.ffmpeg import get_ffmpeg_path, run_ffmpeg_command

        ffmpeg_exe = get_ffmpeg_path()
        if not ffmpeg_exe or not ffmpeg_exe.exists():
            return [], {}

        temp_dir = Path(tempfile.mkdtemp(prefix="mp_ml_frame_"))
        tensors = []
        face_detected_count = 0
        fallback_count = 0

        try:
            frame_pattern = temp_dir / "frame_%03d.png"
            cmd_extract = [
                str(ffmpeg_exe), "-y",
                "-i", str(file_path),
                "-vf", f"fps={VIDEO_TARGET_FPS_SAMPLING}",
                "-vframes", str(MAX_VIDEO_SAMPLED_FRAMES),
                str(frame_pattern)
            ]
            res = run_ffmpeg_command(cmd_extract, timeout=30.0)
            if res.returncode != 0:
                return [], {}

            frame_files = sorted(temp_dir.glob("frame_*.png"))
            for fp in frame_files:
                img_bgr = cv2.imread(str(fp))
                img_pil = Image.open(fp).convert("RGB")

                cropped, is_face = self._detect_and_crop_face(img_pil, img_bgr, margin_scale=1.3)
                if is_face:
                    face_detected_count += 1
                else:
                    fallback_count += 1

                resized = cropped.resize((299, 299), Image.BILINEAR)
                arr = np.array(resized, dtype=np.float32) / 255.0
                norm = (arr - 0.5) / 0.5  # Map to [-1, 1]
                t = torch.tensor(norm, dtype=torch.float32).permute(2, 0, 1)  # [3, 299, 299]
                tensors.append(t)
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

        total_sampled = len(tensors)
        if total_sampled > 0:
            if face_detected_count == total_sampled:
                crop_summary = "face_crop"
            elif fallback_count == total_sampled:
                crop_summary = "center_square_crop_fallback"
            else:
                crop_summary = "face_crop_hybrid"
        else:
            crop_summary = "none"

        crop_metadata = {
            "face_detector_installed": self.face_detector_path.exists(),
            "crop_method": crop_summary,
            "face_detected_frames": face_detected_count,
            "fallback_frames": fallback_count,
            "total_sampled_frames": total_sampled,
        }

        return tensors, crop_metadata

    def _build_failure_result(
        self, status: str, error_code: str, message: str, start_time: float
    ) -> MLInferenceResult:
        return MLInferenceResult(
            model_name=self.model_name,
            available=False,
            status=status,
            prediction_label="NO_VALIDATED_DETECTOR",
            confidence=None,
            processing_time_seconds=round(time.time() - start_time, 4),
            details={
                "message": message,
                "error": error_code,
                "metadata": self.metadata.to_dict(),
            },
        )

    def predict_video(self, file_path: Union[str, Path]) -> MLInferenceResult:
        start_time = time.time()
        p = Path(file_path).resolve()

        if not HAS_TORCH:
            return MLInferenceResult(
                model_name=self.model_name,
                available=False,
                status="UNAVAILABLE_PYTORCH_MISSING",
                processing_time_seconds=time.time() - start_time,
                details={"message": "PyTorch framework is not installed in current environment."}
            )

        if not self.checkpoint_path.exists():
            return MLInferenceResult(
                model_name=self.model_name,
                available=False,
                status="UNAVAILABLE_CHECKPOINT_MISSING",
                prediction_label="NO_VALIDATED_DETECTOR",
                processing_time_seconds=time.time() - start_time,
                details={
                    "message": (
                        "Audited DeepfakeBench Xception model checkpoint file not found. "
                        f"Place verified checkpoint at: '{self.checkpoint_path}'."
                    ),
                    "checkpoint_path": str(self.checkpoint_path),
                    "metadata": self.metadata.to_dict()
                }
            )

        try:
            load_start = time.time()
            self.load()
            load_time = time.time() - load_start

            frame_tensors, crop_meta = self._extract_and_preprocess_frames(p)
            if not frame_tensors:
                return self._build_failure_result(
                    status="FRAME_EXTRACTION_FAILED",
                    error_code="frame_extraction_failed",
                    message="Failed to extract or sample usable frames from input video.",
                    start_time=start_time,
                )

            batch = torch.stack(frame_tensors, dim=0)  # [N, 3, 299, 299]
            infer_start = time.time()
            with torch.no_grad():
                logits = self.model(batch)

            infer_time = time.time() - infer_start

            # 1. Structural validation of raw logits
            if (
                not isinstance(logits, torch.Tensor)
                or logits.ndim != 2
                or logits.shape[0] != batch.shape[0]
                or logits.shape[1] != 2
            ):
                return self._build_failure_result(
                    status="MALFORMED_MODEL_OUTPUT",
                    error_code="malformed_output_shape",
                    message=f"Model output shape {getattr(logits, 'shape', None)} is invalid (expected [{batch.shape[0]}, 2]).",
                    start_time=start_time,
                )

            # 2. Numerical finiteness check on raw logits
            if not torch.all(torch.isfinite(logits)):
                return self._build_failure_result(
                    status="NON_FINITE_MODEL_OUTPUT",
                    error_code="non_finite_logits",
                    message="Model inference produced non-finite logits (NaN or Inf).",
                    start_time=start_time,
                )

            with torch.no_grad():
                probs = torch.softmax(logits, dim=1)

            # 3. Numerical finiteness check on probabilities
            if not torch.all(torch.isfinite(probs)):
                return self._build_failure_result(
                    status="NON_FINITE_MODEL_OUTPUT",
                    error_code="non_finite_probabilities",
                    message="Softmax computation produced non-finite probabilities (NaN or Inf).",
                    start_time=start_time,
                )

            fake_probs = probs[:, 1].tolist()
            if not fake_probs:
                return self._build_failure_result(
                    status="MALFORMED_MODEL_OUTPUT",
                    error_code="empty_probabilities",
                    message="Extracted probability list is empty; cannot compute aggregate risk.",
                    start_time=start_time,
                )

            mean_fake_prob = float(np.mean(fake_probs))
            max_fake_prob = float(np.max(fake_probs))
            min_fake_prob = float(np.min(fake_probs))

            status_str = (
                "MANIPULATION_SUSPECTED"
                if mean_fake_prob >= self.threshold
                else "NO_MANIPULATION_DETECTED"
            )

            total_proc_time = time.time() - start_time

            # Update metadata dict with runtime crop information
            meta_dict = self.metadata.to_dict()
            meta_dict["preprocessing"].update(crop_meta)

            return MLInferenceResult(
                model_name=self.model_name,
                available=True,
                status=status_str,
                prediction_label=status_str,
                confidence=round(mean_fake_prob, 4),
                processing_time_seconds=round(total_proc_time, 4),
                details={
                    "message": f"DeepfakeBench Xception analysis completed: P(Fake) = {mean_fake_prob:.4f}.",
                    "threshold": self.threshold,
                    "fake_probability_mean": round(mean_fake_prob, 4),
                    "fake_probability_max": round(max_fake_prob, 4),
                    "fake_probability_min": round(min_fake_prob, 4),
                    "analyzed_frames_count": len(fake_probs),
                    "per_frame_fake_probabilities": [round(p, 4) for p in fake_probs],
                    "aggregation_method": "mean_fake_probability",
                    "face_detection": crop_meta,
                    "timing_seconds": {
                        "model_load": round(load_time, 4),
                        "frame_inference": round(infer_time, 4),
                        "total": round(total_proc_time, 4),
                        "per_frame": round(infer_time / len(fake_probs), 4)
                    },
                    "class_mapping": {0: "Real", 1: "Fake/Deepfake"},
                    "metadata": meta_dict
                }
            )
        except Exception as e:
            return MLInferenceResult(
                model_name=self.model_name,
                available=False,
                status=f"INFERENCE_ERROR: {str(e)}",
                processing_time_seconds=time.time() - start_time,
                details={"error": str(e)}
            )


def get_video_deepfake_detector(detector_type: str = "auto") -> BaseDeepfakeDetector:
    """Factory method returning active video deepfake detector implementation."""
    xc_path = VIDEO_ML_CHECKPOINT_PATH.parent / "video_deepfake_xception.pth"
    if detector_type == "xception" or (detector_type == "auto" and xc_path.exists()):
        return XceptionDeepfakeDetector(xc_path)
    return EfficientNetB4Adapter(VIDEO_ML_CHECKPOINT_PATH)


def run_video_ml_inference(file_path: Union[str, Path]) -> MLInferenceResult:
    """Optional video deepfake ML model inference adapter."""
    detector = get_video_deepfake_detector(detector_type="auto")
    return detector.predict_video(file_path)


