"""Face detection module using MediaPipe."""

import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from emotion_detection_action.core.config import ModelConfig
from emotion_detection_action.core.types import BoundingBox, FaceDetection
from emotion_detection_action.models.base import BaseModel

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None
    python = None
    vision = None

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
MODEL_NAME = "blaze_face_short_range.tflite"


class FaceDetector(BaseModel[np.ndarray, list[FaceDetection]]):
    """Face detection using MediaPipe.

    Detects faces in images and returns bounding boxes, confidence scores,
    and optionally facial landmarks and cropped face images.

    The model is automatically downloaded on first use and cached locally.

    Example:
        >>> config = ModelConfig(model_id="mediapipe", device="cpu")
        >>> detector = FaceDetector(config)
        >>> detector.load()
        >>> faces = detector.predict(image)
        >>> for face in faces:
        ...     print(f"Face at {face.bbox} with confidence {face.confidence}")
    """

    def __init__(
        self,
        config: ModelConfig,
        threshold: float = 0.5,
        min_face_size: int = 20,
        max_faces: int = 5,
        return_landmarks: bool = True,
        return_face_images: bool = True,
    ) -> None:
        """Initialize face detector.

        Args:
            config: Model configuration.
            threshold: Detection confidence threshold (0-1).
            min_face_size: Minimum face size in pixels.
            max_faces: Maximum number of faces to detect.
            return_landmarks: Whether to return facial landmarks.
            return_face_images: Whether to return cropped face images.
        """
        super().__init__(config)
        self.threshold = threshold
        self.min_face_size = min_face_size
        self.max_faces = max_faces
        self.return_landmarks = return_landmarks
        self.return_face_images = return_face_images

        self._detector: Any = None

    def load(self) -> None:
        """Load the MediaPipe face detection model."""
        if self._is_loaded:
            return

        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError(
                "MediaPipe is not available. Install with: pip install mediapipe"
            )

        model_path = self._get_or_download_model()

        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=self.threshold,
        )

        self._detector = vision.FaceDetector.create_from_options(options)
        self._is_loaded = True

    def _get_or_download_model(self) -> Path:
        """Get the face detection model file, downloading if necessary."""
        cache_locations = [
            Path.home() / ".cache" / "emotion_detection_action" / "models",
            Path(tempfile.gettempdir()) / "emotion_detection_action" / "models",
        ]

        cache_dir = None
        model_path = None

        for loc in cache_locations:
            try:
                loc.mkdir(parents=True, exist_ok=True)
                model_path = loc / MODEL_NAME
                cache_dir = loc
                break
            except (PermissionError, OSError):
                continue

        if cache_dir is None:
            raise RuntimeError(
                "Unable to create cache directory for MediaPipe models. "
                "Please ensure write access to ~/.cache or system temp directory."
            )

        if not model_path.exists():
            try:
                urllib.request.urlretrieve(MODEL_URL, model_path)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download MediaPipe face detection model: {e}\n"
                    f"You can manually download from {MODEL_URL} and place at {model_path}"
                ) from e

        return model_path

    def unload(self) -> None:
        """Unload the model."""
        if self._detector:
            self._detector.close()
        self._detector = None
        self._is_loaded = False

    @property
    def model_type(self) -> str:
        """Get the model type identifier."""
        return "mediapipe-face-detector"

    def predict(self, input_data: np.ndarray) -> list[FaceDetection]:
        """Detect faces in an image.

        Args:
            input_data: Input image as numpy array (H, W, C) in RGB format.

        Returns:
            List of FaceDetection objects.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        if self._detector is None:
            return []

        if len(input_data.shape) == 2:
            input_data = np.stack([input_data] * 3, axis=-1)

        h, w = input_data.shape[:2]

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=input_data)
        result = self._detector.detect(mp_image)

        if not result.detections:
            return []

        detections = []

        sorted_detections = sorted(
            result.detections,
            key=lambda d: d.categories[0].score if d.categories else 0,
            reverse=True,
        )

        for detection in sorted_detections:
            if len(detections) >= self.max_faces:
                break

            score = detection.categories[0].score if detection.categories else 0
            if score < self.threshold:
                continue

            bbox_data = detection.bounding_box
            x1 = int(bbox_data.origin_x)
            y1 = int(bbox_data.origin_y)
            box_w = int(bbox_data.width)
            box_h = int(bbox_data.height)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x1 + box_w)
            y2 = min(h, y1 + box_h)
            box_w = x2 - x1
            box_h = y2 - y1

            if box_w < self.min_face_size or box_h < self.min_face_size:
                continue

            bbox = BoundingBox(
                x=x1,
                y=y1,
                width=box_w,
                height=box_h,
            )

            face_landmarks = None
            if self.return_landmarks and detection.keypoints:
                face_landmarks = np.array([
                    [kp.x * w, kp.y * h] for kp in detection.keypoints
                ])

            face_image = None
            if self.return_face_images:
                face_image = input_data[y1:y2, x1:x2].copy()

            detections.append(
                FaceDetection(
                    bbox=bbox,
                    confidence=float(score),
                    landmarks=face_landmarks,
                    face_image=face_image,
                )
            )

        return detections

    def detect_and_align(
        self,
        image: np.ndarray,
        target_size: tuple[int, int] = (224, 224),
    ) -> list[tuple[FaceDetection, np.ndarray]]:
        """Detect faces and return aligned face crops.

        Args:
            image: Input image in RGB format.
            target_size: Target size for aligned faces.

        Returns:
            List of (detection, aligned_face) tuples.
        """
        import cv2

        detections = self.predict(image)
        results = []

        for det in detections:
            if det.face_image is not None:
                aligned = cv2.resize(det.face_image, target_size)
                results.append((det, aligned))

        return results

    @staticmethod
    def draw_detections(
        image: np.ndarray,
        detections: list[FaceDetection],
        color: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw detection boxes on an image.

        Args:
            image: Input image.
            detections: List of face detections.
            color: Box color in BGR.
            thickness: Line thickness.

        Returns:
            Image with drawn boxes.
        """
        import cv2

        result = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox.to_xyxy()
            cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

            label = f"{det.confidence:.2f}"
            cv2.putText(
                result,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                thickness,
            )

        return result

    def __repr__(self) -> str:
        return (
            f"FaceDetector(model={self.model_type}, "
            f"threshold={self.threshold}, loaded={self._is_loaded})"
        )
