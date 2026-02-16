"""Face detection module using MediaPipe."""

from typing import Any

import numpy as np

from emotion_detection_action.core.config import ModelConfig
from emotion_detection_action.core.types import BoundingBox, FaceDetection
from emotion_detection_action.models.base import BaseModel

# Try to import MediaPipe
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


class FaceDetector(BaseModel[np.ndarray, list[FaceDetection]]):
    """Face detection using MediaPipe.

    Detects faces in images and returns bounding boxes, confidence scores,
    and optionally facial landmarks and cropped face images.

    MediaPipe provides two model types:
    - "short": Optimized for faces within 2 meters of the camera (default)
    - "full": Optimized for faces within 5 meters of the camera

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
        model_selection: int = 0,
    ) -> None:
        """Initialize face detector.

        Args:
            config: Model configuration.
            threshold: Detection confidence threshold (0-1).
            min_face_size: Minimum face size in pixels.
            max_faces: Maximum number of faces to detect.
            return_landmarks: Whether to return facial landmarks.
            return_face_images: Whether to return cropped face images.
            model_selection: MediaPipe model selection.
                0 = short-range (within 2 meters, faster)
                1 = full-range (within 5 meters, more accurate for distant faces)
        """
        super().__init__(config)
        self.threshold = threshold
        self.min_face_size = min_face_size
        self.max_faces = max_faces
        self.return_landmarks = return_landmarks
        self.return_face_images = return_face_images
        self.model_selection = model_selection

        self._detector: Any = None

    def load(self) -> None:
        """Load the MediaPipe face detection model."""
        if self._is_loaded:
            return

        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError(
                "MediaPipe is not available. Install with: pip install mediapipe"
            )

        # Parse model selection from config if specified
        model_id = self.config.model_id.lower()
        if model_id in ("full", "mediapipe-full", "long-range"):
            self.model_selection = 1
        elif model_id in ("short", "mediapipe-short", "short-range", "mediapipe"):
            self.model_selection = 0

        self._detector = mp.solutions.face_detection.FaceDetection(
            model_selection=self.model_selection,
            min_detection_confidence=self.threshold,
        )

        self._is_loaded = True

    def unload(self) -> None:
        """Unload the model."""
        if self._detector:
            self._detector.close()
        self._detector = None
        self._is_loaded = False

    @property
    def model_type(self) -> str:
        """Get the currently loaded model type."""
        return "mediapipe-short" if self.model_selection == 0 else "mediapipe-full"

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

        # Ensure RGB format
        if len(input_data.shape) == 2:
            # Grayscale, convert to RGB
            input_data = np.stack([input_data] * 3, axis=-1)

        h, w = input_data.shape[:2]

        # Process with MediaPipe
        results = self._detector.process(input_data)

        if not results.detections:
            return []

        detections = []

        # Sort by confidence to get highest confidence faces first
        sorted_detections = sorted(
            results.detections,
            key=lambda d: d.score[0] if d.score else 0,
            reverse=True,
        )

        for detection in sorted_detections:
            if len(detections) >= self.max_faces:
                break

            score = detection.score[0] if detection.score else 0
            if score < self.threshold:
                continue

            # Get bounding box
            bbox_data = detection.location_data.relative_bounding_box

            # Convert relative coordinates to absolute
            x1 = int(bbox_data.xmin * w)
            y1 = int(bbox_data.ymin * h)
            box_w = int(bbox_data.width * w)
            box_h = int(bbox_data.height * h)

            # Clamp to image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x1 + box_w)
            y2 = min(h, y1 + box_h)
            box_w = x2 - x1
            box_h = y2 - y1

            # Check minimum face size
            if box_w < self.min_face_size or box_h < self.min_face_size:
                continue

            bbox = BoundingBox(
                x=x1,
                y=y1,
                width=box_w,
                height=box_h,
            )

            # Get landmarks if available and requested
            # MediaPipe provides 6 keypoints: right eye, left eye, nose tip,
            # mouth center, right ear tragion, left ear tragion
            face_landmarks = None
            if self.return_landmarks and detection.location_data.relative_keypoints:
                keypoints = detection.location_data.relative_keypoints
                face_landmarks = np.array([
                    [kp.x * w, kp.y * h] for kp in keypoints
                ])

            # Crop face image if requested
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

            # Draw confidence
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
