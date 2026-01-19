"""Facial emotion recognition module using ViT-based models."""

from typing import Any

import numpy as np

from emotion_detector.core.config import ModelConfig
from emotion_detector.core.types import (
    EmotionScores,
    FaceDetection,
    FacialEmotionResult,
)
from emotion_detector.models.base import BaseModel

# Try to import transformers
try:
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class FacialEmotionRecognizer(BaseModel[FaceDetection, FacialEmotionResult]):
    """Facial emotion recognition using Vision Transformer models.

    Uses HuggingFace transformer models for facial expression recognition.
    Default model: trpakov/vit-face-expression

    Example:
        >>> config = ModelConfig(
        ...     model_id="trpakov/vit-face-expression",
        ...     device="cuda"
        ... )
        >>> recognizer = FacialEmotionRecognizer(config)
        >>> recognizer.load()
        >>> result = recognizer.predict(face_detection)
        >>> print(result.emotions.dominant_emotion)
    """

    # Standard emotion label mapping
    EMOTION_LABELS = [
        "angry",
        "disgusted",
        "fearful",
        "happy",
        "neutral",
        "sad",
        "surprised",
    ]

    def __init__(self, config: ModelConfig) -> None:
        """Initialize facial emotion recognizer.

        Args:
            config: Model configuration including model_id and device.
        """
        super().__init__(config)
        self._processor: Any = None
        self._model: Any = None
        self._device: Any = None
        self._label_mapping: dict[int, str] = {}

    def load(self) -> None:
        """Load the facial emotion model."""
        if self._is_loaded:
            return

        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "transformers not available. Install with: pip install transformers torch"
            )

        model_id = self.config.model_id

        # Load processor and model
        # Use safetensors format to avoid torch.load security issues
        self._processor = AutoImageProcessor.from_pretrained(
            model_id,
            cache_dir=self.config.cache_dir,
        )
        self._model = AutoModelForImageClassification.from_pretrained(
            model_id,
            cache_dir=self.config.cache_dir,
            use_safetensors=True,
        )

        # Set device
        self._device = torch.device(self.config.device)
        self._model = self._model.to(self._device)
        self._model.eval()

        # Build label mapping from model config
        if hasattr(self._model.config, "id2label"):
            self._label_mapping = {
                int(k): v.lower() for k, v in self._model.config.id2label.items()
            }
        else:
            # Use default mapping
            self._label_mapping = {
                i: label for i, label in enumerate(self.EMOTION_LABELS)
            }

        self._is_loaded = True

    def unload(self) -> None:
        """Unload the model."""
        self._processor = None
        self._model = None
        self._device = None
        self._label_mapping = {}
        self._is_loaded = False

    def predict(self, input_data: FaceDetection) -> FacialEmotionResult:
        """Predict emotions from a face detection.

        Args:
            input_data: Face detection with cropped face image.

        Returns:
            Facial emotion result with emotion scores.

        Raises:
            RuntimeError: If model is not loaded.
            ValueError: If face image is not available in detection.
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        if input_data.face_image is None:
            raise ValueError("Face detection must include face_image")

        # Preprocess image
        face_image = input_data.face_image

        # Convert BGR to RGB if needed (OpenCV format)
        if len(face_image.shape) == 3 and face_image.shape[2] == 3:
            # Assume BGR, convert to RGB
            import cv2
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        # Process with HuggingFace processor
        inputs = self._processor(
            images=face_image,
            return_tensors="pt",
        )

        # Move to device
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits

        # Get probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)
        probs = probs.cpu().numpy()[0]

        # Map to emotion scores
        emotion_dict = self._map_to_emotions(probs)

        emotions = EmotionScores.from_dict(emotion_dict)

        # Get confidence (max probability)
        confidence = float(np.max(probs))

        return FacialEmotionResult(
            face_detection=input_data,
            emotions=emotions,
            confidence=confidence,
        )

    def predict_batch(
        self,
        face_detections: list[FaceDetection],
    ) -> list[FacialEmotionResult]:
        """Predict emotions for multiple face detections.

        Args:
            face_detections: List of face detections.

        Returns:
            List of emotion results.
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        if not face_detections:
            return []

        # Filter detections with face images
        valid_detections = [d for d in face_detections if d.face_image is not None]

        if not valid_detections:
            return []

        # Preprocess all images
        import cv2
        face_images = []
        for det in valid_detections:
            img = det.face_image
            if img is not None:
                if len(img.shape) == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                face_images.append(img)

        # Batch process
        inputs = self._processor(
            images=face_images,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits

        probs = torch.nn.functional.softmax(logits, dim=-1)
        probs = probs.cpu().numpy()

        # Create results
        results = []
        for i, det in enumerate(valid_detections):
            emotion_dict = self._map_to_emotions(probs[i])
            emotions = EmotionScores.from_dict(emotion_dict)

            results.append(
                FacialEmotionResult(
                    face_detection=det,
                    emotions=emotions,
                    confidence=float(np.max(probs[i])),
                )
            )

        return results

    def _map_to_emotions(self, probs: np.ndarray) -> dict[str, float]:
        """Map model output probabilities to standard emotion labels.

        Args:
            probs: Probability array from model.

        Returns:
            Dictionary mapping emotion names to probabilities.
        """
        emotion_dict: dict[str, float] = {
            "happy": 0.0,
            "sad": 0.0,
            "angry": 0.0,
            "fearful": 0.0,
            "surprised": 0.0,
            "disgusted": 0.0,
            "neutral": 0.0,
        }

        for idx, prob in enumerate(probs):
            if idx in self._label_mapping:
                label = self._label_mapping[idx].lower()
                # Map model labels to standard labels
                if label in ("happy", "happiness"):
                    emotion_dict["happy"] += prob
                elif label in ("sad", "sadness"):
                    emotion_dict["sad"] += prob
                elif label in ("angry", "anger"):
                    emotion_dict["angry"] += prob
                elif label in ("fearful", "fear"):
                    emotion_dict["fearful"] += prob
                elif label in ("surprised", "surprise"):
                    emotion_dict["surprised"] += prob
                elif label in ("disgusted", "disgust"):
                    emotion_dict["disgusted"] += prob
                elif label in ("neutral",):
                    emotion_dict["neutral"] += prob
                elif label in ("contempt",):
                    # Map contempt to angry/disgusted
                    emotion_dict["angry"] += prob * 0.5
                    emotion_dict["disgusted"] += prob * 0.5

        return emotion_dict

    def predict_from_image(self, image: np.ndarray) -> FacialEmotionResult | None:
        """Predict emotions directly from an image (skipping face detection).

        Useful when face is already cropped.

        Args:
            image: Face image as numpy array (H, W, C).

        Returns:
            Emotion result or None if prediction fails.
        """
        # Create a dummy face detection
        dummy_detection = FaceDetection(
            bbox=type("BBox", (), {
                "x": 0, "y": 0,
                "width": image.shape[1],
                "height": image.shape[0],
                "to_tuple": lambda s: (0, 0, image.shape[1], image.shape[0]),
                "to_xyxy": lambda s: (0, 0, image.shape[1], image.shape[0]),
            })(),  # type: ignore
            confidence=1.0,
            face_image=image,
        )
        return self.predict(dummy_detection)

