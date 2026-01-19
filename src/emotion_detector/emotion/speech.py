"""Speech emotion recognition module using Wav2Vec2-based models."""

from typing import Any

import numpy as np

from emotion_detector.core.config import ModelConfig
from emotion_detector.core.types import (
    EmotionScores,
    SpeechEmotionResult,
    VoiceDetection,
)
from emotion_detector.models.base import BaseModel

# Try to import transformers
try:
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class SpeechEmotionRecognizer(BaseModel[VoiceDetection, SpeechEmotionResult]):
    """Speech emotion recognition using Wav2Vec2-based models.

    Uses HuggingFace transformer models for speech emotion recognition.
    Default model: superb/wav2vec2-base-superb-er (trained on IEMOCAP dataset)
    
    Supported emotions (from speech): happy, sad, angry, neutral
    Note: Other emotions (fearful, surprised, disgusted) are mapped from facial only.

    Example:
        >>> config = ModelConfig(
        ...     model_id="superb/wav2vec2-base-superb-er",
        ...     device="cuda"
        ... )
        >>> recognizer = SpeechEmotionRecognizer(config)
        >>> recognizer.load()
        >>> result = recognizer.predict(voice_detection)
        >>> print(result.emotions.dominant_emotion)
    """

    # SUPERB model outputs 4 emotions: neu, hap, ang, sad
    # We map these to our standard labels
    EMOTION_LABELS = [
        "neutral",
        "happy",
        "angry",
        "sad",
    ]

    def __init__(
        self,
        config: ModelConfig,
        target_sample_rate: int = 16000,
    ) -> None:
        """Initialize speech emotion recognizer.

        Args:
            config: Model configuration including model_id and device.
            target_sample_rate: Target sample rate for audio processing.
        """
        super().__init__(config)
        self.target_sample_rate = target_sample_rate

        self._feature_extractor: Any = None
        self._model: Any = None
        self._device: Any = None
        self._label_mapping: dict[int, str] = {}

    def load(self) -> None:
        """Load the speech emotion model."""
        if self._is_loaded:
            return

        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "transformers not available. Install with: pip install transformers torch"
            )

        model_id = self.config.model_id

        # Load feature extractor and model
        # Use safetensors format to avoid torch.load security issues
        self._feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_id,
            cache_dir=self.config.cache_dir,
        )
        self._model = AutoModelForAudioClassification.from_pretrained(
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
        self._feature_extractor = None
        self._model = None
        self._device = None
        self._label_mapping = {}
        self._is_loaded = False

    def predict(self, input_data: VoiceDetection) -> SpeechEmotionResult:
        """Predict emotions from voice detection.

        Args:
            input_data: Voice detection with audio segment.

        Returns:
            Speech emotion result with emotion scores.

        Raises:
            RuntimeError: If model is not loaded.
            ValueError: If audio segment is not available.
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        if input_data.audio_segment is None:
            raise ValueError("Voice detection must include audio_segment")

        # Get audio data
        audio = input_data.audio_segment

        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Normalize audio
        if np.abs(audio).max() > 1.0:
            audio = audio / np.abs(audio).max()

        # Process with feature extractor
        inputs = self._feature_extractor(
            audio,
            sampling_rate=self.target_sample_rate,
            return_tensors="pt",
            padding=True,
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

        return SpeechEmotionResult(
            voice_detection=input_data,
            emotions=emotions,
            confidence=confidence,
        )

    def predict_from_audio(
        self,
        audio: np.ndarray,
        sample_rate: int | None = None,
    ) -> SpeechEmotionResult | None:
        """Predict emotions directly from audio array.

        Args:
            audio: Audio data as numpy array.
            sample_rate: Sample rate of audio. Uses target_sample_rate if None.

        Returns:
            Emotion result or None if prediction fails.
        """
        if sample_rate is None:
            sample_rate = self.target_sample_rate

        # Resample if needed
        if sample_rate != self.target_sample_rate:
            audio = self._resample(audio, sample_rate, self.target_sample_rate)

        # Create dummy voice detection
        dummy_detection = VoiceDetection(
            is_speech=True,
            confidence=1.0,
            start_time=0.0,
            end_time=len(audio) / self.target_sample_rate,
            audio_segment=audio,
        )

        return self.predict(dummy_detection)

    def predict_batch(
        self,
        voice_detections: list[VoiceDetection],
    ) -> list[SpeechEmotionResult]:
        """Predict emotions for multiple voice detections.

        Args:
            voice_detections: List of voice detections.

        Returns:
            List of emotion results.
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        if not voice_detections:
            return []

        # Filter detections with audio segments
        valid_detections = [
            d for d in voice_detections if d.audio_segment is not None
        ]

        if not valid_detections:
            return []

        # Prepare audio arrays
        audio_arrays = []
        for det in valid_detections:
            audio = det.audio_segment
            if audio is not None:
                if audio.dtype != np.float32:
                    audio = audio.astype(np.float32)
                if np.abs(audio).max() > 1.0:
                    audio = audio / np.abs(audio).max()
                audio_arrays.append(audio)

        # Batch process
        inputs = self._feature_extractor(
            audio_arrays,
            sampling_rate=self.target_sample_rate,
            return_tensors="pt",
            padding=True,
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
                SpeechEmotionResult(
                    voice_detection=det,
                    emotions=emotions,
                    confidence=float(np.max(probs[i])),
                )
            )

        return results

    def _map_to_emotions(self, probs: np.ndarray) -> dict[str, float]:
        """Map model output probabilities to standard emotion labels.

        The SUPERB model (superb/wav2vec2-base-superb-er) outputs 4 classes:
        - neu (neutral), hap (happy), ang (angry), sad (sad)
        
        Other emotions (fearful, surprised, disgusted) are set to 0.0 as
        they are not supported by the speech model and should come from
        facial emotion recognition via fusion.

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
                # SUPERB uses: neu, hap, ang, sad
                if label in ("happy", "happiness", "hap", "joy"):
                    emotion_dict["happy"] += float(prob)
                elif label in ("sad", "sadness"):
                    emotion_dict["sad"] += float(prob)
                elif label in ("angry", "anger", "ang"):
                    emotion_dict["angry"] += float(prob)
                elif label in ("neutral", "neu", "calm"):
                    emotion_dict["neutral"] += float(prob)
                elif label in ("fearful", "fear"):
                    emotion_dict["fearful"] += float(prob)
                elif label in ("surprised", "surprise"):
                    emotion_dict["surprised"] += float(prob)
                elif label in ("disgusted", "disgust"):
                    emotion_dict["disgusted"] += float(prob)
                # Note: Unknown labels are ignored rather than mapped to neutral
                # to avoid artificially inflating neutral scores

        return emotion_dict

    @staticmethod
    def _resample(
        audio: np.ndarray,
        orig_rate: int,
        target_rate: int,
    ) -> np.ndarray:
        """Resample audio to target sample rate.

        Args:
            audio: Input audio.
            orig_rate: Original sample rate.
            target_rate: Target sample rate.

        Returns:
            Resampled audio.
        """
        if orig_rate == target_rate:
            return audio

        duration = len(audio) / orig_rate
        new_length = int(duration * target_rate)
        return np.interp(
            np.linspace(0, len(audio), new_length),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)

