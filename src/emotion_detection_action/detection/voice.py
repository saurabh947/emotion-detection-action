"""Voice activity detection module using Silero VAD."""

from typing import Any

import numpy as np
import torch

from emotion_detection_action.core.config import ModelConfig
from emotion_detection_action.core.types import VoiceDetection
from emotion_detection_action.inputs.base import AudioChunk
from emotion_detection_action.models.base import BaseModel

SILERO_REPO = "snakers4/silero-vad"
SILERO_SAMPLE_RATE = 16000


class VoiceActivityDetector(BaseModel[AudioChunk, VoiceDetection | None]):
    """Voice Activity Detection using Silero VAD.

    Uses a pre-trained deep learning model (Silero) that is robust to
    background noise -- barking, typing, music, HVAC, etc. Loaded via
    torch.hub so no extra packages are needed beyond PyTorch.

    The model expects 16 kHz mono audio. Input at other sample rates is
    automatically resampled.

    Example:
        >>> config = ModelConfig(model_id="silero-vad", device="cpu")
        >>> vad = VoiceActivityDetector(config)
        >>> vad.load()
        >>> detection = vad.predict(audio_chunk)
        >>> if detection and detection.is_speech:
        ...     print("Speech detected!")
    """

    def __init__(
        self,
        config: ModelConfig,
        speech_threshold: float = 0.5,
    ) -> None:
        """Initialize voice activity detector.

        Args:
            config: Model configuration.
            speech_threshold: Probability threshold above which a frame is
                classified as speech (0-1). Lower = more sensitive.
        """
        super().__init__(config)
        self.speech_threshold = speech_threshold
        self._model: Any = None

    def load(self) -> None:
        """Load the Silero VAD model via torch.hub."""
        if self._is_loaded:
            return

        model, _ = torch.hub.load(
            repo_or_dir=SILERO_REPO,
            model="silero_vad",
            trust_repo=True,
        )
        self._model = model
        self._is_loaded = True

    def unload(self) -> None:
        """Unload the model and free memory."""
        if self._model is not None:
            self._model.reset_states()
        self._model = None
        self._is_loaded = False

    def predict(self, input_data: AudioChunk) -> VoiceDetection | None:
        """Detect voice activity in an audio chunk.

        Args:
            input_data: Audio chunk to analyze.

        Returns:
            VoiceDetection result or None if the chunk is empty.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if not self._is_loaded or self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        audio = self._to_float_tensor(input_data.data)
        if audio.numel() == 0:
            return None

        # Resample to 16 kHz if needed
        if input_data.sample_rate != SILERO_SAMPLE_RATE:
            audio = self._resample(audio, input_data.sample_rate, SILERO_SAMPLE_RATE)

        # Silero expects chunks of 512 samples (32 ms @ 16 kHz)
        # Process in 512-sample windows, average the speech probabilities
        window_size = 512
        speech_probs: list[float] = []

        for start in range(0, len(audio) - window_size + 1, window_size):
            chunk = audio[start : start + window_size]
            prob = self._model(chunk, SILERO_SAMPLE_RATE).item()
            speech_probs.append(prob)

        if not speech_probs:
            # Audio shorter than one window -- run on the whole thing
            # (pad to minimum length)
            padded = torch.nn.functional.pad(audio, (0, window_size - len(audio)))
            prob = self._model(padded, SILERO_SAMPLE_RATE).item()
            speech_probs.append(prob)

        avg_prob = float(np.mean(speech_probs))
        is_speech = avg_prob >= self.speech_threshold

        self._model.reset_states()

        return VoiceDetection(
            is_speech=is_speech,
            confidence=avg_prob,
            start_time=input_data.start_time,
            end_time=input_data.end_time,
            audio_segment=input_data.data if is_speech else None,
        )

    @staticmethod
    def _to_float_tensor(audio: np.ndarray) -> torch.Tensor:
        """Convert numpy audio to a float32 torch tensor in [-1, 1]."""
        if audio.dtype in (np.float32, np.float64):
            return torch.from_numpy(audio.astype(np.float32))
        elif audio.dtype == np.int16:
            return torch.from_numpy(audio.astype(np.float32) / 32768.0)
        elif audio.dtype == np.int32:
            return torch.from_numpy(audio.astype(np.float32) / 2147483648.0)
        return torch.from_numpy(audio.astype(np.float32))

    @staticmethod
    def _resample(audio: torch.Tensor, orig_rate: int, target_rate: int) -> torch.Tensor:
        """Resample audio via linear interpolation."""
        if orig_rate == target_rate:
            return audio
        duration = len(audio) / orig_rate
        new_length = int(duration * target_rate)
        return torch.nn.functional.interpolate(
            audio.unsqueeze(0).unsqueeze(0),
            size=new_length,
            mode="linear",
            align_corners=False,
        ).squeeze()
