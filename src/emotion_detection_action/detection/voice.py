"""Voice activity detection module."""

from typing import Any

import numpy as np

from emotion_detection_action.core.config import ModelConfig
from emotion_detection_action.core.types import VoiceDetection
from emotion_detection_action.inputs.base import AudioChunk
from emotion_detection_action.models.base import BaseModel

# Try to import webrtcvad
try:
    import webrtcvad
    WEBRTCVAD_AVAILABLE = True
except ImportError:
    WEBRTCVAD_AVAILABLE = False


class VoiceActivityDetector(BaseModel[AudioChunk, VoiceDetection | None]):
    """Voice Activity Detection using WebRTC VAD.

    Detects speech segments in audio using Google's WebRTC Voice Activity
    Detector. Returns detection results indicating whether speech is present.

    Example:
        >>> config = ModelConfig(model_id="webrtcvad", device="cpu")
        >>> vad = VoiceActivityDetector(config, aggressiveness=2)
        >>> vad.load()
        >>> detection = vad.predict(audio_chunk)
        >>> if detection and detection.is_speech:
        ...     print("Speech detected!")
    """

    # WebRTC VAD only works with specific sample rates
    SUPPORTED_SAMPLE_RATES = {8000, 16000, 32000, 48000}
    # And specific frame durations (in ms)
    SUPPORTED_FRAME_DURATIONS = {10, 20, 30}

    def __init__(
        self,
        config: ModelConfig,
        aggressiveness: int = 2,
        frame_duration_ms: int = 30,
        speech_threshold: float = 0.5,
    ) -> None:
        """Initialize voice activity detector.

        Args:
            config: Model configuration.
            aggressiveness: VAD aggressiveness (0-3). Higher = more aggressive
                filtering of non-speech.
            frame_duration_ms: Frame duration in milliseconds (10, 20, or 30).
            speech_threshold: Threshold for speech detection (0-1).
        """
        super().__init__(config)

        if aggressiveness not in range(4):
            raise ValueError("aggressiveness must be 0, 1, 2, or 3")
        if frame_duration_ms not in self.SUPPORTED_FRAME_DURATIONS:
            raise ValueError(
                f"frame_duration_ms must be one of {self.SUPPORTED_FRAME_DURATIONS}"
            )

        self.aggressiveness = aggressiveness
        self.frame_duration_ms = frame_duration_ms
        self.speech_threshold = speech_threshold

        self._vad: Any = None

    def load(self) -> None:
        """Load the VAD model."""
        if self._is_loaded:
            return

        if not WEBRTCVAD_AVAILABLE:
            raise RuntimeError(
                "webrtcvad not available. Install with: pip install webrtcvad"
            )

        self._vad = webrtcvad.Vad(self.aggressiveness)
        self._is_loaded = True

    def unload(self) -> None:
        """Unload the model."""
        self._vad = None
        self._is_loaded = False

    def predict(self, input_data: AudioChunk) -> VoiceDetection | None:
        """Detect voice activity in an audio chunk.

        Args:
            input_data: Audio chunk to analyze.

        Returns:
            VoiceDetection result or None if detection failed.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        if self._vad is None:
            return None

        sample_rate = input_data.sample_rate

        # Validate sample rate
        if sample_rate not in self.SUPPORTED_SAMPLE_RATES:
            # Resample to nearest supported rate
            target_rate = min(
                self.SUPPORTED_SAMPLE_RATES,
                key=lambda x: abs(x - sample_rate),
            )
            audio_data = self._resample(
                input_data.data, sample_rate, target_rate
            )
            sample_rate = target_rate
        else:
            audio_data = input_data.data

        # Convert to int16 for webrtcvad
        if audio_data.dtype != np.int16:
            audio_data = self._to_int16(audio_data)

        # Calculate frame size
        frame_size = int(sample_rate * self.frame_duration_ms / 1000)

        # Process frames and count speech frames
        speech_frames = 0
        total_frames = 0

        for start in range(0, len(audio_data) - frame_size + 1, frame_size):
            frame = audio_data[start : start + frame_size]
            frame_bytes = frame.tobytes()

            try:
                is_speech = self._vad.is_speech(frame_bytes, sample_rate)
                if is_speech:
                    speech_frames += 1
                total_frames += 1
            except Exception:
                continue

        if total_frames == 0:
            return None

        speech_ratio = speech_frames / total_frames
        is_speech = speech_ratio >= self.speech_threshold

        return VoiceDetection(
            is_speech=is_speech,
            confidence=speech_ratio,
            start_time=input_data.start_time,
            end_time=input_data.end_time,
            audio_segment=input_data.data if is_speech else None,
        )

    @staticmethod
    def _to_int16(audio: np.ndarray) -> np.ndarray:
        """Convert audio to int16 format.

        Args:
            audio: Audio data in any format.

        Returns:
            Audio data as int16.
        """
        if audio.dtype == np.float32 or audio.dtype == np.float64:
            return (audio * 32767).astype(np.int16)
        elif audio.dtype == np.int32:
            return (audio // 65536).astype(np.int16)
        elif audio.dtype == np.int16:
            return audio
        else:
            return audio.astype(np.int16)

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
            audio.astype(np.float32),
        ).astype(audio.dtype)

