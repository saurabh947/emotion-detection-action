"""Main EmotionDetector class orchestrating the full pipeline."""

import asyncio
from typing import AsyncIterator

import numpy as np

from emotion_detection_action.actions.base import BaseActionHandler
from emotion_detection_action.actions.logging_handler import LoggingActionHandler
from emotion_detection_action.core.config import Config, ModelConfig
from emotion_detection_action.core.types import (
    ActionCommand,
    AttentionResult,
    DetectionResult,
    EmotionResult,
    PipelineResult,
)
from emotion_detection_action.detection.attention import AttentionDetector
from emotion_detection_action.detection.face import FaceDetector
from emotion_detection_action.detection.voice import VoiceActivityDetector
from emotion_detection_action.emotion.attention import AttentionAnalyzer
from emotion_detection_action.emotion.facial import FacialEmotionRecognizer
from emotion_detection_action.emotion.fusion import EmotionFusion
from emotion_detection_action.emotion.smoothing import EmotionSmoother, SmoothingConfig
from emotion_detection_action.emotion.speech import SpeechEmotionRecognizer
from emotion_detection_action.inputs.audio import AudioInput
from emotion_detection_action.inputs.base import AudioChunk, VideoFrame
from emotion_detection_action.inputs.video import VideoInput
from emotion_detection_action.models.vla.base import BaseVLAModel, VLAInput
from emotion_detection_action.models.vla.openvla import OpenVLAModel


class EmotionDetector:
    """Main emotion detector class orchestrating the full pipeline.

    This class integrates all components of the emotion detection system:
    - Face detection
    - Voice activity detection
    - Attention analysis (gaze, pupil dilation, fixation)
    - Facial emotion recognition
    - Speech emotion recognition
    - Multimodal fusion
    - VLA-based action generation

    Example:
        >>> from emotion_detection_action import EmotionDetector, Config

        >>> # Create detector with default config
        >>> detector = EmotionDetector()
        >>> detector.initialize()

        >>> # Real-time streaming from webcam + microphone
        >>> async for result in detector.stream(camera=0, microphone=0):
        ...     print(result.emotion.dominant_emotion)
        ...     if result.emotion.attention:
        ...         print(f"Stress: {result.emotion.attention.stress_score:.2f}")

        >>> # Process a single frame (for custom real-time pipelines)
        >>> result = detector.process_frame(frame, audio, timestamp=0.0)
        >>> print(result.emotion.dominant_emotion)
    """

    def __init__(
        self,
        config: Config | None = None,
        action_handler: BaseActionHandler | None = None,
    ) -> None:
        """Initialize the emotion detector.

        Args:
            config: Configuration for the detector. Uses defaults if None.
            action_handler: Custom action handler. Uses stub if None.
        """
        self.config = config or Config()
        self.action_handler = action_handler or LoggingActionHandler(verbose=self.config.verbose)

        # Components (initialized lazily)
        self._face_detector: FaceDetector | None = None
        self._voice_detector: VoiceActivityDetector | None = None
        self._attention_detector: AttentionDetector | None = None
        self._attention_analyzer: AttentionAnalyzer | None = None
        self._facial_emotion: FacialEmotionRecognizer | None = None
        self._speech_emotion: SpeechEmotionRecognizer | None = None
        self._fusion: EmotionFusion | None = None
        self._smoother: EmotionSmoother | None = None
        self._vla_model: BaseVLAModel | None = None

        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        """Check if detector is initialized."""
        return self._initialized

    def initialize(self, load_vla: bool = True) -> None:
        """Initialize all pipeline components.

        Args:
            load_vla: Whether to load the VLA model. Set False for emotion-only mode.
        """
        if self._initialized:
            return

        # Initialize face detector
        face_config = ModelConfig(
            model_id=self.config.face_detection_model,
            device=self.config.device,
            extra_kwargs={"threshold": self.config.face_detection_threshold},
        )
        self._face_detector = FaceDetector(
            face_config,
            threshold=self.config.face_detection_threshold,
            min_face_size=self.config.face_min_size,
            max_faces=self.config.max_faces,
        )
        self._face_detector.load()

        # Initialize voice activity detector
        vad_config = ModelConfig(model_id="webrtcvad", device="cpu")
        self._voice_detector = VoiceActivityDetector(
            vad_config,
            aggressiveness=self.config.vad_aggressiveness,
        )
        self._voice_detector.load()

        # Initialize attention detector (if enabled)
        if self.config.attention_analysis_enabled:
            try:
                attention_config = ModelConfig(model_id="mediapipe", device="cpu")
                self._attention_detector = AttentionDetector(attention_config)
                self._attention_detector.load()
                self._attention_analyzer = AttentionAnalyzer(detector=self._attention_detector)
            except RuntimeError as e:
                # MediaPipe not available, disable attention analysis
                if self.config.verbose:
                    print(f"Attention analysis disabled: {e}")
                self._attention_detector = None
                self._attention_analyzer = None

        # Initialize facial emotion recognizer
        facial_config = self.config.get_facial_emotion_config()
        self._facial_emotion = FacialEmotionRecognizer(facial_config)
        self._facial_emotion.load()

        # Initialize speech emotion recognizer
        speech_config = self.config.get_speech_emotion_config()
        self._speech_emotion = SpeechEmotionRecognizer(
            speech_config,
            target_sample_rate=self.config.sample_rate,
        )
        self._speech_emotion.load()

        # Initialize fusion module
        self._fusion = EmotionFusion(
            strategy=self.config.fusion_strategy,
            visual_weight=self.config.facial_weight,
            audio_weight=self.config.speech_weight,
            confidence_threshold=self.config.fusion_confidence_threshold,
            attention_weight=self.config.attention_weight,
            attention_stress_amplification=self.config.attention_stress_amplification,
            attention_engagement_threshold=self.config.attention_engagement_threshold,
        )

        # Initialize temporal smoother
        smoothing_config = SmoothingConfig(
            strategy=self.config.smoothing_strategy,
            window_size=self.config.smoothing_window,
            ema_alpha=self.config.smoothing_ema_alpha,
            hysteresis_threshold=self.config.smoothing_hysteresis_threshold,
            hysteresis_frames=self.config.smoothing_hysteresis_frames,
        )
        self._smoother = EmotionSmoother(smoothing_config)

        # Initialize VLA model (optional)
        if load_vla and self.config.vla_enabled:
            vla_config = self.config.get_vla_config()
            self._vla_model = OpenVLAModel(vla_config)
            self._vla_model.load()

        # Connect action handler
        self.action_handler.connect()

        self._initialized = True

    def shutdown(self) -> None:
        """Shutdown and release all resources."""
        if self._face_detector:
            self._face_detector.unload()
        if self._voice_detector:
            self._voice_detector.unload()
        if self._attention_detector:
            self._attention_detector.unload()
        if self._facial_emotion:
            self._facial_emotion.unload()
        if self._speech_emotion:
            self._speech_emotion.unload()
        if self._vla_model:
            self._vla_model.unload()

        self.action_handler.disconnect()
        self._initialized = False

    async def stream(
        self,
        camera: int = 0,
        microphone: int | str | None = None,
    ) -> AsyncIterator[PipelineResult]:
        """Stream real-time emotion detection from camera and microphone.

        Args:
            camera: Camera device index (0 = default camera).
            microphone: Audio device index or name. None to disable audio.

        Yields:
            Pipeline results for each processed frame.
        """
        if not self._initialized:
            self.initialize()

        video_input = VideoInput(frame_skip=self.config.frame_skip)
        video_input.open(camera)

        audio_input: AudioInput | None = None
        if microphone is not None:
            audio_input = AudioInput(sample_rate=self.config.sample_rate)
            audio_input.open(microphone)

        try:
            async for frame in video_input:
                # Get audio chunk if available
                audio_chunk = None
                if audio_input:
                    audio_chunk = audio_input.read()

                result = self._process_frame(frame, audio_chunk)
                if result:
                    yield result

                # Small delay to prevent CPU overload
                await asyncio.sleep(0.001)

        finally:
            video_input.close()
            if audio_input:
                audio_input.close()

    def process_frame(
        self,
        frame: np.ndarray,
        audio: np.ndarray | None = None,
        timestamp: float = 0.0,
    ) -> PipelineResult | None:
        """Process a single frame with optional audio.

        Lower-level API for custom processing pipelines.

        Args:
            frame: Video frame as numpy array (H, W, C) in BGR format.
            audio: Optional audio data as numpy array.
            timestamp: Frame timestamp in seconds.

        Returns:
            Pipeline result or None if processing failed.
        """
        if not self._initialized:
            self.initialize()

        video_frame = VideoFrame(data=frame, timestamp=timestamp, frame_number=0)

        audio_chunk = None
        if audio is not None:
            audio_chunk = AudioChunk(
                data=audio,
                sample_rate=self.config.sample_rate,
                start_time=timestamp,
            )

        return self._process_frame(video_frame, audio_chunk)

    def _process_frame(
        self,
        frame: VideoFrame,
        audio_chunk: AudioChunk | None,
    ) -> PipelineResult | None:
        """Internal method to process a frame through the pipeline.

        Args:
            frame: Video frame to process.
            audio_chunk: Optional audio chunk.

        Returns:
            Pipeline result or None.
        """
        assert self._face_detector is not None
        assert self._facial_emotion is not None
        assert self._fusion is not None

        timestamp = frame.timestamp

        # Convert BGR to RGB for face detection
        rgb_frame = VideoInput.bgr_to_rgb(frame.data)

        # Detect faces
        faces = self._face_detector.predict(rgb_frame)

        # Process voice activity if audio available
        voice_detection = None
        if audio_chunk and self._voice_detector:
            voice_detection = self._voice_detector.predict(audio_chunk)

        # Process attention/gaze analysis
        gaze_detection = None
        attention_result: AttentionResult | None = None
        if self._attention_detector and self._attention_analyzer:
            gaze_detection = self._attention_detector.predict(rgb_frame)
            if gaze_detection:
                # Get metrics from detector
                blink_rate = self._attention_detector.get_blink_rate()
                pupil_dilation = self._attention_detector.get_pupil_dilation()
                gaze_stability = self._attention_detector.get_gaze_stability()

                # Analyze attention
                attention_result = self._attention_analyzer.analyze(
                    gaze=gaze_detection,
                    blink_rate=blink_rate,
                    pupil_dilation=pupil_dilation,
                    gaze_stability=gaze_stability,
                    timestamp=timestamp,
                )

        # Create detection result
        detection = DetectionResult(
            timestamp=timestamp,
            faces=faces,
            voice=voice_detection,
            gaze=gaze_detection,
            frame=frame.data,
        )

        # Facial emotion recognition
        facial_result = None
        if faces:
            # Process first face (could extend to multiple)
            facial_result = self._facial_emotion.predict(faces[0])

        # Speech emotion recognition
        speech_result = None
        if voice_detection and voice_detection.is_speech and self._speech_emotion:
            speech_result = self._speech_emotion.predict(voice_detection)

        # Fuse emotions (including attention analysis)
        if facial_result is None and speech_result is None:
            return None

        emotion_result = self._fusion.fuse(
            facial_result, speech_result, attention_result, timestamp
        )

        # Apply temporal smoothing
        if self._smoother is not None:
            emotion_result = self._smoother.smooth(emotion_result)

        # Generate action
        action = self._generate_action(emotion_result, frame.data)

        # Execute action through handler
        self.action_handler.execute(action)

        return PipelineResult(
            timestamp=timestamp,
            detection=detection,
            emotion=emotion_result,
            action=action,
        )

    def _generate_action(
        self,
        emotion: EmotionResult,
        image: np.ndarray | None,
    ) -> ActionCommand:
        """Generate action based on emotion.

        Args:
            emotion: Detected emotion result.
            image: Optional visual context.

        Returns:
            Generated action command.
        """
        # Use VLA model if available
        if self._vla_model and self._vla_model.is_loaded:
            return self._vla_model.generate_emotion_response(emotion, image)

        # Fall back to stub action
        return ActionCommand.stub(emotion)

    def get_emotion_only(
        self,
        frame: np.ndarray,
        audio: np.ndarray | None = None,
    ) -> EmotionResult | None:
        """Get emotion result without action generation.

        Useful when you only need emotion detection.

        Args:
            frame: Video frame in BGR format.
            audio: Optional audio data.

        Returns:
            Emotion result or None.
        """
        result = self.process_frame(frame, audio)
        return result.emotion if result else None

    def __enter__(self) -> "EmotionDetector":
        """Context manager entry - initialize."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - shutdown."""
        self.shutdown()

    def __repr__(self) -> str:
        return (
            f"EmotionDetector(vla={self.config.vla_model}, "
            f"initialized={self._initialized})"
        )
