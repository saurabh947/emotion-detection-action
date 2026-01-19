#!/usr/bin/env python3
"""Real-time multimodal emotion detection example.

This example shows video (facial) and audio (speech) emotion detection
SEPARATELY without fusion. Each modality has its own panel.

Supported emotions:
    - Facial: happy, sad, angry, fearful, surprised, disgusted, neutral (7)
    - Speech: happy, sad, angry, neutral (4) - using SUPERB model

Requirements:
    - Webcam connected to the system
    - Microphone connected to the system
    - OpenCV for visualization

Usage:
    python realtime_multimodal.py
    python realtime_multimodal.py --camera 1  # Use different camera
"""

import argparse
import signal
import sys
import threading
import time
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Any

import cv2
import numpy as np

from emotion_detector.core.config import ModelConfig
from emotion_detector.core.types import EmotionScores
from emotion_detector.detection.face import FaceDetector
from emotion_detector.detection.voice import VoiceActivityDetector
from emotion_detector.emotion.facial import FacialEmotionRecognizer
from emotion_detector.emotion.speech import SpeechEmotionRecognizer
from emotion_detector.inputs.audio import AudioInput
from emotion_detector.inputs.video import VideoInput


@dataclass
class FacialEmotionState:
    """Current state of facial emotion detection."""
    emotions: EmotionScores | None = None
    dominant: str = "none"
    confidence: float = 0.0
    face_detected: bool = False
    timestamp: float = 0.0


@dataclass
class AudioEmotionState:
    """Current state of audio emotion detection."""
    emotions: EmotionScores | None = None
    dominant: str = "none"
    confidence: float = 0.0
    speech_detected: bool = False
    timestamp: float = 0.0
    is_listening: bool = False


# Color scheme for emotions (BGR)
EMOTION_COLORS = {
    "happy": (0, 255, 255),      # Yellow
    "sad": (255, 0, 0),          # Blue
    "angry": (0, 0, 255),        # Red
    "fearful": (128, 0, 128),    # Purple
    "surprised": (0, 255, 0),    # Green
    "disgusted": (0, 128, 0),    # Dark green
    "neutral": (128, 128, 128),  # Gray
    "none": (100, 100, 100),     # Dark gray
}


class AudioEmotionProcessor:
    """Process audio in a separate thread for emotion detection."""

    def __init__(self, sample_rate: int = 16000, chunk_duration: float = 1.0):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        
        self._audio_input: AudioInput | None = None
        self._vad: VoiceActivityDetector | None = None
        self._speech_emotion: SpeechEmotionRecognizer | None = None
        
        self._state = AudioEmotionState()
        self._state_lock = threading.Lock()
        
        self._running = False
        self._thread: threading.Thread | None = None

    def initialize(self) -> None:
        """Initialize audio processing components."""
        print("[Audio] Initializing voice activity detector...")
        vad_config = ModelConfig(model_id="webrtcvad", device="cpu")
        self._vad = VoiceActivityDetector(vad_config, aggressiveness=2)
        self._vad.load()

        print("[Audio] Initializing speech emotion recognizer...")
        speech_config = ModelConfig(
            model_id="superb/wav2vec2-base-superb-er",
            device="cpu",
        )
        self._speech_emotion = SpeechEmotionRecognizer(
            speech_config,
            target_sample_rate=self.sample_rate,
        )
        self._speech_emotion.load()
        print("[Audio] Models loaded!")

    def start(self, audio_device: int | None = None) -> None:
        """Start audio processing in background thread."""
        if self._running:
            return

        self._audio_input = AudioInput(
            sample_rate=self.sample_rate,
            chunk_duration=self.chunk_duration,
        )
        
        try:
            self._audio_input.open_microphone(audio_device)
            print(f"[Audio] Microphone opened (device: {audio_device or 'default'})")
        except Exception as e:
            print(f"[Audio] Failed to open microphone: {e}")
            return

        self._running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop audio processing."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._audio_input:
            self._audio_input.close()

    def get_state(self) -> AudioEmotionState:
        """Get current audio emotion state (thread-safe)."""
        with self._state_lock:
            return AudioEmotionState(
                emotions=self._state.emotions,
                dominant=self._state.dominant,
                confidence=self._state.confidence,
                speech_detected=self._state.speech_detected,
                timestamp=self._state.timestamp,
                is_listening=self._state.is_listening,
            )

    def _process_loop(self) -> None:
        """Main audio processing loop (runs in thread)."""
        while self._running:
            try:
                self._update_state(is_listening=True)
                
                # Read audio chunk
                chunk = self._audio_input.read() if self._audio_input else None
                if chunk is None:
                    time.sleep(0.01)
                    continue

                # Voice activity detection
                vad_result = self._vad.predict(chunk) if self._vad else None
                
                if vad_result and vad_result.is_speech:
                    self._update_state(speech_detected=True)
                    
                    # Speech emotion recognition
                    if self._speech_emotion:
                        emotion_result = self._speech_emotion.predict(vad_result)
                        self._update_state(
                            emotions=emotion_result.emotions,
                            dominant=emotion_result.emotions.dominant_emotion.value,
                            confidence=emotion_result.confidence,
                            timestamp=time.time(),
                        )
                        print(f"[Audio] Detected: {emotion_result.emotions.dominant_emotion.value} "
                              f"({emotion_result.confidence:.0%})")
                else:
                    # No speech - decay the state after a while
                    current = self.get_state()
                    if time.time() - current.timestamp > 2.0:
                        self._update_state(speech_detected=False, dominant="none")

            except Exception as e:
                print(f"[Audio] Error: {e}")
                time.sleep(0.1)

    def _update_state(self, **kwargs: Any) -> None:
        """Update state thread-safely."""
        with self._state_lock:
            for key, value in kwargs.items():
                if hasattr(self._state, key):
                    setattr(self._state, key, value)


class MultimodalDisplay:
    """Display for showing both video and audio emotions separately."""

    def __init__(self, window_name: str = "Multimodal Emotion Detector"):
        self.window_name = window_name
        self._facial_state = FacialEmotionState()
        self._audio_state = AudioEmotionState()

    def start(self) -> None:
        """Start the display window."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1000, 600)

    def stop(self) -> None:
        """Close the display window."""
        cv2.destroyAllWindows()

    def update(
        self,
        frame: np.ndarray,
        facial_state: FacialEmotionState,
        audio_state: AudioEmotionState,
    ) -> bool:
        """Update display with current states.

        Returns:
            False if window closed, True otherwise.
        """
        self._facial_state = facial_state
        self._audio_state = audio_state

        display = frame.copy()

        # Draw face detection box and label
        self._draw_face_overlay(display)

        # Draw facial emotion panel (left side)
        self._draw_facial_panel(display)

        # Draw audio emotion panel (right side)
        self._draw_audio_panel(display)

        cv2.imshow(self.window_name, display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            return False

        return cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) >= 1

    def _draw_face_overlay(self, frame: np.ndarray) -> None:
        """Draw face detection and emotion on the video."""
        if not self._facial_state.face_detected:
            # Show "No face" message
            cv2.putText(
                frame,
                "No face detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            return

        emotion = self._facial_state.dominant
        color = EMOTION_COLORS.get(emotion, (255, 255, 255))

        # Draw emotion label at top
        label = f"Face: {emotion} ({self._facial_state.confidence:.0%})"
        cv2.putText(
            frame,
            label,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )

    def _draw_facial_panel(self, frame: np.ndarray) -> None:
        """Draw facial emotion panel on left side."""
        panel_width = 200
        panel_height = 220
        panel_x = 10
        panel_y = 50

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + panel_width, panel_y + panel_height),
            (0, 0, 0),
            -1,
        )
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Title with icon
        cv2.putText(
            frame,
            "FACIAL",
            (panel_x + 10, panel_y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 255),
            2,
        )

        # Status indicator
        status_color = (0, 255, 0) if self._facial_state.face_detected else (0, 0, 255)
        cv2.circle(frame, (panel_x + panel_width - 20, panel_y + 20), 8, status_color, -1)

        # Emotion bars
        self._draw_emotion_bars(
            frame,
            panel_x + 10,
            panel_y + 45,
            panel_width - 20,
            self._facial_state.emotions,
        )

    def _draw_audio_panel(self, frame: np.ndarray) -> None:
        """Draw audio emotion panel on right side."""
        panel_width = 200
        panel_height = 220
        panel_x = frame.shape[1] - panel_width - 10
        panel_y = 50

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + panel_width, panel_y + panel_height),
            (0, 0, 0),
            -1,
        )
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Title
        cv2.putText(
            frame,
            "AUDIO",
            (panel_x + 10, panel_y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 200, 0),
            2,
        )

        # Status indicator (listening / speech detected)
        if self._audio_state.speech_detected:
            status_color = (0, 255, 0)  # Green - speech detected
            status_text = "Speaking"
        elif self._audio_state.is_listening:
            status_color = (0, 255, 255)  # Yellow - listening
            status_text = "Listening"
        else:
            status_color = (0, 0, 255)  # Red - not active
            status_text = "Inactive"

        cv2.circle(frame, (panel_x + panel_width - 20, panel_y + 20), 8, status_color, -1)
        cv2.putText(
            frame,
            status_text,
            (panel_x + 70, panel_y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            status_color,
            1,
        )

        # Emotion bars
        self._draw_emotion_bars(
            frame,
            panel_x + 10,
            panel_y + 45,
            panel_width - 20,
            self._audio_state.emotions,
        )

    def _draw_emotion_bars(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        width: int,
        emotions: EmotionScores | None,
    ) -> None:
        """Draw emotion probability bars."""
        bar_height = 18
        label_width = 65
        max_bar_width = width - label_width - 10

        if emotions is None:
            # Show placeholder
            cv2.putText(
                frame,
                "No data",
                (x + 20, y + 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (100, 100, 100),
                1,
            )
            return

        emotion_dict = emotions.to_dict()
        y_offset = y

        for emotion, score in emotion_dict.items():
            color = EMOTION_COLORS.get(emotion, (255, 255, 255))
            bar_width = int(score * max_bar_width)

            # Emotion name
            cv2.putText(
                frame,
                emotion[:7],
                (x, y_offset + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (200, 200, 200),
                1,
            )

            # Bar background
            bar_x = x + label_width
            cv2.rectangle(
                frame,
                (bar_x, y_offset + 2),
                (bar_x + max_bar_width, y_offset + bar_height - 2),
                (50, 50, 50),
                -1,
            )

            # Bar fill
            if bar_width > 0:
                cv2.rectangle(
                    frame,
                    (bar_x, y_offset + 2),
                    (bar_x + bar_width, y_offset + bar_height - 2),
                    color,
                    -1,
                )

            # Score text
            cv2.putText(
                frame,
                f"{score:.0%}",
                (bar_x + max_bar_width + 5, y_offset + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (150, 150, 150),
                1,
            )

            y_offset += bar_height + 4


def run_multimodal_detection(camera_index: int = 0) -> None:
    """Run multimodal emotion detection with separate video/audio processing."""
    
    print("=" * 50)
    print("MULTIMODAL EMOTION DETECTOR")
    print("Video (Facial) and Audio (Speech) - SEPARATE")
    print("=" * 50)
    print("\nInitializing components...\n")

    # Initialize facial emotion components
    print("[Video] Initializing face detector...")
    face_config = ModelConfig(model_id="mtcnn", device="cpu")
    face_detector = FaceDetector(face_config, threshold=0.9)
    face_detector.load()

    print("[Video] Initializing facial emotion recognizer...")
    facial_config = ModelConfig(
        model_id="trpakov/vit-face-expression",
        device="cpu",
    )
    facial_emotion = FacialEmotionRecognizer(facial_config)
    facial_emotion.load()
    print("[Video] Models loaded!")

    # Initialize audio processor (runs in separate thread)
    audio_processor = AudioEmotionProcessor()
    audio_processor.initialize()

    # Initialize video input
    video_input = VideoInput(frame_skip=2)
    video_input.open(camera_index)

    # Initialize display
    display = MultimodalDisplay()
    display.start()

    # Start audio processing
    audio_processor.start(audio_device=None)  # Default microphone

    print("\n" + "=" * 50)
    print("Running! Press ESC or Q to quit")
    print("=" * 50 + "\n")

    facial_state = FacialEmotionState()
    frame_count = 0

    try:
        while True:
            # Read video frame
            frame_obj = video_input.read()
            if frame_obj is None:
                break

            frame = frame_obj.data
            frame_count += 1

            # Convert BGR to RGB for face detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Face detection
            faces = face_detector.predict(rgb_frame)

            if faces:
                # Get emotion for first face
                face = faces[0]
                result = facial_emotion.predict(face)

                facial_state = FacialEmotionState(
                    emotions=result.emotions,
                    dominant=result.emotions.dominant_emotion.value,
                    confidence=result.confidence,
                    face_detected=True,
                    timestamp=time.time(),
                )

                # Draw face box
                x1, y1, x2, y2 = face.bbox.to_xyxy()
                color = EMOTION_COLORS.get(facial_state.dominant, (255, 255, 255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            else:
                facial_state = FacialEmotionState(face_detected=False)

            # Get audio state
            audio_state = audio_processor.get_state()

            # Update display
            if not display.update(frame, facial_state, audio_state):
                break

            # Console output every 60 frames
            if frame_count % 60 == 0:
                print(f"[Status] Frame {frame_count} | "
                      f"Face: {facial_state.dominant} | "
                      f"Audio: {audio_state.dominant}")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        print("\nShutting down...")
        audio_processor.stop()
        video_input.close()
        display.stop()
        face_detector.unload()
        facial_emotion.unload()
        print("Done!")


def main() -> None:
    """Parse arguments and run multimodal detection."""
    parser = argparse.ArgumentParser(
        description="Real-time multimodal emotion detection (video + audio separate)"
    )
    parser.add_argument(
        "--camera", "-c",
        type=int,
        default=0,
        help="Camera device index (default: 0)",
    )

    args = parser.parse_args()

    # Handle Ctrl+C gracefully
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))

    run_multimodal_detection(camera_index=args.camera)


if __name__ == "__main__":
    main()

