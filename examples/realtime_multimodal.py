#!/usr/bin/env python3
"""Real-time multimodal emotion detection example.

This example uses the high-level EmotionDetector API while showing
separate panels for facial, audio, attention, and fused emotion results.

Four panels show:
    - FACIAL: Emotion from face (top left panel)
    - AUDIO: Emotion from speech (top right panel)
    - ATTENTION: Gaze, pupil, and attention metrics (bottom left panel)
    - FUSED: Combined emotion using configured fusion strategy (bottom center panel)

Supported emotions:
    - Facial: happy, sad, angry, fearful, surprised, disgusted, neutral (7)
    - Speech: happy, sad, angry, neutral (4) - using SUPERB model
    - Fused: All 7 emotions (confidence-weighted combination, influenced by attention)

Attention analysis metrics:
    - Stress: Based on pupil dilation and blink rate
    - Engagement: Based on eye contact and fixation stability
    - Nervousness: Based on gaze aversion and instability

Face detection models:
    - mtcnn: Fast, good for real-time (default)
    - retinaface: More accurate, better for challenging poses (requires: pip install retinaface)

Temporal smoothing strategies:
    - none: No smoothing (raw per-frame output)
    - rolling: Rolling average over N frames
    - ema: Exponential Moving Average (default, recommended)
    - hysteresis: Requires sustained change before switching

Requirements:
    - Webcam connected to the system
    - Microphone connected to the system
    - OpenCV for visualization
    - MediaPipe (for attention analysis)

Usage:
    python realtime_multimodal.py
    python realtime_multimodal.py --camera 1  # Use different camera
    python realtime_multimodal.py --face-detection retinaface  # Use RetinaFace
    python realtime_multimodal.py --smoothing ema --smoothing-alpha 0.2  # Smoother output
    python realtime_multimodal.py --no-attention  # Disable attention analysis
"""

import argparse
import asyncio
import signal
import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np

from emotion_detection_action import Config, EmotionDetector
from emotion_detection_action.core.types import (
    EmotionScores,
    PipelineResult,
)


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


@dataclass
class FusedEmotionState:
    """Current state of fused (combined) emotion detection."""
    emotions: EmotionScores | None = None
    dominant: str = "none"
    confidence: float = 0.0
    facial_used: bool = False
    audio_used: bool = False
    attention_used: bool = False
    timestamp: float = 0.0


@dataclass
class AttentionState:
    """Current state of attention analysis."""
    stress_score: float = 0.0
    engagement_score: float = 0.0
    nervousness_score: float = 0.0
    pupil_dilation: float = 0.0
    blink_rate: float = 0.0
    gaze_direction: tuple[float, float] = (0.0, 0.0)
    eye_detected: bool = False
    confidence: float = 0.0
    timestamp: float = 0.0


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


def extract_states(result: PipelineResult) -> tuple[FacialEmotionState, AudioEmotionState, AttentionState, FusedEmotionState]:
    """Extract facial, audio, attention, and fused states from a PipelineResult.
    
    Args:
        result: Pipeline result from EmotionDetector.
        
    Returns:
        Tuple of (facial_state, audio_state, attention_state, fused_state).
    """
    now = time.time()
    
    # Extract facial state
    facial_state = FacialEmotionState(timestamp=now)
    if result.emotion.facial_result is not None:
        fr = result.emotion.facial_result
        facial_state = FacialEmotionState(
            emotions=fr.emotions,
            dominant=fr.emotions.dominant_emotion.value,
            confidence=fr.confidence,
            face_detected=True,
            timestamp=now,
        )
    
    # Extract audio state
    audio_state = AudioEmotionState(timestamp=now)
    if result.emotion.speech_result is not None:
        sr = result.emotion.speech_result
        audio_state = AudioEmotionState(
            emotions=sr.emotions,
            dominant=sr.emotions.dominant_emotion.value,
            confidence=sr.confidence,
            speech_detected=True,
            timestamp=now,
        )
    
    # Extract attention state
    attention_state = AttentionState(timestamp=now)
    if result.emotion.attention_result is not None:
        ar = result.emotion.attention_result
        attention_state = AttentionState(
            stress_score=ar.metrics.stress_score,
            engagement_score=ar.metrics.engagement_score,
            nervousness_score=ar.metrics.nervousness_score,
            pupil_dilation=ar.metrics.pupil_dilation,
            blink_rate=ar.metrics.blink_rate,
            gaze_direction=ar.gaze.gaze_direction if ar.gaze else (0.0, 0.0),
            eye_detected=ar.gaze is not None,
            confidence=ar.confidence,
            timestamp=now,
        )
    
    # Extract fused state
    fused_state = FusedEmotionState(
        emotions=result.emotion.emotions,
        dominant=result.emotion.dominant_emotion.value,
        confidence=result.emotion.fusion_confidence,
        facial_used=result.emotion.facial_result is not None,
        audio_used=result.emotion.speech_result is not None,
        attention_used=result.emotion.attention_result is not None,
        timestamp=now,
    )
    
    return facial_state, audio_state, attention_state, fused_state


class MultimodalDisplay:
    """Display for showing facial, audio, attention, and fused emotions."""

    def __init__(self, window_name: str = "Multimodal Emotion Detector"):
        self.window_name = window_name
        self._facial_state = FacialEmotionState()
        self._audio_state = AudioEmotionState()
        self._attention_state = AttentionState()
        self._fused_state = FusedEmotionState()

    def start(self) -> None:
        """Start the display window."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1000, 750)

    def stop(self) -> None:
        """Close the display window."""
        cv2.destroyAllWindows()

    def update(
        self,
        frame: np.ndarray,
        facial_state: FacialEmotionState,
        audio_state: AudioEmotionState,
        attention_state: AttentionState,
        fused_state: FusedEmotionState,
        faces: list | None = None,
    ) -> bool:
        """Update display with current states.

        Args:
            frame: Video frame to display.
            facial_state: Current facial emotion state.
            audio_state: Current audio emotion state.
            attention_state: Current attention analysis state.
            fused_state: Current fused emotion state.
            faces: List of detected faces (for drawing bounding boxes).

        Returns:
            False if window closed, True otherwise.
        """
        self._facial_state = facial_state
        self._audio_state = audio_state
        self._attention_state = attention_state
        self._fused_state = fused_state

        display = frame.copy()

        # Draw face bounding boxes
        if faces:
            for face in faces:
                x1, y1, x2, y2 = face.bbox.to_xyxy()
                color = EMOTION_COLORS.get(facial_state.dominant, (255, 255, 255))
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

        # Draw gaze direction indicator if attention is being tracked
        if attention_state.eye_detected:
            self._draw_gaze_indicator(display, attention_state)

        # Draw face detection status and label
        self._draw_face_overlay(display)

        # Draw facial emotion panel (top left)
        self._draw_facial_panel(display)

        # Draw audio emotion panel (top right)
        self._draw_audio_panel(display)

        # Draw attention panel (bottom left)
        self._draw_attention_panel(display)

        # Draw fused emotion panel (bottom center)
        self._draw_fused_panel(display)

        cv2.imshow(self.window_name, display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            return False

        return cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) >= 1

    def _draw_gaze_indicator(self, frame: np.ndarray, attention: AttentionState) -> None:
        """Draw a gaze direction indicator on the frame."""
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2

        # Gaze direction (-1 to 1) maps to screen
        gx, gy = attention.gaze_direction
        target_x = int(center_x + gx * (w // 4))
        target_y = int(center_y + gy * (h // 4))

        # Draw gaze target
        color = (0, 255, 255) if attention.engagement_score > 0.5 else (0, 165, 255)
        cv2.circle(frame, (target_x, target_y), 10, color, 2)
        cv2.line(frame, (center_x, center_y - 100), (target_x, target_y), color, 1)

    def _draw_face_overlay(self, frame: np.ndarray) -> None:
        """Draw face detection and emotion on the video."""
        if not self._facial_state.face_detected:
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

        # Draw solid black background (not transparent)
        cv2.rectangle(
            frame,
            (panel_x, panel_y),
            (panel_x + panel_width, panel_y + panel_height),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            frame,
            "FACIAL",
            (panel_x + 10, panel_y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 255),
            2,
        )

        status_color = (0, 255, 0) if self._facial_state.face_detected else (0, 0, 255)
        cv2.circle(frame, (panel_x + panel_width - 20, panel_y + 20), 8, status_color, -1)

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

        # Draw solid black background (not transparent)
        cv2.rectangle(
            frame,
            (panel_x, panel_y),
            (panel_x + panel_width, panel_y + panel_height),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            frame,
            "AUDIO",
            (panel_x + 10, panel_y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 200, 0),
            2,
        )

        if self._audio_state.speech_detected:
            status_color = (0, 255, 0)
            status_text = "Speaking"
        else:
            status_color = (0, 255, 255)
            status_text = "Listening"

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

        self._draw_emotion_bars(
            frame,
            panel_x + 10,
            panel_y + 45,
            panel_width - 20,
            self._audio_state.emotions,
        )

    def _draw_attention_panel(self, frame: np.ndarray) -> None:
        """Draw attention analysis panel on bottom left."""
        panel_width = 200
        panel_height = 180
        panel_x = 10
        panel_y = frame.shape[0] - panel_height - 10

        # Draw solid black background
        cv2.rectangle(
            frame,
            (panel_x, panel_y),
            (panel_x + panel_width, panel_y + panel_height),
            (0, 0, 0),
            -1,
        )

        # Title
        cv2.putText(
            frame,
            "ATTENTION",
            (panel_x + 10, panel_y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 128, 0),  # Orange
            2,
        )

        # Status indicator
        status_color = (0, 255, 0) if self._attention_state.eye_detected else (0, 0, 255)
        cv2.circle(frame, (panel_x + panel_width - 20, panel_y + 20), 8, status_color, -1)

        if not self._attention_state.eye_detected:
            cv2.putText(
                frame,
                "Eyes not detected",
                (panel_x + 20, panel_y + 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (100, 100, 100),
                1,
            )
            return

        # Draw attention metrics as bars
        metrics = [
            ("Stress", self._attention_state.stress_score, (0, 0, 255)),      # Red
            ("Engage", self._attention_state.engagement_score, (0, 255, 0)),  # Green
            ("Nervous", self._attention_state.nervousness_score, (255, 0, 255)),  # Magenta
        ]

        y_offset = panel_y + 45
        bar_height = 18
        label_width = 55
        max_bar_width = panel_width - label_width - 30

        for label, value, color in metrics:
            # Label
            cv2.putText(
                frame,
                label,
                (panel_x + 10, y_offset + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (200, 200, 200),
                1,
            )

            # Background bar
            bar_x = panel_x + label_width
            cv2.rectangle(
                frame,
                (bar_x, y_offset + 2),
                (bar_x + max_bar_width, y_offset + bar_height - 2),
                (50, 50, 50),
                -1,
            )

            # Value bar
            bar_width = int(value * max_bar_width)
            if bar_width > 0:
                cv2.rectangle(
                    frame,
                    (bar_x, y_offset + 2),
                    (bar_x + bar_width, y_offset + bar_height - 2),
                    color,
                    -1,
                )

            # Percentage
            cv2.putText(
                frame,
                f"{value:.0%}",
                (bar_x + max_bar_width + 5, y_offset + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (150, 150, 150),
                1,
            )

            y_offset += bar_height + 6

        # Additional metrics
        y_offset += 5
        cv2.putText(
            frame,
            f"Blink rate: {self._attention_state.blink_rate:.1f}/min",
            (panel_x + 10, y_offset + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (150, 150, 150),
            1,
        )

        y_offset += 18
        gx, gy = self._attention_state.gaze_direction
        gaze_str = "Center" if abs(gx) < 0.3 and abs(gy) < 0.3 else f"({gx:.1f}, {gy:.1f})"
        cv2.putText(
            frame,
            f"Gaze: {gaze_str}",
            (panel_x + 10, y_offset + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (150, 150, 150),
            1,
        )

    def _draw_fused_panel(self, frame: np.ndarray) -> None:
        """Draw fused emotion panel at bottom center."""
        panel_width = 280
        panel_height = 240
        panel_x = (frame.shape[1] - panel_width) // 2
        panel_y = frame.shape[0] - panel_height - 10

        # Draw solid black background (not transparent)
        cv2.rectangle(
            frame,
            (panel_x, panel_y),
            (panel_x + panel_width, panel_y + panel_height),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            frame,
            "FUSED",
            (panel_x + 10, panel_y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 128),
            2,
        )

        sources = []
        if self._fused_state.facial_used:
            sources.append("F")
        if self._fused_state.audio_used:
            sources.append("A")
        if self._fused_state.attention_used:
            sources.append("Att")
        source_text = "+".join(sources) if sources else "No data"

        cv2.putText(
            frame,
            f"({source_text})",
            (panel_x + 80, panel_y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (150, 150, 150),
            1,
        )

        conf_color = (
            (0, 255, 0) if self._fused_state.confidence > 0.5
            else (0, 255, 255) if self._fused_state.confidence > 0.3
            else (0, 0, 255)
        )
        cv2.circle(frame, (panel_x + panel_width - 20, panel_y + 20), 8, conf_color, -1)
        cv2.putText(
            frame,
            f"{self._fused_state.confidence:.0%}",
            (panel_x + panel_width - 55, panel_y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            conf_color,
            1,
        )

        if self._fused_state.dominant != "none" and self._fused_state.emotions is not None:
            emotion_color = EMOTION_COLORS.get(self._fused_state.dominant, (255, 255, 255))
            cv2.putText(
                frame,
                f">> {self._fused_state.dominant.upper()} <<",
                (panel_x + 70, panel_y + 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                emotion_color,
                2,
            )

        self._draw_emotion_bars(
            frame,
            panel_x + 10,
            panel_y + 60,
            panel_width - 20,
            self._fused_state.emotions,
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

            cv2.putText(
                frame,
                emotion[:7],
                (x, y_offset + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (200, 200, 200),
                1,
            )

            bar_x = x + label_width
            cv2.rectangle(
                frame,
                (bar_x, y_offset + 2),
                (bar_x + max_bar_width, y_offset + bar_height - 2),
                (50, 50, 50),
                -1,
            )

            if bar_width > 0:
                cv2.rectangle(
                    frame,
                    (bar_x, y_offset + 2),
                    (bar_x + bar_width, y_offset + bar_height - 2),
                    color,
                    -1,
                )

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


async def run_multimodal_detection(
    camera_index: int = 0,
    face_detection_model: str = "mtcnn",
    facial_model: str = "trpakov/vit-face-expression",
    speech_model: str = "superb/wav2vec2-base-superb-er",
    device: str = "cpu",
    smoothing: str = "ema",
    smoothing_alpha: float = 0.3,
    attention_enabled: bool = True,
) -> None:
    """Run multimodal emotion detection using high-level EmotionDetector API."""

    print("=" * 50)
    print("MULTIMODAL EMOTION DETECTOR")
    print("Using EmotionDetector high-level API")
    print("Video (Facial) + Audio (Speech) + Attention + FUSION")
    print("=" * 50)
    print(f"\nFace detection: {face_detection_model}")
    print(f"Facial model: {facial_model}")
    print(f"Speech model: {speech_model}")
    print(f"Attention analysis: {'enabled' if attention_enabled else 'disabled'}")
    print(f"Smoothing: {smoothing}" + (f" (alpha={smoothing_alpha})" if smoothing == "ema" else ""))
    print(f"Device: {device}")
    print("\nInitializing detector...\n")

    # Configure detector with configurable models
    config = Config(
        device=device,
        vla_enabled=False,
        face_detection_model=face_detection_model,
        facial_emotion_model=facial_model,
        speech_emotion_model=speech_model,
        fusion_strategy="confidence",
        fusion_confidence_threshold=0.3,
        smoothing_strategy=smoothing,
        smoothing_ema_alpha=smoothing_alpha,
        attention_analysis_enabled=attention_enabled,
        frame_skip=2,
        verbose=False,
    )

    display = MultimodalDisplay()
    display.start()

    # Track last states for when no result is returned
    last_facial = FacialEmotionState()
    last_audio = AudioEmotionState()
    last_attention = AttentionState()
    last_fused = FusedEmotionState()
    frame_count = 0

    try:
        with EmotionDetector(config) as detector:
            print("\n" + "=" * 50)
            print("Running! Press ESC or Q to quit")
            print("=" * 50 + "\n")

            # Stream with audio always enabled (microphone=0 for default mic)
            async for result in detector.stream(camera=camera_index, microphone=0):
                frame_count += 1

                # Extract states from result
                facial_state, audio_state, attention_state, fused_state = extract_states(result)

                # Update last known states
                if facial_state.face_detected:
                    last_facial = facial_state
                if audio_state.speech_detected:
                    last_audio = audio_state
                if attention_state.eye_detected:
                    last_attention = attention_state
                last_fused = fused_state

                # Get frame and faces for display
                frame = result.detection.frame
                if frame is None:
                    continue

                faces = result.detection.faces

                # Update display
                if not display.update(frame, last_facial, last_audio, last_attention, last_fused, faces):
                    break

                # Console output every 60 frames
                if frame_count % 60 == 0:
                    attn_str = ""
                    if attention_enabled and last_attention.eye_detected:
                        attn_str = f" | Stress: {last_attention.stress_score:.0%} | Engage: {last_attention.engagement_score:.0%}"
                    print(
                        f"[Status] Frame {frame_count} | "
                        f"Face: {last_facial.dominant} ({last_facial.confidence:.0%}) | "
                        f"Audio: {last_audio.dominant} ({last_audio.confidence:.0%}) | "
                        f"Fused: {last_fused.dominant} ({last_fused.confidence:.0%})"
                        f"{attn_str}"
                    )

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        print("\nShutting down...")
        display.stop()
        print("Done!")


def main() -> None:
    """Parse arguments and run multimodal detection."""
    parser = argparse.ArgumentParser(
        description="Real-time multimodal emotion detection with 4 panels (facial + audio + attention + fused)"
    )
    parser.add_argument(
        "--camera",
        "-c",
        type=int,
        default=0,
        help="Camera device index (default: 0)",
    )
    parser.add_argument(
        "--face-detection",
        type=str,
        default="mtcnn",
        choices=["mtcnn", "retinaface"],
        help="Face detection model: mtcnn (fast) or retinaface (accurate)",
    )
    parser.add_argument(
        "--facial-model",
        type=str,
        default="trpakov/vit-face-expression",
        help="HuggingFace model ID for facial emotion recognition",
    )
    parser.add_argument(
        "--speech-model",
        type=str,
        default="superb/wav2vec2-base-superb-er",
        help="HuggingFace model ID for speech emotion recognition",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use: cpu, cuda, or mps (default: cpu)",
    )
    parser.add_argument(
        "--smoothing",
        type=str,
        default="ema",
        choices=["none", "rolling", "ema", "hysteresis"],
        help="Temporal smoothing strategy (default: ema)",
    )
    parser.add_argument(
        "--smoothing-alpha",
        type=float,
        default=0.3,
        help="EMA smoothing factor 0-1, lower=smoother (default: 0.3)",
    )
    parser.add_argument(
        "--no-attention",
        action="store_true",
        help="Disable attention analysis (gaze, pupil, stress detection)",
    )

    args = parser.parse_args()

    # Handle Ctrl+C gracefully
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))

    asyncio.run(run_multimodal_detection(
        camera_index=args.camera,
        face_detection_model=args.face_detection,
        facial_model=args.facial_model,
        speech_model=args.speech_model,
        device=args.device,
        smoothing=args.smoothing,
        smoothing_alpha=args.smoothing_alpha,
        attention_enabled=not args.no_attention,
    ))


if __name__ == "__main__":
    main()
