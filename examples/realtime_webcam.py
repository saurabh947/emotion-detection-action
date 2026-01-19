#!/usr/bin/env python3
"""Real-time webcam emotion detection example.

This example demonstrates how to use the EmotionDetector for
real-time emotion detection from a webcam feed with optional audio.

Supported emotions:
    - Facial: happy, sad, angry, fearful, surprised, disgusted, neutral (7)
    - Speech (with --audio): happy, sad, angry, neutral (4)

Requirements:
    - Webcam connected to the system
    - OpenCV for visualization
    - Microphone (optional, for --audio flag)

Usage:
    python realtime_webcam.py
    python realtime_webcam.py --camera 1  # Use different camera
    python realtime_webcam.py --audio     # Enable audio emotion detection
    python realtime_webcam.py --no-display  # Run without visualization
"""

import argparse
import asyncio
import signal
import sys
from typing import Any

import cv2
import numpy as np

from emotion_detector import Config, EmotionDetector
from emotion_detector.core.types import PipelineResult


class RealtimeEmotionDisplay:
    """Real-time emotion display with OpenCV visualization."""

    # Color scheme for emotions (BGR)
    EMOTION_COLORS = {
        "happy": (0, 255, 255),      # Yellow
        "sad": (255, 0, 0),          # Blue
        "angry": (0, 0, 255),        # Red
        "fearful": (128, 0, 128),    # Purple
        "surprised": (0, 255, 0),    # Green
        "disgusted": (0, 128, 0),    # Dark green
        "neutral": (128, 128, 128),  # Gray
    }

    def __init__(self, window_name: str = "Emotion Detector") -> None:
        """Initialize the display.

        Args:
            window_name: Name of the OpenCV window.
        """
        self.window_name = window_name
        self._running = True

    def start(self) -> None:
        """Start the display window."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 800, 600)

    def stop(self) -> None:
        """Stop the display and close window."""
        self._running = False
        cv2.destroyAllWindows()

    def update(self, frame: np.ndarray, result: PipelineResult | None) -> bool:
        """Update the display with new frame and result.

        Args:
            frame: Video frame to display.
            result: Emotion detection result.

        Returns:
            False if window was closed, True otherwise.
        """
        display_frame = frame.copy()

        if result:
            self._draw_detections(display_frame, result)
            self._draw_emotion_panel(display_frame, result)

        cv2.imshow(self.window_name, display_frame)

        # Check for key press (ESC or Q to quit)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            return False

        return cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) >= 1

    def _draw_detections(
        self,
        frame: np.ndarray,
        result: PipelineResult,
    ) -> None:
        """Draw face detections on frame.

        Args:
            frame: Frame to draw on.
            result: Pipeline result with detections.
        """
        emotion = result.emotion.dominant_emotion.value
        color = self.EMOTION_COLORS.get(emotion, (255, 255, 255))

        for face in result.detection.faces:
            x1, y1, x2, y2 = face.bbox.to_xyxy()

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw emotion label
            label = f"{emotion} ({result.emotion.fusion_confidence:.0%})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

            # Background for label
            cv2.rectangle(
                frame,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0] + 10, y1),
                color,
                -1,
            )

            # Label text
            cv2.putText(
                frame,
                label,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )

    def _draw_emotion_panel(
        self,
        frame: np.ndarray,
        result: PipelineResult,
    ) -> None:
        """Draw emotion scores panel.

        Args:
            frame: Frame to draw on.
            result: Pipeline result with emotions.
        """
        panel_width = 200
        panel_height = 180
        panel_x = frame.shape[1] - panel_width - 10
        panel_y = 10

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
            "Emotions",
            (panel_x + 10, panel_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        # Emotion bars
        y_offset = panel_y + 35
        bar_height = 15
        max_bar_width = panel_width - 80

        for emotion, score in result.emotion.emotions.to_dict().items():
            color = self.EMOTION_COLORS.get(emotion, (255, 255, 255))
            bar_width = int(score * max_bar_width)

            # Emotion name
            cv2.putText(
                frame,
                emotion[:8],
                (panel_x + 10, y_offset + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
            )

            # Bar background
            cv2.rectangle(
                frame,
                (panel_x + 70, y_offset),
                (panel_x + 70 + max_bar_width, y_offset + bar_height),
                (50, 50, 50),
                -1,
            )

            # Bar fill
            if bar_width > 0:
                cv2.rectangle(
                    frame,
                    (panel_x + 70, y_offset),
                    (panel_x + 70 + bar_width, y_offset + bar_height),
                    color,
                    -1,
                )

            y_offset += bar_height + 5


async def run_realtime_detection(
    camera_index: int = 0,
    show_display: bool = True,
    use_audio: bool = False,
) -> None:
    """Run real-time emotion detection.

    Args:
        camera_index: Camera device index.
        show_display: Whether to show visualization window.
        use_audio: Whether to enable audio emotion detection.
    """
    print("Starting real-time emotion detection...")
    print("Press ESC or Q to quit\n")

    config = Config(
        device="cpu",  # Use CPU for compatibility
        vla_enabled=False,
        frame_skip=2,  # Process every other frame
        verbose=False,
    )

    display = None
    if show_display:
        display = RealtimeEmotionDisplay()
        display.start()

    # Track statistics
    frame_count = 0
    emotion_counts: dict[str, int] = {}

    try:
        with EmotionDetector(config) as detector:
            async for result in detector.stream(
                video_source=camera_index,
                audio_source=0 if use_audio else None,
            ):
                frame_count += 1

                # Track emotions
                emotion = result.emotion.dominant_emotion.value
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

                # Update display
                if display and result.detection.frame is not None:
                    if not display.update(result.detection.frame, result):
                        break

                # Print status every 30 frames
                if frame_count % 30 == 0:
                    print(
                        f"Frame {frame_count}: {emotion} "
                        f"({result.emotion.fusion_confidence:.0%})"
                    )

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        if display:
            display.stop()

        # Print summary
        print("\n" + "=" * 40)
        print(f"Processed {frame_count} frames")
        print("\nEmotion Distribution:")
        for emotion, count in sorted(emotion_counts.items(), key=lambda x: -x[1]):
            pct = count / frame_count * 100 if frame_count > 0 else 0
            print(f"  {emotion:12s}: {pct:5.1f}% ({count} frames)")


def main() -> None:
    """Parse arguments and run real-time detection."""
    parser = argparse.ArgumentParser(
        description="Real-time webcam emotion detection"
    )
    parser.add_argument(
        "--camera", "-c",
        type=int,
        default=0,
        help="Camera device index (default: 0)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run without visualization window",
    )
    parser.add_argument(
        "--audio",
        action="store_true",
        help="Enable audio emotion detection",
    )

    args = parser.parse_args()

    # Handle Ctrl+C gracefully
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))

    asyncio.run(run_realtime_detection(
        camera_index=args.camera,
        show_display=not args.no_display,
        use_audio=args.audio,
    ))


if __name__ == "__main__":
    main()

