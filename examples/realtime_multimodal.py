#!/usr/bin/env python3
"""Real-time multimodal emotion detection — Two-Tower Transformer.

Architecture
------------
The detector uses a Two-Tower Multimodal Emotion Recognition Transformer:

    VideoMAE backbone  ──┐
                          ├─► Cross-Attention Fusion ─► emotion probs (7)
    AST backbone       ──┘                           └► attention metrics (3)

Two display panels are shown:

- **EMOTION** (left): per-clip softmax probabilities for the 7 standard
  emotions (Angry, Disgusted, Fearful, Happy, Neutral, Sad, Surprised).
- **ATTENTION** (right): continuous attention metrics from the neural
  attention head — Stress, Engagement, Nervousness — each in [0, 1].

MediaPipe gaze tracking provides the gaze-direction overlay arrow; the
attention *scores* come from the Two-Tower attention head (not a
rule-based algorithm).

Usage
-----
::

    python realtime_multimodal.py
    python realtime_multimodal.py --camera 1
    python realtime_multimodal.py --no-pretrained   # fast offline test
    python realtime_multimodal.py --no-attention    # disable gaze overlay
    python realtime_multimodal.py --smoothing ema --smoothing-alpha 0.2

Requirements
------------
- Webcam + microphone
- OpenCV for visualisation
- PyTorch + torchaudio
- MediaPipe (for face detection / gaze overlay)
- HuggingFace ``transformers`` (for pretrained VideoMAE / AST backbones)
"""

import argparse
import asyncio
import signal
import sys
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from emotion_detection_action import Config, EmotionDetector
from emotion_detection_action.core.types import EmotionScores, PipelineResult


# -------------------------------------------------------------------------
# State dataclasses
# -------------------------------------------------------------------------


@dataclass
class EmotionState:
    """Current Two-Tower emotion output."""

    emotions: EmotionScores | None = None
    dominant: str = "none"
    confidence: float = 0.0
    video_missing: bool = False
    audio_missing: bool = False
    timestamp: float = 0.0


@dataclass
class AttentionState:
    """Current Two-Tower attention output + MediaPipe gaze info."""

    stress_score: float = 0.0
    engagement_score: float = 0.0
    nervousness_score: float = 0.0
    gaze_direction: tuple[float, float] = (0.0, 0.0)
    eye_detected: bool = False
    confidence: float = 0.0
    timestamp: float = 0.0


# -------------------------------------------------------------------------
# Colour palette (BGR)
# -------------------------------------------------------------------------

EMOTION_COLORS: dict[str, tuple[int, int, int]] = {
    "happy":     (0, 255, 255),
    "sad":       (255, 0, 0),
    "angry":     (0, 0, 255),
    "fearful":   (128, 0, 128),
    "surprised": (0, 255, 0),
    "disgusted": (0, 128, 0),
    "neutral":   (128, 128, 128),
    "none":      (100, 100, 100),
}


# -------------------------------------------------------------------------
# State extraction
# -------------------------------------------------------------------------


def extract_states(result: PipelineResult) -> tuple[EmotionState, AttentionState]:
    """Extract emotion and attention states from a pipeline result."""
    now = time.time()

    emotion_state = EmotionState(
        emotions=result.emotion.emotions,
        dominant=result.emotion.dominant_emotion.value,
        confidence=result.emotion.fusion_confidence,
        timestamp=now,
    )

    attention_state = AttentionState(timestamp=now)
    if result.emotion.attention_result is not None:
        ar = result.emotion.attention_result
        gaze = ar.gaze
        attention_state = AttentionState(
            stress_score=ar.metrics.stress_score,
            engagement_score=ar.metrics.engagement_score,
            nervousness_score=ar.metrics.nervousness_score,
            gaze_direction=gaze.gaze_direction if gaze else (0.0, 0.0),
            eye_detected=gaze is not None,
            confidence=ar.confidence,
            timestamp=now,
        )

    return emotion_state, attention_state


# -------------------------------------------------------------------------
# Display
# -------------------------------------------------------------------------


class TwoTowerDisplay:
    """OpenCV display with EMOTION and ATTENTION panels."""

    PANEL_W = 220
    PANEL_H = 260
    BAR_H = 18

    def __init__(self, window_name: str = "Two-Tower Emotion Detector") -> None:
        self.window_name = window_name
        self._emotion = EmotionState()
        self._attention = AttentionState()

    def start(self) -> None:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1000, 700)

    def stop(self) -> None:
        cv2.destroyAllWindows()

    def update(
        self,
        frame: np.ndarray,
        emotion: EmotionState,
        attention: AttentionState,
        faces: list | None = None,
    ) -> bool:
        """Render frame + panels.  Returns False when the user closes the window."""
        self._emotion = emotion
        self._attention = attention

        canvas = frame.copy()

        # Face bounding boxes
        if faces:
            for face in faces:
                x1, y1, x2, y2 = face.bbox.to_xyxy()
                color = EMOTION_COLORS.get(emotion.dominant, (255, 255, 255))
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        # Gaze overlay (anchored to face)
        if attention.eye_detected and faces:
            self._draw_gaze(canvas, attention, faces[0].bbox.to_xyxy())

        # Dominant emotion label at top of frame
        self._draw_headline(canvas)

        # Side panels
        self._draw_emotion_panel(canvas)
        self._draw_attention_panel(canvas)

        cv2.imshow(self.window_name, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            return False
        return cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) >= 1

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_gaze(
        self,
        frame: np.ndarray,
        att: AttentionState,
        face_bbox: tuple[int, int, int, int],
    ) -> None:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = face_bbox
        ox = (x1 + x2) // 2
        oy = y1 + (y2 - y1) // 4          # ~eye level
        gx, gy = att.gaze_direction
        tx = int(ox + gx * w // 4)
        ty = int(oy + gy * h // 4)
        color = (0, 255, 255) if att.engagement_score > 0.5 else (0, 165, 255)
        cv2.circle(frame, (tx, ty), 10, color, 2)
        cv2.line(frame, (ox, oy), (tx, ty), color, 1)

    def _draw_headline(self, frame: np.ndarray) -> None:
        dominant = self._emotion.dominant
        color = EMOTION_COLORS.get(dominant, (255, 255, 255))
        label = f"{dominant.upper()}  ({self._emotion.confidence:.0%})"
        cv2.putText(frame, label, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    def _draw_emotion_panel(self, frame: np.ndarray) -> None:
        """Left panel — 7 emotion probability bars."""
        px, py = 10, 50
        self._panel_bg(frame, px, py)
        cv2.putText(
            frame, "EMOTION", (px + 8, py + 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2,
        )
        self._draw_bars(
            frame, px + 8, py + 38, self.PANEL_W - 16, self._emotion.emotions
        )

    def _draw_attention_panel(self, frame: np.ndarray) -> None:
        """Right panel — attention metrics from Two-Tower head."""
        frame_w = frame.shape[1]
        px = frame_w - self.PANEL_W - 10
        py = 50
        self._panel_bg(frame, px, py)
        cv2.putText(
            frame, "ATTENTION", (px + 8, py + 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2,
        )

        status_color = (0, 255, 0) if self._attention.eye_detected else (0, 0, 255)
        cv2.circle(frame, (px + self.PANEL_W - 18, py + 18), 8, status_color, -1)

        metrics = [
            ("Stress",   self._attention.stress_score,      (0, 0, 255)),
            ("Engage",   self._attention.engagement_score,  (0, 255, 0)),
            ("Nervous",  self._attention.nervousness_score, (255, 0, 255)),
        ]
        label_w = 60
        bar_max = self.PANEL_W - label_w - 30
        y = py + 40
        for label, value, color in metrics:
            cv2.putText(
                frame, label, (px + 8, y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1,
            )
            bx = px + label_w
            cv2.rectangle(frame, (bx, y + 2), (bx + bar_max, y + self.BAR_H - 2), (50, 50, 50), -1)
            bw = int(value * bar_max)
            if bw > 0:
                cv2.rectangle(frame, (bx, y + 2), (bx + bw, y + self.BAR_H - 2), color, -1)
            cv2.putText(
                frame, f"{value:.0%}", (bx + bar_max + 4, y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1,
            )
            y += self.BAR_H + 6

        # Gaze direction text
        if self._attention.eye_detected:
            gx, gy = self._attention.gaze_direction
            gaze_str = "Center" if abs(gx) < 0.3 and abs(gy) < 0.3 else f"({gx:.1f},{gy:.1f})"
            cv2.putText(
                frame, f"Gaze: {gaze_str}", (px + 8, y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1,
            )

    def _panel_bg(self, frame: np.ndarray, px: int, py: int) -> None:
        cv2.rectangle(
            frame,
            (px, py),
            (px + self.PANEL_W, py + self.PANEL_H),
            (0, 0, 0),
            -1,
        )

    def _draw_bars(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        width: int,
        emotions: EmotionScores | None,
    ) -> None:
        label_w = 68
        bar_max = width - label_w - 10
        if emotions is None:
            cv2.putText(frame, "No data", (x + 20, y + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            return
        for emotion, score in emotions.to_dict().items():
            color = EMOTION_COLORS.get(emotion, (255, 255, 255))
            bw = int(score * bar_max)
            cv2.putText(
                frame, emotion[:7], (x, y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1,
            )
            bx = x + label_w
            cv2.rectangle(frame, (bx, y + 2), (bx + bar_max, y + self.BAR_H - 2), (50, 50, 50), -1)
            if bw > 0:
                cv2.rectangle(frame, (bx, y + 2), (bx + bw, y + self.BAR_H - 2), color, -1)
            cv2.putText(
                frame, f"{score:.0%}", (bx + bar_max + 4, y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1,
            )
            y += self.BAR_H + 4


# -------------------------------------------------------------------------
# Main detection loop
# -------------------------------------------------------------------------


async def run(
    camera_index: int = 0,
    device: str = "cpu",
    pretrained: bool = True,
    smoothing: str = "none",
    smoothing_alpha: float = 0.3,
    attention_enabled: bool = True,
    video_frames: int = 16,
) -> None:
    """Run the Two-Tower real-time emotion detection loop."""
    print("=" * 55)
    print("TWO-TOWER MULTIMODAL EMOTION DETECTOR")
    print("=" * 55)
    print(f"  Device          : {device}")
    print(f"  Pretrained      : {pretrained}")
    print(f"  Video frames    : {video_frames}")
    print(f"  Smoothing       : {smoothing}")
    print(f"  Attention overlay: {'on' if attention_enabled else 'off'}")
    print("\nInitialising — this may take a minute (downloading backbone weights)...")

    config = Config(
        device=device,
        vla_enabled=False,
        two_tower_pretrained=pretrained,
        two_tower_video_frames=video_frames,
        two_tower_device=device,
        smoothing_strategy=smoothing,
        smoothing_ema_alpha=smoothing_alpha,
        attention_analysis_enabled=attention_enabled,
        frame_skip=2,
        verbose=False,
    )

    display = TwoTowerDisplay()
    display.start()

    last_emotion = EmotionState()
    last_attention = AttentionState()
    frame_count = 0

    try:
        with EmotionDetector(config, action_handler=None) as detector:
            print("\nRunning — press ESC or Q to quit.\n")
            async for result in detector.stream(camera=camera_index, microphone=0):
                frame_count += 1
                emotion, attention = extract_states(result)
                last_emotion = emotion
                if attention.eye_detected:
                    last_attention = attention

                frame = result.detection.frame
                if frame is None:
                    continue

                alive = display.update(
                    frame,
                    last_emotion,
                    last_attention,
                    result.detection.faces,
                )
                if not alive:
                    break

                if frame_count % 60 == 0:
                    print(
                        f"[{frame_count:5d}] {last_emotion.dominant:<10} "
                        f"({last_emotion.confidence:.0%}) | "
                        f"stress={last_attention.stress_score:.2f}  "
                        f"engage={last_attention.engagement_score:.2f}  "
                        f"nervous={last_attention.nervousness_score:.2f}"
                    )

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        display.stop()
        print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real-time Two-Tower multimodal emotion detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--camera", "-c", type=int, default=0,
                        help="Camera device index")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device: cpu | cuda | mps")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Use stub (random) backbones — no download, for testing only")
    parser.add_argument("--video-frames", type=int, default=16,
                        help="Clip length fed to the video backbone")
    parser.add_argument("--smoothing", type=str, default="none",
                        choices=["none", "rolling", "ema", "hysteresis"],
                        help="Temporal smoothing strategy")
    parser.add_argument("--smoothing-alpha", type=float, default=0.3,
                        help="EMA alpha (lower = smoother)")
    parser.add_argument("--no-attention", action="store_true",
                        help="Disable MediaPipe gaze overlay")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))

    asyncio.run(run(
        camera_index=args.camera,
        device=args.device,
        pretrained=not args.no_pretrained,
        video_frames=args.video_frames,
        smoothing=args.smoothing,
        smoothing_alpha=args.smoothing_alpha,
        attention_enabled=not args.no_attention,
    ))


if __name__ == "__main__":
    main()
