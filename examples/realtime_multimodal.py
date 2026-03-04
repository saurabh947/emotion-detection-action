#!/usr/bin/env python3
"""Real-time multimodal emotion detection — Two-Tower Transformer.

Architecture
------------
The detector uses a Two-Tower Multimodal Emotion Recognition Transformer:

    VideoMAE backbone  ──┐
                          ├─► Cross-Attention Fusion ─► emotion probs (8)
    AST backbone       ──┘                           └► attention metrics (3)

Two display panels are shown:

- **EMOTION** (left): per-clip softmax probabilities for the 8 emotion classes
  (Angry, Disgusted, Fearful, Happy, Neutral, Sad, Surprised, Unclear).
- **ATTENTION** (right): continuous attention metrics from the neural
  attention head — Stress, Engagement, Arousal — each in [0, 1].

Usage
-----
::

    python realtime_multimodal.py
    python realtime_multimodal.py --camera 1
    python realtime_multimodal.py --no-pretrained   # fast offline test
    python realtime_multimodal.py --device mps      # Apple Silicon GPU
    python realtime_multimodal.py --model-path outputs/phase2_best.pt

Requirements
------------
- Webcam
- OpenCV for visualisation
- PyTorch + torchaudio
- HuggingFace ``transformers`` (for pretrained VideoMAE / AST backbones)
"""

from __future__ import annotations

import argparse
import signal
import sys
import time

import numpy as np

from emotion_detection_action import Config, EmotionDetector, InferenceWorker, NeuralEmotionResult


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
    "unclear":   (80, 80, 80),
}


# -------------------------------------------------------------------------
# Display
# -------------------------------------------------------------------------


class TwoTowerDisplay:
    """OpenCV display with EMOTION and ATTENTION panels."""

    PANEL_W = 220
    PANEL_H = 280
    BAR_H = 18

    def __init__(self, window_name: str = "Two-Tower Emotion Detector") -> None:
        self.window_name = window_name
        self._result: NeuralEmotionResult | None = None

    def start(self) -> None:
        try:
            import cv2
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 1000, 700)
        except ImportError:
            raise RuntimeError("opencv-python required: pip3 install opencv-python")

    def stop(self) -> None:
        try:
            import cv2
            cv2.destroyAllWindows()
        except ImportError:
            pass

    def update(self, frame: np.ndarray, result: NeuralEmotionResult | None) -> bool:
        """Render frame + side panels.  Returns False when the user closes the window."""
        import cv2

        self._result = result
        canvas = frame.copy()

        if result is not None:
            self._draw_headline(canvas, result)
            self._draw_emotion_panel(canvas, result)
            self._draw_attention_panel(canvas, result)
        else:
            cv2.putText(canvas, "Warming up…", (10, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (150, 150, 150), 2)

        cv2.imshow(self.window_name, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            return False
        return cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) >= 1

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_headline(self, frame: np.ndarray, result: NeuralEmotionResult) -> None:
        import cv2
        dominant = result.dominant_emotion
        color = EMOTION_COLORS.get(dominant, (255, 255, 255))
        label = f"{dominant.upper()}  ({result.confidence:.0%})"
        cv2.putText(frame, label, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    def _draw_emotion_panel(self, frame: np.ndarray, result: NeuralEmotionResult) -> None:
        """Left panel — 8 emotion probability bars."""
        import cv2
        px, py = 10, 50
        self._panel_bg(frame, px, py)
        cv2.putText(
            frame, "EMOTION", (px + 8, py + 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2,
        )
        label_w = 68
        bar_max = self.PANEL_W - label_w - 30
        y = py + 40
        for emotion, score in result.emotion_scores.items():
            color = EMOTION_COLORS.get(emotion, (255, 255, 255))
            bw = int(score * bar_max)
            cv2.putText(
                frame, emotion[:7], (px + 8, y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1,
            )
            bx = px + 8 + label_w
            cv2.rectangle(frame, (bx, y + 2), (bx + bar_max, y + self.BAR_H - 2), (50, 50, 50), -1)
            if bw > 0:
                cv2.rectangle(frame, (bx, y + 2), (bx + bw, y + self.BAR_H - 2), color, -1)
            cv2.putText(
                frame, f"{score:.0%}", (bx + bar_max + 4, y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1,
            )
            y += self.BAR_H + 4

    def _draw_attention_panel(self, frame: np.ndarray, result: NeuralEmotionResult) -> None:
        """Right panel — stress / engagement / arousal metrics."""
        import cv2
        frame_w = frame.shape[1]
        px = frame_w - self.PANEL_W - 10
        py = 50
        self._panel_bg(frame, px, py)
        cv2.putText(
            frame, "ATTENTION", (px + 8, py + 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2,
        )
        metrics = [
            ("Stress",  result.metrics.get("stress", 0.0),     (0, 0, 255)),
            ("Engage",  result.metrics.get("engagement", 0.0), (0, 255, 0)),
            ("Arousal", result.metrics.get("arousal", 0.0),    (255, 0, 255)),
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

    def _panel_bg(self, frame: np.ndarray, px: int, py: int) -> None:
        import cv2
        cv2.rectangle(
            frame,
            (px, py),
            (px + self.PANEL_W, py + self.PANEL_H),
            (0, 0, 0),
            -1,
        )


# -------------------------------------------------------------------------
# Main detection loop
# -------------------------------------------------------------------------


def run(
    camera_index: int = 0,
    device: str = "cpu",
    pretrained: bool = True,
    video_frames: int = 16,
    model_path: str | None = None,
    num_workers: int = 1,
    max_queue_size: int = 4,
) -> None:
    """Run the Two-Tower real-time emotion detection loop."""
    try:
        import cv2
    except ImportError:
        raise RuntimeError("opencv-python required: pip3 install opencv-python")

    print("=" * 55)
    print("TWO-TOWER MULTIMODAL EMOTION DETECTOR")
    print("=" * 55)
    print(f"  Device       : {device}")
    print(f"  Pretrained   : {pretrained}")
    print(f"  Video frames : {video_frames}")
    if model_path:
        print(f"  Checkpoint   : {model_path}")
    print("\nInitialising…  (may take a moment if downloading backbone weights)")

    config = Config(
        vla_enabled=False,
        two_tower_pretrained=pretrained,
        two_tower_video_frames=video_frames,
        two_tower_device=device,
        two_tower_model_path=model_path,
        verbose=False,
    )
    detector = EmotionDetector(config)
    detector.initialize()

    worker = InferenceWorker(
        detector,
        num_workers=num_workers,
        max_queue_size=max_queue_size,
    )

    display = TwoTowerDisplay()
    display.start()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        worker.stop()
        raise RuntimeError(f"Cannot open camera {camera_index}")

    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Camera       : {cam_w}×{cam_h}  index={camera_index}")
    print("\nRunning — press ESC or Q to quit.\n")

    frame_count = 0
    last_result: NeuralEmotionResult | None = None

    try:
        with worker:
            while True:
                ret, bgr_frame = cap.read()
                if not ret:
                    break

                worker.push_frame(bgr_frame, audio=None, timestamp=time.time())

                result = worker.latest_result
                if result is not None:
                    last_result = result

                alive = display.update(bgr_frame, last_result)
                if not alive:
                    break

                frame_count += 1
                if frame_count % 60 == 0 and last_result is not None:
                    r = last_result
                    print(
                        f"[{frame_count:5d}] {r.dominant_emotion:<10} "
                        f"({r.confidence:.0%}) | "
                        f"stress={r.metrics.get('stress', 0):.2f}  "
                        f"engage={r.metrics.get('engagement', 0):.2f}  "
                        f"arousal={r.metrics.get('arousal', 0):.2f}"
                    )

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cap.release()
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
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to a fine-tuned checkpoint (e.g. outputs/phase2_best.pt)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Inference worker threads")
    parser.add_argument("--queue-size", type=int, default=4,
                        help="Max frames queued before oldest is dropped")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))

    run(
        camera_index=args.camera,
        device=args.device,
        pretrained=not args.no_pretrained,
        video_frames=args.video_frames,
        model_path=args.model_path,
        num_workers=args.workers,
        max_queue_size=args.queue_size,
    )


if __name__ == "__main__":
    main()
