#!/usr/bin/env python3
"""Neural Emotion Detector — real-time streaming demo.

Demonstrates two streaming modes:

1. **Webcam mode** (``--webcam``) — reads from a real camera and microphone.
   The :class:`~emotion_detection_action.InferenceWorker` runs inference on a
   background thread so the capture loop never stalls.  The latest result is
   overlaid on the live video window in the top-left corner.

2. **Simulation mode** (default) — generates synthetic numpy arrays at a
   controlled fps to benchmark throughput and latency without hardware.

Thread model
------------
::

    ┌──────────────────────────────────────────────────────────────┐
    │  Main thread                                                 │
    │  cap.read() → worker.push_frame() → result = worker.latest  │
    │                     │                          ▲            │
    │              [FrameQueue]  max N items          │            │
    │              drop-oldest when full              │            │
    │                     │                          │            │
    │              ┌──────▼──────────────────────────┤            │
    │              │  Inference thread(s)  (×workers)│            │
    │              │  detector.process_frame()        │            │
    │              │  → latest_result ──────────────►│            │
    │              └─────────────────────────────────┘            │
    └──────────────────────────────────────────────────────────────┘

Usage
-----
::

    # Offline simulation (no camera, no download)
    python examples/neural_stream_demo.py

    # Webcam + microphone with real pretrained weights
    python examples/neural_stream_demo.py --webcam --pretrained

    # Webcam, GPU, 2 inference workers, INT8 quantized
    python examples/neural_stream_demo.py --webcam --device cuda --workers 2 --quantize

    # Simulation benchmark: ViViT backbone
    python examples/neural_stream_demo.py --video-model vivit --fps 60 --duration 10

    # High-throughput GPU config
    python examples/neural_stream_demo.py --webcam --device cuda --workers 4 --queue-size 16
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from collections import deque
from typing import Iterator

import numpy as np

from emotion_detection_action import (
    Config,
    EmotionDetector,
    InferenceWorker,
    NeuralEmotionResult,
)
from emotion_detection_action.actions.base import BaseActionHandler

# ---------------------------------------------------------------------------
# Custom action handler — subclass for your robot platform
# ---------------------------------------------------------------------------


class ConsoleActionHandler(BaseActionHandler):
    """Example handler: logs notable emotional events to stdout.

    Replace the body of ``execute()`` with your robot SDK calls
    (ROS publisher, serial command, WebSocket message, …).
    """

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def execute(self, result: NeuralEmotionResult) -> None:  # type: ignore[override]
        """Dispatch robot action based on dominant emotion.

        Always check the ``"unclear"`` label before acting — the model
        returns it when no person is detected, the signal is too noisy, or
        confidence is too low to classify reliably.
        """
        if result.dominant_emotion == "unclear":
            return  # nothing to react to
        if result.dominant_emotion in ("sad", "fearful") and result.confidence > 0.6:
            pass  # e.g. robot.comfort_gesture()
        if result.metrics.get("stress", 0) > 0.8:
            pass  # e.g. robot.reduce_intensity()


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

_ICONS: dict[str, str] = {
    "angry": "😠", "disgusted": "🤢", "fearful": "😨",
    "happy": "😊", "neutral": "😐", "sad": "😢", "surprised": "😲",
    "unclear": "?",  # no person present, noisy signal, or low confidence
}
_W = 18  # terminal bar width


def _bar(value: float, width: int = _W) -> str:
    n = max(0, min(width, int(value * width)))
    return "█" * n + "░" * (width - n)


def _print_status(
    frame_idx: int,
    result: NeuralEmotionResult | None,
    stats: "emotion_detection_action.WorkerStats",  # type: ignore[name-defined]  # noqa: F821
) -> None:
    if result is None:
        print(f"\r  [{frame_idx:5d}] (warming up…)  "
              f"queue={stats.queue_depth}  drop={stats.drop_rate:.0%}", end="", flush=True)
        return

    icon = _ICONS.get(result.dominant_emotion, "?")

    if result.dominant_emotion == "unclear":
        # Compact line — no metrics when signal is unclear
        print(
            f"\r  [{frame_idx:5d}] {icon} unclear        "
            f"conf={result.confidence:4.0%} | "
            f"q={stats.queue_depth} drop={stats.drop_rate:.0%} "
            f"p50={stats.inference_latency_ms_p50:.0f}ms "
            f"fps={stats.effective_fps:.1f}",
            end="",
            flush=True,
        )
        return

    print(
        f"\r  [{frame_idx:5d}] {icon} {result.dominant_emotion:<10} "
        f"{_bar(result.confidence)} {result.confidence:4.0%} | "
        f"stress {_bar(result.metrics.get('stress', 0), 10)}  "
        f"engage {_bar(result.metrics.get('engagement', 0), 10)}  "
        f"arousal {_bar(result.metrics.get('arousal', 0), 10)} | "
        f"q={stats.queue_depth} drop={stats.drop_rate:.0%} "
        f"p50={stats.inference_latency_ms_p50:.0f}ms "
        f"fps={stats.effective_fps:.1f}",
        end="",
        flush=True,
    )


def _draw_overlay(frame: np.ndarray, result: NeuralEmotionResult) -> np.ndarray:
    """Draw the inference result as an overlay in the top-left of the frame.

    Renders:
    * Emotion label + confidence on the first line
    * A filled confidence bar
    * Three metric rows (stress / engagement / arousal) with mini bars

    Args:
        frame: BGR uint8 frame to annotate (modified in-place copy).
        result: Latest :class:`NeuralEmotionResult` from the detector.

    Returns:
        Annotated BGR frame.
    """
    try:
        import cv2  # type: ignore[import]
    except ImportError:
        return frame

    out = frame.copy()
    h, w = out.shape[:2]

    # ── Layout constants ──────────────────────────────────────────────
    PAD = 10           # pixels from frame edge
    LINE_H = 26        # pixels per text row
    BAR_H = 8          # height of metric bars
    BAR_W = 120        # width of metric bars
    BOX_W = 240        # overlay background width
    BOX_H = PAD + LINE_H + BAR_H + PAD + 3 * (LINE_H + 2) + PAD
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE_LG = 0.65
    FONT_SCALE_SM = 0.48
    THICK = 1

    # Emotion-specific accent colour (BGR)
    _COLOURS: dict[str, tuple[int, int, int]] = {
        "angry":     (0,   50,  220),
        "disgusted": (0,  140,   70),
        "fearful":   (200, 100,   0),
        "happy":     (0,  200,  255),
        "neutral":   (180, 180, 180),
        "sad":       (200,  80,   0),
        "surprised": (0,  220,  220),
        "unclear":   (80,  80,   80),  # dark grey — unclear / no person present
    }
    accent = _COLOURS.get(result.dominant_emotion, (100, 200, 100))

    # ── Semi-transparent background ───────────────────────────────────
    x0, y0 = PAD, PAD
    x1, y1 = x0 + BOX_W, y0 + BOX_H
    # Clamp to frame bounds
    x1 = min(x1, w - 1)
    y1 = min(y1, h - 1)

    overlay = out.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)

    # Accent left-edge bar
    cv2.rectangle(out, (x0, y0), (x0 + 4, y1), accent, -1)

    # ── Unclear state: minimal overlay ───────────────────────────────
    if result.dominant_emotion == "unclear":
        cv2.putText(out, "UNCLEAR", (x0 + 12, y0 + LINE_H),
                    FONT, FONT_SCALE_LG, accent, THICK, cv2.LINE_AA)
        cv2.putText(out, f"conf {result.confidence:.0%}", (x0 + 12, y0 + LINE_H * 2),
                    FONT, FONT_SCALE_SM, (130, 130, 130), THICK, cv2.LINE_AA)
        return out

    # ── Emotion label + confidence ────────────────────────────────────
    ty = y0 + LINE_H
    emotion_text = f"{result.dominant_emotion.upper()}  {result.confidence:.0%}"
    cv2.putText(out, emotion_text, (x0 + 12, ty),
                FONT, FONT_SCALE_LG, accent, THICK + 1, cv2.LINE_AA)

    # Confidence bar (below emotion label)
    bar_y = ty + 4
    bar_filled = int(BAR_W * result.confidence)
    cv2.rectangle(out, (x0 + 12, bar_y), (x0 + 12 + BAR_W, bar_y + BAR_H),
                  (60, 60, 60), -1)
    cv2.rectangle(out, (x0 + 12, bar_y), (x0 + 12 + bar_filled, bar_y + BAR_H),
                  accent, -1)

    # ── Metric rows ───────────────────────────────────────────────────
    metrics_order = [
        ("stress",     result.metrics.get("stress", 0.0)),
        ("engagement", result.metrics.get("engagement", 0.0)),
        ("arousal",    result.metrics.get("arousal", 0.0)),
    ]
    metric_y = bar_y + BAR_H + PAD

    for label, value in metrics_order:
        metric_y += LINE_H
        # Label
        cv2.putText(out, f"{label:<10} {value:.2f}", (x0 + 12, metric_y),
                    FONT, FONT_SCALE_SM, (200, 200, 200), THICK, cv2.LINE_AA)
        # Mini bar
        bar_x = x0 + 12 + 105
        filled = int((BAR_W - 30) * value)
        cv2.rectangle(out, (bar_x, metric_y - 10),
                      (bar_x + BAR_W - 30, metric_y - 10 + BAR_H - 2),
                      (60, 60, 60), -1)
        cv2.rectangle(out, (bar_x, metric_y - 10),
                      (bar_x + filled, metric_y - 10 + BAR_H - 2),
                      (100, 180, 100), -1)

    return out


# ---------------------------------------------------------------------------
# Frame sources
# ---------------------------------------------------------------------------


def _webcam_source(
    camera_index: int = 0,
    sample_rate: int = 16000,
    chunk_ms: int = 33,
) -> Iterator[tuple[np.ndarray, np.ndarray | None]]:
    """Yield ``(bgr_frame, audio_chunk)`` from the real webcam + microphone.

    Audio capture requires ``sounddevice``.  If not installed, audio is ``None``.
    """
    try:
        import cv2
    except ImportError:
        raise RuntimeError("opencv-python is required for webcam mode: pip3 install opencv-python")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}")

    # Optional audio capture
    mic = None
    try:
        import sounddevice as sd
        chunk_samples = int(sample_rate * chunk_ms / 1000)
        mic = sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32")
        mic.start()
    except Exception:
        pass

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            audio: np.ndarray | None = None
            if mic is not None:
                try:
                    raw, _ = mic.read(chunk_samples)
                    audio = raw[:, 0]  # mono
                except Exception:
                    pass
            yield frame, audio
    finally:
        cap.release()
        if mic is not None:
            mic.stop()
            mic.close()


def _synthetic_source(
    fps: int = 30,
    duration_s: float = 5.0,
    frame_h: int = 120,
    frame_w: int = 160,
    sample_rate: int = 16000,
    scene_transitions: int = 3,
) -> Iterator[tuple[np.ndarray, np.ndarray | None, float, bool]]:
    """Yield ``(frame, audio, timestamp, should_reset)`` synthetic tuples."""
    total = int(fps * duration_s)
    chunk = int(sample_rate * 33 / 1000)
    transition_every = max(1, total // max(scene_transitions, 1))
    rng = np.random.default_rng(42)
    t0 = time.perf_counter()

    for i in range(total):
        ts = t0 + i / fps
        brightness = 80 + (i // transition_every) * 40
        frame = rng.integers(
            max(0, brightness - 40), min(255, brightness + 40),
            size=(frame_h, frame_w, 3), dtype=np.uint8,
        )
        audio = rng.normal(0, 0.1, chunk).astype(np.float32) if i % 60 < 45 else None
        should_reset = i > 0 and i % transition_every == 0
        yield frame, audio, ts, should_reset


# ---------------------------------------------------------------------------
# Webcam streaming mode
# ---------------------------------------------------------------------------


def run_webcam(
    camera_index: int = 0,
    device: str = "cpu",
    pretrained: bool = False,
    quantize: bool = False,
    video_model: str = "videomae",
    num_workers: int = 1,
    max_queue_size: int = 4,
    verbose: bool = False,
) -> None:
    """Run live webcam + microphone detection with the background worker.

    The inference result is overlaid on the live video window in the top-left
    corner, showing the dominant emotion, confidence bar, and the three
    attention metrics (stress, engagement, arousal).
    """
    try:
        import cv2
    except ImportError:
        raise RuntimeError("opencv-python required: pip3 install opencv-python")

    print("=" * 65)
    print("  Neural Emotion Detector — Live Webcam")
    print("=" * 65)
    print(f"  Backbone : {video_model.upper()}  device={device}")
    print(f"  Workers  : {num_workers}   queue={max_queue_size}")
    print(f"  Weights  : {'pretrained' if pretrained else 'stub (no download)'}")
    print(f"  Quantize : {'INT8 dynamic' if quantize else 'off'}")
    print("=" * 65)
    print("  Press Q (in the OpenCV window) or Ctrl-C to quit.\n")

    config = Config(
        two_tower_pretrained=pretrained,
        two_tower_video_model=video_model,   # type: ignore[arg-type]
        two_tower_device=device,
        vla_enabled=False,
        verbose=verbose,
    )
    detector = EmotionDetector(config)
    detector.initialize()

    if quantize:
        detector.quantize("dynamic")

    worker = InferenceWorker(
        detector,
        num_workers=num_workers,
        max_queue_size=max_queue_size,
    )
    worker.start()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        worker.stop()
        raise RuntimeError(f"Cannot open camera {camera_index}")

    # Read actual camera resolution for the window title
    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Camera   : {cam_w}×{cam_h}  index={camera_index}")
    print(f"\n  {'Frame':>5}  {'Emotion':<10}  Confidence            Metrics")
    print("-" * 65)

    frame_idx = 0
    last_result: NeuralEmotionResult | None = None

    try:
        while True:
            ret, bgr_frame = cap.read()
            if not ret:
                break

            # Capture microphone audio for this frame
            audio: np.ndarray | None = None  # InferenceWorker handles None gracefully

            # Push to background worker — never blocks the camera loop
            worker.push_frame(bgr_frame, audio=audio, timestamp=time.time())

            # Poll the latest result — always instant (no waiting)
            result = worker.latest_result
            if result is not None:
                last_result = result

            # ── Terminal status line ──────────────────────────────────
            _print_status(frame_idx, last_result, worker.stats)
            frame_idx += 1

            # ── OpenCV window with top-left overlay ───────────────────
            if last_result is not None:
                display = _draw_overlay(bgr_frame, last_result)
            else:
                display = bgr_frame.copy()
                cv2.putText(display, "Warming up…", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1,
                            cv2.LINE_AA)

            cv2.imshow("Emotion Detector", display)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        worker.stop()

    print()
    _print_final_stats(worker.stats)


# ---------------------------------------------------------------------------
# Simulation mode
# ---------------------------------------------------------------------------


def run_simulation(
    fps: int = 30,
    duration_s: float = 5.0,
    device: str = "cpu",
    pretrained: bool = False,
    quantize: bool = False,
    video_model: str = "videomae",
    num_workers: int = 1,
    max_queue_size: int = 4,
    verbose: bool = False,
) -> None:
    """Simulate a camera stream with synthetic frames and measure throughput."""
    total_frames = int(fps * duration_s)

    print("=" * 65)
    print("  Neural Emotion Detector — Stream Simulation")
    print("=" * 65)
    print(f"  Backbone : {video_model.upper()}  device={device}")
    print(f"  Workers  : {num_workers}   queue={max_queue_size}")
    print(f"  Weights  : {'pretrained' if pretrained else 'stub (no download)'}")
    print(f"  Quantize : {'INT8 dynamic' if quantize else 'off'}")
    print(f"  Simulate : {fps}fps × {duration_s}s = {total_frames} frames")
    print("=" * 65)

    config = Config(
        two_tower_pretrained=pretrained,
        two_tower_video_model=video_model,   # type: ignore[arg-type]
        two_tower_device=device,
        vla_enabled=False,
        verbose=verbose,
    )

    detector = EmotionDetector(config)
    detector.initialize()

    if quantize:
        print("\n  Quantizing to INT8 dynamic…")
        detector.quantize("dynamic")

    worker = InferenceWorker(
        detector,
        num_workers=num_workers,
        max_queue_size=max_queue_size,
    )

    print(f"\n  {'Frame':>5}  {'Emotion':<10}  Confidence            Metrics")
    print("-" * 65)

    frame_idx = 0
    last_result: NeuralEmotionResult | None = None
    reset_count = 0
    frame_timestamps: deque[float] = deque(maxlen=60)

    # --- Capture loop (main thread) ---
    with worker:
        for frame, audio, ts, should_reset in _synthetic_source(
            fps=fps, duration_s=duration_s
        ):
            if should_reset:
                worker.reset()
                reset_count += 1
                print(f"\n  --- Subject change #{reset_count}: GRU + buffer reset ---")

            # Push to background worker — always non-blocking
            worker.push_frame(frame, audio, timestamp=ts)

            # Poll latest result — always instant
            result = worker.latest_result
            if result is not None:
                last_result = result

            _print_status(frame_idx, last_result, worker.stats)
            frame_timestamps.append(time.perf_counter())
            frame_idx += 1

            # Pace the simulation to the target fps
            time.sleep(max(0.0, 1.0 / fps - 0.001))

        # Drain remaining frames in the worker queue
        _drain_worker(worker, timeout=10.0)

    print()
    _print_final_stats(worker.stats)

    # Show latent embedding (VLA integration demo)
    if last_result is not None:
        emb = last_result.latent_embedding
        print(
            f"\n  latent_embedding [{len(emb)} dims, first 8]: "
            f"{[round(v, 4) for v in emb[:8]]}"
        )
        print("  → Feed this directly into your VLA model (e.g., OpenVLA).")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _drain_worker(worker: InferenceWorker, timeout: float = 5.0) -> None:
    """Wait for the worker to process remaining queued frames."""
    deadline = time.perf_counter() + timeout
    while worker.stats.queue_depth > 0 and time.perf_counter() < deadline:
        time.sleep(0.05)


def _print_final_stats(stats: "WorkerStats") -> None:  # type: ignore[name-defined]
    print("-" * 65)
    print(f"\n  Frames submitted : {stats.frames_submitted}")
    print(f"  Frames processed : {stats.frames_processed}")
    print(f"  Frames dropped   : {stats.frames_dropped}  ({stats.drop_rate:.1%})")
    print(f"  Results produced : {stats.results_produced}")
    print(f"  Inference p50    : {stats.inference_latency_ms_p50:.1f} ms")
    print(f"  Inference p95    : {stats.inference_latency_ms_p95:.1f} ms")
    print(f"  Effective fps    : {stats.effective_fps:.1f}")
    status = "✓ keeping up" if stats.is_keeping_up else "⚠ dropping frames — consider GPU or reducing fps"
    print(f"  Status           : {status}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Neural Emotion Detector — streaming demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--webcam", action="store_true",
                        help="Use real webcam + microphone instead of synthetic data")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--fps", type=int, default=30,
                        help="Target fps (simulation mode only)")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Simulation duration in seconds")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device: cpu | cuda | mps")
    parser.add_argument("--pretrained", action="store_true",
                        help="Download HuggingFace pretrained weights (~1.8 GB)")
    parser.add_argument("--quantize", action="store_true",
                        help="Apply INT8 dynamic quantization")
    parser.add_argument("--video-model", choices=["videomae", "vivit"], default="videomae",
                        help="Video backbone (videomae=16 frames, vivit=32 frames)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Inference worker threads (>1 for GPU parallelism)")
    parser.add_argument("--queue-size", type=int, default=4,
                        help="Max frames queued before oldest is dropped")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))

    if args.webcam:
        run_webcam(
            camera_index=args.camera,
            device=args.device,
            pretrained=args.pretrained,
            quantize=args.quantize,
            video_model=args.video_model,
            num_workers=args.workers,
            max_queue_size=args.queue_size,
            verbose=args.verbose,
        )
    else:
        run_simulation(
            fps=args.fps,
            duration_s=args.duration,
            device=args.device,
            pretrained=args.pretrained,
            quantize=args.quantize,
            video_model=args.video_model,
            num_workers=args.workers,
            max_queue_size=args.queue_size,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()
