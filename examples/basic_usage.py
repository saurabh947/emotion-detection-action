#!/usr/bin/env python3
"""Basic usage example for the emotion detector SDK.

This example demonstrates how to use the EmotionDetector for
processing images and video files with optional audio.

Supported emotions:
    - Facial (from images/video): happy, sad, angry, fearful, surprised, disgusted, neutral
    - Speech (from audio): happy, sad, angry, neutral
"""

from pathlib import Path

from emotion_detector import Config, EmotionDetector


def process_image(image_path: str) -> None:
    """Process a single image and detect emotions.

    Args:
        image_path: Path to the image file.
    """
    print(f"\n=== Processing Image: {image_path} ===\n")

    # Create config (use CPU for this example)
    config = Config(
        device="cpu",
        vla_enabled=False,  # Disable VLA for faster loading
        verbose=True,
    )

    # Create detector
    with EmotionDetector(config) as detector:
        # Process the image
        results = detector.process(image_path=image_path)

        if not results:
            print("No faces detected in the image.")
            return

        for result in results:
            print(f"Timestamp: {result.timestamp:.2f}s")
            print(f"Dominant Emotion: {result.emotion.dominant_emotion.value}")
            print(f"Confidence: {result.emotion.fusion_confidence:.2%}")
            print("\nEmotion Scores:")
            for emotion, score in result.emotion.emotions.to_dict().items():
                bar = "█" * int(score * 20)
                print(f"  {emotion:12s}: {bar} {score:.2%}")
            print()


def process_video(video_path: str, audio_path: str | None = None) -> None:
    """Process a video file with optional audio.

    Args:
        video_path: Path to the video file.
        audio_path: Optional path to audio file.
    """
    print(f"\n=== Processing Video: {video_path} ===\n")

    config = Config(
        device="cpu",
        vla_enabled=False,
        frame_skip=5,  # Process every 5th frame
        verbose=True,
    )

    with EmotionDetector(config) as detector:
        results = detector.process(
            video_path=video_path,
            audio_path=audio_path,
        )

        print(f"\nProcessed {len(results)} frames\n")

        # Summarize emotions
        emotion_counts: dict[str, int] = {}
        for result in results:
            emotion = result.emotion.dominant_emotion.value
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

        print("Emotion Distribution:")
        total = len(results)
        for emotion, count in sorted(emotion_counts.items(), key=lambda x: -x[1]):
            pct = count / total * 100
            bar = "█" * int(pct / 5)
            print(f"  {emotion:12s}: {bar} {pct:.1f}% ({count} frames)")


def process_frame_by_frame() -> None:
    """Demonstrate frame-by-frame processing API."""
    import cv2
    import numpy as np

    print("\n=== Frame-by-Frame Processing Demo ===\n")

    config = Config(
        device="cpu",
        vla_enabled=False,
        verbose=False,
    )

    # Create a dummy frame (colored rectangle simulating a face)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[100:380, 200:440] = (200, 180, 160)  # Skin-like color

    with EmotionDetector(config) as detector:
        result = detector.process_frame(frame, timestamp=0.0)

        if result:
            print(f"Detected emotion: {result.emotion.dominant_emotion.value}")
        else:
            print("No face detected in frame")


def main() -> None:
    """Run basic usage examples."""
    print("Emotion Detector SDK - Basic Usage Examples")
    print("=" * 50)

    # Check for example files
    examples_dir = Path(__file__).parent
    test_image = examples_dir / "test_image.jpg"
    test_video = examples_dir / "test_video.mp4"

    if test_image.exists():
        process_image(str(test_image))
    else:
        print(f"\nNote: Place a test image at {test_image} to test image processing")

    if test_video.exists():
        process_video(str(test_video))
    else:
        print(f"\nNote: Place a test video at {test_video} to test video processing")

    # Run frame-by-frame demo with synthetic data
    process_frame_by_frame()

    print("\n" + "=" * 50)
    print("Examples completed!")


if __name__ == "__main__":
    main()

