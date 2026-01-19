#!/usr/bin/env python3
"""Batch processing example for the emotion detector SDK.

This example demonstrates how to process multiple files and
generate emotion analysis reports.

Supported file types:
    - Video: .mp4, .avi, .mov, .mkv, .webm
    - Image: .jpg, .jpeg, .png, .bmp, .gif
    - Audio: .wav, .mp3 (speech emotions: happy, sad, angry, neutral)

Usage:
    python batch_processing.py --input-dir ./videos --output-dir ./results
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from emotion_detector import Config, EmotionDetector
from emotion_detector.core.types import PipelineResult


def analyze_results(results: list[PipelineResult]) -> dict[str, Any]:
    """Analyze pipeline results and generate statistics.

    Args:
        results: List of pipeline results.

    Returns:
        Analysis dictionary with statistics.
    """
    if not results:
        return {
            "total_frames": 0,
            "emotions": {},
            "timeline": [],
        }

    # Count emotions
    emotion_counts: dict[str, int] = {}
    confidence_sums: dict[str, float] = {}

    timeline: list[dict[str, Any]] = []

    for result in results:
        emotion = result.emotion.dominant_emotion.value
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        confidence_sums[emotion] = (
            confidence_sums.get(emotion, 0) + result.emotion.fusion_confidence
        )

        # Build timeline entry
        timeline.append({
            "timestamp": result.timestamp,
            "dominant_emotion": emotion,
            "confidence": result.emotion.fusion_confidence,
            "scores": result.emotion.emotions.to_dict(),
        })

    # Calculate percentages and average confidences
    total = len(results)
    emotion_stats = {}
    for emotion, count in emotion_counts.items():
        emotion_stats[emotion] = {
            "count": count,
            "percentage": count / total * 100,
            "avg_confidence": confidence_sums[emotion] / count,
        }

    # Find emotion transitions
    transitions: list[dict[str, Any]] = []
    prev_emotion = None
    for entry in timeline:
        if prev_emotion and entry["dominant_emotion"] != prev_emotion:
            transitions.append({
                "timestamp": entry["timestamp"],
                "from": prev_emotion,
                "to": entry["dominant_emotion"],
            })
        prev_emotion = entry["dominant_emotion"]

    return {
        "total_frames": total,
        "duration_seconds": results[-1].timestamp if results else 0,
        "emotions": emotion_stats,
        "transitions": transitions,
        "timeline": timeline,
    }


def process_file(
    detector: EmotionDetector,
    file_path: Path,
) -> dict[str, Any]:
    """Process a single file.

    Args:
        detector: Initialized EmotionDetector.
        file_path: Path to media file.

    Returns:
        Processing results and analysis.
    """
    suffix = file_path.suffix.lower()

    # Determine file type
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    audio_exts = {".wav", ".mp3", ".flac", ".ogg"}

    results: list[PipelineResult] = []

    if suffix in video_exts:
        results = detector.process(video_path=str(file_path))
    elif suffix in image_exts:
        results = detector.process(image_path=str(file_path))
    elif suffix in audio_exts:
        results = detector.process(audio_path=str(file_path))
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    return {
        "file": str(file_path),
        "type": "video" if suffix in video_exts else "image" if suffix in image_exts else "audio",
        "analysis": analyze_results(results),
    }


def generate_report(
    all_results: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Generate a JSON report.

    Args:
        all_results: Results from all processed files.
        output_path: Path to save report.
    """
    report = {
        "generated_at": datetime.now().isoformat(),
        "total_files": len(all_results),
        "files": all_results,
        "summary": generate_summary(all_results),
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to: {output_path}")


def generate_summary(all_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Generate overall summary across all files.

    Args:
        all_results: Results from all files.

    Returns:
        Summary statistics.
    """
    total_frames = 0
    total_duration = 0
    global_emotions: dict[str, int] = {}

    for result in all_results:
        analysis = result["analysis"]
        total_frames += analysis["total_frames"]
        total_duration += analysis.get("duration_seconds", 0)

        for emotion, stats in analysis["emotions"].items():
            global_emotions[emotion] = (
                global_emotions.get(emotion, 0) + stats["count"]
            )

    # Calculate global percentages
    emotion_percentages = {}
    if total_frames > 0:
        emotion_percentages = {
            e: count / total_frames * 100
            for e, count in global_emotions.items()
        }

    return {
        "total_frames_processed": total_frames,
        "total_duration_seconds": total_duration,
        "overall_emotion_distribution": emotion_percentages,
    }


def print_analysis(analysis: dict[str, Any], file_name: str) -> None:
    """Print analysis to console.

    Args:
        analysis: Analysis dictionary.
        file_name: Name of the processed file.
    """
    print(f"\n{'=' * 50}")
    print(f"File: {file_name}")
    print(f"{'=' * 50}")
    print(f"Frames processed: {analysis['total_frames']}")

    if analysis["duration_seconds"]:
        print(f"Duration: {analysis['duration_seconds']:.1f}s")

    print("\nEmotion Distribution:")
    for emotion, stats in sorted(
        analysis["emotions"].items(),
        key=lambda x: -x[1]["percentage"],
    ):
        bar = "█" * int(stats["percentage"] / 5)
        print(
            f"  {emotion:12s}: {bar} {stats['percentage']:5.1f}% "
            f"(confidence: {stats['avg_confidence']:.0%})"
        )

    if analysis["transitions"]:
        print(f"\nEmotion transitions: {len(analysis['transitions'])}")
        for i, trans in enumerate(analysis["transitions"][:5]):
            print(
                f"  {trans['timestamp']:.1f}s: {trans['from']} → {trans['to']}"
            )
        if len(analysis["transitions"]) > 5:
            print(f"  ... and {len(analysis['transitions']) - 5} more")


def main() -> None:
    """Run batch processing."""
    parser = argparse.ArgumentParser(
        description="Batch emotion detection processing"
    )
    parser.add_argument(
        "--input-dir", "-i",
        type=Path,
        required=True,
        help="Directory containing media files",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("./emotion_results"),
        help="Directory for output reports",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (cpu, cuda, mps)",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=5,
        help="Process every nth frame (default: 5)",
    )

    args = parser.parse_args()

    # Validate input directory
    if not args.input_dir.exists():
        print(f"Error: Input directory not found: {args.input_dir}")
        return

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Find media files
    extensions = {
        ".mp4", ".avi", ".mov", ".mkv",
        ".jpg", ".jpeg", ".png",
        ".wav", ".mp3",
    }
    files = [
        f for f in args.input_dir.iterdir()
        if f.suffix.lower() in extensions
    ]

    if not files:
        print(f"No media files found in {args.input_dir}")
        return

    print(f"Found {len(files)} files to process")

    # Configure detector
    config = Config(
        device=args.device,
        vla_enabled=False,
        frame_skip=args.frame_skip,
        verbose=False,
    )

    all_results: list[dict[str, Any]] = []

    with EmotionDetector(config) as detector:
        for i, file_path in enumerate(files, 1):
            print(f"\nProcessing [{i}/{len(files)}]: {file_path.name}")

            try:
                result = process_file(detector, file_path)
                all_results.append(result)
                print_analysis(result["analysis"], file_path.name)
            except Exception as e:
                print(f"  Error processing: {e}")
                all_results.append({
                    "file": str(file_path),
                    "error": str(e),
                })

    # Generate report
    report_path = args.output_dir / f"emotion_report_{datetime.now():%Y%m%d_%H%M%S}.json"
    generate_report(all_results, report_path)

    # Print summary
    print("\n" + "=" * 50)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 50)
    print(f"Files processed: {len(all_results)}")
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()

