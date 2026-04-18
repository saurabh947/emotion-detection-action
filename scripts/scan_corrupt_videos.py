#!/usr/bin/env python3
"""Scan data/combined/ for corrupt video files and optionally remove them.

A file is considered corrupt if ffprobe cannot read its container metadata
(covers: truncated files, moov-atom-missing MP4s, broken FLV headers, zero-
duration clips, and empty files).

Usage
-----
    # Dry-run: list corrupt files, print counts, don't delete
    python3 scripts/scan_corrupt_videos.py

    # Remove corrupt files
    python3 scripts/scan_corrupt_videos.py --remove

    # Scan a different data root
    python3 scripts/scan_corrupt_videos.py --data-dir path/to/data --remove
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# ---------------------------------------------------------------------------
# Probe helpers
# ---------------------------------------------------------------------------

SUPPORTED_EXTS = {".mp4", ".flv", ".avi", ".mov", ".mkv", ".webm"}


def _is_corrupt(path: Path) -> tuple[bool, str]:
    """Return (is_corrupt, reason) for a single video file."""
    if path.stat().st_size == 0:
        return True, "empty file"

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",               # suppress normal output
                "-select_streams", "v:0",    # first video stream
                "-show_entries", "stream=nb_frames,duration",
                "-of", "default=noprint_wrappers=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except FileNotFoundError:
        print("ERROR: ffprobe not found. Install ffmpeg: brew install ffmpeg", file=sys.stderr)
        sys.exit(1)
    except subprocess.TimeoutExpired:
        return True, "ffprobe timeout (likely severely corrupt)"

    # Any error output from ffprobe = problem
    stderr = result.stderr.strip()
    if stderr:
        # Pick out the most informative part
        reason = stderr.split("\n")[0][:120]
        return True, reason

    # ffprobe returned no error but also no stream info = no video stream found
    if not result.stdout.strip():
        return True, "no video stream found"

    return False, ""


def _scan_file(path: Path) -> tuple[Path, bool, str]:
    corrupt, reason = _is_corrupt(path)
    return path, corrupt, reason


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Scan for corrupt video files")
    parser.add_argument(
        "--data-dir",
        default="data/combined",
        help="Root directory to scan (default: data/combined)",
    )
    parser.add_argument(
        "--remove",
        action="store_true",
        help="Delete corrupt files (default: dry-run only)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel ffprobe workers (default: 8)",
    )
    args = parser.parse_args()

    data_root = Path(args.data_dir).resolve()
    if not data_root.is_dir():
        print(f"ERROR: data dir not found: {data_root}", file=sys.stderr)
        sys.exit(1)

    # Collect all video files
    all_files: list[Path] = [
        p for p in data_root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    ]

    total_before = len(all_files)
    print(f"\n{'='*60}")
    print(f"Scanning {total_before:,} video files in {data_root}")
    print(f"Workers: {args.workers} | Mode: {'REMOVE' if args.remove else 'DRY-RUN'}")
    print(f"{'='*60}\n")

    # Per-class before counts
    before_counts: dict[str, int] = defaultdict(int)
    for f in all_files:
        before_counts[f.parent.name] += 1

    # Scan in parallel
    corrupt_files: list[tuple[Path, str]] = []
    scanned = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_scan_file, f): f for f in all_files}
        for future in as_completed(futures):
            path, is_corrupt, reason = future.result()
            scanned += 1
            if scanned % 500 == 0 or scanned == total_before:
                pct = scanned / total_before * 100
                print(f"  [{scanned:>6,}/{total_before:,}  {pct:5.1f}%]  corrupt so far: {len(corrupt_files)}", end="\r")
            if is_corrupt:
                corrupt_files.append((path, reason))

    print()  # newline after progress

    # Sort for readable output
    corrupt_files.sort(key=lambda x: (x[0].parent.name, x[0].name))

    # Print corrupt file details
    if corrupt_files:
        print(f"\n--- Corrupt files ({len(corrupt_files):,}) ---")
        for path, reason in corrupt_files:
            print(f"  [{path.parent.name}] {path.name}  →  {reason}")

    # Remove if requested
    removed = 0
    if args.remove and corrupt_files:
        print(f"\nRemoving {len(corrupt_files):,} corrupt files...")
        for path, _ in corrupt_files:
            try:
                path.unlink()
                removed += 1
            except OSError as e:
                print(f"  WARNING: could not remove {path}: {e}", file=sys.stderr)

    # Per-class after counts
    after_counts: dict[str, int] = defaultdict(int)
    remaining = [
        p for p in data_root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    ]
    for f in remaining:
        after_counts[f.parent.name] += 1

    total_after = len(remaining)

    # Summary table
    print(f"\n{'='*60}")
    print(f"{'Class':<14} {'Before':>8} {'Corrupt':>8} {'After':>8}")
    print(f"{'-'*14} {'-'*8} {'-'*8} {'-'*8}")
    all_classes = sorted(set(list(before_counts.keys()) + list(after_counts.keys())))
    for cls in all_classes:
        b = before_counts[cls]
        a = after_counts[cls]
        c = b - a if args.remove else sum(1 for p, _ in corrupt_files if p.parent.name == cls)
        marker = "  ← " if c > 0 else ""
        print(f"{cls:<14} {b:>8,} {c:>8,} {a:>8,}{marker}")

    print(f"{'-'*14} {'-'*8} {'-'*8} {'-'*8}")
    total_corrupt = len(corrupt_files)
    print(f"{'TOTAL':<14} {total_before:>8,} {total_corrupt:>8,} {total_after:>8,}")
    print(f"{'='*60}")

    if not args.remove and corrupt_files:
        print(f"\nDRY-RUN: {total_corrupt:,} corrupt files found but NOT removed.")
        print(f"Re-run with --remove to delete them.\n")
    elif args.remove:
        print(f"\nRemoved {removed:,} files. {total_after:,} clean files remain.\n")
    else:
        print(f"\nNo corrupt files found. Dataset is clean.\n")


if __name__ == "__main__":
    main()
