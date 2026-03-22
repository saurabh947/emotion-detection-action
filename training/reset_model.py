#!/usr/bin/env python3
"""Factory-reset the emotion detection SDK model.

What "reset" means
------------------
The SDK has three layers of learned weights:

1. **HuggingFace pretrained backbones** (AffectNet ViT + emotion2vec) — downloaded
   once to the HuggingFace / FunASR cache.
   These are the *factory defaults* the SDK ships with.

2. **Phase 1 checkpoint** (``outputs/phase1_best.pt``) — heads + fusion
   trained on your data.

3. **Phase 2 checkpoint** (``outputs/phase2_best.pt``) — full end-to-end
   fine-tune on your data.

A factory reset removes checkpoints 2 and 3 so the SDK reverts to using
the original pretrained backbone weights + randomly-initialised heads — the
exact state it was in before you ran any training.

Optionally, you can also delete the HuggingFace backbone cache (option 3
below), which forces a fresh re-download on the next ``--pretrained`` run.
Only do this if you want to start from completely fresh weights.

Usage
-----
::

    # Interactive mode (recommended — confirms each step):
    python training/reset_model.py

    # Delete all training checkpoints silently:
    python training/reset_model.py --yes --checkpoints-only

    # Full reset including HuggingFace backbone cache:
    python training/reset_model.py --yes --full

    # Custom output directory:
    python training/reset_model.py --output-dir /my/training/outputs

    # Preview what would be deleted (dry run):
    python training/reset_model.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

# HuggingFace model IDs whose local caches can be cleared.
# emotion2vec is stored in the FunASR / ModelScope cache (~/.cache/modelscope),
# not in the HuggingFace hub cache — it is not listed here.
_HF_MODEL_IDS: dict[str, str] = {
    "AffectNet ViT (video backbone, default)": "trpakov/vit-face-expression",
    "VideoMAE (legacy video backbone)": "MCG-NJU/videomae-base",
    "ViViT (legacy video backbone)": "google/vivit-b-16x2-kinetics400",
    "AST (legacy audio backbone)": "MIT/ast-finetuned-audioset-10-10-0.4593",
}

# Checkpoint filenames produced by the training scripts
_CHECKPOINT_NAMES: list[str] = [
    "phase1_best.pt",
    "phase1_last.pt",
    "phase2_best.pt",
    "phase2_last.pt",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _confirm(prompt: str, default: bool = False) -> bool:
    """Prompt the user for a yes/no answer.  Returns ``True`` for yes."""
    options = " [Y/n] " if default else " [y/N] "
    try:
        answer = input(prompt + options).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False

    if not answer:
        return default
    return answer in ("y", "yes")


def _sizeof_fmt(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes //= 1024
    return f"{num_bytes:.1f} TB"


def _dir_size(path: Path) -> int:
    """Recursively sum file sizes under ``path``."""
    total = 0
    try:
        for entry in path.rglob("*"):
            if entry.is_file():
                total += entry.stat().st_size
    except PermissionError:
        pass
    return total


def _delete_path(path: Path, dry_run: bool) -> None:
    if dry_run:
        print(f"    [dry-run] would delete: {path}")
        return
    if path.is_file():
        path.unlink()
        print(f"    ✓ Deleted file  : {path}")
    elif path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
        print(f"    ✓ Deleted folder: {path}")
    else:
        print(f"    (not found, skipping): {path}")


def _find_hf_cache_dirs() -> dict[str, Path | None]:
    """Locate the HuggingFace cached model directories for each backbone."""
    hf_cache = Path(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    ) / "hub"

    result: dict[str, Path | None] = {}
    for label, model_id in _HF_MODEL_IDS.items():
        # HuggingFace stores models as models--{org}--{name} under hub/
        folder_name = "models--" + model_id.replace("/", "--")
        candidate = hf_cache / folder_name
        result[label] = candidate if candidate.exists() else None

    return result


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Factory-reset the emotion detection SDK model weights",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--output-dir", default="outputs",
        help="Directory where training checkpoints are stored",
    )
    p.add_argument(
        "--yes", "-y", action="store_true",
        help="Skip all confirmation prompts (use with care)",
    )
    p.add_argument(
        "--checkpoints-only", action="store_true",
        help="Only delete training checkpoints (phases 1 + 2), leave HF cache untouched",
    )
    p.add_argument(
        "--full", action="store_true",
        help="Full reset: delete checkpoints AND HuggingFace backbone cache "
             "(forces re-download on next --pretrained run)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be deleted without actually deleting anything",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Reset actions
# ---------------------------------------------------------------------------


def reset_checkpoints(output_dir: str, dry_run: bool) -> int:
    """Delete Phase 1 and Phase 2 checkpoint files.

    Returns the number of files deleted (or that would be deleted).
    """
    root = Path(output_dir)
    deleted = 0
    for name in _CHECKPOINT_NAMES:
        path = root / name
        if path.exists():
            size = _sizeof_fmt(path.stat().st_size)
            print(f"    {name}  ({size})")
            _delete_path(path, dry_run)
            deleted += 1
    return deleted


def reset_hf_cache(dry_run: bool) -> int:
    """Delete cached HuggingFace backbone model files.

    Returns the number of cache directories deleted.
    """
    cache_dirs = _find_hf_cache_dirs()
    deleted = 0
    for label, path in cache_dirs.items():
        if path is None:
            print(f"    {label}: not cached (skipping)")
            continue
        size = _sizeof_fmt(_dir_size(path))
        print(f"    {label}  ({size})  →  {path}")
        _delete_path(path, dry_run)
        deleted += 1
    return deleted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║       Emotion Detection SDK — Model Factory Reset            ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    if args.dry_run:
        print("  *** DRY RUN — nothing will actually be deleted ***\n")

    # ── Decide what to reset ──────────────────────────────────────────────
    do_checkpoints = True
    do_hf_cache = False

    if args.full:
        do_hf_cache = True
    elif not args.checkpoints_only:
        # Interactive mode
        print("  What would you like to reset?\n")
        print("  [1] Training checkpoints only (phase1_best.pt, phase2_best.pt, etc.)")
        print("      The SDK will revert to pretrained backbone + random heads.")
        print()
        print("  [2] Training checkpoints  +  HuggingFace backbone cache")
        print("      Forces re-download of AffectNet ViT + emotion2vec on next run.")
        print()

        if not args.yes:
            try:
                choice = input("  Enter choice [1/2] (default=1): ").strip() or "1"
            except (EOFError, KeyboardInterrupt):
                print("\n  Aborted.")
                sys.exit(0)
        else:
            choice = "1"

        if choice == "2":
            do_hf_cache = True

    # ── Summary ───────────────────────────────────────────────────────────
    print()
    print("  The following will be deleted:")
    print()

    root = Path(args.output_dir)
    found_checkpoints = [
        root / name for name in _CHECKPOINT_NAMES if (root / name).exists()
    ]
    if found_checkpoints:
        print(f"  Training checkpoints in {root!r}:")
        for p in found_checkpoints:
            print(f"    • {p.name}  ({_sizeof_fmt(p.stat().st_size)})")
    else:
        print(f"  No training checkpoints found in {root!r}")

    if do_hf_cache:
        print()
        print("  HuggingFace model cache:")
        for label, path in _find_hf_cache_dirs().items():
            if path:
                size = _sizeof_fmt(_dir_size(path))
                print(f"    • {label}  ({size})")
            else:
                print(f"    • {label}: not cached")

    print()

    if not found_checkpoints and not do_hf_cache:
        print("  Nothing to reset — model is already at factory state.")
        return

    if not args.yes and not args.dry_run:
        if not _confirm("  Proceed with reset?", default=False):
            print("  Aborted — no files changed.")
            return

    # ── Execute ───────────────────────────────────────────────────────────
    print()
    print("  Resetting …")

    ckpt_count = reset_checkpoints(args.output_dir, args.dry_run)
    hf_count = 0
    if do_hf_cache:
        print()
        print("  Clearing HuggingFace backbone cache …")
        hf_count = reset_hf_cache(args.dry_run)

    # ── Summary ───────────────────────────────────────────────────────────
    print()
    print("─" * 65)
    if args.dry_run:
        print("  Dry run complete — no files were actually deleted.")
    else:
        print(f"  Reset complete.")
        print(f"    Training checkpoints removed : {ckpt_count}")
        if do_hf_cache:
            print(f"    HuggingFace cache entries    : {hf_count}")

    print()
    print("  The SDK is now at factory state.")
    print("  On the next run it will use:")
    print("    • AffectNet ViT pretrained on 450 K facial emotion images (if --pretrained)")
    print("    • emotion2vec via FunASR                                  (if --pretrained)")
    print("    • Randomly-initialised fusion + emotion heads")
    print()
    print("  Note: emotion2vec weights are cached separately by FunASR at")
    print("    ~/.cache/modelscope/hub/iic/emotion2vec_base")
    print("  To also clear the emotion2vec cache, delete that directory manually.")
    print()
    print("  To start fresh training:")
    print("    python training/train_phase1.py --data-dir /data/RAVDESS --pretrained \\")
    print("        --class-weights 5.70,35.50,20.50,1.00,1.90,5.50,9.60,20.50")
    print("─" * 65)
    print()


if __name__ == "__main__":
    main()
