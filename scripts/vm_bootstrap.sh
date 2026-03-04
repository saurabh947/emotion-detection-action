#!/usr/bin/env bash
# =============================================================================
# vm_bootstrap.sh — One-time VM environment setup
# =============================================================================
# Runs on the remote GPU VM to install system dependencies, create a Python
# virtual environment, and install all SDK requirements.
#
# Designed for NVIDIA CUDA VMs running Ubuntu 20.04 / 22.04 (the standard
# image on SF Compute, AWS p5, GCP A3, Azure NC H100 v5, RunPod, etc.).
#
# Usage (called automatically by remote.sh setup, or run manually):
#   bash scripts/vm_bootstrap.sh
#
# Re-running is safe — it is idempotent.
# =============================================================================

set -euo pipefail

REMOTE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$REMOTE_DIR/venv"

log() { echo "  [bootstrap] $*"; }
section() { echo ""; echo "  ── $* ──────────────────────────────────"; }

section "System info"
log "Hostname  : $(hostname)"
log "User      : $(whoami)"
log "Directory : $REMOTE_DIR"
uname -a || true

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------

section "System packages"
apt-get update -qq
apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    git \
    curl \
    rsync \
    htop \
    tmux

log "System packages installed."

# ---------------------------------------------------------------------------
# 2. NVIDIA / CUDA check
# ---------------------------------------------------------------------------

section "GPU check"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    log "WARNING: nvidia-smi not found. Ensure CUDA drivers are installed."
    log "Training will fall back to CPU (very slow for real data)."
fi

# ---------------------------------------------------------------------------
# 3. Python virtual environment
# ---------------------------------------------------------------------------

section "Python virtual environment"
if [[ ! -d "$VENV_DIR" ]]; then
    log "Creating venv at $VENV_DIR …"
    python3 -m venv "$VENV_DIR"
else
    log "Venv already exists at $VENV_DIR"
fi

# Activate venv for the rest of this script
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

pip install --upgrade pip wheel setuptools -q
log "pip $(pip --version)"

# ---------------------------------------------------------------------------
# 4. PyTorch (CUDA build)
# ---------------------------------------------------------------------------

section "PyTorch (CUDA)"
# Install the CUDA-enabled PyTorch build. This is the standard index for
# CUDA 12.x which matches H100, A100, RTX 40xx, etc.
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    log "PyTorch with CUDA already installed and working:"
    python3 -c "import torch; print(f'    torch {torch.__version__}  CUDA {torch.version.cuda}  GPUs: {torch.cuda.device_count()}')"
else
    log "Installing PyTorch with CUDA 12.x support …"
    pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu121 -q
    python3 -c "import torch; print(f'    torch {torch.__version__}  CUDA available: {torch.cuda.is_available()}')"
fi

# ---------------------------------------------------------------------------
# 5. SDK dependencies
# ---------------------------------------------------------------------------

section "SDK dependencies"
cd "$REMOTE_DIR"

# Install the SDK itself in editable mode (pulls all dependencies from pyproject.toml)
pip install -e ".[vla]" -q

log "SDK installed."

# ---------------------------------------------------------------------------
# 6. Create required directories
# ---------------------------------------------------------------------------

section "Directory structure"
mkdir -p "$REMOTE_DIR/outputs"
mkdir -p "$REMOTE_DIR/data/combined"
mkdir -p "$REMOTE_DIR/results"
log "Directories ready."

# ---------------------------------------------------------------------------
# 7. Smoke test
# ---------------------------------------------------------------------------

section "Smoke test"
python3 - <<'EOF'
import torch
import torchvision
import torchaudio
import cv2
import numpy as np

print(f"    torch       {torch.__version__}")
print(f"    torchvision {torchvision.__version__}")
print(f"    torchaudio  {torchaudio.__version__}")
print(f"    opencv      {cv2.__version__}")
print(f"    numpy       {np.__version__}")
print(f"    CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        gb = props.total_memory / 1024**3
        print(f"    GPU {i}: {props.name}  {gb:.1f} GB")
EOF

section "Bootstrap complete"
log "Environment is ready."
log ""
log "Next steps:"
log "  1. From your local machine, sync your dataset:"
log "       ./scripts/remote.sh sync-data"
log ""
log "  2. Then start Phase 1 training:"
log "       ./scripts/remote.sh train-phase1"
log ""
log "  3. Monitor with:"
log "       ./scripts/remote.sh status"
log "       ./scripts/remote.sh logs"
