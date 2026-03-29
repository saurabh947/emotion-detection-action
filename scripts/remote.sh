#!/usr/bin/env bash
# =============================================================================
# remote.sh — Cloud-agnostic remote compute management
# =============================================================================
# Supports two provider modes set via COMPUTE_PROVIDER in compute.env:
#
#   sfcompute   SF Compute nodes (no public IP — uses "sf nodes ssh" CLI proxy)
#   direct      Any VM with a public IP: AWS, GCP, Azure, RunPod, Vast.ai, etc.
#
# Commands:
#   ssh             Open an interactive SSH session on the remote VM
#   sync            Push local code to the remote VM (excludes data/venv/outputs)
#   sync-data       Push data/combined dataset to the remote VM
#   setup           Bootstrap the remote VM (install deps, create venv)
#   push-checkpoint Upload a local .pt checkpoint to outputs/ on the remote VM
#   train-phase1    Run Phase 1 training on the remote VM
#   train-phase2    Run Phase 2 training on the remote VM
#   download        Pull outputs/ checkpoints back to your local machine
#   status          Show running processes and GPU usage on the remote VM
#   logs            Tail the latest training log on the remote VM
#   run <cmd>       Run an arbitrary command on the remote VM
#
# Configuration:
#   Edit compute.env with your provider and node/host details.
#
# Examples:
#   ./scripts/remote.sh ssh
#   ./scripts/remote.sh sync
#   ./scripts/remote.sh sync-data
#   ./scripts/remote.sh setup
#   ./scripts/remote.sh train-phase1
#   ./scripts/remote.sh download
#   ./scripts/remote.sh run "nvidia-smi"
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve project root and load config
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$PROJECT_ROOT/compute.env"

if [[ ! -f "$ENV_FILE" ]]; then
    echo ""
    echo "  ERROR: compute.env not found."
    echo "  Copy the template and fill in your details:"
    echo "    cp compute.env.example compute.env"
    echo ""
    exit 1
fi

# shellcheck source=/dev/null
source "$ENV_FILE"

# Defaults
: "${COMPUTE_PROVIDER:=direct}"
: "${COMPUTE_USER:=root}"
: "${COMPUTE_KEY:=~/.ssh/id_ed25519}"
: "${COMPUTE_PORT:=22}"
: "${REMOTE_DIR:=/root/emotion-detection-action}"
: "${COMPUTE_DEVICE:=cuda}"
: "${TRAIN_NUM_GPUS:=8}"
: "${TRAIN_BATCH_SIZE:=16}"
: "${TRAIN_BATCH_SIZE_PHASE2:=8}"
: "${TRAIN_NUM_WORKERS:=4}"
: "${TRAIN_EPOCHS_PHASE1:=20}"
: "${TRAIN_EPOCHS_PHASE2:=10}"
: "${TRAIN_VIDEO_MODEL:=affectnet_vit}"
: "${TRAIN_AUDIO_MODEL:=emotion2vec}"
: "${TRAIN_UNFREEZE_LAYERS:=4}"
: "${TRAIN_AUGMENT:=true}"
: "${TRAIN_LR_PHASE1:=2e-4}"
: "${TRAIN_SCALE_LR:=false}"
: "${TRAIN_WARMUP_EPOCHS:=2}"
: "${TRAIN_WEIGHT_DECAY:=1e-3}"
: "${TRAIN_CLASS_WEIGHTS:=5.70,35.50,20.50,1.00,1.90,5.50,9.60,20.50}"
: "${LOCAL_DATA_DIR:=./data/combined}"
: "${SKIP_DATA_SYNC:=false}"

COMPUTE_KEY="${COMPUTE_KEY/#\~/$HOME}"

# ---------------------------------------------------------------------------
# Validate provider-specific required fields
# ---------------------------------------------------------------------------

if [[ "$COMPUTE_PROVIDER" == "sfcompute" ]]; then
    : "${COMPUTE_NODE_NAME:?  compute.env: COMPUTE_NODE_NAME is required for COMPUTE_PROVIDER=sfcompute}"

    if ! command -v sf &>/dev/null; then
        echo ""
        echo "  ERROR: SF Compute CLI not found."
        echo "  Install it with:"
        echo "    curl -fsSL https://sfcompute.com/cli/install | bash"
        echo "    source ~/.zshrc"
        echo "    sf login"
        echo ""
        exit 1
    fi
elif [[ "$COMPUTE_PROVIDER" == "direct" ]]; then
    : "${COMPUTE_HOST:?  compute.env: COMPUTE_HOST is required for COMPUTE_PROVIDER=direct}"
else
    echo "  ERROR: Unknown COMPUTE_PROVIDER='$COMPUTE_PROVIDER'. Use 'sfcompute' or 'direct'."
    exit 1
fi

# ---------------------------------------------------------------------------
# SSH / transfer helpers — provider-aware
# ---------------------------------------------------------------------------
#
# SF Compute nodes have no public IPs and the `sf nodes ssh` CLI only accepts
# a destination argument — it cannot pass commands or extra SSH flags.
#
# Workarounds used here:
#   ssh_run()   — pipes the command through stdin (sf nodes ssh reads it as
#                 a non-interactive shell command when stdin is not a tty)
#   push_file() — base64-encodes a tar archive piped through stdin so binary
#                 data is safe through the terminal layer
#   push_dir()  — same base64+tar approach for directories
#   pull_dir()  — pipes a remote tar+base64 command through stdin, captures
#                 the base64-encoded archive from stdout, and decodes locally
# ---------------------------------------------------------------------------

_sf_ssh() {
    # Internal: pipe a shell command string to sf nodes ssh via stdin.
    # The SF CLI detects non-tty stdin and runs non-interactively.
    # Append "exit $?" so the session closes with the correct exit code.
    local cmd="$*"
    printf '%s\nexit $?\n' "$cmd" | sf nodes ssh -q "root@$COMPUTE_NODE_NAME"
}

ssh_run() {
    # Run a non-interactive command on the remote VM
    if [[ "$COMPUTE_PROVIDER" == "sfcompute" ]]; then
        _sf_ssh "$@"
    else
        ssh \
            -i "$COMPUTE_KEY" \
            -p "$COMPUTE_PORT" \
            -o StrictHostKeyChecking=no \
            -o ServerAliveInterval=60 \
            -o ConnectTimeout=15 \
            "$COMPUTE_USER@$COMPUTE_HOST" \
            "$@"
    fi
}

ssh_interactive() {
    # Open an interactive SSH session, or run a command with live output.
    # SF Compute's "sf nodes ssh" only accepts a destination argument (no
    # trailing command), so we pipe any command through stdin instead.
    if [[ "$COMPUTE_PROVIDER" == "sfcompute" ]]; then
        if [[ $# -eq 0 ]]; then
            # No command — open a truly interactive shell
            sf nodes ssh "root@$COMPUTE_NODE_NAME"
        else
            # Run command via stdin so live output streams back to the terminal
            local cmd="$*"
            printf '%s\nexit $?\n' "$cmd" | sf nodes ssh "root@$COMPUTE_NODE_NAME"
        fi
    else
        ssh \
            -i "$COMPUTE_KEY" \
            -p "$COMPUTE_PORT" \
            -o StrictHostKeyChecking=no \
            -o ServerAliveInterval=60 \
            -o ServerAliveCountMax=10 \
            -o ConnectTimeout=15 \
            -t \
            "$COMPUTE_USER@$COMPUTE_HOST" \
            "$@"
    fi
}

push_file() {
    # Upload a single file to the remote VM
    local src="$1"
    local dest="$2"     # full remote path including filename
    local dest_dir
    dest_dir="$(dirname "$dest")"

    if [[ "$COMPUTE_PROVIDER" == "sfcompute" ]]; then
        # Encode as base64 so binary data survives the terminal layer
        local b64
        b64="$(tar -czf - -C "$(dirname "$src")" "$(basename "$src")" | base64)"
        printf 'mkdir -p %s && echo "%s" | base64 -d | tar -xzf - -C %s\nexit $?\n' \
            "$dest_dir" "$b64" "$dest_dir" \
            | sf nodes ssh -q "root@$COMPUTE_NODE_NAME"
    else
        scp \
            -i "$COMPUTE_KEY" \
            -P "$COMPUTE_PORT" \
            -o StrictHostKeyChecking=no \
            "$src" \
            "$COMPUTE_USER@$COMPUTE_HOST:$dest"
    fi
}

push_dir() {
    # Sync a local directory to the remote VM
    local src="$1"        # local path (trailing / = contents only)
    local dest="$2"       # remote path
    shift 2
    local extra_excludes=("$@")

    if [[ "$COMPUTE_PROVIDER" == "sfcompute" ]]; then
        echo "  [sfcompute] Packing and uploading (base64+tar) …"
        local tar_excludes=()
        for ex in "${extra_excludes[@]:-}"; do
            [[ -n "$ex" ]] && tar_excludes+=(--exclude="$ex")
        done
        local src_dir="${src%/}"
        local b64
        b64="$(tar -czf - "${tar_excludes[@]}" -C "$(dirname "$src_dir")" "$(basename "$src_dir")" | base64)"
        printf 'mkdir -p %s && echo "%s" | base64 -d | tar -xzf - --strip-components=1 -C %s\nexit $?\n' \
            "$dest" "$b64" "$dest" \
            | sf nodes ssh -q "root@$COMPUTE_NODE_NAME"
    else
        local rsync_excludes=()
        for ex in "${extra_excludes[@]:-}"; do
            [[ -n "$ex" ]] && rsync_excludes+=(--exclude="$ex")
        done
        rsync -avz --progress \
            -e "ssh -i $COMPUTE_KEY -p $COMPUTE_PORT -o StrictHostKeyChecking=no" \
            "${rsync_excludes[@]}" \
            "$src" \
            "$COMPUTE_USER@$COMPUTE_HOST:$dest"
    fi
}

pull_dir() {
    # Download a remote directory to local
    local remote_src="$1"
    local local_dest="$2"

    mkdir -p "$local_dest"

    if [[ "$COMPUTE_PROVIDER" == "sfcompute" ]]; then
        echo "  [sfcompute] Downloading (streaming base64+tar) …"
        # Stream directly through pipes — never capture into a bash variable.
        # Capturing into $() would load the entire base64-encoded archive into
        # memory, which crashes bash for large checkpoints (hundreds of MB).
        # awk strips any MOTD / login banner lines before the <<EDA_START>> marker.
        printf \
            'echo "<<EDA_START>>"; tar -czf - -C %s %s 2>/dev/null | base64; echo "<<EDA_END>>"\nexit 0\n' \
            "$(dirname "$remote_src")" "$(basename "$remote_src")" \
            | sf nodes ssh -q "root@$COMPUTE_NODE_NAME" \
            | awk '/<<EDA_START>>/{f=1;next} /<<EDA_END>>/{f=0;next} f' \
            | base64 -d \
            | tar -xzf - --strip-components=1 -C "$local_dest"
    else
        rsync -avz --progress \
            -e "ssh -i $COMPUTE_KEY -p $COMPUTE_PORT -o StrictHostKeyChecking=no" \
            "$COMPUTE_USER@$COMPUTE_HOST:$remote_src/" \
            "$local_dest/"
    fi
}

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

cmd_ssh() {
    if [[ "$COMPUTE_PROVIDER" == "sfcompute" ]]; then
        echo "  Connecting to SF Compute node: $COMPUTE_NODE_NAME …"
    else
        echo "  Connecting to $COMPUTE_USER@$COMPUTE_HOST:$COMPUTE_PORT …"
    fi
    ssh_interactive
}

cmd_sync() {
    if [[ "$COMPUTE_PROVIDER" == "sfcompute" ]]; then
        echo "  Syncing code → SF Compute node: $COMPUTE_NODE_NAME:$REMOTE_DIR"
    else
        echo "  Syncing code → $COMPUTE_USER@$COMPUTE_HOST:$REMOTE_DIR"
    fi
    echo "  (Excludes: data/, venv/, outputs/, results/, __pycache__, *.pt)"
    echo ""

    ssh_run "mkdir -p $REMOTE_DIR"

    push_dir "$PROJECT_ROOT/" "$REMOTE_DIR" \
        "data/" "venv/" "outputs/" "results/" \
        "__pycache__/" "*.pyc" "*.pyo" "*.egg-info/" \
        ".git/" ".DS_Store" "*.pt" "*.pth" "compute.env"

    echo ""
    echo "  Sync complete."
}

cmd_sync_data() {
    if [[ "$SKIP_DATA_SYNC" == "true" ]]; then
        echo "  SKIP_DATA_SYNC=true — skipping dataset upload."
        return 0
    fi

    local local_data
    local_data="$(cd "$PROJECT_ROOT" && realpath "$LOCAL_DATA_DIR")"

    if [[ ! -d "$local_data" ]]; then
        echo "  ERROR: Local data directory not found: $local_data"
        echo "  Set LOCAL_DATA_DIR in compute.env or run combine.py first."
        exit 1
    fi

    local file_count tar_size
    file_count=$(find "$local_data" -type f | wc -l | tr -d ' ')
    tar_size=$(du -sh "$local_data" | cut -f1)
    echo "  Dataset: $file_count files, ~$tar_size"
    echo ""

    if [[ "$COMPUTE_PROVIDER" == "sfcompute" ]]; then
        # SF Compute has no public IP so we can't use rsync.
        # Use upload_data.py which opens ONE ssh connection and streams raw
        # binary tar data — much faster than per-chunk connections.
        echo "  Using streaming upload (single SSH connection) …"
        python3 "$SCRIPT_DIR/upload_data.py" \
            "$COMPUTE_NODE_NAME" \
            "$local_data" \
            "$REMOTE_DIR/data/combined"
    else
        echo "  Syncing via rsync …"
        echo "  (First upload takes 15–60 min depending on connection speed)"
        echo ""
        ssh_run "mkdir -p $REMOTE_DIR/data/combined"
        rsync -avz --progress \
            -e "ssh -i $COMPUTE_KEY -p $COMPUTE_PORT -o StrictHostKeyChecking=no" \
            "$local_data/" \
            "$COMPUTE_USER@$COMPUTE_HOST:$REMOTE_DIR/data/combined/"
    fi

    echo ""
    echo "  Dataset sync complete."
}

cmd_setup() {
    echo "  Bootstrapping remote VM …"
    echo ""

    ssh_run "mkdir -p $REMOTE_DIR/scripts"
    push_file "$SCRIPT_DIR/vm_bootstrap.sh" "$REMOTE_DIR/scripts/vm_bootstrap.sh"
    ssh_interactive "chmod +x $REMOTE_DIR/scripts/vm_bootstrap.sh && cd $REMOTE_DIR && bash scripts/vm_bootstrap.sh"

    echo ""
    echo "  Setup complete."
    echo "  Next: ./scripts/remote.sh sync-data"
}

_train_launcher() {
    # Emit the correct launcher prefix:
    #   TRAIN_NUM_GPUS > 1  →  torchrun (single-node DDP via --standalone)
    #   TRAIN_NUM_GPUS = 1  →  python3
    # --standalone is the correct flag for a single physical node; it handles
    # rendezvous automatically and surfaces child-process errors clearly.
    if [ "${TRAIN_NUM_GPUS:-1}" -gt 1 ] 2>/dev/null; then
        echo "torchrun --standalone --nnodes=1 --nproc_per_node=${TRAIN_NUM_GPUS}"
    else
        echo "python3"
    fi
}

cmd_train_phase1() {
    local extra_args="${*:-}"
    local launcher
    launcher="$(_train_launcher)"

    echo "  Starting Phase 1 training …"
    echo "  GPUs: $TRAIN_NUM_GPUS | Launcher: $launcher"
    echo "  Batch/GPU: $TRAIN_BATCH_SIZE | Epochs: $TRAIN_EPOCHS_PHASE1"
    echo ""

    # Build optional flags
    local augment_flag=""
    [[ "$TRAIN_AUGMENT" == "true" ]] && augment_flag="--augment"

    local scale_lr_flag="--no-scale-lr"
    [[ "$TRAIN_SCALE_LR" == "true" ]] && scale_lr_flag="--scale-lr"

    local class_weights_flag=""
    [[ -n "$TRAIN_CLASS_WEIGHTS" ]] && class_weights_flag="--class-weights $TRAIN_CLASS_WEIGHTS"

    ssh_interactive "
        cd $REMOTE_DIR && \
        source venv/bin/activate && \
        mkdir -p outputs && \
        export HF_TOKEN='${HF_TOKEN:-}' && \
        TORCHELASTIC_ERROR_FILE=/tmp/torchrun_phase1_err.json \
        $launcher training/train_phase1.py \
            --data-dir data/combined \
            --pretrained \
            --video-model $TRAIN_VIDEO_MODEL \
            --audio-model $TRAIN_AUDIO_MODEL \
            $augment_flag \
            $scale_lr_flag \
            $class_weights_flag \
            --lr $TRAIN_LR_PHASE1 \
            --warmup-epochs $TRAIN_WARMUP_EPOCHS \
            --weight-decay $TRAIN_WEIGHT_DECAY \
            --batch-size $TRAIN_BATCH_SIZE \
            --num-workers $TRAIN_NUM_WORKERS \
            --epochs $TRAIN_EPOCHS_PHASE1 \
            --output-dir outputs \
            $extra_args \
        2>&1 | tee outputs/train_phase1.log; \
        if [ -s /tmp/torchrun_phase1_err.json ]; then \
            echo '--- torchrun error detail ---'; \
            cat /tmp/torchrun_phase1_err.json; \
        fi
    "
}

cmd_train_phase2() {
    local extra_args="${*:-}"
    local launcher
    launcher="$(_train_launcher)"

    echo "  Starting Phase 2 training …"
    echo "  GPUs: $TRAIN_NUM_GPUS | Launcher: $launcher"
    echo "  Batch/GPU: $TRAIN_BATCH_SIZE_PHASE2 | Epochs: $TRAIN_EPOCHS_PHASE2 | Unfreeze layers: $TRAIN_UNFREEZE_LAYERS"
    echo ""

    # Build optional flags
    local scale_lr_flag="--no-scale-lr"
    [[ "$TRAIN_SCALE_LR" == "true" ]] && scale_lr_flag="--scale-lr"

    local augment_flag=""
    [[ "$TRAIN_AUGMENT" == "true" ]] && augment_flag="--augment"

    local class_weights_flag=""
    [[ -n "$TRAIN_CLASS_WEIGHTS" ]] && class_weights_flag="--class-weights $TRAIN_CLASS_WEIGHTS"

    ssh_interactive "
        cd $REMOTE_DIR && \
        source venv/bin/activate && \
        mkdir -p outputs && \
        export HF_TOKEN='${HF_TOKEN:-}' && \
        TORCHELASTIC_ERROR_FILE=/tmp/torchrun_phase2_err.json \
        $launcher training/train_phase2.py \
            --checkpoint outputs/phase1_best.pt \
            --data-dir data/combined \
            --pretrained \
            --video-model $TRAIN_VIDEO_MODEL \
            --audio-model $TRAIN_AUDIO_MODEL \
            --unfreeze-layers $TRAIN_UNFREEZE_LAYERS \
            $scale_lr_flag \
            $augment_flag \
            $class_weights_flag \
            --warmup-epochs $TRAIN_WARMUP_EPOCHS \
            --weight-decay $TRAIN_WEIGHT_DECAY \
            --batch-size $TRAIN_BATCH_SIZE_PHASE2 \
            --num-workers $TRAIN_NUM_WORKERS \
            --epochs $TRAIN_EPOCHS_PHASE2 \
            --output-dir outputs \
            $extra_args \
        2>&1 | tee outputs/train_phase2.log; \
        if [ -s /tmp/torchrun_phase2_err.json ]; then \
            echo '--- torchrun error detail ---'; \
            cat /tmp/torchrun_phase2_err.json; \
        fi
    "
}

cmd_push_checkpoint() {
    local local_pt="${1:-}"
    if [[ -z "$local_pt" ]]; then
        # Default to phase1_best.pt if no argument given
        local_pt="$PROJECT_ROOT/outputs/phase1_best.pt"
    fi

    if [[ ! -f "$local_pt" ]]; then
        echo "  ERROR: File not found: $local_pt"
        echo "  Usage: ./scripts/remote.sh push-checkpoint [path/to/checkpoint.pt]"
        exit 1
    fi

    local filename
    filename="$(basename "$local_pt")"
    local remote_dest="$REMOTE_DIR/outputs/$filename"

    local size
    size="$(du -sh "$local_pt" | cut -f1)"
    echo "  Uploading checkpoint: $filename ($size) → remote:outputs/"
    echo "  (This may take a few minutes over the SSH proxy)"
    echo ""

    ssh_run "mkdir -p $REMOTE_DIR/outputs"
    push_file "$local_pt" "$remote_dest"

    echo ""
    echo "  Upload complete: $remote_dest"
}

cmd_download() {
    echo "  Downloading outputs/ from remote VM …"
    echo ""

    mkdir -p "$PROJECT_ROOT/outputs"
    pull_dir "$REMOTE_DIR/outputs" "$PROJECT_ROOT/outputs"

    echo ""
    echo "  Downloaded to: $PROJECT_ROOT/outputs/"
    ls -lh "$PROJECT_ROOT/outputs/" 2>/dev/null || true
}

cmd_status() {
    echo "  === Remote VM Status ==="
    echo ""
    echo "  [GPU]"
    ssh_run "nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu \
        --format=csv,noheader 2>/dev/null || echo '  No NVIDIA GPU / nvidia-smi not found.'"
    echo ""
    echo "  [Running Python processes]"
    ssh_run "ps aux | grep '[p]ython' | awk '{print \$2, \$3, \$4, \$11, \$12, \$13}' || echo '  None.'"
    echo ""
    echo "  [Disk usage in $REMOTE_DIR]"
    ssh_run "du -sh $REMOTE_DIR/* 2>/dev/null | sort -h || true"
}

cmd_logs() {
    echo "  Tailing training logs … (Ctrl+C to stop)"
    echo ""
    ssh_run "tail -f \
        $REMOTE_DIR/outputs/train_phase1.log \
        $REMOTE_DIR/outputs/train_phase2.log 2>/dev/null \
        || echo 'No log files found. Start training first.'"
}

cmd_run() {
    if [[ $# -eq 0 ]]; then
        echo "  Usage: ./scripts/remote.sh run <command>"
        exit 1
    fi
    echo "  Running on remote: $*"
    echo ""
    ssh_run "cd $REMOTE_DIR && $*"
}

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

print_help() {
    grep '^#' "$0" | head -30 | sed 's/^# \?//'
}

COMMAND="${1:-help}"
shift || true

case "$COMMAND" in
    ssh)           cmd_ssh "$@" ;;
    sync)          cmd_sync "$@" ;;
    sync-data)     cmd_sync_data "$@" ;;
    setup)         cmd_setup "$@" ;;
    push-checkpoint) cmd_push_checkpoint "$@" ;;
    train-phase1)  cmd_train_phase1 "$@" ;;
    train-phase2)  cmd_train_phase2 "$@" ;;
    download)      cmd_download "$@" ;;
    status)        cmd_status "$@" ;;
    logs)          cmd_logs "$@" ;;
    run)           cmd_run "$@" ;;
    help|--help|-h) print_help ;;
    *)
        echo "  Unknown command: $COMMAND"
        echo "  Run './scripts/remote.sh help' for usage."
        exit 1
        ;;
esac
