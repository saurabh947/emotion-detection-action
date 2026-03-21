# Emotion Detection Action SDK

A **100% pure-neural, platform-agnostic** multimodal emotion detection SDK
for robotics, built on a Two-Tower Multimodal Transformer.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     NeuralFusionModel  (models/fusion.py)                    │
│                                                                              │
│  ┌─────────────────┐     ┌────────────────────────────────────┐              │
│  │  VideoBackbone  │     │  Bidirectional CrossAttentionBlock │ × N layers   │
│  │  (VideoMAE ✓    │     │                                    │              │
│  │   or ViViT)     │──┐  │   video tokens ↔ audio tokens      │              │
│  └─────────────────┘  │  │   Q=video  K/V=audio (V→A)         │              │
│                       ├─►│   Q=audio  K/V=video (A→V)         │              │
│  ┌─────────────────┐  │  └────────────────┬───────────────────┘              │
│  │  AudioBackbone  │──┘                   │ mean-pool                        │
│  │  (AST)          │              ┌───────▼───────┐                          │
│  └─────────────────┘              │  Fused CLS    │  (B, d_model=512)        │
│                                   └───────┬───────┘                          │
│                                           │                                  │
│                              ┌────────────▼────────────┐                     │
│                              │  TemporalContextBuffer  │  GRU (2 layers)     │
│                              │  ~2-second rolling      │  prevents flicker   │
│                              │  window                 │                     │
│                              └────────────┬────────────┘                     │
│                                           │  latent_embedding (512-dim)      │
│                              ┌────────────┴────────────┐                     │
│                              ▼                         ▼                     │
│                   ┌──────────────────┐     ┌────────────────────────────┐    │
│                   │  Emotion head    │     │     Metrics head           │    │
│                   │  Softmax (8)     │     │     Sigmoid (3)            │    │
│                   └──────────────────┘     └────────────────────────────┘    │
│                   angry · disgusted ·       stress · engagement · arousal    │
│                   fearful · happy ·                                          │
│                   neutral · sad ·                                            │
│                   surprised · unclear                                        │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Output contract

Every call returns a **Pydantic** `NeuralEmotionResult`:

```python
NeuralEmotionResult(
    dominant_emotion = "happy",
    emotion_scores   = {"angry": 0.02, "happy": 0.81, ..., "unclear": 0.01},  # 8 classes
    latent_embedding = [0.12, -0.34, ...],                     # 512-dim VLA vector
    metrics          = {"stress": 0.23, "engagement": 0.71, "arousal": 0.45},
    confidence       = 0.81,
    timestamp        = 1706123456.78,
    video_missing    = False,
    audio_missing    = False,
)
```

> **Tip:** Always gate on the `"unclear"` label before acting.  The model
> predicts `"unclear"` when no person is present, the signal is too noisy, or
> confidence is low.
>
> ```python
> if result.dominant_emotion != "unclear":
>     robot.react_to(result.dominant_emotion)
> ```

The `latent_embedding` is a 512-dim GRU-smoothed fused vector — feed this
directly into VLA models (e.g., OpenVLA) as the emotion context.

---

## Video backbone comparison: VideoMAE vs ViViT

| Property | **VideoMAE** ✓ recommended | ViViT-B/16x2 |
|---|---|---|
| HuggingFace ID | `MCG-NJU/videomae-base` | `google/vivit-b-16x2-kinetics400` |
| Clip length | **16 frames** | 32 frames |
| Pretraining | Masked Autoencoding (MAE) | Supervised (Kinetics-400) |
| Parameters | 87 M | 86 M |
| CPU latency (1 clip) | **~300 ms** | ~580 ms |
| Fine-tuning on small data | **Excellent** | Good |
| Latent space quality | **High** (great for VLA) | Moderate |
| Attention window @ 30fps | **0.53 s** | 1.07 s |

**Why VideoMAE?**

1. Shorter clip → **~2× faster inference**, lower real-time latency.
2. MAE pretraining produces richer internal representations — the `latent_embedding`
   captures more nuanced emotional features for VLA integration.
3. Emotion datasets are small; MAE-pretrained models transfer significantly
   better than supervised ones to small target datasets (AffectNet, RAVDESS).

---

## Code structure

```
src/emotion_detection_action/
├── models/
│   ├── backbones.py        ← VideoBackbone (VideoMAE/ViViT) + AudioBackbone (AST)
│   ├── fusion.py           ← NeuralFusionModel · CrossAttentionBlock · TemporalContextBuffer
│   └── vla/                ← OpenVLA integration (optional)
├── core/
│   ├── detector.py         ← EmotionDetector (pure-neural, platform-agnostic)
│   ├── inference_worker.py ← InferenceWorker · WorkerStats (background thread + queue)
│   ├── config.py           ← Config dataclass
│   └── types.py            ← NeuralEmotionResult (Pydantic) · ActionCommand · EmotionLabel
├── actions/
│   └── base.py             ← BaseActionHandler (subclass for your robot)
└── inputs/
    ├── video.py            ← Camera capture helper
    └── audio.py            ← Microphone capture helper
```

---

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip3 install -e ".[dev]"
```

Core dependencies: `torch`, `torchaudio`, `transformers`, `pydantic>=2`, `numpy`, `opencv-python`.

---

## Quick start

### Architecture test (no download, stub weights)

```bash
python3 src/emotion_detection_action/models/fusion.py
```

### 30fps stream simulation

```bash
# Stub backbones — instant, no download required
python3 examples/neural_stream_demo.py

# Pretrained backbones (downloads ~1.8 GB VideoMAE + AST weights once)
python3 examples/neural_stream_demo.py --pretrained

# INT8 quantization for low-latency deployment
python3 examples/neural_stream_demo.py --quantize

# ViViT backbone instead of VideoMAE
python3 examples/neural_stream_demo.py --video-model vivit --pretrained
```

### Real-time webcam demo (OpenCV visualisation)

```bash
# Stub backbones — instant, no download
python3 examples/neural_stream_demo.py --webcam

# Pretrained backbones (downloads ~1.8 GB once)
python3 examples/neural_stream_demo.py --webcam --pretrained

# With your fine-tuned checkpoint
python3 examples/neural_stream_demo.py --webcam --pretrained --model-path outputs/phase2_best.pt
```

### Python API

```python
from emotion_detection_action import Config, EmotionDetector
import numpy as np

detector = EmotionDetector(Config(
    two_tower_pretrained=True,   # downloads VideoMAE + AST weights once
    two_tower_device="cpu",      # "cuda" or "mps" for GPU
    vla_enabled=False,
))
detector.initialize()

# Pass a (T, H, W, 3) RGB clip + optional raw PCM audio
clip  = np.random.randint(0, 255, (16, 480, 640, 3), dtype=np.uint8)
audio = np.random.randn(8000).astype("float32")   # 0.5 s @ 16 kHz
result = detector.process(clip, audio)

print(result.dominant_emotion)   # e.g. "happy" — "unclear" when no person / low confidence
print(result.confidence)         # 0.0 – 1.0
print(result.metrics)            # {"stress": …, "engagement": …, "arousal": …}
print(result.latent_embedding)   # 512-dim vector → feed directly to a VLA model
```

> For frame-by-frame webcam use, call `detector.process_frame(bgr_frame)` in a loop —
> it accumulates a rolling buffer internally.  See `examples/neural_stream_demo.py`.

### Custom robot action handler

```python
from emotion_detection_action.actions.base import BaseActionHandler
from emotion_detection_action.core.types import ActionCommand, NeuralEmotionResult

class ReachyHandler(BaseActionHandler):
    def connect(self) -> bool:    ...   # open connection to your robot SDK
    def disconnect(self) -> None: ...

    def execute(self, action: ActionCommand) -> bool:
        # action.action_type is set by the SDK's default emotion→action mapping.
        # Override react() instead if you want to work with NeuralEmotionResult directly.
        ...

detector = EmotionDetector(Config(...), action_handler=ReachyHandler())
```

---

## Configuration reference

| Field | Default | Description |
|---|---|---|
| `two_tower_video_model` | `"videomae"` | `"videomae"` or `"vivit"` |
| `two_tower_video_backbone` | `"MCG-NJU/videomae-base"` | HuggingFace model ID |
| `two_tower_audio_backbone` | `"MIT/ast-finetuned-audioset-10-10-0.4593"` | HuggingFace model ID |
| `two_tower_pretrained` | `True` | Download pretrained HuggingFace weights |
| `two_tower_model_path` | `None` | Path to a fine-tuned `NeuralFusionModel` checkpoint (e.g. `outputs/phase2_best.pt`) |
| `two_tower_device` | `"cpu"` | `"cpu"`, `"cuda"`, or `"mps"` |
| `two_tower_d_model` | `512` | Shared projection / cross-attention dim |
| `two_tower_cross_attn_layers` | `2` | Bidirectional cross-attention layers |
| `two_tower_gru_layers` | `2` | GRU depth in temporal context buffer |
| `two_tower_video_frames` | `16` | Frames per clip (auto-set to 32 for ViViT) |
| `two_tower_video_freeze_layers` | `8` | VideoMAE encoder layers to freeze |
| `two_tower_audio_freeze_layers` | `6` | AST encoder layers to freeze |
| `two_tower_n_mels` | `128` | Mel bins for audio spectrogram |
| `two_tower_n_fft` | `400` | FFT window size for mel spectrogram |
| `two_tower_hop_length` | `160` | Hop length for mel spectrogram (10 ms @ 16 kHz) |

---

## Training / fine-tuning

Two-phase training with scripts in `training/`.  Place labelled video clips in:

```
data/combined/{emotion}/   ← one folder per class (angry, happy, …, unclear)
```

Supported sources: **RAVDESS**, **CREMA-D**, **MELD**.

### Phase 1 — freeze backbones, train heads

```bash
python3 training/train_phase1.py --data-dir data/combined --epochs 20 --batch-size 4
# Multi-GPU: torchrun --standalone --nproc_per_node=N training/train_phase1.py ...
```

### Phase 2 — unfreeze top layers, fine-tune end-to-end

```bash
python3 training/train_phase2.py --data-dir data/combined --epochs 10 --no-scale-lr
# Automatically loads outputs/phase1_best.pt; saves outputs/phase2_best.pt
```

### Load a fine-tuned checkpoint

```python
detector = EmotionDetector(Config(
    two_tower_pretrained=True,
    two_tower_model_path="outputs/phase2_best.pt",
    two_tower_device="cpu",
))
detector.initialize()
```

```bash
python3 training/reset_model.py   # factory reset — deletes outputs/ checkpoints
```

---

## Missing-modality handling

| Scenario | Behaviour |
|---|---|
| Both modalities | Full cross-attention fusion |
| `video_frames=None` | Learned absent-video token substituted |
| `audio_spectrograms=None` | Learned absent-audio token substituted |
| Both `None` | `ValueError` raised |

---

## Quantization

```python
# After initialize(), before streaming:
detector.quantize("dynamic")   # returns INT8-quantized model internally
                               # Platform: qnnpack (ARM/macOS), fbgemm (x86)
```

Expected gains on CPU:
- **~2–3× throughput increase** for cross-attention layers
- **~4× memory reduction** for linear weights
- Negligible accuracy loss (< 0.5% on held-out emotion benchmarks)

---

## License

MIT
