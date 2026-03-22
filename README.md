# Emotion Detection Action SDK

A **100% pure-neural, platform-agnostic** multimodal emotion detection SDK
for robotics, built on a Two-Tower Multimodal Transformer.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                     NeuralFusionModel  (models/fusion.py)                      │
│                                                                                │
│  ┌──────────────────────┐   ┌──────────────────────────────────────────┐       │
│  │     VideoBackbone    │   │  VideoTemporalBlock  (intra-clip self-   │       │
│  │  AffectNet ViT ✓     │──►│  attention + positional encoding over T  │       │
│  │  (trpakov/vit-face-  │   │  per-frame CLS tokens)                   │       │
│  │   expression)        │   └──────────────────┬───────────────────────┘       │
│  └──────────────────────┘                      │                               │
│  Face crop per frame ──────────────────────────┘                               │
│  (MediaPipe, auto-applied)                     ├─┐                             │
│                                                │ │  ┌────────────────────────┐ │
│  ┌──────────────────────┐                      │ └─►│  Bidirectional Cross   │ │
│  │     AudioBackbone    │                      │    │  AttentionBlock × N    │ │
│  │  emotion2vec ✓       │──────────────────────┘    │  video ↔ audio         │ │
│  │  (iic/emotion2vec_   │   raw waveform (16 kHz)   └──────────┬─────────────┘ │
│  │   base via FunASR)   │                                      │ mean-pool     │
│  └──────────────────────┘                              ┌───────▼─────────┐     │
│                                                        │   Fused CLS     │     │
│                                                        │ (B, d_model=512)│     │
│                                                        └────────┬────────┘     │
│                                              ┌──────────────────▼──────┐       │
│                                              │  TemporalContextBuffer  │       │
│                                              │  GRU (2 layers) — 2-s   │       │
│                                              │  inter-clip rolling ctx │       │
│                                              └──────────┬──────────────┘       │
│                                                         │  latent_embedding    │
│                                         ┌───────────────┴───────────┐          │
│                                         ▼                           ▼          │
│                              ┌──────────────────┐   ┌────────────────────┐     │
│                              │  Emotion head    │   │   Metrics head     │     │
│                              │  Softmax (8)     │   │   Sigmoid (3)      │     │
│                              └──────────────────┘   └────────────────────┘     │
│                     angry · disgusted · fearful · happy · neutral · sad ·      │
│                     surprised · unclear        stress · engagement · arousal   │
└────────────────────────────────────────────────────────────────────────────────┘
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

## Backbone selection

### Video — AffectNet ViT (default, recommended)

| Property | **AffectNet ViT** ✓ default | VideoMAE (legacy) | ViViT (legacy) |
|---|---|---|---|
| HuggingFace ID | `trpakov/vit-face-expression` | `MCG-NJU/videomae-base` | `google/vivit-b-16x2-kinetics400` |
| Pre-training task | **Emotion classification** on 450 K faces | Masked autoencoding | Action recognition |
| Input | Per-frame face crops (auto-applied) | 16-frame clip | 32-frame clip |
| Output | `(B, T, 768)` — 1 CLS per frame | `(B, ~1568, 768)` patch tokens | `(B, ~3137, 768)` |
| Domain fit | **Direct** — trained on emotion | Indirect | Indirect |
| Requires face crop | Yes (MediaPipe, automatic) | No | No |

### Audio — emotion2vec (default, recommended)

| Property | **emotion2vec** ✓ default | AST (legacy) |
|---|---|---|
| Loader | FunASR (required: `funasr`, `modelscope`) | HuggingFace transformers |
| Pre-training task | **Speech emotion** (IEMOCAP, MSP, RAVDESS, CREMA-D) | AudioSet sound classification |
| Input | `(B, samples)` raw 16 kHz waveform | `(B, time, mel)` mel-spectrogram |
| Output | `(B, T_a, 768)` frame features | `(B, T_a, 768)` CLS + patch tokens |

emotion2vec is a **required** dependency. `funasr` and `modelscope` are installed
automatically with the SDK — no extra steps needed.

---

## Code structure

```
src/emotion_detection_action/
├── models/
│   ├── backbones.py        ← VideoBackbone (AffectNet ViT/VideoMAE/ViViT)
│   │                          AudioBackbone (emotion2vec/AST)
│   │                          FaceCropPipeline (MediaPipe face detection)
│   ├── fusion.py           ← NeuralFusionModel · VideoTemporalBlock
│   │                          CrossAttentionBlock · TemporalContextBuffer
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

Core dependencies: `torch`, `torchaudio`, `transformers`, `pydantic>=2`, `numpy`, `opencv-python`, `mediapipe`, `funasr`, `modelscope`.

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

# Pretrained backbones (downloads AffectNet ViT + emotion2vec weights)
python3 examples/neural_stream_demo.py --pretrained

# INT8 quantization for low-latency deployment
python3 examples/neural_stream_demo.py --quantize

# Legacy VideoMAE + AST backbone
python3 examples/neural_stream_demo.py --video-model videomae --audio-model ast --pretrained
```

### Real-time webcam demo (OpenCV visualisation)

```bash
# Stub backbones — instant, no download
python3 examples/neural_stream_demo.py --webcam

# Pretrained backbones + face crop (best accuracy)
python3 examples/neural_stream_demo.py --webcam --pretrained

# With your fine-tuned checkpoint
python3 examples/neural_stream_demo.py --webcam --pretrained --model-path outputs/phase2_best.pt
```

### Python API

```python
from emotion_detection_action import Config, EmotionDetector
import numpy as np

detector = EmotionDetector(Config(
    two_tower_pretrained=True,     # downloads AffectNet ViT + emotion2vec weights
    two_tower_device="cpu",        # "cuda" or "mps" for GPU
    two_tower_face_crop_enabled=True,  # auto face-crop per frame (MediaPipe)
    vla_enabled=False,
))
detector.initialize()

# Pass a (T, H, W, 3) RGB clip + optional raw PCM audio (16 kHz)
clip  = np.random.randint(0, 255, (16, 480, 640, 3), dtype=np.uint8)
audio = np.random.randn(16000 * 3).astype("float32")  # 3 s @ 16 kHz
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
| `two_tower_video_model` | `"affectnet_vit"` | `"affectnet_vit"` (best), `"videomae"`, or `"vivit"` |
| `two_tower_audio_model` | `"emotion2vec"` | `"emotion2vec"` (best) or `"ast"` (legacy) |
| `two_tower_video_backbone` | `"trpakov/vit-face-expression"` | HuggingFace model ID |
| `two_tower_audio_backbone` | `"iic/emotion2vec_base"` | FunASR / HuggingFace model ID |
| `two_tower_pretrained` | `True` | Download pretrained weights |
| `two_tower_model_path` | `None` | Path to fine-tuned checkpoint (e.g. `outputs/phase2_best.pt`) |
| `two_tower_device` | `"cpu"` | `"cpu"`, `"cuda"`, or `"mps"` |
| `two_tower_d_model` | `512` | Shared projection / cross-attention dim |
| `two_tower_cross_attn_layers` | `2` | Bidirectional cross-attention layers |
| `two_tower_gru_layers` | `2` | GRU depth in inter-clip temporal buffer |
| `two_tower_video_frames` | `16` | Frames per clip (auto-set to 32 for ViViT) |
| `two_tower_video_freeze_layers` | `6` | Video encoder layers to freeze during training |
| `two_tower_audio_freeze_layers` | `6` | Audio encoder layers to freeze during training |
| `two_tower_face_crop_enabled` | `True` | Auto face-crop via MediaPipe (AffectNet ViT only) |
| `two_tower_face_crop_margin` | `0.2` | Fractional bbox expansion before crop |
| `two_tower_face_min_confidence` | `0.5` | MediaPipe min face confidence (falls back to centre crop) |
| `two_tower_emotion_class_weights` | `None` | 8-element list for `CrossEntropyLoss(weight=...)` — corrects AffectNet class imbalance |
| `two_tower_n_mels` | `128` | Mel bins (AST legacy only) |
| `two_tower_n_fft` | `400` | FFT window (AST legacy only) |
| `two_tower_hop_length` | `160` | Hop length (AST legacy only) |

**AffectNet class-imbalance correction** (no custom code needed):

```python
from emotion_detection_action.core.config import _DEFAULT_AFFECTNET_CLASS_WEIGHTS

cfg = Config(two_tower_emotion_class_weights=_DEFAULT_AFFECTNET_CLASS_WEIGHTS)
# These weights are then passed to nn.CrossEntropyLoss(weight=...) during training.
# EMOTION_ORDER = ["angry","disgusted","fearful","happy","neutral","sad","surprised","unclear"]
# Weights  ≈    [5.70,    35.50,     20.50,    1.00,   1.90,   5.50,   9.60,     20.50]
```

---

## Training / fine-tuning

Two-phase training with scripts in `training/`.  Place labelled video clips in:

```
data/combined/{emotion}/   ← one folder per class (angry, happy, …, unclear)
```

Supported sources: **RAVDESS**, **CREMA-D**, **MELD**.

### Phase 1 — freeze backbones, train heads

```bash
# AffectNet ViT + emotion2vec (recommended, with class-imbalance correction)
python3 training/train_phase1.py \
    --data-dir data/combined --pretrained --epochs 20 --batch-size 4 \
    --class-weights "5.70,35.50,20.50,1.00,1.90,5.50,9.60,20.50"

# Multi-GPU:
torchrun --standalone --nproc_per_node=N training/train_phase1.py \
    --data-dir data/combined --pretrained --batch-size 16 \
    --class-weights "5.70,35.50,20.50,1.00,1.90,5.50,9.60,20.50"
```

### Phase 2 — unfreeze top layers, fine-tune end-to-end

```bash
python3 training/train_phase2.py \
    --checkpoint outputs/phase1_best.pt \
    --data-dir data/combined --pretrained --epochs 10 --no-scale-lr \
    --class-weights "5.70,35.50,20.50,1.00,1.90,5.50,9.60,20.50"
# Saves outputs/phase2_best.pt
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
| `audio=None` | Learned absent-audio token substituted |
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
