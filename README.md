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

### Missing-modality handling

| Scenario | Behaviour |
|---|---|
| Both modalities | Full cross-attention fusion |
| `video_frames=None` | Learned absent-video token substituted |
| `audio=None` | Learned absent-audio token substituted |
| Both `None` | `ValueError` raised |

---

## Layer breakdown

| Component | Layers | What they do |
|---|---|---|
| **Video Tower** — AffectNet ViT-B/16 | 12 transformer encoder layers | Processes each face-cropped frame independently → one 768-dim CLS token per frame |
| **Audio Tower** — emotion2vec (data2vec-base) | 12 transformer encoder layers | Extracts speech emotion features → 768-dim frame tokens; **always frozen** (FunASR, no gradient exposed) |
| **Video projection** | 1 (Linear + LayerNorm) | Maps ViT 768-dim output → shared 512-dim embedding space |
| **Audio projection** | 1 (Linear + LayerNorm) | Maps emotion2vec 768-dim output → shared 512-dim embedding space |
| **VideoTemporalBlock** | 1 (pos-embed + MHA + LayerNorm) | Injects frame-order positional encoding; lets video frame tokens attend to each other before cross-modal fusion |
| **CrossAttentionBlock × 2** | 2 (each: 2 MHA + 2 FFN + 4 LayerNorm) | Bidirectional video ↔ audio cross-attention — each tower queries the other |
| **TemporalContextBuffer** | 1 (2-layer GRU) | 2-second inter-clip rolling memory; hidden state persists across clips during live inference |
| **Emotion head** | 1 (Linear → softmax) | 512-dim fused vector → 8-class emotion probabilities |
| **Metrics head** | 1 (2× Linear + GELU + Sigmoid) | 512-dim fused vector → stress, engagement, arousal |
| **Total** | **32** | 24 pretrained backbone layers + 8 custom fusion/head layers |

### Frozen vs. trainable per phase

| | Phase 1 | Phase 2 |
|---|---|---|
| **Video ViT layers 0–7** (bottom 2/3) | Frozen | Frozen |
| **Video ViT layers 8–11** (top 1/3) | Frozen | **Trainable** (`lr=1e-5`) |
| **Audio emotion2vec** (all 12 layers) | Frozen | Frozen |
| **Projections, VideoTemporalBlock, CrossAttentionBlocks, GRU, heads** | **Trainable** (`lr=1e-4`) | **Trainable** (`lr=1e-4`) |
| **Trainable / Total** | **8 / 32** | **12 / 32** |
| **Frozen / Total** | **24 / 32** | **20 / 32** |

> Phase 2 defaults to unfreezing the top 4 ViT layers (`--unfreeze-layers 4`). Pass `--unfreeze-layers 6` for larger datasets.

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

## Backbone selection

The default video backbone is `trpakov/vit-face-expression` (AffectNet ViT-B/16), pre-trained directly on 450K emotion-labelled faces — `VideoMAE` (`MCG-NJU/videomae-base`) and `ViViT` (`google/vivit-b-16x2-kinetics400`) are supported as legacy drop-in swaps via `two_tower_video_model`. The default audio backbone is `iic/emotion2vec_base` (loaded via FunASR), pre-trained on speech emotion corpora (IEMOCAP, MSP-Podcast, RAVDESS, CREMA-D) — the legacy `AST` (`MIT/ast-finetuned-audioset-10-10-0.4593`) can be swapped in via `two_tower_audio_model="ast"`. Note that emotion2vec is always frozen regardless of phase, as FunASR does not expose gradient flow.

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

## License

MIT
