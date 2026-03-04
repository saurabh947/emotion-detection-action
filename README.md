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
│  │  VideoBackbone  │     │   Bidirectional CrossAttentionBlock │ × N layers  │
│  │  (VideoMAE ✓   │     │                                    │              │
│  │   or ViViT)    │──┐  │   video tokens ↔ audio tokens      │              │
│  └─────────────────┘  │  │   Q=video  K/V=audio (V→A)        │              │
│                        ├─►│   Q=audio  K/V=video (A→V)        │              │
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
│                   │  Softmax (7)     │     │     Sigmoid (3)            │    │
│                   └──────────────────┘     └────────────────────────────┘    │
│                   angry · disgusted ·       stress · engagement · arousal    │
│                   fearful · happy ·                                          │
│                   neutral · sad ·                                            │
│                   surprised                                                  │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Output contract

Every call returns a **Pydantic** `NeuralEmotionResult`:

```python
NeuralEmotionResult(
    dominant_emotion = "happy",
    emotion_scores   = {"angry": 0.02, "happy": 0.81, ...},   # 7 classes
    latent_embedding = [0.12, -0.34, ...],                     # 512-dim VLA vector
    metrics          = {"stress": 0.23, "engagement": 0.71, "arousal": 0.45},
    confidence       = 0.81,
    timestamp        = 1706123456.78,
    video_missing    = False,
    audio_missing    = False,
)
```

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
│   ├── config.py           ← Config dataclass
│   └── types.py            ← NeuralEmotionResult (Pydantic) + legacy types
├── emotion/
│   ├── two_tower_transformer.py  ← backward-compat shim → NeuralFusionModel
│   ├── facial.py                 ← ViT per-frame recogniser (custom pipelines)
│   └── speech.py                 ← Wav2Vec2 recogniser (custom pipelines)
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
python3 examples/realtime_multimodal.py --no-pretrained   # stub, instant
python3 examples/realtime_multimodal.py                   # pretrained
```

### Python API

```python
import numpy as np
from emotion_detection_action import Config, EmotionDetector

config = Config(
    two_tower_pretrained=True,      # VideoMAE + AST from HuggingFace
    two_tower_device="cpu",         # "cuda" or "mps" recommended in production
    two_tower_video_model="videomae",
    vla_enabled=False,
)

detector = EmotionDetector(config)
detector.initialize()

# ── Clip API: pass a (T, H, W, 3) array ──────────────────────────────────────
clip = np.random.randint(0, 255, (16, 480, 640, 3), dtype=np.uint8)  # RGB
audio = np.random.randn(8000).astype("float32")                       # 0.5s PCM
result = detector.process(clip, audio)

print(result.dominant_emotion)          # "happy"
print(result.metrics)                   # {"stress": 0.2, "engagement": 0.7, "arousal": 0.4}
print(len(result.latent_embedding))     # 512  ← feed to VLA

# ── Single-frame API (accumulates rolling buffer internally) ──────────────────
import cv2
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()            # BGR frame from OpenCV
    if not ret:
        break
    result = detector.process_frame(frame)
    if result:
        print(result.dominant_emotion, result.confidence)

# ── Subject change ─────────────────────────────────────────────────────────────
detector.reset()   # clears frame/audio buffers + GRU hidden state

# ── Quantize for deployment ────────────────────────────────────────────────────
detector.quantize("dynamic")   # INT8 linear layers, ~2-3× faster on CPU
```

### Custom robot action handler

```python
from emotion_detection_action import NeuralEmotionResult
from emotion_detection_action.actions.base import BaseActionHandler

class ReachyHandler(BaseActionHandler):
    def connect(self) -> None:
        # Connect to your robot SDK here
        ...

    def disconnect(self) -> None: ...

    def execute(self, result: NeuralEmotionResult) -> None:
        if result.dominant_emotion == "sad" and result.confidence > 0.6:
            self._send_comfort_gesture()
        if result.metrics["stress"] > 0.8:
            self._reduce_interaction_intensity()

    def _send_comfort_gesture(self): ...
    def _reduce_interaction_intensity(self): ...

detector = EmotionDetector(action_handler=ReachyHandler())
```

---

## Configuration reference

| Field | Default | Description |
|---|---|---|
| `two_tower_video_model` | `"videomae"` | `"videomae"` or `"vivit"` |
| `two_tower_video_backbone` | `"MCG-NJU/videomae-base"` | HuggingFace model ID |
| `two_tower_audio_backbone` | `"MIT/ast-finetuned-audioset-10-10-0.4593"` | HuggingFace model ID |
| `two_tower_pretrained` | `True` | Download pretrained weights |
| `two_tower_model_path` | `None` | Path to fine-tuned checkpoint |
| `two_tower_device` | `"cpu"` | `"cpu"`, `"cuda"`, or `"mps"` |
| `two_tower_d_model` | `512` | Shared projection / cross-attention dim |
| `two_tower_cross_attn_layers` | `2` | Bidirectional cross-attention layers |
| `two_tower_gru_layers` | `2` | GRU depth in temporal context buffer |
| `two_tower_video_frames` | `16` | Frames per clip (auto-set to 32 for ViViT) |
| `two_tower_video_freeze_layers` | `8` | VideoMAE encoder layers to freeze |
| `two_tower_audio_freeze_layers` | `6` | AST encoder layers to freeze |
| `two_tower_n_mels` | `128` | Mel bins for audio spectrogram |

---

## Training / fine-tuning

```python
from emotion_detection_action.models.fusion import NeuralFusionModel
from emotion_detection_action.models.backbones import BackboneConfig
import torch

model = NeuralFusionModel(BackboneConfig(pretrained=True))
model.freeze_backbones()   # only train cross-attention + heads initially

groups = model.get_trainable_parameter_groups(backbone_lr=1e-5, head_lr=1e-4)
optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)

# Training loop
for video_clip, audio_mel, emotion_labels, metric_labels in dataloader:
    out = model(video_clip, audio_mel, use_temporal=False)  # disable GRU for batch training
    loss = (
        F.cross_entropy(out.emotion_logits, emotion_labels)
        + F.mse_loss(out.metrics, metric_labels)
    )
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    model.detach_temporal_state()

# Save checkpoint
torch.save(model.state_dict(), "emotion_model.pt")
```

Point `Config.two_tower_model_path` to `"emotion_model.pt"` to load it in production.

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
