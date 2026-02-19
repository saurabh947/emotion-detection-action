# Emotion Detection Action SDK

Real-time human emotion detection SDK for robotics. Uses AI models for facial, speech, and attention analysis, with ML-based fusion that works out-of-the-box.

## Get Started in 60 Seconds

```bash
# Clone and install
git clone https://github.com/saurabh947/emotion-detection-action.git
cd emotion-detection-action
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e .

# Run the real-time demo (requires webcam + microphone)
python examples/realtime_multimodal.py
```

Press **ESC** or **Q** to quit the demo.

## Features

- **Real-time emotion detection**: Live webcam + microphone processing
- **Face detection**: Automatic face detection using MediaPipe (fast and lightweight)
- **Voice activity detection**: Robust speech detection using Silero VAD (deep learning, noise-resistant)
- **Attention analysis**: Eye tracking, gaze detection, pupil dilation, and stress/engagement metrics
- **Facial emotion recognition**: ViT-based classification (`trpakov/vit-face-expression`)
- **Speech emotion recognition**: Wav2Vec2-based analysis (`superb/wav2vec2-base-superb-er`)
- **ML-based multimodal fusion**: Neural network automatically combines facial, speech, and attention signals (works out-of-the-box)
- **Temporal smoothing**: Reduce flickering with rolling average, EMA, or hysteresis smoothing
- **VLA action generation**: OpenVLA-7B for emotion-aware robot actions (swappable via model registry)
- **Built-in action handlers**: HTTP, WebSocket, Serial/Arduino, ROS1/ROS2 integration
- **Extensible action handlers**: Plug in custom robot control logic

## Installation

### From Source (Recommended)

```bash
git clone https://github.com/saurabh947/emotion-detection-action
cd emotion-detection-action
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e .
```

### Optional Dependencies

```bash
pip install -e ".[robot]"       # Serial + WebSocket handlers
pip install -e ".[vla]"         # VLA model support (requires GPU)
pip install -e ".[dev]"         # Development tools (pytest, black, ruff)
```

## Quick Start

### Real-time Streaming (Webcam + Microphone)

```python
import asyncio
from emotion_detection_action import EmotionDetector, Config

config = Config(device="cuda", vla_enabled=False)

async def main():
    with EmotionDetector(config) as detector:
        async for result in detector.stream(camera=0, microphone=0):
            print(f"Emotion: {result.emotion.dominant_emotion.value}")
            print(f"Confidence: {result.emotion.fusion_confidence:.2%}")

asyncio.run(main())
```

### Frame-by-Frame API (for custom real-time pipelines)

```python
import cv2

# Capture frame from your own camera/video pipeline
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

# Process the frame
result = detector.process_frame(frame, audio=None, timestamp=0.0)
print(f"Emotion: {result.emotion.dominant_emotion.value}")

# Or get emotion only (without action generation)
emotion = detector.get_emotion_only(frame, audio=None)
```

## Architecture

```
Real-time Input → Detection Layer → Analysis Layer → ML Fusion → VLA Model → Actions
      │                 │                  │              │           │           │
   Camera          FaceDetector     FacialEmotion    Neural Net   OpenVLA    ActionHandler
   Microphone      SileroVAD        SpeechEmotion    (MLP)                   (extensible)
                   AttentionDet.    AttentionMetrics
```

The SDK processes webcam and microphone input through three AI-powered analysis pipelines:
1. **Facial**: Face detection → ViT emotion recognition (7 emotions)
2. **Audio**: Silero VAD (voice activity detection) → Wav2Vec2 speech emotion (4 emotions)
3. **Attention**: Eye tracking → Stress/engagement/nervousness metrics

All outputs are automatically fed into an **ML-based fusion model** (neural network) that combines them into a unified emotion prediction. The fusion works out-of-the-box with sensible default weights, or can be trained on custom data for improved accuracy.

## Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `vla_model` | `"openvla/openvla-7b"` | VLA model for action generation |
| `vla_enabled` | `True` | Set `False` for emotion-only mode |
| `device` | `"cuda"` | `"cuda"`, `"cpu"`, or `"mps"` |
| `face_detection_model` | `"mediapipe"` | Face detector model |
| `face_detection_threshold` | `0.5` | Face detection confidence threshold |
| `facial_emotion_model` | `"trpakov/vit-face-expression"` | Facial emotion model (HuggingFace) |
| `speech_emotion_model` | `"superb/wav2vec2-base-superb-er"` | Speech emotion model (HuggingFace) |
| `fusion_model_path` | `None` | Path to trained fusion MLP model |
| `fusion_device` | `"cpu"` | Device for fusion model (`"cpu"`, `"cuda"`, `"mps"`) |
| `frame_skip` | `1` | Process every nth frame |
| `max_faces` | `5` | Maximum faces per frame |

### Face Detection

The SDK uses MediaPipe Tasks API for fast, lightweight face detection that works well on CPU.

```python
config = Config(face_detection_model="mediapipe")
detector = EmotionDetector(config)
```

The face detector automatically downloads the model (~200KB) on first use and caches it locally.

### Temporal Smoothing

Reduce emotion flickering with built-in smoothing strategies:

| Strategy | Description | Best For |
|----------|-------------|----------|
| `none` | No smoothing (default) | Testing, debugging |
| `rolling` | Rolling average over N frames | Gentle smoothing |
| `ema` | Exponential Moving Average | Real-time, balanced response |
| `hysteresis` | Requires sustained change | Stable output, prevents rapid switching |

| Option | Default | Description |
|--------|---------|-------------|
| `smoothing_strategy` | `"none"` | Smoothing algorithm to use |
| `smoothing_window` | `5` | Window size for rolling average |
| `smoothing_ema_alpha` | `0.3` | EMA factor (0-1, lower = smoother) |
| `smoothing_hysteresis_threshold` | `0.15` | Min confidence diff to change |
| `smoothing_hysteresis_frames` | `3` | Frames emotion must persist |

```python
# Smooth with EMA (recommended for real-time)
config = Config(
    smoothing_strategy="ema",
    smoothing_ema_alpha=0.3,  # Lower = smoother
)

# Hysteresis for very stable output
config = Config(
    smoothing_strategy="hysteresis",
    smoothing_hysteresis_threshold=0.2,
    smoothing_hysteresis_frames=5,
)
```

### Attention Analysis

The SDK includes attention analysis using MediaPipe Face Landmarker to track eye movements, pupil size, and gaze patterns. This provides additional psychological indicators that modulate the emotion output.

| Metric | Description | Range |
|--------|-------------|-------|
| `stress_score` | Based on pupil dilation and blink rate | 0-1 |
| `engagement_score` | Based on eye contact and fixation stability | 0-1 |
| `nervousness_score` | Based on gaze aversion and instability | 0-1 |
| `blink_rate` | Blinks per minute | 0-60+ |
| `gaze_direction` | Where user is looking (x, y) | -1 to 1 |

**How attention affects fusion:**
The ML-based fusion model learns how attention metrics (stress, engagement, nervousness) influence the final emotion output from training data. The model can learn patterns such as:
- High stress correlating with negative emotions
- Low engagement reducing prediction confidence
- Nervousness affecting fear-related signals

| Option | Default | Description |
|--------|---------|-------------|
| `attention_analysis_enabled` | `True` | Enable attention tracking |
| `mediapipe_delegate` | `"cpu"` | MediaPipe acceleration: `"cpu"` or `"gpu"` |

```python
# Access attention metrics from result
async for result in detector.stream(camera=0, microphone=0):
    if result.emotion.attention:
        attn = result.emotion.attention
        print(f"Stress: {attn.stress_score:.0%}")
        print(f"Engagement: {attn.engagement_score:.0%}")
        print(f"Nervousness: {attn.nervousness_score:.0%}")

# Disable attention analysis
config = Config(attention_analysis_enabled=False)

# Use GPU acceleration (faster but may fail on some systems, especially macOS)
config = Config(mediapipe_delegate="gpu")
```
        print(f"Nervousness: {attn.nervousness_score:.0%}")

# Disable attention analysis
config = Config(attention_analysis_enabled=False)
```

### Multimodal Fusion (ML-based)

The SDK uses a neural network (MLP) to fuse emotion predictions from facial recognition, speech recognition, and attention analysis. This is the default fusion method - it works automatically after the facial, audio, and attention AI models produce their results.

By default, the fusion model uses sensible weights (60% facial, 40% speech) that work out-of-the-box. For better accuracy, you can train the model on your own labeled data to learn nuanced patterns specific to your use case.

**Architecture:**
```
Facial (7) + Speech (7) + Attention (3) → Dense(64) → Dense(32) → Emotions (7) + Confidence (1)
```

**Using a trained fusion model:**
```python
# Use a pre-trained fusion model
config = Config(
    fusion_model_path="models/fusion_mlp.pt",
    fusion_device="cpu",  # or "cuda", "mps"
)

detector = EmotionDetector(config)
```

**Training your own model:**
```bash
# Train with synthetic data (for testing)
python scripts/train_fusion_mlp.py --output models/fusion_mlp.pt

# Train with your own labeled data
python scripts/train_fusion_mlp.py --data data/emotions.csv --output models/fusion_mlp.pt
```

**Training data format (CSV):**
```csv
facial_angry,facial_disgusted,facial_fearful,facial_happy,facial_neutral,facial_sad,facial_surprised,speech_angry,speech_disgusted,speech_fearful,speech_happy,speech_neutral,speech_sad,speech_surprised,stress_score,engagement_score,nervousness_score,label
0.1,0.05,0.05,0.6,0.1,0.05,0.05,0.08,0.02,0.1,0.5,0.2,0.05,0.05,0.2,0.8,0.1,3
```

Where `label` is the emotion index (0=angry, 1=disgusted, 2=fearful, 3=happy, 4=neutral, 5=sad, 6=surprised).

**Model size:** ~15KB, ~3,500 parameters, <1ms inference on CPU.

**Default behavior:** When no `fusion_model_path` is provided, the model uses sensible default weights that approximate weighted average fusion (60% facial, 40% speech). This works out-of-the-box for basic use cases. For better accuracy, train a model with your own data.

## Supported Emotions

| Emotion | Facial | Speech |
|---------|--------|--------|
| Happy | ✅ | ✅ |
| Sad | ✅ | ✅ |
| Angry | ✅ | ✅ |
| Neutral | ✅ | ✅ |
| Fearful | ✅ | ❌ |
| Surprised | ✅ | ❌ |
| Disgusted | ✅ | ❌ |

*Speech model (SUPERB) supports 4 emotions; facial model supports all 7.*

## Action Handlers

Built-in handlers for robot integration:

| Handler | Protocol | Use Case | Install |
|---------|----------|----------|---------|
| `LoggingActionHandler` | - | Testing, debugging | Built-in |
| `HTTPActionHandler` | REST API | Cloud robots, web services | Built-in |
| `WebSocketActionHandler` | WebSocket | Real-time control | `pip install .[websocket]` |
| `SerialActionHandler` | UART/Serial | Arduino, embedded | `pip install .[serial]` |
| `ROSActionHandler` | ROS1/ROS2 | ROS robots | ROS installation |

### HTTP Handler

```python
from emotion_detection_action.actions import HTTPActionHandler

handler = HTTPActionHandler(
    endpoint="http://robot.local:8080/api/action",
    headers={"Authorization": "Bearer token123"}
)
detector = EmotionDetector(config, action_handler=handler)
```

### Serial Handler (Arduino)

```python
from emotion_detection_action.actions import SerialActionHandler

handler = SerialActionHandler(
    port="/dev/ttyUSB0",  # or "COM3" on Windows
    baudrate=115200,
    message_format="json"  # or "csv", "binary", "simple"
)
detector = EmotionDetector(config, action_handler=handler)
```

### ROS Handler

```python
from emotion_detection_action.actions import ROSActionHandler

handler = ROSActionHandler(
    node_name="emotion_detector",
    action_topic="/robot/emotion_action"
)
detector = EmotionDetector(config, action_handler=handler)
```

### Custom Handler

```python
from emotion_detection_action.actions.base import BaseActionHandler

class MyRobotHandler(BaseActionHandler):
    def connect(self) -> bool: ...
    def disconnect(self) -> None: ...
    def execute(self, action: ActionCommand) -> bool: ...

detector = EmotionDetector(config, action_handler=MyRobotHandler())
```

## Public API

**Main exports**: `EmotionDetector`, `Config`, `EmotionResult`, `DetectionResult`, `ActionCommand`, `FaceDetection`, `VoiceDetection`, `GazeDetection`, `AttentionResult`, `AttentionMetrics`

## Examples

### Real-time Demo

```bash
# Basic usage (default camera + microphone)
python examples/realtime_multimodal.py

# With options
python examples/realtime_multimodal.py --camera 1                    # Different camera
python examples/realtime_multimodal.py --smoothing ema               # Smoother output
python examples/realtime_multimodal.py --no-attention                # Disable attention tracking
```

The demo shows 4 panels:
- **FACIAL** (top-left): Face emotion detection with 7 emotions
- **AUDIO** (top-right): Speech emotion detection with 4 emotions
- **ATTENTION** (bottom-left): Stress, engagement, nervousness metrics
- **FUSED** (bottom-center): Combined emotion result with attention influence

### Robot Handlers Demo

```bash
python examples/robot_handlers.py --handler logging  # Test with console output
python examples/robot_handlers.py --handler http --endpoint http://localhost:8080/api
python examples/robot_handlers.py --handler serial --port /dev/ttyUSB0
```

## Development

```bash
pip install -e ".[dev]"
pytest
black src tests && ruff check src tests --fix
```

## Requirements

- Python 3.10+, PyTorch 2.0+, OpenCV, HuggingFace Transformers, MediaPipe
- VLA: CUDA GPU (~16GB VRAM, 8-bit quantization available)

## License

MIT License - see [LICENSE](LICENSE) for details.
