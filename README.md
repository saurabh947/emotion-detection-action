# Emotion Detection Action SDK

Real-time human emotion detection SDK for robotics using Vision-Language-Action (VLA) models.

## Features

- **Real-time emotion detection**: Live webcam + microphone processing
- **Face detection**: Automatic face detection using MTCNN or RetinaFace (configurable)
- **Voice activity detection**: Detect human speech using WebRTC VAD
- **Facial emotion recognition**: ViT-based classification (`trpakov/vit-face-expression`)
- **Speech emotion recognition**: Wav2Vec2-based analysis (`superb/wav2vec2-base-superb-er`)
- **Multimodal fusion**: Combine visual and audio signals (average, weighted, max, confidence strategies)
- **Temporal smoothing**: Reduce flickering with rolling average, EMA, or hysteresis smoothing
- **VLA action generation**: OpenVLA-7B for emotion-aware robot actions (swappable via model registry)
- **Built-in action handlers**: HTTP, WebSocket, Serial/Arduino, ROS1/ROS2 integration
- **Extensible action handlers**: Plug in custom robot control logic

## Installation

```bash
pip install emotion-detection-action
pip install emotion-detection-action[vla]        # For VLA support (requires GPU)
pip install emotion-detection-action[retinaface] # For RetinaFace face detection
pip install emotion-detection-action[robot]      # For serial + websocket handlers
pip install emotion-detection-action[serial]     # For serial/Arduino support only
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
Real-time Input → Detection Layer → Emotion Analysis → VLA Model → Actions
      │                 │                  │               │           │
   Camera          FaceDetector     FacialEmotion     OpenVLA    ActionHandler
   Microphone      VoiceActivity    SpeechEmotion                (extensible)
                   Detector         EmotionFusion
```

## Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `vla_model` | `"openvla/openvla-7b"` | VLA model for action generation |
| `vla_enabled` | `True` | Set `False` for emotion-only mode |
| `device` | `"cuda"` | `"cuda"`, `"cpu"`, or `"mps"` |
| `face_detection_model` | `"retinaface"` | Face detector: `"mtcnn"` or `"retinaface"` |
| `face_detection_threshold` | `0.9` | Face detection confidence threshold |
| `facial_emotion_model` | `"trpakov/vit-face-expression"` | Facial emotion model (HuggingFace) |
| `speech_emotion_model` | `"superb/wav2vec2-base-superb-er"` | Speech emotion model (HuggingFace) |
| `fusion_strategy` | `"confidence"` | `"average"`, `"weighted"`, `"max"`, `"confidence"` |
| `facial_weight` / `speech_weight` | `0.6` / `0.4` | Fusion weights for multimodal |
| `frame_skip` | `1` | Process every nth frame |
| `max_faces` | `5` | Maximum faces per frame |
| `vad_aggressiveness` | `2` | VAD filtering (0-3, higher = stricter) |

### Face Detection Models

| Model | Speed | Accuracy | Best For |
|-------|-------|----------|----------|
| `mtcnn` | Fast | Good | Real-time, well-lit conditions |
| `retinaface` | Slower | Better | Challenging poses, small/occluded faces |

```python
# Use RetinaFace for better accuracy (default)
config = Config(face_detection_model="retinaface")

# Use MTCNN for faster real-time processing
config = Config(face_detection_model="mtcnn")
```

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

**Main exports**: `EmotionDetector`, `Config`, `EmotionResult`, `DetectionResult`, `ActionCommand`, `FaceDetection`, `VoiceDetection`

## Examples

| Script | Description |
|--------|-------------|
| `examples/realtime_multimodal.py` | Real-time webcam + mic with 3 panels (facial, audio, fused) |
| `examples/robot_handlers.py` | Demo of HTTP, Serial, ROS action handlers |

## Development

```bash
pip install -e ".[dev]"
pytest
black src tests && ruff check src tests --fix
```

## Requirements

- Python 3.10+, PyTorch 2.0+, OpenCV, HuggingFace Transformers
- VLA: CUDA GPU (~16GB VRAM, 8-bit quantization available)

## License

MIT License - see [LICENSE](LICENSE) for details.
