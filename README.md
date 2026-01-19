# Emotion Detector SDK

Human emotion detection SDK for robotics using Vision-Language-Action (VLA) models.

## Features

- **Multi-modal emotion detection**: Analyze emotions from video, images, and audio
- **Face detection**: Automatic face detection using MTCNN (RetinaFace planned)
- **Voice activity detection**: Detect human speech using WebRTC VAD
- **Facial emotion recognition**: ViT-based classification (`trpakov/vit-face-expression`)
- **Speech emotion recognition**: Wav2Vec2-based analysis (`superb/wav2vec2-base-superb-er`)
- **Multimodal fusion**: Combine visual and audio signals (average, weighted, max, confidence strategies)
- **VLA action generation**: OpenVLA-7B for emotion-aware robot actions (swappable via model registry)
- **Real-time & batch processing**: Streaming from webcam/mic or file-based processing
- **Extensible action handlers**: Plug in custom robot control logic

## Installation

```bash
pip install emotion-detector
pip install emotion-detector[vla]  # For VLA support (requires GPU)
```

## Quick Start

```python
from emotion_detector import EmotionDetector, Config

config = Config(device="cuda", vla_enabled=False)

with EmotionDetector(config) as detector:
    # Process video/image/audio files
    results = detector.process(video_path="recording.mp4", audio_path="recording.wav")
    
    for result in results:
        print(f"Emotion: {result.emotion.dominant_emotion.value}")
        print(f"Confidence: {result.emotion.fusion_confidence:.2%}")
```

### Real-time Streaming

```python
async for result in detector.stream(video_source=0, audio_source=0):
    print(f"Emotion: {result.emotion.dominant_emotion.value}")
```

### Frame-by-Frame API

- `detector.process_frame(frame, audio, timestamp)` - Process single frame with optional audio
- `detector.get_emotion_only(frame, audio)` - Get emotion without action generation

## Architecture

```
Input Sources → Detection Layer → Emotion Analysis → VLA Model → Actions
     │                │                  │               │           │
  VideoInput     FaceDetector     FacialEmotion     OpenVLA    ActionHandler
  ImageInput     VoiceActivity    SpeechEmotion                (extensible)
  AudioInput     Detector         EmotionFusion
```

## Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `vla_model` | `"openvla/openvla-7b"` | VLA model for action generation |
| `vla_enabled` | `True` | Set `False` for emotion-only mode |
| `device` | `"cuda"` | `"cuda"`, `"cpu"`, or `"mps"` |
| `mode` | `"batch"` | `"batch"` or `"realtime"` |
| `face_detection_threshold` | `0.9` | Face detection confidence threshold |
| `facial_emotion_model` | `"trpakov/vit-face-expression"` | Facial emotion model |
| `speech_emotion_model` | `"superb/wav2vec2-base-superb-er"` | Speech emotion model |
| `fusion_strategy` | `"weighted"` | `"average"`, `"weighted"`, `"max"`, `"confidence"` |
| `facial_weight` / `speech_weight` | `0.6` / `0.4` | Fusion weights for multimodal |
| `frame_skip` | `1` | Process every nth frame |
| `max_faces` | `5` | Maximum faces per frame |
| `vad_aggressiveness` | `2` | VAD filtering (0-3, higher = stricter) |

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

## Custom Action Handlers

Extend `BaseActionHandler` to integrate with your robot:

```python
from emotion_detector.actions.base import BaseActionHandler

class MyRobotHandler(BaseActionHandler):
    def connect(self) -> bool: ...
    def disconnect(self) -> None: ...
    def execute(self, action: ActionCommand) -> bool: ...

detector = EmotionDetector(config, action_handler=MyRobotHandler())
```

## Public API

**Main exports**: `EmotionDetector`, `Config`, `EmotionResult`, `DetectionResult`, `ActionCommand`, `FaceDetection`, `VoiceDetection`, `ProcessingMode`

## Examples

| Script | Description |
|--------|-------------|
| `examples/basic_usage.py` | Image and video processing |
| `examples/realtime_webcam.py` | Webcam with visualization |
| `examples/realtime_multimodal.py` | Separate video + audio panels |
| `examples/batch_processing.py` | Multi-file processing with reports |

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
