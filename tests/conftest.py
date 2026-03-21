"""Pytest configuration and shared fixtures."""

import numpy as np
import pytest

from emotion_detection_action.core.types import NeuralEmotionResult

_EMOTION_KEYS = [
    "angry", "disgusted", "fearful", "happy",
    "neutral", "sad", "surprised", "unclear",
]


@pytest.fixture
def sample_frame():
    """Create a sample video frame (H, W, 3) uint8."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_clip():
    """Create a sample 16-frame video clip (T, H, W, 3) uint8."""
    return np.random.randint(0, 255, (16, 480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_audio():
    """Create sample audio data (1 second at 16 kHz)."""
    return np.random.randn(16000).astype(np.float32) * 0.1


@pytest.fixture
def happy_result() -> NeuralEmotionResult:
    """NeuralEmotionResult with dominant emotion 'happy'."""
    scores = {k: 0.02 for k in _EMOTION_KEYS}
    scores["happy"] = 0.9
    return NeuralEmotionResult(
        dominant_emotion="happy",
        emotion_scores=scores,
        latent_embedding=[0.0] * 512,
        metrics={"stress": 0.1, "engagement": 0.8, "arousal": 0.5},
        confidence=0.9,
        timestamp=0.0,
    )


@pytest.fixture
def sad_result() -> NeuralEmotionResult:
    """NeuralEmotionResult with dominant emotion 'sad'."""
    scores = {k: 0.02 for k in _EMOTION_KEYS}
    scores["sad"] = 0.85
    return NeuralEmotionResult(
        dominant_emotion="sad",
        emotion_scores=scores,
        latent_embedding=[0.0] * 512,
        metrics={"stress": 0.4, "engagement": 0.3, "arousal": 0.2},
        confidence=0.85,
        timestamp=0.0,
    )


@pytest.fixture
def angry_result() -> NeuralEmotionResult:
    """NeuralEmotionResult with dominant emotion 'angry'."""
    scores = {k: 0.02 for k in _EMOTION_KEYS}
    scores["angry"] = 0.88
    return NeuralEmotionResult(
        dominant_emotion="angry",
        emotion_scores=scores,
        latent_embedding=[0.0] * 512,
        metrics={"stress": 0.8, "engagement": 0.5, "arousal": 0.9},
        confidence=0.88,
        timestamp=0.0,
    )


@pytest.fixture
def neutral_result() -> NeuralEmotionResult:
    """NeuralEmotionResult with dominant emotion 'neutral'."""
    scores = {k: 0.02 for k in _EMOTION_KEYS}
    scores["neutral"] = 0.9
    return NeuralEmotionResult(
        dominant_emotion="neutral",
        emotion_scores=scores,
        latent_embedding=[0.0] * 512,
        metrics={"stress": 0.1, "engagement": 0.5, "arousal": 0.3},
        confidence=0.9,
        timestamp=0.0,
    )
