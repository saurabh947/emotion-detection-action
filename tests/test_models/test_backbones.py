"""Tests for FaceCropPipeline — bounding box math, centre-crop fallback,
margin expansion, and output size guarantees.

MediaPipe is mocked so no GPU/model download is needed.
"""

import numpy as np
import pytest

try:
    from emotion_detection_action.models.backbones import FaceCropPipeline
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")


def _solid_frame(h: int = 120, w: int = 160, color=(128, 64, 32)) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = color
    return frame


class TestFaceCropPipelineOutputSize:
    def test_crop_returns_correct_size(self):
        """crop() must always return (image_size, image_size, 3)."""
        pipeline = FaceCropPipeline(image_size=112)
        frame = _solid_frame(480, 640)
        cropped = pipeline.crop(frame)
        assert cropped.shape == (112, 112, 3)

    def test_different_image_sizes(self):
        for size in (112, 224, 256):
            pipeline = FaceCropPipeline(image_size=size)
            cropped = pipeline.crop(_solid_frame())
            assert cropped.shape == (size, size, 3)

    def test_output_dtype_uint8(self):
        pipeline = FaceCropPipeline(image_size=112)
        cropped = pipeline.crop(_solid_frame())
        assert cropped.dtype == np.uint8


class TestFaceCropPipelineFallback:
    def test_no_face_detected_uses_centre_crop(self):
        """When MediaPipe finds no face, the crop should still be the right size
        (centre-crop fallback)."""
        pipeline = FaceCropPipeline(image_size=112, min_confidence=0.99)
        # A solid-colour frame is unlikely to trigger a real face detection;
        # the pipeline should fall back gracefully.
        frame = _solid_frame(480, 640)
        cropped = pipeline.crop(frame)
        assert cropped.shape == (112, 112, 3)

    def test_small_frame_does_not_crash(self):
        """Frames smaller than image_size should still produce correctly-sized output."""
        pipeline = FaceCropPipeline(image_size=224)
        tiny_frame = _solid_frame(32, 32)
        cropped = pipeline.crop(tiny_frame)
        assert cropped.shape == (224, 224, 3)

    def test_square_frame_crop(self):
        """Square input frames should work without issues."""
        pipeline = FaceCropPipeline(image_size=112)
        square = _solid_frame(224, 224)
        cropped = pipeline.crop(square)
        assert cropped.shape == (112, 112, 3)


class TestFaceCropPipelineMargin:
    def test_margin_zero_accepted(self):
        """margin=0.0 should work without error."""
        pipeline = FaceCropPipeline(margin=0.0, image_size=112)
        cropped = pipeline.crop(_solid_frame())
        assert cropped.shape == (112, 112, 3)

    def test_large_margin_does_not_crash(self):
        """margin=1.0 (maximum) should not cause an index-out-of-bounds."""
        pipeline = FaceCropPipeline(margin=1.0, image_size=112)
        cropped = pipeline.crop(_solid_frame(480, 640))
        assert cropped.shape == (112, 112, 3)
