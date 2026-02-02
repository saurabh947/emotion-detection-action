"""Tests for multimodal emotion fusion."""

import pytest

from emotion_detection_action.core.types import (
    AttentionMetrics,
    AttentionResult,
    BoundingBox,
    EmotionLabel,
    EmotionScores,
    FaceDetection,
    FacialEmotionResult,
    SpeechEmotionResult,
    VoiceDetection,
)
from emotion_detection_action.emotion.fusion import EmotionFusion


def create_facial_result(
    happy: float = 0.0,
    sad: float = 0.0,
    angry: float = 0.0,
    neutral: float = 0.0,
    confidence: float = 0.9,
) -> FacialEmotionResult:
    """Helper to create facial emotion result."""
    bbox = BoundingBox(x=0, y=0, width=100, height=100)
    face = FaceDetection(bbox=bbox, confidence=0.95)
    scores = EmotionScores(happy=happy, sad=sad, angry=angry, neutral=neutral)
    return FacialEmotionResult(face_detection=face, emotions=scores, confidence=confidence)


def create_speech_result(
    happy: float = 0.0,
    sad: float = 0.0,
    angry: float = 0.0,
    neutral: float = 0.0,
    confidence: float = 0.85,
) -> SpeechEmotionResult:
    """Helper to create speech emotion result."""
    voice = VoiceDetection(is_speech=True, confidence=0.9, start_time=0.0, end_time=1.0)
    scores = EmotionScores(happy=happy, sad=sad, angry=angry, neutral=neutral)
    return SpeechEmotionResult(voice_detection=voice, emotions=scores, confidence=confidence)


class TestEmotionFusion:
    """Tests for EmotionFusion class."""

    def test_initialization(self):
        """Test fusion initialization."""
        fusion = EmotionFusion(strategy="weighted", visual_weight=0.6, audio_weight=0.4)
        assert fusion.strategy == "weighted"
        assert fusion.visual_weight == 0.6
        assert fusion.audio_weight == 0.4

    def test_weight_normalization(self):
        """Test that weights are normalized to sum to 1."""
        fusion = EmotionFusion(visual_weight=3.0, audio_weight=1.0)
        assert fusion.visual_weight == 0.75
        assert fusion.audio_weight == 0.25

    def test_fuse_facial_only(self):
        """Test fusion with only facial result."""
        fusion = EmotionFusion()
        facial = create_facial_result(happy=0.8, neutral=0.2)

        result = fusion.fuse(facial_result=facial, speech_result=None, timestamp=1.0)

        assert result.timestamp == 1.0
        assert result.emotions.happy == 0.8
        assert result.facial_result is not None
        assert result.speech_result is None

    def test_fuse_speech_only(self):
        """Test fusion with only speech result."""
        fusion = EmotionFusion()
        speech = create_speech_result(sad=0.7, neutral=0.3)

        result = fusion.fuse(facial_result=None, speech_result=speech, timestamp=2.0)

        assert result.timestamp == 2.0
        assert result.emotions.sad == 0.7
        assert result.facial_result is None
        assert result.speech_result is not None

    def test_fuse_no_results_raises(self):
        """Test that fusion with no results raises error."""
        fusion = EmotionFusion()
        with pytest.raises(ValueError, match="No emotion results pass confidence threshold"):
            fusion.fuse(facial_result=None, speech_result=None)

    def test_fuse_average_strategy(self):
        """Test average fusion strategy."""
        fusion = EmotionFusion(strategy="average")
        facial = create_facial_result(happy=0.8, sad=0.2)
        speech = create_speech_result(happy=0.4, sad=0.6)

        result = fusion.fuse(facial, speech)

        # Average of 0.8 and 0.4
        assert result.emotions.happy == pytest.approx(0.6, abs=0.01)
        # Average of 0.2 and 0.6
        assert result.emotions.sad == pytest.approx(0.4, abs=0.01)

    def test_fuse_weighted_strategy(self):
        """Test weighted fusion strategy."""
        fusion = EmotionFusion(strategy="weighted", visual_weight=0.6, audio_weight=0.4)
        facial = create_facial_result(happy=1.0)
        speech = create_speech_result(happy=0.0)

        result = fusion.fuse(facial, speech)

        # 1.0 * 0.6 + 0.0 * 0.4 = 0.6
        assert result.emotions.happy == pytest.approx(0.6, abs=0.01)

    def test_fuse_max_strategy(self):
        """Test max fusion strategy."""
        fusion = EmotionFusion(strategy="max")
        facial = create_facial_result(happy=0.3, sad=0.7)
        speech = create_speech_result(happy=0.8, sad=0.2)

        result = fusion.fuse(facial, speech)

        # Max is taken for each emotion, then normalized
        # happy: max(0.3, 0.8) = 0.8, sad: max(0.7, 0.2) = 0.7
        # Total = 1.5, normalized: happy = 0.53, sad = 0.47
        assert result.emotions.happy > result.emotions.sad

    def test_fuse_confidence_strategy(self):
        """Test confidence-weighted fusion strategy."""
        fusion = EmotionFusion(strategy="confidence")
        # Facial with high confidence
        facial = create_facial_result(happy=1.0, confidence=0.9)
        # Speech with low confidence
        speech = create_speech_result(happy=0.0, confidence=0.1)

        result = fusion.fuse(facial, speech)

        # Should be weighted heavily toward facial due to higher confidence
        assert result.emotions.happy > 0.8

    def test_fusion_confidence_agreement(self):
        """Test that agreement between modalities increases confidence."""
        fusion = EmotionFusion()

        # Both agree on happy
        facial = create_facial_result(happy=0.9, confidence=0.8)
        speech = create_speech_result(happy=0.9, confidence=0.8)
        result_agree = fusion.fuse(facial, speech)

        # They disagree
        facial_diff = create_facial_result(happy=0.9, confidence=0.8)
        speech_diff = create_speech_result(angry=0.9, confidence=0.8)
        result_disagree = fusion.fuse(facial_diff, speech_diff)

        # Agreement should boost confidence
        assert result_agree.fusion_confidence > result_disagree.fusion_confidence

    def test_dominant_emotion_after_fusion(self):
        """Test that dominant emotion is correctly identified after fusion."""
        fusion = EmotionFusion(strategy="average")
        facial = create_facial_result(happy=0.6, sad=0.4)
        speech = create_speech_result(happy=0.8, sad=0.2)

        result = fusion.fuse(facial, speech)

        assert result.dominant_emotion == EmotionLabel.HAPPY

    def test_fuse_multiple(self):
        """Test fusing multiple result pairs."""
        fusion = EmotionFusion()

        facial_results = [
            create_facial_result(happy=0.9),
            create_facial_result(sad=0.8),
        ]
        speech_results = [
            create_speech_result(happy=0.8),
            create_speech_result(sad=0.7),
        ]

        results = fusion.fuse_multiple(facial_results, speech_results)

        assert len(results) == 2
        assert results[0].emotions.happy > 0.5
        assert results[1].emotions.sad > 0.5

    def test_fuse_multiple_unequal_lengths(self):
        """Test fusing with different numbers of facial and speech results."""
        fusion = EmotionFusion()

        facial_results = [
            create_facial_result(happy=0.9),
            create_facial_result(neutral=0.9),
            create_facial_result(sad=0.9),
        ]
        speech_results = [
            create_speech_result(happy=0.8),
        ]

        results = fusion.fuse_multiple(facial_results, speech_results)

        # Should have 3 results (max of the two lists)
        assert len(results) == 3
        # First has both modalities
        assert results[0].facial_result is not None
        assert results[0].speech_result is not None
        # Others only have facial
        assert results[1].facial_result is not None
        assert results[1].speech_result is None


class TestFusionEdgeCases:
    """Test edge cases for emotion fusion."""

    def test_zero_confidence_both(self):
        """Test fusion when both have zero confidence."""
        fusion = EmotionFusion(strategy="confidence")
        facial = create_facial_result(happy=0.5, confidence=0.0)
        speech = create_speech_result(sad=0.5, confidence=0.0)

        # Should fall back to average when both have zero confidence
        result = fusion.fuse(facial, speech)
        assert result is not None

    def test_all_emotions_zero(self):
        """Test fusion when all emotion scores are zero."""
        fusion = EmotionFusion()
        facial = create_facial_result()  # All zeros
        speech = create_speech_result()  # All zeros

        result = fusion.fuse(facial, speech)

        # Should still return a result
        assert result is not None
        # Dominant emotion will be one of them (likely first alphabetically)
        assert result.dominant_emotion is not None


def create_attention_result(
    stress: float = 0.0,
    engagement: float = 0.5,
    nervousness: float = 0.0,
    confidence: float = 0.9,
) -> AttentionResult:
    """Helper to create attention result."""
    metrics = AttentionMetrics(
        stress_score=stress,
        engagement_score=engagement,
        nervousness_score=nervousness,
    )
    return AttentionResult(timestamp=0.0, metrics=metrics, confidence=confidence)


class TestAttentionModulatedFusion:
    """Tests for attention-modulated emotion fusion."""

    def test_fusion_with_attention_init(self):
        """Test fusion initialization with attention parameters."""
        fusion = EmotionFusion(
            attention_weight=0.3,
            attention_stress_amplification=2.0,
            attention_engagement_threshold=0.4,
        )
        assert fusion.attention_weight == 0.3
        assert fusion.attention_stress_amplification == 2.0
        assert fusion.attention_engagement_threshold == 0.4

    def test_fuse_with_attention_result(self):
        """Test fusion includes attention result."""
        fusion = EmotionFusion()
        facial = create_facial_result(happy=0.8)
        attention = create_attention_result(stress=0.3, engagement=0.8)

        result = fusion.fuse(
            facial_result=facial,
            speech_result=None,
            attention_result=attention,
        )

        assert result.attention_result is not None
        assert result.attention_result.stress_score == 0.3
        assert result.attention_result.engagement_score == 0.8

    def test_stress_amplifies_negative_emotions(self):
        """Test that high stress amplifies negative emotions."""
        fusion = EmotionFusion(attention_weight=0.3, attention_stress_amplification=1.5)

        facial = create_facial_result(sad=0.5, happy=0.5)

        # No stress
        no_stress_attn = create_attention_result(stress=0.0, engagement=0.8)
        result_no_stress = fusion.fuse(facial, None, no_stress_attn)

        # High stress
        high_stress_attn = create_attention_result(stress=0.8, engagement=0.8)
        result_high_stress = fusion.fuse(facial, None, high_stress_attn)

        # High stress should amplify sad relative to happy
        no_stress_ratio = result_no_stress.emotions.sad / max(result_no_stress.emotions.happy, 0.01)
        high_stress_ratio = result_high_stress.emotions.sad / max(result_high_stress.emotions.happy, 0.01)

        assert high_stress_ratio > no_stress_ratio

    def test_low_engagement_reduces_confidence(self):
        """Test that low engagement reduces fusion confidence."""
        fusion = EmotionFusion(
            attention_weight=0.3,
            attention_engagement_threshold=0.5,
        )

        facial = create_facial_result(happy=0.8, confidence=0.9)

        # High engagement
        high_eng_attn = create_attention_result(engagement=0.8)
        result_high_eng = fusion.fuse(facial, None, high_eng_attn)

        # Low engagement
        low_eng_attn = create_attention_result(engagement=0.2)
        result_low_eng = fusion.fuse(facial, None, low_eng_attn)

        # Low engagement should reduce confidence
        assert result_low_eng.fusion_confidence < result_high_eng.fusion_confidence

    def test_nervousness_boosts_fearful(self):
        """Test that nervousness boosts fearful emotion."""
        fusion = EmotionFusion(attention_weight=0.3)

        # Create facial result with some fearful
        bbox = BoundingBox(x=0, y=0, width=100, height=100)
        face = FaceDetection(bbox=bbox, confidence=0.95)
        scores = EmotionScores(fearful=0.3, neutral=0.7)
        facial = FacialEmotionResult(face_detection=face, emotions=scores, confidence=0.9)

        # No nervousness
        no_nerv_attn = create_attention_result(nervousness=0.0, engagement=0.8)
        result_no_nerv = fusion.fuse(facial, None, no_nerv_attn)

        # High nervousness
        high_nerv_attn = create_attention_result(nervousness=0.8, engagement=0.8)
        result_high_nerv = fusion.fuse(facial, None, high_nerv_attn)

        # Nervousness should boost fearful
        assert result_high_nerv.emotions.fearful > result_no_nerv.emotions.fearful

    def test_attention_with_zero_confidence_ignored(self):
        """Test that attention with zero confidence is ignored."""
        fusion = EmotionFusion(attention_weight=0.3)

        facial = create_facial_result(happy=0.8)

        # Attention with zero confidence
        zero_conf_attn = create_attention_result(stress=0.9, confidence=0.0)

        # Fusion without attention
        result_no_attn = fusion.fuse(facial, None, None)

        # Fusion with zero-confidence attention
        result_zero_conf = fusion.fuse(facial, None, zero_conf_attn)

        # Should be identical since zero confidence attention is ignored
        assert result_no_attn.emotions.happy == result_zero_conf.emotions.happy

    def test_attention_preserves_in_result(self):
        """Test that attention result is preserved in output."""
        fusion = EmotionFusion()
        facial = create_facial_result(neutral=0.9)
        attention = create_attention_result(stress=0.5, engagement=0.7, nervousness=0.3)

        result = fusion.fuse(facial, None, attention)

        # Original attention should be accessible
        assert result.attention is not None
        assert result.attention.stress_score == 0.5
        assert result.attention.engagement_score == 0.7
        assert result.attention.nervousness_score == 0.3

