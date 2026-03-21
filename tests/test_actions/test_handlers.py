"""Tests for action handlers."""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock

from emotion_detection_action.actions.base import BaseActionHandler
from emotion_detection_action.actions.http_handler import HTTPActionHandler, WebSocketActionHandler
from emotion_detection_action.actions.logging_handler import LoggingActionHandler, MockActionHandler
from emotion_detection_action.actions.ros_handler import ROSActionHandler
from emotion_detection_action.actions.serial_handler import SerialActionHandler
from emotion_detection_action.core.types import (
    ActionCommand,
    EmotionLabel,
    NeuralEmotionResult,
)

_EMOTION_KEYS = [
    "angry", "disgusted", "fearful", "happy",
    "neutral", "sad", "surprised", "unclear",
]


def _result(dominant: str, confidence: float = 0.9) -> NeuralEmotionResult:
    scores = {k: 0.02 for k in _EMOTION_KEYS}
    scores[dominant] = confidence
    return NeuralEmotionResult(
        dominant_emotion=dominant,
        emotion_scores=scores,
        latent_embedding=[0.0] * 512,
        metrics={"stress": 0.1, "engagement": 0.5, "arousal": 0.3},
        confidence=confidence,
        timestamp=0.0,
    )


class TestBaseActionHandler:
    """Tests for BaseActionHandler abstract class."""

    def test_generate_default_action_happy(self):
        """Test default action for happy emotion."""
        handler = LoggingActionHandler(verbose=False)
        action = handler._generate_default_action(_result("happy"))
        assert action.action_type == "acknowledge"
        assert action.parameters.get("gesture") == "nod"

    def test_generate_default_action_sad(self):
        """Test default action for sad emotion."""
        handler = LoggingActionHandler(verbose=False)
        action = handler._generate_default_action(_result("sad"))
        assert action.action_type == "comfort"

    def test_generate_default_action_angry(self):
        """Test default action for angry emotion."""
        handler = LoggingActionHandler(verbose=False)
        action = handler._generate_default_action(_result("angry"))
        assert action.action_type == "de_escalate"

    def test_generate_default_action_neutral(self):
        """Test default action for neutral emotion."""
        handler = LoggingActionHandler(verbose=False)
        action = handler._generate_default_action(_result("neutral"))
        assert action.action_type == "idle"

    def test_get_supported_actions(self):
        """Test getting list of supported actions."""
        handler = LoggingActionHandler(verbose=False)
        actions = handler.get_supported_actions()

        assert "idle" in actions
        assert "acknowledge" in actions
        assert "comfort" in actions

    def test_validate_action_valid(self):
        """Test validating a valid action."""
        handler = LoggingActionHandler(verbose=False)
        action = ActionCommand(action_type="acknowledge")

        is_valid, error = handler.validate_action(action)

        assert is_valid is True
        assert error == ""

    def test_validate_action_invalid(self):
        """Test validating an invalid action."""
        handler = LoggingActionHandler(verbose=False)
        action = ActionCommand(action_type="unsupported_action_xyz")

        is_valid, error = handler.validate_action(action)

        assert is_valid is False
        assert "Unsupported action type" in error


class TestLoggingActionHandler:
    """Tests for LoggingActionHandler."""

    def test_connect_disconnect(self):
        """Test connect and disconnect."""
        handler = LoggingActionHandler(verbose=False)

        assert not handler.is_connected
        result = handler.connect()
        assert result is True
        assert handler.is_connected

        handler.disconnect()
        assert not handler.is_connected

    def test_execute_records_action(self):
        """Test that execute records action in history."""
        handler = LoggingActionHandler(verbose=False)
        handler.connect()

        action = ActionCommand(
            action_type="greeting",
            parameters={"message": "hello"},
            confidence=0.9,
        )
        result = handler.execute(action)

        assert result is True
        history = handler.get_action_history()
        assert len(history) == 1
        assert history[0]["action_type"] == "greeting"
        assert history[0]["parameters"]["message"] == "hello"

    def test_execute_multiple_actions(self):
        """Test executing multiple actions."""
        handler = LoggingActionHandler(verbose=False)
        handler.connect()

        handler.execute(ActionCommand(action_type="action1"))
        handler.execute(ActionCommand(action_type="action2"))
        handler.execute(ActionCommand(action_type="action3"))

        history = handler.get_action_history()
        assert len(history) == 3
        assert history[0]["index"] == 1
        assert history[2]["index"] == 3

    def test_get_last_action(self):
        """Test getting the last executed action."""
        handler = LoggingActionHandler(verbose=False)
        handler.connect()

        handler.execute(ActionCommand(action_type="first"))
        handler.execute(ActionCommand(action_type="last"))

        last = handler.get_last_action()
        assert last is not None
        assert last["action_type"] == "last"

    def test_get_last_action_empty(self):
        """Test getting last action when none executed."""
        handler = LoggingActionHandler(verbose=False)
        assert handler.get_last_action() is None

    def test_clear_history(self):
        """Test clearing action history."""
        handler = LoggingActionHandler(verbose=False)
        handler.connect()

        handler.execute(ActionCommand(action_type="test"))
        assert len(handler.get_action_history()) == 1

        handler.clear_history()
        assert len(handler.get_action_history()) == 0

    def test_get_statistics(self):
        """Test getting action statistics."""
        handler = LoggingActionHandler(verbose=False)
        handler.connect()

        handler.execute(ActionCommand(action_type="greeting", confidence=0.9))
        handler.execute(ActionCommand(action_type="greeting", confidence=0.8))
        handler.execute(ActionCommand(action_type="idle", confidence=1.0))

        stats = handler.get_statistics()

        assert stats["total_actions"] == 3
        assert stats["action_types"]["greeting"] == 2
        assert stats["action_types"]["idle"] == 1
        assert stats["avg_confidence"] == pytest.approx(0.9, abs=0.01)

    def test_get_statistics_empty(self):
        """Test statistics when no actions executed."""
        handler = LoggingActionHandler(verbose=False)
        stats = handler.get_statistics()

        assert stats["total_actions"] == 0
        assert stats["avg_confidence"] == 0.0

    def test_callback(self):
        """Test custom callback on execute."""
        received_actions = []

        def callback(action: ActionCommand) -> None:
            received_actions.append(action.action_type)

        handler = LoggingActionHandler(verbose=False, callback=callback)
        handler.connect()

        handler.execute(ActionCommand(action_type="test1"))
        handler.execute(ActionCommand(action_type="test2"))

        assert received_actions == ["test1", "test2"]

    def test_context_manager(self):
        """Test using handler as context manager."""
        with LoggingActionHandler(verbose=False) as handler:
            assert handler.is_connected
            handler.execute(ActionCommand(action_type="test"))

        assert not handler.is_connected

    def test_execute_for_emotion(self):
        """Test executing action based on NeuralEmotionResult."""
        handler = LoggingActionHandler(verbose=False)
        handler.connect()

        result = handler.execute_for_emotion(_result("happy"))

        assert result is True
        last = handler.get_last_action()
        assert last["action_type"] == "acknowledge"


class TestMockActionHandler:
    """Tests for MockActionHandler."""

    def test_expect_and_verify_success(self):
        """Test expecting and verifying actions successfully."""
        handler = MockActionHandler()
        handler.connect()

        handler.expect_action("greeting")
        handler.expect_action("idle")

        handler.execute(ActionCommand(action_type="greeting"))
        handler.execute(ActionCommand(action_type="idle"))

        success, message = handler.verify_expectations()
        assert success is True
        assert message == "All expectations met"

    def test_expect_and_verify_failure_wrong_action(self):
        """Test verification fails with wrong action."""
        handler = MockActionHandler()
        handler.connect()

        handler.expect_action("greeting")
        handler.execute(ActionCommand(action_type="idle"))

        success, message = handler.verify_expectations()
        assert success is False
        assert "expected 'greeting'" in message

    def test_expect_and_verify_failure_wrong_count(self):
        """Test verification fails with wrong number of actions."""
        handler = MockActionHandler()
        handler.connect()

        handler.expect_action("greeting")
        handler.expect_action("idle")

        handler.execute(ActionCommand(action_type="greeting"))

        success, message = handler.verify_expectations()
        assert success is False
        assert "Expected 2 actions" in message

    def test_expect_actions_list(self):
        """Test expecting multiple actions at once."""
        handler = MockActionHandler()
        handler.connect()

        handler.expect_actions(["a", "b", "c"])

        handler.execute(ActionCommand(action_type="a"))
        handler.execute(ActionCommand(action_type="b"))
        handler.execute(ActionCommand(action_type="c"))

        success, _ = handler.verify_expectations()
        assert success is True

    def test_assert_action_executed(self):
        """Test asserting specific action was executed."""
        handler = MockActionHandler()
        handler.connect()

        handler.execute(ActionCommand(action_type="greeting"))
        handler.execute(ActionCommand(action_type="idle"))

        handler.assert_action_executed("greeting")  # Should not raise
        handler.assert_action_executed("idle")  # Should not raise

    def test_assert_action_executed_raises(self):
        """Test assertion raises when action not executed."""
        handler = MockActionHandler()
        handler.connect()

        handler.execute(ActionCommand(action_type="greeting"))

        with pytest.raises(AssertionError, match="Action 'idle' was not executed"):
            handler.assert_action_executed("idle")

    def test_assert_no_actions(self):
        """Test asserting no actions were executed."""
        handler = MockActionHandler()
        handler.connect()

        handler.assert_no_actions()  # Should not raise

    def test_assert_no_actions_raises(self):
        """Test assertion raises when actions were executed."""
        handler = MockActionHandler()
        handler.connect()

        handler.execute(ActionCommand(action_type="greeting"))

        with pytest.raises(AssertionError, match="Expected no actions"):
            handler.assert_no_actions()

    def test_reset_expectations(self):
        """Test resetting expectations and history."""
        handler = MockActionHandler()
        handler.connect()

        handler.expect_action("test")
        handler.execute(ActionCommand(action_type="test"))

        handler.reset_expectations()

        assert len(handler.get_action_history()) == 0
        success, _ = handler.verify_expectations()
        assert success is True  # No expectations, no actions


class TestHTTPActionHandler:
    """Tests for HTTPActionHandler."""

    def test_initialization(self):
        """Test handler initialization."""
        handler = HTTPActionHandler(
            endpoint="http://localhost:8080/api/action",
            method="POST",
            timeout=5.0,
        )

        assert handler.endpoint == "http://localhost:8080/api/action"
        assert handler.method == "POST"
        assert handler.timeout == 5.0

    def test_build_payload(self):
        """Test payload building from action command."""
        handler = HTTPActionHandler(endpoint="http://test.com/api")

        action = ActionCommand(
            action_type="greeting",
            parameters={"message": "hello"},
            confidence=0.9,
        )

        payload = handler._build_payload(action)

        assert payload["action_type"] == "greeting"
        assert payload["parameters"]["message"] == "hello"
        assert payload["confidence"] == 0.9

    @patch("emotion_detection_action.actions.http_handler.request.urlopen")
    def test_execute_success(self, mock_urlopen):
        """Test successful HTTP execution."""
        # Mock response
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"status": "ok"}'
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_urlopen.return_value = mock_response

        handler = HTTPActionHandler(endpoint="http://test.com/api")
        handler._is_connected = True

        action = ActionCommand(action_type="test")
        result = handler.execute(action)

        assert result is True
        assert handler._success_count == 1

    def test_get_statistics(self):
        """Test getting statistics."""
        handler = HTTPActionHandler(endpoint="http://test.com/api")
        handler._success_count = 5
        handler._error_count = 2

        stats = handler.get_statistics()

        assert stats["success_count"] == 5
        assert stats["error_count"] == 2
        assert stats["success_rate"] == pytest.approx(5/7, abs=0.01)


class TestWebSocketActionHandler:
    """Tests for WebSocketActionHandler."""

    def test_initialization(self):
        """Test handler initialization."""
        handler = WebSocketActionHandler(
            url="ws://localhost:8080/ws",
            reconnect=True,
        )

        assert handler.url == "ws://localhost:8080/ws"
        assert handler.reconnect is True

    def test_not_connected_without_websocket(self):
        """Test that connection fails without websocket library."""
        handler = WebSocketActionHandler(url="ws://test.com/ws")

        # Without websocket-client installed, this may fail
        # We just test the handler was created correctly
        assert handler._ws is None
        assert handler.message_count == 0


class TestSerialActionHandler:
    """Tests for SerialActionHandler."""

    def test_initialization(self):
        """Test handler initialization."""
        handler = SerialActionHandler(
            port="/dev/ttyUSB0",
            baudrate=115200,
            message_format="json",
        )

        assert handler.port == "/dev/ttyUSB0"
        assert handler.baudrate == 115200
        assert handler.message_format == "json"

    def test_format_json(self):
        """Test JSON message formatting."""
        handler = SerialActionHandler(port="/dev/ttyUSB0")

        action = ActionCommand(
            action_type="greeting",
            parameters={"emotion": "happy"},
            confidence=0.9,
        )

        message = handler._format_json(action)
        data = json.loads(message.strip())

        assert data["type"] == "greeting"
        assert data["params"]["emotion"] == "happy"
        assert data["conf"] == 0.9

    def test_format_csv(self):
        """Test CSV message formatting."""
        handler = SerialActionHandler(port="/dev/ttyUSB0", message_format="csv")

        action = ActionCommand(
            action_type="acknowledge",
            parameters={"emotion": "happy"},
            confidence=0.9,
        )

        message = handler._format_csv(action)

        # acknowledge=1, happy=0, confidence=90
        assert message.strip() == "1,0,90"

    def test_format_binary(self):
        """Test binary message formatting."""
        handler = SerialActionHandler(port="/dev/ttyUSB0", message_format="binary")

        action = ActionCommand(
            action_type="idle",
            parameters={"emotion": "neutral"},
            confidence=0.85,
        )

        message = handler._format_binary(action)

        # idle=0, neutral=6, confidence=85
        assert message.strip() == "A00E6C085"

    def test_format_simple(self):
        """Test simple message formatting."""
        handler = SerialActionHandler(port="/dev/ttyUSB0", message_format="simple")

        action = ActionCommand(
            action_type="greeting",
            parameters={"speed": 100, "angle": 45},
            confidence=0.9,
        )

        message = handler._format_simple(action)

        assert "greeting:" in message
        assert "speed=100" in message
        assert "angle=45" in message

    def test_list_ports_static_method(self):
        """Test listing available ports."""
        # This just verifies the method exists and returns a list
        ports = SerialActionHandler.list_ports()
        assert isinstance(ports, list)

    def test_action_codes(self):
        """Test that all expected action codes are defined."""
        expected_actions = [
            "idle", "acknowledge", "comfort", "de_escalate",
            "reassure", "wait", "retreat", "approach",
        ]

        for action in expected_actions:
            assert action in SerialActionHandler.ACTION_CODES

    def test_emotion_codes(self):
        """Test that all expected emotion codes are defined."""
        expected_emotions = [
            "happy", "sad", "angry", "fearful",
            "surprised", "disgusted", "neutral",
        ]

        for emotion in expected_emotions:
            assert emotion in SerialActionHandler.EMOTION_CODES


class TestROSActionHandler:
    """Tests for ROSActionHandler."""

    def test_initialization(self):
        """Test handler initialization."""
        handler = ROSActionHandler(
            node_name="test_node",
            action_topic="/test/action",
        )

        assert handler.node_name == "test_node"
        assert handler.action_topic == "/test/action"

    def test_is_ros_available(self):
        """Test checking ROS availability."""
        availability = ROSActionHandler.is_ros_available()

        assert "ros1" in availability
        assert "ros2" in availability
        assert isinstance(availability["ros1"], bool)
        assert isinstance(availability["ros2"], bool)

    def test_get_statistics(self):
        """Test getting statistics."""
        handler = ROSActionHandler(
            action_topic="/emotion_action",
            emotion_topic="/emotion_result",
        )
        handler._message_count = 10

        stats = handler.get_statistics()

        assert stats["message_count"] == 10
        assert stats["action_topic"] == "/emotion_action"
        assert stats["emotion_topic"] == "/emotion_result"

    def test_ros_version_detection(self):
        """Test that ROS version is detected."""
        handler = ROSActionHandler()

        # Should be 0, 1, or 2 depending on installation
        assert handler.ros_version in (0, 1, 2)

