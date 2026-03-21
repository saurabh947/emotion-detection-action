"""Base interface for action handlers."""

from abc import ABC, abstractmethod
from typing import Any

from emotion_detection_action.core.types import ActionCommand, NeuralEmotionResult


class BaseActionHandler(ABC):
    """Abstract base class for action handlers.

    Action handlers convert ActionCommand objects from the VLA model
    into actual robot actions. This abstraction allows different
    robot platforms to implement their own action execution logic.

    Example:
        >>> class MyRobotHandler(BaseActionHandler):
        ...     def execute(self, action: ActionCommand) -> bool:
        ...         # Send action to robot
        ...         return True

        >>> handler = MyRobotHandler()
        >>> handler.execute(action_command)
    """

    def __init__(self, name: str = "base") -> None:
        """Initialize the action handler.

        Args:
            name: Handler identifier name.
        """
        self.name = name
        self._is_connected = False

    @property
    def is_connected(self) -> bool:
        """Check if handler is connected to robot/system."""
        return self._is_connected

    @abstractmethod
    def connect(self) -> bool:
        """Connect to the robot or action system.

        Returns:
            True if connection successful.
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the robot or action system."""
        pass

    @abstractmethod
    def execute(self, action: ActionCommand) -> bool:
        """Execute an action command.

        Args:
            action: The action command to execute.

        Returns:
            True if execution was successful.
        """
        pass

    def execute_for_emotion(
        self,
        emotion_result: NeuralEmotionResult,
        action: ActionCommand | None = None,
    ) -> bool:
        """Execute an action based on a neural emotion result.

        If no action is provided, generates a default action based on emotion.

        Args:
            emotion_result: The detected emotion (NeuralEmotionResult).
            action: Optional pre-generated action command.

        Returns:
            True if execution was successful.
        """
        if action is None:
            action = self._generate_default_action(emotion_result)

        return self.execute(action)

    def _generate_default_action(self, emotion: NeuralEmotionResult) -> ActionCommand:
        """Generate a default action based on detected emotion.

        Args:
            emotion: The neural emotion result.

        Returns:
            Default action command for the emotion.
        """
        emotion_label = emotion.dominant_emotion

        default_actions = {
            "happy": ActionCommand(
                action_type="acknowledge",
                parameters={"gesture": "nod", "intensity": 0.8},
                confidence=0.9,
            ),
            "sad": ActionCommand(
                action_type="comfort",
                parameters={"gesture": "approach_slowly", "intensity": 0.5},
                confidence=0.7,
            ),
            "angry": ActionCommand(
                action_type="de_escalate",
                parameters={"gesture": "step_back", "intensity": 0.3},
                confidence=0.8,
            ),
            "fearful": ActionCommand(
                action_type="reassure",
                parameters={"gesture": "slow_wave", "intensity": 0.4},
                confidence=0.7,
            ),
            "surprised": ActionCommand(
                action_type="wait",
                parameters={"duration": 2.0},
                confidence=0.9,
            ),
            "disgusted": ActionCommand(
                action_type="retreat",
                parameters={"distance": 0.5},
                confidence=0.8,
            ),
            "neutral": ActionCommand(
                action_type="idle",
                parameters={},
                confidence=1.0,
            ),
            "unclear": ActionCommand(
                action_type="idle",
                parameters={"reason": "no_clear_subject"},
                confidence=0.5,
            ),
        }

        return default_actions.get(
            emotion_label,
            ActionCommand(action_type="idle", parameters={}, confidence=0.5),
        )

    def get_supported_actions(self) -> list[str]:
        """Get list of action types supported by this handler.

        Returns:
            List of supported action type names.
        """
        return [
            "idle",
            "acknowledge",
            "comfort",
            "de_escalate",
            "reassure",
            "wait",
            "retreat",
            "approach",
            "gesture",
            "speak",
        ]

    def validate_action(self, action: ActionCommand) -> tuple[bool, str]:
        """Validate an action command before execution.

        Args:
            action: The action command to validate.

        Returns:
            Tuple of (is_valid, error_message).
        """
        supported = self.get_supported_actions()

        if action.action_type not in supported:
            return False, f"Unsupported action type: {action.action_type}"

        return True, ""

    def __enter__(self) -> "BaseActionHandler":
        """Context manager entry - connect."""
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - disconnect."""
        self.disconnect()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, connected={self._is_connected})"

