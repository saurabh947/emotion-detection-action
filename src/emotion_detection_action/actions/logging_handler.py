"""Logging action handler for testing and development."""

from typing import Any, Callable

from emotion_detection_action.actions.base import BaseActionHandler
from emotion_detection_action.core.types import ActionCommand, EmotionResult


class LoggingActionHandler(BaseActionHandler):
    """Action handler that logs actions without executing them.

    This handler is useful for:
    - Testing the emotion detection pipeline without robot hardware
    - Development and debugging
    - Demonstrating the SDK capabilities
    - Integration testing

    Example:
        >>> handler = LoggingActionHandler(verbose=True)
        >>> handler.connect()
        >>> handler.execute(action_command)  # Logs instead of executing
        >>> handler.get_action_history()  # Get all logged actions
    """

    def __init__(
        self,
        name: str = "logging",
        verbose: bool = True,
        callback: Callable[[ActionCommand], None] | None = None,
    ) -> None:
        """Initialize logging action handler.

        Args:
            name: Handler identifier name.
            verbose: Whether to print actions to stdout.
            callback: Optional callback function called for each action.
        """
        super().__init__(name)
        self.verbose = verbose
        self.callback = callback

        self._action_history: list[dict[str, Any]] = []
        self._execution_count = 0

    def connect(self) -> bool:
        """Simulate connecting to a robot.

        Returns:
            Always returns True for logging handler.
        """
        self._is_connected = True
        if self.verbose:
            print(f"[{self.name}] Connected (logging mode - no robot attached)")
        return True

    def disconnect(self) -> None:
        """Simulate disconnecting from a robot."""
        self._is_connected = False
        if self.verbose:
            print(f"[{self.name}] Disconnected")

    def execute(self, action: ActionCommand) -> bool:
        """Log an action command without actually executing it.

        Args:
            action: The action command to "execute".

        Returns:
            Always returns True for logging handler.
        """
        self._execution_count += 1

        # Record action in history
        record = {
            "index": self._execution_count,
            "action_type": action.action_type,
            "parameters": action.parameters.copy(),
            "confidence": action.confidence,
            "raw_output": action.raw_output,
        }
        self._action_history.append(record)

        # Print if verbose
        if self.verbose:
            self._print_action(action)

        # Call callback if provided
        if self.callback is not None:
            self.callback(action)

        return True

    def _print_action(self, action: ActionCommand) -> None:
        """Pretty print an action command.

        Args:
            action: Action to print.
        """
        print(f"\n[{self.name}] Action #{self._execution_count}")
        print(f"  Type: {action.action_type}")
        print(f"  Confidence: {action.confidence:.2f}")

        if action.parameters:
            print("  Parameters:")
            for key, value in action.parameters.items():
                if key == "raw_response":
                    # Truncate long responses
                    value_str = str(value)[:100] + "..." if len(str(value)) > 100 else value
                    print(f"    {key}: {value_str}")
                else:
                    print(f"    {key}: {value}")

    def get_action_history(self) -> list[dict[str, Any]]:
        """Get the history of all executed actions.

        Returns:
            List of action records.
        """
        return self._action_history.copy()

    def get_last_action(self) -> dict[str, Any] | None:
        """Get the most recently executed action.

        Returns:
            Last action record or None if no actions executed.
        """
        if self._action_history:
            return self._action_history[-1].copy()
        return None

    def clear_history(self) -> None:
        """Clear the action history."""
        self._action_history.clear()
        self._execution_count = 0

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about executed actions.

        Returns:
            Dictionary with action statistics.
        """
        if not self._action_history:
            return {
                "total_actions": 0,
                "action_types": {},
                "avg_confidence": 0.0,
            }

        action_types: dict[str, int] = {}
        total_confidence = 0.0

        for record in self._action_history:
            action_type = record["action_type"]
            action_types[action_type] = action_types.get(action_type, 0) + 1
            total_confidence += record["confidence"]

        return {
            "total_actions": len(self._action_history),
            "action_types": action_types,
            "avg_confidence": total_confidence / len(self._action_history),
        }


class MockActionHandler(LoggingActionHandler):
    """Mock action handler for unit testing.

    Extends LoggingActionHandler with additional testing utilities
    like assertions and expected action sequences.

    Example:
        >>> handler = MockActionHandler()
        >>> handler.expect_action("acknowledge")
        >>> handler.execute(ActionCommand(action_type="acknowledge", ...))
        >>> handler.verify_expectations()  # Passes
    """

    def __init__(self, name: str = "mock") -> None:
        """Initialize mock handler.

        Args:
            name: Handler identifier name.
        """
        super().__init__(name, verbose=False)
        self._expected_actions: list[str] = []
        self._received_actions: list[str] = []

    def expect_action(self, action_type: str) -> None:
        """Add an expected action type.

        Args:
            action_type: Expected action type.
        """
        self._expected_actions.append(action_type)

    def expect_actions(self, action_types: list[str]) -> None:
        """Add multiple expected action types.

        Args:
            action_types: List of expected action types.
        """
        self._expected_actions.extend(action_types)

    def execute(self, action: ActionCommand) -> bool:
        """Execute and record action for verification.

        Args:
            action: The action command to execute.

        Returns:
            True if execution successful.
        """
        self._received_actions.append(action.action_type)
        return super().execute(action)

    def verify_expectations(self) -> tuple[bool, str]:
        """Verify all expected actions were received.

        Returns:
            Tuple of (success, error_message).
        """
        if len(self._received_actions) != len(self._expected_actions):
            return False, (
                f"Expected {len(self._expected_actions)} actions, "
                f"received {len(self._received_actions)}"
            )

        for i, (expected, received) in enumerate(
            zip(self._expected_actions, self._received_actions)
        ):
            if expected != received:
                return False, (
                    f"Action {i}: expected '{expected}', received '{received}'"
                )

        return True, "All expectations met"

    def reset_expectations(self) -> None:
        """Reset expected and received actions."""
        self._expected_actions.clear()
        self._received_actions.clear()
        self.clear_history()

    def assert_action_executed(self, action_type: str) -> None:
        """Assert that a specific action type was executed.

        Args:
            action_type: Action type to check for.

        Raises:
            AssertionError: If action was not executed.
        """
        if action_type not in self._received_actions:
            raise AssertionError(
                f"Action '{action_type}' was not executed. "
                f"Executed actions: {self._received_actions}"
            )

    def assert_no_actions(self) -> None:
        """Assert that no actions were executed.

        Raises:
            AssertionError: If any actions were executed.
        """
        if self._received_actions:
            raise AssertionError(
                f"Expected no actions, but received: {self._received_actions}"
            )

