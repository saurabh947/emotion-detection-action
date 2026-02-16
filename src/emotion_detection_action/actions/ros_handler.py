"""ROS/ROS2 action handler for robot control via ROS topics and services.

Supports both ROS1 (rospy) and ROS2 (rclpy) interfaces.
"""

import json
from typing import Any, Callable

from emotion_detection_action.actions.base import BaseActionHandler
from emotion_detection_action.core.types import ActionCommand, EmotionResult

# Try to import ROS libraries
ROS1_AVAILABLE = False
ROS2_AVAILABLE = False

try:
    import rospy
    from std_msgs.msg import String
    ROS1_AVAILABLE = True
except ImportError:
    pass

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String as String2
    ROS2_AVAILABLE = True
except ImportError:
    pass


class ROSActionHandler(BaseActionHandler):
    """Action handler that publishes to ROS topics.

    Supports both ROS1 (rospy) and ROS2 (rclpy). Automatically detects
    which version is available.

    Topics published:
    - /emotion_action (String): JSON-encoded action commands
    - /emotion_result (String): JSON-encoded emotion results (optional)

    Example (ROS1):
        >>> handler = ROSActionHandler(
        ...     node_name="emotion_action_node",
        ...     action_topic="/robot/emotion_action"
        ... )
        >>> handler.connect()
        >>> handler.execute(action_command)

    Example (ROS2):
        >>> # Initialize rclpy first
        >>> import rclpy
        >>> rclpy.init()
        >>> handler = ROSActionHandler(ros_version=2)
        >>> handler.connect()
        >>> handler.execute(action_command)
    """

    def __init__(
        self,
        node_name: str = "emotion_detector",
        action_topic: str = "/emotion_action",
        emotion_topic: str | None = "/emotion_result",
        name: str = "ros",
        ros_version: int | None = None,
        queue_size: int = 10,
        publish_emotion: bool = True,
    ) -> None:
        """Initialize ROS action handler.

        Args:
            node_name: Name for the ROS node.
            action_topic: Topic to publish action commands.
            emotion_topic: Topic to publish emotion results (optional).
            name: Handler identifier name.
            ros_version: Force ROS version (1 or 2). Auto-detect if None.
            queue_size: Publisher queue size.
            publish_emotion: Whether to also publish emotion results.
        """
        super().__init__(name)

        self.node_name = node_name
        self.action_topic = action_topic
        self.emotion_topic = emotion_topic
        self.queue_size = queue_size
        self._publish_emotion_enabled = publish_emotion

        # Determine ROS version
        if ros_version is not None:
            self.ros_version = ros_version
        elif ROS2_AVAILABLE:
            self.ros_version = 2
        elif ROS1_AVAILABLE:
            self.ros_version = 1
        else:
            self.ros_version = 0

        self._node: Any = None
        self._action_pub: Any = None
        self._emotion_pub: Any = None
        self._message_count = 0

    def connect(self) -> bool:
        """Initialize ROS node and publishers.

        Returns:
            True if initialization successful.
        """
        if self.ros_version == 2:
            return self._connect_ros2()
        elif self.ros_version == 1:
            return self._connect_ros1()
        else:
            print(
                "No ROS installation detected. Install ROS1 (rospy) or ROS2 (rclpy)."
            )
            return False

    def _connect_ros1(self) -> bool:
        """Initialize ROS1 node and publishers."""
        try:
            # Check if node is already initialized
            if not rospy.core.is_initialized():
                rospy.init_node(self.node_name, anonymous=True)

            # Create publishers
            self._action_pub = rospy.Publisher(
                self.action_topic,
                String,
                queue_size=self.queue_size,
            )

            if self.publish_emotion and self.emotion_topic:
                self._emotion_pub = rospy.Publisher(
                    self.emotion_topic,
                    String,
                    queue_size=self.queue_size,
                )

            self._is_connected = True
            return True

        except Exception as e:
            print(f"ROS1 initialization failed: {e}")
            return False

    def _connect_ros2(self) -> bool:
        """Initialize ROS2 node and publishers."""
        try:
            # Initialize rclpy if not done
            if not rclpy.ok():
                rclpy.init()

            # Create node
            self._node = rclpy.create_node(self.node_name)

            # Create publishers
            self._action_pub = self._node.create_publisher(
                String2,
                self.action_topic,
                self.queue_size,
            )

            if self.publish_emotion and self.emotion_topic:
                self._emotion_pub = self._node.create_publisher(
                    String2,
                    self.emotion_topic,
                    self.queue_size,
                )

            self._is_connected = True
            return True

        except Exception as e:
            print(f"ROS2 initialization failed: {e}")
            return False

    def disconnect(self) -> None:
        """Shutdown ROS node."""
        if self.ros_version == 2 and self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:
                pass
            self._node = None

        # ROS1 cleanup is handled by rospy shutdown

        self._action_pub = None
        self._emotion_pub = None
        self._is_connected = False

    def execute(self, action: ActionCommand) -> bool:
        """Publish action command to ROS topic.

        Args:
            action: The action command to publish.

        Returns:
            True if published successfully.
        """
        if not self._is_connected or self._action_pub is None:
            return False

        try:
            # Build message
            message_data = {
                "action_type": action.action_type,
                "parameters": action.parameters,
                "confidence": action.confidence,
            }
            message_json = json.dumps(message_data)

            # Publish
            if self.ros_version == 2:
                msg = String2()
                msg.data = message_json
                self._action_pub.publish(msg)
            else:
                msg = String()
                msg.data = message_json
                self._action_pub.publish(msg)

            self._message_count += 1
            return True

        except Exception as e:
            print(f"ROS publish failed: {e}")
            return False

    def _publish_emotion_to_topic(self, emotion: EmotionResult) -> bool:
        """Publish emotion result to ROS topic.

        Args:
            emotion: Emotion result to publish.

        Returns:
            True if published successfully.
        """
        if not self._is_connected or self._emotion_pub is None:
            return False

        try:
            message_data = {
                "timestamp": emotion.timestamp,
                "dominant_emotion": emotion.dominant_emotion.value,
                "emotions": emotion.emotions.to_dict(),
                "fusion_confidence": emotion.fusion_confidence,
            }
            message_json = json.dumps(message_data)

            if self.ros_version == 2:
                msg = String2()
                msg.data = message_json
                self._emotion_pub.publish(msg)
            else:
                msg = String()
                msg.data = message_json
                self._emotion_pub.publish(msg)

            return True

        except Exception as e:
            print(f"ROS emotion publish failed: {e}")
            return False

    def execute_for_emotion(
        self,
        emotion_result: EmotionResult,
        action: ActionCommand | None = None,
    ) -> bool:
        """Execute action and optionally publish emotion.

        Args:
            emotion_result: The detected emotion.
            action: Optional pre-generated action command.

        Returns:
            True if execution was successful.
        """
        # Publish emotion if enabled
        if self._publish_emotion_enabled and self._emotion_pub is not None:
            self._publish_emotion_to_topic(emotion_result)

        # Execute action
        return super().execute_for_emotion(emotion_result, action)

    @staticmethod
    def is_ros_available() -> dict[str, bool]:
        """Check ROS availability.

        Returns:
            Dictionary with ROS1 and ROS2 availability.
        """
        return {
            "ros1": ROS1_AVAILABLE,
            "ros2": ROS2_AVAILABLE,
        }

    @property
    def message_count(self) -> int:
        """Get the number of messages published."""
        return self._message_count

    def get_statistics(self) -> dict[str, Any]:
        """Get handler statistics.

        Returns:
            Dictionary with message count and ROS info.
        """
        return {
            "message_count": self._message_count,
            "ros_version": self.ros_version,
            "action_topic": self.action_topic,
            "emotion_topic": self.emotion_topic,
        }


class ROS2ActionHandler(ROSActionHandler):
    """Convenience class for ROS2-specific handler."""

    def __init__(
        self,
        node_name: str = "emotion_detector",
        action_topic: str = "/emotion_action",
        **kwargs: Any,
    ) -> None:
        """Initialize ROS2 action handler.

        Args:
            node_name: Name for the ROS2 node.
            action_topic: Topic to publish action commands.
            **kwargs: Additional arguments for ROSActionHandler.
        """
        super().__init__(
            node_name=node_name,
            action_topic=action_topic,
            ros_version=2,
            **kwargs,
        )


class ROS1ActionHandler(ROSActionHandler):
    """Convenience class for ROS1-specific handler."""

    def __init__(
        self,
        node_name: str = "emotion_detector",
        action_topic: str = "/emotion_action",
        **kwargs: Any,
    ) -> None:
        """Initialize ROS1 action handler.

        Args:
            node_name: Name for the ROS1 node.
            action_topic: Topic to publish action commands.
            **kwargs: Additional arguments for ROSActionHandler.
        """
        super().__init__(
            node_name=node_name,
            action_topic=action_topic,
            ros_version=1,
            **kwargs,
        )
