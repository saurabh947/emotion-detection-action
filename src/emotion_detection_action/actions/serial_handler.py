"""Serial/UART action handler for robot control via serial communication.

Useful for Arduino, embedded systems, and other serial-connected devices.
"""

import json
import time
from typing import Any, Callable

from emotion_detection_action.actions.base import BaseActionHandler
from emotion_detection_action.core.types import ActionCommand

# Try to import pyserial
try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


class SerialActionHandler(BaseActionHandler):
    """Action handler that sends commands via serial/UART.

    Useful for:
    - Arduino and other microcontrollers
    - Embedded robot controllers
    - Serial-connected motor controllers
    - Any device with UART interface

    Supports multiple message formats: JSON, CSV, or custom.

    Example:
        >>> handler = SerialActionHandler(
        ...     port="/dev/ttyUSB0",
        ...     baudrate=115200,
        ...     message_format="json"
        ... )
        >>> handler.connect()
        >>> handler.execute(action_command)

    Arduino sketch example:
        ```cpp
        void loop() {
            if (Serial.available()) {
                String json = Serial.readStringUntil('\\n');
                // Parse and execute action
            }
        }
        ```
    """

    # Pre-defined action codes for efficient transmission
    ACTION_CODES = {
        "idle": 0,
        "acknowledge": 1,
        "comfort": 2,
        "de_escalate": 3,
        "reassure": 4,
        "wait": 5,
        "retreat": 6,
        "approach": 7,
        "greeting": 8,
        "gesture": 9,
        "speak": 10,
        "generated": 11,
        "stub": 12,
    }

    EMOTION_CODES = {
        "happy": 0,
        "sad": 1,
        "angry": 2,
        "fearful": 3,
        "surprised": 4,
        "disgusted": 5,
        "neutral": 6,
    }

    def __init__(
        self,
        port: str | None = None,
        baudrate: int = 115200,
        name: str = "serial",
        message_format: str = "json",
        timeout: float = 1.0,
        write_timeout: float = 1.0,
        end_marker: str = "\n",
        on_receive: Callable[[str], None] | None = None,
        auto_detect: bool = False,
    ) -> None:
        """Initialize serial action handler.

        Args:
            port: Serial port (e.g., "/dev/ttyUSB0", "COM3").
            baudrate: Baud rate for communication.
            name: Handler identifier name.
            message_format: Format for messages: "json", "csv", "binary", "simple".
            timeout: Read timeout in seconds.
            write_timeout: Write timeout in seconds.
            end_marker: Line ending marker.
            on_receive: Callback for received data.
            auto_detect: Auto-detect Arduino/serial port if port is None.
        """
        super().__init__(name)

        self.port = port
        self.baudrate = baudrate
        self.message_format = message_format
        self.timeout = timeout
        self.write_timeout = write_timeout
        self.end_marker = end_marker
        self.on_receive = on_receive
        self.auto_detect = auto_detect

        self._serial: Any = None
        self._bytes_sent = 0
        self._bytes_received = 0
        self._message_count = 0

    def connect(self) -> bool:
        """Connect to the serial port.

        Returns:
            True if connection successful.
        """
        if not SERIAL_AVAILABLE:
            raise RuntimeError(
                "pyserial not available. Install with: pip install pyserial"
            )

        # Auto-detect port if needed
        if self.port is None and self.auto_detect:
            self.port = self._auto_detect_port()
            if self.port is None:
                print("No serial port detected")
                return False

        if self.port is None:
            print("No serial port specified")
            return False

        try:
            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                write_timeout=self.write_timeout,
            )

            # Wait for Arduino reset (if applicable)
            time.sleep(2.0)

            # Clear any startup messages
            self._serial.reset_input_buffer()

            self._is_connected = True
            return True

        except serial.SerialException as e:
            print(f"Serial connection failed: {e}")
            self._is_connected = False
            return False

    def disconnect(self) -> None:
        """Disconnect from the serial port."""
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
            self._serial = None
        self._is_connected = False

    def execute(self, action: ActionCommand) -> bool:
        """Send action command via serial.

        Args:
            action: The action command to send.

        Returns:
            True if message was sent successfully.
        """
        if not self._is_connected or self._serial is None:
            return False

        try:
            message = self._format_message(action)
            message_bytes = message.encode("utf-8")

            self._serial.write(message_bytes)
            self._bytes_sent += len(message_bytes)
            self._message_count += 1

            # Read response if available
            if self._serial.in_waiting > 0:
                response = self._serial.readline().decode("utf-8").strip()
                self._bytes_received += len(response)
                if self.on_receive:
                    self.on_receive(response)

            return True

        except Exception as e:
            print(f"Serial write failed: {e}")
            return False

    def _format_message(self, action: ActionCommand) -> str:
        """Format action command as message string.

        Args:
            action: Action command to format.

        Returns:
            Formatted message string.
        """
        if self.message_format == "json":
            return self._format_json(action)
        elif self.message_format == "csv":
            return self._format_csv(action)
        elif self.message_format == "binary":
            return self._format_binary(action)
        elif self.message_format == "simple":
            return self._format_simple(action)
        else:
            return self._format_json(action)

    def _format_json(self, action: ActionCommand) -> str:
        """Format as JSON."""
        data = {
            "type": action.action_type,
            "params": action.parameters,
            "conf": round(action.confidence, 2),
        }
        return json.dumps(data, separators=(",", ":")) + self.end_marker

    def _format_csv(self, action: ActionCommand) -> str:
        """Format as CSV: action_code,emotion_code,confidence"""
        action_code = self.ACTION_CODES.get(action.action_type, 99)
        emotion = action.parameters.get("emotion", "neutral")
        emotion_code = self.EMOTION_CODES.get(emotion, 6)
        confidence = int(action.confidence * 100)
        return f"{action_code},{emotion_code},{confidence}{self.end_marker}"

    def _format_binary(self, action: ActionCommand) -> str:
        """Format as compact binary-like string: A<code>E<code>C<conf>"""
        action_code = self.ACTION_CODES.get(action.action_type, 99)
        emotion = action.parameters.get("emotion", "neutral")
        emotion_code = self.EMOTION_CODES.get(emotion, 6)
        confidence = int(action.confidence * 100)
        return f"A{action_code:02d}E{emotion_code}C{confidence:03d}{self.end_marker}"

    def _format_simple(self, action: ActionCommand) -> str:
        """Format as simple human-readable: ACTION:params"""
        params_str = ",".join(f"{k}={v}" for k, v in action.parameters.items())
        return f"{action.action_type}:{params_str}{self.end_marker}"

    def _auto_detect_port(self) -> str | None:
        """Auto-detect Arduino or similar serial port.

        Returns:
            Detected port path or None.
        """
        common_descriptions = [
            "arduino",
            "ch340",
            "cp210",
            "ftdi",
            "usb serial",
            "usb-serial",
        ]

        ports = serial.tools.list_ports.comports()
        for port in ports:
            desc_lower = (port.description or "").lower()
            for keyword in common_descriptions:
                if keyword in desc_lower:
                    return port.device

        # Fall back to first available USB port
        for port in ports:
            if "usb" in port.device.lower():
                return port.device

        return None

    @staticmethod
    def list_ports() -> list[dict[str, Any]]:
        """List available serial ports.

        Returns:
            List of port info dictionaries.
        """
        if not SERIAL_AVAILABLE:
            return []

        ports = serial.tools.list_ports.comports()
        return [
            {
                "device": port.device,
                "description": port.description,
                "hwid": port.hwid,
            }
            for port in ports
        ]

    def get_statistics(self) -> dict[str, Any]:
        """Get handler statistics.

        Returns:
            Dictionary with bytes sent/received and message count.
        """
        return {
            "bytes_sent": self._bytes_sent,
            "bytes_received": self._bytes_received,
            "message_count": self._message_count,
            "port": self.port,
            "baudrate": self.baudrate,
        }
