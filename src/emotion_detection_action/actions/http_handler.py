"""HTTP/REST action handler for robot control via web APIs."""

import json
import logging
import ssl
from typing import Any, Callable
from urllib import request, error

_log = logging.getLogger(__name__)

from emotion_detection_action.actions.base import BaseActionHandler
from emotion_detection_action.core.types import ActionCommand


class HTTPActionHandler(BaseActionHandler):
    """Action handler that sends commands via HTTP REST API.

    Useful for robots with web-based control interfaces, cloud services,
    or any system with a REST API endpoint.

    Example:
        >>> handler = HTTPActionHandler(
        ...     endpoint="http://robot.local:8080/api/action",
        ...     headers={"Authorization": "Bearer token123"}
        ... )
        >>> handler.connect()
        >>> handler.execute(action_command)
    """

    def __init__(
        self,
        endpoint: str,
        name: str = "http",
        method: str = "POST",
        headers: dict[str, str] | None = None,
        timeout: float = 5.0,
        verify_ssl: bool = True,
        on_response: Callable[[dict], None] | None = None,
    ) -> None:
        """Initialize HTTP action handler.

        Args:
            endpoint: URL endpoint to send actions to.
            name: Handler identifier name.
            method: HTTP method (POST, PUT, PATCH).
            headers: Additional HTTP headers.
            timeout: Request timeout in seconds.
            verify_ssl: Whether to verify SSL certificates.
            on_response: Optional callback for API responses.
        """
        super().__init__(name)
        self.endpoint = endpoint
        self.method = method.upper()
        self.headers = headers or {}
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self.on_response = on_response

        # Statistics
        self._success_count = 0
        self._error_count = 0
        self._last_response: dict | None = None

    def connect(self) -> bool:
        """Verify connection to the HTTP endpoint.

        Sends a HEAD request to verify the endpoint is reachable.

        Returns:
            True if the endpoint responded (any HTTP status code counts).
            Returns False on network-level errors (DNS failure, refused connection).
        """
        import warnings

        if not self.verify_ssl:
            warnings.warn(
                "HTTPActionHandler: SSL certificate verification is disabled "
                "(verify_ssl=False). This exposes the connection to man-in-the-middle "
                "attacks. Never use in production with sensitive data.",
                stacklevel=2,
            )

        try:
            req = request.Request(
                self.endpoint,
                method="HEAD",
                headers=self.headers,
            )
            request.urlopen(req, timeout=self.timeout, context=self._ssl_context())
            self._is_connected = True
            return True

        except error.HTTPError:
            # HTTP error (4xx/5xx) still means the server is reachable.
            self._is_connected = True
            return True
        except error.URLError:
            # Network-level failure — endpoint is genuinely unreachable.
            self._is_connected = False
            return False
        except Exception as exc:
            _log.warning("HTTPActionHandler.connect() failed: %s", exc, exc_info=True)
            self._is_connected = False
            return False

    def disconnect(self) -> None:
        """Disconnect from the HTTP endpoint."""
        self._is_connected = False

    def execute(self, action: ActionCommand) -> bool:
        """Send action command via HTTP.

        Args:
            action: The action command to send.

        Returns:
            True if request was successful.
        """
        if not self._is_connected:
            return False

        # Build request payload
        payload = self._build_payload(action)

        try:
            data = json.dumps(payload).encode("utf-8")
            headers = {
                "Content-Type": "application/json",
                **self.headers,
            }
            req = request.Request(
                self.endpoint,
                data=data,
                method=self.method,
                headers=headers,
            )

            with request.urlopen(req, timeout=self.timeout, context=self._ssl_context()) as response:
                response_data = response.read().decode("utf-8")

                try:
                    self._last_response = json.loads(response_data)
                except json.JSONDecodeError:
                    self._last_response = {"raw": response_data}

                if self.on_response:
                    self.on_response(self._last_response)

            self._success_count += 1
            return True

        except error.HTTPError as e:
            self._error_count += 1
            self._last_response = {"error": str(e), "code": e.code}
            _log.warning("HTTPActionHandler.execute() HTTP error %s: %s", e.code, e)
            return False

        except error.URLError as e:
            self._error_count += 1
            self._last_response = {"error": str(e)}
            _log.warning("HTTPActionHandler.execute() network error: %s", e)
            return False

        except Exception as e:
            self._error_count += 1
            self._last_response = {"error": str(e)}
            _log.warning("HTTPActionHandler.execute() unexpected error: %s", e, exc_info=True)
            return False

    def _ssl_context(self) -> ssl.SSLContext | None:
        """Return an SSL context, or None to use the default verified context."""
        if not self.verify_ssl:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            return ctx
        return None

    def _build_payload(self, action: ActionCommand) -> dict[str, Any]:
        """Build HTTP payload from action command.

        Args:
            action: Action command to convert.

        Returns:
            Dictionary payload for HTTP request.
        """
        return {
            "action_type": action.action_type,
            "parameters": action.parameters,
            "confidence": action.confidence,
        }

    def get_last_response(self) -> dict | None:
        """Get the last API response.

        Returns:
            Last response dictionary or None.
        """
        return self._last_response

    def get_statistics(self) -> dict[str, Any]:
        """Get handler statistics.

        Returns:
            Dictionary with success/error counts.
        """
        return {
            "success_count": self._success_count,
            "error_count": self._error_count,
            "success_rate": (
                self._success_count / (self._success_count + self._error_count)
                if (self._success_count + self._error_count) > 0
                else 0.0
            ),
        }


class WebSocketActionHandler(BaseActionHandler):
    """Action handler that sends commands via WebSocket.

    Useful for real-time robot control with low latency bidirectional
    communication.

    Note: Requires the 'websockets' package for async WebSocket support,
    or falls back to synchronous implementation.

    Example:
        >>> handler = WebSocketActionHandler(
        ...     url="ws://robot.local:8080/ws/actions"
        ... )
        >>> handler.connect()
        >>> handler.execute(action_command)
    """

    def __init__(
        self,
        url: str,
        name: str = "websocket",
        on_message: Callable[[str], None] | None = None,
        reconnect: bool = True,
        ping_interval: float = 30.0,
    ) -> None:
        """Initialize WebSocket action handler.

        Args:
            url: WebSocket URL (ws:// or wss://).
            name: Handler identifier name.
            on_message: Callback for incoming messages.
            reconnect: Whether to auto-reconnect on disconnect.
            ping_interval: Ping interval in seconds to keep connection alive.
        """
        super().__init__(name)
        self.url = url
        self.on_message = on_message
        self.reconnect = reconnect
        self.ping_interval = ping_interval

        self._ws: Any = None
        self._message_count = 0

    def connect(self) -> bool:
        """Connect to the WebSocket server.

        Returns:
            True if connection successful.
        """
        try:
            # Try to use websocket-client (sync library)
            import websocket

            self._ws = websocket.create_connection(
                self.url,
                timeout=10,
            )
            self._is_connected = True
            return True

        except ImportError:
            # websocket-client not available
            print(
                "Warning: websocket-client not installed. "
                "Install with: pip install websocket-client"
            )
            return False

        except Exception as e:
            print(f"WebSocket connection failed: {e}")
            self._is_connected = False
            return False

    def disconnect(self) -> None:
        """Disconnect from the WebSocket server."""
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception as exc:
                _log.debug("WebSocketActionHandler.disconnect() close error: %s", exc)
            self._ws = None
        self._is_connected = False

    def execute(self, action: ActionCommand) -> bool:
        """Send action command via WebSocket.

        Args:
            action: The action command to send.

        Returns:
            True if message was sent successfully.
        """
        if not self._is_connected or self._ws is None:
            if self.reconnect:
                if not self.connect():
                    return False
            else:
                return False

        try:
            # Build message
            message = json.dumps({
                "action_type": action.action_type,
                "parameters": action.parameters,
                "confidence": action.confidence,
            })

            self._ws.send(message)
            self._message_count += 1

            # Try to receive response (non-blocking); timeout is expected.
            try:
                self._ws.settimeout(0.1)
                response = self._ws.recv()
                if self.on_message:
                    self.on_message(response)
            except Exception:
                pass  # Timeout or no data available — not an error

            return True

        except Exception as e:
            print(f"WebSocket send failed: {e}")
            self._is_connected = False
            return False

    def receive(self, timeout: float = 1.0) -> str | None:
        """Receive a message from the WebSocket.

        Args:
            timeout: Receive timeout in seconds.

        Returns:
            Received message or None.
        """
        if not self._is_connected or self._ws is None:
            return None

        try:
            self._ws.settimeout(timeout)
            return self._ws.recv()
        except Exception:
            return None

    @property
    def message_count(self) -> int:
        """Get the number of messages sent."""
        return self._message_count
