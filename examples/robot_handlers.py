#!/usr/bin/env python3
"""Example demonstrating different robot action handlers.

This example shows how to use the built-in action handlers to connect
emotion detection to real robot systems.

Available handlers:
    - LoggingActionHandler: Testing and debugging (default)
    - HTTPActionHandler: REST API integration
    - WebSocketActionHandler: Real-time WebSocket control
    - SerialActionHandler: Arduino/UART communication
    - ROSActionHandler: ROS1/ROS2 topic publishing

Usage:
    python robot_handlers.py --handler logging
    python robot_handlers.py --handler http --endpoint http://localhost:8080/api/action
    python robot_handlers.py --handler serial --port /dev/ttyUSB0
    python robot_handlers.py --handler ros --topic /robot/action
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from emotion_detection_action import Config, EmotionDetector, NeuralEmotionResult
from emotion_detection_action.actions import (
    LoggingActionHandler,
    HTTPActionHandler,
    SerialActionHandler,
    ROSActionHandler,
    WebSocketActionHandler,
)
from emotion_detection_action.core.types import ActionCommand


def create_handler(args: argparse.Namespace) -> object | None:
    """Create the appropriate action handler based on arguments."""
    if args.handler == "logging":
        return LoggingActionHandler(verbose=True)

    elif args.handler == "http":
        if not args.endpoint:
            print("Error: --endpoint required for HTTP handler")
            return None
        return HTTPActionHandler(
            endpoint=args.endpoint,
            method="POST",
            timeout=5.0,
        )

    elif args.handler == "websocket":
        if not args.endpoint:
            print("Error: --endpoint required for WebSocket handler")
            return None
        return WebSocketActionHandler(url=args.endpoint)

    elif args.handler == "serial":
        if not args.port:
            print("Using auto-detect for serial port...")
        return SerialActionHandler(
            port=args.port,
            baudrate=args.baudrate,
            message_format=args.format,
            auto_detect=args.port is None,
        )

    elif args.handler == "ros":
        return ROSActionHandler(
            node_name="emotion_detector_example",
            action_topic=args.topic or "/emotion_action",
        )

    else:
        print(f"Unknown handler: {args.handler}")
        return None


def test_handler_standalone(handler: object) -> bool:
    """Test the handler with synthetic action commands."""
    print(f"\n{'='*50}")
    print(f"Testing {handler.name} handler standalone")
    print(f"{'='*50}\n")

    # Connect
    if not handler.connect():
        print("Failed to connect handler!")
        return False
    print("Handler connected successfully")

    # Test actions
    test_actions = [
        ActionCommand(
            action_type="acknowledge",
            parameters={"gesture": "nod", "emotion": "happy"},
            confidence=0.9,
        ),
        ActionCommand(
            action_type="comfort",
            parameters={"gesture": "approach", "emotion": "sad"},
            confidence=0.8,
        ),
        ActionCommand(
            action_type="de_escalate",
            parameters={"gesture": "step_back", "emotion": "angry"},
            confidence=0.85,
        ),
        ActionCommand(
            action_type="idle",
            parameters={"emotion": "neutral"},
            confidence=1.0,
        ),
    ]

    print("\nSending test actions...")
    for i, action in enumerate(test_actions, 1):
        print(f"\n[{i}/{len(test_actions)}] Sending: {action.action_type}")
        success = handler.execute(action)
        print(f"  Result: {'Success' if success else 'Failed'}")
        time.sleep(0.5)

    # Disconnect
    handler.disconnect()
    print("\nHandler disconnected")
    return True


def demo_with_detector(handler: object, test_image: str | None = None) -> None:
    """Demo handler with actual emotion detection."""
    print(f"\n{'='*50}")
    print(f"Demo with EmotionDetector using {getattr(handler, 'name', handler)} handler")
    print(f"{'='*50}\n")

    config = Config(
        two_tower_pretrained=False,  # stub mode — no download required for demo
        two_tower_device="cpu",
        vla_enabled=False,
        verbose=False,
    )

    detector = EmotionDetector(config)
    detector.initialize()

    if test_image and Path(test_image).exists():
        try:
            import cv2
        except ImportError:
            raise RuntimeError("opencv-python required to load images: pip3 install opencv-python")

        print(f"Processing image: {test_image}")
        frame = cv2.imread(test_image)
        if frame is not None:
            # Accumulate enough frames for the rolling buffer, then run inference.
            import numpy as np
            for _ in range(20):
                detector.process_frame(frame)
            result: NeuralEmotionResult | None = detector.process_frame(frame)
            if result is not None:
                print(f"\nDetected  : {result.dominant_emotion}")
                print(f"Confidence: {result.confidence:.2%}")
                print(f"Metrics   : {result.metrics}")
            else:
                print("Detector returned no result yet (buffer still filling).")
        else:
            print(f"Failed to load image: {test_image}")
    else:
        print("No test image provided or not found.")
        print("Provide --image path/to/image.jpg to test with real detection")
        print("\nRunning with synthetic data instead...")
        test_handler_standalone(handler)

    detector.shutdown()


def show_serial_ports():
    """Display available serial ports."""
    print("\nAvailable serial ports:")
    print("-" * 40)
    ports = SerialActionHandler.list_ports()
    if not ports:
        print("  No serial ports found")
    for port in ports:
        print(f"  {port['device']}: {port['description']}")
    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Demonstrate emotion detection with different robot action handlers"
    )
    parser.add_argument(
        "--handler",
        type=str,
        default="logging",
        choices=["logging", "http", "websocket", "serial", "ros"],
        help="Action handler type to use",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        help="URL endpoint for HTTP/WebSocket handlers",
    )
    parser.add_argument(
        "--port",
        type=str,
        help="Serial port for Serial handler (e.g., /dev/ttyUSB0, COM3)",
    )
    parser.add_argument(
        "--baudrate",
        type=int,
        default=115200,
        help="Baud rate for Serial handler",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="json",
        choices=["json", "csv", "binary", "simple"],
        help="Message format for Serial handler",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="/emotion_action",
        help="ROS topic for ROS handler",
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to test image for emotion detection",
    )
    parser.add_argument(
        "--list-ports",
        action="store_true",
        help="List available serial ports and exit",
    )

    args = parser.parse_args()

    # List ports if requested
    if args.list_ports:
        show_serial_ports()
        return

    print("=" * 50)
    print("Robot Action Handler Demo")
    print("=" * 50)

    # Create handler
    handler = create_handler(args)
    if handler is None:
        return

    print(f"\nUsing handler: {handler.name}")

    # Run demo
    if args.image:
        demo_with_detector(handler, args.image)
    else:
        test_handler_standalone(handler)

    print("\n" + "=" * 50)
    print("Demo completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
