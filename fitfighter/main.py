"""
FitFighter Main Application Module

This is the main application module for FitFighter, providing a complete
exercise detection and tracking solution.
"""

import cv2
import time
import argparse
import asyncio
import threading

from fitfighter.utils.camera_utils import (
    CameraManager,
    preprocess_frame,
    display_frame,
)
from fitfighter.core.pose_detector import PoseDetector
from fitfighter.core.detector_manager import ExerciseDetectorManager
from fitfighter.utils.data_sender import DataSender


class VisibilityChecker:
    """Checks if required body parts are visible in the frame."""

    def __init__(self, confidence_threshold=0.5):
        """
        Initialize the visibility checker.

        Args:
            confidence_threshold: Minimum confidence to consider a landmark visible
        """
        self.confidence_threshold = confidence_threshold
        self.key_landmarks = {
            "upper_body": [
                "left_shoulder",
                "right_shoulder",
                "left_elbow",
                "right_elbow",
                "left_wrist",
                "right_wrist",
                "left_hip",
                "right_hip",
            ],
            "lower_body": ["left_knee", "right_knee", "left_ankle", "right_ankle"],
        }

    def check_visibility(self, landmarks):
        """
        Check if all required body parts are visible.

        Args:
            landmarks: List of landmarks from pose detector

        Returns:
            dict: Visibility status for different body regions
        """
        if not landmarks:
            return {
                "full_body_visible": False,
                "upper_body_visible": False,
                "lower_body_visible": False,
                "missing_parts": self.key_landmarks["upper_body"]
                + self.key_landmarks["lower_body"],
            }

        from fitfighter.constants import LANDMARK_INDICES

        # Check upper body landmarks
        upper_body_visible = True
        missing_upper = []

        for landmark_name in self.key_landmarks["upper_body"]:
            idx = LANDMARK_INDICES[landmark_name]
            if landmarks[idx][3] < self.confidence_threshold:
                upper_body_visible = False
                missing_upper.append(landmark_name)

        # Check lower body landmarks
        lower_body_visible = True
        missing_lower = []

        for landmark_name in self.key_landmarks["lower_body"]:
            idx = LANDMARK_INDICES[landmark_name]
            if landmarks[idx][3] < self.confidence_threshold:
                lower_body_visible = False
                missing_lower.append(landmark_name)

        return {
            "full_body_visible": upper_body_visible and lower_body_visible,
            "upper_body_visible": upper_body_visible,
            "lower_body_visible": lower_body_visible,
            "missing_parts": missing_upper + missing_lower,
        }


class WebSocketThread(threading.Thread):
    """Thread for running the WebSocket server."""

    def __init__(self, host="127.0.0.1", port=5678):
        """Initialize the WebSocket thread."""
        super().__init__()
        self.host = host
        self.port = port
        self.data_sender = DataSender(host=host, port=port)
        self.loop = None
        self.daemon = True  # Thread will exit when main program exits

    def run(self):
        """Run the WebSocket server in a separate thread."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        # Start the WebSocket server
        self.loop.run_until_complete(self.data_sender.start_server())

        # Start the message sender task
        sender_task = self.loop.create_task(self.data_sender._send_to_clients())

        try:
            self.loop.run_forever()
        except Exception as e:
            print(f"WebSocket thread exception: {str(e)}")
        finally:
            self.loop.run_until_complete(self.data_sender.stop())
            self.loop.close()

    async def send_data(self, landmarks, exercise_states, fps):
        """
        Send data to connected clients.

        Args:
            landmarks: Pose landmarks
            exercise_states: Dictionary of exercise states
            fps: Current frames per second
        """
        if self.loop and self.data_sender:
            future = asyncio.run_coroutine_threadsafe(
                self.data_sender.send_data(landmarks, exercise_states, fps), self.loop
            )
            try:
                # Wait for the result with a timeout
                future.result(timeout=0.1)
            except (asyncio.TimeoutError, Exception) as e:
                pass  # Ignore timeout or other errors

    def close(self):
        """Close the WebSocket server."""
        if self.loop:
            asyncio.run_coroutine_threadsafe(self.data_sender.stop(), self.loop)


def add_status_to_frame(frame, detection_results):
    """
    Add exercise status text to the frame.

    Args:
        frame: Frame to add text to
        detection_results: Results from the detector manager
    """
    # Background for status text
    cv2.rectangle(frame, (10, 50), (350, 250), (0, 0, 0), -1)

    # Add active exercises
    cv2.putText(
        frame,
        "Active Exercises:",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    active_exercises = detection_results["active_exercises"]
    counts = detection_results["counts"]

    if not active_exercises:
        cv2.putText(
            frame,
            "None detected",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 165, 255),
            2,
        )
    else:
        for i, exercise in enumerate(active_exercises):
            count = counts.get(exercise, 0)
            cv2.putText(
                frame,
                f"{exercise.capitalize()}: {count} reps",
                (20, 110 + (i * 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

    # Add total reps
    total_reps = detection_results["session_stats"]["total_reps"]
    cv2.putText(
        frame,
        f"Total Reps: {total_reps}",
        (20, 220),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )


def add_visibility_guidance(frame, visibility_status):
    """
    Add guidance text when body parts are not visible.

    Args:
        frame: Frame to add text to
        visibility_status: Visibility status from VisibilityChecker
    """
    h, w = frame.shape[:2]

    # Create semi-transparent overlay for better text visibility
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 120), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    if not visibility_status["lower_body_visible"]:
        message = "MOVE BACK: Lower body not fully visible"
        color = (0, 0, 255)  # Red for important guidance
    elif not visibility_status["upper_body_visible"]:
        message = "MOVE BACK: Upper body not fully visible"
        color = (0, 0, 255)
    else:
        message = "Adjust position to ensure full visibility"
        color = (0, 165, 255)  # Orange for general guidance

    # Add primary guidance message
    cv2.putText(
        frame,
        message,
        (int(w / 2) - 250, h - 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        2,
        cv2.LINE_AA,
    )

    # Add additional detail about missing parts
    if len(visibility_status["missing_parts"]) > 0:
        parts_text = "Missing: " + ", ".join(
            [p.replace("_", " ") for p in visibility_status["missing_parts"][:3]]
        )
        if len(visibility_status["missing_parts"]) > 3:
            parts_text += f" and {len(visibility_status['missing_parts']) - 3} more"

        cv2.putText(
            frame,
            parts_text,
            (int(w / 2) - 250, h - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def main():
    """Main entry point for the application."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="FitFighter Application")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument("--width", type=int, default=640, help="Camera frame width")
    parser.add_argument("--height", type=int, default=480, help="Camera frame height")
    parser.add_argument(
        "--model-complexity",
        type=int,
        default=1,
        help="MediaPipe model complexity (0, 1, or 2)",
    )
    parser.add_argument(
        "--websocket-host", type=str, default="127.0.0.1", help="WebSocket server host"
    )
    parser.add_argument(
        "--websocket-port", type=int, default=5678, help="WebSocket server port"
    )
    parser.add_argument(
        "--enable-websocket", action="store_true", help="Enable WebSocket server"
    )
    args = parser.parse_args()

    # Initialize components
    camera = CameraManager(camera_id=args.camera, width=args.width, height=args.height)
    pose_detector = PoseDetector(model_complexity=args.model_complexity)
    detector_manager = ExerciseDetectorManager()
    visibility_checker = VisibilityChecker()

    # Start WebSocket server if enabled
    websocket_thread = None
    if args.enable_websocket:
        print(
            f"Starting WebSocket server at ws://{args.websocket_host}:{args.websocket_port}"
        )
        websocket_thread = WebSocketThread(
            host=args.websocket_host, port=args.websocket_port
        )
        websocket_thread.start()

    # Start camera
    if not camera.start():
        print("Failed to open camera. Exiting.")
        return

    print("FitFighter Application")
    print("Press 'q' to quit, 'r' to reset exercise counts")

    try:
        # Main loop
        while True:
            # Read frame from camera
            success, frame = camera.read_frame()
            if not success:
                print("Failed to read frame. Exiting.")
                break

            # Preprocess frame
            processed_frame = preprocess_frame(frame)

            # Detect pose
            results = pose_detector.process_frame(processed_frame)

            # Extract landmarks
            landmarks = pose_detector.get_pose_landmarks(results)

            # Process landmarks with detector manager
            detection_results = {}
            visibility_status = None

            if landmarks:
                # Check visibility of body parts
                visibility_status = visibility_checker.check_visibility(landmarks)

                # Process landmarks if body is visible
                detection_results = detector_manager.process_landmarks(landmarks)

            # Draw landmarks on frame
            pose_frame = pose_detector.draw_landmarks(frame, results)

            # Add exercise status to frame
            add_status_to_frame(pose_frame, detection_results)

            # Add visibility guidance if needed
            if visibility_status and not visibility_status["full_body_visible"]:
                add_visibility_guidance(pose_frame, visibility_status)

            # Add FPS counter
            pose_frame = camera.add_fps_to_frame(pose_frame)

            # Display frame
            display_frame(pose_frame)

            # Send data to WebSocket clients if enabled
            if args.enable_websocket and websocket_thread and landmarks:
                asyncio.run(
                    websocket_thread.send_data(
                        landmarks, detection_results, camera.get_fps()
                    )
                )

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                detector_manager.reset_session()
                print("Exercise counts reset")

    finally:
        # Clean up
        camera.release()
        pose_detector.release()
        if websocket_thread:
            websocket_thread.close()
        cv2.destroyAllWindows()
        print("Application closed")


if __name__ == "__main__":
    main()
