"""
FitFighter Motion Detection Proof of Concept

This is the main application for the FitFighter motion detection proof of concept.
It demonstrates camera input processing and basic exercise detection.
"""

import cv2
import time
import argparse
from camera_utils import CameraManager, preprocess_frame, display_frame
from pose_detector import PoseDetector, LANDMARK_INDICES
from motion_analyzer import MotionAnalyzer


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


def main():
    """Main entry point for the application."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="FitFighter Motion Detection PoC")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument("--width", type=int, default=640, help="Camera frame width")
    parser.add_argument("--height", type=int, default=480, help="Camera frame height")
    parser.add_argument(
        "--model-complexity",
        type=int,
        default=1,
        help="MediaPipe model complexity (0, 1, or 2)",
    )
    args = parser.parse_args()

    # Initialize components
    camera = CameraManager(camera_id=args.camera, width=args.width, height=args.height)
    pose_detector = PoseDetector(model_complexity=args.model_complexity)
    motion_analyzer = MotionAnalyzer()
    visibility_checker = VisibilityChecker()

    # Start camera
    if not camera.start():
        print("Failed to open camera. Exiting.")
        return

    print("FitFighter Motion Detection PoC")
    print("Press 'q' to quit")

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

            # Check visibility of body parts
            visibility_status = None
            if landmarks:
                # Add landmarks to analyzer
                motion_analyzer.add_landmarks(landmarks)
                # Check visibility
                visibility_status = visibility_checker.check_visibility(landmarks)

            # Analyze motion
            exercise_states = motion_analyzer.analyze_motion()

            # Draw landmarks on frame
            pose_frame = pose_detector.draw_landmarks(frame, results)

            # Add exercise status to frame
            add_status_to_frame(pose_frame, exercise_states)

            # Add visibility guidance if needed
            if visibility_status and not visibility_status["full_body_visible"]:
                add_visibility_guidance(pose_frame, visibility_status)

            # Add FPS counter
            pose_frame = camera.add_fps_to_frame(pose_frame)

            # Display frame
            display_frame(pose_frame)

            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        # Clean up
        camera.release()
        pose_detector.release()
        cv2.destroyAllWindows()
        print("Application closed")


def add_status_to_frame(frame, exercise_states):
    """
    Add exercise status text to the frame.

    Args:
        frame: Frame to add text to
        exercise_states: Dictionary of exercise states
    """
    # Background for status text
    cv2.rectangle(frame, (10, 50), (250, 150), (0, 0, 0), -1)

    # Add status text
    for i, (exercise, detected) in enumerate(exercise_states.items()):
        status = "DETECTED" if detected else "Not detected"
        color = (0, 255, 0) if detected else (0, 165, 255)

        cv2.putText(
            frame,
            f"{exercise.capitalize()}: {status}",
            (20, 80 + (i * 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
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


if __name__ == "__main__":
    main()
