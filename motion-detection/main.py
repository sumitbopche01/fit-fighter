"""
FitFighter Motion Detection Proof of Concept

This is the main application for the FitFighter motion detection proof of concept.
It demonstrates camera input processing and basic exercise detection.
"""

import cv2
import time
import argparse
from camera_utils import CameraManager, preprocess_frame, display_frame
from pose_detector import PoseDetector
from motion_analyzer import MotionAnalyzer


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

            # Add landmarks to analyzer
            if landmarks:
                motion_analyzer.add_landmarks(landmarks)

            # Analyze motion
            exercise_states = motion_analyzer.analyze_motion()

            # Draw landmarks on frame
            pose_frame = pose_detector.draw_landmarks(frame, results)

            # Add exercise status to frame
            add_status_to_frame(pose_frame, exercise_states)

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


if __name__ == "__main__":
    main()
