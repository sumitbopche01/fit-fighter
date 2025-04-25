#!/usr/bin/env python3
"""
Webcam demo for FitFighter exercise detection.

This script demonstrates real-time exercise detection using a webcam.
"""

import cv2
import mediapipe as mp
import argparse
import logging
import sys
import os

# Add the parent directory to sys.path to import fitfighter
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fitfighter.core.detector_manager import ExerciseDetectorManager
from fitfighter.utils.pose_processor import convert_mediapipe_landmarks
from fitfighter.utils.visualization import (
    draw_pose_landmarks,
    draw_detection_results,
    create_debug_visualization,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="FitFighter Webcam Demo")
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug visualization"
    )
    parser.add_argument(
        "--confidence", type=float, default=0.6, help="Landmark confidence threshold"
    )
    parser.add_argument("--history", type=int, default=30, help="Landmark history size")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--width", type=int, default=640, help="Camera width")
    parser.add_argument("--height", type=int, default=480, help="Camera height")
    return parser.parse_args()


def main():
    """Run the webcam demo."""
    # Parse command line arguments
    args = parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("fitfighter_demo")
    logger.info("Starting FitFighter Webcam Demo")

    # Initialize MediaPipe Pose
    logger.info("Initializing MediaPipe Pose")
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Initialize ExerciseDetectorManager
    logger.info("Initializing Exercise Detector Manager")
    detector_manager = ExerciseDetectorManager(
        confidence_threshold=args.confidence, history_size=args.history
    )

    # Print available exercises
    available_exercises = detector_manager.get_available_exercises()
    logger.info(f'Available exercise detectors: {", ".join(available_exercises)}')

    # Open webcam
    logger.info(f"Opening webcam at index {args.camera}")
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        logger.error("Failed to open webcam")
        return

    frame_count = 0

    try:
        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame from webcam")
                break

            # Flip the image horizontally for a selfie-view display
            frame = cv2.flip(frame, 1)

            # Convert the image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame with MediaPipe Pose
            results = pose.process(rgb_frame)

            # Create output frame
            output_frame = frame.copy()

            if results.pose_landmarks:
                # Draw pose landmarks on frame
                mp_drawing.draw_landmarks(
                    output_frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(
                        color=(0, 255, 0), thickness=2, circle_radius=2
                    ),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2),
                )

                # Convert MediaPipe landmarks to our format
                landmarks = convert_mediapipe_landmarks(results.pose_landmarks)

                # Process landmarks with our detector manager
                detection_results = detector_manager.process_landmarks(landmarks)

                # Draw detection results on output frame
                output_frame = draw_detection_results(output_frame, detection_results)

                # Print debug info for active exercises every 30 frames
                frame_count += 1
                if args.debug and frame_count % 30 == 0:
                    debug_info = detector_manager.get_debug_info()
                    for exercise_name, data in debug_info.items():
                        if data.get("is_active", False):
                            logger.info(f"DEBUG - {exercise_name}: {data}")

                # Create debug visualization if requested
                if args.debug:
                    output_frame = create_debug_visualization(
                        output_frame, landmarks, detector_manager.get_debug_info()
                    )

            # Display the frame
            cv2.imshow("FitFighter Exercise Detection", output_frame)

            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        # Release resources
        logger.info("Cleaning up")
        cap.release()
        cv2.destroyAllWindows()
        pose.close()


if __name__ == "__main__":
    main()
