"""
Visualization utilities.

This module provides functions for visualizing exercise detection results.
"""

import cv2
import numpy as np
import mediapipe as mp


def draw_pose_landmarks(frame, landmarks, connections=None):
    """
    Draw pose landmarks and connections on an image.

    Args:
        frame: Image/frame to draw on (numpy array)
        landmarks: Dictionary or list of pose landmarks
        connections: Optional list of landmark connections to draw

    Returns:
        numpy array: Image with landmarks drawn
    """
    # Create a copy of the frame to avoid modifying the original
    output_frame = frame.copy()

    # Initialize Mediapipe drawing utils if connections are provided
    if connections is None:
        mp_pose = mp.solutions.pose
        connections = mp_pose.POSE_CONNECTIONS

    # Draw landmarks
    height, width, _ = output_frame.shape
    landmark_color = (0, 255, 0)  # Green
    connection_color = (255, 0, 0)  # Blue

    # Draw each landmark
    for idx, landmark in landmarks.items():
        # Convert normalized coordinates to pixel coordinates
        x = int(landmark[0] * width)
        y = int(landmark[1] * height)

        # Draw the landmark point
        cv2.circle(output_frame, (x, y), 5, landmark_color, -1)

    # Draw connections between landmarks
    for connection in connections:
        start_idx, end_idx = connection

        if start_idx in landmarks and end_idx in landmarks:
            start_point = landmarks[start_idx]
            end_point = landmarks[end_idx]

            start_x = int(start_point[0] * width)
            start_y = int(start_point[1] * height)
            end_x = int(end_point[0] * width)
            end_y = int(end_point[1] * height)

            cv2.line(
                output_frame, (start_x, start_y), (end_x, end_y), connection_color, 2
            )

    return output_frame


def draw_detection_results(
    frame, results, position=(10, 30), font_scale=0.7, font_thickness=2
):
    """
    Draw exercise detection results on an image.

    Args:
        frame: Image/frame to draw on (numpy array)
        results: Detection results dictionary
        position: Starting position for text (x, y)
        font_scale: Font scale for text
        font_thickness: Thickness of font

    Returns:
        numpy array: Image with detection results drawn
    """
    # Create a copy of the frame to avoid modifying the original
    output_frame = frame.copy()

    # Set font
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Draw active exercises
    active_exercises = results.get("active_exercises", [])
    active_text = (
        f"Active: {', '.join(active_exercises) if active_exercises else 'None'}"
    )
    cv2.putText(
        output_frame,
        active_text,
        position,
        font,
        font_scale,
        (0, 255, 0),  # Green
        font_thickness,
    )

    # Draw rep counts for each exercise
    counts = results.get("counts", {})
    y_offset = position[1] + 30

    for exercise, count in counts.items():
        if count > 0:
            count_text = f"{exercise.capitalize()}: {count} reps"
            cv2.putText(
                output_frame,
                count_text,
                (position[0], y_offset),
                font,
                font_scale,
                (0, 255, 0),  # Green
                font_thickness,
            )
            y_offset += 30

    return output_frame


def create_debug_visualization(frame, landmarks, debug_info):
    """
    Create a detailed debug visualization showing landmark tracking
    and exercise detection metrics.

    Args:
        frame: Image/frame to draw on (numpy array)
        landmarks: Dictionary of pose landmarks
        debug_info: Debug information from detectors

    Returns:
        numpy array: Debug visualization image
    """
    # Create a copy of the frame
    debug_frame = frame.copy()

    # Draw the pose landmarks
    debug_frame = draw_pose_landmarks(debug_frame, landmarks)

    # Add debug text
    font = cv2.FONT_HERSHEY_SIMPLEX
    y_pos = 30
    line_height = 20

    # Add a semi-transparent overlay for better text readability
    overlay = debug_frame.copy()
    cv2.rectangle(overlay, (0, 0), (400, 400), (0, 0, 0), -1)
    alpha = 0.4  # Transparency factor
    debug_frame = cv2.addWeighted(overlay, alpha, debug_frame, 1 - alpha, 0)

    # Add debug info for each active detector
    for detector_name, info in debug_info.items():
        if info.get("is_active", False):
            # Add detector name as header
            cv2.putText(
                debug_frame,
                f"{detector_name.upper()}:",
                (10, y_pos),
                font,
                0.6,
                (255, 255, 0),  # Yellow
                2,
            )
            y_pos += line_height

            # Add relevant metrics
            for key, value in info.items():
                if key not in ["name", "is_active"]:
                    metric_text = f"  {key}: {value}"
                    cv2.putText(
                        debug_frame,
                        metric_text,
                        (10, y_pos),
                        font,
                        0.5,
                        (255, 255, 255),  # White
                        1,
                    )
                    y_pos += line_height

            # Add spacing between detector sections
            y_pos += 5

    return debug_frame
