"""
Pose processing utilities.

This module provides functions for processing pose landmarks.
"""

import numpy as np
from fitfighter.constants import landmark_indices as lm


def calculate_midpoint(landmarks, idx1, idx2):
    """
    Calculate the midpoint between two landmarks.

    Args:
        landmarks: Dictionary of landmarks
        idx1: Index of first landmark
        idx2: Index of second landmark

    Returns:
        tuple: (x, y, z) coordinates of the midpoint
    """
    if idx1 not in landmarks or idx2 not in landmarks:
        return None

    lm1 = landmarks[idx1]
    lm2 = landmarks[idx2]

    # Check if landmarks have sufficient visibility
    if len(lm1) > 3 and len(lm2) > 3 and (lm1[3] < 0.5 or lm2[3] < 0.5):
        return None

    # Calculate midpoint
    midpoint = ((lm1[0] + lm2[0]) / 2, (lm1[1] + lm2[1]) / 2, (lm1[2] + lm2[2]) / 2)

    return midpoint


def calculate_distance(point1, point2):
    """
    Calculate Euclidean distance between two points.

    Args:
        point1: First point (x, y, z) or (x, y, z, visibility)
        point2: Second point (x, y, z) or (x, y, z, visibility)

    Returns:
        float: Euclidean distance
    """
    if point1 is None or point2 is None:
        return float("inf")

    # Extract coordinates
    if len(point1) > 3:
        p1 = point1[:3]
    else:
        p1 = point1

    if len(point2) > 3:
        p2 = point2[:3]
    else:
        p2 = point2

    # Calculate distance
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)


def are_landmarks_visible(landmarks, indices, threshold=0.5):
    """
    Check if specified landmarks are visible with sufficient confidence.

    Args:
        landmarks: Dictionary of landmarks
        indices: List of landmark indices to check
        threshold: Minimum visibility threshold

    Returns:
        bool: True if all landmarks are visible, False otherwise
    """
    if not landmarks:
        return False

    for idx in indices:
        if idx not in landmarks:
            return False

        lm = landmarks[idx]
        if len(lm) > 3 and lm[3] < threshold:
            return False

    return True


def calculate_body_reference_height(landmarks):
    """
    Calculate a reference height for the person's body.

    Args:
        landmarks: Dictionary of landmarks

    Returns:
        float: Reference height value or None if can't be determined
    """
    # Try to use nose to ankle distance as height reference
    if are_landmarks_visible(landmarks, [lm.NOSE, lm.LEFT_ANKLE, lm.RIGHT_ANKLE]):
        nose = landmarks[lm.NOSE]
        left_ankle = landmarks[lm.LEFT_ANKLE]
        right_ankle = landmarks[lm.RIGHT_ANKLE]

        ankle_midpoint = calculate_midpoint(landmarks, lm.LEFT_ANKLE, lm.RIGHT_ANKLE)
        if ankle_midpoint:
            return abs(nose[1] - ankle_midpoint[1])

    # Fall back to shoulder-to-hip distance if full body not visible
    if are_landmarks_visible(
        landmarks, [lm.LEFT_SHOULDER, lm.RIGHT_SHOULDER, lm.LEFT_HIP, lm.RIGHT_HIP]
    ):
        shoulder_midpoint = calculate_midpoint(
            landmarks, lm.LEFT_SHOULDER, lm.RIGHT_SHOULDER
        )
        hip_midpoint = calculate_midpoint(landmarks, lm.LEFT_HIP, lm.RIGHT_HIP)

        if shoulder_midpoint and hip_midpoint:
            # Torso height is typically around 0.4 of total height
            torso_height = abs(shoulder_midpoint[1] - hip_midpoint[1])
            return torso_height / 0.4

    return None


def estimate_depth(landmarks):
    """
    Estimate relative depth values for landmarks.

    In 2D images, this uses heuristics based on relative positions
    to estimate Z-coordinates.

    Args:
        landmarks: Dictionary of landmarks

    Returns:
        dict: Dictionary of landmarks with estimated depth values
    """
    # If we already have depth values (from 3D model), just return
    if landmarks and len(next(iter(landmarks.values()))) > 2:
        return landmarks

    # Create a copy to avoid modifying the original
    result = {}

    # Use shoulders width to normalize depth estimates
    shoulder_width = None
    if lm.LEFT_SHOULDER in landmarks and lm.RIGHT_SHOULDER in landmarks:
        left_shoulder = landmarks[lm.LEFT_SHOULDER]
        right_shoulder = landmarks[lm.RIGHT_SHOULDER]
        shoulder_width = abs(left_shoulder[0] - right_shoulder[0])

    # Default depth if we can't calculate
    default_depth = 0.0

    # Process each landmark
    for idx, coords in landmarks.items():
        # Add estimated depth (z-coordinate)
        if len(coords) <= 2:
            # For 2D landmarks, add a default depth
            result[idx] = (*coords, default_depth)
        else:
            # Copy existing landmarks
            result[idx] = coords

    return result


def convert_mediapipe_landmarks(mediapipe_landmarks):
    """
    Convert MediaPipe landmarks format to our internal format.

    Args:
        mediapipe_landmarks: MediaPipe pose landmarks result

    Returns:
        dict: Dictionary mapping landmark indices to (x, y, z, visibility) tuples
    """
    if not mediapipe_landmarks or not mediapipe_landmarks.landmark:
        return {}

    landmarks = {}
    for idx, landmark in enumerate(mediapipe_landmarks.landmark):
        landmarks[idx] = (landmark.x, landmark.y, landmark.z, landmark.visibility)

    return landmarks
