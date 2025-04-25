"""
Angle calculation utilities.

This module provides functions for calculating angles between landmarks.
"""

import numpy as np


def calculate_2d_angle(p1, p2, p3):
    """
    Calculate the angle between three 2D points.

    Args:
        p1: First point (x, y) or (x, y, z, visibility)
        p2: Second point (vertex) (x, y) or (x, y, z, visibility)
        p3: Third point (x, y) or (x, y, z, visibility)

    Returns:
        float: Angle in degrees
    """
    # Extract x, y coordinates
    if len(p1) >= 3:  # Handle (x, y, z) or (x, y, z, visibility) format
        a = np.array([p1[0], p1[1]])
    else:
        a = np.array(p1[:2])

    if len(p2) >= 3:
        b = np.array([p2[0], p2[1]])
    else:
        b = np.array(p2[:2])

    if len(p3) >= 3:
        c = np.array([p3[0], p3[1]])
    else:
        c = np.array(p3[:2])

    # Calculate vectors
    ba = a - b
    bc = c - b

    # Calculate angle
    dot_product = np.dot(ba, bc)
    magnitude_ba = np.linalg.norm(ba)
    magnitude_bc = np.linalg.norm(bc)

    # Avoid division by zero
    if magnitude_ba * magnitude_bc < 1e-10:
        return 0

    cosine_angle = dot_product / (magnitude_ba * magnitude_bc)
    angle_rad = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def calculate_3d_angle(p1, p2, p3):
    """
    Calculate the angle between three 3D points.

    Args:
        p1: First point (x, y, z) or (x, y, z, visibility)
        p2: Second point (vertex) (x, y, z) or (x, y, z, visibility)
        p3: Third point (x, y, z) or (x, y, z, visibility)

    Returns:
        float: Angle in degrees
    """
    # Extract x, y, z coordinates
    if len(p1) > 3:  # Handle (x, y, z, visibility) format
        a = np.array([p1[0], p1[1], p1[2]])
    else:
        a = np.array(p1[:3])

    if len(p2) > 3:
        b = np.array([p2[0], p2[1], p2[2]])
    else:
        b = np.array(p2[:3])

    if len(p3) > 3:
        c = np.array([p3[0], p3[1], p3[2]])
    else:
        c = np.array(p3[:3])

    # Calculate vectors
    ba = a - b
    bc = c - b

    # Calculate angle
    dot_product = np.dot(ba, bc)
    magnitude_ba = np.linalg.norm(ba)
    magnitude_bc = np.linalg.norm(bc)

    # Avoid division by zero
    if magnitude_ba * magnitude_bc < 1e-10:
        return 0

    cosine_angle = dot_product / (magnitude_ba * magnitude_bc)
    angle_rad = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def calculate_body_alignment(
    shoulder_midpoint, hip_midpoint, horizontal_vector=(1, 0, 0)
):
    """
    Calculate the angle between a body segment and a reference vector.

    Args:
        shoulder_midpoint: Coordinates of the shoulder midpoint (x, y, z)
        hip_midpoint: Coordinates of the hip midpoint (x, y, z)
        horizontal_vector: Reference vector, defaults to x-axis (1, 0, 0)

    Returns:
        float: Angle in degrees between the body segment and reference vector
    """
    # Calculate shoulder-to-hip vector
    body_vector = np.array(
        [
            hip_midpoint[0] - shoulder_midpoint[0],
            hip_midpoint[1] - shoulder_midpoint[1],
            hip_midpoint[2] - shoulder_midpoint[2],
        ]
    )

    # Convert reference vector to numpy array
    ref_vector = np.array(horizontal_vector)

    # Calculate magnitudes
    magnitude_body = np.linalg.norm(body_vector)
    magnitude_ref = np.linalg.norm(ref_vector)

    # Avoid division by zero
    if magnitude_body * magnitude_ref < 1e-10:
        return 0

    # Calculate dot product and angle
    dot_product = np.dot(body_vector, ref_vector)
    cosine_angle = dot_product / (magnitude_body * magnitude_ref)
    angle_rad = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)

    return angle_deg
