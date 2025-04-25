"""
Utility functions for the FitFighter system.

This module provides utility functions for angle calculation,
pose processing, visualization, camera management, and data sending.
"""

from .angle_calculator import (
    calculate_2d_angle,
    calculate_3d_angle,
    calculate_body_alignment,
)
from .pose_processor import (
    calculate_midpoint,
    calculate_distance,
    are_landmarks_visible,
    calculate_body_reference_height,
    convert_mediapipe_landmarks,
)
from .visualization import (
    draw_pose_landmarks,
    draw_detection_results,
    create_debug_visualization,
)
from .camera_utils import (
    CameraManager,
    FPSCounter,
    display_frame,
    preprocess_frame,
)
from .data_sender import DataSender
