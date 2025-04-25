"""
Plank Exercise Detector.

This module provides a detector for plank exercises based on pose landmarks.
"""

import numpy as np
import time
from fitfighter.core.base_detector import BaseExerciseDetector
from fitfighter.constants import landmark_indices as lm
from fitfighter.utils.pose_processor import calculate_distance


class PlankDetector(BaseExerciseDetector):
    """
    Detector for plank exercises.

    Detects planks by analyzing the alignment of the body and stability
    over time during the plank position.
    """

    def __init__(self, confidence_threshold=0.5, history_size=30):
        """
        Initialize the plank detector.

        Args:
            confidence_threshold (float): Minimum visibility threshold for landmarks
            history_size (int): Number of frames to consider for detection
        """
        super().__init__(confidence_threshold, history_size)
        self.name = "Plank"
        self.is_in_plank_position = False
        self.plank_start_time = None
        self.plank_duration = 0
        self.last_update_time = None
        self.debug_values = {}

        # Required landmarks for this detector
        self.required_landmarks = [
            lm.LEFT_SHOULDER,
            lm.RIGHT_SHOULDER,
            lm.LEFT_ELBOW,
            lm.RIGHT_ELBOW,
            lm.LEFT_HIP,
            lm.RIGHT_HIP,
            lm.LEFT_ANKLE,
            lm.RIGHT_ANKLE,
        ]

        # Thresholds for plank detection
        self.hip_elevation_threshold = (
            0.15  # Maximum acceptable hip elevation relative to body length
        )
        self.body_angle_threshold = (
            15.0  # Maximum acceptable angle of torso from horizontal (degrees)
        )
        self.stability_threshold = 0.03  # Maximum allowed movement between frames
        self.min_plank_time = 1.0  # Minimum time (seconds) to register as a plank

        # For tracking stability
        self.previous_positions = None

    def detect(self, landmark_history):
        """
        Detect plank exercises from landmark history.

        Args:
            landmark_history (list): List of landmarks for each frame

        Returns:
            bool: True if in plank position, False otherwise
        """
        if not landmark_history or len(landmark_history) < 2:
            return False

        # Get the most recent landmarks
        landmarks = landmark_history[-1]

        # Get current time for duration tracking
        current_time = time.time()
        if self.last_update_time is None:
            self.last_update_time = current_time

        # Check if required landmarks are visible
        if not self.are_landmarks_visible(landmarks, self.required_landmarks):
            self._reset_plank_state()
            self.debug_values.update(
                {"state": "invalid", "reason": "Required landmarks not visible"}
            )
            return False

        # Get positions
        left_shoulder = landmarks[lm.LEFT_SHOULDER]
        right_shoulder = landmarks[lm.RIGHT_SHOULDER]
        left_hip = landmarks[lm.LEFT_HIP]
        right_hip = landmarks[lm.RIGHT_HIP]
        left_ankle = landmarks[lm.LEFT_ANKLE]
        right_ankle = landmarks[lm.RIGHT_ANKLE]

        # Calculate midpoints for more robust detection
        shoulder_midpoint = [
            (left_shoulder[i] + right_shoulder[i]) / 2 for i in range(3)
        ]
        hip_midpoint = [(left_hip[i] + right_hip[i]) / 2 for i in range(3)]
        ankle_midpoint = [(left_ankle[i] + right_ankle[i]) / 2 for i in range(3)]

        # Calculate body length for normalization
        body_length = calculate_distance(shoulder_midpoint, ankle_midpoint)

        # Check if the body is horizontal (shoulders to hips nearly horizontal)
        shoulder_to_hip_vector = [
            hip_midpoint[i] - shoulder_midpoint[i] for i in range(3)
        ]
        horizontal_vector = [1, 0, 0]  # X-axis reference vector

        # Calculate angle between shoulder-hip line and horizontal
        dot_product = sum(
            shoulder_to_hip_vector[i] * horizontal_vector[i] for i in range(3)
        )
        magnitude_sh = np.sqrt(sum(shoulder_to_hip_vector[i] ** 2 for i in range(3)))
        magnitude_h = 1  # Unit vector

        body_angle = np.degrees(
            np.arccos(max(-1, min(1, dot_product / (magnitude_sh * magnitude_h))))
        )
        # Adjust angle to measure deviation from horizontal
        if shoulder_midpoint[1] > hip_midpoint[1]:  # Y increases downward
            body_angle = 180 - body_angle

        # Check if hips are aligned properly (not sagging or raised too high)
        hip_elevation = abs(hip_midpoint[1] - shoulder_midpoint[1]) / max(
            0.1, body_length
        )

        # Check stability by comparing with previous frame
        is_stable = True
        if self.previous_positions is not None:
            prev_shoulder = self.previous_positions["shoulder"]
            prev_hip = self.previous_positions["hip"]

            shoulder_movement = calculate_distance(shoulder_midpoint, prev_shoulder)
            hip_movement = calculate_distance(hip_midpoint, prev_hip)

            # Normalize movement by body length
            normalized_movement = (shoulder_movement + hip_movement) / (
                2 * max(0.1, body_length)
            )
            is_stable = normalized_movement < self.stability_threshold

        # Update previous positions
        self.previous_positions = {"shoulder": shoulder_midpoint, "hip": hip_midpoint}

        # Update debug information
        self.debug_values.update(
            {
                "body_angle": body_angle,
                "hip_elevation": hip_elevation,
                "is_stable": is_stable,
                "duration": self.plank_duration,
                "state": "analyzing",
            }
        )

        # Determine if in plank position
        is_plank = (
            abs(body_angle) < self.body_angle_threshold
            and hip_elevation < self.hip_elevation_threshold
            and is_stable
        )

        # State tracking for plank
        if is_plank:
            if not self.is_in_plank_position:
                # Just entered plank position
                self.is_in_plank_position = True
                self.plank_start_time = current_time
                self.debug_values.update({"state": "plank_started"})
            else:
                # Continue tracking duration
                self.plank_duration = current_time - self.plank_start_time
                self.debug_values.update(
                    {"state": "plank_in_progress", "duration": self.plank_duration}
                )
        else:
            if self.is_in_plank_position:
                # Just exited plank position
                self.plank_duration = current_time - self.plank_start_time
                # Only count as completed if held for minimum time
                plank_completed = self.plank_duration >= self.min_plank_time

                if plank_completed:
                    self.rep_count += 1
                    self.debug_values.update(
                        {"state": "plank_completed", "duration": self.plank_duration}
                    )
                    # We don't return True here because the plank has ended
                else:
                    self.debug_values.update(
                        {"state": "plank_too_short", "duration": self.plank_duration}
                    )

                self._reset_plank_state()
            else:
                # Not in plank and wasn't in plank
                self._reset_plank_state()
                self.debug_values.update({"state": "not_in_plank"})

        # Update last update time
        self.last_update_time = current_time

        # Update active state
        self.is_active = self.is_in_plank_position

        # For plank, we return True while in the plank position
        return self.is_in_plank_position

    def _reset_plank_state(self):
        """Reset the plank detection state."""
        self.is_in_plank_position = False
        self.plank_start_time = None

    def reset(self):
        """Reset the detector completely, including count and duration."""
        super().reset()
        self._reset_plank_state()
        self.plank_duration = 0
        self.previous_positions = None
