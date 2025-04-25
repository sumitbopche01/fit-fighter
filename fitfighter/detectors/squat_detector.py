"""
Squat Exercise Detector.

This module provides a detector for squat exercises based on pose landmarks.
"""

import numpy as np
from collections import deque
from fitfighter.core.base_detector import BaseExerciseDetector
from fitfighter.constants import landmark_indices as lm
from fitfighter.utils.angle_calculator import calculate_angle


class SquatDetector(BaseExerciseDetector):
    """
    Detector for squat exercises.

    Detects squats by analyzing the angle of the knees and hips,
    tracking the up and down movement of the body.
    """

    def __init__(self, confidence_threshold=0.6, history_size=30):
        """
        Initialize the squat detector.

        Args:
            confidence_threshold (float): Minimum visibility threshold for landmarks
            history_size (int): Number of frames to consider for detection
        """
        super().__init__(confidence_threshold, history_size)
        self.name = "Squat"

        # State tracking
        self.position_state = "up"  # "up", "down", or "transitioning"
        self.position_history = deque(maxlen=5)  # For smoother detection
        self.cooldown_frames = 10
        self.cooldown_counter = 0
        self.last_detection_state = False
        self.debug_values = {}

        # Required landmarks for this detector
        self.required_landmarks = [
            lm.LEFT_HIP,
            lm.RIGHT_HIP,
            lm.LEFT_KNEE,
            lm.RIGHT_KNEE,
            lm.LEFT_ANKLE,
            lm.RIGHT_ANKLE,
        ]

        # Thresholds for squat detection
        self.knee_angle_standing_threshold = 160  # Knee angle when standing (degrees)
        self.knee_angle_squat_threshold = 100  # Knee angle when squatting (degrees)
        self.hip_angle_standing_threshold = 170  # Hip angle when standing (degrees)
        self.hip_angle_squat_threshold = 90  # Hip angle when squatting (degrees)
        self.min_vertical_movement = 0.15  # Minimum vertical movement for a squat

        # Track vertical positions for movement detection
        self.lowest_hip_position = None
        self.highest_hip_position = None

    def detect(self, landmark_history):
        """
        Detect squat exercises from landmark history.

        Args:
            landmark_history (list): List of landmarks for each frame

        Returns:
            bool: True if actively doing squats, False otherwise
        """
        if not landmark_history or len(landmark_history) < 2:
            return False

        # Apply cooldown if active
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return False

        # Get the most recent landmarks
        landmarks = landmark_history[-1]

        # Check if required landmarks are visible
        if not self.are_landmarks_visible(landmarks, self.required_landmarks):
            self._reset_detection_state()
            self.debug_values.update(
                {"state": "invalid", "reason": "Required landmarks not visible"}
            )
            return False

        # Get key positions
        left_hip = landmarks[lm.LEFT_HIP]
        right_hip = landmarks[lm.RIGHT_HIP]
        left_knee = landmarks[lm.LEFT_KNEE]
        right_knee = landmarks[lm.RIGHT_KNEE]
        left_ankle = landmarks[lm.LEFT_ANKLE]
        right_ankle = landmarks[lm.RIGHT_ANKLE]

        # Calculate midpoints for more robust detection
        hip_midpoint = [(left_hip[i] + right_hip[i]) / 2 for i in range(3)]
        knee_midpoint = [(left_knee[i] + right_knee[i]) / 2 for i in range(3)]
        ankle_midpoint = [(left_ankle[i] + right_ankle[i]) / 2 for i in range(3)]

        # Calculate knee angle (angle between hip, knee, and ankle)
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        avg_knee_angle = (left_knee_angle + right_knee_angle) / 2

        # Calculate hip angle (angle between shoulder, hip, and knee)
        if self.are_landmarks_visible(landmarks, [lm.LEFT_SHOULDER, lm.RIGHT_SHOULDER]):
            left_shoulder = landmarks[lm.LEFT_SHOULDER]
            right_shoulder = landmarks[lm.RIGHT_SHOULDER]
            shoulder_midpoint = [
                (left_shoulder[i] + right_shoulder[i]) / 2 for i in range(3)
            ]

            left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
            right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
            avg_hip_angle = (left_hip_angle + right_hip_angle) / 2
        else:
            # If shoulders aren't visible, use a heuristic based on knee angle
            avg_hip_angle = avg_knee_angle * 1.2  # Simple approximation

        # Track vertical position of hip for movement detection
        current_hip_y = hip_midpoint[1]

        # Initialize position tracking if needed
        if self.lowest_hip_position is None or self.highest_hip_position is None:
            self.lowest_hip_position = current_hip_y
            self.highest_hip_position = current_hip_y

        # Update lowest and highest positions
        if current_hip_y > self.lowest_hip_position:
            self.lowest_hip_position = current_hip_y
        if current_hip_y < self.highest_hip_position:
            self.highest_hip_position = current_hip_y

        # Calculate vertical range and current position in the range
        vertical_range = max(
            0.001, self.lowest_hip_position - self.highest_hip_position
        )  # Avoid division by zero
        hip_position_in_range = (
            current_hip_y - self.highest_hip_position
        ) / vertical_range

        # Determine squat state based on knee angle, hip angle, and vertical position
        old_position_state = self.position_state

        # Check if in up position (standing)
        if (
            avg_knee_angle > self.knee_angle_standing_threshold
            and avg_hip_angle > self.hip_angle_standing_threshold
            and hip_position_in_range < 0.3
        ):
            new_position_state = "up"
        # Check if in down position (squatting)
        elif (
            avg_knee_angle < self.knee_angle_squat_threshold
            and avg_hip_angle < self.hip_angle_squat_threshold
            and hip_position_in_range > 0.7
        ):
            new_position_state = "down"
        # Otherwise, transitioning between positions
        else:
            new_position_state = "transitioning"

        # Add current position to history for smoother detection
        self.position_history.append(new_position_state)

        # Use majority voting from history for more stable detection
        if len(self.position_history) >= 3:
            down_count = self.position_history.count("down")
            up_count = self.position_history.count("up")
            trans_count = self.position_history.count("transitioning")

            if down_count > max(up_count, trans_count):
                new_position_state = "down"
            elif up_count > max(down_count, trans_count):
                new_position_state = "up"
            else:
                new_position_state = "transitioning"

        # Count a squat rep when transitioning from down to up
        if old_position_state == "down" and new_position_state == "up":
            vertical_movement = abs(
                self.lowest_hip_position - self.highest_hip_position
            )
            if vertical_movement > self.min_vertical_movement:
                self.rep_count += 1
                self.cooldown_counter = self.cooldown_frames

        self.position_state = new_position_state

        # Update active state
        is_active = new_position_state in ["down", "transitioning"]
        self.is_active = is_active

        # Update detection state for event tracking
        detection_changed = self.is_active != self.last_detection_state
        self.last_detection_state = self.is_active

        # Update debug information
        self.debug_values.update(
            {
                "knee_angle": avg_knee_angle,
                "hip_angle": avg_hip_angle,
                "position_state": self.position_state,
                "vertical_position": hip_position_in_range,
                "squat_count": self.rep_count,
                "is_active": self.is_active,
            }
        )

        return self.is_active

    def _reset_detection_state(self):
        """Reset the squat detection state."""
        self.position_state = "up"
        self.position_history.clear()
        self.lowest_hip_position = None
        self.highest_hip_position = None
        self.is_active = False

    def reset(self):
        """Reset the detector completely, including count."""
        super().reset()
        self._reset_detection_state()
        self.cooldown_counter = 0
        self.rep_count = 0
