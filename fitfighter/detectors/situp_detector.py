"""
Sit-Up Exercise Detector.

This module provides a detector for sit-up exercises based on pose landmarks.
"""

import numpy as np
from collections import deque
from fitfighter.core.base_detector import BaseExerciseDetector


class SitupDetector(BaseExerciseDetector):
    """
    Detector for sit-up exercises.

    Detects sit-ups by analyzing the angle between the shoulders, hip, and knees
    as well as the vertical movement of the upper body.
    """

    def __init__(self, confidence_threshold=0.6, history_size=30):
        """
        Initialize the sit-up detector.

        Args:
            confidence_threshold (float): Minimum visibility threshold for landmarks
            history_size (int): Number of frames to consider for detection
        """
        super().__init__(confidence_threshold, history_size)
        self.name = "Sit-Up"

        # State tracking
        self.is_in_situp_position = False
        self.position_state = "down"  # "down", "up", or "transitioning"
        self.position_history = deque(maxlen=5)  # For smoother detection
        self.cooldown_frames = 10
        self.cooldown_counter = 0

        # Landmarks for MediaPipe model
        self.nose = 0
        self.left_shoulder = 11
        self.right_shoulder = 12
        self.left_elbow = 13
        self.right_elbow = 14
        self.left_hip = 23
        self.right_hip = 24
        self.left_knee = 25
        self.right_knee = 26
        self.left_ankle = 27
        self.right_ankle = 28

        # Required landmarks for this detector
        self.required_landmarks = [
            self.left_shoulder,
            self.right_shoulder,
            self.left_hip,
            self.right_hip,
            self.left_knee,
            self.right_knee,
        ]

        # Thresholds for sit-up detection
        self.hip_angle_down_threshold = 120  # Hip angle when lying down (degrees)
        self.hip_angle_up_threshold = 70  # Hip angle when sitting up (degrees)
        self.min_vertical_movement = 0.15  # Minimum movement for a sit-up

        # Track vertical positions for movement detection
        self.lowest_position = None
        self.highest_position = None

    def detect(self, landmark_history):
        """
        Detect sit-up exercises from landmark history.

        Args:
            landmark_history (list): List of landmarks for each frame

        Returns:
            bool: True if actively doing sit-ups, False otherwise
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
        left_shoulder = landmarks[self.left_shoulder]
        right_shoulder = landmarks[self.right_shoulder]
        left_hip = landmarks[self.left_hip]
        right_hip = landmarks[self.right_hip]
        left_knee = landmarks[self.left_knee]
        right_knee = landmarks[self.right_knee]

        # Calculate midpoints for more robust detection
        shoulder_midpoint = [
            (left_shoulder[i] + right_shoulder[i]) / 2 for i in range(3)
        ]
        hip_midpoint = [(left_hip[i] + right_hip[i]) / 2 for i in range(3)]
        knee_midpoint = [(left_knee[i] + right_knee[i]) / 2 for i in range(3)]

        # Calculate hip angle (angle between shoulders, hips, and knees)
        hip_angle = self.calculate_angle(shoulder_midpoint, hip_midpoint, knee_midpoint)

        # Track vertical position of upper body (using y-coordinate of shoulder midpoint)
        current_y_position = shoulder_midpoint[1]

        # Initialize position tracking if needed
        if self.lowest_position is None or self.highest_position is None:
            self.lowest_position = current_y_position
            self.highest_position = current_y_position

        # Update lowest and highest positions
        if current_y_position > self.lowest_position:
            self.lowest_position = current_y_position
        if current_y_position < self.highest_position:
            self.highest_position = current_y_position

        # Calculate vertical range and current position in the range
        vertical_range = max(
            0.001, self.lowest_position - self.highest_position
        )  # Avoid division by zero
        position_in_range = (
            current_y_position - self.highest_position
        ) / vertical_range

        # Determine sit-up state based on hip angle and vertical position
        old_position_state = self.position_state

        # Check if in down position (lying down)
        if hip_angle > self.hip_angle_down_threshold and position_in_range > 0.7:
            new_position_state = "down"
        # Check if in up position (sitting up)
        elif hip_angle < self.hip_angle_up_threshold and position_in_range < 0.3:
            new_position_state = "up"
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

        # Count a sit-up rep when transitioning from down to up
        if old_position_state == "down" and new_position_state == "up":
            vertical_movement = abs(self.lowest_position - self.highest_position)
            if vertical_movement > self.min_vertical_movement:
                self.rep_count += 1
                self.cooldown_counter = self.cooldown_frames

        self.position_state = new_position_state

        # Update active state
        self.is_in_situp_position = new_position_state in [
            "down",
            "transitioning",
            "up",
        ]
        self.is_active = self.is_in_situp_position

        # Update detection state for event tracking
        detection_changed = self.is_active != self.last_detection_state
        self.last_detection_state = self.is_active

        # Update debug information
        self.debug_values.update(
            {
                "hip_angle": hip_angle,
                "position_state": self.position_state,
                "vertical_position": position_in_range,
                "sit_up_count": self.rep_count,
                "is_active": self.is_active,
            }
        )

        return self.is_active

    def _reset_detection_state(self):
        """Reset the sit-up detection state."""
        self.is_in_situp_position = False
        self.position_state = "down"
        self.position_history.clear()
        self.lowest_position = None
        self.highest_position = None
        self.is_active = False

    def reset(self):
        """Reset the detector completely, including count."""
        super().reset()
        self._reset_detection_state()
        self.cooldown_counter = 0
