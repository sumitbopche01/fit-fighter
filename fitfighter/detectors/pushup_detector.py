"""
Push-Up Exercise Detector.

This module provides a detector for push-up exercises based on pose landmarks.
"""

import numpy as np
import time
from fitfighter.core.base_detector import BaseExerciseDetector
from fitfighter.constants import landmark_indices as lm
from fitfighter.utils.angle_calculator import calculate_angle


class PushUpDetector(BaseExerciseDetector):
    """
    Detector for push-up exercises.

    Detects push-ups by analyzing the vertical movement of the body
    and the angle of the arms during the exercise.
    """

    def __init__(self, confidence_threshold=0.5, history_size=30):
        """
        Initialize the push-up detector.

        Args:
            confidence_threshold (float): Minimum visibility threshold for landmarks
            history_size (int): Number of frames to consider for detection
        """
        super().__init__(confidence_threshold, history_size)
        self.name = "Push-Up"
        self.is_in_pushup_position = False
        self.position_state = "up"  # "up", "down", or "transitioning"
        self.last_update_time = None

        # Required landmarks for this detector
        self.required_landmarks = [
            lm.LEFT_SHOULDER,
            lm.RIGHT_SHOULDER,
            lm.LEFT_ELBOW,
            lm.RIGHT_ELBOW,
            lm.LEFT_WRIST,
            lm.RIGHT_WRIST,
            lm.LEFT_HIP,
            lm.RIGHT_HIP,
        ]

        # Thresholds for push-up detection
        self.elbow_angle_down_threshold = (
            90  # Maximum elbow angle for down position (degrees)
        )
        self.elbow_angle_up_threshold = (
            160  # Minimum elbow angle for up position (degrees)
        )
        self.body_angle_threshold = (
            30.0  # Maximum acceptable angle of torso from horizontal (degrees)
        )
        self.min_vertical_movement = 0.15  # Minimum shoulder movement for a push-up

        # Track vertical positions for movement detection
        self.lowest_position = None
        self.highest_position = None
        self.transitioning_threshold = (
            0.05  # Movement threshold to consider transitioning
        )
        self.debug_values = {}

    def detect(self, landmark_history):
        """
        Detect push-up exercises from landmark history.

        Args:
            landmark_history (list): List of landmarks for each frame

        Returns:
            bool: True if actively doing push-ups, False otherwise
        """
        if not landmark_history or len(landmark_history) < 2:
            return False

        # Get the most recent landmarks
        landmarks = landmark_history[-1]

        # Get current time for tracking
        current_time = time.time()
        if self.last_update_time is None:
            self.last_update_time = current_time

        # Check if required landmarks are visible
        if not self.are_landmarks_visible(landmarks, self.required_landmarks):
            self._reset_detection_state()
            self.debug_values.update(
                {"state": "invalid", "reason": "Required landmarks not visible"}
            )
            return False

        # Get key positions
        left_shoulder = landmarks[lm.LEFT_SHOULDER]
        right_shoulder = landmarks[lm.RIGHT_SHOULDER]
        left_elbow = landmarks[lm.LEFT_ELBOW]
        right_elbow = landmarks[lm.RIGHT_ELBOW]
        left_wrist = landmarks[lm.LEFT_WRIST]
        right_wrist = landmarks[lm.RIGHT_WRIST]
        left_hip = landmarks[lm.LEFT_HIP]
        right_hip = landmarks[lm.RIGHT_HIP]

        # Calculate midpoints for more robust detection
        shoulder_midpoint = [
            (left_shoulder[i] + right_shoulder[i]) / 2 for i in range(3)
        ]
        hip_midpoint = [(left_hip[i] + right_hip[i]) / 2 for i in range(3)]

        # Calculate elbow angles (average of left and right)
        left_elbow_angle = calculate_angle(left_wrist, left_elbow, left_shoulder)
        right_elbow_angle = calculate_angle(right_wrist, right_elbow, right_shoulder)
        avg_elbow_angle = (left_elbow_angle + right_elbow_angle) / 2

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

        # Check if body is in correct push-up position (nearly horizontal)
        is_body_horizontal = abs(body_angle) < self.body_angle_threshold

        # Track vertical position (using y-coordinate of shoulder midpoint)
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

        # Determine push-up state based on elbow angle and vertical position
        old_position_state = self.position_state

        # Check if we're in the down position (arms bent, body lowered)
        if (
            avg_elbow_angle < self.elbow_angle_down_threshold
            and position_in_range > 0.7
        ):
            new_position_state = "down"
        # Check if we're in the up position (arms straight, body raised)
        elif (
            avg_elbow_angle > self.elbow_angle_up_threshold and position_in_range < 0.3
        ):
            new_position_state = "up"
        # Otherwise, we're transitioning between positions
        else:
            new_position_state = "transitioning"

        # Count a push-up rep when transitioning from down to up
        if old_position_state == "down" and new_position_state == "up":
            vertical_movement = abs(self.lowest_position - self.highest_position)
            if vertical_movement > self.min_vertical_movement and is_body_horizontal:
                self.rep_count += 1

        self.position_state = new_position_state

        # Update state and detection info
        self.is_in_pushup_position = is_body_horizontal
        self.is_active = self.is_in_pushup_position

        # Reset tracking if body is not horizontal for too long
        if not is_body_horizontal:
            self._partial_reset()

        # Update debug information
        self.debug_values.update(
            {
                "body_angle": body_angle,
                "elbow_angle": avg_elbow_angle,
                "position_state": self.position_state,
                "pushup_count": self.rep_count,
                "is_horizontal": is_body_horizontal,
                "state": "active" if self.is_active else "inactive",
            }
        )

        # Update last update time
        self.last_update_time = current_time

        return self.is_active

    def _reset_detection_state(self):
        """Reset the push-up detection state completely."""
        self.is_in_pushup_position = False
        self.position_state = "up"
        self.lowest_position = None
        self.highest_position = None
        self.is_active = False

    def _partial_reset(self):
        """Reset only the position tracking, but keep the count."""
        self.lowest_position = None
        self.highest_position = None

    def reset(self):
        """Reset the detector completely, including count."""
        super().reset()
        self._reset_detection_state()
        self.rep_count = 0
