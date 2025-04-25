"""
Jumping Jacks exercise detector.

This module provides a detector for jumping jacks exercises.
"""

import numpy as np
from collections import deque
from fitfighter.core.base_detector import BaseExerciseDetector
from fitfighter.constants import landmark_indices as lm
from fitfighter.utils.angle_calculator import calculate_2d_angle
from fitfighter.utils.pose_processor import calculate_distance


class JumpingJackDetector(BaseExerciseDetector):
    """Detector for jumping jacks exercise."""

    def __init__(self, confidence_threshold=0.6, history_size=30):
        """
        Initialize the jumping jacks detector.

        Args:
            confidence_threshold: Minimum landmark visibility to consider valid
            history_size: Number of frames to keep in history
        """
        super().__init__(confidence_threshold, history_size)

        # Thresholds for detection
        self.arm_angle_threshold = 60  # Minimum angle for arms to be considered open
        self.leg_width_threshold = 0.15  # Minimum normalized width between ankles
        self.cooldown_frames = 10  # Frames to wait before detecting another rep

        # State tracking
        self.cooldown_counter = 0
        self.in_open_position = False
        self.in_closed_position = False
        self.position_history = deque(
            maxlen=5
        )  # Track recent positions for smoother detection

        # For detecting the current phase of the movement
        self.current_phase = "unknown"  # "closed", "opening", "open", "closing"
        self.debug_values = {}

    def detect(self, landmark_history):
        """
        Detect if a jumping jack is being performed.

        Args:
            landmark_history: History of pose landmarks

        Returns:
            bool: True if jumping jack is detected, False otherwise
        """
        if len(landmark_history) < 2:
            return False

        # Apply cooldown if active
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return False

        # Get current frame landmarks
        current = landmark_history[-1]

        # Required landmarks for jumping jacks
        required_landmarks = [
            lm.LEFT_SHOULDER,
            lm.RIGHT_SHOULDER,  # Shoulders
            lm.LEFT_ELBOW,
            lm.RIGHT_ELBOW,  # Elbows
            lm.LEFT_WRIST,
            lm.RIGHT_WRIST,  # Wrists
            lm.LEFT_HIP,
            lm.RIGHT_HIP,  # Hips
            lm.LEFT_KNEE,
            lm.RIGHT_KNEE,  # Knees
            lm.LEFT_ANKLE,
            lm.RIGHT_ANKLE,  # Ankles
        ]

        # Check if we have all required landmarks
        if not self.are_landmarks_visible(current, required_landmarks):
            self.current_phase = "unknown"
            return False

        # Get shoulder landmarks
        left_shoulder = current[lm.LEFT_SHOULDER]
        right_shoulder = current[lm.RIGHT_SHOULDER]
        left_wrist = current[lm.LEFT_WRIST]
        right_wrist = current[lm.RIGHT_WRIST]
        left_ankle = current[lm.LEFT_ANKLE]
        right_ankle = current[lm.RIGHT_ANKLE]
        left_hip = current[lm.LEFT_HIP]
        right_hip = current[lm.RIGHT_HIP]

        # Calculate arm angle (angle between shoulders and wrists)
        left_arm_angle = calculate_2d_angle(right_shoulder, left_shoulder, left_wrist)
        right_arm_angle = calculate_2d_angle(left_shoulder, right_shoulder, right_wrist)

        # Calculate normalized leg width (distance between ankles relative to shoulder width)
        shoulder_width = calculate_distance(left_shoulder, right_shoulder)
        ankle_width = calculate_distance(left_ankle, right_ankle)
        normalized_leg_width = ankle_width / shoulder_width if shoulder_width > 0 else 0

        # Calculate height (to detect jumping)
        hip_height = (left_hip[1] + right_hip[1]) / 2

        # Store metrics for debugging
        self.debug_values.update(
            {
                "left_arm_angle": left_arm_angle,
                "right_arm_angle": right_arm_angle,
                "normalized_leg_width": normalized_leg_width,
                "hip_height": hip_height,
                "phase": self.current_phase,
            }
        )

        # Check if in open position (arms up, legs apart)
        is_open = (
            left_arm_angle > self.arm_angle_threshold
            and right_arm_angle > self.arm_angle_threshold
            and normalized_leg_width > self.leg_width_threshold
        )

        # Check if in closed position (arms down, legs together)
        is_closed = (
            left_arm_angle < self.arm_angle_threshold / 2
            and right_arm_angle < self.arm_angle_threshold / 2
            and normalized_leg_width < self.leg_width_threshold / 2
        )

        # Add current position to history
        self.position_history.append((is_open, is_closed))

        # Determine current phase based on position history
        if len(self.position_history) >= 3:
            # Count recent positions
            open_count = sum(pos[0] for pos in self.position_history)
            closed_count = sum(pos[1] for pos in self.position_history)

            # Stable positions
            if open_count >= len(self.position_history) - 1:
                new_phase = "open"
            elif closed_count >= len(self.position_history) - 1:
                new_phase = "closed"
            # Transitions
            elif self.current_phase == "closed" or self.current_phase == "closing":
                new_phase = "opening"
            elif self.current_phase == "open" or self.current_phase == "opening":
                new_phase = "closing"
            else:
                new_phase = "unknown"

            # Detect a complete rep (closed -> open -> closed)
            if self.current_phase == "open" and new_phase == "closing":
                # Going from open to closing, we've completed half a rep
                self.is_active = True
            elif self.current_phase == "closing" and new_phase == "closed":
                # Completed a full rep
                if self.is_active:
                    self.rep_count += 1
                    self.cooldown_counter = self.cooldown_frames
                    self.is_active = False

            # Update phase
            self.current_phase = new_phase

        # Return true if we're actively doing a jumping jack
        is_detected = self.is_active

        # Update detection state for event tracking
        detection_changed = is_detected != self.last_detection_state
        self.last_detection_state = is_detected

        return is_detected

    def reset(self):
        """Reset the detector state."""
        super().reset()
        self.cooldown_counter = 0
        self.in_open_position = False
        self.in_closed_position = False
        self.position_history.clear()
        self.current_phase = "unknown"
