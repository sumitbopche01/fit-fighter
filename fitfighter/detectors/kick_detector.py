"""
Kick detector module.

This module provides a detector for kicking motions using pose landmarks.
"""

import numpy as np
from collections import deque
from fitfighter.core.base_detector import BaseExerciseDetector


class KickDetector(BaseExerciseDetector):
    """Detector for kicking motions."""

    def __init__(self, confidence_threshold=0.6, history_size=30):
        """
        Initialize the kick detector.

        Args:
            confidence_threshold (float): Minimum visibility threshold for landmarks
            history_size (int): Number of frames to consider for detection
        """
        super().__init__(confidence_threshold, history_size)

        # Define landmark indices
        self.left_hip = 23
        self.right_hip = 24
        self.left_knee = 25
        self.right_knee = 26
        self.left_ankle = 27
        self.right_ankle = 28
        self.left_foot_index = 31
        self.right_foot_index = 32

        # Thresholds for detection
        self.velocity_threshold = 0.05  # Minimum velocity to consider a kick
        self.extension_threshold = 140  # Minimum angle for leg extension
        self.cooldown_frames = 15  # Frames to wait before detecting another kick

        # State tracking
        self.cooldown_counter = 0
        self.consecutive_frames_required = 2  # Frames needed for detection
        self.consecutive_detection_counter = 0
        self.last_detection_state = False

        # For tracking leg position and movement
        self.ankle_position_history = deque(maxlen=10)
        self.state = "neutral"  # neutral, extending, retracted

        # For debug information
        self.debug_info = {
            "left_leg_vel": 0,
            "right_leg_vel": 0,
            "left_extension": 0,
            "right_extension": 0,
        }

        # Required landmarks for kick detection
        self.required_landmarks = [
            self.left_hip,
            self.right_hip,
            self.left_knee,
            self.right_knee,
            self.left_ankle,
            self.right_ankle,
        ]

    def detect(self, landmark_history):
        """
        Detect if a kick is being performed.

        Args:
            landmark_history (list): History of pose landmarks

        Returns:
            bool: True if kick detected, False otherwise
        """
        # Apply cooldown if active
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return False

        if not landmark_history or len(landmark_history) < 3:
            return False

        # Get landmarks from last three frames
        current = landmark_history[-1]
        previous = landmark_history[-2]
        prev_prev = landmark_history[-3]

        # Check if required landmarks are visible
        if not self.check_landmarks_visibility(current, self.required_landmarks):
            self.update_debug_info(
                state=self.state, reason="Required landmarks not visible"
            )
            return False

        # Initialize detection flags
        left_kick = False
        right_kick = False

        # Check left leg kick
        left_extension = self._calculate_leg_extension(
            current, self.left_hip, self.left_knee, self.left_ankle
        )

        left_velocity = self._calculate_leg_velocity(
            current, previous, self.left_ankle, self.left_knee
        )

        # Check right leg kick
        right_extension = self._calculate_leg_extension(
            current, self.right_hip, self.right_knee, self.right_ankle
        )

        right_velocity = self._calculate_leg_velocity(
            current, previous, self.right_ankle, self.right_knee
        )

        # Store debug info
        self.debug_info["left_leg_vel"] = left_velocity
        self.debug_info["right_leg_vel"] = right_velocity
        self.debug_info["left_extension"] = left_extension
        self.debug_info["right_extension"] = right_extension

        # Detect kick condition: high velocity and extended leg
        left_kick = (
            left_velocity > self.velocity_threshold
            and left_extension > self.extension_threshold
        )

        right_kick = (
            right_velocity > self.velocity_threshold
            and right_extension > self.extension_threshold
        )

        # Combined detection result
        current_detection = left_kick or right_kick

        # Apply consecutive frame requirement for more robust detection
        if current_detection:
            self.consecutive_detection_counter += 1
        else:
            self.consecutive_detection_counter = 0

        is_kick = self.consecutive_detection_counter >= self.consecutive_frames_required

        # Set cooldown when a kick is detected
        if is_kick and not self.last_detection_state:
            self.cooldown_counter = self.cooldown_frames
            # Increment rep count
            self.rep_count += 1

        self.last_detection_state = is_kick
        self.is_active = is_kick

        return is_kick

    def _calculate_leg_extension(self, landmarks, hip_idx, knee_idx, ankle_idx):
        """
        Calculate the leg extension angle (hip-knee-ankle).

        Args:
            landmarks (dict): Current frame landmarks
            hip_idx (int): Hip landmark index
            knee_idx (int): Knee landmark index
            ankle_idx (int): Ankle landmark index

        Returns:
            float: Leg extension angle in degrees
        """
        hip = self.get_landmark_position(landmarks, hip_idx)
        knee = self.get_landmark_position(landmarks, knee_idx)
        ankle = self.get_landmark_position(landmarks, ankle_idx)

        if hip is None or knee is None or ankle is None:
            return 0

        return self.calculate_3d_angle(hip, knee, ankle)

    def _calculate_leg_velocity(self, current, previous, ankle_idx, knee_idx):
        """
        Calculate the velocity of the leg.

        Args:
            current (dict): Current frame landmarks
            previous (dict): Previous frame landmarks
            ankle_idx (int): Ankle landmark index
            knee_idx (int): Knee landmark index

        Returns:
            float: Leg movement velocity
        """
        current_ankle = self.get_landmark_position(current, ankle_idx)
        previous_ankle = self.get_landmark_position(previous, ankle_idx)

        # Reference point (knee) for normalization
        current_knee = self.get_landmark_position(current, knee_idx)
        previous_knee = self.get_landmark_position(previous, knee_idx)

        if (
            current_ankle is None
            or previous_ankle is None
            or current_knee is None
            or previous_knee is None
        ):
            return 0

        # Calculate ankle displacement relative to knee
        rel_current_ankle = (
            current_ankle[0] - current_knee[0],
            current_ankle[1] - current_knee[1],
            current_ankle[2] - current_knee[2],
        )

        rel_previous_ankle = (
            previous_ankle[0] - previous_knee[0],
            previous_ankle[1] - previous_knee[1],
            previous_ankle[2] - previous_knee[2],
        )

        # Calculate velocity as change in position
        dx = rel_current_ankle[0] - rel_previous_ankle[0]
        dy = rel_current_ankle[1] - rel_previous_ankle[1]
        dz = rel_current_ankle[2] - rel_previous_ankle[2]

        # Prioritize forward motion (z-axis) and vertical motion (y-axis)
        # for kick detection
        velocity = np.sqrt(dx**2 + dy**2 + 1.5 * dz**2)  # Weight z more

        return velocity
