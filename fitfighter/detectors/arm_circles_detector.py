"""
Arm Circles exercise detector.

This module provides a detector for arm circles exercises.
"""

import numpy as np
import math
from collections import deque
from fitfighter.core.base_detector import BaseExerciseDetector


class ArmCirclesDetector(BaseExerciseDetector):
    """Detector for arm circles exercise."""

    def __init__(self, confidence_threshold=0.6, history_size=30):
        """
        Initialize the arm circles detector.

        Args:
            confidence_threshold: Minimum landmark visibility to consider valid
            history_size: Number of frames to keep in history
        """
        super().__init__(confidence_threshold, history_size)

        # Thresholds for detection
        self.min_wrist_movement = 0.05  # Minimum wrist movement to consider valid
        self.circularity_threshold = (
            0.7  # Minimum circularity score (0-1) to consider a circular motion
        )
        self.cooldown_frames = 15  # Frames to wait before detecting another rep

        # State tracking
        self.cooldown_counter = 0

        # Tracking wrist positions for detecting circles
        self.left_wrist_positions = deque(maxlen=20)
        self.right_wrist_positions = deque(maxlen=20)

        # Track which arm is moving in circles
        self.active_arm = None  # "left", "right", "both", or None

        # Store completion state of circle
        self.circle_completion = {"left": 0.0, "right": 0.0}  # 0.0 to 1.0
        self.last_angle = {"left": None, "right": None}
        self.angle_history = {"left": deque(maxlen=10), "right": deque(maxlen=10)}

    def detect(self, landmark_history):
        """
        Detect if arm circles are being performed.

        Args:
            landmark_history: History of pose landmarks

        Returns:
            bool: True if arm circles are detected, False otherwise
        """
        if len(landmark_history) < 2:
            return False

        # Apply cooldown if active
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return False

        # Get current and previous frame landmarks
        current = landmark_history[-1]
        previous = landmark_history[-2]

        # Required landmarks for arm circles
        required_landmarks = [11, 12, 13, 14, 15, 16]  # Shoulders  # Elbows  # Wrists

        # Check if we have all required landmarks
        if not self.are_landmarks_visible(current, required_landmarks):
            self.reset_tracking()
            return False

        # Get landmarks
        left_shoulder = current[11]
        right_shoulder = current[12]
        left_elbow = current[13]
        right_elbow = current[14]
        left_wrist = current[15]
        right_wrist = current[16]

        # Previous wrist positions
        prev_left_wrist = previous[15]
        prev_right_wrist = previous[16]

        # Calculate wrist movement distance since last frame
        left_wrist_movement = self.calculate_distance(left_wrist, prev_left_wrist)
        right_wrist_movement = self.calculate_distance(right_wrist, prev_right_wrist)

        # Normalize to shoulder width for consistent movement detection
        shoulder_width = self.calculate_distance(left_shoulder, right_shoulder)
        normalized_left_movement = (
            left_wrist_movement / shoulder_width if shoulder_width > 0 else 0
        )
        normalized_right_movement = (
            right_wrist_movement / shoulder_width if shoulder_width > 0 else 0
        )

        # Store metrics for debugging
        self.debug_values.update(
            {
                "left_wrist_movement": normalized_left_movement,
                "right_wrist_movement": normalized_right_movement,
                "left_circle_completion": self.circle_completion["left"],
                "right_circle_completion": self.circle_completion["right"],
                "active_arm": self.active_arm,
            }
        )

        # Check if wrists are moving enough to potentially be arm circles
        left_moving = normalized_left_movement > self.min_wrist_movement
        right_moving = normalized_right_movement > self.min_wrist_movement

        # Add wrist positions to tracking (for circular motion detection)
        if left_moving:
            # Normalize position relative to shoulder to handle body movement
            relative_position = (
                left_wrist[0] - left_shoulder[0],
                left_wrist[1] - left_shoulder[1],
                left_wrist[2] - left_shoulder[2],
            )
            self.left_wrist_positions.append(relative_position)

            # Calculate angle from shoulder to wrist
            angle = self.calculate_angle_from_vertical(left_shoulder, left_wrist)
            self.track_circle_movement("left", angle)

        if right_moving:
            # Normalize position relative to shoulder
            relative_position = (
                right_wrist[0] - right_shoulder[0],
                right_wrist[1] - right_shoulder[1],
                right_wrist[2] - right_shoulder[2],
            )
            self.right_wrist_positions.append(relative_position)

            # Calculate angle from shoulder to wrist
            angle = self.calculate_angle_from_vertical(right_shoulder, right_wrist)
            self.track_circle_movement("right", angle)

        # Check if we have enough points to detect circle
        left_completed = self.check_circle_completion("left") >= 0.9
        right_completed = self.check_circle_completion("right") >= 0.9

        # Determine which arm is actively doing circles
        if left_completed and right_completed:
            self.active_arm = "both"
            self.rep_count += 1
            self.reset_circle_tracking()
            self.cooldown_counter = self.cooldown_frames
        elif left_completed:
            self.active_arm = "left"
            self.rep_count += 1
            self.reset_circle_tracking()
            self.cooldown_counter = self.cooldown_frames
        elif right_completed:
            self.active_arm = "right"
            self.rep_count += 1
            self.reset_circle_tracking()
            self.cooldown_counter = self.cooldown_frames

        # If no arm is making circular movements, reset tracking
        if not (left_moving or right_moving):
            self.active_arm = None

        # We're actively doing arm circles if we have a detection for either arm
        is_detected = self.active_arm is not None

        # Update detection state
        detection_changed = is_detected != self.last_detection_state
        self.last_detection_state = is_detected
        self.is_active = is_detected

        return is_detected

    def reset_tracking(self):
        """Reset all tracking data."""
        self.left_wrist_positions.clear()
        self.right_wrist_positions.clear()
        self.active_arm = None
        self.reset_circle_tracking()

    def reset_circle_tracking(self):
        """Reset circle completion tracking."""
        self.circle_completion = {"left": 0.0, "right": 0.0}
        self.last_angle = {"left": None, "right": None}
        self.angle_history["left"].clear()
        self.angle_history["right"].clear()

    def calculate_angle_from_vertical(self, shoulder, wrist):
        """
        Calculate the angle of the arm relative to vertical.

        Args:
            shoulder: Shoulder landmark (x, y, z, visibility)
            wrist: Wrist landmark (x, y, z, visibility)

        Returns:
            float: Angle in degrees (0-360)
        """
        # Calculate vector from shoulder to wrist
        dx = wrist[0] - shoulder[0]
        dy = wrist[1] - shoulder[1]

        # Calculate angle to vertical (negative y is up in screen coordinates)
        angle = math.degrees(math.atan2(dx, -dy))

        # Normalize to 0-360 degrees
        if angle < 0:
            angle += 360

        return angle

    def track_circle_movement(self, side, angle):
        """
        Track circular movement of the arm.

        Args:
            side: "left" or "right"
            angle: Current angle in degrees
        """
        # Add to angle history
        self.angle_history[side].append(angle)

        # If this is the first angle, just store it
        if self.last_angle[side] is None:
            self.last_angle[side] = angle
            return

        # Calculate angle difference (accounting for wrapping around 360°)
        diff = angle - self.last_angle[side]
        if diff > 180:
            diff -= 360
        elif diff < -180:
            diff += 360

        # For forward (clockwise) circles, angle increases
        # For reverse (counterclockwise) circles, angle decreases
        # We'll detect both, but need consistent direction for a single circle

        # Ignore very small movements
        if abs(diff) < 5:
            return

        # Check if we're moving in a consistent direction
        is_consistent = True
        if len(self.angle_history[side]) >= 3:
            # Get last few angle differences
            diffs = []
            for i in range(1, min(len(self.angle_history[side]), 3)):
                d = self.angle_history[side][i] - self.angle_history[side][i - 1]
                if d > 180:
                    d -= 360
                elif d < -180:
                    d += 360
                diffs.append(d)

            # Check if all differences have the same sign (consistent direction)
            is_consistent = all(d > 0 for d in diffs) or all(d < 0 for d in diffs)

        # If moving consistently, update completion percentage
        if is_consistent:
            # Calculate how much of a full circle we've completed
            completion_increment = abs(diff) / 360.0
            self.circle_completion[side] += completion_increment

            # Cap at 1.0 (one full circle)
            self.circle_completion[side] = min(1.0, self.circle_completion[side])

        # Store the current angle for next comparison
        self.last_angle[side] = angle

    def check_circle_completion(self, side):
        """
        Check how much of a circle has been completed.

        Args:
            side: "left" or "right"

        Returns:
            float: Completion percentage (0.0 to 1.0)
        """
        return self.circle_completion[side]
