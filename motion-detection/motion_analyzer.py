"""
Motion analyzer module for FitFighter.

This module processes pose landmarks to detect specific exercises.
"""

import math
import numpy as np
from collections import deque
from pose_detector import LANDMARK_INDICES


class MotionAnalyzer:
    """Analyzes pose landmarks to detect exercises."""

    def __init__(self, history_length=30):
        """
        Initialize the motion analyzer.

        Args:
            history_length: Number of frames to keep in history for motion analysis
        """
        self.history_length = history_length
        self.landmark_history = deque(maxlen=history_length)
        self.exercise_detectors = {
            "punch": PunchDetector(),
            "squat": SquatDetector(),
            "plank": PlankDetector(),
        }
        self.current_states = {"punch": False, "squat": False, "plank": False}

    def add_landmarks(self, landmarks):
        """
        Add landmarks to history.

        Args:
            landmarks: List of landmarks from pose detector
        """
        if landmarks:
            self.landmark_history.append(landmarks)

    def analyze_motion(self):
        """
        Analyze the landmark history to detect exercises.

        Returns:
            dict: Current state of each exercise
        """
        if len(self.landmark_history) < 2:
            return self.current_states

        # Update each exercise detector
        for exercise, detector in self.exercise_detectors.items():
            self.current_states[exercise] = detector.detect(self.landmark_history)

        return self.current_states


class ExerciseDetector:
    """Base class for exercise detectors."""

    def detect(self, landmark_history):
        """
        Detect if the exercise is being performed.

        Args:
            landmark_history: History of pose landmarks

        Returns:
            bool: True if exercise detected, False otherwise
        """
        raise NotImplementedError("Subclasses must implement this method")


class PunchDetector(ExerciseDetector):
    """Detects punching motion."""

    def __init__(self, velocity_threshold=0.05, confidence_threshold=0.6):
        """
        Initialize the punch detector.

        Args:
            velocity_threshold: Minimum velocity to consider a punch
            confidence_threshold: Minimum landmark visibility to consider valid
        """
        self.velocity_threshold = velocity_threshold
        self.confidence_threshold = confidence_threshold

    def detect(self, landmark_history):
        """
        Detect if a punch is being performed.

        Args:
            landmark_history: History of pose landmarks

        Returns:
            bool: True if punch detected, False otherwise
        """
        if len(landmark_history) < 2:
            return False

        # Get current and previous landmarks
        current = landmark_history[-1]
        previous = landmark_history[-2]

        # Check if landmarks are valid
        if not current or not previous:
            return False

        # Calculate wrist velocity for both hands
        left_wrist_idx = LANDMARK_INDICES["left_wrist"]
        right_wrist_idx = LANDMARK_INDICES["right_wrist"]

        # Check if landmarks have sufficient visibility
        if (
            current[left_wrist_idx][3] < self.confidence_threshold
            and current[right_wrist_idx][3] < self.confidence_threshold
        ):
            return False

        # Calculate velocity
        left_velocity = self._calculate_velocity(
            current[left_wrist_idx], previous[left_wrist_idx]
        )
        right_velocity = self._calculate_velocity(
            current[right_wrist_idx], previous[right_wrist_idx]
        )

        # Detect punch if either wrist has high forward velocity
        return (
            left_velocity > self.velocity_threshold
            or right_velocity > self.velocity_threshold
        )

    def _calculate_velocity(self, current, previous):
        """
        Calculate the velocity of a landmark.

        Args:
            current: Current landmark position (x, y, z, visibility)
            previous: Previous landmark position (x, y, z, visibility)

        Returns:
            float: Velocity magnitude
        """
        # Mainly interested in forward motion (z-axis)
        z_velocity = previous[2] - current[2]

        # Also consider x and y components
        x_velocity = current[0] - previous[0]
        y_velocity = current[1] - previous[1]

        # Return velocity magnitude with emphasis on forward motion
        return z_velocity * 2 + math.sqrt(x_velocity**2 + y_velocity**2)


class SquatDetector(ExerciseDetector):
    """Detects squatting motion."""

    def __init__(self, hip_threshold=0.1, confidence_threshold=0.6):
        """
        Initialize the squat detector.

        Args:
            hip_threshold: Minimum hip movement to consider a squat
            confidence_threshold: Minimum landmark visibility to consider valid
        """
        self.hip_threshold = hip_threshold
        self.confidence_threshold = confidence_threshold
        self.hip_height_history = deque(maxlen=10)

    def detect(self, landmark_history):
        """
        Detect if a squat is being performed.

        Args:
            landmark_history: History of pose landmarks

        Returns:
            bool: True if squat detected, False otherwise
        """
        if len(landmark_history) < 1:
            return False

        # Get current landmarks
        current = landmark_history[-1]

        # Check if landmarks are valid
        if not current:
            return False

        # Get hip and knee landmarks
        left_hip_idx = LANDMARK_INDICES["left_hip"]
        right_hip_idx = LANDMARK_INDICES["right_hip"]
        left_knee_idx = LANDMARK_INDICES["left_knee"]
        right_knee_idx = LANDMARK_INDICES["right_knee"]
        left_ankle_idx = LANDMARK_INDICES["left_ankle"]
        right_ankle_idx = LANDMARK_INDICES["right_ankle"]

        # Check if landmarks have sufficient visibility
        if (
            current[left_hip_idx][3] < self.confidence_threshold
            or current[right_hip_idx][3] < self.confidence_threshold
            or current[left_knee_idx][3] < self.confidence_threshold
            or current[right_knee_idx][3] < self.confidence_threshold
        ):
            return False

        # Calculate average hip height
        hip_height = (current[left_hip_idx][1] + current[right_hip_idx][1]) / 2
        self.hip_height_history.append(hip_height)

        # Not enough history to detect squat
        if len(self.hip_height_history) < 5:
            return False

        # Calculate hip angle
        left_hip_angle = self._calculate_angle(
            current[left_hip_idx], current[left_knee_idx], current[left_ankle_idx]
        )
        right_hip_angle = self._calculate_angle(
            current[right_hip_idx], current[right_knee_idx], current[right_ankle_idx]
        )

        # Average angle of both hips
        avg_hip_angle = (left_hip_angle + right_hip_angle) / 2

        # Detect squat if hip angle is small (bent knees) and hip has moved down
        hip_min = min(self.hip_height_history)
        hip_max = max(self.hip_height_history)
        hip_range = hip_max - hip_min

        return hip_range > self.hip_threshold and avg_hip_angle < 120

    def _calculate_angle(self, p1, p2, p3):
        """
        Calculate the angle between three points.

        Args:
            p1, p2, p3: Points as (x, y, z, visibility) tuples

        Returns:
            float: Angle in degrees
        """
        a = np.array([p1[0], p1[1]])
        b = np.array([p2[0], p2[1]])
        c = np.array([p3[0], p3[1]])

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

        return np.degrees(angle)


class PlankDetector(ExerciseDetector):
    """Detects plank position."""

    def __init__(
        self,
        stability_threshold=0.02,
        alignment_threshold=0.1,
        confidence_threshold=0.6,
    ):
        """
        Initialize the plank detector.

        Args:
            stability_threshold: Maximum movement to consider stable
            alignment_threshold: Maximum misalignment allowed
            confidence_threshold: Minimum landmark visibility to consider valid
        """
        self.stability_threshold = stability_threshold
        self.alignment_threshold = alignment_threshold
        self.confidence_threshold = confidence_threshold

    def detect(self, landmark_history):
        """
        Detect if a plank is being performed.

        Args:
            landmark_history: History of pose landmarks

        Returns:
            bool: True if plank detected, False otherwise
        """
        if len(landmark_history) < 5:  # Need several frames to determine stability
            return False

        # Check stability of position
        stable = self._check_stability(landmark_history)
        if not stable:
            return False

        # Get current landmarks
        current = landmark_history[-1]

        # Check if landmarks are valid
        if not current:
            return False

        # Get relevant landmarks
        left_shoulder_idx = LANDMARK_INDICES["left_shoulder"]
        right_shoulder_idx = LANDMARK_INDICES["right_shoulder"]
        left_hip_idx = LANDMARK_INDICES["left_hip"]
        right_hip_idx = LANDMARK_INDICES["right_hip"]
        left_ankle_idx = LANDMARK_INDICES["left_ankle"]
        right_ankle_idx = LANDMARK_INDICES["right_ankle"]

        # Check if landmarks have sufficient visibility
        if (
            current[left_shoulder_idx][3] < self.confidence_threshold
            or current[right_shoulder_idx][3] < self.confidence_threshold
            or current[left_hip_idx][3] < self.confidence_threshold
            or current[right_hip_idx][3] < self.confidence_threshold
            or current[left_ankle_idx][3] < self.confidence_threshold
            or current[right_ankle_idx][3] < self.confidence_threshold
        ):
            return False

        # Check if body is horizontal (shoulders and hips at similar height)
        left_alignment = abs(current[left_shoulder_idx][1] - current[left_hip_idx][1])
        right_alignment = abs(
            current[right_shoulder_idx][1] - current[right_hip_idx][1]
        )
        hip_alignment = abs(current[left_hip_idx][1] - current[right_hip_idx][1])

        # Check if body is straight (shoulders, hips, and ankles in line)
        body_straight = self._check_body_alignment(current)

        return (
            left_alignment < self.alignment_threshold
            and right_alignment < self.alignment_threshold
            and hip_alignment < self.alignment_threshold
            and body_straight
        )

    def _check_stability(self, landmark_history):
        """
        Check if the pose is stable over time.

        Args:
            landmark_history: History of pose landmarks

        Returns:
            bool: True if pose is stable, False otherwise
        """
        # Compare the last 5 frames
        samples = min(5, len(landmark_history))
        if samples < 3:
            return False

        # Check stability of key points (shoulders, hips)
        left_shoulder_idx = LANDMARK_INDICES["left_shoulder"]
        right_shoulder_idx = LANDMARK_INDICES["right_shoulder"]
        left_hip_idx = LANDMARK_INDICES["left_hip"]
        right_hip_idx = LANDMARK_INDICES["right_hip"]

        # Calculate movement of key points
        movement = []
        for i in range(1, samples):
            current = landmark_history[-i]
            previous = landmark_history[-(i + 1)]

            if not current or not previous:
                continue

            # Average movement of key points
            points_movement = [
                self._calculate_distance(
                    current[left_shoulder_idx], previous[left_shoulder_idx]
                ),
                self._calculate_distance(
                    current[right_shoulder_idx], previous[right_shoulder_idx]
                ),
                self._calculate_distance(current[left_hip_idx], previous[left_hip_idx]),
                self._calculate_distance(
                    current[right_hip_idx], previous[right_hip_idx]
                ),
            ]

            movement.append(sum(points_movement) / len(points_movement))

        # Return True if average movement is below threshold
        return (
            sum(movement) / len(movement) < self.stability_threshold
            if movement
            else False
        )

    def _check_body_alignment(self, landmarks):
        """
        Check if the body is properly aligned for a plank.

        Args:
            landmarks: Current frame landmarks

        Returns:
            bool: True if body is aligned, False otherwise
        """
        # Get relevant landmarks
        shoulder_idx = LANDMARK_INDICES["left_shoulder"]
        hip_idx = LANDMARK_INDICES["left_hip"]
        ankle_idx = LANDMARK_INDICES["left_ankle"]

        # Check if the shoulder, hip, and ankle form approximately a straight line
        angle = self._calculate_angle(
            landmarks[shoulder_idx], landmarks[hip_idx], landmarks[ankle_idx]
        )

        # A straight line would be close to 180 degrees
        return abs(angle - 170) < 20

    def _calculate_distance(self, p1, p2):
        """
        Calculate the distance between two landmarks.

        Args:
            p1, p2: Points as (x, y, z, visibility) tuples

        Returns:
            float: Euclidean distance
        """
        return math.sqrt(
            (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2
        )
