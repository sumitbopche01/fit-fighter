"""
Base exercise detector.

This module provides a base class for exercise detectors with common functionality.
"""

import numpy as np
from abc import ABC, abstractmethod
from collections import deque


class BaseExerciseDetector(ABC):
    """
    Base class for all exercise detectors.

    Provides common functionality and defines the interface that all exercise
    detectors must implement.
    """

    def __init__(self, confidence_threshold=0.6, history_size=30):
        """
        Initialize the base exercise detector.

        Args:
            confidence_threshold (float): Minimum visibility threshold for landmarks
            history_size (int): Number of frames to consider for detection
        """
        self.confidence_threshold = confidence_threshold
        self.history_size = history_size
        self.name = self.__class__.__name__.replace("Detector", "")
        self.rep_count = 0
        self.is_active = False
        self.cooldown_frames = 0
        self.debug_info = {}
        self.last_detection_state = False

    @abstractmethod
    def detect(self, landmark_history):
        """
        Detect if the exercise is being performed.

        Args:
            landmark_history (list): History of pose landmarks

        Returns:
            bool: True if exercise is detected, False otherwise
        """
        pass

    def check_landmarks_visibility(self, landmarks, required_landmarks):
        """
        Check if required landmarks are visible with sufficient confidence.

        Args:
            landmarks (dict): Current frame landmarks
            required_landmarks (list): List of landmark indices to check

        Returns:
            bool: True if all required landmarks are visible, False otherwise
        """
        if not landmarks:
            return False

        for lm_idx in required_landmarks:
            if lm_idx not in landmarks:
                return False

            visibility = landmarks[lm_idx].get("visibility", 0)
            if visibility < self.confidence_threshold:
                return False

        return True

    def get_landmark_position(self, landmarks, landmark_idx):
        """
        Get normalized position of a landmark.

        Args:
            landmarks (dict): Frame landmarks
            landmark_idx (int): Index of the landmark

        Returns:
            tuple: (x, y, z) position of the landmark, or None if not visible
        """
        if not landmarks or landmark_idx not in landmarks:
            return None

        lm = landmarks[landmark_idx]
        visibility = lm.get("visibility", 0)

        if visibility < self.confidence_threshold:
            return None

        return (lm.get("x", 0), lm.get("y", 0), lm.get("z", 0))

    def calculate_distance(self, point1, point2):
        """
        Calculate Euclidean distance between two 3D points.

        Args:
            point1 (tuple): First point (x, y, z)
            point2 (tuple): Second point (x, y, z)

        Returns:
            float: Euclidean distance between points
        """
        if point1 is None or point2 is None:
            return float("inf")

        return (
            (point1[0] - point2[0]) ** 2
            + (point1[1] - point2[1]) ** 2
            + (point1[2] - point2[2]) ** 2
        ) ** 0.5

    def get_debug_info(self):
        """
        Get debug information for the detector.

        Returns:
            dict: Debug information
        """
        debug_info = {
            "name": self.name,
            "is_active": self.is_active,
            "rep_count": self.rep_count,
        }

        # Add any custom debug values
        debug_info.update(self.debug_info)

        return debug_info

    def update_debug_info(self, **kwargs):
        """
        Update debug information.

        Args:
            **kwargs: Key-value pairs to add to debug information
        """
        self.debug_info.update(kwargs)

    def calculate_angle(self, p1, p2, p3):
        """
        Calculate the angle between three points.

        Args:
            p1, p2, p3: Points as (x, y, z, visibility) tuples, where p2 is the vertex

        Returns:
            float: Angle in degrees
        """
        # Convert to 2D coordinates (x,y)
        a = np.array([p1[0], p1[1]])
        b = np.array([p2[0], p2[1]])
        c = np.array([p3[0], p3[1]])

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

        return np.degrees(angle)

    def calculate_3d_angle(self, p1, p2, p3):
        """
        Calculate the angle between three points in 3D space.

        Args:
            p1, p2, p3: Points as (x, y, z, visibility) tuples, where p2 is the vertex

        Returns:
            float: Angle in degrees
        """
        # Use 3D coordinates (x,y,z)
        a = np.array([p1[0], p1[1], p1[2]])
        b = np.array([p2[0], p2[1], p2[2]])
        c = np.array([p3[0], p3[1], p3[2]])

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

        return np.degrees(angle)

    def is_landmark_visible(self, landmark, threshold=None):
        """
        Check if a landmark has sufficient visibility.

        Args:
            landmark: Landmark as (x, y, z, visibility) tuple
            threshold: Optional visibility threshold, defaults to self.confidence_threshold

        Returns:
            bool: True if landmark is visible, False otherwise
        """
        if threshold is None:
            threshold = self.confidence_threshold

        return landmark[3] >= threshold

    def are_landmarks_visible(self, landmarks, indices, threshold=None):
        """
        Check if multiple landmarks have sufficient visibility.

        Args:
            landmarks: List of landmarks
            indices: List of landmark indices to check
            threshold: Optional visibility threshold

        Returns:
            bool: True if all specified landmarks are visible, False otherwise
        """
        if threshold is None:
            threshold = self.confidence_threshold

        for idx in indices:
            if not self.is_landmark_visible(landmarks[idx], threshold):
                return False

        return True

    def reset(self):
        """Reset the detector state, including counters."""
        self.rep_count = 0
        self.is_active = False
        self.last_detection_state = False
        self.debug_info = {}
