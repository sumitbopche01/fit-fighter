"""
Pose detection module using MediaPipe.

This module handles pose detection and landmark extraction using MediaPipe.
"""

import cv2
import numpy as np
import mediapipe as mp


class PoseDetector:
    """Handles pose detection and landmark extraction."""

    def __init__(
        self,
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        """
        Initialize the pose detector.

        Args:
            static_image_mode: Whether to treat input as static images (False for video)
            model_complexity: Model complexity (0, 1, or 2)
            smooth_landmarks: Whether to smooth landmarks across frames
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
        """
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose

        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process_frame(self, frame):
        """
        Process a frame to detect pose landmarks.

        Args:
            frame: RGB frame to process

        Returns:
            tuple: (processed_frame, pose_results)
        """
        results = self.pose.process(frame)
        return results

    def draw_landmarks(self, frame, results):
        """
        Draw pose landmarks on the frame.

        Args:
            frame: Frame to draw on (BGR format)
            results: Pose detection results

        Returns:
            Frame with landmarks drawn
        """
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style(),
            )
        return frame

    def get_pose_landmarks(self, results):
        """
        Extract pose landmarks from results.

        Args:
            results: Pose detection results

        Returns:
            list: List of landmarks as (x, y, z, visibility) tuples or None if no pose detected
        """
        if not results.pose_landmarks:
            return None

        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmarks.append((landmark.x, landmark.y, landmark.z, landmark.visibility))
        return landmarks

    def release(self):
        """Release resources used by the pose detector."""
        self.pose.close()


# Landmark indices for reference
LANDMARK_INDICES = {
    "nose": 0,
    "left_eye_inner": 1,
    "left_eye": 2,
    "left_eye_outer": 3,
    "right_eye_inner": 4,
    "right_eye": 5,
    "right_eye_outer": 6,
    "left_ear": 7,
    "right_ear": 8,
    "mouth_left": 9,
    "mouth_right": 10,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_pinky": 17,
    "right_pinky": 18,
    "left_index": 19,
    "right_index": 20,
    "left_thumb": 21,
    "right_thumb": 22,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}
