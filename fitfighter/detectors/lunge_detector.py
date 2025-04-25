"""
Lunge Exercise Detector.

This module provides a detector for lunge exercises based on pose landmarks.
"""

from fitfighter.core.base_detector import BaseExerciseDetector
import numpy as np


class LungeDetector(BaseExerciseDetector):
    """
    Detector for lunge exercises.

    Detects lunges by analyzing the position of hips, knees, and ankles,
    along with the vertical displacement during the movement.
    """

    def __init__(self, confidence_threshold=0.6, history_size=30):
        """
        Initialize the lunge detector.

        Args:
            confidence_threshold (float): Minimum visibility threshold for landmarks
            history_size (int): Number of frames to consider for detection
        """
        super().__init__(confidence_threshold, history_size)
        self.name = "Lunge"
        self.state = "up"  # up, down
        self.consecutive_frames_threshold = 3
        self.consecutive_matching_frames = 0
        self.cooldown_counter = 0

        # Landmarks for MediaPipe model
        self.left_shoulder = 11
        self.right_shoulder = 12
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
            self.left_ankle,
            self.right_ankle,
        ]

    def detect(self, landmark_history):
        """
        Detect lunge exercises from landmark history.

        Args:
            landmark_history (list): List of landmarks for each frame

        Returns:
            bool: True if lunge detected, False otherwise
        """
        if not landmark_history or len(landmark_history) < 2:
            return False

        # Get the most recent landmarks
        landmarks = landmark_history[-1]

        # Check if required landmarks are visible
        if not self.check_landmarks_visibility(landmarks, self.required_landmarks):
            self.update_debug_info(
                state=self.state, reason="Required landmarks not visible"
            )
            return False

        # Get positions
        left_shoulder = self.get_landmark_position(landmarks, self.left_shoulder)
        right_shoulder = self.get_landmark_position(landmarks, self.right_shoulder)
        left_hip = self.get_landmark_position(landmarks, self.left_hip)
        right_hip = self.get_landmark_position(landmarks, self.right_hip)
        left_knee = self.get_landmark_position(landmarks, self.left_knee)
        right_knee = self.get_landmark_position(landmarks, self.right_knee)
        left_ankle = self.get_landmark_position(landmarks, self.left_ankle)
        right_ankle = self.get_landmark_position(landmarks, self.right_ankle)

        # Calculate midpoints
        hip_midpoint = [
            (left_hip[0] + right_hip[0]) / 2,
            (left_hip[1] + right_hip[1]) / 2,
            (left_hip[2] + right_hip[2]) / 2,
        ]

        # Calculate knee angles
        left_knee_angle = self._calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = self._calculate_angle(right_hip, right_knee, right_ankle)

        # Determine which leg is forward (the one with the smaller knee angle typically)
        forward_knee_angle = min(left_knee_angle, right_knee_angle)
        rear_knee_angle = max(left_knee_angle, right_knee_angle)

        # Check hip height
        shoulder_hip_y_diff = abs(left_shoulder[1] - hip_midpoint[1])
        hip_y_position = hip_midpoint[1]

        # Track hip height across frames to detect vertical movement
        if len(landmark_history) > 5:
            # Get hip position from 5 frames ago for comparison
            previous_landmarks = landmark_history[-5]
            if self.check_landmarks_visibility(
                previous_landmarks, [self.left_hip, self.right_hip]
            ):
                prev_left_hip = self.get_landmark_position(
                    previous_landmarks, self.left_hip
                )
                prev_right_hip = self.get_landmark_position(
                    previous_landmarks, self.right_hip
                )

                prev_hip_midpoint = [
                    (prev_left_hip[0] + prev_right_hip[0]) / 2,
                    (prev_left_hip[1] + prev_right_hip[1]) / 2,
                    (prev_left_hip[2] + prev_right_hip[2]) / 2,
                ]

                hip_movement = hip_midpoint[1] - prev_hip_midpoint[1]
            else:
                hip_movement = 0
        else:
            hip_movement = 0

        # Update debug information
        self.update_debug_info(
            state=self.state,
            left_knee_angle=left_knee_angle,
            right_knee_angle=right_knee_angle,
            forward_knee_angle=forward_knee_angle,
            rear_knee_angle=rear_knee_angle,
            hip_y_position=hip_y_position,
            hip_movement=hip_movement,
        )

        # Determine state based on knee angles and hip position
        prev_state = self.state

        # In a lunge, forward knee angle is ~90 degrees, rear knee angle is closer to 90-120 degrees
        # and hip is low
        if forward_knee_angle < 110 and rear_knee_angle < 150 and hip_movement <= 0:
            new_state = "down"
        else:
            new_state = "up"

        # Handle state transitions with consecutive frame requirement
        if new_state == prev_state:
            self.consecutive_matching_frames += 1
        else:
            self.consecutive_matching_frames = 1

        # Only change state if we've seen enough consecutive frames
        if self.consecutive_matching_frames >= self.consecutive_frames_threshold:
            self.state = new_state

        # Detect repetition when transitioning from down to up
        if self.cooldown_counter <= 0 and prev_state == "down" and self.state == "up":
            self.rep_count += 1
            self.cooldown_counter = (
                15  # Prevent counting too many reps in quick succession
            )
            self.is_active = True
            return True

        # Update cooldown counter
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1

        # Exercise is considered active if at least one knee is bent
        self.is_active = forward_knee_angle < 130

        return self.is_active

    def _calculate_angle(self, a, b, c):
        """
        Calculate the angle between three points, with b as the vertex.

        Args:
            a (list): 3D coordinates of the first point
            b (list): 3D coordinates of the vertex
            c (list): 3D coordinates of the third point

        Returns:
            float: Angle in degrees
        """
        # Calculate vectors
        ba = [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
        bc = [c[0] - b[0], c[1] - b[1], c[2] - b[2]]

        # Calculate dot product
        dot_product = sum(ba[i] * bc[i] for i in range(3))

        # Calculate magnitudes
        ba_magnitude = np.sqrt(sum(ba[i] * ba[i] for i in range(3)))
        bc_magnitude = np.sqrt(sum(bc[i] * bc[i] for i in range(3)))

        # Calculate angle in radians
        cosine = dot_product / (ba_magnitude * bc_magnitude)
        # Clamp cosine to [-1, 1] to avoid domain errors
        cosine = max(-1, min(1, cosine))
        angle_rad = np.arccos(cosine)

        # Convert to degrees
        angle_deg = np.degrees(angle_rad)

        return angle_deg
