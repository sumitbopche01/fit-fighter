"""
Burpee Exercise Detector.

This module provides a detector for burpee exercises based on pose landmarks.
"""

import numpy as np
from collections import deque
from fitfighter.core.base_detector import BaseExerciseDetector


class BurpeeDetector(BaseExerciseDetector):
    """
    Detector for burpee exercises.

    Detects burpees by analyzing multiple phases:
    1. Standing position
    2. Squat position
    3. Plank position
    4. Push-up (optional)
    5. Jump
    """

    def __init__(self, confidence_threshold=0.6, history_size=30):
        """
        Initialize the burpee detector.

        Args:
            confidence_threshold (float): Minimum visibility threshold for landmarks
            history_size (int): Number of frames to consider for detection
        """
        super().__init__(confidence_threshold, history_size)
        self.name = "Burpee"

        # State tracking
        self.current_phase = (
            "standing"  # "standing", "squat", "plank", "pushup", "jump"
        )
        self.phase_history = deque(maxlen=10)  # Longer history for complex movement
        self.cooldown_frames = 15  # Longer cooldown for complex movement
        self.cooldown_counter = 0
        self.last_detection_state = False
        self.debug_values = {}

        # Track the sequence of phases for a complete burpee
        self.phase_sequence = []
        self.max_sequence_length = 20  # Maximum phases to track

        # Timestamps for phase transitions
        self.phase_start_time = 0
        self.current_phase_duration = 0

        # MediaPipe landmark indices
        self.nose = 0
        self.left_shoulder = 11
        self.right_shoulder = 12
        self.left_elbow = 13
        self.right_elbow = 14
        self.left_wrist = 15
        self.right_wrist = 16
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

        # Thresholds for phase detection
        # Standing position thresholds
        self.standing_hip_height_threshold = 0.85  # Hip height relative to height
        self.standing_knee_angle_threshold = 160  # Knee angle when standing

        # Squat position thresholds
        self.squat_knee_angle_threshold = 100  # Knee angle when squatting
        self.squat_hip_height_threshold = 0.6  # Hip height relative to height

        # Plank position thresholds
        self.plank_body_angle_threshold = 30  # Body angle in plank position
        self.plank_hip_height_threshold = 0.4  # Hip height in plank position

        # Jump thresholds
        self.jump_height_threshold = 0.05  # Minimum jump height
        self.max_frames_between_phases = (
            15  # Max frames to allow between phase transitions
        )

        # Track vertical positions
        self.lowest_hip_position = None
        self.highest_hip_position = None
        self.reference_height = None  # For normalizing heights

    def detect(self, landmark_history):
        """
        Detect burpee exercises from landmark history.

        Args:
            landmark_history (list): List of landmarks for each frame

        Returns:
            bool: True if a burpee sequence is in progress
        """
        if not landmark_history or len(landmark_history) < 2:
            return False

        # Apply cooldown if active
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return False

        # Get the most recent landmarks
        landmarks = landmark_history[-1]
        prev_landmarks = landmark_history[-2]

        # Check if required landmarks are visible
        if not self.are_landmarks_visible(landmarks, self.required_landmarks):
            self._reset_detection_state()
            self.debug_values.update(
                {"state": "invalid", "reason": "Required landmarks not visible"}
            )
            return False

        # Calculate key metrics for phase detection
        metrics = self._calculate_metrics(landmarks, prev_landmarks)

        # Determine the current phase based on metrics
        new_phase = self._determine_phase(metrics)

        # Track phase transitions for sequence detection
        prev_phase = self.current_phase
        if new_phase != prev_phase:
            # Phase transition
            if len(self.phase_sequence) >= self.max_sequence_length:
                self.phase_sequence.pop(0)
            self.phase_sequence.append(new_phase)

        self.current_phase = new_phase

        # Add to history for smoother detection
        self.phase_history.append(new_phase)

        # Check if we have completed a burpee sequence
        completed_burpee = self._check_sequence_completion()

        if completed_burpee:
            self.rep_count += 1
            self.cooldown_counter = self.cooldown_frames
            self._reset_sequence()

        # Update active state based on current phase
        self.is_active = self.current_phase != "standing" or completed_burpee

        # Update detection state for event tracking
        detection_changed = self.is_active != self.last_detection_state
        self.last_detection_state = self.is_active

        # Update debug information
        self.debug_values.update(
            {
                **metrics,
                "current_phase": self.current_phase,
                "phase_sequence": self.phase_sequence.copy(),
                "burpee_count": self.rep_count,
                "is_active": self.is_active,
            }
        )

        return self.is_active

    def _calculate_metrics(self, landmarks, prev_landmarks):
        """Calculate key metrics for phase detection."""
        # Get key positions
        nose = landmarks[self.nose]
        left_shoulder = landmarks[self.left_shoulder]
        right_shoulder = landmarks[self.right_shoulder]
        left_hip = landmarks[self.left_hip]
        right_hip = landmarks[self.right_hip]
        left_knee = landmarks[self.left_knee]
        right_knee = landmarks[self.right_knee]
        left_ankle = landmarks[self.left_ankle]
        right_ankle = landmarks[self.right_ankle]

        # Calculate midpoints for more robust detection
        shoulder_midpoint = [
            (left_shoulder[i] + right_shoulder[i]) / 2 for i in range(3)
        ]
        hip_midpoint = [(left_hip[i] + right_hip[i]) / 2 for i in range(3)]
        knee_midpoint = [(left_knee[i] + right_knee[i]) / 2 for i in range(3)]
        ankle_midpoint = [(left_ankle[i] + right_ankle[i]) / 2 for i in range(3)]

        # Calculate joint angles
        left_knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = self.calculate_angle(right_hip, right_knee, right_ankle)
        avg_knee_angle = (left_knee_angle + right_knee_angle) / 2

        # Calculate body angle (angle between shoulders, hips, and knees)
        body_angle = self.calculate_angle(
            shoulder_midpoint, hip_midpoint, knee_midpoint
        )

        # Calculate approximate person height if not already set
        if self.reference_height is None:
            # Use distance from ankle to nose as reference height
            self.reference_height = abs(nose[1] - ankle_midpoint[1])

        # Calculate normalized heights (y increases downward)
        nose_height = nose[1]
        hip_height = hip_midpoint[1]
        shoulder_height = shoulder_midpoint[1]

        # Track hip movement for jump detection
        if self.lowest_hip_position is None or hip_height > self.lowest_hip_position:
            self.lowest_hip_position = hip_height
        if self.highest_hip_position is None or hip_height < self.highest_hip_position:
            self.highest_hip_position = hip_height

        # Normalize heights relative to person's height
        norm_hip_height = 1 - (
            hip_height / self.reference_height if self.reference_height else 1
        )

        # Calculate vertical velocity for jump detection
        prev_hip_midpoint = [
            (prev_landmarks[self.left_hip][i] + prev_landmarks[self.right_hip][i]) / 2
            for i in range(3)
        ]
        vertical_velocity = (
            prev_hip_midpoint[1] - hip_midpoint[1]
        )  # Positive means moving up

        # Calculate height relative to full height
        height_ratio = (
            abs(nose_height - ankle_midpoint[1]) / self.reference_height
            if self.reference_height
            else 1
        )

        # Calculate distance between feet for jump detection
        feet_distance = self.calculate_distance(left_ankle, right_ankle)

        return {
            "knee_angle": avg_knee_angle,
            "body_angle": body_angle,
            "hip_height": hip_height,
            "norm_hip_height": norm_hip_height,
            "vertical_velocity": vertical_velocity,
            "height_ratio": height_ratio,
            "feet_distance": feet_distance,
        }

    def _determine_phase(self, metrics):
        """Determine the current phase of the burpee."""
        # Extract metrics
        knee_angle = metrics["knee_angle"]
        body_angle = metrics["body_angle"]
        norm_hip_height = metrics["norm_hip_height"]
        vertical_velocity = metrics["vertical_velocity"]

        # Phase detection logic
        if vertical_velocity > self.jump_height_threshold:
            # If moving up rapidly, it's a jump
            return "jump"
        elif (
            norm_hip_height > self.standing_hip_height_threshold
            and knee_angle > self.standing_knee_angle_threshold
        ):
            # Standing upright with straight legs
            return "standing"
        elif (
            norm_hip_height > self.squat_hip_height_threshold
            and knee_angle < self.squat_knee_angle_threshold
        ):
            # Low position with bent knees
            return "squat"
        elif (
            norm_hip_height < self.plank_hip_height_threshold
            and abs(body_angle - 180) < self.plank_body_angle_threshold
        ):
            # Low position with straight body
            return "plank"
        else:
            # Transitioning between phases
            return "transitioning"

    def _check_sequence_completion(self):
        """Check if we have completed a valid burpee sequence."""
        # A complete burpee sequence: standing → squat → plank → (optional pushup) → squat → jump → standing
        if len(self.phase_sequence) < 4:
            return False

        # Look for the pattern in the recent sequence
        # We'll use a simplified approach: check if all required phases occurred in the correct order
        found_standing_start = False
        found_squat_down = False
        found_plank = False
        found_squat_up = False
        found_jump = False
        found_standing_end = False

        for phase in self.phase_sequence:
            if not found_standing_start and phase == "standing":
                found_standing_start = True
            elif found_standing_start and not found_squat_down and phase == "squat":
                found_squat_down = True
            elif found_squat_down and not found_plank and phase == "plank":
                found_plank = True
            elif found_plank and not found_squat_up and phase == "squat":
                found_squat_up = True
            elif found_squat_up and not found_jump and phase == "jump":
                found_jump = True
            elif found_jump and not found_standing_end and phase == "standing":
                found_standing_end = True

        # Return true if we found all required phases in order
        sequence_complete = (
            found_standing_start
            and found_squat_down
            and found_plank
            and found_squat_up
            and found_jump
        )

        return sequence_complete

    def _reset_sequence(self):
        """Reset the sequence tracking after a complete burpee."""
        self.phase_sequence = []

    def _reset_detection_state(self):
        """Reset the burpee detection state."""
        self.current_phase = "standing"
        self.phase_history.clear()
        self.phase_sequence = []
        self.lowest_hip_position = None
        self.highest_hip_position = None
        self.is_active = False

    def reset(self):
        """Reset the detector completely, including count."""
        super().reset()
        self._reset_detection_state()
        self.cooldown_counter = 0
        self.rep_count = 0
        self.reference_height = None
