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

    def __init__(
        self,
        velocity_threshold=0.08,
        confidence_threshold=0.6,
        cooldown_frames=8,
        direction_weight=0.4,
    ):
        """
        Initialize the punch detector.

        Args:
            velocity_threshold: Minimum velocity to consider a punch
            confidence_threshold: Minimum landmark visibility to consider valid
            cooldown_frames: Number of frames to wait before detecting another punch
            direction_weight: Weight to apply to forward motion (z-axis)
        """
        self.velocity_threshold = velocity_threshold
        self.confidence_threshold = confidence_threshold
        self.cooldown_frames = cooldown_frames
        self.direction_weight = direction_weight
        self.cooldown_counter = 0
        self.consecutive_frames_required = (
            1  # Boxers punch fast, so we need quicker detection
        )
        self.consecutive_detection_counter = 0
        self.last_detection_state = False
        # Debug info
        self.debug_info = {
            "left_vel": 0,
            "right_vel": 0,
            "left_dir": False,
            "right_dir": False,
        }

    def detect(self, landmark_history):
        """
        Detect if a punch is being performed.

        Args:
            landmark_history: History of pose landmarks

        Returns:
            bool: True if punch detected, False otherwise
        """
        # Apply cooldown if active
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return False

        if len(landmark_history) < 3:  # Need at least 3 frames for better detection
            return False

        # Get landmarks from last three frames
        current = landmark_history[-1]
        previous = landmark_history[-2]
        prev_prev = landmark_history[-3]

        # Check if landmarks are valid
        if not current or not previous or not prev_prev:
            return False

        # Get indices for all required landmarks
        left_wrist_idx = LANDMARK_INDICES["left_wrist"]
        right_wrist_idx = LANDMARK_INDICES["right_wrist"]
        left_shoulder_idx = LANDMARK_INDICES["left_shoulder"]
        right_shoulder_idx = LANDMARK_INDICES["right_shoulder"]
        left_elbow_idx = LANDMARK_INDICES["left_elbow"]
        right_elbow_idx = LANDMARK_INDICES["right_elbow"]
        nose_idx = LANDMARK_INDICES["nose"]  # For determining body orientation

        # Check if landmarks have sufficient visibility
        left_visible = (
            current[left_wrist_idx][3] >= self.confidence_threshold
            and current[left_shoulder_idx][3] >= self.confidence_threshold
            and current[left_elbow_idx][3] >= self.confidence_threshold
        )

        right_visible = (
            current[right_wrist_idx][3] >= self.confidence_threshold
            and current[right_shoulder_idx][3] >= self.confidence_threshold
            and current[right_elbow_idx][3] >= self.confidence_threshold
        )

        if not left_visible and not right_visible:
            self.consecutive_detection_counter = 0
            return False

        # Determine body orientation (which way user is facing)
        # This helps adjust for boxing stance where punches might have horizontal components
        body_orientation = self._determine_body_orientation(
            current, nose_idx, left_shoulder_idx, right_shoulder_idx
        )

        # Calculate velocity and check for punch motion
        left_punch = False
        right_punch = False

        if left_visible:
            # Calculate wrist velocity considering both forward and horizontal components
            left_wrist_vel = self._calculate_boxing_velocity(
                current[left_wrist_idx],
                previous[left_wrist_idx],
                current[left_shoulder_idx],
                previous[left_shoulder_idx],
                body_orientation,
                "left",
            )

            # Check directional motion (depends on boxing stance)
            left_direction = self._is_punch_direction(
                current[left_wrist_idx],
                previous[left_wrist_idx],
                prev_prev[left_wrist_idx],
                body_orientation,
                "left",
            )

            # Arms don't always need to be fully extending in boxing
            left_extending = self._is_boxing_extension(
                current[left_shoulder_idx],
                current[left_elbow_idx],
                current[left_wrist_idx],
                previous[left_shoulder_idx],
                previous[left_elbow_idx],
                previous[left_wrist_idx],
            )

            # Store debug info
            self.debug_info["left_vel"] = left_wrist_vel
            self.debug_info["left_dir"] = left_direction

            left_punch = left_wrist_vel > self.velocity_threshold and left_direction

        if right_visible:
            # Calculate wrist velocity
            right_wrist_vel = self._calculate_boxing_velocity(
                current[right_wrist_idx],
                previous[right_wrist_idx],
                current[right_shoulder_idx],
                previous[right_shoulder_idx],
                body_orientation,
                "right",
            )

            # Check direction
            right_direction = self._is_punch_direction(
                current[right_wrist_idx],
                previous[right_wrist_idx],
                prev_prev[right_wrist_idx],
                body_orientation,
                "right",
            )

            # Check extension
            right_extending = self._is_boxing_extension(
                current[right_shoulder_idx],
                current[right_elbow_idx],
                current[right_wrist_idx],
                previous[right_shoulder_idx],
                previous[right_elbow_idx],
                previous[right_wrist_idx],
            )

            # Store debug info
            self.debug_info["right_vel"] = right_wrist_vel
            self.debug_info["right_dir"] = right_direction

            right_punch = right_wrist_vel > self.velocity_threshold and right_direction

        # Check if either hand is punching
        current_detection = left_punch or right_punch

        # Require consecutive frames for more robust detection (reduced for boxing)
        if current_detection:
            self.consecutive_detection_counter += 1
        else:
            self.consecutive_detection_counter = 0

        is_punch = (
            self.consecutive_detection_counter >= self.consecutive_frames_required
        )

        # Start cooldown when a punch is detected
        if is_punch and not self.last_detection_state:
            self.cooldown_counter = self.cooldown_frames

        self.last_detection_state = is_punch
        return is_punch

    def _determine_body_orientation(
        self, landmarks, nose_idx, left_shoulder_idx, right_shoulder_idx
    ):
        """
        Determine which way the user is facing based on shoulder and nose positions.

        Returns:
            dict: Containing orientation information (side_facing, left_forward)
        """
        # Check if user is facing sideways based on shoulder alignment with camera
        shoulders_x_diff = abs(
            landmarks[left_shoulder_idx][0] - landmarks[right_shoulder_idx][0]
        )
        shoulders_z_diff = abs(
            landmarks[left_shoulder_idx][2] - landmarks[right_shoulder_idx][2]
        )

        # If x difference is small and z difference is large, user is likely sideways
        side_facing = shoulders_z_diff > shoulders_x_diff

        # Determine which side is forward
        left_forward = False
        if side_facing:
            left_forward = (
                landmarks[left_shoulder_idx][2] < landmarks[right_shoulder_idx][2]
            )

        return {"side_facing": side_facing, "left_forward": left_forward}

    def _calculate_boxing_velocity(
        self,
        wrist_current,
        wrist_prev,
        shoulder_current,
        shoulder_prev,
        orientation,
        side,
    ):
        """
        Calculate punch velocity accounting for boxing stance.

        Args:
            wrist_current, wrist_prev: Wrist positions
            shoulder_current, shoulder_prev: Shoulder positions
            orientation: Body orientation information
            side: "left" or "right" to indicate which arm

        Returns:
            float: Velocity magnitude adjusted for boxing stance
        """
        # Calculate wrist movement relative to shoulder
        wrist_rel_current = [
            wrist_current[0] - shoulder_current[0],  # x - horizontal sideways
            wrist_current[1] - shoulder_current[1],  # y - vertical
            wrist_current[2] - shoulder_current[2],  # z - depth
        ]

        wrist_rel_prev = [
            wrist_prev[0] - shoulder_prev[0],
            wrist_prev[1] - shoulder_prev[1],
            wrist_prev[2] - shoulder_prev[2],
        ]

        # Calculate change in relative position
        dx = wrist_rel_current[0] - wrist_rel_prev[0]  # Horizontal change
        dy = wrist_rel_current[1] - wrist_rel_prev[1]  # Vertical change
        dz = wrist_rel_prev[2] - wrist_rel_current[2]  # Forward change (inverted)

        # For a boxer stance, we should weigh the horizontal component more
        # depending on which way they're facing
        if orientation["side_facing"]:
            # In side stance, horizontal movement is more important
            # and the forward arm uses more x-movement while the back arm uses more z-movement
            if (orientation["left_forward"] and side == "left") or (
                not orientation["left_forward"] and side == "right"
            ):
                # Forward arm: emphasize x movement
                return abs(dx) * 1.5 + dz * self.direction_weight + abs(dy) * 0.3
            else:
                # Back arm: emphasize z movement
                return (
                    abs(dx) * 0.7 + dz * (self.direction_weight * 1.5) + abs(dy) * 0.3
                )
        else:
            # Front facing stance - more balanced
            return dz * self.direction_weight + math.sqrt(dx**2 + dy**2)

    def _is_punch_direction(self, current, previous, prev_prev, orientation, side):
        """
        Check if movement direction is consistent with punching in boxing stance.

        Args:
            current, previous, prev_prev: Landmark positions
            orientation: Body orientation information
            side: "left" or "right" to indicate which arm

        Returns:
            bool: True if direction matches punch motion
        """
        # Calculate differences in each dimension
        x_diff = current[0] - previous[0]
        y_diff = current[1] - previous[1]
        z_diff = previous[2] - current[2]  # Inverted because z decreases moving forward

        # For boxing stance, the direction depends on which arm and orientation
        if orientation["side_facing"]:
            # For side stance, the leading arm punches more horizontally
            # while the rear arm punches more forward
            if (orientation["left_forward"] and side == "left") or (
                not orientation["left_forward"] and side == "right"
            ):
                # Leading arm - should have significant horizontal component in correct direction
                correct_x_direction = (side == "left" and x_diff > 0.005) or (
                    side == "right" and x_diff < -0.005
                )
                return (
                    correct_x_direction and abs(y_diff) < 0.05
                )  # Not much vertical movement
            else:
                # Rear arm - should have forward component
                return (
                    z_diff > 0.005 and abs(y_diff) < 0.05
                )  # Moving forward, not much vertical
        else:
            # Front facing - more traditional forward punch
            return (
                z_diff > 0.005 and abs(y_diff) < 0.05
            )  # Moving forward, not much vertical

    def _is_boxing_extension(
        self,
        shoulder_current,
        elbow_current,
        wrist_current,
        shoulder_prev,
        elbow_prev,
        wrist_prev,
    ):
        """
        Check if the arm motion resembles a boxing punch extension.

        Boxing punches often don't fully extend the arm, so we're less strict.

        Returns:
            bool: True if extension matches boxing punch
        """
        # Calculate current and previous distances from shoulder to wrist
        current_dist = math.sqrt(
            (shoulder_current[0] - wrist_current[0]) ** 2
            + (shoulder_current[1] - wrist_current[1]) ** 2
            + (shoulder_current[2] - wrist_current[2]) ** 2
        )

        prev_dist = math.sqrt(
            (shoulder_prev[0] - wrist_prev[0]) ** 2
            + (shoulder_prev[1] - wrist_prev[1]) ** 2
            + (shoulder_prev[2] - wrist_prev[2]) ** 2
        )

        # For boxing, we just need some extension, not necessarily full
        return current_dist > prev_dist * 1.01  # Just 1% increase is enough

    def _calculate_relative_velocity(
        self, wrist_current, wrist_prev, shoulder_current, shoulder_prev
    ):
        """
        Calculate the velocity of wrist relative to shoulder to filter out body movement.

        Args:
            wrist_current: Current wrist position
            wrist_prev: Previous wrist position
            shoulder_current: Current shoulder position
            shoulder_prev: Previous shoulder position

        Returns:
            float: Relative velocity magnitude
        """
        # Calculate wrist movement relative to shoulder
        wrist_rel_current = [
            wrist_current[0] - shoulder_current[0],
            wrist_current[1] - shoulder_current[1],
            wrist_current[2] - shoulder_current[2],
        ]

        wrist_rel_prev = [
            wrist_prev[0] - shoulder_prev[0],
            wrist_prev[1] - shoulder_prev[1],
            wrist_prev[2] - shoulder_prev[2],
        ]

        # Calculate change in relative position
        dx = wrist_rel_current[0] - wrist_rel_prev[0]
        dy = wrist_rel_current[1] - wrist_rel_prev[1]
        dz = (
            wrist_rel_prev[2] - wrist_rel_current[2]
        )  # Inverted because z gets smaller as you move forward

        # Apply directional weight to emphasize forward motion
        return dz * self.direction_weight + math.sqrt(dx**2 + dy**2)

    def _is_forward_motion(self, current, previous, prev_prev):
        """
        Check if the motion is predominantly forward.

        Args:
            current: Current landmark position
            previous: Previous landmark position
            prev_prev: Position from two frames ago

        Returns:
            bool: True if motion is forward, False otherwise
        """
        # Z decreases as you move forward in camera space
        z_diff_current = previous[2] - current[2]
        z_diff_prev = prev_prev[2] - previous[2]

        # Consistent forward motion across frames
        return z_diff_current > 0.005  # Reduced threshold for detection

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
        return z_velocity * self.direction_weight + math.sqrt(
            x_velocity**2 + y_velocity**2
        )


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
