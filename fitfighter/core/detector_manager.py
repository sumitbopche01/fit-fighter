"""
Exercise detector manager.

This module provides a manager class for handling multiple exercise detectors.
"""

from collections import deque
import numpy as np
import importlib
import logging

logger = logging.getLogger(__name__)

# Import will be done dynamically to allow customization of loaded detectors


class ExerciseDetectorManager:
    """Manager for multiple exercise detectors."""

    def __init__(
        self, history_size=30, confidence_threshold=0.6, detectors_to_load=None
    ):
        """
        Initialize the exercise detector manager.

        Args:
            history_size: Number of frames to keep in landmark history
            confidence_threshold: Minimum landmark visibility threshold
            detectors_to_load: List of detector class names to load (default: load all available)
        """
        self.history_size = history_size
        self.confidence_threshold = confidence_threshold

        # Store landmark history
        self.landmark_history = deque(maxlen=history_size)

        # Initialize detectors dict
        self.detectors = {}

        # Load all available detectors
        self._load_detectors(detectors_to_load)

        # Track active exercises and their counts
        self.active_exercises = set()
        self.exercise_counts = {name: 0 for name in self.detectors.keys()}

        # Stats for exercise session
        self.session_stats = {
            "total_reps": 0,
            "exercise_durations": {name: 0 for name in self.detectors.keys()},
            "last_active": None,
        }

    def _load_detectors(self, detectors_to_load=None):
        """
        Load the specified detector classes or all available detectors.

        Args:
            detectors_to_load: List of detector class names to load, or None to load defaults
        """
        # Define the default set of detectors if none specified
        if detectors_to_load is None:
            detectors_to_load = [
                "JumpingJackDetector",
                "PushupDetector",
                "LungeDetector",
                "ArmCirclesDetector",
                "PlankDetector",
                "SitupDetector",
                "SquatDetector",
                "BurpeeDetector",
            ]

        # Import and initialize each detector
        for detector_class_name in detectors_to_load:
            try:
                # Convert class name to module name (e.g., JumpingJackDetector -> jumping_jacks_detector)
                module_name = detector_class_name.replace("Detector", "").lower()
                if not module_name.endswith("s"):
                    module_name += "_detector"
                else:
                    module_name = module_name[:-1] + "_detector"

                # Import the module
                module = importlib.import_module(f"fitfighter.detectors.{module_name}")

                # Get the detector class
                detector_class = getattr(module, detector_class_name)

                # Initialize the detector
                detector_instance = detector_class(
                    self.confidence_threshold, self.history_size
                )

                # Add to detectors dict using normalized name
                detector_name = module_name.replace("_detector", "")
                self.detectors[detector_name] = detector_instance

                logger.info(f"Loaded detector: {detector_name}")
            except (ImportError, AttributeError) as e:
                logger.warning(f"Failed to load detector {detector_class_name}: {e}")

    def process_landmarks(self, landmarks):
        """
        Process new pose landmarks and detect exercises.

        Args:
            landmarks: Dictionary of pose landmarks from MediaPipe

        Returns:
            dict: Detection results with active exercises and counts
        """
        # Add current landmarks to history
        self.landmark_history.append(landmarks)

        if len(self.landmark_history) < 2:
            return self._get_current_state()

        # Check each detector
        for exercise_name, detector in self.detectors.items():
            is_detected = detector.detect(list(self.landmark_history))

            # Track active state changes
            if is_detected and exercise_name not in self.active_exercises:
                self.active_exercises.add(exercise_name)
                self.session_stats["last_active"] = exercise_name
            elif not is_detected and exercise_name in self.active_exercises:
                self.active_exercises.remove(exercise_name)

            # Update rep counts
            self.exercise_counts[exercise_name] = detector.rep_count

            # Update session duration stats if this exercise is active
            if is_detected:
                self.session_stats["exercise_durations"][exercise_name] += 1

        # Update total rep count across all exercises
        self.session_stats["total_reps"] = sum(self.exercise_counts.values())

        return self._get_current_state()

    def _get_current_state(self):
        """
        Get the current detection state.

        Returns:
            dict: Current state with active exercises and counts
        """
        return {
            "active_exercises": list(self.active_exercises),
            "counts": self.exercise_counts.copy(),
            "session_stats": self.session_stats.copy(),
        }

    def get_debug_info(self):
        """
        Get debug information from all detectors.

        Returns:
            dict: Debug information for each detector
        """
        debug_info = {}
        for name, detector in self.detectors.items():
            debug_info[name] = detector.get_debug_info()
        return debug_info

    def reset_session(self):
        """Reset all detectors and session statistics."""
        for detector in self.detectors.values():
            detector.reset()

        self.active_exercises.clear()
        self.exercise_counts = {name: 0 for name in self.detectors.keys()}
        self.session_stats = {
            "total_reps": 0,
            "exercise_durations": {name: 0 for name in self.detectors.keys()},
            "last_active": None,
        }

    def get_available_exercises(self):
        """
        Get list of available exercise detectors.

        Returns:
            list: Names of available exercise detectors
        """
        return list(self.detectors.keys())

    def add_detector(self, name, detector_instance):
        """
        Add a new detector to the manager.

        Args:
            name: Name to use for the detector
            detector_instance: Initialized detector instance
        """
        # Add the detector
        self.detectors[name] = detector_instance

        # Update tracking
        self.exercise_counts[name] = 0
        self.session_stats["exercise_durations"][name] = 0

        logger.info(f"Added detector: {name}")

    def remove_detector(self, name):
        """
        Remove a detector from the manager.

        Args:
            name: Name of the detector to remove

        Returns:
            bool: True if removed successfully, False if not found
        """
        if name in self.detectors:
            del self.detectors[name]

            # Update tracking
            if name in self.exercise_counts:
                del self.exercise_counts[name]

            if name in self.session_stats["exercise_durations"]:
                del self.session_stats["exercise_durations"][name]

            if name in self.active_exercises:
                self.active_exercises.remove(name)

            logger.info(f"Removed detector: {name}")
            return True

        logger.warning(f"Detector not found: {name}")
        return False
