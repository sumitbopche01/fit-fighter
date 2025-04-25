"""
Tests for the Exercise Detector Manager.
"""

import unittest
from fitfighter.core.detector_manager import ExerciseDetectorManager


class TestDetectorManager(unittest.TestCase):
    """Test the ExerciseDetectorManager class."""

    def setUp(self):
        """Set up the test."""
        self.detector_manager = ExerciseDetectorManager()

    def test_initialization(self):
        """Test that the detector manager initializes correctly."""
        self.assertIsNotNone(self.detector_manager)
        self.assertIsInstance(self.detector_manager, ExerciseDetectorManager)

        # Check that the detector manager has loaded detectors
        self.assertGreater(len(self.detector_manager.detectors), 0)

        # Check that the detector manager has session stats
        self.assertIn("total_reps", self.detector_manager.session_stats)
        self.assertIn("exercise_durations", self.detector_manager.session_stats)

    def test_reset_session(self):
        """Test that the session can be reset."""
        # Reset the session
        self.detector_manager.reset_session()

        # Check that all exercise counts are 0
        for count in self.detector_manager.exercise_counts.values():
            self.assertEqual(count, 0)

        # Check that total reps is 0
        self.assertEqual(self.detector_manager.session_stats["total_reps"], 0)

        # Check that active exercises is empty
        self.assertEqual(len(self.detector_manager.active_exercises), 0)

    def test_get_available_exercises(self):
        """Test that the available exercises can be retrieved."""
        exercises = self.detector_manager.get_available_exercises()

        # Check that there are exercises
        self.assertGreater(len(exercises), 0)

        # Check that the exercises are strings
        for exercise in exercises:
            self.assertIsInstance(exercise, str)


if __name__ == "__main__":
    unittest.main()
