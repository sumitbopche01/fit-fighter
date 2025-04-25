"""
Tests for the mock pose utilities.
"""

import unittest
import numpy as np
from fitfighter.testing import create_mock_pose, create_pose_sequence


class TestMockUtilities(unittest.TestCase):
    """Test the mock pose utilities."""

    def test_create_mock_pose(self):
        """Test that mock poses can be created."""
        # Test with predefined pose
        pose = create_mock_pose("standing")
        self.assertIsNotNone(pose)
        self.assertEqual(len(pose), 33)  # Should have 33 landmarks

        # Test with custom arm angles
        pose = create_mock_pose(arm_angles=(45, 135))
        self.assertIsNotNone(pose)
        self.assertEqual(len(pose), 33)

        # Test with custom leg angles
        pose = create_mock_pose(leg_angles=(30, 150))
        self.assertIsNotNone(pose)
        self.assertEqual(len(pose), 33)

        # Test with custom landmarks
        pose = create_mock_pose(custom_landmarks={0: (0.1, 0.2, 0.3, 1.0)})
        self.assertIsNotNone(pose)
        self.assertEqual(len(pose), 33)
        self.assertEqual(pose[0], (0.1, 0.2, 0.3, 1.0))

    def test_create_pose_sequence(self):
        """Test that pose sequences can be created."""
        # Test jumping jack sequence
        sequence = create_pose_sequence("jumping_jack", frames=30)
        self.assertIsNotNone(sequence)
        self.assertEqual(len(sequence), 30)  # Should have 30 frames

        # Test pushup sequence
        sequence = create_pose_sequence("pushup", frames=30)
        self.assertIsNotNone(sequence)
        self.assertEqual(len(sequence), 30)

        # Test squat sequence
        sequence = create_pose_sequence("squat", frames=30)
        self.assertIsNotNone(sequence)
        self.assertEqual(len(sequence), 30)

        # Test default sequence
        sequence = create_pose_sequence("unknown", frames=15)
        self.assertIsNotNone(sequence)
        self.assertEqual(len(sequence), 15)


if __name__ == "__main__":
    unittest.main()
