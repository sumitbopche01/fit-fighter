"""
Common test fixtures for the FitFighter test suite.

This module provides fixtures that can be reused across different test files.
"""

import pytest
import numpy as np
from collections import deque
from fitfighter.constants import landmark_indices as lm
from fitfighter.testing import create_mock_pose, create_pose_sequence, POSES


@pytest.fixture
def sample_landmarks():
    """Generate a set of sample pose landmarks."""
    return create_mock_pose("standing")


@pytest.fixture
def landmark_history():
    """Generate a sample landmark history."""
    return [create_mock_pose("standing") for _ in range(10)]


@pytest.fixture
def standing_pose():
    """Create landmarks for a person standing with arms at sides."""
    return create_mock_pose("standing")


@pytest.fixture
def jumping_jack_closed_pose():
    """Create landmarks for a person in jumping jack closed position."""
    return create_mock_pose("jumping_jack_closed")


@pytest.fixture
def jumping_jack_open_pose():
    """Create landmarks for a person in jumping jack open position."""
    return create_mock_pose("jumping_jack_open")


@pytest.fixture
def create_landmark_frame():
    """Factory function to create a landmark frame with specified positions."""

    def _create_frame(positions):
        frame = {}
        for landmark_idx, position in positions.items():
            # Handle both tuples and lists with variable length
            if len(position) == 4:
                frame[landmark_idx] = position
            elif len(position) == 3:
                frame[landmark_idx] = (*position, 1.0)  # Add visibility
            else:
                frame[landmark_idx] = (*position, 0.0, 1.0)  # Add z and visibility
        return frame

    return _create_frame


@pytest.fixture
def create_landmark_history():
    """Factory function to create a landmark history from a list of frames."""

    def _create_history(frames):
        return list(frames)

    return _create_history


@pytest.fixture
def create_exercise_sequence():
    """Create a sequence of frames for a given exercise."""

    def _create_sequence(exercise_name, num_frames=30):
        return create_pose_sequence(exercise_name, frames=num_frames)

    return _create_sequence
