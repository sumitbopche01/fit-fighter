"""
Tests for the jumping jacks detector.
"""

import pytest
import numpy as np
import sys
import os

# Add the parent directory to the path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fitfighter.detectors.jumping_jack_detector import JumpingJackDetector
from fitfighter.constants import landmark_indices as lm


def create_test_landmark(x, y, z=0.0, visibility=1.0):
    """Create a test landmark with the given coordinates and visibility."""
    return (x, y, z, visibility)


def create_landmark_frame(positions):
    """Create a landmark frame with the given positions."""
    frame = {}
    for landmark_idx, position in positions.items():
        frame[landmark_idx] = create_test_landmark(*position)
    return frame


def create_test_history(frames):
    """Create a test landmark history."""
    return frames


def test_jumping_jack_detector_initialization():
    """Test that the jumping jack detector initializes correctly."""
    detector = JumpingJackDetector()
    assert detector.name == "JumpingJack"
    assert detector.rep_count == 0
    assert detector.is_active is False
    assert detector.current_phase == "unknown"


def test_detect_with_insufficient_history():
    """Test that the detector returns False with insufficient history."""
    detector = JumpingJackDetector()
    history = []
    assert detector.detect(history) is False

    # Add just one frame
    frame = create_landmark_frame({})
    history = [frame]
    assert detector.detect(history) is False


def test_detect_with_missing_landmarks():
    """Test that the detector returns False when landmarks are missing."""
    detector = JumpingJackDetector()

    # Create a history with missing landmarks
    frame = create_landmark_frame(
        {
            lm.LEFT_SHOULDER: (0.4, 0.3, 0, 1.0),
            lm.RIGHT_SHOULDER: (0.6, 0.3, 0, 1.0),
            # Missing other required landmarks
        }
    )

    history = [frame, frame]  # Two identical frames
    assert detector.detect(history) is False


def test_jumping_jack_full_cycle():
    """Test a full jumping jack cycle (closed -> open -> closed)."""
    detector = JumpingJackDetector()

    # Create landmarks for closed position (arms down, legs together)
    closed_frame = create_landmark_frame(
        {
            lm.LEFT_SHOULDER: (0.4, 0.3, 0, 1.0),
            lm.RIGHT_SHOULDER: (0.6, 0.3, 0, 1.0),
            lm.LEFT_ELBOW: (0.35, 0.45, 0, 1.0),
            lm.RIGHT_ELBOW: (0.65, 0.45, 0, 1.0),
            lm.LEFT_WRIST: (0.35, 0.6, 0, 1.0),
            lm.RIGHT_WRIST: (0.65, 0.6, 0, 1.0),
            lm.LEFT_HIP: (0.45, 0.6, 0, 1.0),
            lm.RIGHT_HIP: (0.55, 0.6, 0, 1.0),
            lm.LEFT_KNEE: (0.45, 0.75, 0, 1.0),
            lm.RIGHT_KNEE: (0.55, 0.75, 0, 1.0),
            lm.LEFT_ANKLE: (0.45, 0.9, 0, 1.0),
            lm.RIGHT_ANKLE: (0.55, 0.9, 0, 1.0),
        }
    )

    # Create landmarks for open position (arms up, legs apart)
    open_frame = create_landmark_frame(
        {
            lm.LEFT_SHOULDER: (0.4, 0.3, 0, 1.0),
            lm.RIGHT_SHOULDER: (0.6, 0.3, 0, 1.0),
            lm.LEFT_ELBOW: (0.3, 0.2, 0, 1.0),
            lm.RIGHT_ELBOW: (0.7, 0.2, 0, 1.0),
            lm.LEFT_WRIST: (0.2, 0.1, 0, 1.0),
            lm.RIGHT_WRIST: (0.8, 0.1, 0, 1.0),
            lm.LEFT_HIP: (0.45, 0.6, 0, 1.0),
            lm.RIGHT_HIP: (0.55, 0.6, 0, 1.0),
            lm.LEFT_KNEE: (0.35, 0.75, 0, 1.0),
            lm.RIGHT_KNEE: (0.65, 0.75, 0, 1.0),
            lm.LEFT_ANKLE: (0.25, 0.9, 0, 1.0),
            lm.RIGHT_ANKLE: (0.75, 0.9, 0, 1.0),
        }
    )

    # Initial state: closed position
    history = [closed_frame, closed_frame, closed_frame]
    assert detector.detect(history) is False
    assert detector.current_phase == "closed"

    # Transition to open position
    history = [closed_frame, closed_frame, open_frame]
    assert detector.detect(history) is False
    assert detector.current_phase == "opening"

    # Fully open position
    history = [closed_frame, open_frame, open_frame]
    assert detector.detect(history) is False
    assert detector.current_phase == "open"

    # Start closing
    history = [open_frame, open_frame, closed_frame]
    assert detector.detect(history) is True  # Active during transition
    assert detector.current_phase == "closing"

    # Back to closed position
    history = [open_frame, closed_frame, closed_frame]
    assert detector.detect(history) is False  # Completed rep
    assert detector.current_phase == "closed"
    assert detector.rep_count == 1


def test_reset():
    """Test that the reset method resets all state variables."""
    detector = JumpingJackDetector()

    # Setup some state
    detector.rep_count = 5
    detector.is_active = True
    detector.current_phase = "open"
    detector.cooldown_counter = 3

    # Reset
    detector.reset()

    # Check all state is reset
    assert detector.rep_count == 0
    assert detector.is_active is False
    assert detector.current_phase == "unknown"
    assert detector.cooldown_counter == 0
    assert len(detector.position_history) == 0
