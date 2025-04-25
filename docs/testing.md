# Testing Guide

This guide explains how to test components of the FitFighter library and write
new tests for custom detectors.

## Test Structure

The FitFighter test suite is organized to test each component of the library:

```
tests/
├── test_base_detector.py      # Tests for the base detector functionality
├── test_detector_manager.py   # Tests for the detector manager
├── test_jumping_jack_detector.py  # Tests for specific detectors
├── test_utils/               # Tests for utility functions
│   ├── test_angle_calculator.py
│   ├── test_pose_processor.py
│   └── test_visualization.py
└── conftest.py               # Common test fixtures
```

## Running Tests

To run the entire test suite:

```bash
# From the project root
pytest

# With coverage report
pytest --cov=fitfighter
```

To run specific test files:

```bash
# Test a specific detector
pytest tests/test_jumping_jack_detector.py

# Test a specific module
pytest tests/test_utils/
```

## Test Fixtures

The `conftest.py` file contains common fixtures used across tests:

```python
import pytest
import numpy as np
from fitfighter.constants import landmark_indices as lm

@pytest.fixture
def sample_landmarks():
    """Generate a set of sample pose landmarks."""
    landmarks = {}
    for i in range(33):  # MediaPipe has 33 pose landmarks
        landmarks[i] = (0.5, 0.5, 0.0, 1.0)  # x, y, z, visibility
    return landmarks

@pytest.fixture
def landmark_history():
    """Generate a sample landmark history."""
    history = []
    for i in range(10):
        landmarks = {}
        for j in range(33):
            landmarks[j] = (0.5, 0.5, 0.0, 1.0)
        history.append(landmarks)
    return history
```

## Writing Tests for Detectors

When creating a new exercise detector, follow this template for writing tests:

```python
import pytest
import numpy as np
from fitfighter.detectors.your_detector import YourDetector
from fitfighter.constants import landmark_indices as lm

# Helper functions for creating test data
def create_test_landmark(x, y, z=0.0, visibility=1.0):
    """Create a test landmark with given coordinates and visibility."""
    return (x, y, z, visibility)

def create_landmark_frame(positions):
    """Create a landmark frame with given positions."""
    frame = {}
    for landmark_idx, position in positions.items():
        frame[landmark_idx] = create_test_landmark(*position)
    return frame

# Tests
def test_detector_initialization():
    """Test that the detector initializes correctly."""
    detector = YourDetector()
    assert detector.name == "YourExercise"
    assert detector.rep_count == 0
    assert detector.is_active is False

def test_detect_with_insufficient_history():
    """Test behavior with insufficient history."""
    detector = YourDetector()
    assert detector.detect([]) is False
    
def test_detect_with_missing_landmarks():
    """Test behavior with missing landmarks."""
    detector = YourDetector()
    frame = create_landmark_frame({
        # Include only some landmarks
        lm.NOSE: (0.5, 0.5, 0, 1.0),
    })
    assert detector.detect([frame, frame]) is False
    
def test_exercise_detection():
    """Test complete exercise detection."""
    detector = YourDetector()
    
    # Create landmark frames representing exercise phases
    start_frame = create_landmark_frame({
        # Add landmarks for starting position
    })
    
    mid_frame = create_landmark_frame({
        # Add landmarks for middle position
    })
    
    end_frame = create_landmark_frame({
        # Add landmarks for ending position
    })
    
    # Test exercise cycle
    # For example: not detected in start position
    assert detector.detect([start_frame, start_frame]) is False
    
    # Detected during motion
    assert detector.detect([start_frame, mid_frame]) is True
    
    # Count increments after completing a rep
    assert detector.detect([mid_frame, end_frame]) is False
    assert detector.rep_count == 1
    
def test_reset():
    """Test that reset properly clears the state."""
    detector = YourDetector()
    
    # Set up some state
    detector.rep_count = 5
    detector.is_active = True
    
    # Reset
    detector.reset()
    
    # Verify state is reset
    assert detector.rep_count == 0
    assert detector.is_active is False
```

## Mock Pose Data

For testing with realistic data, use the `create_mock_pose` utility:

```python
from fitfighter.testing import create_mock_pose

# Create mock landmarks for a standing pose
standing_pose = create_mock_pose("standing")

# Create mock landmarks for a jumping jack with arms up
jumping_jack_pose = create_mock_pose("jumping_jack_up")

# Create custom pose with specific joint angles
custom_pose = create_mock_pose(
    arm_angles=(90, 45),  # Left and right arm angles in degrees
    leg_angles=(180, 180)  # Left and right leg angles in degrees
)
```

## Integration Tests

To test the entire pipeline, create integration tests that simulate a sequence
of frames:

```python
from fitfighter.core import ExerciseDetectorManager
from fitfighter.detectors import JumpingJackDetector
from fitfighter.testing import create_pose_sequence

def test_jumping_jack_integration():
    # Create detector manager
    detector_manager = ExerciseDetectorManager()
    detector_manager.add_detector("jumping_jack", JumpingJackDetector())
    
    # Create a sequence of poses (0-10% of motion, 11-20%, etc.)
    sequence = create_pose_sequence("jumping_jack", frames=20)
    
    # Process sequence
    for landmarks in sequence:
        detector_manager.process_landmarks(landmarks)
    
    # Check if correct number of reps was counted
    assert detector_manager.get_counts()["jumping_jack"] == 1
```

## Testing Visualization

To test visualization functions, use OpenCV to write images and verify them
manually:

```python
import os
import cv2
import numpy as np
from fitfighter.utils import draw_pose_landmarks, draw_detection_results

def test_visualization(tmpdir):
    # Create a blank image
    image = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # Create sample landmarks
    landmarks = {
        # Add landmarks here
    }
    
    # Draw landmarks
    annotated = draw_pose_landmarks(image, landmarks)
    
    # Save for visual inspection
    output_path = os.path.join(tmpdir, "test_visualization.jpg")
    cv2.imwrite(output_path, annotated)
    
    # For automated testing, check image properties
    assert annotated.shape == image.shape
    assert not np.array_equal(annotated, image)  # Should have changed
```

## Continuous Integration

The FitFighter project uses GitHub Actions for CI/CD. The workflow runs all
tests on multiple Python versions to ensure compatibility.

To see the current CI status, visit the Actions tab in the GitHub repository.

## Test Coverage

Maintain high test coverage by following these guidelines:

1. Every detector should have its own test file
2. All utility functions should have tests
3. Test both normal operation and edge cases
4. Test with invalid inputs to ensure proper error handling

Check coverage with:

```bash
pytest --cov=fitfighter --cov-report=html
```

Then open `htmlcov/index.html` to view the coverage report.
