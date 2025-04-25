# Exercise Detector Development Guide

This guide explains how to create new exercise detectors for the FitFighter
system.

## Overview

The FitFighter system uses a modular architecture where each exercise type is
detected by a dedicated detector class. All detectors inherit from the
`BaseExerciseDetector` abstract base class, which provides common functionality
and a standard interface.

## Creating a New Detector

To create a new exercise detector:

1. Create a new file in the `fitfighter/detectors/` directory
2. Implement a class that inherits from `BaseExerciseDetector`
3. Add the detector to the `ExerciseDetectorManager` (or it will be loaded
   automatically)
4. Import your detector in `fitfighter/detectors/__init__.py` to make it
   accessible directly

### File Naming

Follow these naming conventions:

- File name: `exercise_name_detector.py` (lowercase, underscores)
- Class name: `ExerciseNameDetector` (CamelCase)

For example:

- `squat_detector.py` for the file
- `SquatDetector` for the class

### Class Implementation

Here's a template for a new exercise detector:

```python
"""
Exercise Name detector.

This module provides a detector for Exercise Name exercises.
"""

import numpy as np
from collections import deque
from fitfighter.core import BaseExerciseDetector
from fitfighter.constants import landmark_indices as lm
from fitfighter.utils import calculate_2d_angle, calculate_3d_angle, calculate_distance

class ExerciseNameDetector(BaseExerciseDetector):
    """Detector for Exercise Name exercise."""

    def __init__(self, confidence_threshold=0.6, history_size=30):
        """
        Initialize the Exercise Name detector.

        Args:
            confidence_threshold: Minimum landmark visibility to consider valid
            history_size: Number of frames to keep in history
        """
        super().__init__(confidence_threshold, history_size)
        
        # Define thresholds and parameters specific to this exercise
        self.parameter1 = value1
        self.parameter2 = value2
        
        # Setup state tracking variables
        self.debug_values = {}

    def detect(self, landmark_history):
        """
        Detect if the exercise is being performed.

        Args:
            landmark_history: History of pose landmarks

        Returns:
            bool: True if exercise is detected, False otherwise
        """
        if len(landmark_history) < 2:
            return False
            
        # Get the current frame landmarks
        current = landmark_history[-1]
        
        # Define required landmarks for this exercise
        required_landmarks = [
            lm.LANDMARK1, lm.LANDMARK2, ...
        ]
        
        # Check if required landmarks are visible
        if not self.are_landmarks_visible(current, required_landmarks):
            return False
            
        # Implement detection logic
        
        # Update rep count if a repetition is completed
        # self.rep_count += 1
        
        # Store debug values
        self.debug_values.update({
            "key1": value1,
            "key2": value2,
        })
        
        # Return whether exercise is currently being performed
        return is_detected

    def reset(self):
        """Reset the detector state."""
        super().reset()
        # Reset detector-specific state variables
```

## Adding to the Detector Registry

After creating your detector, add it to the `fitfighter/detectors/__init__.py`
file:

```python
try:
    from .jumping_jack_detector import JumpingJackDetector
    from .exercise_name_detector import ExerciseNameDetector  # Your new detector
except ImportError:
    pass
```

This makes your detector accessible through the package's import system.

## Detection Logic Considerations

When implementing the detection logic for a new exercise, consider:

1. **Exercise Phases**: Most exercises have distinct phases (e.g., up/down for
   squats)
2. **Landmark Selection**: Identify which landmarks are most relevant for
   detection
3. **Angle/Distance Metrics**: Determine which angles or distances best
   characterize the exercise
4. **Smoothing**: Use history and averaging to reduce false positives
5. **Thresholds**: Set appropriate thresholds for detection, possibly using a
   hysteresis approach

## Testing Your Detector

Create a test file in the `tests/` directory following the naming convention
`test_exercise_name_detector.py`. Implement test cases for:

1. Initialization
2. Detection with insufficient history
3. Detection with missing landmarks
4. Successful detection of the exercise
5. Reset functionality

Use the test utilities provided to create sample landmark data.

## Adding to Detector Manager

Once your detector is implemented and tested, you can add it to the system by:

1. Ensuring the file is in the correct location
2. The manager will auto-discover it if following the naming convention

Or, you can manually add it to a running system using:

```python
from fitfighter.detectors import ExerciseNameDetector  # If added to __init__.py
# OR
from fitfighter.detectors.exercise_name_detector import ExerciseNameDetector

detector_manager.add_detector("exercise_name", ExerciseNameDetector())
```

## Best Practices

1. **Documentation**: Document thresholds and detection logic thoroughly
2. **Configurability**: Make thresholds configurable via the constructor
3. **Debug Info**: Provide useful debug information for troubleshooting
4. **Robustness**: Handle edge cases and missing/noisy landmarks gracefully
5. **Performance**: Keep the detector computationally efficient
6. **Consistency**: Follow the existing code style and patterns
