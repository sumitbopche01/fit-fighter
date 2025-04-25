# FitFighter API Reference

This document provides detailed information about the main classes and methods
in the FitFighter library.

## Core Components

### ExerciseDetectorManager

The main class that coordinates all exercise detection.

```python
from fitfighter.core import ExerciseDetectorManager

# Initialize with default parameters
detector_manager = ExerciseDetectorManager()

# Initialize with custom parameters
detector_manager = ExerciseDetectorManager(
    history_size=45,          # Number of frames to keep in history
    confidence_threshold=0.7  # Minimum landmark visibility to consider valid
)
```

#### Methods

| Method                         | Description                                  |
| ------------------------------ | -------------------------------------------- |
| `process_landmarks(landmarks)` | Process pose landmarks for a single frame    |
| `get_active_exercises()`       | Get currently active exercises               |
| `get_rep_count(exercise_name)` | Get repetition count for a specific exercise |
| `get_session_stats()`          | Get statistics for the current session       |
| `get_available_exercises()`    | Get list of all available exercises          |
| `reset_session()`              | Reset all exercise counters and states       |
| `get_debug_info()`             | Get debug information from all detectors     |
| `add_detector(name, detector)` | Add a custom detector to the manager         |

### BaseExerciseDetector

Abstract base class that all exercise detectors inherit from.

```python
from fitfighter.core import BaseExerciseDetector

# This is an abstract class and should be subclassed, not instantiated directly
```

#### Methods to Implement in Subclasses

| Method                     | Description                        |
| -------------------------- | ---------------------------------- |
| `detect(landmark_history)` | Implement exercise detection logic |
| `reset()`                  | Reset detector state               |

#### Helper Methods Provided

| Method                                      | Description                             |
| ------------------------------------------- | --------------------------------------- |
| `are_landmarks_visible(landmarks, indices)` | Check if required landmarks are visible |
| `get_rep_count()`                           | Get current repetition count            |
| `get_debug_info()`                          | Get debug information dictionary        |

## Built-in Detectors

FitFighter comes with several built-in exercise detectors:

```python
from fitfighter.detectors import (
    JumpingJackDetector,
    PushupDetector,
    LungeDetector,
    ArmCirclesDetector,
    PlankDetector,
    SitupDetector
)
```

Each detector can be initialized with custom parameters:

```python
detector = JumpingJackDetector(
    confidence_threshold=0.7,  # Minimum landmark visibility
    history_size=30            # Number of frames to track
)
```

## Utility Functions

### Angle Calculation

```python
from fitfighter.utils import calculate_2d_angle, calculate_3d_angle

# Calculate 2D angle between three points (in the x-y plane)
angle_2d = calculate_2d_angle(point1, point2, point3)

# Calculate 3D angle between three points (in x-y-z space)
angle_3d = calculate_3d_angle(point1, point2, point3)
```

### Distance Calculation

```python
from fitfighter.utils import calculate_distance

# Calculate Euclidean distance between two points
distance = calculate_distance(point1, point2)
```

### Visualization

```python
from fitfighter.utils import draw_landmarks, visualize_angles

# Draw landmarks on an image
annotated_image = draw_landmarks(image, landmarks)

# Visualize joint angles on an image
annotated_image = visualize_angles(image, landmarks, angles_dict)
```

## Constants

### Landmark Indices

FitFighter provides convenient access to landmark indices compatible with
MediaPipe Pose:

```python
from fitfighter.constants import landmark_indices as lm

# Examples
nose = lm.NOSE
left_shoulder = lm.LEFT_SHOULDER
right_hip = lm.RIGHT_HIP
```

## Example Usage

```python
from fitfighter.core import ExerciseDetectorManager
import mediapipe as mp

# Setup MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize detector manager
detector_manager = ExerciseDetectorManager()

# Process video frames
for frame in video_stream:
    # Process with MediaPipe
    results = pose.process(frame)
    landmarks = results.pose_landmarks
    
    if landmarks:
        # Convert to numpy array if needed
        landmarks_array = [[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks.landmark]
        
        # Process landmarks with FitFighter
        detector_manager.process_landmarks(landmarks_array)
        
        # Get active exercises
        active = detector_manager.get_active_exercises()
        
        # Get repetition counts
        counts = {ex: detector_manager.get_rep_count(ex) for ex in active}
        
        # Display or process results
        print(f"Active exercises: {active}")
        print(f"Repetition counts: {counts}")
```

## Error Handling

Most FitFighter methods will handle invalid input gracefully. Common errors:

- `ValueError`: When invalid parameters are provided
- `KeyError`: When requesting an exercise that doesn't exist
- `TypeError`: When providing landmarks in an incorrect format

Always check that landmarks are in the correct format before processing.
