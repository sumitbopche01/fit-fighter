# FitFighter

A real-time exercise detection system for fitness applications, leveraging
computer vision and pose estimation.

## Overview

FitFighter is a Python library that provides accurate detection and counting of
various exercises using pose estimation. The system can:

- Detect multiple exercise types simultaneously
- Count repetitions
- Track exercise session stats
- Provide debugging visualizations

## Features

- Support for multiple exercises:
  - Jumping Jacks
  - Push-ups
  - Lunges
  - Arm Circles
  - Planks
  - Sit-ups
  - Squats
  - Burpees

- Modular architecture:
  - Easy to add new exercise detectors
  - Configurable detection parameters
  - Utilities for landmark processing and angle calculations

- Visualization tools:
  - Real-time pose detection overlay
  - Exercise stats visualization
  - Debug mode for development

## Documentation

- [Installation Guide](docs/installation.md): How to install FitFighter and its
  dependencies
- [Usage Guide](docs/usage.md): Examples and instructions for using FitFighter
- [API Reference](docs/api_reference.md): Detailed information about classes and
  methods
- [Exercise Detector Development Guide](docs/detector_guide.md): How to create
  custom exercise detectors
- [Testing Guide](docs/testing.md): How to test FitFighter components

## Installation

### Option 1: Install from source

```bash
# Clone the repository
git clone https://github.com/yourusername/fitfighter.git
cd fitfighter

# Install the package
pip install -e .
```

### Option 2: Install dependencies only

```bash
# Install required dependencies
pip install -r requirements.txt
```

## Quick Start

```python
import cv2
import mediapipe as mp
from fitfighter.core import ExerciseDetectorManager
from fitfighter.utils import convert_mediapipe_landmarks, draw_detection_results

# Initialize MediaPipe pose detector
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# Initialize ExerciseDetectorManager
detector_manager = ExerciseDetectorManager()

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    # Convert image to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process frame with MediaPipe
    results = pose.process(rgb_frame)
    
    if results.pose_landmarks:
        # Convert landmarks to our format
        landmarks = convert_mediapipe_landmarks(results.pose_landmarks)
        
        # Process landmarks with detector manager
        detection_results = detector_manager.process_landmarks(landmarks)
        
        # Draw detection results on frame
        frame = draw_detection_results(frame, detection_results)
    
    # Show frame
    cv2.imshow('FitFighter', frame)
    
    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
pose.close()
```

## Examples

Check out the `examples` directory for more detailed examples:

- `webcam_demo.py`: A fully featured webcam-based exercise detection demo
  ```bash
  python examples/webcam_demo.py --debug
  ```

## Project Structure

```
fitfighter/
├── core/              # Core functionality
│   ├── __init__.py              # Exports core components
│   ├── base_detector.py         # Base class for all detectors 
│   └── detector_manager.py      # Manager for multiple detectors
├── detectors/         # Exercise detectors
│   ├── __init__.py              # Exports available detectors
│   ├── jumping_jack_detector.py # Jumping jack detector
│   ├── ...                      # Other exercise detectors
├── utils/             # Utility functions
│   ├── __init__.py              # Exports utility functions
│   ├── angle_calculator.py      # Angle calculation utilities
│   ├── pose_processor.py        # Pose landmark processing
│   └── visualization.py         # Visualization utilities
├── constants/         # Constants and configuration
│   ├── __init__.py              # Exports constants
│   └── landmark_indices.py      # MediaPipe landmark indices
├── testing/           # Testing utilities
│   └── __init__.py              # Mock data generation for tests
└── __init__.py        # Main package initialization
```

## Creating a Custom Exercise Detector

To add a new exercise detector, create a new file in the `detectors` directory
and implement a class that inherits from `BaseExerciseDetector`:

```python
from fitfighter.core import BaseExerciseDetector
from fitfighter.constants import landmark_indices as lm

class MyNewExerciseDetector(BaseExerciseDetector):
    def __init__(self, confidence_threshold=0.6, history_size=30):
        super().__init__(confidence_threshold, history_size)
        self.name = "MyNewExercise"
        # Initialize detector-specific properties
        
    def detect(self, landmark_history):
        # Implement detection logic
        # Return True if exercise is detected, False otherwise
        # Update self.rep_count when a repetition is completed
        return is_detected
```

See the [Exercise Detector Development Guide](docs/detector_guide.md) for more
detailed instructions.

## Requirements

- Python 3.7+
- NumPy
- OpenCV
- MediaPipe

## License

This project is licensed under the MIT License - see the LICENSE file for
details.

## Acknowledgments

- [MediaPipe](https://google.github.io/mediapipe/) for the pose estimation model
- [OpenCV](https://opencv.org/) for computer vision utilities
