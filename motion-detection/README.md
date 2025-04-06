# FitFighter Motion Detection

This module provides the motion detection functionality for FitFighter,
analyzing camera input to detect specific exercises.

## Features

- Real-time pose detection using MediaPipe
- Detection of three key exercises:
  - Punches
  - Squats
  - Plank positions
- Performance metrics tracking
- Visualization of detection results

## Components

- `camera_utils.py`: Camera input and frame processing utilities
- `pose_detector.py`: MediaPipe integration for pose detection
- `motion_analyzer.py`: Exercise detection algorithms
- `main.py`: Proof of concept application

## Setup

1. Create a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Proof of Concept

To run the basic proof of concept application:

```bash
python main.py
```

Optional parameters:

- `--camera`: Camera device ID (default: 0)
- `--width`: Camera frame width (default: 640)
- `--height`: Camera frame height (default: 480)
- `--model-complexity`: MediaPipe model complexity (0, 1, or 2) (default: 1)

Example:

```bash
python main.py --camera 1 --width 1280 --height 720 --model-complexity 1
```

## Performance Considerations

- Model complexity 0: Fastest but less accurate
- Model complexity 1: Balanced performance and accuracy
- Model complexity 2: Most accurate but slower

Lower resolution improves performance but may reduce detection accuracy.
