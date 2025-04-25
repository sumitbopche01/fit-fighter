# FitFighter

A computer vision-based exercise detection and fitness tracking system.

## Features

- Real-time pose detection using MediaPipe
- Detection of multiple exercises:
  - Jumping Jacks
  - Push-ups
  - Squats
  - Lunges
  - Burpees
  - Sit-ups
  - Planks
  - Arm Circles
- Exercise repetition counting
- Performance metrics tracking
- WebSocket integration for external applications

## Project Structure

```
fitFighter/
├── docs/            # Documentation files
├── src/             # Source code
│   ├── core/        # Core components
│   ├── detectors/   # Exercise detectors
│   ├── utils/       # Utility functions
│   └── constants/   # Constant values
├── tests/           # Test files
└── requirements.txt # Dependencies
```

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

## Running the Application

To run the basic application:

```bash
python -m src.main
```

Optional parameters:

- `--camera`: Camera device ID (default: 0)
- `--width`: Camera frame width (default: 640)
- `--height`: Camera frame height (default: 480)
- `--model-complexity`: MediaPipe model complexity (0, 1, or 2) (default: 1)
- `--websocket-host`: WebSocket server host (default: 127.0.0.1)
- `--websocket-port`: WebSocket server port (default: 5678)

Example:

```bash
python -m src.main --camera 1 --width 1280 --height 720 --model-complexity 1
```

## Performance Considerations

- Model complexity 0: Fastest but less accurate
- Model complexity 1: Balanced performance and accuracy
- Model complexity 2: Most accurate but slower

Lower resolution improves performance but may reduce detection accuracy.
