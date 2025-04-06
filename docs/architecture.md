# FitFighter Architecture

This document provides an overview of the FitFighter architecture, describing
how the various components interact.

## System Components

FitFighter consists of two main components:

1. **Motion Detection Module** (Python/MediaPipe)
   - Camera input processing
   - Pose detection and landmark extraction
   - Exercise recognition algorithms
   - Performance monitoring

2. **Game Engine** (Unity)
   - 3D environment and visualization
   - Game mechanics and scoring
   - User interface
   - Mobile deployment

## Architecture Overview

```
┌─────────────────────┐                  ┌────────────────────┐
│                     │                  │                    │
│   Motion Detection  │  Motion Data     │    Unity Game      │
│      (Python)       │ ───────────────► │     Engine         │
│                     │                  │                    │
└─────────────────────┘                  └────────────────────┘
        │                                        │
        │                                        │
        ▼                                        ▼
┌─────────────────────┐                  ┌────────────────────┐
│    Camera Input     │                  │  Game Mechanics    │
│                     │                  │                    │
└─────────────────────┘                  └────────────────────┘
```

## Data Flow

1. Camera captures video frames
2. Motion detection processes frames to detect pose landmarks
3. Exercise recognition algorithms analyze landmarks
4. Motion data is sent to Unity game engine
5. Unity visualizes the data and controls game mechanics
6. Game provides feedback to the user

## Communication Protocol

The communication between the Motion Detection module and Unity will use:

- WebSocket or UDP for real-time communication
- JSON format for data exchange
- Simple protocol with motion state updates:

```json
{
  "timestamp": 1234567890,
  "exercises": {
    "punch": true,
    "squat": false,
    "plank": false
  },
  "landmarks": [
    {"name": "left_wrist", "x": 0.5, "y": 0.6, "z": 0.1, "confidence": 0.9},
    {"name": "right_wrist", "x": 0.6, "y": 0.6, "z": 0.1, "confidence": 0.9},
    ...
  ],
  "performance": {
    "fps": 25.5,
    "latency_ms": 45
  }
}
```

## Module Dependencies

### Motion Detection

- MediaPipe
- OpenCV
- NumPy
- WebSockets

### Unity

- WebSockets package
- JSON utilities
- Physics system
- Mobile Input system

## Deployment Architecture

For the MVP, the motion detection will run on the same device as the Unity game
(mobile phone or tablet).

Future versions may explore:

- Cloud-based processing for improved performance
- Server-side motion analysis for multiplayer features
- Edge computing optimizations

## Performance Considerations

- Frame rate target: 30+ FPS for motion detection
- Latency target: <100ms from motion to game response
- Memory usage: <500MB on target devices
- CPU usage: <30% of available resources
