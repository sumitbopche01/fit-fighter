# Usage Guide

This guide provides examples and instructions for using the FitFighter library
to detect exercises in real-time or from pre-recorded videos.

## Basic Usage

### Setting Up the Detector Manager

The core component of FitFighter is the `ExerciseDetectorManager`, which manages
multiple exercise detectors:

```python
from fitfighter import ExerciseDetectorManager
from fitfighter.detectors import JumpingJackDetector, PushupDetector

# Create detector manager with default settings
detector_manager = ExerciseDetectorManager(
    history_size=30,  # Number of frames to keep in history
    confidence_threshold=0.6  # Minimum confidence to consider a detection valid
)

# Add specific detectors you want to use
detector_manager.add_detector(JumpingJackDetector())
detector_manager.add_detector(PushupDetector())
```

### Processing a Single Frame

To process a single frame with landmarks:

```python
# Assuming landmarks is a list of (x, y, z, visibility) tuples from MediaPipe
results = detector_manager.process_landmarks(landmarks)

# Get currently active exercises
active_exercises = detector_manager.get_active_exercises()

# Get repetition counts
counts = detector_manager.get_counts()
print(f"Repetition counts: {counts}")
```

### Real-time Detection with Webcam

Here's a complete example for real-time exercise detection using a webcam:

```python
import cv2
import mediapipe as mp
from fitfighter import ExerciseDetectorManager
from fitfighter.detectors import JumpingJackDetector, PushupDetector

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize detector manager
detector_manager = ExerciseDetectorManager()
detector_manager.add_detector(JumpingJackDetector())
detector_manager.add_detector(PushupDetector())

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break
        
    # Convert the image to RGB and process with MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if results.pose_landmarks:
        # Extract landmarks
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmarks.append((landmark.x, landmark.y, landmark.z, landmark.visibility))
        
        # Process landmarks with detector manager
        detector_manager.process_landmarks(landmarks)
        
        # Draw pose landmarks on the image
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Display active exercises and counts
        active = detector_manager.get_active_exercises()
        counts = detector_manager.get_counts()
        
        y_position = 30
        for exercise, count in counts.items():
            status = "ACTIVE" if exercise in active else "INACTIVE"
            cv2.putText(image, f"{exercise}: {count} reps ({status})", 
                       (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_position += 30
    
    # Display the image
    cv2.imshow('FitFighter Exercise Detection', image)
    
    # Exit on 'q' press
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Advanced Usage

### Customizing Detectors

You can customize detectors by passing parameters during initialization:

```python
from fitfighter.detectors import PushupDetector

# Create a push-up detector with custom parameters
pushup_detector = PushupDetector(
    angle_threshold=40.0,  # Customize the angle threshold
    confidence_threshold=0.7  # Higher confidence for detection
)
```

### Processing Video Files

To process a pre-recorded video file:

```python
import cv2
import mediapipe as mp
from fitfighter import ExerciseDetectorManager
from fitfighter.detectors import LungeDetector

# Initialize MediaPipe and detector
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
detector_manager = ExerciseDetectorManager()
detector_manager.add_detector(LungeDetector())

# Open video file
video_path = "workout_video.mp4"
cap = cv2.VideoCapture(video_path)

# Process each frame
while cap.isOpened():
    success, image = cap.read()
    if not success:
        break
        
    # Process with MediaPipe and FitFighter
    # (same code as in the webcam example)
    
    # Optional: Save the processed frames to a new video
    # out.write(image)

cap.release()
```

### Retrieving Session Statistics

After processing a workout session, you can retrieve statistics:

```python
# Get session stats
session_stats = detector_manager.get_session_stats()

# Print detailed statistics
for exercise, stats in session_stats.items():
    print(f"\n{exercise} statistics:")
    print(f"  Total repetitions: {stats['count']}")
    print(f"  Average confidence: {stats['avg_confidence']:.2f}")
    print(f"  Average duration per rep: {stats['avg_duration']:.2f} seconds")
```

### Debugging and Visualization

For debugging purposes, you can get additional information:

```python
# Get debug information
debug_info = detector_manager.get_debug_info()

# Visualize angles and positions
from fitfighter.utils import visualize_landmarks, visualize_angle

# Assuming 'image' is your frame and 'landmarks' are pose landmarks
visualize_landmarks(image, landmarks)
visualize_angle(image, landmarks, point1_idx=11, point2_idx=13, point3_idx=15, color=(0, 255, 0))
```

## Working with Specific Exercises

### Jumping Jacks

```python
from fitfighter.detectors import JumpingJackDetector

jumping_jack_detector = JumpingJackDetector()
detector_manager.add_detector(jumping_jack_detector)
```

### Push-ups

```python
from fitfighter.detectors import PushupDetector

pushup_detector = PushupDetector()
detector_manager.add_detector(pushup_detector)
```

### Lunges

```python
from fitfighter.detectors import LungeDetector

lunge_detector = LungeDetector()
detector_manager.add_detector(lunge_detector)
```

## Error Handling

Always handle potential errors in your application:

```python
try:
    detector_manager.process_landmarks(landmarks)
except ValueError as e:
    print(f"Error processing landmarks: {e}")
except KeyError as e:
    print(f"Missing landmark index: {e}")
```

## Performance Tips

- To improve performance, consider reducing the webcam resolution or processing
  every 2-3 frames instead of every frame
- For deployment on edge devices, use a smaller history size for the detector
  manager
- Close the webcam and release resources when done with `cap.release()` and
  `cv2.destroyAllWindows()`
