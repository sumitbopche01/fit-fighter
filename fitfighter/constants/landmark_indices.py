"""
MediaPipe pose landmark indices.

This module provides constants for the MediaPipe Pose landmark indices.
"""

# Face landmarks
NOSE = 0
LEFT_EYE_INNER = 1
LEFT_EYE = 2
LEFT_EYE_OUTER = 3
RIGHT_EYE_INNER = 4
RIGHT_EYE = 5
RIGHT_EYE_OUTER = 6
LEFT_EAR = 7
RIGHT_EAR = 8
MOUTH_LEFT = 9
MOUTH_RIGHT = 10

# Upper body landmarks
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_PINKY = 17
RIGHT_PINKY = 18
LEFT_INDEX = 19
RIGHT_INDEX = 20
LEFT_THUMB = 21
RIGHT_THUMB = 22

# Lower body landmarks
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32

# Landmark groups for convenience
SHOULDER_LANDMARKS = [LEFT_SHOULDER, RIGHT_SHOULDER]
HIP_LANDMARKS = [LEFT_HIP, RIGHT_HIP]
KNEE_LANDMARKS = [LEFT_KNEE, RIGHT_KNEE]
ANKLE_LANDMARKS = [LEFT_ANKLE, RIGHT_ANKLE]

# Upper body groups
ARM_LANDMARKS = [
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_ELBOW,
    RIGHT_ELBOW,
    LEFT_WRIST,
    RIGHT_WRIST,
]

# Lower body groups
LEG_LANDMARKS = [LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE]

# Landmark groups by side
LEFT_SIDE_LANDMARKS = [
    LEFT_SHOULDER,
    LEFT_ELBOW,
    LEFT_WRIST,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_ANKLE,
]

RIGHT_SIDE_LANDMARKS = [
    RIGHT_SHOULDER,
    RIGHT_ELBOW,
    RIGHT_WRIST,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_ANKLE,
]

# Dictionary for landmark indices by name
LANDMARK_INDICES = {
    "nose": NOSE,
    "left_eye_inner": LEFT_EYE_INNER,
    "left_eye": LEFT_EYE,
    "left_eye_outer": LEFT_EYE_OUTER,
    "right_eye_inner": RIGHT_EYE_INNER,
    "right_eye": RIGHT_EYE,
    "right_eye_outer": RIGHT_EYE_OUTER,
    "left_ear": LEFT_EAR,
    "right_ear": RIGHT_EAR,
    "mouth_left": MOUTH_LEFT,
    "mouth_right": MOUTH_RIGHT,
    "left_shoulder": LEFT_SHOULDER,
    "right_shoulder": RIGHT_SHOULDER,
    "left_elbow": LEFT_ELBOW,
    "right_elbow": RIGHT_ELBOW,
    "left_wrist": LEFT_WRIST,
    "right_wrist": RIGHT_WRIST,
    "left_pinky": LEFT_PINKY,
    "right_pinky": RIGHT_PINKY,
    "left_index": LEFT_INDEX,
    "right_index": RIGHT_INDEX,
    "left_thumb": LEFT_THUMB,
    "right_thumb": RIGHT_THUMB,
    "left_hip": LEFT_HIP,
    "right_hip": RIGHT_HIP,
    "left_knee": LEFT_KNEE,
    "right_knee": RIGHT_KNEE,
    "left_ankle": LEFT_ANKLE,
    "right_ankle": RIGHT_ANKLE,
    "left_heel": LEFT_HEEL,
    "right_heel": RIGHT_HEEL,
    "left_foot_index": LEFT_FOOT_INDEX,
    "right_foot_index": RIGHT_FOOT_INDEX,
}
