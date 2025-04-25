"""
Exercise detectors for the FitFighter system.

This module provides detectors for various exercises.
"""

# Import all detectors for convenience
try:
    from .jumping_jack_detector import JumpingJackDetector
except ImportError:
    pass

try:
    from .squat_detector import SquatDetector
except ImportError:
    pass

try:
    from .pushup_detector import PushupDetector
except ImportError:
    pass

try:
    from .plank_detector import PlankDetector
except ImportError:
    pass

try:
    from .arm_circles_detector import ArmCirclesDetector
except ImportError:
    pass

try:
    from .burpee_detector import BurpeeDetector
except ImportError:
    pass

try:
    from .lunge_detector import LungeDetector
except ImportError:
    pass

try:
    from .situp_detector import SitupDetector
except ImportError:
    pass

try:
    from .kick_detector import KickDetector
except ImportError:
    pass
