"""
Camera utilities for FitFighter motion detection.

This module provides utilities for camera setup, frame processing, and display.
"""

import cv2
import numpy as np
import time


class CameraManager:
    """Manages camera input and frame processing."""

    def __init__(self, camera_id=0, width=640, height=480):
        """
        Initialize the camera manager.

        Args:
            camera_id: Camera device ID (default: 0 for primary camera)
            width: Frame width (default: 640)
            height: Frame height (default: 480)
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.cap = None
        self.fps_counter = FPSCounter()

    def start(self):
        """Start the camera capture."""
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera with ID {self.camera_id}")

        return self.cap.isOpened()

    def read_frame(self):
        """
        Read a frame from the camera.

        Returns:
            tuple: (success, frame)
        """
        if self.cap is None:
            raise RuntimeError("Camera not started. Call start() first.")

        success, frame = self.cap.read()
        self.fps_counter.update()

        return success, frame

    def get_fps(self):
        """Get the current FPS."""
        return self.fps_counter.get_fps()

    def add_fps_to_frame(self, frame):
        """
        Add FPS counter to the frame.

        Args:
            frame: The frame to add FPS counter to

        Returns:
            frame with FPS text overlay
        """
        fps = self.fps_counter.get_fps()
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        return frame

    def release(self):
        """Release the camera resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class FPSCounter:
    """FPS counter for performance monitoring."""

    def __init__(self, avg_frames=30):
        """
        Initialize the FPS counter.

        Args:
            avg_frames: Number of frames to average FPS over
        """
        self.frame_times = []
        self.avg_frames = avg_frames
        self.last_time = time.time()

    def update(self):
        """Update the FPS counter with a new frame."""
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time

        self.frame_times.append(dt)
        if len(self.frame_times) > self.avg_frames:
            self.frame_times.pop(0)

    def get_fps(self):
        """
        Calculate the current FPS.

        Returns:
            float: Current FPS
        """
        if not self.frame_times:
            return 0

        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0


def display_frame(frame, window_name="FitFighter"):
    """
    Display a frame in a window.

    Args:
        frame: The frame to display
        window_name: Name of the window
    """
    cv2.imshow(window_name, frame)


def preprocess_frame(frame, target_width=None, target_height=None):
    """
    Preprocess a frame for analysis.

    Args:
        frame: Input frame
        target_width: Target width for resizing (optional)
        target_height: Target height for resizing (optional)

    Returns:
        Preprocessed frame
    """
    # Convert to RGB (MediaPipe uses RGB, OpenCV uses BGR)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize if dimensions provided
    if target_width and target_height:
        rgb_frame = cv2.resize(rgb_frame, (target_width, target_height))

    return rgb_frame
