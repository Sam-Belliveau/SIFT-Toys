"""
Video Capture
Uses: nothing
Used by: main.py
"""

import cv2

# === PARAMETERS ===
DEVICE_INDEX = 0


def get_stream():
    """Open webcam and return camera handle."""
    return cv2.VideoCapture(DEVICE_INDEX)


def get_frame(camera):
    """Grab a single frame from camera. Returns None if invalid."""
    valid, frame = camera.read()
    if not valid:
        return None
    return frame


def release(camera):
    """Release camera resources."""
    camera.release()
