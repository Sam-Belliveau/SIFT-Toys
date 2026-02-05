"""
Grayscale Conversion
Uses: nothing
Used by: main.py
"""

import cv2


def convert(image):
    """Convert BGR image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
