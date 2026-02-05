"""
Grayscale Conversion (CPU)
Uses: nothing
Used by: main.py

Simple grayscale conversion using OpenCV.
"""

import cv2


def convert(image):
    """
    Convert BGR image to grayscale.

    Args:
        image: HxWx3 uint8 BGR numpy array

    Returns:
        HxW uint8 grayscale numpy array
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
