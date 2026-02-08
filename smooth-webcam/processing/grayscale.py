"""
Grayscale Conversion (CPU)
Uses: nothing
Used by: main.py

Simple grayscale conversion with local contrast normalization.
"""

import cv2
import numpy as np


def convert(image):
    """
    Convert BGR image to grayscale with local contrast normalization.

    Args:
        image: HxWx3 uint8 BGR numpy array

    Returns:
        HxW uint8 grayscale numpy array
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    ksize = (41, 41)  # must be odd, ~4*sigma+1 for sigma=10
    gray_blur = cv2.GaussianBlur(gray, ksize, sigmaX=10)
    gray_sq_blur = cv2.GaussianBlur(np.square(gray), ksize, sigmaX=10)

    variance = gray_sq_blur - np.square(gray_blur)
    std = np.sqrt(variance + 16)

    norm_gray = (gray - gray_blur) / std
    return np.clip(40 * norm_gray + 128, 0, 255).astype(np.uint8)
