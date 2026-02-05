"""
Display Output
Uses: nothing
Used by: main.py
"""

import cv2

# === PARAMETERS ===
WINDOW_NAME = "RBF Warp"
EXIT_KEY = 27  # ESC


def show(image):
    """
    Display image in window.
    Returns True to continue, False to exit.
    """
    cv2.imshow(WINDOW_NAME, image)
    key = cv2.waitKey(1)
    return key != EXIT_KEY


def cleanup():
    """Destroy all windows."""
    cv2.destroyAllWindows()
