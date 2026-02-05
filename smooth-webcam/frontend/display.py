"""
Display Output
Uses: nothing
Used by: main.py
"""

import cv2

# === PARAMETERS ===
WINDOW_NAME = "RBF Warp"
EXIT_KEY_ESC = 27
EXIT_KEY_Q = ord("q")


def show(image):
    """
    Display image in window.
    Returns True to continue, False to exit (ESC, 'q', or window closed).
    """
    cv2.imshow(WINDOW_NAME, image)
    key = cv2.waitKey(1) & 0xFF

    # Check for quit keys
    if key == EXIT_KEY_ESC or key == EXIT_KEY_Q:
        return False

    # Check if window was closed
    try:
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            return False
    except cv2.error:
        return False

    return True


def cleanup():
    """Destroy all windows."""
    cv2.destroyAllWindows()
