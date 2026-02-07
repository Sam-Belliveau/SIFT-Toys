"""
Video Capture
Uses: nothing
Used by: main.py

Async capture: background thread grabs frames so main loop never blocks.
"""

import cv2
import threading

# === PARAMETERS ===
DEVICE_INDEX = 0


class AsyncCamera:
    def __init__(self, device_index=DEVICE_INDEX):
        self.cap = cv2.VideoCapture(device_index)
        self.frame = None
        self.grabbed = False
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.running:
            grabbed, frame = self.cap.read()
            with self.lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.lock:
            return self.frame.copy() if self.grabbed else None

    def release(self):
        self.running = False
        self.thread.join()
        self.cap.release()


def get_stream():
    return AsyncCamera()


def get_frame(camera):
    return camera.read()


def release(camera):
    camera.release()
