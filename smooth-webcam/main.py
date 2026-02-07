"""
Smooth Webcam - Optical Flow Feature Tracking

Main orchestrator with downsampling for performance.
"""

import time
import cv2
import numpy as np

from frontend import video_capture, interface
from processing import grayscale, image_warp, draw
from processing.feature_tracker import FeatureTracker
from profiler import profiler

# === PARAMETERS ===
MIN_FEATURES = 4
DOWNSAMPLE_FACTOR = 4

COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_WHITE = (255, 255, 255)


def main():
    print(f"Running on CPU with {DOWNSAMPLE_FACTOR}x downsampling")

    camera = video_capture.get_stream()
    tracker = FeatureTracker()

    app, window = interface.create()

    last_time = time.time()

    def tick():
        nonlocal last_time

        if not interface.is_running():
            app.quit()
            return

        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time

        with profiler.section("capture"):
            frame = video_capture.get_frame(camera)
            if frame is None:
                app.quit()
                return

        max_features, rc_ms = interface.read_params()

        with profiler.section("downsample"):
            h, w = frame.shape[:2]
            small_h, small_w = h // DOWNSAMPLE_FACTOR, w // DOWNSAMPLE_FACTOR
            small_frame = cv2.resize(frame, (small_w, small_h))

        with profiler.section("grayscale"):
            gray = grayscale.convert(small_frame)

        with profiler.section("tracking"):
            detected, tracked = tracker.update(
                gray,
                dt,
                rc_ms,
                max_features,
            )

        if detected.shape[0] < MIN_FEATURES:
            interface.show_frame(frame)
            interface.process_events()
            profiler.report()
            return

        with profiler.section("warp"):
            warped_small = image_warp.warp_image(
                small_frame,
                detected,
                tracked,
            )

        with profiler.section("upsample"):
            warped = cv2.resize(warped_small, (w, h))

        with profiler.section("scale_points"):
            detected_full = detected * DOWNSAMPLE_FACTOR
            tracked_full = tracked * DOWNSAMPLE_FACTOR

        with profiler.section("draw"):
            output = draw.draw_lines(
                warped,
                detected_full,
                tracked_full,
                COLOR_WHITE,
            )
            output = draw.draw_points(
                output,
                detected_full,
                COLOR_RED,
                radius=3,
            )
            output = draw.draw_points(
                output,
                tracked_full,
                COLOR_GREEN,
                radius=3,
            )

        with profiler.section("display"):
            combined = np.hstack([frame, output])
            interface.show_frame(combined)
            interface.process_events()

        profiler.report()

    from PyQt6.QtCore import QTimer

    timer = QTimer()
    timer.timeout.connect(tick)
    timer.start(0)

    try:
        app.exec()
    finally:
        video_capture.release(camera)


if __name__ == "__main__":
    main()
