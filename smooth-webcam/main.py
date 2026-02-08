"""
Smooth Webcam - Optical Flow Feature Tracking

Main orchestrator. Reads all tunable values from the global params registry.
"""

import time
import cv2
import numpy as np

from frontend import video_capture, interface
from processing import grayscale, image_warp, draw
from processing.feature_tracker import FeatureTracker
from profiler import profiler
from params import params

# === PARAMETERS ===
MIN_FEATURES = 4

COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_WHITE = (255, 255, 255)


def main():
    print("Running on CPU with async capture")

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
                return

        downsample = params["downsample"]

        with profiler.section("downsample"):
            h, w = frame.shape[:2]
            small_h, small_w = h // downsample, w // downsample
            small_frame = cv2.resize(frame, (small_w, small_h))

        with profiler.section("grayscale"):
            gray = grayscale.convert(small_frame)

        with profiler.section("tracking"):
            detected, tracked = tracker.update(
                gray,
                params["decay_iters"],
                params["grid_points"],
            )

        if detected.shape[0] < MIN_FEATURES:
            interface.show_frame(frame)
            interface.process_events()
            profiler.report()
            return

        with profiler.section("warp"):
            warped_small, flow_small = image_warp.warp_image(
                small_frame,
                detected,
                tracked,
            )

        with profiler.section("upsample"):
            warped = cv2.resize(warped_small, (w, h))

        with profiler.section("scale_points"):
            detected_full = detected * downsample
            tracked_full = tracked * downsample

        with profiler.section("draw"):
            output = draw.draw_points(
                warped,
                tracked_full,
                COLOR_GREEN,
                radius=1,
            )
            output = draw.draw_lines(
                warped,
                detected_full,
                tracked_full,
                COLOR_WHITE,
            )
            output = draw.draw_points(
                warped,
                detected_full,
                COLOR_RED,
                radius=3,
            )

        with profiler.section("flow_viz"):
            flow_y = flow_small[:, :, 0]
            flow_x = flow_small[:, :, 1]
            mag = np.sqrt(flow_x**2 + flow_y**2)
            ang = np.arctan2(flow_y, flow_x)

            hsv = np.zeros((*flow_small.shape[:2], 3), dtype=np.uint8)
            hsv[..., 0] = ((ang + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
            hsv[..., 1] = 255
            max_mag = max(mag.max(), 1e-5)
            hsv[..., 2] = np.clip(mag / max_mag * 255, 0, 255).astype(np.uint8)

            flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            flow_vis = cv2.resize(flow_bgr, (w, h), interpolation=cv2.INTER_NEAREST)

        with profiler.section("display"):
            gray_bgr = cv2.cvtColor(cv2.resize(gray, (w, h)), cv2.COLOR_GRAY2BGR)
            top = np.hstack([frame, output])
            bottom = np.hstack([gray_bgr, flow_vis])
            combined = np.vstack([top, bottom])
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
