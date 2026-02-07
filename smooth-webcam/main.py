"""
Smooth Webcam - CPU-based SIFT Feature Tracking

Main orchestrator with downsampling for performance.
"""

import time
import cv2

from frontend import video_capture, interface, display
from sift.detector import ORBDetector
from processing import grayscale, image_warp, draw
from processing.feature_tracker import FeatureTracker
from profiler import profiler

# === PARAMETERS ===
MIN_FEATURES = 4
DOWNSAMPLE_FACTOR = 4  # Process at 1/4 resolution

# Colors in BGR
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_WHITE = (255, 255, 255)


def main():
    print(f"Running on CPU with {DOWNSAMPLE_FACTOR}x downsampling")
    print("Press 'q' or ESC to quit, or close the window")

    camera = video_capture.get_stream()
    orb = ORBDetector()
    tracker = FeatureTracker()

    interface.setup()

    last_time = time.time()

    try:
        while True:
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time

            with profiler.section("capture"):
                frame = video_capture.get_frame(camera)
                if frame is None:
                    break

            max_features, rc_ms = interface.read_params()

            # Downsample for faster processing
            with profiler.section("downsample"):
                h, w = frame.shape[:2]
                small_h, small_w = h // DOWNSAMPLE_FACTOR, w // DOWNSAMPLE_FACTOR
                small_frame = cv2.resize(frame, (small_w, small_h))

            with profiler.section("grayscale"):
                gray = grayscale.convert(small_frame)

            with profiler.section("sift"):
                keypoints, descriptors = orb.detect(gray, max_features)

            if keypoints.shape[0] < MIN_FEATURES:
                display.show(frame)
                profiler.report()
                continue

            with profiler.section("tracking"):
                detected, tracked = tracker.update(keypoints, descriptors, dt, rc_ms)

            if detected.shape[0] < MIN_FEATURES:
                display.show(frame)
                profiler.report()
                continue

            # Warp at small resolution
            with profiler.section("warp"):
                warped_small = image_warp.warp_image(small_frame, detected, tracked)

            # Upsample result
            with profiler.section("upsample"):
                warped = cv2.resize(warped_small, (w, h))

            # Scale points for drawing at full resolution
            with profiler.section("scale_points"):
                detected_full = detected * DOWNSAMPLE_FACTOR
                tracked_full = tracked * DOWNSAMPLE_FACTOR

            # Draw overlay
            with profiler.section("draw"):
                output = draw.draw_lines(
                    warped, detected_full, tracked_full, COLOR_WHITE
                )
                output = draw.draw_points(output, detected_full, COLOR_RED, radius=3)
                output = draw.draw_points(output, tracked_full, COLOR_GREEN, radius=3)

            # Resize for smaller display window
            with profiler.section("resize_output"):
                output_small = cv2.resize(output, (640, 480))

            with profiler.section("display"):
                if not display.show(output_small):
                    break

            profiler.report()

    finally:
        video_capture.release(camera)
        display.cleanup()


if __name__ == "__main__":
    main()
