"""
Smooth Webcam - Image Warp with SIFT Feature Tracking

Main orchestrator that combines all modules.

Uses:
  - frontend/video_capture
  - frontend/interface
  - frontend/debug_overlay
  - frontend/display
  - sift/detector
  - processing/grayscale
  - processing/feature_tracker
  - processing/image_warp
"""

import time

from frontend import video_capture, interface, debug_overlay, display
from sift.detector import SIFTDetector
from processing import grayscale, image_warp
from processing.feature_tracker import FeatureTracker


def main():
    # Initialize stateful components
    camera = video_capture.get_stream()
    sift = SIFTDetector()
    tracker = FeatureTracker()

    interface.setup()

    last_time = time.time()

    try:
        while True:
            # Timing
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time

            # Capture
            frame = video_capture.get_frame(camera)
            if frame is None:
                break

            # Read UI params
            max_features, rc_ms = interface.read_params()

            # Extract features
            gray = grayscale.convert(frame)
            keypoints, descriptors = sift.detect(gray, max_features)

            if len(keypoints) < 4:
                display.show(frame)
                continue

            # Track features (matches internally, smooths positions)
            detected, tracked = tracker.update(keypoints, descriptors, dt, rc_ms)

            if len(detected) < 4:
                display.show(frame)
                continue

            # Warp image
            map_x, map_y = image_warp.generate_warp_map(
                frame.shape,
                source_points=detected,
                destination_points=tracked,
            )
            warped = image_warp.apply_warp(frame, map_x, map_y)

            # Draw debug overlay
            output = debug_overlay.draw(warped, detected, tracked)

            # Display
            if not display.show(output):
                break

    finally:
        video_capture.release(camera)
        display.cleanup()


if __name__ == "__main__":
    main()
