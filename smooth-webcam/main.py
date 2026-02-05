"""
Smooth Webcam - RBF Warp with SIFT Feature Tracking

Main orchestrator that combines all modules.

Uses:
  - frontend/video_capture
  - frontend/interface
  - frontend/debug_overlay
  - frontend/display
  - sift/detector
  - processing/grayscale
  - processing/optimal_transport
  - processing/lowpass_filter
  - processing/rbf_warp
"""

import time

from frontend import video_capture, interface, debug_overlay, display
from sift.detector import SIFTDetector
from processing import grayscale, optimal_transport, rbf_warp
from processing.lowpass_filter import LowPassFilter


def main():
    # Initialize stateful components
    camera = video_capture.get_stream()
    sift = SIFTDetector()
    smoother = LowPassFilter()

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
            keypoints = sift.detect(gray, max_features)

            if len(keypoints) < 4:
                display.show(frame)
                continue

            # Match to previous frame
            matched = optimal_transport.match_points(
                keypoints,
                smoother.get_smoothed(),
            )

            # Smooth keypoints
            smoothed = smoother.update(matched, dt, rc_ms)

            # Warp image
            map_x, map_y = rbf_warp.generate_warp_map(
                frame.shape,
                source_points=keypoints,
                destination_points=smoothed,
            )
            warped = rbf_warp.apply_warp(frame, map_x, map_y)

            # Draw debug overlay
            output = debug_overlay.draw(warped, keypoints, smoothed)

            # Display
            if not display.show(output):
                break

    finally:
        video_capture.release(camera)
        display.cleanup()


if __name__ == "__main__":
    main()
