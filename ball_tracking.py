"""Camera-only ball tracking and a rough line/length estimate.

Best-effort only: a single camera with no depth information can't
measure line/length precisely. These are coarse, frame-relative
estimates meant as a talking point, not a measurement.
"""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

DEFAULT_MIN_RADIUS = 4
DEFAULT_MAX_RADIUS = 25
MIN_TRACK_POINTS = 3
LINE_CENTER_TOLERANCE_PX = 15


@dataclass
class BallDetection:
    center_px: tuple
    radius_px: float


class BallDetector:
    """Common interface every ball detector implements."""

    def detect(self, prev_frame_bgr, curr_frame_bgr) -> Optional[BallDetection]:
        raise NotImplementedError


class MotionHoughBallDetector(BallDetector):
    """
    Frame-differencing to isolate what's moving, then Hough circle
    detection restricted to that motion mask - avoids picking up
    static round/bright objects in the background.
    """

    def __init__(self, min_radius=DEFAULT_MIN_RADIUS, max_radius=DEFAULT_MAX_RADIUS):
        self.min_radius = min_radius
        self.max_radius = max_radius

    def detect(self, prev_frame_bgr, curr_frame_bgr) -> Optional[BallDetection]:
        prev_gray = cv2.cvtColor(prev_frame_bgr, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame_bgr, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(prev_gray, curr_gray)
        diff = cv2.GaussianBlur(diff, (5, 5), 0)
        _, motion_mask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

        blurred = cv2.GaussianBlur(curr_gray, (7, 7), 0)
        masked = cv2.bitwise_and(blurred, blurred, mask=motion_mask)

        circles = cv2.HoughCircles(
            masked, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
            param1=100, param2=15,
            minRadius=self.min_radius, maxRadius=self.max_radius,
        )
        if circles is None:
            return None

        circles = np.round(circles[0]).astype(int)

        # Several circles can match on noise; keep the one with the most
        # motion-mask overlap - the real (moving) ball should win.
        best, best_score = None, -1
        for (x, y, r) in circles:
            x0, x1 = max(0, x - r), min(motion_mask.shape[1], x + r)
            y0, y1 = max(0, y - r), min(motion_mask.shape[0], y + r)
            score = int(motion_mask[y0:y1, x0:x1].sum())
            if score > best_score:
                best_score, best = score, (x, y, r)

        if best is None:
            return None
        x, y, r = best
        return BallDetection(center_px=(float(x), float(y)), radius_px=float(r))


def get_default_ball_detector() -> BallDetector:
    return MotionHoughBallDetector()


def estimate_line_and_length(ball_track, stump_x_px: Optional[float] = None, cm_per_px: Optional[float] = None):
    """
    Rough, camera-relative line/length estimate from a ball track.
    Off-side/leg-side isn't determined here (that needs the batter's
    handedness/stance, which this pipeline doesn't infer) - "line" is
    reported purely relative to the frame/stumps instead.
    """
    if not ball_track or len(ball_track) < MIN_TRACK_POINTS:
        return {
            "available": False,
            "note": "Ball wasn't tracked clearly enough (too few detections) to estimate line/length.",
        }

    sorted_track = sorted(ball_track, key=lambda item: item["timestamp_ms"])
    first_pt, last_pt = sorted_track[0], sorted_track[-1]

    result = {
        "available": True,
        "trajectory_px": [item["center_px"] for item in sorted_track],
        "line": "Unknown (no stump reference given)",
        "length": "Unknown",
        "note": "Approximate, monocular, no-depth estimate - not a measurement.",
    }

    if stump_x_px is not None:
        avg_x = sum(p["center_px"][0] for p in sorted_track) / len(sorted_track)
        offset = avg_x - stump_x_px
        if abs(offset) < LINE_CENTER_TOLERANCE_PX:
            result["line"] = "In line with stumps"
        elif offset > 0:
            result["line"] = "Right of stumps (frame-relative)"
        else:
            result["line"] = "Left of stumps (frame-relative)"

    time_span_s = max(0.001, (last_pt["timestamp_ms"] - first_pt["timestamp_ms"]) / 1000.0)
    descent_rate_px_s = (last_pt["center_px"][1] - first_pt["center_px"][1]) / time_span_s

    if descent_rate_px_s < 150:
        result["length"] = "Full / overpitched (rough estimate)"
    elif descent_rate_px_s < 400:
        result["length"] = "Good length (rough estimate)"
    else:
        result["length"] = "Short (rough estimate)"

    return result
