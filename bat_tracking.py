"""Camera-only bat tracking: detectors (ArUco marker, color tape, and a
YOLOv8 stub for later) plus swing metrics derived from the resulting
track.

No sensors involved - the user attaches a printed ArUco marker or a
strip of bright tape to the bat.
"""

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

import pose_utils

# ---- Color-tape defaults (tune per lighting / tape color) ----
# OpenCV HSV ranges: H in [0,179], S/V in [0,255]. Defaults target bright
# green tape; for a different color, sample a few pixels of your tape
# under your actual lighting and widen the range around their HSV values.
DEFAULT_HSV_LOWER = (40, 80, 80)
DEFAULT_HSV_UPPER = (85, 255, 255)
MIN_CONTOUR_AREA = 150

MIN_TRACK_POINTS = 2


@dataclass
class BatDetection:
    center_px: tuple
    angle_deg: Optional[float]
    confidence: float


class BatDetector:
    """Common interface every bat detector implements."""

    def detect(self, frame_bgr) -> Optional[BatDetection]:
        raise NotImplementedError


class ArucoBatDetector(BatDetector):
    """Primary detector: a printed ArUco marker taped to the bat."""

    def __init__(self, dictionary=cv2.aruco.DICT_4X4_50, marker_id: Optional[int] = None):
        aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
        params = cv2.aruco.DetectorParameters()
        self._detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        self.marker_id = marker_id

    def detect(self, frame_bgr) -> Optional[BatDetection]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)
        if ids is None or len(corners) == 0:
            return None

        idx = 0
        if self.marker_id is not None:
            matches = [i for i, mid in enumerate(ids.flatten()) if mid == self.marker_id]
            if not matches:
                return None
            idx = matches[0]

        pts = corners[idx][0]  # 4x2: top-left, top-right, bottom-right, bottom-left
        center = (float(pts[:, 0].mean()), float(pts[:, 1].mean()))
        top_edge = pts[1] - pts[0]
        angle_deg = float(np.degrees(np.arctan2(top_edge[1], top_edge[0])))
        return BatDetection(center_px=center, angle_deg=angle_deg, confidence=1.0)


class ColorMarkerBatDetector(BatDetector):
    """Fallback detector: a strip of bright tape on the bat."""

    def __init__(self, hsv_lower=DEFAULT_HSV_LOWER, hsv_upper=DEFAULT_HSV_UPPER, min_area=MIN_CONTOUR_AREA):
        self.hsv_lower = np.array(hsv_lower, dtype=np.uint8)
        self.hsv_upper = np.array(hsv_upper, dtype=np.uint8)
        self.min_area = min_area

    def detect(self, frame_bgr) -> Optional[BatDetection]:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < self.min_area:
            return None

        moments = cv2.moments(largest)
        if moments["m00"] == 0:
            return None
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]

        rect = cv2.minAreaRect(largest)
        angle_deg = float(rect[-1])

        frame_area = frame_bgr.shape[0] * frame_bgr.shape[1]
        confidence = min(1.0, (area / max(1, frame_area)) * 20)  # rough, unlike ArUco's exact 1.0

        return BatDetection(center_px=(cx, cy), angle_deg=angle_deg, confidence=confidence)


class YoloBatDetector(BatDetector):
    """
    Placeholder for a trained YOLOv8 (Ultralytics) bat detector.

    TODO(Phase 3): once labeled bat/ball clips exist, train with
    ultralytics.YOLO and load the resulting weights here; detect()
    should return a BatDetection built from the highest-confidence
    'bat' box's center. Until then, use get_default_bat_detector().
    """

    def __init__(self, weights_path: Optional[str] = None):
        self.weights_path = weights_path

    def detect(self, frame_bgr) -> Optional[BatDetection]:
        raise NotImplementedError(
            "YoloBatDetector has no trained weights yet - use get_default_bat_detector() "
            "(ArUco marker / color tape) until a YOLOv8 model is trained."
        )


class CompositeBatDetector(BatDetector):
    """Tries the ArUco marker first (robust); falls back to color tape."""

    def __init__(self):
        self._aruco = ArucoBatDetector()
        self._color = ColorMarkerBatDetector()

    def detect(self, frame_bgr) -> Optional[BatDetection]:
        detection = self._aruco.detect(frame_bgr)
        if detection is not None:
            return detection
        return self._color.detect(frame_bgr)


def get_default_bat_detector() -> BatDetector:
    return CompositeBatDetector()


def _nearest(track, target_ms):
    return min(track, key=lambda item: abs(item["timestamp_ms"] - target_ms))


def _angle_from_vertical(p_from, p_to):
    """0deg = straight up, positive = leaning toward +x (frame-relative)."""
    dx = p_to[0] - p_from[0]
    dy = p_to[1] - p_from[1]
    return float(math.degrees(math.atan2(dx, -dy)))


def compute_bat_metrics(bat_track, phase_markers, impact_landmarks, w, h, cm_per_px: Optional[float] = None):
    """
    Derive backlift/downswing angles, bat speed at impact, and
    point-of-contact offset from a bat track + the Phase 1 phase
    markers. Degrades to {"available": False, "note": ...} if the bat
    marker wasn't tracked clearly enough (too sparse) or segmentation
    failed - callers must not assume the numeric fields are present.
    """
    if not bat_track or len(bat_track) < MIN_TRACK_POINTS or not phase_markers:
        return {
            "available": False,
            "note": "No bat marker (ArUco or bright tape) tracked clearly enough to compute swing metrics.",
        }

    sorted_track = sorted(bat_track, key=lambda item: item["timestamp_ms"])

    stance_pt = _nearest(sorted_track, phase_markers["stance_ms"])
    backlift_pt = _nearest(sorted_track, phase_markers["backlift_ms"])
    impact_pt = _nearest(sorted_track, phase_markers["impact_ms"])

    backlift_height_px = stance_pt["center_px"][1] - backlift_pt["center_px"][1]
    backlift_angle_deg = _angle_from_vertical(stance_pt["center_px"], backlift_pt["center_px"])
    downswing_angle_deg = _angle_from_vertical(backlift_pt["center_px"], impact_pt["center_px"])

    impact_idx = sorted_track.index(impact_pt)
    neighbor_idx = impact_idx - 1 if impact_idx > 0 else min(impact_idx + 1, len(sorted_track) - 1)
    neighbor_pt = sorted_track[neighbor_idx]
    dt_s = abs(impact_pt["timestamp_ms"] - neighbor_pt["timestamp_ms"]) / 1000.0
    dist_px = math.hypot(
        impact_pt["center_px"][0] - neighbor_pt["center_px"][0],
        impact_pt["center_px"][1] - neighbor_pt["center_px"][1],
    )
    bat_speed_px_s = (dist_px / dt_s) if dt_s > 0 else None

    front_pt = pose_utils.compute_weight_transfer(impact_landmarks, w, h)["front_ankle_pt"]
    contact_offset_px = (
        impact_pt["center_px"][0] - front_pt[0],
        impact_pt["center_px"][1] - front_pt[1],
    )

    metrics = {
        "available": True,
        "backlift_height_px": backlift_height_px,
        "backlift_angle_deg": backlift_angle_deg,
        "downswing_angle_deg": downswing_angle_deg,
        "bat_speed_px_s": bat_speed_px_s,
        "contact_offset_px": contact_offset_px,
        "bat_speed_kmh": None,
        "contact_offset_cm": None,
    }

    if cm_per_px:
        if bat_speed_px_s is not None:
            # px/s * cm/px = cm/s; cm/s * 0.036 = km/h
            metrics["bat_speed_kmh"] = bat_speed_px_s * cm_per_px * 0.036
        metrics["contact_offset_cm"] = (
            contact_offset_px[0] * cm_per_px,
            contact_offset_px[1] * cm_per_px,
        )

    return metrics
