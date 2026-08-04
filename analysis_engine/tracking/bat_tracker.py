"""Bat detection and swing-path analysis.

The bat is tracked independently from the body.  The grip is anchored on the
hands (a reliable pose landmark cluster) and the blade is then found from image
evidence: a Canny + probabilistic-Hough search inside a hand-centred ROI keeps
only near-straight segments that start at the grip, and the longest consistent
one defines the bat toe.  When no line survives (blur, low contrast) the blade
falls back to the forearm/hand direction extrapolated by an anthropometric bat
length, so the tracker never drops out mid-swing.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..config import CONFIG, TrackingConfig
from ..datatypes import BatObservation, PoseFrame
from ..pose import landmarks as lms


def _hand_center(pose: PoseFrame, side: int) -> np.ndarray:
    wrist = lms.LEFT_WRIST if side == 0 else lms.RIGHT_WRIST
    index = lms.LEFT_INDEX if side == 0 else lms.RIGHT_INDEX
    thumb = lms.LEFT_THUMB if side == 0 else lms.RIGHT_THUMB
    return (pose.pixels[wrist] + pose.pixels[index] + pose.pixels[thumb]) / 3.0


class BatTracker:
    """Stateful bat tracker; call :meth:`update` once per frame in order."""

    def __init__(self, config: Optional[TrackingConfig] = None) -> None:
        self.config = config or CONFIG.tracking
        self._smoothed_tip: Optional[np.ndarray] = None
        self._last_length: Optional[float] = None
        self.observations: List[BatObservation] = []

    def update(
        self, frame_bgr: np.ndarray, frame_index: int, timestamp: float, pose: Optional[PoseFrame]
    ) -> BatObservation:
        observation = BatObservation(
            frame_index=frame_index, timestamp=timestamp, grip=None, tip=None
        )
        if pose is None or not pose.detected:
            self.observations.append(observation)
            return observation

        left_hand = _hand_center(pose, 0)
        right_hand = _hand_center(pose, 1)
        grip = (left_hand + right_hand) / 2.0
        shoulder_width = max(
            12.0,
            float(np.linalg.norm(pose.pixels[lms.LEFT_SHOULDER] - pose.pixels[lms.RIGHT_SHOULDER])),
        )

        # Default blade direction: from the top hand through the bottom hand.
        hand_axis = right_hand - left_hand
        if np.linalg.norm(hand_axis) < 1e-3:
            elbow_mid = lms.midpoint(pose.pixels, lms.LEFT_ELBOW, lms.RIGHT_ELBOW)
            hand_axis = grip - elbow_mid
        default_dir = hand_axis / max(1e-6, float(np.linalg.norm(hand_axis)))
        default_length = self._last_length or shoulder_width * self.config.bat_length_shoulder_ratio

        tip, confidence, from_image = self._find_blade(
            frame_bgr, grip, shoulder_width, default_dir, default_length
        )

        if self._smoothed_tip is not None:
            alpha = self.config.bat_smoothing
            tip = alpha * tip + (1.0 - alpha) * self._smoothed_tip
        self._smoothed_tip = tip
        self._last_length = float(np.linalg.norm(tip - grip))

        observation.grip = (float(grip[0]), float(grip[1]))
        observation.tip = (float(tip[0]), float(tip[1]))
        observation.angle_deg = lms.line_angle_deg(observation.grip, observation.tip)
        observation.length_px = self._last_length
        observation.confidence = confidence
        observation.from_image_evidence = from_image
        self.observations.append(observation)
        return observation

    # ---------------------------------------------------------------- helpers
    def _find_blade(
        self,
        frame_bgr: np.ndarray,
        grip: np.ndarray,
        shoulder_width: float,
        default_dir: np.ndarray,
        default_length: float,
    ) -> Tuple[np.ndarray, float, bool]:
        height, width = frame_bgr.shape[:2]
        half = int(shoulder_width * self.config.bat_roi_scale)
        x0, x1 = int(max(0, grip[0] - half)), int(min(width, grip[0] + half))
        y0, y1 = int(max(0, grip[1] - half)), int(min(height, grip[1] + half))
        fallback = grip + default_dir * default_length
        if x1 - x0 < 16 or y1 - y0 < 16:
            return fallback, 0.25, False

        roi = frame_bgr[y0:y1, x0:x1]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(gray, self.config.bat_canny_low, self.config.bat_canny_high)
        diagonal = float(np.hypot(x1 - x0, y1 - y0))
        min_len = diagonal * self.config.bat_min_line_len_ratio
        segments = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180.0,
            threshold=28,
            minLineLength=int(min_len),
            maxLineGap=int(max(4, min_len * 0.25)),
        )
        if segments is None or len(segments) == 0:
            return fallback, 0.3, False
        # OpenCV returns (N, 1, 4) historically and (N, 4) in newer builds.
        segments = np.asarray(segments).reshape(-1, 4)

        grip_local = grip - np.array([x0, y0])
        best_tip: Optional[np.ndarray] = None
        best_score = 0.0
        for x_a, y_a, x_b, y_b in segments:
            p_a = np.array([float(x_a), float(y_a)])
            p_b = np.array([float(x_b), float(y_b)])
            d_a = float(np.linalg.norm(p_a - grip_local))
            d_b = float(np.linalg.norm(p_b - grip_local))
            near, far = (p_a, p_b) if d_a <= d_b else (p_b, p_a)
            near_distance = min(d_a, d_b)
            if near_distance > shoulder_width * 1.1:
                continue  # not attached to the hands
            length = float(np.linalg.norm(far - near))
            if length < min_len:
                continue
            direction = (far - near) / max(1e-6, length)
            alignment = float(np.dot(direction, default_dir))  # -1..1
            score = (
                0.45 * min(1.0, length / (shoulder_width * 2.4))
                + 0.35 * max(0.0, abs(alignment))
                + 0.20 * (1.0 - min(1.0, near_distance / (shoulder_width * 1.1)))
            )
            if score > best_score:
                best_score = score
                best_tip = far + np.array([x0, y0])

        if best_tip is None:
            return fallback, 0.3, False
        # Guard against absurd blade lengths from stray background edges.
        length = float(np.linalg.norm(best_tip - grip))
        if not (shoulder_width * 0.8 <= length <= shoulder_width * 4.0):
            return fallback, 0.35, False
        return best_tip, min(1.0, 0.45 + best_score * 0.55), True


# --------------------------------------------------------------------- track
def analyse_bat_track(
    observations: Sequence[BatObservation],
    px_per_meter: float,
    fps: float,
    impact_index: Optional[int] = None,
    body_height_px: float = 0.0,
    shoulder_y: Optional[float] = None,
) -> Dict[str, object]:
    """Derive swing metrics from the bat-tip path.

    Includes bat speed (km/h), swing plane, arc length, backlift height, bat
    face angle at impact, sweet-spot position and follow-through extent.
    """
    tips = [(i, np.array(o.tip)) for i, o in enumerate(observations) if o.tip is not None]
    result: Dict[str, object] = {
        "detected": bool(tips),
        "path": [
            {"frame": observations[i].frame_index, "x": float(p[0]), "y": float(p[1])}
            for i, p in tips
        ],
        "speed_kmh": 0.0,
        "max_speed_kmh": 0.0,
        "max_speed_frame": None,
        "swing_plane_deg": 0.0,
        "swing_plane_label": "Unknown",
        "swing_arc_px": 0.0,
        "backlift_angle_deg": 0.0,
        "backlift_height_ratio": 0.0,
        "bat_angle_at_impact_deg": 0.0,
        "bat_face_angle_deg": 0.0,
        "follow_through_ratio": 0.0,
        "follow_through_label": "Unknown",
        "sweet_spot_ratio": 0.72,
    }
    if len(tips) < 3 or fps <= 0:
        return result

    scale = (fps / max(1e-6, px_per_meter)) * 3.6
    speeds: List[Tuple[int, float]] = []
    arc = 0.0
    for (i0, p0), (i1, p1) in zip(tips, tips[1:]):
        step = float(np.linalg.norm(p1 - p0))
        arc += step
        speeds.append((observations[i1].frame_index, step / max(1, i1 - i0) * scale))
    result["swing_arc_px"] = round(arc, 1)
    if speeds:
        values = np.array([s for _, s in speeds])
        result["speed_kmh"] = round(float(np.median(values)), 1)
        peak = int(np.argmax(values))
        result["max_speed_kmh"] = round(float(values[peak]), 1)
        result["max_speed_frame"] = speeds[peak][0]

    # Swing plane: principal axis of the tip trajectory.
    coords = np.array([p for _, p in tips])
    centred = coords - coords.mean(axis=0)
    if len(centred) >= 3:
        _, _, vectors = np.linalg.svd(centred, full_matrices=False)
        axis = vectors[0]
        plane = lms.line_angle_deg((0.0, 0.0), (float(axis[0]), float(axis[1])))
        result["swing_plane_deg"] = round(plane, 1)
        result["swing_plane_label"] = (
            "Vertical (V-shaped)"
            if plane >= 55
            else "Diagonal"
            if plane >= 30
            else "Horizontal (cross-batted)"
        )

    # Backlift: highest tip position (smallest y) before the downswing.
    limit = impact_index if impact_index is not None else len(observations)
    pre = [(i, p) for i, p in tips if observations[i].frame_index <= limit]
    if pre:
        highest = min(pre, key=lambda item: item[1][1])
        result["backlift_angle_deg"] = round(observations[highest[0]].angle_deg, 1)
        if body_height_px > 0 and shoulder_y is not None:
            result["backlift_height_ratio"] = round(
                float((shoulder_y - highest[1][1]) / body_height_px), 3
            )

    if impact_index is not None:
        at_impact = min(tips, key=lambda item: abs(observations[item[0]].frame_index - impact_index))
        observation = observations[at_impact[0]]
        result["bat_angle_at_impact_deg"] = round(observation.angle_deg, 1)
        # Bat face angle: deviation of the blade from the vertical plane.
        result["bat_face_angle_deg"] = round(90.0 - observation.angle_deg, 1)
        post = [p for i, p in tips if observations[i].frame_index > impact_index]
        if post and body_height_px > 0:
            travel = float(np.linalg.norm(post[-1] - at_impact[1]))
            ratio = travel / body_height_px
            result["follow_through_ratio"] = round(ratio, 3)
            result["follow_through_label"] = (
                "Full" if ratio >= 0.45 else "Partial" if ratio >= 0.2 else "Checked"
            )
    return result


__all__ = ["BatTracker", "analyse_bat_track"]
