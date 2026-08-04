"""Cricket-ball detection and trajectory analysis.

The ball is small, fast and often motion-blurred, so a single cue is never
enough.  Each frame is scored with a fusion of four cues:

* **motion** – frame differencing against the previous frame isolates fast movers;
* **shape** – contour circularity / radius gating rejects limbs and bat blobs;
* **colour** – red-leather *and* white-ball colour models are both evaluated;
* **prediction** – a constant-velocity gate keeps the track locked once started.

The resulting per-frame observations are then post-processed into a trajectory
with speed, bounce frame, impact point, post-impact direction and a predicted
forward path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..config import CONFIG, TrackingConfig
from ..datatypes import BallObservation, PoseFrame
from ..pose import landmarks as lms


@dataclass
class _Candidate:
    position: Tuple[float, float]
    radius: float
    score: float


def _colour_score(patch_bgr: np.ndarray) -> float:
    """Likelihood that a patch is a red-leather or white cricket ball."""
    if patch_bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = ((hue <= 12) | (hue >= 168)) & (sat >= 70) & (val >= 45)
    white = (sat <= 70) & (val >= 165)
    return float(max(red.mean(), white.mean()))


class BallTracker:
    """Frame-by-frame ball tracker (stateful; feed frames in order)."""

    def __init__(self, config: Optional[TrackingConfig] = None) -> None:
        self.config = config or CONFIG.tracking
        self._prev_gray: Optional[np.ndarray] = None
        self._last: Optional[Tuple[float, float]] = None
        self._velocity = np.zeros(2, dtype=float)
        self._missing = 0
        self.observations: List[BallObservation] = []

    # ------------------------------------------------------------------ frame
    def update(
        self, frame_bgr: np.ndarray, frame_index: int, timestamp: float, pose: Optional[PoseFrame]
    ) -> BallObservation:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        observation = BallObservation(frame_index=frame_index, timestamp=timestamp, position=None)

        if self._prev_gray is not None:
            candidates = self._candidates(frame_bgr, gray, pose)
            best = self._pick(candidates)
            if best is not None:
                if self._last is not None:
                    self._velocity = np.array(best.position) - np.array(self._last)
                self._last = best.position
                self._missing = 0
                observation.position = best.position
                observation.radius = best.radius
                observation.confidence = min(1.0, best.score)
            else:
                self._missing += 1
                if self._last is not None and self._missing <= self.config.ball_max_missing_frames:
                    predicted = np.array(self._last) + self._velocity
                    self._last = (float(predicted[0]), float(predicted[1]))
                    observation.position = self._last
                    observation.interpolated = True
                    observation.confidence = 0.25
                else:
                    self._last = None
                    self._velocity[:] = 0.0

        self._prev_gray = gray
        self.observations.append(observation)
        return observation

    # ------------------------------------------------------------- candidates
    def _candidates(
        self, frame_bgr: np.ndarray, gray: np.ndarray, pose: Optional[PoseFrame]
    ) -> List[_Candidate]:
        diff = cv2.absdiff(gray, self._prev_gray)
        _, mask = cv2.threshold(diff, self.config.ball_diff_threshold, 255, cv2.THRESH_BINARY)
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        torso_center, torso_radius = self._torso(pose)
        height, width = gray.shape[:2]
        out: List[_Candidate] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.config.ball_min_area_px:
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            if not (self.config.ball_min_radius_px <= radius <= self.config.ball_max_radius_px):
                continue
            circularity = area / max(1e-6, np.pi * radius * radius)
            if circularity < 0.35:
                continue

            x0, x1 = int(max(0, cx - radius)), int(min(width, cx + radius + 1))
            y0, y1 = int(max(0, cy - radius)), int(min(height, cy + radius + 1))
            colour = _colour_score(frame_bgr[y0:y1, x0:x1])

            score = 0.45 * circularity + 0.45 * colour + 0.10
            if torso_center is not None:
                distance = float(np.linalg.norm(np.array([cx, cy]) - torso_center))
                if distance < torso_radius * 0.55:
                    score *= 0.35  # deep inside the body: almost certainly a limb
            out.append(_Candidate((float(cx), float(cy)), float(radius), float(score)))
        return out

    def _pick(self, candidates: Sequence[_Candidate]) -> Optional[_Candidate]:
        if not candidates:
            return None
        if self._last is None:
            best = max(candidates, key=lambda c: c.score)
            return best if best.score >= 0.34 else None

        predicted = np.array(self._last) + self._velocity
        gate = self.config.ball_max_jump_px * (1 + 0.35 * self._missing)
        scored: List[Tuple[float, _Candidate]] = []
        for candidate in candidates:
            distance = float(np.linalg.norm(np.array(candidate.position) - predicted))
            if distance > gate:
                continue
            proximity = 1.0 - min(1.0, distance / gate)
            scored.append((candidate.score * 0.5 + proximity * 0.5, candidate))
        if not scored:
            return None
        return max(scored, key=lambda item: item[0])[1]

    @staticmethod
    def _torso(pose: Optional[PoseFrame]) -> Tuple[Optional[np.ndarray], float]:
        if pose is None or not pose.detected:
            return None, 0.0
        shoulders = lms.midpoint(pose.pixels, lms.LEFT_SHOULDER, lms.RIGHT_SHOULDER)
        hips = lms.midpoint(pose.pixels, lms.LEFT_HIP, lms.RIGHT_HIP)
        center = (shoulders + hips) / 2.0
        radius = float(np.linalg.norm(shoulders - hips)) + 1.0
        return center, radius


# --------------------------------------------------------------------- track
def _clean_series(observations: Sequence[BallObservation]) -> List[Optional[np.ndarray]]:
    return [np.array(o.position) if o.position is not None else None for o in observations]


def longest_consistent_tracklet(
    observations: Sequence[BallObservation], max_jump_px: float
) -> List[BallObservation]:
    """Keep only the longest temporally consistent run of detections.

    Crowd movement, scoreboards and camera shake produce isolated false
    positives.  A real ball produces a smooth run of detections, so the
    observations are split wherever the displacement exceeds the motion gate and
    only the longest run survives.
    """
    tracklets: List[List[BallObservation]] = []
    current: List[BallObservation] = []
    previous: Optional[np.ndarray] = None
    for observation in observations:
        if observation.position is None:
            if current:
                tracklets.append(current)
            current, previous = [], None
            continue
        point = np.array(observation.position)
        if previous is not None and float(np.linalg.norm(point - previous)) > max_jump_px:
            tracklets.append(current)
            current = []
        current.append(observation)
        previous = point
    if current:
        tracklets.append(current)
    if not tracklets:
        return []
    best = max(tracklets, key=len)
    return best if len(best) >= 3 else []


def analyse_ball_track(
    observations: Sequence[BallObservation],
    px_per_meter: float,
    fps: float,
    impact_index: Optional[int] = None,
) -> Dict[str, object]:
    """Turn raw ball observations into a trajectory description.

    Returns speed (km/h), bounce frame, impact point, post-impact direction and
    a short predicted path extrapolated with gravity-corrected motion.
    """
    observations = longest_consistent_tracklet(
        observations, CONFIG.tracking.ball_max_jump_px
    ) or list(observations)
    #: ``(frame_index, position)`` pairs — frame indices, never list positions.
    valid = [
        (observation.frame_index, np.array(observation.position))
        for observation in observations
        if observation.position is not None
    ]
    interpolated = {o.frame_index: o.interpolated for o in observations}
    result: Dict[str, object] = {
        "detected": bool(valid),
        "path": [
            {
                "frame": frame,
                "x": float(point[0]),
                "y": float(point[1]),
                "interpolated": interpolated.get(frame, False),
            }
            for frame, point in valid
        ],
        "speed_kmh": 0.0,
        "max_speed_kmh": 0.0,
        "bounce_frame": None,
        "impact_point": None,
        "post_impact_direction_deg": None,
        "post_impact_speed_kmh": 0.0,
        "predicted_path": [],
        "direction_change_deg": 0.0,
    }
    if len(valid) < 3 or px_per_meter <= 0 or fps <= 0:
        return result

    scale = (fps / px_per_meter) * 3.6  # px/frame -> km/h
    speeds: List[float] = []
    for (i0, p0), (i1, p1) in zip(valid, valid[1:]):
        frames = max(1, i1 - i0)
        speeds.append(float(np.linalg.norm(p1 - p0)) / frames * scale)
    result["speed_kmh"] = round(float(np.median(speeds)), 1)
    result["max_speed_kmh"] = round(float(np.percentile(speeds, 90)), 1)

    # --- bounce: local maximum of y (lowest point on screen) before impact ---
    ys = np.array([p[1] for _, p in valid])
    idxs = [i for i, _ in valid]
    search_end = len(valid)
    if impact_index is not None:
        search_end = next((k for k, i in enumerate(idxs) if i >= impact_index), len(valid))
    if search_end >= 4:
        window = ys[:search_end]
        local = int(np.argmax(window))
        if 0 < local < search_end - 1:
            result["bounce_frame"] = idxs[local]

    # --- direction change around impact ---
    if impact_index is not None:
        before = [(i, p) for i, p in valid if i < impact_index]
        after = [(i, p) for i, p in valid if i > impact_index]
        if len(before) >= 2 and len(after) >= 2:
            in_vec = before[-1][1] - before[-2][1]
            out_vec = after[1][1] - after[0][1]
            result["direction_change_deg"] = round(
                abs(
                    lms.signed_line_angle_deg((0, 0), out_vec)
                    - lms.signed_line_angle_deg((0, 0), in_vec)
                ),
                1,
            )
            result["post_impact_direction_deg"] = round(
                lms.signed_line_angle_deg((0, 0), out_vec), 1
            )
            result["post_impact_speed_kmh"] = round(
                float(np.linalg.norm(out_vec)) * scale, 1
            )
        at_impact = min(valid, key=lambda item: abs(item[0] - impact_index))
        result["impact_point"] = {"x": float(at_impact[1][0]), "y": float(at_impact[1][1])}

    # --- predicted forward path (constant velocity + gravity) ---
    tail = valid[-min(4, len(valid)) :]
    if len(tail) >= 2:
        velocity = (tail[-1][1] - tail[0][1]) / max(1, tail[-1][0] - tail[0][0])
        gravity_px = 9.81 * px_per_meter / (fps * fps)
        position = tail[-1][1].astype(float).copy()
        predicted = []
        for step in range(1, 13):
            position = position + velocity
            velocity = velocity + np.array([0.0, gravity_px])
            predicted.append({"step": step, "x": float(position[0]), "y": float(position[1])})
        result["predicted_path"] = predicted

    return result


__all__ = ["BallTracker", "analyse_ball_track"]
