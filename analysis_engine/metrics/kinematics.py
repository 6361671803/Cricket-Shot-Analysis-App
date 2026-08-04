"""Low-level kinematics helpers: centre of mass, rotations, smoothing.

These are pure functions over landmark arrays so they can be unit-tested and
reused by the per-frame feature extractor, the weight-transfer module and the
real-time analyser alike.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..config import CONFIG
from ..pose import landmarks as lms

#: Segment definitions as (proximal, distal, mass-key, whether it is paired).
_SEGMENTS: Tuple[Tuple[int, int, str], ...] = (
    (lms.LEFT_SHOULDER, lms.LEFT_ELBOW, "upper_arm"),
    (lms.RIGHT_SHOULDER, lms.RIGHT_ELBOW, "upper_arm"),
    (lms.LEFT_ELBOW, lms.LEFT_WRIST, "forearm"),
    (lms.RIGHT_ELBOW, lms.RIGHT_WRIST, "forearm"),
    (lms.LEFT_HIP, lms.LEFT_KNEE, "thigh"),
    (lms.RIGHT_HIP, lms.RIGHT_KNEE, "thigh"),
    (lms.LEFT_KNEE, lms.LEFT_ANKLE, "shank"),
    (lms.RIGHT_KNEE, lms.RIGHT_ANKLE, "shank"),
    (lms.LEFT_ANKLE, lms.LEFT_FOOT_INDEX, "foot"),
    (lms.RIGHT_ANKLE, lms.RIGHT_FOOT_INDEX, "foot"),
)


def center_of_mass(points: np.ndarray) -> np.ndarray:
    """Whole-body centre of mass from segment-mass fractions (Winter's model).

    Works for both pixel (N, 2) and world (N, 3) landmark arrays.
    """
    masses = CONFIG.biomech.segment_mass
    total = 0.0
    accumulator = np.zeros(points.shape[1], dtype=float)

    head = (points[lms.LEFT_EAR] + points[lms.RIGHT_EAR] + points[lms.NOSE]) / 3.0
    accumulator += head * masses["head"]
    total += masses["head"]

    shoulders = lms.midpoint(points, lms.LEFT_SHOULDER, lms.RIGHT_SHOULDER)
    hips = lms.midpoint(points, lms.LEFT_HIP, lms.RIGHT_HIP)
    trunk = shoulders + (hips - shoulders) * 0.44  # trunk COM sits above mid-torso
    accumulator += trunk * masses["trunk"]
    total += masses["trunk"]

    for proximal, distal, key in _SEGMENTS:
        segment_com = points[proximal] + (points[distal] - points[proximal]) * 0.45
        accumulator += segment_com * masses[key]
        total += masses[key]

    return accumulator / max(1e-6, total)


def foot_line_projection(
    com: np.ndarray, back_foot: np.ndarray, front_foot: np.ndarray
) -> float:
    """Fraction of weight on the front foot from the COM projected on the base.

    0.0 = fully over the back foot, 1.0 = fully over the front foot.
    """
    base = front_foot - back_foot
    length_sq = float(np.dot(base, base))
    if length_sq < 1e-6:
        return 0.5
    t = float(np.dot(com - back_foot, base) / length_sq)
    return float(np.clip(t, 0.0, 1.0))


def rotation_about_vertical(points_world: np.ndarray, left: int, right: int) -> float:
    """Rotation of a body line (shoulders / hips) about the vertical axis.

    Uses the metric world landmarks' horizontal plane (x, z).  0 deg means the
    line is square to the camera; +-90 deg means the batter is fully side-on.
    The result is folded into (-90, 90] because a body line has no direction —
    a shoulder line rotated by 180 deg is the same line.
    """
    if points_world.size == 0:
        return 0.0
    vector = points_world[right] - points_world[left]
    angle = float(np.degrees(np.arctan2(vector[2], vector[0])))
    return (angle + 90.0) % 180.0 - 90.0


def knee_flexion(points: np.ndarray, side: str) -> float:
    """Knee flexion in degrees (0 = fully straight leg)."""
    if side == "left":
        hip, knee, ankle = lms.LEFT_HIP, lms.LEFT_KNEE, lms.LEFT_ANKLE
    else:
        hip, knee, ankle = lms.RIGHT_HIP, lms.RIGHT_KNEE, lms.RIGHT_ANKLE
    return max(0.0, 180.0 - lms.angle_deg(points[hip], points[knee], points[ankle]))


def heel_lift_ratio(points: np.ndarray, side: str, body_height_px: float) -> float:
    """How far the heel is raised above the toe, as a fraction of body height."""
    if body_height_px <= 0:
        return 0.0
    if side == "left":
        heel, toe = lms.LEFT_HEEL, lms.LEFT_FOOT_INDEX
    else:
        heel, toe = lms.RIGHT_HEEL, lms.RIGHT_FOOT_INDEX
    return float(max(0.0, points[toe][1] - points[heel][1]) / body_height_px)


def wrist_rotation(points: np.ndarray, side: str) -> float:
    """Angle between the forearm and the hand axis (a proxy for wrist roll)."""
    if side == "left":
        elbow, wrist, index, pinky = lms.LEFT_ELBOW, lms.LEFT_WRIST, lms.LEFT_INDEX, lms.LEFT_PINKY
    else:
        elbow, wrist, index, pinky = (
            lms.RIGHT_ELBOW,
            lms.RIGHT_WRIST,
            lms.RIGHT_INDEX,
            lms.RIGHT_PINKY,
        )
    hand = (points[index] + points[pinky]) / 2.0
    return lms.angle_deg(points[elbow], points[wrist], hand)


def moving_average(values: Sequence[float], window: int) -> List[float]:
    """Centred moving average that keeps the input length."""
    if window <= 1 or len(values) <= 2:
        return list(map(float, values))
    array = np.asarray(values, dtype=float)
    kernel = np.ones(window) / window
    padded = np.pad(array, (window // 2, window // 2), mode="edge")
    return list(np.convolve(padded, kernel, mode="valid")[: len(array)])


def velocity_series(
    positions: Sequence[Optional[Tuple[float, float]]], fps: float, px_per_meter: float
) -> List[float]:
    """Speed in km/h between consecutive (possibly missing) positions."""
    if fps <= 0 or px_per_meter <= 0:
        return [0.0] * len(positions)
    scale = (fps / px_per_meter) * 3.6
    out = [0.0]
    last_index, last_point = None, None
    for index, point in enumerate(positions):
        if point is None:
            out.append(out[-1] if out else 0.0)
            continue
        if last_point is not None:
            gap = max(1, index - last_index)
            distance = float(np.linalg.norm(np.array(point) - np.array(last_point)))
            out.append(distance / gap * scale)
        last_index, last_point = index, point
    return out[1 : len(positions) + 1] + [0.0] * max(0, len(positions) - len(out) + 1)


__all__ = [
    "center_of_mass",
    "foot_line_projection",
    "heel_lift_ratio",
    "knee_flexion",
    "moving_average",
    "rotation_about_vertical",
    "velocity_series",
    "wrist_rotation",
]
