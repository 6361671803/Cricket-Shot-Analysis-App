"""Biomechanics metrics computed from a single frame's pose landmarks,
compared against illustrative technique-archetype benchmark ranges.

See benchmarks.json's "_note" field: these ranges are coaching
heuristics authored for this app, not validated sports-science data.
"""

import json
import math
from typing import Optional

import pose_utils

with open("benchmarks.json", "r", encoding="utf-8") as f:
    BENCHMARKS = json.load(f)

# MediaPipe Pose landmark indices used here.
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28

# Maps every shots.json key to a technique archetype with its own
# benchmark ranges in benchmarks.json. "Drive" is the fallback for any
# unmapped shot (shouldn't happen in practice - all 24 current shots
# are covered).
SHOT_ARCHETYPES = {
    "Straight Drive": "Drive",
    "Cover Drive": "Drive",
    "On Drive": "Drive",
    "Off Drive": "Drive",
    "Square Drive": "Drive",
    "Lofted Drive": "Drive",
    "Pull Shot": "Cut-Pull-Hook",
    "Hook Shot": "Cut-Pull-Hook",
    "Square Cut": "Cut-Pull-Hook",
    "Late Cut": "Cut-Pull-Hook",
    "Upper Cut": "Cut-Pull-Hook",
    "Sweep": "Sweep",
    "Slog Sweep": "Sweep",
    "Reverse Sweep": "Sweep",
    "Flick Shot": "Wristy",
    "Leg Glance": "Wristy",
    "Forward Defense": "Defensive",
    "Backward Defense": "Defensive",
    "Paddle Scoop": "Modern-Power",
    "Helicopter Shot": "Modern-Power",
    "Ramp Shot": "Modern-Power",
    "Dilscoop": "Modern-Power",
    "Switch Hit": "Modern-Power",
    "Reverse Pull": "Modern-Power",
}

DEFAULT_ARCHETYPE = "Drive"


def angle_at_joint(a, b, c) -> float:
    """Angle in degrees at vertex b, formed by points a-b-c."""
    bax, bay = a[0] - b[0], a[1] - b[1]
    bcx, bcy = c[0] - b[0], c[1] - b[1]
    dot = bax * bcx + bay * bcy
    mag = math.hypot(bax, bay) * math.hypot(bcx, bcy)
    if mag == 0:
        return 0.0
    cos_angle = max(-1.0, min(1.0, dot / mag))
    return math.degrees(math.acos(cos_angle))


def _line_angle_diff_deg(p1, p2, q1, q2) -> float:
    """Angular difference (0-90deg) between line p1->p2 and line q1->q2."""
    v1x, v1y = p2[0] - p1[0], p2[1] - p1[1]
    v2x, v2y = q2[0] - q1[0], q2[1] - q1[1]
    dot = v1x * v2x + v1y * v2y
    mag = math.hypot(v1x, v1y) * math.hypot(v2x, v2y)
    if mag == 0:
        return 0.0
    cos_angle = max(-1.0, min(1.0, dot / mag))
    diff = math.degrees(math.acos(cos_angle))
    return min(diff, 180 - diff)


def compute_biomechanics(landmarks, w, h, front_side: str, bat_metrics: Optional[dict] = None,
                          cm_per_px: Optional[float] = None) -> dict:
    """
    Compute biomechanics metrics from one frame's pose landmarks.
    front_side is "left" or "right" (from compute_weight_transfer's
    front-foot decision) - which side's knee/elbow count as "front".
    """
    def pt(idx):
        return pose_utils.get_point_and_vis(landmarks[idx], w, h)[0]

    nose_pt = pt(NOSE)
    if front_side == "left":
        hip_pt, knee_pt, ankle_pt = pt(L_HIP), pt(L_KNEE), pt(L_ANKLE)
        shoulder_pt, elbow_pt, wrist_pt = pt(L_SHOULDER), pt(L_ELBOW), pt(L_WRIST)
    else:
        hip_pt, knee_pt, ankle_pt = pt(R_HIP), pt(R_KNEE), pt(R_ANKLE)
        shoulder_pt, elbow_pt, wrist_pt = pt(R_SHOULDER), pt(R_ELBOW), pt(R_WRIST)

    l_shoulder_pt, r_shoulder_pt = pt(L_SHOULDER), pt(R_SHOULDER)
    l_hip_pt, r_hip_pt = pt(L_HIP), pt(R_HIP)

    head_knee_offset_px = abs(nose_pt[0] - knee_pt[0])
    front_knee_flex_deg = angle_at_joint(hip_pt, knee_pt, ankle_pt)
    elbow_extension_deg = angle_at_joint(shoulder_pt, elbow_pt, wrist_pt)
    shoulder_hip_rotation_deg = _line_angle_diff_deg(l_shoulder_pt, r_shoulder_pt, l_hip_pt, r_hip_pt)

    bat_speed_kmh = None
    if bat_metrics and bat_metrics.get("available"):
        bat_speed_kmh = bat_metrics.get("bat_speed_kmh")

    return {
        "front_side": front_side,
        "head_knee_offset_px": head_knee_offset_px,
        "head_knee_offset_cm": (head_knee_offset_px * cm_per_px) if cm_per_px else None,
        "front_knee_flex_deg": front_knee_flex_deg,
        "elbow_extension_deg": elbow_extension_deg,
        "shoulder_hip_rotation_deg": shoulder_hip_rotation_deg,
        "bat_speed_kmh": bat_speed_kmh,
    }


def _status_for(value, value_range):
    lo, hi = value_range
    if value < lo:
        return "under"
    if value > hi:
        return "over"
    return "within"


def compare_to_benchmarks(shot_key: str, metrics: dict) -> dict:
    """
    Compare computed biomechanics metrics against the shot's archetype
    benchmark ranges. Returns per-metric {value, unit, range, status}
    plus an overall_score (% of evaluated metrics within range, or
    None if nothing could be evaluated).
    """
    archetype = SHOT_ARCHETYPES.get(shot_key, DEFAULT_ARCHETYPE)
    ranges = BENCHMARKS["archetypes"][archetype]
    generic = BENCHMARKS["generic"]

    entries = {}
    within_count = 0
    evaluated = 0

    def add_entry(key, value, unit, value_range):
        nonlocal within_count, evaluated
        if value is None or value_range is None:
            entries[key] = {"value": value, "unit": unit, "range": value_range, "status": "unavailable"}
            return
        status = _status_for(value, value_range)
        entries[key] = {"value": value, "unit": unit, "range": list(value_range), "status": status}
        evaluated += 1
        if status == "within":
            within_count += 1

    add_entry("head_knee_offset", metrics.get("head_knee_offset_cm"), "cm", generic.get("head_knee_offset_cm"))
    add_entry("front_knee_flex", metrics.get("front_knee_flex_deg"), "deg", ranges.get("front_knee_flex_deg"))
    add_entry("elbow_extension", metrics.get("elbow_extension_deg"), "deg", ranges.get("elbow_extension_deg"))
    add_entry("shoulder_hip_rotation", metrics.get("shoulder_hip_rotation_deg"), "deg", ranges.get("shoulder_hip_rotation_deg"))
    add_entry("bat_speed", metrics.get("bat_speed_kmh"), "km/h", ranges.get("bat_speed_kmh"))

    overall_score = round(100 * within_count / evaluated) if evaluated > 0 else None

    return {
        "archetype": archetype,
        "metrics": entries,
        "overall_score": overall_score,
    }
