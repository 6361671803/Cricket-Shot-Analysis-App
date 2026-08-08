"""Illustrative wagon-wheel shot map (Phase 6).

Plots tagged shots by type + outcome around a field diagram. The
angle-per-shot-type mapping below is an illustrative approximation of
where each shot typically goes, not derived from real ball-tracking
or trajectory data - Phase 2's ball tracking is monocular and can't
produce true spatial/pitch-referenced coordinates. Frontend rendering
should caption this clearly.

Angle convention: 0deg = straight down the ground (12 o'clock),
increasing clockwise. This is a fixed illustrative convention, not
tied to striker handedness.
"""

from typing import Optional

SHOT_WAGON_ANGLE_DEG = {
    "Straight Drive": 0,
    "Lofted Drive": 0,
    "On Drive": 340,
    "Off Drive": 20,
    "Cover Drive": 45,
    "Square Drive": 70,
    "Square Cut": 90,
    "Late Cut": 120,
    "Upper Cut": 100,
    "Pull Shot": 260,
    "Hook Shot": 200,
    "Sweep": 230,
    "Slog Sweep": 270,
    "Reverse Sweep": 110,
    "Flick Shot": 320,
    "Leg Glance": 190,
    "Forward Defense": 0,
    "Backward Defense": 0,
    "Paddle Scoop": 190,
    "Helicopter Shot": 280,
    "Ramp Shot": 150,
    "Dilscoop": 170,
    "Switch Hit": 300,
    "Reverse Pull": 100,
}

DEFAULT_ANGLE_DEG = 0

RUN_COLORS = {
    0: "#9aa39c",
    1: "#3d8f5f",
    2: "#3d8f5f",
    3: "#2f6fb0",
    4: "#2f6fb0",
    5: "#c98b1f",
    6: "#c98b1f",
}
DISMISSAL_COLOR = "#c0392b"


def _radius_pct(runs: int) -> float:
    return min(95.0, 15.0 + runs * 12.0)


def build_wagon_wheel_points(sessions: list) -> list:
    """
    sessions: rows from db.get_tagged_sessions() (dicts with shot_key,
    outcome_runs, outcome_dismissal). Returns a list of
    {angle_deg, radius_pct, shot_key, outcome_label, color} for the
    frontend to plot as an SVG scatter.
    """
    points = []
    for session in sessions:
        shot_key = session.get("shot_key")
        angle_deg = SHOT_WAGON_ANGLE_DEG.get(shot_key, DEFAULT_ANGLE_DEG)
        dismissal = bool(session.get("outcome_dismissal"))
        runs = session.get("outcome_runs")

        if dismissal:
            radius_pct = 95.0
            color = DISMISSAL_COLOR
            outcome_label = "Out"
        else:
            runs = runs if runs is not None else 0
            radius_pct = _radius_pct(runs)
            color = RUN_COLORS.get(runs, RUN_COLORS[6])
            outcome_label = "Dot ball" if runs == 0 else f"{runs} run{'s' if runs != 1 else ''}"

        points.append({
            "angle_deg": angle_deg,
            "radius_pct": radius_pct,
            "shot_key": shot_key,
            "outcome_label": outcome_label,
            "color": color,
        })
    return points
