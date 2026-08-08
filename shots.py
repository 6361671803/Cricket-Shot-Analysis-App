"""Shot database access and rule-based shot classification.

Kept separate from app.py so both the photo path (app.py) and the
video path (video_analysis.py) can classify shots without either
module importing the other.
"""

import json

import numpy as np

with open("shots.json", "r") as f:
    SHOT_DB = json.load(f)


def classify_shot_key(lm, w, h, forward_pct, angle: str) -> str:
    """
    Use pose + camera angle to pick a reasonable shot key from SHOT_DB.
    This is a simple rule-based classifier (not perfect, but decent).
    """

    lw = lm[15]
    rw = lm[16]
    ls = lm[11]
    rs = lm[12]
    lh = lm[23]
    rh = lm[24]

    lw_pt = np.array([lw.x * w, lw.y * h])
    rw_pt = np.array([rw.x * w, rw.y * h])
    ls_pt = np.array([ls.x * w, ls.y * h])
    rs_pt = np.array([rs.x * w, rs.y * h])
    lh_pt = np.array([lh.x * w, lh.y * h])
    rh_pt = np.array([rh.x * w, rh.y * h])

    bat_vec = rw_pt - lw_pt
    bat_angle_deg = abs(np.degrees(np.arctan2(bat_vec[1], bat_vec[0])))

    shoulder_rotation = abs(ls_pt[0] - rs_pt[0]) / max(1, abs(lh_pt[0] - rh_pt[0]))

    # Default
    shot_key = "Straight Drive"

    if angle == "A":  # Side view
        if forward_pct >= 60:
            if bat_angle_deg < 30:
                shot_key = "Lofted Drive"
            else:
                shot_key = "Straight Drive"
        else:
            if bat_angle_deg < 35:
                shot_key = "Pull Shot"
            else:
                shot_key = "Square Cut"

    elif angle == "B":  # Front view (bowler end)
        if forward_pct >= 60:
            shot_key = "Cover Drive"
        else:
            if shoulder_rotation > 1.15:
                shot_key = "Pull Shot"
            else:
                shot_key = "Square Cut"

    elif angle == "C":  # Diagonal / 45°
        if forward_pct >= 60:
            if bat_angle_deg < 28:
                shot_key = "Sweep"
            elif bat_angle_deg < 55:
                shot_key = "Straight Drive"
            else:
                shot_key = "Lofted Drive"
        else:
            if bat_angle_deg < 32:
                shot_key = "Pull Shot"
            else:
                shot_key = "Late Cut"

    if shot_key not in SHOT_DB:
        shot_key = "Straight Drive"

    return shot_key


def get_shot_info_from_db(shot_key: str):
    """Return shot info fields for templates."""
    data = SHOT_DB.get(shot_key, SHOT_DB["Straight Drive"])

    shot_name = shot_key
    shot_summary = data.get("summary", "")
    field_suggestion = data.get("fields", "")
    variations = data.get("variations", [])
    master_player = data.get("masters", "")
    shot_history = data.get("history", "")
    improvement_summary = data.get("improve", "")
    alt_safe = data.get("alt_safe", [])
    alt_aggressive = data.get("alt_aggressive", [])
    final_feedback = data.get("final_feedback", [])

    return (
        shot_name,
        shot_summary,
        field_suggestion,
        variations,
        master_player,
        shot_history,
        improvement_summary,
        alt_safe,
        alt_aggressive,
        final_feedback,
    )
