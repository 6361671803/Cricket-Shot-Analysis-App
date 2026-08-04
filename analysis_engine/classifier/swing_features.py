"""Aggregate a whole swing into the feature vector used for classification.

Per-frame features describe *instants*; shots are defined by the **sequence**
(stance -> backlift -> downswing -> contact -> follow-through).  This module
compresses the sequence into the vocabulary documented in
:mod:`analysis_engine.classifier.shot_definitions`, using a window around the
detected impact frame instead of one isolated frame.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from ..config import CONFIG
from ..datatypes import FrameFeatures, ImpactInfo


def _index_of_frame(features: Sequence[FrameFeatures], frame_index: int) -> int:
    for position, feature in enumerate(features):
        if feature.frame_index >= frame_index:
            return position
    return len(features) - 1


def _median(values: Sequence[float], default: float = 0.0) -> float:
    cleaned = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.median(cleaned)) if cleaned else default


def build_swing_features(
    features: Sequence[FrameFeatures],
    impact: ImpactInfo,
    bat_track: Optional[Dict[str, object]] = None,
    ball_track: Optional[Dict[str, object]] = None,
) -> Dict[str, float]:
    """Compress a swing into the classifier's feature vector."""
    valid = [f for f in features if f.pose_detected]
    if not valid:
        return {}

    bat_track = bat_track or {}
    temporal = CONFIG.temporal
    impact_pos = _index_of_frame(valid, impact.frame_index) if impact.frame_index >= 0 else len(valid) - 1
    impact_pos = int(np.clip(impact_pos, 0, len(valid) - 1))
    at_impact = valid[impact_pos]

    stance_slice = valid[: max(1, min(4, impact_pos))] or [valid[0]]
    pre = valid[max(0, impact_pos - temporal.pre_impact_window) : impact_pos + 1] or [at_impact]
    post = valid[impact_pos + 1 : impact_pos + 1 + temporal.post_impact_window]

    body_height = _median([f.body_height_px for f in valid], 1.0) or 1.0
    ankle_y = _median(
        [max(f.front_ankle[1], f.back_ankle[1]) for f in valid if f.front_ankle and f.back_ankle],
        0.0,
    )

    # --- contact point: prefer the tracked ball, fall back to the bat toe ---
    contact = at_impact.ball or at_impact.bat_tip
    if contact is None:
        contact = at_impact.hip_center or (0.0, ankle_y)
    contact_height = float((ankle_y - contact[1]) / body_height)

    facing_sign = at_impact.facing_sign or 1.0
    hip_x = (at_impact.hip_center or contact)[0]
    contact_offset = float((contact[0] - hip_x) * facing_sign / body_height)

    hip_height = float((ankle_y - (at_impact.hip_center or (0.0, ankle_y))[1]) / body_height)

    # --- bat travel direction at contact, relative to "down the ground" ---
    swing_direction = _swing_direction(valid, impact_pos)

    stance_rotation = _median([f.shoulder_rotation_deg for f in stance_slice], 0.0)
    body_rotation = float((at_impact.shoulder_rotation_deg - stance_rotation) * facing_sign)
    if body_rotation > 180:
        body_rotation -= 360
    elif body_rotation < -180:
        body_rotation += 360

    stance_wrist = _median([f.wrist_rotation_deg for f in stance_slice], 0.0)
    wrist_roll = float(
        max((abs(f.wrist_rotation_deg - stance_wrist) for f in (pre + post) or [at_impact]), default=0.0)
    )

    bat_finish_height = float(max((f.bat_height_ratio for f in post), default=at_impact.bat_height_ratio))

    stance_hands = _median([f.hands_axis_deg for f in stance_slice], 0.0)
    hands_delta = abs(((at_impact.hands_axis_deg - stance_hands) + 180) % 360 - 180)
    stance_facing = _median([f.facing_sign for f in stance_slice], facing_sign)

    swing_plane = float(bat_track.get("swing_plane_deg") or 0.0)
    if swing_plane <= 0.0:
        # Too few bat samples for a trajectory fit: fall back to the blade angle.
        swing_plane = float(at_impact.bat_angle_deg)

    vector: Dict[str, float] = {
        "forward_pct": float(at_impact.forward_pct),
        "contact_height": contact_height,
        "swing_plane": swing_plane,
        "contact_offset": contact_offset,
        "swing_direction": swing_direction,
        "follow_through": float(bat_track.get("follow_through_ratio", 0.0) or 0.0),
        "hip_height": hip_height,
        "body_rotation": body_rotation,
        "wrist_roll": wrist_roll,
        "bat_finish_height": bat_finish_height,
        "backlift": float(bat_track.get("backlift_height_ratio", 0.0) or 0.0),
        "stride": float(at_impact.stride_length_m),
        "reverse": 1.0 if hands_delta > 95 else 0.0,
        "switch": 1.0 if stance_facing * facing_sign < 0 else 0.0,
    }

    # Features that need a *sequence* are dropped when only a photo (or a very
    # short clip) was supplied, so the classifier is not fed zeros that would
    # masquerade as real measurements.
    if len(valid) < CONFIG.temporal.min_frames_for_sequence:
        for key in (
            "follow_through",
            "bat_finish_height",
            "backlift",
            "wrist_roll",
            "body_rotation",
            "swing_direction",
        ):
            vector.pop(key, None)

    return {k: float(np.nan_to_num(v)) for k, v in vector.items()}


def _swing_direction(valid: Sequence[FrameFeatures], impact_pos: int) -> float:
    """Angle (0-90 deg) between the bat's travel at contact and down the ground.

    ``0`` means the bat swings straight down the pitch (drives, defence) and
    ``90`` means fully across the line (pull, cut, sweep).
    """
    tips: List[np.ndarray] = []
    for feature in valid[max(0, impact_pos - 3) : impact_pos + 3]:
        if feature.bat_tip is not None:
            tips.append(np.array(feature.bat_tip))
    if len(tips) < 2:
        return 45.0
    travel = tips[-1] - tips[0]
    if np.linalg.norm(travel) < 1e-3:
        return 45.0

    reference = None
    at_impact = valid[impact_pos]
    if at_impact.front_ankle and at_impact.back_ankle:
        reference = np.array(at_impact.front_ankle) - np.array(at_impact.back_ankle)
    if reference is None or np.linalg.norm(reference) < 1e-3:
        reference = np.array([1.0, 0.0])
    reference = reference / np.linalg.norm(reference)
    travel = travel / np.linalg.norm(travel)
    cos = float(np.clip(abs(np.dot(travel, reference)), 0.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


__all__ = ["build_swing_features"]
