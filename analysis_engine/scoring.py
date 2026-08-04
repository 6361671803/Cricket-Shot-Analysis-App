"""0-100 shot scoring.

Each score is a small, documented function of measured quantities rather than a
black box, so a coach can see exactly which measurement moved a score.  All ten
categories requested by the report card are produced:

Footwork, Balance, Weight Transfer, Timing, Head Position, Shot Execution,
Follow Through, Power, Control and an Overall Technique score (a weighted mean).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from .datatypes import FrameFeatures, ImpactInfo, ShotPrediction, WeightTransferResult

#: Weight of each category inside the overall technique score.
OVERALL_WEIGHTS = {
    "footwork": 0.16,
    "balance": 0.14,
    "weight_transfer": 0.14,
    "timing": 0.16,
    "head_position": 0.12,
    "shot_execution": 0.12,
    "follow_through": 0.06,
    "power": 0.08,
    "control": 0.12,
}

#: Reference peak bat speed (km/h) used to normalise the power score.
REFERENCE_BAT_SPEED_KMH = 105.0

BACK_FOOT_SHOTS = {
    "Back Foot Defence",
    "Pull Shot",
    "Hook Shot",
    "Cut Shot",
    "Square Cut",
    "Upper Cut",
    "Late Cut",
}


#: Score used when a component could not be measured (single photo, no ball
#: track): neutral, so it neither rewards nor punishes missing evidence.
NEUTRAL_SCORE = 60.0


def _num(metrics: Dict[str, object], key: str) -> Optional[float]:
    """Metric as a float, or ``None`` when the engine could not measure it."""
    value = metrics.get(key)
    if value is None:
        return None
    return float(value)


def _band(value: float, ideal_low: float, ideal_high: float, span: float) -> float:
    """100 inside the ideal band, decaying to 0 over ``span`` outside it."""
    if ideal_low <= value <= ideal_high:
        return 100.0
    distance = ideal_low - value if value < ideal_low else value - ideal_high
    return float(np.clip(100.0 - (distance / max(1e-6, span)) * 100.0, 0.0, 100.0))


def compute_scores(
    features: Sequence[FrameFeatures],
    impact: ImpactInfo,
    weight: WeightTransferResult,
    metrics: Dict[str, object],
    shot: Optional[ShotPrediction] = None,
) -> Dict[str, int]:
    """Compute the ten 0-100 scores for one shot."""
    if not metrics.get("available"):
        return {key: 0 for key in list(OVERALL_WEIGHTS) + ["overall_technique"]}

    back_foot_shot = bool(shot and shot.name in BACK_FOOT_SHOTS)
    forward = float(weight.forward_pct)

    # --- footwork: decisive weight placement, a flexed front knee, a real stride ---
    decisiveness = abs(forward - 50.0) / 50.0 * 100.0
    knee = _band(float(metrics["front_knee_flexion_deg"]), 15.0, 55.0, 45.0)
    stride = _band(float(metrics["stride_length_m"]), 0.35, 1.05, 0.5)
    pivot_deg = _num(metrics, "back_foot_pivot_deg")
    pivot = _band(pivot_deg, 5.0, 60.0, 45.0) if pivot_deg is not None else NEUTRAL_SCORE
    footwork = 0.35 * decisiveness + 0.3 * knee + 0.2 * stride + 0.15 * pivot

    balance = 0.6 * float(metrics["balance_score"]) + 0.4 * float(metrics["average_balance_score"])

    # --- weight transfer: right direction for the shot type, well timed ---
    target = (28.0, 45.0) if back_foot_shot else (58.0, 82.0)
    direction = _band(forward, target[0], target[1], 30.0)
    timing_bonus = {
        "On Time": 100.0,
        "Single Frame": 70.0,
        "Early Transfer": 55.0,
        "Late Transfer": 40.0,
    }.get(weight.timing_label, 60.0)
    quality_bonus = {
        "Excellent Transfer": 100.0,
        "Good Transfer": 80.0,
        "Average Transfer": 60.0,
        "Poor Transfer": 35.0,
    }.get(weight.quality_label, 55.0)
    weight_transfer = 0.45 * direction + 0.3 * timing_bonus + 0.25 * quality_bonus

    timing_map = {"Well Timed": 100.0, "Acceptable": 72.0, "Early": 55.0, "Late": 42.0}
    timing = 0.55 * timing_map.get(str(metrics["shot_timing"]), 60.0) + 0.45 * float(
        metrics["contact_quality_score"]
    )

    # Head stability needs several frames; a photo cannot prove a still head.
    head_stability = _num(metrics, "head_stability")
    eye_level = _num(metrics, "eye_level_stability")
    head_position = (
        0.6 * head_stability + 0.4 * eye_level
        if head_stability is not None and eye_level is not None
        else NEUTRAL_SCORE
    )

    execution = 0.45 * float(shot.confidence if shot else 50.0) + 0.55 * float(
        metrics["contact_quality_score"]
    )

    follow_ratio = _num(metrics, "follow_through_ratio")
    checked_shot = bool(shot and shot.name in {"Forward Defence", "Back Foot Defence", "Paddle Sweep", "Glance"})
    if follow_ratio is None:
        follow_through = NEUTRAL_SCORE
    elif checked_shot:
        follow_through = _band(follow_ratio, 0.0, 0.2, 0.3)
    else:
        follow_through = _band(follow_ratio, 0.4, 1.3, 0.45)

    bat_speed = _num(metrics, "bat_speed_kmh")
    if bat_speed:
        power = float(np.clip(bat_speed / REFERENCE_BAT_SPEED_KMH * 100.0, 0.0, 100.0))
    else:  # no bat-speed evidence: infer from the base the batter is hitting from
        power = 0.5 * balance + 0.5 * footwork * 0.6

    # Without a tracked ball there is no sweet-spot evidence: stay neutral
    # instead of rewarding a measurement that was never made.
    offset = _num(metrics, "sweet_spot_offset_ratio")
    sweet_spot = _band(offset, 0.0, 0.12, 0.4) if offset is not None else NEUTRAL_SCORE
    control = (
        0.4 * sweet_spot
        + 0.3 * _band(abs(float(metrics["bat_face_angle_deg"])), 0.0, 22.0, 45.0)
        + 0.3 * (head_stability if head_stability is not None else NEUTRAL_SCORE)
    )

    scores = {
        "footwork": footwork,
        "balance": balance,
        "weight_transfer": weight_transfer,
        "timing": timing,
        "head_position": head_position,
        "shot_execution": execution,
        "follow_through": follow_through,
        "power": power,
        "control": control,
    }
    overall = sum(scores[key] * OVERALL_WEIGHTS[key] for key in OVERALL_WEIGHTS)
    scores["overall_technique"] = overall
    return {key: int(round(float(np.clip(value, 0.0, 100.0)))) for key, value in scores.items()}


def score_label(score: int) -> str:
    """Coach-friendly band for a 0-100 score."""
    if score >= 85:
        return "Excellent"
    if score >= 70:
        return "Good"
    if score >= 55:
        return "Developing"
    if score >= 40:
        return "Needs Work"
    return "Poor"


__all__ = ["compute_scores", "score_label", "OVERALL_WEIGHTS", "REFERENCE_BAT_SPEED_KMH"]
