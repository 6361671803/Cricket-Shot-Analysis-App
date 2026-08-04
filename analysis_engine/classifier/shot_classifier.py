"""Temporal, explainable shot classifier for 20 cricket strokes.

Scoring works in three steps:

1. **Fuzzy membership** – every template feature contributes a score in
   ``[0, 1]``: ``1`` inside the ideal band, decaying linearly across the
   feature's tolerance outside it.  Weighted mean = raw shot score.
2. **Softmax confidence** – raw scores are converted into a probability
   distribution so similar shots (Cut vs Square Cut) split confidence honestly
   instead of both reporting "95 %".
3. **Explanation** – the highest-weighted matching features of the winner are
   rendered into a sentence, so every prediction ships with its reason.

Confidence is finally damped by *data quality* (pose/ball/bat availability and
sequence length), because a one-frame photo cannot justify the same certainty
as a tracked 60-frame swing.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..datatypes import ShotPrediction
from .shot_definitions import (
    FEATURE_PHRASES,
    FEATURE_TOLERANCE,
    SHOT_TEMPLATES,
    SUPPORTED_SHOTS,
)


def _membership(value: float, low: float, high: float, tolerance: float) -> float:
    if low <= value <= high:
        return 1.0
    distance = low - value if value < low else value - high
    if tolerance <= 0:
        return 0.0
    return float(max(0.0, 1.0 - distance / tolerance))


def score_shots(vector: Dict[str, float]) -> Dict[str, float]:
    """Raw ``[0, 1]`` match score for every supported shot."""
    scores: Dict[str, float] = {}
    for shot, template in SHOT_TEMPLATES.items():
        total_weight = 0.0
        accumulated = 0.0
        for feature, (low, high, weight) in template.items():
            if feature not in vector:
                continue
            tolerance = FEATURE_TOLERANCE.get(feature, 1.0)
            accumulated += weight * _membership(vector[feature], low, high, tolerance)
            total_weight += weight
        scores[shot] = accumulated / total_weight if total_weight else 0.0
    return scores


def _phrase(feature: str, vector: Dict[str, float]) -> str:
    value = vector.get(feature, 0.0)
    template = FEATURE_PHRASES.get(feature, feature)
    context = {
        "value": value,
        "foot": "front" if vector.get("forward_pct", 50) >= 50 else "back",
        "plane": (
            "vertical"
            if vector.get("swing_plane", 45) >= 55
            else "diagonal"
            if vector.get("swing_plane", 45) >= 30
            else "horizontal"
        ),
        "offset_words": (
            "well in front of the body"
            if vector.get("contact_offset", 0) > 0.18
            else "in front of the hips"
            if vector.get("contact_offset", 0) > 0.05
            else "beside the body"
            if vector.get("contact_offset", 0) > -0.06
            else "late, behind the hips"
        ),
        "direction_words": (
            "straight down the ground"
            if vector.get("swing_direction", 45) < 22
            else "through the off/on side"
            if vector.get("swing_direction", 45) < 55
            else "across the line of the ball"
        ),
        "follow_words": (
            "a full"
            if vector.get("follow_through", 0) >= 0.45
            else "a controlled"
            if vector.get("follow_through", 0) >= 0.2
            else "a checked"
        ),
        "hip_words": (
            "dropped low (crouched)" if vector.get("hip_height", 0.5) < 0.42 else "upright"
        ),
        "rotation": abs(vector.get("body_rotation", 0.0)),
        "finish_words": (
            "high above the shoulders"
            if vector.get("bat_finish_height", 0) >= 1.1
            else "at shoulder height"
            if vector.get("bat_finish_height", 0) >= 0.7
            else "low"
        ),
    }
    try:
        return template.format(**context)
    except (KeyError, IndexError, ValueError):
        return feature.replace("_", " ")


def explain(shot: str, vector: Dict[str, float], max_reasons: int = 3) -> str:
    """Human-readable justification for one shot prediction."""
    template = SHOT_TEMPLATES.get(shot, {})
    contributions: List[Tuple[float, str]] = []
    for feature, (low, high, weight) in template.items():
        if feature not in vector:
            continue
        tolerance = FEATURE_TOLERANCE.get(feature, 1.0)
        match = _membership(vector[feature], low, high, tolerance)
        if match >= 0.6:
            contributions.append((weight * match, _phrase(feature, vector)))
    contributions.sort(reverse=True, key=lambda item: item[0])
    if not contributions:
        return f"Closest match to a {shot} from the overall swing shape."
    reasons = [text for _, text in contributions[:max_reasons]]
    return f"Detected {shot}: " + ", ".join(reasons) + "."


def data_quality(
    frames_analysed: int,
    pose_frames: int,
    ball_detected: bool,
    bat_from_image: bool,
) -> float:
    """Confidence multiplier in ``[0.35, 1.0]`` reflecting available evidence."""
    quality = 0.4
    quality += 0.25 * min(1.0, pose_frames / 24.0)
    quality += 0.15 if ball_detected else 0.0
    quality += 0.10 if bat_from_image else 0.0
    quality += 0.10 * min(1.0, frames_analysed / 40.0)
    return float(np.clip(quality, 0.35, 1.0))


def classify_shot(
    vector: Dict[str, float],
    frames_analysed: int = 1,
    pose_frames: int = 1,
    ball_detected: bool = False,
    bat_from_image: bool = False,
    prior: Optional[Dict[str, float]] = None,
    temperature: float = 0.085,
) -> ShotPrediction:
    """Classify a swing feature vector into one of the 20 supported shots."""
    if not vector:
        return ShotPrediction(
            name="Unknown",
            confidence=0.0,
            reason="Pose could not be detected, so the shot could not be classified.",
        )

    scores = score_shots(vector)
    if prior:
        for shot in scores:
            scores[shot] *= float(prior.get(shot, 1.0))

    names = list(scores.keys())
    values = np.array([scores[name] for name in names], dtype=float)
    exponent = np.exp((values - values.max()) / max(1e-6, temperature))
    probabilities = exponent / exponent.sum()

    order = np.argsort(-probabilities)
    best = names[int(order[0])]
    quality = data_quality(frames_analysed, pose_frames, ball_detected, bat_from_image)
    confidence = float(np.clip(probabilities[int(order[0])] * quality, 0.0, 1.0))

    alternatives = [
        (names[int(i)], round(float(probabilities[int(i)] * quality * 100.0), 1))
        for i in order[1:4]
    ]
    return ShotPrediction(
        name=best,
        confidence=round(confidence * 100.0, 1),
        reason=explain(best, vector),
        scores={name: round(float(scores[name]), 3) for name in names},
        alternatives=alternatives,
    )


__all__ = ["classify_shot", "explain", "score_shots", "data_quality", "SUPPORTED_SHOTS"]
