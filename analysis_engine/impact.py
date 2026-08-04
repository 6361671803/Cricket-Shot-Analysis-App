"""Impact-frame detection through multi-cue temporal fusion.

Rather than guessing the impact from a single frame, every frame of the swing
receives an *impact likelihood* built from five independent cues:

===========================  ======  ============================================
cue                          weight  intuition
===========================  ======  ============================================
bat / ball proximity          0.30   contact happens where the blade meets the ball
bat velocity peak             0.25   the bat is fastest at (or just before) contact
ball direction change         0.20   the ball reverses/deflects only at contact
bat deceleration              0.15   energy transfer slows the blade abruptly
whole-body motion energy      0.10   the swing's kinematic peak brackets contact
===========================  ======  ============================================

The frame with the highest fused likelihood is the impact frame; the winning
cues are reported as human-readable evidence.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .datatypes import BallObservation, BatObservation, FrameFeatures, ImpactInfo

WEIGHTS = {
    "proximity": 0.30,
    "bat_velocity": 0.25,
    "ball_deflection": 0.20,
    "bat_deceleration": 0.15,
    "motion": 0.10,
}
#: Sweet spot sits ~72 % down the blade from the grip.
SWEET_SPOT_RATIO = 0.72


def _normalise(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return array
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    spread = float(array.max() - array.min())
    if spread < 1e-9:
        return np.zeros_like(array)
    return (array - array.min()) / spread


def _ball_deflection_series(balls: Sequence[BallObservation]) -> List[float]:
    """Per-frame angular change of the ball's direction of travel (degrees)."""
    out = [0.0] * len(balls)
    known = [(i, np.array(b.position)) for i, b in enumerate(balls) if b.position is not None]
    for k in range(1, len(known) - 1):
        (_, p_prev), (i_cur, p_cur), (_, p_next) = known[k - 1], known[k], known[k + 1]
        incoming = p_cur - p_prev
        outgoing = p_next - p_cur
        n_in = float(np.linalg.norm(incoming))
        n_out = float(np.linalg.norm(outgoing))
        if n_in < 1e-3 or n_out < 1e-3:
            continue
        cos = float(np.clip(np.dot(incoming, outgoing) / (n_in * n_out), -1.0, 1.0))
        out[i_cur] = float(np.degrees(np.arccos(cos)))
    return out


def detect_impact(
    features: Sequence[FrameFeatures],
    bat_observations: Sequence[BatObservation],
    ball_observations: Sequence[BallObservation],
    fps: float,
) -> ImpactInfo:
    """Locate the impact frame and describe the contact."""
    if not features:
        return ImpactInfo(reason="No frames were analysed.")

    bat_speed = [f.bat_speed_kmh for f in features]
    motion = [f.motion_energy for f in features]
    distances = [f.bat_ball_distance_px for f in features]
    body_scale = float(
        np.median([f.body_height_px for f in features if f.body_height_px > 0] or [1.0])
    )

    finite = [d for d in distances if np.isfinite(d)]
    if finite:
        # Proximity: 1 when the ball touches the blade, decaying over 25 % of body height.
        span = max(1.0, body_scale * 0.25)
        proximity = np.array(
            [float(np.exp(-d / span)) if np.isfinite(d) else 0.0 for d in distances]
        )
    else:
        proximity = np.zeros(len(features))

    velocity = _normalise(bat_speed)
    deceleration = np.zeros(len(features))
    for i in range(1, len(bat_speed) - 1):
        deceleration[i] = max(0.0, bat_speed[i] - bat_speed[i + 1])
    deceleration = _normalise(deceleration)
    deflection = _normalise(_ball_deflection_series(ball_observations)) if ball_observations else np.zeros(len(features))
    if len(deflection) != len(features):
        deflection = np.resize(deflection, len(features))
    motion_n = _normalise(motion)

    total = (
        WEIGHTS["proximity"] * proximity
        + WEIGHTS["bat_velocity"] * velocity
        + WEIGHTS["ball_deflection"] * deflection
        + WEIGHTS["bat_deceleration"] * deceleration
        + WEIGHTS["motion"] * motion_n
    )
    # Impact cannot be in the very first frames (the backlift is not a contact).
    if len(total) > 6:
        total[:2] *= 0.35
    best = int(np.argmax(total))
    feature = features[best]

    evidence = {
        "proximity": round(float(proximity[best]), 3),
        "bat_velocity": round(float(velocity[best]), 3),
        "ball_deflection": round(float(deflection[best]), 3),
        "bat_deceleration": round(float(deceleration[best]), 3),
        "motion": round(float(motion_n[best]), 3),
    }
    dominant = max(evidence.items(), key=lambda item: item[1] * WEIGHTS[item[0]])[0]
    reason_map = {
        "proximity": "ball was closest to the bat blade",
        "bat_velocity": "bat reached its maximum velocity",
        "ball_deflection": "ball changed direction sharply",
        "bat_deceleration": "bat decelerated abruptly (energy transfer)",
        "motion": "whole-body motion peaked",
    }

    ball_speed = _impact_ball_speed(features, best)
    info = ImpactInfo(
        frame_index=feature.frame_index,
        timestamp=round(feature.timestamp, 3),
        confidence=round(float(np.clip(total[best] / max(1e-6, sum(WEIGHTS.values())), 0.0, 1.0)), 3),
        bat_speed_kmh=round(float(np.max(bat_speed[max(0, best - 2) : best + 2] or [0.0])), 1),
        ball_speed_kmh=ball_speed,
        bat_ball_distance_px=(
            round(float(distances[best]), 1) if np.isfinite(distances[best]) else -1.0
        ),
        evidence=evidence,
        reason=f"Impact at frame {feature.frame_index}: {reason_map[dominant]}.",
    )
    _describe_contact(info, features, bat_observations, best, body_scale)
    return info


def _impact_ball_speed(features: Sequence[FrameFeatures], index: int) -> float:
    window = [
        f.ball_speed_kmh
        for f in features[max(0, index - 4) : index + 1]
        if f.ball_speed_kmh > 0
    ]
    return round(float(np.median(window)), 1) if window else 0.0


def _describe_contact(
    info: ImpactInfo,
    features: Sequence[FrameFeatures],
    bat_observations: Sequence[BatObservation],
    index: int,
    body_scale: float,
) -> None:
    """Fill in contact quality, sweet-spot offset and impact position."""
    feature = features[index]
    scores: List[float] = []
    ball_evidence = feature.ball is not None

    # 1) how close to the sweet spot along the blade
    if feature.ball and feature.bat_grip and feature.bat_tip:
        grip = np.array(feature.bat_grip)
        tip = np.array(feature.bat_tip)
        blade = tip - grip
        length_sq = float(np.dot(blade, blade))
        if length_sq > 1e-6:
            t = float(np.clip(np.dot(np.array(feature.ball) - grip, blade) / length_sq, 0.0, 1.5))
            info.sweet_spot_offset_ratio = round(abs(t - SWEET_SPOT_RATIO), 3)
            scores.append(float(np.clip(1.0 - info.sweet_spot_offset_ratio / 0.5, 0.0, 1.0)))
    # 2) blade-to-ball distance
    if info.bat_ball_distance_px >= 0 and body_scale > 0:
        scores.append(
            float(np.clip(1.0 - info.bat_ball_distance_px / (body_scale * 0.12), 0.0, 1.0))
        )
    # 3) energy transfer: quick ball off a fast bat
    if info.bat_speed_kmh > 0 and info.ball_speed_kmh > 0:
        scores.append(float(np.clip(info.ball_speed_kmh / (info.bat_speed_kmh * 1.4), 0.0, 1.0)))
    # 4) a still head at contact means a controlled strike
    if body_scale > 0:
        scores.append(
            float(np.clip(1.0 - feature.head_travel_px / (body_scale * 0.35), 0.0, 1.0))
        )

    quality = float(np.mean(scores)) * 100.0 if scores else 0.0
    if not ball_evidence:
        # Without a tracked ball the only cue is body stability, so the estimate
        # is deliberately damped and labelled as such rather than reported as a
        # confident "middled" contact.
        quality *= 0.7
    info.contact_quality_score = round(quality, 1)
    info.contact_quality = (
        "Middled (sweet spot)"
        if quality >= 78
        else "Well struck"
        if quality >= 62
        else "Off-centre"
        if quality >= 42
        else "Edged / mistimed"
        if quality > 0
        else "Unknown"
    )
    if not ball_evidence and quality > 0:
        info.contact_quality += " (estimated, ball not tracked)"

    # Impact position relative to the body, in the batter's facing direction.
    if feature.ball and feature.hip_center and feature.front_ankle:
        direction = 1.0 if feature.front_ankle[0] >= feature.hip_center[0] else -1.0
        offset = (feature.ball[0] - feature.hip_center[0]) * direction / max(1.0, body_scale)
        info.impact_position = (
            "Well in front of the body"
            if offset > 0.18
            else "Under the eyes (ideal)"
            if offset > 0.05
            else "Beside the body"
            if offset > -0.06
            else "Late, behind the body"
        )
    elif feature.bat_tip and feature.hip_center:
        info.impact_position = "Estimated from bat position (ball not tracked)"


__all__ = ["detect_impact", "SWEET_SPOT_RATIO", "WEIGHTS"]
