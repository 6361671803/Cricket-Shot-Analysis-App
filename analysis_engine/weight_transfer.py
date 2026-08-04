"""Weight-transfer analysis from centre-of-mass dynamics.

Weight distribution is estimated per frame (see
:func:`analysis_engine.metrics.kinematics.center_of_mass`) and then read as a
*time series* so the engine can judge not only how much weight moved but also
**when** it moved relative to contact — which is what separates an early lunge
from a well-timed transfer.

Outputs a label set (Balanced / Front Foot Dominant / Late Transfer / ...), a
per-frame series for the overlay, and an optional movement graph rendered with
matplotlib.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np

from .datatypes import FrameFeatures, ImpactInfo, WeightTransferResult
from .metrics import kinematics as kin


def analyse_weight_transfer(
    features: Sequence[FrameFeatures],
    impact: Optional[ImpactInfo] = None,
    fps: float = 25.0,
    graph_path: Optional[str] = None,
) -> WeightTransferResult:
    """Full weight-transfer report for a swing (or a single frame)."""
    result = WeightTransferResult()
    valid = [f for f in features if f.pose_detected]
    if not valid:
        result.notes.append("Pose not detected, weight transfer unavailable.")
        return result

    raw = [f.forward_pct for f in valid]
    smoothed = kin.moving_average(raw, min(5, max(1, len(raw) // 3 * 2 + 1)))
    for feature, value in zip(valid, smoothed):
        feature.forward_pct = round(float(value), 1)
        feature.back_pct = round(100.0 - float(value), 1)

    impact_position = _impact_position(valid, impact)
    forward_at_impact = float(smoothed[impact_position])
    result.forward_pct = int(round(forward_at_impact))
    result.back_pct = 100 - result.forward_pct
    result.series = [
        {
            "frame": feature.frame_index,
            "timestamp": round(feature.timestamp, 3),
            "forward_pct": feature.forward_pct,
            "back_pct": feature.back_pct,
            "com_x": None if feature.com is None else round(feature.com[0], 1),
            "com_y": None if feature.com is None else round(feature.com[1], 1),
            "front_knee_flexion": feature.front_knee_flexion,
            "back_knee_flexion": feature.back_knee_flexion,
            "heel_lift": round(feature.heel_lift, 4),
            "balance_score": feature.balance_score,
        }
        for feature in valid
    ]

    # --- distribution label ---
    if forward_at_impact >= 72:
        result.label = "Front Foot Dominant"
    elif forward_at_impact >= 58:
        result.label = "Forward Weight Transfer"
    elif forward_at_impact >= 43:
        result.label = "Balanced"
    elif forward_at_impact >= 30:
        result.label = "Backward Weight Transfer"
    else:
        result.label = "Back Foot Dominant"

    # --- timing label: where the biggest shift happened vs impact ---
    if len(smoothed) >= 4:
        deltas = np.diff(np.asarray(smoothed))
        shift_index = int(np.argmax(np.abs(deltas))) + 1
        offset_frames = shift_index - impact_position
        window = max(2.0, len(smoothed) * 0.12)
        if offset_frames < -window:
            result.timing_label = "Early Transfer"
        elif offset_frames > window:
            result.timing_label = "Late Transfer"
        else:
            result.timing_label = "On Time"
        span = max(1e-6, len(smoothed) / max(1e-6, fps))
        result.transfer_rate_pct_per_s = round(
            float((smoothed[impact_position] - smoothed[0]) / span), 1
        )
    else:
        result.timing_label = "Single Frame"

    # --- overall quality ---
    balance = float(np.mean([f.balance_score for f in valid]))
    stability = 100.0 - min(100.0, float(np.std(smoothed)) * 2.2)
    directional = 100.0 - abs(60.0 - forward_at_impact) * 1.6
    quality = float(np.clip(0.4 * directional + 0.35 * balance + 0.25 * stability, 0.0, 100.0))
    if quality >= 78 and result.timing_label in {"On Time", "Single Frame"}:
        result.quality_label = "Excellent Transfer"
    elif quality >= 62:
        result.quality_label = "Good Transfer"
    elif quality >= 45:
        result.quality_label = "Average Transfer"
    else:
        result.quality_label = "Poor Transfer"

    if result.timing_label == "Late Transfer":
        result.notes.append(
            "Your weight arrives after contact — start the front-foot stride earlier."
        )
    if result.timing_label == "Early Transfer":
        result.notes.append(
            "Weight commits before the ball arrives; hold your shape a fraction longer."
        )
    if valid[impact_position].heel_lift < 0.005 and result.forward_pct >= 60:
        result.notes.append(
            "Back heel stays flat during a front-foot shot — allow it to lift and pivot."
        )

    if graph_path:
        result.graph_path = render_weight_graph(result, impact, graph_path)
    return result


def _impact_position(valid: Sequence[FrameFeatures], impact: Optional[ImpactInfo]) -> int:
    if impact is None or impact.frame_index < 0:
        return len(valid) - 1
    for position, feature in enumerate(valid):
        if feature.frame_index >= impact.frame_index:
            return position
    return len(valid) - 1


def render_weight_graph(
    result: WeightTransferResult, impact: Optional[ImpactInfo], path: str
) -> Optional[str]:
    """Render the weight-transfer / balance movement graph as a PNG."""
    if not result.series:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - optional dependency
        return None

    frames = [point["frame"] for point in result.series]
    forward = [point["forward_pct"] for point in result.series]
    balance = [point["balance_score"] for point in result.series]

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.6, 3.2), dpi=110)
    figure.patch.set_facecolor("#042114")
    axis.set_facecolor("#042114")
    axis.plot(frames, forward, color="#ffc400", linewidth=2.2, label="Front-foot weight %")
    axis.plot(frames, balance, color="#26c6da", linewidth=1.4, alpha=0.85, label="Balance score")
    axis.axhline(50, color="#ffffff", alpha=0.25, linestyle="--", linewidth=1)
    if impact is not None and impact.frame_index in frames:
        axis.axvline(impact.frame_index, color="#ff5252", linewidth=1.8, label="Impact")
    axis.set_ylim(0, 100)
    axis.set_xlabel("Frame", color="#e0f2f1")
    axis.set_ylabel("Percent / score", color="#e0f2f1")
    axis.tick_params(colors="#b2dfdb")
    for spine in axis.spines.values():
        spine.set_color("#37474f")
    axis.legend(facecolor="#062b1b", edgecolor="#37474f", labelcolor="#e0f2f1", fontsize=8)
    axis.set_title("Weight transfer through the shot", color="#ffffff", fontsize=11)
    figure.tight_layout()
    figure.savefig(path, facecolor=figure.get_facecolor())
    plt.close(figure)
    return path


__all__ = ["analyse_weight_transfer", "render_weight_graph"]
