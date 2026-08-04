"""Aggregation of the full technical-metric set for one shot.

Everything a coach would put on a report card is computed here from the
per-frame features plus the bat/ball trajectory summaries: bat metrics, body
rotations, footwork, stability, timing and reaction time.  The output is a flat
``dict`` of primitives so it can be rendered in templates, embedded in the PDF
report or serialised to JSON without further processing.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from ..config import CONFIG
from ..datatypes import FrameFeatures, ImpactInfo, WeightTransferResult

#: Metrics that only mean something across a swing sequence, and are therefore
#: reported as unavailable when a single photo is analysed.
SEQUENCE_ONLY_METRICS = (
    "backlift_height_ratio",
    "backlift_angle_deg",
    "bat_swing_plane_deg",
    "bat_speed_kmh",
    "swing_speed_kmh",
    "swing_arc_px",
    "follow_through_ratio",
    "back_foot_pivot_deg",
    "head_stability",
    "eye_level_stability",
    "head_travel_px",
)

#: Metrics that require the ball to have actually been tracked.
BALL_ONLY_METRICS = (
    "ball_speed_kmh",
    "ball_max_speed_kmh",
    "ball_direction_change_deg",
    "post_impact_speed_kmh",
    "reaction_time_s",
    "sweet_spot_offset_ratio",
)


def _at(features: Sequence[FrameFeatures], frame_index: int) -> FrameFeatures:
    for feature in features:
        if feature.frame_index >= frame_index:
            return feature
    return features[-1]


def _series(features: Sequence[FrameFeatures], attribute: str) -> np.ndarray:
    values = [getattr(f, attribute) for f in features]
    return np.array([float(v) for v in values if v is not None and np.isfinite(float(v))])


def compute_technical_metrics(
    features: Sequence[FrameFeatures],
    impact: ImpactInfo,
    bat_track: Dict[str, object],
    ball_track: Dict[str, object],
    weight: WeightTransferResult,
    fps: float,
) -> Dict[str, object]:
    """Build the complete technical-metric dictionary for a shot."""
    valid = [f for f in features if f.pose_detected]
    if not valid:
        return {"available": False}

    at_impact = _at(valid, impact.frame_index if impact.frame_index >= 0 else valid[-1].frame_index)
    body_height = float(np.median(_series(valid, "body_height_px")) or 1.0)

    head_positions = np.array([f.head for f in valid if f.head is not None])
    head_stability_px = float(np.std(head_positions, axis=0).mean()) if len(head_positions) > 1 else 0.0
    head_stability = float(np.clip(100.0 - (head_stability_px / max(1e-6, body_height)) * 420.0, 0.0, 100.0))
    eye_level_stability = (
        float(np.clip(100.0 - (float(np.std(head_positions[:, 1])) / max(1e-6, body_height)) * 520.0, 0.0, 100.0))
        if len(head_positions) > 1
        else 100.0
    )

    reaction_time_s = 0.0
    path = ball_track.get("path") or []
    if path and impact.frame_index >= 0 and fps > 0:
        first_seen = float(path[0]["frame"])
        reaction_time_s = max(0.0, (impact.frame_index - first_seen) / fps)

    shot_timing = _timing_label(weight, impact)

    metrics: Dict[str, object] = {
        "available": True,
        # --- bat ---
        "backlift_height_ratio": bat_track.get("backlift_height_ratio", 0.0),
        "backlift_angle_deg": bat_track.get("backlift_angle_deg", 0.0),
        "bat_swing_plane_deg": bat_track.get("swing_plane_deg", 0.0),
        "bat_swing_plane": bat_track.get("swing_plane_label", "Unknown"),
        "bat_path_points": len(bat_track.get("path") or []),
        "bat_face_angle_deg": bat_track.get("bat_face_angle_deg", 0.0),
        "bat_angle_deg": at_impact.bat_angle_deg,
        "bat_speed_kmh": bat_track.get("max_speed_kmh", 0.0),
        "swing_speed_kmh": bat_track.get("speed_kmh", 0.0),
        "swing_arc_px": bat_track.get("swing_arc_px", 0.0),
        "follow_through": bat_track.get("follow_through_label", "Unknown"),
        "follow_through_ratio": bat_track.get("follow_through_ratio", 0.0),
        "sweet_spot_offset_ratio": impact.sweet_spot_offset_ratio,
        # --- body ---
        "shoulder_rotation_deg": at_impact.shoulder_rotation_deg,
        "hip_rotation_deg": at_impact.hip_rotation_deg,
        "hip_shoulder_separation_deg": at_impact.separation_deg,
        "body_alignment_deg": at_impact.body_alignment_deg,
        "spine_lean_deg": at_impact.spine_lean_deg,
        "front_elbow_deg": at_impact.front_elbow_deg,
        "back_elbow_deg": at_impact.back_elbow_deg,
        "wrist_rotation_deg": at_impact.wrist_rotation_deg,
        # --- footwork ---
        "front_knee_flexion_deg": at_impact.front_knee_flexion,
        "back_knee_flexion_deg": at_impact.back_knee_flexion,
        "stride_length_m": at_impact.stride_length_m,
        "front_foot_direction_deg": at_impact.front_foot_direction_deg,
        "back_foot_pivot_deg": round(
            float(abs(at_impact.back_foot_pivot_deg - valid[0].back_foot_pivot_deg)), 1
        ),
        "heel_lift_ratio": round(at_impact.heel_lift, 4),
        # --- stability / balance ---
        "balance_score": at_impact.balance_score,
        "average_balance_score": round(float(np.mean(_series(valid, "balance_score")) or 0.0), 1),
        "head_stability": round(head_stability, 1),
        "head_travel_px": at_impact.head_travel_px,
        "eye_level_stability": round(eye_level_stability, 1),
        "center_of_gravity": (
            {"x": round(at_impact.com[0], 1), "y": round(at_impact.com[1], 1)}
            if at_impact.com
            else None
        ),
        # --- timing / impact ---
        "shot_timing": shot_timing,
        "impact_position": impact.impact_position,
        "impact_frame": impact.frame_index,
        "impact_timestamp_s": impact.timestamp,
        "contact_quality": impact.contact_quality,
        "contact_quality_score": impact.contact_quality_score,
        "reaction_time_s": round(reaction_time_s, 3),
        # --- ball ---
        "ball_speed_kmh": ball_track.get("speed_kmh", 0.0),
        "ball_max_speed_kmh": ball_track.get("max_speed_kmh", 0.0),
        "ball_bounce_frame": ball_track.get("bounce_frame"),
        "ball_direction_change_deg": ball_track.get("direction_change_deg", 0.0),
        "post_impact_direction_deg": ball_track.get("post_impact_direction_deg"),
        "post_impact_speed_kmh": ball_track.get("post_impact_speed_kmh", 0.0),
        "ball_tracked": bool(ball_track.get("detected")),
        # --- weight transfer summary ---
        "forward_weight_pct": weight.forward_pct,
        "backward_weight_pct": weight.back_pct,
        "weight_transfer_label": weight.label,
        "weight_transfer_timing": weight.timing_label,
        "weight_transfer_quality": weight.quality_label,
        "weight_transfer_rate_pct_per_s": weight.transfer_rate_pct_per_s,
    }
    metrics["sequence_available"] = len(valid) >= CONFIG.temporal.min_frames_for_sequence
    if not metrics["sequence_available"]:
        for key in SEQUENCE_ONLY_METRICS:
            metrics[key] = None
        metrics["bat_swing_plane"] = "Not measurable from a single frame"
        metrics["follow_through"] = "Not measurable from a single frame"
    if not metrics["ball_tracked"]:
        for key in BALL_ONLY_METRICS:
            metrics[key] = None
    metrics["display"] = _display_rows(metrics)
    return metrics


def _timing_label(weight: WeightTransferResult, impact: ImpactInfo) -> str:
    if weight.timing_label == "Late Transfer" or "Late" in impact.impact_position:
        return "Late"
    if weight.timing_label == "Early Transfer":
        return "Early"
    if "ideal" in impact.impact_position.lower() or weight.timing_label == "On Time":
        return "Well Timed"
    return "Acceptable"


def _display_rows(metrics: Dict[str, object]) -> List[Dict[str, str]]:
    """Ordered ``label``/``value`` rows for templates and the PDF report."""

    def fmt(value: object, suffix: str = "") -> str:
        if value is None:
            return "\u2014"
        if isinstance(value, float):
            return f"{value:.2f}{suffix}"
        return f"{value}{suffix}"

    def fmt_point(point: object) -> str:
        if not isinstance(point, dict):
            return "\u2014"
        return f"x {point['x']:.0f}, y {point['y']:.0f} px"

    def fmt_plane(label: object, degrees: object) -> str:
        if degrees is None:
            return fmt(label)
        return f"{label} ({fmt(degrees, chr(176))})"

    rows = [
        ("Backlift Height", fmt(metrics["backlift_height_ratio"], " body-heights")),
        ("Backlift Angle", fmt(metrics["backlift_angle_deg"], "\u00b0")),
        ("Bat Swing Plane", fmt_plane(metrics["bat_swing_plane"], metrics["bat_swing_plane_deg"])),
        ("Bat Path Samples", fmt(metrics["bat_path_points"])),
        ("Bat Face Angle", fmt(metrics["bat_face_angle_deg"], "\u00b0")),
        ("Bat Speed (peak)", fmt(metrics["bat_speed_kmh"], " km/h")),
        ("Swing Speed (median)", fmt(metrics["swing_speed_kmh"], " km/h")),
        ("Follow Through", fmt(metrics["follow_through"])),
        ("Shoulder Rotation", fmt(metrics["shoulder_rotation_deg"], "\u00b0")),
        ("Hip Rotation", fmt(metrics["hip_rotation_deg"], "\u00b0")),
        ("Hip-Shoulder Separation", fmt(metrics["hip_shoulder_separation_deg"], "\u00b0")),
        ("Head Stability", fmt(metrics["head_stability"], "/100")),
        ("Eye Level Stability", fmt(metrics["eye_level_stability"], "/100")),
        ("Front Elbow Angle", fmt(metrics["front_elbow_deg"], "\u00b0")),
        ("Bottom-hand Wrist Rotation", fmt(metrics["wrist_rotation_deg"], "\u00b0")),
        ("Front Knee Flexion", fmt(metrics["front_knee_flexion_deg"], "\u00b0")),
        ("Back Knee Flexion", fmt(metrics["back_knee_flexion_deg"], "\u00b0")),
        ("Stride Length", fmt(metrics["stride_length_m"], " m")),
        ("Front Foot Direction", fmt(metrics["front_foot_direction_deg"], "\u00b0")),
        ("Back Foot Pivot", fmt(metrics["back_foot_pivot_deg"], "\u00b0")),
        ("Balance Score", fmt(metrics["balance_score"], "/100")),
        ("Centre of Gravity", fmt_point(metrics["center_of_gravity"])),
        ("Body Alignment", fmt(metrics["body_alignment_deg"], "\u00b0")),
        ("Shot Timing", fmt(metrics["shot_timing"])),
        ("Impact Position", fmt(metrics["impact_position"])),
        ("Contact Quality", f"{metrics['contact_quality']} ({fmt(metrics['contact_quality_score'])}/100)"),
        ("Reaction Time", fmt(metrics["reaction_time_s"], " s")),
        ("Ball Speed", fmt(metrics["ball_speed_kmh"], " km/h")),
    ]
    return [{"label": label, "value": value} for label, value in rows]


__all__ = ["BALL_ONLY_METRICS", "SEQUENCE_ONLY_METRICS", "compute_technical_metrics"]
