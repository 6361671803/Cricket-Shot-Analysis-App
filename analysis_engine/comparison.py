"""Comparison mode: measured shot vs an ideal professional benchmark.

The reference numbers are *biomechanical benchmarks* for each player's textbook
version of a shot (front-foot weight share, head stability, front-elbow angle,
swing plane, stride, follow-through, bat speed and balance).  They act as
coaching targets: the comparison reports your value, the benchmark, the signed
gap and a verdict per metric, plus a match percentage over all metrics.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .datatypes import ShotPrediction, WeightTransferResult

#: Metric key -> (label, unit, tolerance considered "matching").
COMPARISON_METRICS: Dict[str, tuple] = {
    "forward_weight_pct": ("Front-foot weight", "%", 10.0),
    "head_stability": ("Head stability", "/100", 12.0),
    "front_elbow_deg": ("Front elbow angle", "\u00b0", 18.0),
    "bat_swing_plane_deg": ("Bat swing plane", "\u00b0", 15.0),
    "stride_length_m": ("Stride length", " m", 0.2),
    "follow_through_ratio": ("Follow-through", " body-heights", 0.3),
    "balance_score": ("Balance", "/100", 12.0),
    "bat_speed_kmh": ("Bat speed", " km/h", 18.0),
    "back_foot_pivot_deg": ("Back-foot pivot", "\u00b0", 18.0),
    "contact_quality_score": ("Contact quality", "/100", 15.0),
}

PRO_PROFILES: Dict[str, Dict[str, object]] = {
    "Virat Kohli": {
        "style": "Front-foot dominant, wrists through the line, head absolutely still.",
        "signature": ["Cover Drive", "Straight Drive", "On Drive", "Flick"],
        "benchmarks": {
            "forward_weight_pct": 78,
            "head_stability": 93,
            "front_elbow_deg": 148,
            "bat_swing_plane_deg": 72,
            "stride_length_m": 0.88,
            "follow_through_ratio": 0.85,
            "balance_score": 91,
            "bat_speed_kmh": 92,
            "back_foot_pivot_deg": 34,
            "contact_quality_score": 88,
        },
    },
    "Steve Smith": {
        "style": "Deep in the crease, late hands, huge hip-shoulder separation.",
        "signature": ["Back Foot Defence", "Flick", "Glance", "Pull Shot"],
        "benchmarks": {
            "forward_weight_pct": 58,
            "head_stability": 90,
            "front_elbow_deg": 128,
            "bat_swing_plane_deg": 63,
            "stride_length_m": 0.62,
            "follow_through_ratio": 0.6,
            "balance_score": 88,
            "bat_speed_kmh": 86,
            "back_foot_pivot_deg": 42,
            "contact_quality_score": 86,
        },
    },
    "Babar Azam": {
        "style": "Silky timing, high front elbow, minimal head movement.",
        "signature": ["Cover Drive", "Straight Drive", "Square Cut", "Flick"],
        "benchmarks": {
            "forward_weight_pct": 74,
            "head_stability": 92,
            "front_elbow_deg": 150,
            "bat_swing_plane_deg": 74,
            "stride_length_m": 0.82,
            "follow_through_ratio": 0.9,
            "balance_score": 90,
            "bat_speed_kmh": 88,
            "back_foot_pivot_deg": 32,
            "contact_quality_score": 89,
        },
    },
    "Joe Root": {
        "style": "Compact, tight base, plays late with soft hands and busy sweeps.",
        "signature": ["Forward Defence", "Glance", "Sweep", "Reverse Sweep"],
        "benchmarks": {
            "forward_weight_pct": 70,
            "head_stability": 91,
            "front_elbow_deg": 138,
            "bat_swing_plane_deg": 68,
            "stride_length_m": 0.72,
            "follow_through_ratio": 0.55,
            "balance_score": 89,
            "bat_speed_kmh": 84,
            "back_foot_pivot_deg": 30,
            "contact_quality_score": 87,
        },
    },
    "Kane Williamson": {
        "style": "Still head, orthodox base, immaculate balance on both feet.",
        "signature": ["Forward Defence", "Straight Drive", "Cut Shot", "Late Cut"],
        "benchmarks": {
            "forward_weight_pct": 68,
            "head_stability": 94,
            "front_elbow_deg": 140,
            "bat_swing_plane_deg": 70,
            "stride_length_m": 0.75,
            "follow_through_ratio": 0.65,
            "balance_score": 92,
            "bat_speed_kmh": 83,
            "back_foot_pivot_deg": 31,
            "contact_quality_score": 88,
        },
    },
}

PRO_PLAYERS = tuple(PRO_PROFILES.keys())


def default_player(shot_name: str) -> str:
    """Pick the professional whose signature repertoire fits the shot best."""
    for player, profile in PRO_PROFILES.items():
        if shot_name in profile["signature"]:
            return player
    return "Virat Kohli"


def compare_with_professional(
    metrics: Dict[str, object],
    weight: WeightTransferResult,
    shot: Optional[ShotPrediction] = None,
    player: Optional[str] = None,
) -> Dict[str, object]:
    """Side-by-side metric comparison against a professional benchmark."""
    shot_name = shot.name if shot else "Straight Drive"
    player = player if player in PRO_PROFILES else default_player(shot_name)
    profile = PRO_PROFILES[player]
    benchmarks: Dict[str, float] = profile["benchmarks"]  # type: ignore[assignment]

    if not metrics.get("available"):
        return {
            "player": player,
            "style": profile["style"],
            "shot": shot_name,
            "rows": [],
            "match_pct": 0,
            "available": False,
            "players": list(PRO_PLAYERS),
        }

    measured = dict(metrics)
    measured.setdefault("forward_weight_pct", weight.forward_pct)

    rows: List[Dict[str, object]] = []
    matches: List[float] = []
    skipped: List[str] = []
    for key, (label, unit, tolerance) in COMPARISON_METRICS.items():
        if key not in benchmarks or key not in measured:
            continue
        # ``None`` means the engine could not measure it (e.g. bat speed from a
        # single photo); comparing that against a professional would be noise.
        try:
            mine = float(measured[key])  # type: ignore[arg-type]
        except (TypeError, ValueError):
            skipped.append(label)
            continue
        target = float(benchmarks[key])
        delta = mine - target
        closeness = float(np.clip(1.0 - abs(delta) / max(1e-6, tolerance * 2.5), 0.0, 1.0))
        matches.append(closeness)
        rows.append(
            {
                "metric": label,
                "you": f"{mine:.1f}{unit}",
                "pro": f"{target:.1f}{unit}",
                "delta": f"{delta:+.1f}{unit}",
                "verdict": (
                    "Matches the benchmark"
                    if abs(delta) <= tolerance
                    else "Below benchmark"
                    if delta < 0
                    else "Above benchmark"
                ),
                "closeness_pct": int(round(closeness * 100)),
            }
        )

    match_pct = int(round(float(np.mean(matches)) * 100)) if matches else 0
    gaps = sorted(rows, key=lambda row: row["closeness_pct"])[:3]
    return {
        "player": player,
        "style": profile["style"],
        "shot": shot_name,
        "rows": rows,
        "match_pct": match_pct,
        "available": True,
        "biggest_gaps": [f"{row['metric']}: {row['you']} vs {row['pro']}" for row in gaps],
        "not_measured": skipped,
        "players": list(PRO_PLAYERS),
    }


__all__ = [
    "COMPARISON_METRICS",
    "PRO_PLAYERS",
    "PRO_PROFILES",
    "compare_with_professional",
    "default_player",
]
