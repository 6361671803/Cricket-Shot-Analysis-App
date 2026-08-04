"""Shared dataclasses passed between the engine's modules.

The pipeline is deliberately built around small, serialisable value objects so
each stage (pose -> tracking -> temporal features -> classification -> scoring)
can be tested, replaced or reused in isolation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class PoseFrame:
    """A single frame of 2D + 3D pose data (33 BlazePose GHUM landmarks)."""

    index: int
    timestamp: float
    #: (33, 2) landmark positions in pixels.
    pixels: np.ndarray
    #: (33, 3) metric world landmarks in metres, origin at the hip centre.
    world: np.ndarray
    #: (33,) per-landmark visibility in [0, 1].
    visibility: np.ndarray
    width: int
    height: int

    @property
    def detected(self) -> bool:
        return self.pixels is not None and len(self.pixels) > 0


@dataclass
class BallObservation:
    """Ball detection for one frame."""

    frame_index: int
    timestamp: float
    position: Optional[Tuple[float, float]]  # pixels
    radius: float = 0.0
    confidence: float = 0.0
    interpolated: bool = False


@dataclass
class BatObservation:
    """Bat pose for one frame: grip (handle) and tip (toe of the bat)."""

    frame_index: int
    timestamp: float
    grip: Optional[Tuple[float, float]]
    tip: Optional[Tuple[float, float]]
    angle_deg: float = 0.0  # 0 = horizontal, 90 = vertical (upright)
    length_px: float = 0.0
    confidence: float = 0.0
    from_image_evidence: bool = False


@dataclass
class FrameFeatures:
    """Per-frame biomechanical feature vector used by every downstream stage."""

    frame_index: int
    timestamp: float
    pose_detected: bool = False

    # --- global scale / geometry ---
    px_per_meter: float = 0.0
    body_height_px: float = 0.0

    # --- weight transfer / balance ---
    forward_pct: float = 50.0
    back_pct: float = 50.0
    com: Optional[Tuple[float, float]] = None
    hip_center: Optional[Tuple[float, float]] = None
    shoulder_center: Optional[Tuple[float, float]] = None
    head: Optional[Tuple[float, float]] = None
    front_ankle: Optional[Tuple[float, float]] = None
    back_ankle: Optional[Tuple[float, float]] = None
    front_knee_flexion: float = 0.0
    back_knee_flexion: float = 0.0
    heel_lift: float = 0.0
    balance_score: float = 0.0
    stride_length_m: float = 0.0

    # --- rotation / alignment ---
    shoulder_rotation_deg: float = 0.0
    hip_rotation_deg: float = 0.0
    separation_deg: float = 0.0  # shoulder-hip separation ("X-factor")
    spine_lean_deg: float = 0.0
    body_alignment_deg: float = 0.0

    # --- arms / bat ---
    front_elbow_deg: float = 0.0
    back_elbow_deg: float = 0.0
    wrist_rotation_deg: float = 0.0
    bat_angle_deg: float = 0.0
    bat_tip: Optional[Tuple[float, float]] = None
    bat_grip: Optional[Tuple[float, float]] = None
    bat_speed_kmh: float = 0.0
    bat_height_ratio: float = 0.0  # bat tip height relative to body (1 = head)
    hands_height_ratio: float = 0.0

    # --- ball ---
    ball: Optional[Tuple[float, float]] = None
    ball_speed_kmh: float = 0.0
    bat_ball_distance_px: float = 0.0

    # --- misc ---
    motion_energy: float = 0.0
    head_travel_px: float = 0.0
    facing: str = "unknown"  # "right"/"left" handed batter
    front_foot_direction_deg: float = 0.0
    back_foot_pivot_deg: float = 0.0
    #: Signed angle of the left-wrist -> right-wrist axis (detects reversed hands).
    hands_axis_deg: float = 0.0
    #: +1 when the front foot is to the right of the hips, -1 otherwise (detects switch hits).
    facing_sign: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for key, value in self.__dict__.items():
            if isinstance(value, (int, float, str, bool)) or value is None:
                out[key] = value
        return out


@dataclass
class ImpactInfo:
    """Result of impact-frame detection."""

    frame_index: int = -1
    timestamp: float = 0.0
    confidence: float = 0.0
    bat_speed_kmh: float = 0.0
    ball_speed_kmh: float = 0.0
    contact_quality: str = "Unknown"
    contact_quality_score: float = 0.0
    bat_ball_distance_px: float = 0.0
    sweet_spot_offset_ratio: float = 0.0
    impact_position: str = "Unknown"
    evidence: Dict[str, float] = field(default_factory=dict)
    reason: str = ""


@dataclass
class ShotPrediction:
    """Classifier output for one candidate shot."""

    name: str
    confidence: float
    reason: str
    scores: Dict[str, float] = field(default_factory=dict)
    alternatives: List[Tuple[str, float]] = field(default_factory=list)


@dataclass
class WeightTransferResult:
    forward_pct: int = 50
    back_pct: int = 50
    label: str = "Balanced"
    timing_label: str = "On Time"
    quality_label: str = "Average Transfer"
    transfer_rate_pct_per_s: float = 0.0
    series: List[Dict[str, float]] = field(default_factory=list)
    graph_path: Optional[str] = None
    notes: List[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    """Everything the engine knows about one delivery / shot."""

    shot: ShotPrediction
    impact: ImpactInfo
    weight_transfer: WeightTransferResult
    metrics: Dict[str, object] = field(default_factory=dict)
    scores: Dict[str, int] = field(default_factory=dict)
    coaching: Dict[str, List[str]] = field(default_factory=dict)
    ball_track: List[Dict[str, float]] = field(default_factory=list)
    bat_track: List[Dict[str, float]] = field(default_factory=list)
    frames_analysed: int = 0
    fps: float = 0.0
    source: str = "image"
    annotated_image_path: Optional[str] = None
    annotated_video_path: Optional[str] = None
    graph_path: Optional[str] = None
    comparison: Dict[str, object] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        """JSON-serialisable view (used by the ``/api`` endpoints and reports)."""
        return {
            "shot": asdict(self.shot),
            "impact": asdict(self.impact),
            "weight_transfer": asdict(self.weight_transfer),
            "metrics": self.metrics,
            "scores": self.scores,
            "coaching": self.coaching,
            "comparison": self.comparison,
            "ball_track": self.ball_track,
            "bat_track": self.bat_track,
            "frames_analysed": self.frames_analysed,
            "fps": self.fps,
            "source": self.source,
            "annotated_image_path": self.annotated_image_path,
            "annotated_video_path": self.annotated_video_path,
            "graph_path": self.graph_path,
            "warnings": self.warnings,
        }


__all__ = [
    "AnalysisResult",
    "BallObservation",
    "BatObservation",
    "FrameFeatures",
    "ImpactInfo",
    "PoseFrame",
    "ShotPrediction",
    "WeightTransferResult",
]
