"""Central configuration for the cricket analysis engine.

Every tunable constant used by the pipeline lives here so the engine can be
re-tuned without touching algorithmic code.  Values may be overridden through
environment variables (useful for weaker machines / real-time mode).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class PoseConfig:
    """Pose estimation settings (BlazePose GHUM / MediaPipe Pose Landmarker)."""

    model_complexity: int = _env_int("CRICKET_POSE_COMPLEXITY", 1)
    realtime_model_complexity: int = _env_int("CRICKET_POSE_COMPLEXITY_RT", 0)
    min_detection_confidence: float = _env_float("CRICKET_POSE_MIN_DET", 0.5)
    min_tracking_confidence: float = _env_float("CRICKET_POSE_MIN_TRK", 0.5)
    #: Optional path to a `.task` bundle enabling the modern Pose Landmarker API.
    landmarker_task_path: str = os.environ.get(
        "CRICKET_POSE_TASK", os.path.join("models", "pose_landmarker_full.task")
    )
    use_gpu: bool = _env_bool("CRICKET_USE_GPU", False)


@dataclass(frozen=True)
class TrackingConfig:
    """Bat / ball tracking settings."""

    # --- ball ---
    ball_min_radius_px: int = 3
    ball_max_radius_px: int = 26
    ball_min_area_px: int = 12
    ball_diff_threshold: int = 22
    ball_max_jump_px: int = 190
    ball_max_missing_frames: int = 8
    # --- bat ---
    bat_roi_scale: float = 2.6  # ROI size around the grip, in shoulder widths
    bat_min_line_len_ratio: float = 0.25  # of ROI diagonal
    bat_canny_low: int = 55
    bat_canny_high: int = 165
    bat_smoothing: float = 0.55  # exponential smoothing on the bat tip
    bat_length_shoulder_ratio: float = 2.1  # fallback bat length estimate


@dataclass(frozen=True)
class BiomechConfig:
    """Body / biomechanics assumptions."""

    assumed_body_height_m: float = _env_float("CRICKET_BODY_HEIGHT_M", 1.75)
    #: Fractional body-mass of each segment used for centre-of-mass estimation.
    segment_mass: dict = field(
        default_factory=lambda: {
            "head": 0.081,
            "trunk": 0.497,
            "upper_arm": 0.028,
            "forearm": 0.022,
            "thigh": 0.100,
            "shank": 0.0465,
            "foot": 0.0145,
        }
    )
    head_stability_good_px_ratio: float = 0.035  # of body height


@dataclass(frozen=True)
class TemporalConfig:
    """Temporal-window settings for swing-sequence analysis."""

    max_history: int = _env_int("CRICKET_HISTORY", 240)
    smoothing_window: int = 5
    velocity_window: int = 3
    #: frames analysed before/after impact when describing the swing
    pre_impact_window: int = 18
    post_impact_window: int = 14
    min_frames_for_sequence: int = 6


@dataclass(frozen=True)
class RealtimeConfig:
    """Live-camera settings."""

    target_fps: int = _env_int("CRICKET_TARGET_FPS", 25)
    capture_width: int = _env_int("CRICKET_CAM_WIDTH", 960)
    capture_height: int = _env_int("CRICKET_CAM_HEIGHT", 540)
    jpeg_quality: int = _env_int("CRICKET_JPEG_QUALITY", 72)
    #: Analyse only every Nth frame when the worker cannot keep up.
    adaptive_skip: bool = True
    max_skip: int = 4
    idle_timeout_s: float = _env_float("CRICKET_CAM_IDLE_TIMEOUT", 90.0)
    #: Consecutive failed reads tolerated before the source is declared finished
    #: (a camera drops the odd frame; a video file simply ends).
    max_read_misses: int = _env_int("CRICKET_CAM_READ_MISSES", 25)


@dataclass(frozen=True)
class EngineConfig:
    pose: PoseConfig = field(default_factory=PoseConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    biomech: BiomechConfig = field(default_factory=BiomechConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    realtime: RealtimeConfig = field(default_factory=RealtimeConfig)
    #: Cap on frames analysed for an uploaded video (protects the web worker).
    max_video_frames: int = _env_int("CRICKET_MAX_VIDEO_FRAMES", 900)
    worker_threads: int = _env_int("CRICKET_WORKERS", 2)


CONFIG = EngineConfig()

__all__ = [
    "CONFIG",
    "EngineConfig",
    "PoseConfig",
    "TrackingConfig",
    "BiomechConfig",
    "TemporalConfig",
    "RealtimeConfig",
]
