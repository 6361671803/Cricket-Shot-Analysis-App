"""Temporal shot-classification subpackage."""

from .shot_classifier import classify_shot, explain, score_shots
from .shot_definitions import SHOT_DB_ALIASES, SHOT_TEMPLATES, SUPPORTED_SHOTS
from .swing_features import build_swing_features

__all__ = [
    "SHOT_DB_ALIASES",
    "SHOT_TEMPLATES",
    "SUPPORTED_SHOTS",
    "build_swing_features",
    "classify_shot",
    "explain",
    "score_shots",
]
