"""Declarative biomechanical templates for the 20 supported cricket shots.

Each shot is described by the *swing signature* a coach would look for: where
the weight is, how high contact is, whether the bat travels vertically or
across the line, how far in front of the body contact happens, how the body
rotates and how the shot finishes.  Keeping this knowledge as data (rather than
nested ``if`` statements) is what lets the classifier distinguish shots that
look similar — e.g. Cut vs Square Cut vs Late Cut, or Sweep vs Paddle Sweep vs
Reverse Sweep — and explain *why* it chose one.

Feature vocabulary (all produced by :mod:`analysis_engine.classifier.swing_features`):

``forward_pct``        front-foot weight percentage at impact (0-100)
``contact_height``     contact height as a fraction of body height (0 = ground, 1 = head)
``swing_plane``        principal bat-path angle (0 = horizontal, 90 = vertical)
``contact_offset``     contact point ahead of the hips, in body heights (+ = in front)
``swing_direction``    bat travel at contact: 0 = down the ground, 90 = across the body
``follow_through``     bat travel after contact, in body heights
``hip_height``         hip height above the feet, as a fraction of body height
``body_rotation``      shoulder rotation from stance to impact (degrees)
``wrist_roll``         wrist rotation change through contact (degrees)
``bat_finish_height``  highest bat-tip height after contact, in body heights
``backlift``           peak backlift height above the shoulders, in body heights
``stride``             distance between the feet at impact (metres)
``reverse``            1 when the hands/bat face are reversed
``switch``             1 when the batter has swapped stance/handedness
"""

from __future__ import annotations

from typing import Dict, Tuple

#: ``feature -> (low, high, weight)``.  ``low``/``high`` bound the ideal band.
ShotTemplate = Dict[str, Tuple[float, float, float]]

SHOT_TEMPLATES: Dict[str, ShotTemplate] = {
    "Forward Defence": {
        "forward_pct": (62, 95, 1.4),
        "contact_height": (0.28, 0.52, 1.0),
        "swing_plane": (62, 90, 1.3),
        "contact_offset": (0.05, 0.32, 1.0),
        "swing_direction": (0, 22, 1.2),
        "follow_through": (0.0, 0.16, 1.5),
        "hip_height": (0.40, 0.55, 0.6),
        "body_rotation": (0, 18, 0.8),
        "bat_finish_height": (0.30, 0.72, 0.8),
    },
    "Back Foot Defence": {
        "forward_pct": (5, 38, 1.4),
        "contact_height": (0.32, 0.60, 1.0),
        "swing_plane": (60, 90, 1.2),
        "contact_offset": (-0.05, 0.16, 1.0),
        "swing_direction": (0, 25, 1.0),
        "follow_through": (0.0, 0.18, 1.4),
        "hip_height": (0.40, 0.56, 0.6),
        "body_rotation": (0, 22, 0.7),
    },
    "Straight Drive": {
        "forward_pct": (60, 92, 1.3),
        "contact_height": (0.22, 0.48, 1.0),
        "swing_plane": (58, 90, 1.2),
        "contact_offset": (0.12, 0.45, 1.2),
        "swing_direction": (0, 18, 1.5),
        "follow_through": (0.35, 1.10, 1.3),
        "body_rotation": (5, 30, 0.8),
        "bat_finish_height": (0.75, 1.35, 1.0),
    },
    "Cover Drive": {
        "forward_pct": (58, 92, 1.2),
        "contact_height": (0.22, 0.50, 0.9),
        "swing_plane": (52, 85, 1.1),
        "contact_offset": (0.10, 0.42, 1.1),
        "swing_direction": (20, 48, 1.6),
        "follow_through": (0.35, 1.10, 1.2),
        "body_rotation": (18, 48, 1.2),
        "bat_finish_height": (0.75, 1.35, 0.9),
    },
    "On Drive": {
        "forward_pct": (58, 92, 1.2),
        "contact_height": (0.22, 0.50, 0.9),
        "swing_plane": (56, 88, 1.1),
        "contact_offset": (0.08, 0.38, 1.0),
        "swing_direction": (12, 34, 1.4),
        "follow_through": (0.32, 1.05, 1.1),
        "body_rotation": (-10, 16, 1.4),
        "bat_finish_height": (0.70, 1.30, 0.9),
    },
    "Lofted Drive": {
        "forward_pct": (55, 92, 1.0),
        "contact_height": (0.18, 0.46, 0.9),
        "swing_plane": (58, 90, 1.1),
        "contact_offset": (0.12, 0.48, 1.0),
        "swing_direction": (0, 34, 1.0),
        "follow_through": (0.55, 1.40, 1.3),
        "bat_finish_height": (1.15, 2.00, 1.8),
        "body_rotation": (10, 45, 0.7),
    },
    "Pull Shot": {
        "forward_pct": (5, 42, 1.4),
        "contact_height": (0.52, 0.82, 1.3),
        "swing_plane": (0, 34, 1.6),
        "contact_offset": (-0.08, 0.18, 1.0),
        "swing_direction": (55, 90, 1.6),
        "follow_through": (0.40, 1.20, 1.0),
        "body_rotation": (35, 90, 1.3),
        "wrist_roll": (10, 60, 0.7),
    },
    "Hook Shot": {
        "forward_pct": (5, 40, 1.2),
        "contact_height": (0.80, 1.25, 1.8),
        "swing_plane": (0, 40, 1.3),
        "contact_offset": (-0.14, 0.12, 1.0),
        "swing_direction": (58, 90, 1.5),
        "follow_through": (0.45, 1.30, 1.0),
        "body_rotation": (45, 110, 1.4),
    },
    "Cut Shot": {
        "forward_pct": (5, 42, 1.2),
        "contact_height": (0.40, 0.70, 1.1),
        "swing_plane": (0, 32, 1.5),
        "contact_offset": (-0.16, 0.10, 1.2),
        "swing_direction": (52, 90, 1.4),
        "follow_through": (0.28, 0.95, 0.9),
        "body_rotation": (18, 60, 1.0),
        "wrist_roll": (15, 70, 1.0),
    },
    "Square Cut": {
        "forward_pct": (5, 42, 1.1),
        "contact_height": (0.42, 0.72, 1.1),
        "swing_plane": (0, 26, 1.6),
        "contact_offset": (-0.10, 0.14, 1.1),
        "swing_direction": (68, 90, 1.7),
        "follow_through": (0.30, 1.00, 0.9),
        "body_rotation": (25, 70, 1.1),
    },
    "Upper Cut": {
        "forward_pct": (5, 42, 1.1),
        "contact_height": (0.85, 1.30, 1.9),
        "swing_plane": (18, 55, 1.2),
        "contact_offset": (-0.20, 0.06, 1.2),
        "swing_direction": (48, 88, 1.2),
        "bat_finish_height": (1.20, 2.10, 1.5),
        "body_rotation": (20, 70, 0.9),
    },
    "Sweep": {
        "forward_pct": (48, 88, 0.9),
        "contact_height": (0.05, 0.30, 1.8),
        "swing_plane": (0, 30, 1.5),
        "contact_offset": (-0.05, 0.28, 0.9),
        "swing_direction": (55, 90, 1.4),
        "hip_height": (0.16, 0.40, 1.8),
        "follow_through": (0.35, 1.20, 0.9),
        "body_rotation": (25, 80, 0.9),
    },
    "Reverse Sweep": {
        "forward_pct": (45, 88, 0.8),
        "contact_height": (0.05, 0.32, 1.5),
        "swing_plane": (0, 32, 1.3),
        "swing_direction": (55, 90, 1.2),
        "hip_height": (0.18, 0.42, 1.5),
        "reverse": (1, 1, 2.6),
        "wrist_roll": (40, 140, 1.1),
    },
    "Paddle Sweep": {
        "forward_pct": (45, 85, 0.8),
        "contact_height": (0.05, 0.28, 1.6),
        "swing_plane": (0, 34, 1.0),
        "contact_offset": (-0.20, 0.08, 1.2),
        "swing_direction": (40, 85, 1.0),
        "hip_height": (0.16, 0.40, 1.6),
        "follow_through": (0.0, 0.30, 1.7),
        "bat_finish_height": (0.10, 0.55, 1.2),
    },
    "Flick": {
        "forward_pct": (48, 85, 0.9),
        "contact_height": (0.18, 0.48, 0.9),
        "swing_plane": (34, 72, 1.0),
        "contact_offset": (0.0, 0.28, 0.9),
        "swing_direction": (18, 50, 1.0),
        "wrist_roll": (45, 140, 2.0),
        "follow_through": (0.25, 0.95, 0.8),
        "body_rotation": (-16, 20, 1.1),
    },
    "Glance": {
        "forward_pct": (30, 72, 0.8),
        "contact_height": (0.28, 0.62, 0.9),
        "swing_plane": (30, 70, 0.9),
        "contact_offset": (-0.18, 0.10, 1.3),
        "swing_direction": (25, 60, 0.9),
        "follow_through": (0.0, 0.32, 1.6),
        "wrist_roll": (30, 110, 1.4),
        "bat_finish_height": (0.20, 0.70, 1.0),
    },
    "Late Cut": {
        "forward_pct": (10, 50, 0.9),
        "contact_height": (0.28, 0.58, 0.9),
        "swing_plane": (0, 34, 1.2),
        "contact_offset": (-0.42, -0.06, 2.0),
        "swing_direction": (55, 90, 1.2),
        "follow_through": (0.15, 0.70, 0.8),
        "wrist_roll": (25, 100, 1.0),
    },
    "Ramp Shot": {
        "forward_pct": (30, 78, 0.7),
        "contact_height": (0.60, 1.15, 1.4),
        "swing_plane": (24, 62, 1.0),
        "contact_offset": (-0.30, 0.02, 1.5),
        "swing_direction": (30, 80, 0.9),
        "bat_finish_height": (1.05, 1.90, 1.3),
        "follow_through": (0.05, 0.45, 1.2),
        "hip_height": (0.20, 0.46, 1.0),
    },
    "Switch Hit": {
        "switch": (1, 1, 3.0),
        "forward_pct": (45, 90, 0.7),
        "swing_plane": (30, 85, 0.7),
        "follow_through": (0.40, 1.35, 0.8),
        "body_rotation": (60, 180, 1.6),
        "bat_finish_height": (0.85, 1.90, 0.8),
    },
    "Helicopter Shot": {
        "forward_pct": (35, 82, 0.7),
        "contact_height": (0.08, 0.38, 1.1),
        "swing_plane": (40, 90, 0.9),
        "contact_offset": (-0.05, 0.30, 0.8),
        "wrist_roll": (70, 180, 2.2),
        "bat_finish_height": (1.25, 2.20, 1.7),
        "follow_through": (0.60, 1.60, 1.2),
        "body_rotation": (25, 95, 0.8),
    },
}

#: Tolerance (in feature units) applied outside each ideal band before the
#: membership score decays to zero.
FEATURE_TOLERANCE: Dict[str, float] = {
    "forward_pct": 26.0,
    "contact_height": 0.24,
    "swing_plane": 26.0,
    "contact_offset": 0.24,
    "swing_direction": 26.0,
    "follow_through": 0.36,
    "hip_height": 0.16,
    "body_rotation": 38.0,
    "wrist_roll": 55.0,
    "bat_finish_height": 0.55,
    "backlift": 0.35,
    "stride": 0.35,
    "reverse": 0.5,
    "switch": 0.5,
}

#: Human wording used when explaining which features matched.
FEATURE_PHRASES: Dict[str, str] = {
    "forward_pct": "{value:.0f}% weight on the {foot} foot",
    "contact_height": "contact at {value:.0%} of body height",
    "swing_plane": "{value:.0f}deg bat swing plane ({plane})",
    "contact_offset": "contact {offset_words}",
    "swing_direction": "bat travelling {direction_words}",
    "follow_through": "{follow_words} follow-through",
    "hip_height": "hips {hip_words}",
    "body_rotation": "{rotation:.0f}deg shoulder rotation",
    "wrist_roll": "{value:.0f}deg of wrist roll through contact",
    "bat_finish_height": "bat finishing {finish_words}",
    "backlift": "backlift {value:.2f} of body height above the shoulders",
    "stride": "{value:.2f} m stride",
    "reverse": "reversed hands / bat face",
    "switch": "switched stance",
}

#: Engine shot name -> the equivalent key in the app's ``shots.json`` library,
#: for the few shots the library stores under a different name.
SHOT_DB_ALIASES: Dict[str, str] = {
    "Flick": "Flick Shot",
    "Paddle Sweep": "Paddle Scoop",
}

SUPPORTED_SHOTS = tuple(SHOT_TEMPLATES.keys())

__all__ = [
    "FEATURE_PHRASES",
    "FEATURE_TOLERANCE",
    "SHOT_DB_ALIASES",
    "SHOT_TEMPLATES",
    "SUPPORTED_SHOTS",
    "ShotTemplate",
]
