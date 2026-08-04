"""BlazePose GHUM landmark indices, connections and small geometry helpers.

Keeping the indices in one module means the rest of the engine never depends on
a specific MediaPipe API version (``mp.solutions`` vs the Tasks API).
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

NOSE = 0
LEFT_EYE_INNER, LEFT_EYE, LEFT_EYE_OUTER = 1, 2, 3
RIGHT_EYE_INNER, RIGHT_EYE, RIGHT_EYE_OUTER = 4, 5, 6
LEFT_EAR, RIGHT_EAR = 7, 8
MOUTH_LEFT, MOUTH_RIGHT = 9, 10
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_PINKY, RIGHT_PINKY = 17, 18
LEFT_INDEX, RIGHT_INDEX = 19, 20
LEFT_THUMB, RIGHT_THUMB = 21, 22
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
LEFT_HEEL, RIGHT_HEEL = 29, 30
LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX = 31, 32

NUM_LANDMARKS = 33

#: Skeleton edges used when drawing (mirrors ``POSE_CONNECTIONS``).
POSE_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW),
    (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW),
    (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_HIP, LEFT_KNEE),
    (LEFT_KNEE, LEFT_ANKLE),
    (LEFT_ANKLE, LEFT_HEEL),
    (LEFT_HEEL, LEFT_FOOT_INDEX),
    (LEFT_ANKLE, LEFT_FOOT_INDEX),
    (RIGHT_HIP, RIGHT_KNEE),
    (RIGHT_KNEE, RIGHT_ANKLE),
    (RIGHT_ANKLE, RIGHT_HEEL),
    (RIGHT_HEEL, RIGHT_FOOT_INDEX),
    (RIGHT_ANKLE, RIGHT_FOOT_INDEX),
    (NOSE, LEFT_EYE),
    (NOSE, RIGHT_EYE),
    (LEFT_EYE, LEFT_EAR),
    (RIGHT_EYE, RIGHT_EAR),
    (LEFT_WRIST, LEFT_INDEX),
    (LEFT_WRIST, LEFT_PINKY),
    (LEFT_WRIST, LEFT_THUMB),
    (RIGHT_WRIST, RIGHT_INDEX),
    (RIGHT_WRIST, RIGHT_PINKY),
    (RIGHT_WRIST, RIGHT_THUMB),
)


def midpoint(points: np.ndarray, a: int, b: int) -> np.ndarray:
    """Midpoint between two landmarks."""
    return (points[a] + points[b]) / 2.0


def angle_deg(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
    """Interior angle at ``b`` for the joint chain ``a-b-c``, in degrees."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-6:
        return 0.0
    cos = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def line_angle_deg(p1: Sequence[float], p2: Sequence[float]) -> float:
    """Angle of the segment ``p1->p2`` in degrees, 0 = horizontal, 90 = vertical.

    Image coordinates (y down) are converted so that "up" is positive.
    """
    dx = float(p2[0]) - float(p1[0])
    dy = -(float(p2[1]) - float(p1[1]))
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return 0.0
    ang = np.degrees(np.arctan2(dy, dx))
    ang = abs(float(ang))
    if ang > 90.0:
        ang = 180.0 - ang
    return ang


def signed_line_angle_deg(p1: Sequence[float], p2: Sequence[float]) -> float:
    """Signed direction of ``p1->p2`` in degrees within (-180, 180], "up" positive."""
    dx = float(p2[0]) - float(p1[0])
    dy = -(float(p2[1]) - float(p1[1]))
    return float(np.degrees(np.arctan2(dy, dx)))


def body_height_px(pixels: np.ndarray, visibility: Optional[np.ndarray] = None) -> float:
    """Estimate the batter's on-screen height (head to lowest foot)."""
    head_y = float(min(pixels[NOSE][1], pixels[LEFT_EAR][1], pixels[RIGHT_EAR][1]))
    foot_candidates = [
        pixels[LEFT_ANKLE][1],
        pixels[RIGHT_ANKLE][1],
        pixels[LEFT_FOOT_INDEX][1],
        pixels[RIGHT_FOOT_INDEX][1],
    ]
    foot_y = float(max(foot_candidates))
    height = foot_y - head_y
    if height <= 1.0:
        # Fall back to a torso-scaled estimate when the legs are cropped out.
        torso = abs(
            float(midpoint(pixels, LEFT_HIP, RIGHT_HIP)[1])
            - float(midpoint(pixels, LEFT_SHOULDER, RIGHT_SHOULDER)[1])
        )
        height = max(1.0, torso * 3.3)
    return height


def visible(visibility: np.ndarray, indices: Iterable[int], threshold: float = 0.4) -> bool:
    """True when every requested landmark is visible enough to trust."""
    return all(float(visibility[i]) >= threshold for i in indices)


__all__ = [name for name in dir() if name.isupper() or name.islower()]
