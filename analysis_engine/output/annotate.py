"""Video / frame annotation: skeleton, trajectories, impact and live HUD.

All drawing happens here so the analysis modules stay free of OpenCV drawing
code.  Two entry points are used by the app:

* :func:`draw_analysis` – rich overlay for an analysed frame (skeleton, bat
  path, ball path, impact marker, centre of gravity, weight-transfer arrow,
  rotation arcs, shot name and confidence);
* :func:`draw_live_hud` – the compact real-time heads-up display.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..datatypes import FrameFeatures, ImpactInfo, PoseFrame, ShotPrediction
from ..pose import landmarks as lms

WHITE = (255, 255, 255)
YELLOW = (0, 196, 255)
ORANGE = (0, 109, 255)
GREEN = (80, 220, 100)
RED = (60, 60, 255)
CYAN = (218, 198, 38)
MAGENTA = (200, 80, 220)
PANEL = (24, 20, 14)


def _point(value: Optional[Sequence[float]]) -> Optional[Tuple[int, int]]:
    if value is None:
        return None
    return (int(round(float(value[0]))), int(round(float(value[1]))))


def draw_skeleton(frame: np.ndarray, pose: Optional[PoseFrame], thickness: int = 2) -> None:
    """Draw the 33-point BlazePose skeleton."""
    if pose is None or not pose.detected:
        return
    pixels = pose.pixels
    for start, end in lms.POSE_CONNECTIONS:
        p1, p2 = _point(pixels[start]), _point(pixels[end])
        if p1 and p2:
            cv2.line(frame, p1, p2, (200, 255, 200), thickness, cv2.LINE_AA)
    for index, position in enumerate(pixels):
        radius = 4 if index in (lms.NOSE, lms.LEFT_WRIST, lms.RIGHT_WRIST) else 3
        cv2.circle(frame, _point(position), radius, (40, 180, 255), -1, cv2.LINE_AA)


def draw_path(
    frame: np.ndarray,
    path: Sequence[Dict[str, float]],
    colour: Tuple[int, int, int],
    thickness: int = 2,
    up_to_frame: Optional[int] = None,
    dotted: bool = False,
    max_points: int = 18,
) -> None:
    """Draw a tracked trajectory (bat toe or ball) as a short trailing tail.

    Only the most recent ``max_points`` samples are drawn so the overlay reads as
    a motion trail rather than a scribble over the whole clip.
    """
    points = [
        _point((item["x"], item["y"]))
        for item in path
        if up_to_frame is None or item.get("frame", 0) <= up_to_frame
    ]
    points = [p for p in points if p is not None][-max_points:]
    if len(points) < 2:
        if points:
            cv2.circle(frame, points[0], thickness + 1, colour, -1, cv2.LINE_AA)
        return
    # A tracker that jumps across the frame between two samples has changed
    # target rather than moved, so the trail is broken there instead of drawing
    # a line right across the picture.
    max_jump = 0.12 * float(np.hypot(frame.shape[1], frame.shape[0]))
    for index in range(1, len(points)):
        if dotted and index % 2 == 0:
            continue
        previous, current = points[index - 1], points[index]
        if float(np.hypot(current[0] - previous[0], current[1] - previous[1])) > max_jump:
            continue
        cv2.line(frame, previous, current, colour, thickness, cv2.LINE_AA)


def draw_rotation_arc(
    frame: np.ndarray,
    center: Optional[Sequence[float]],
    radius: int,
    angle_deg: float,
    colour: Tuple[int, int, int],
    label: str,
) -> None:
    """Draw a rotation indicator arc for the hips / shoulders."""
    origin = _point(center)
    if origin is None or radius <= 4:
        return
    sweep = float(np.clip(abs(angle_deg), 0.0, 180.0))
    cv2.ellipse(frame, origin, (radius, radius), 0, -sweep / 2.0, sweep / 2.0, colour, 2, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"{label} {angle_deg:+.0f}",
        (origin[0] + radius + 6, origin[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        colour,
        1,
        cv2.LINE_AA,
    )


def draw_weight_transfer(frame: np.ndarray, features: FrameFeatures) -> None:
    """Draw the centre of gravity plus a weight-transfer arrow along the base."""
    com = _point(features.com)
    front = _point(features.front_ankle)
    back = _point(features.back_ankle)
    if com is None:
        return
    cv2.drawMarker(frame, com, MAGENTA, cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)
    cv2.putText(frame, "COG", (com[0] + 10, com[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, MAGENTA, 1, cv2.LINE_AA)
    if front is None or back is None:
        return
    cv2.circle(frame, front, 7, GREEN, -1, cv2.LINE_AA)
    cv2.circle(frame, back, 7, RED, -1, cv2.LINE_AA)
    base = np.array(front, dtype=float) - np.array(back, dtype=float)
    tip = np.array(back, dtype=float) + base * (features.forward_pct / 100.0)
    cv2.arrowedLine(frame, back, _point(tip), YELLOW, 3, cv2.LINE_AA, tipLength=0.25)
    mid = _point((np.array(back, dtype=float) + base * 0.5))
    cv2.putText(
        frame,
        f"{features.forward_pct:.0f}% front / {features.back_pct:.0f}% back",
        (mid[0] - 60, mid[1] + 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        YELLOW,
        2,
        cv2.LINE_AA,
    )


def draw_impact_marker(frame: np.ndarray, impact: ImpactInfo, features: FrameFeatures) -> None:
    """Highlight the contact point at the impact frame."""
    position = _point(features.ball) or _point(features.bat_tip)
    if position is None:
        return
    cv2.circle(frame, position, 16, RED, 2, cv2.LINE_AA)
    cv2.circle(frame, position, 4, RED, -1, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"IMPACT f{impact.frame_index} @ {impact.timestamp:.2f}s",
        (position[0] + 20, position[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        RED,
        2,
        cv2.LINE_AA,
    )


def _panel(frame: np.ndarray, x: int, y: int, width: int, height: int, alpha: float = 0.55) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), PANEL, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _lines(frame: np.ndarray, rows: Sequence[Tuple[str, str]], x: int = 12, y: int = 12) -> None:
    width = 288
    height = 22 * len(rows) + 16
    _panel(frame, x, y, width, height)
    for index, (label, value) in enumerate(rows):
        text_y = y + 26 + index * 22
        cv2.putText(frame, label, (x + 10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, WHITE, 1, cv2.LINE_AA)
        cv2.putText(
            frame,
            str(value),
            (x + 168, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            YELLOW,
            1,
            cv2.LINE_AA,
        )


def draw_analysis(
    frame: np.ndarray,
    pose: Optional[PoseFrame],
    features: Optional[FrameFeatures],
    shot: Optional[ShotPrediction] = None,
    impact: Optional[ImpactInfo] = None,
    bat_path: Optional[Sequence[Dict[str, float]]] = None,
    ball_path: Optional[Sequence[Dict[str, float]]] = None,
    predicted_path: Optional[Sequence[Dict[str, float]]] = None,
    is_impact_frame: bool = False,
    show_panel: bool = True,
) -> np.ndarray:
    """Full analysis overlay for one frame (modifies and returns ``frame``)."""
    draw_skeleton(frame, pose)
    current_frame = None if features is None else features.frame_index
    if bat_path:
        draw_path(frame, bat_path, CYAN, 2, up_to_frame=current_frame)
    if ball_path:
        draw_path(frame, ball_path, ORANGE, 2, up_to_frame=current_frame)
    if predicted_path:
        draw_path(frame, predicted_path, (120, 200, 255), 1, dotted=True, max_points=12)

    if features is not None:
        draw_weight_transfer(frame, features)
        if features.bat_grip and features.bat_tip:
            cv2.line(frame, _point(features.bat_grip), _point(features.bat_tip), CYAN, 3, cv2.LINE_AA)
        if features.hip_center is not None:
            draw_rotation_arc(frame, features.hip_center, 34, features.hip_rotation_deg, GREEN, "HIP")
        if features.shoulder_center is not None:
            draw_rotation_arc(frame, features.shoulder_center, 46, features.shoulder_rotation_deg, CYAN, "SHD")
        if impact is not None and (is_impact_frame or impact.frame_index == features.frame_index):
            draw_impact_marker(frame, impact, features)

    if shot is not None:
        label = f"{shot.name}  {shot.confidence:.0f}%"
        cv2.putText(frame, label, (12, frame.shape[0] - 18), cv2.FONT_HERSHEY_DUPLEX, 0.78, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, label, (12, frame.shape[0] - 18), cv2.FONT_HERSHEY_DUPLEX, 0.78, YELLOW, 1, cv2.LINE_AA)

    if show_panel and features is not None:
        rows = [
            ("Weight fwd", f"{features.forward_pct:.0f}%"),
            ("Bat angle", f"{features.bat_angle_deg:.0f} deg"),
            ("Bat speed", f"{features.bat_speed_kmh:.0f} km/h"),
            ("Balance", f"{features.balance_score:.0f}/100"),
            ("Front elbow", f"{features.front_elbow_deg:.0f} deg"),
        ]
        if impact is not None and impact.frame_index >= 0:
            rows.append(("Impact frame", str(impact.frame_index)))
        _lines(frame, rows)
    return frame


def draw_live_hud(frame: np.ndarray, live: Dict[str, object]) -> np.ndarray:
    """Real-time HUD: the live metric set requested for camera mode."""
    rows: List[Tuple[str, str]] = [
        ("Shot", str(live.get("shot", "-"))),
        ("Confidence", f"{float(live.get('confidence', 0) or 0):.0f}%"),
        ("Weight transfer", f"{float(live.get('forward_pct', 0) or 0):.0f}% front"),
        ("Head stability", f"{float(live.get('head_stability', 0) or 0):.0f}/100"),
        ("Bat angle", f"{float(live.get('bat_angle_deg', 0) or 0):.0f} deg"),
        ("Backlift angle", f"{float(live.get('backlift_angle_deg', 0) or 0):.0f} deg"),
        ("Front foot angle", f"{float(live.get('front_foot_angle_deg', 0) or 0):.0f} deg"),
        ("Swing speed", f"{float(live.get('swing_speed_kmh', 0) or 0):.0f} km/h"),
        ("Impact", str(live.get("impact_prediction", "-"))),
        ("Timing", str(live.get("timing", "-"))),
        ("Footwork", str(live.get("footwork_quality", "-"))),
        ("FPS", f"{float(live.get('fps', 0) or 0):.1f}"),
    ]
    _lines(frame, rows)
    return frame


__all__ = [
    "draw_analysis",
    "draw_live_hud",
    "draw_impact_marker",
    "draw_path",
    "draw_skeleton",
    "draw_weight_transfer",
]
