"""Per-frame feature extraction: pose + bat + ball -> :class:`FrameFeatures`.

This is the single place where raw detections are turned into the
biomechanical vocabulary the rest of the engine speaks (weight transfer,
rotations, bat geometry, balance...).  Every downstream module — shot
classifier, impact detector, scoring, coaching, overlays — consumes only
``FrameFeatures``, which is what makes the pipeline modular.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .config import CONFIG
from .datatypes import BallObservation, BatObservation, FrameFeatures, PoseFrame
from .metrics import kinematics as kin
from .pose import landmarks as lms


def _tuple(point: np.ndarray) -> Tuple[float, float]:
    return (float(point[0]), float(point[1]))


class FeatureExtractor:
    """Stateful extractor (needs the previous frame for velocities/energy)."""

    def __init__(self) -> None:
        self._prev_pixels: Optional[np.ndarray] = None
        self._prev_bat_tip: Optional[np.ndarray] = None
        self._prev_ball: Optional[np.ndarray] = None
        self._stance_head: Optional[np.ndarray] = None
        self._fps: float = 25.0
        self.facing: str = "unknown"

    def set_fps(self, fps: float) -> None:
        self._fps = fps if fps and fps > 0 else 25.0

    # ------------------------------------------------------------------ public
    def extract(
        self,
        pose: Optional[PoseFrame],
        bat: Optional[BatObservation],
        ball: Optional[BallObservation],
        frame_index: int,
        timestamp: float,
    ) -> FrameFeatures:
        features = FrameFeatures(frame_index=frame_index, timestamp=timestamp)
        if pose is None or not pose.detected:
            self._prev_pixels = None
            return features

        pixels = pose.pixels
        world = pose.world
        features.pose_detected = True
        features.body_height_px = lms.body_height_px(pixels, pose.visibility)
        features.px_per_meter = features.body_height_px / CONFIG.biomech.assumed_body_height_m

        # ---------------- geometry anchors ----------------
        head = (pixels[lms.NOSE] + pixels[lms.LEFT_EAR] + pixels[lms.RIGHT_EAR]) / 3.0
        hip_center = lms.midpoint(pixels, lms.LEFT_HIP, lms.RIGHT_HIP)
        shoulder_center = lms.midpoint(pixels, lms.LEFT_SHOULDER, lms.RIGHT_SHOULDER)
        features.head = _tuple(head)
        features.hip_center = _tuple(hip_center)
        features.shoulder_center = _tuple(shoulder_center)

        front_side, back_side = self._front_back_side(pixels)
        features.facing = self.facing = "right" if front_side == "left" else "left"
        front_ankle = pixels[lms.LEFT_ANKLE if front_side == "left" else lms.RIGHT_ANKLE]
        back_ankle = pixels[lms.LEFT_ANKLE if back_side == "left" else lms.RIGHT_ANKLE]
        features.front_ankle = _tuple(front_ankle)
        features.back_ankle = _tuple(back_ankle)

        # ---------------- weight transfer ----------------
        com = kin.center_of_mass(pixels)
        features.com = _tuple(com)
        base_front = np.array([front_ankle[0], front_ankle[1]])
        base_back = np.array([back_ankle[0], back_ankle[1]])
        forward = kin.foot_line_projection(np.array([com[0], com[1]]), base_back, base_front)

        # Refine with joint evidence: a loaded leg is flexed, an unloaded heel lifts.
        features.front_knee_flexion = kin.knee_flexion(pixels, front_side)
        features.back_knee_flexion = kin.knee_flexion(pixels, back_side)
        features.heel_lift = kin.heel_lift_ratio(pixels, back_side, features.body_height_px)
        flexion_bias = np.clip(
            (features.front_knee_flexion - features.back_knee_flexion) / 120.0, -0.25, 0.25
        )
        heel_bias = np.clip(features.heel_lift * 4.0, 0.0, 0.2)
        forward = float(np.clip(0.72 * forward + 0.18 * (0.5 + flexion_bias) + 0.10 * (0.5 + heel_bias), 0.0, 1.0))
        features.forward_pct = round(forward * 100.0, 1)
        features.back_pct = round(100.0 - features.forward_pct, 1)

        # ---------------- rotations & alignment ----------------
        if world.size:
            features.shoulder_rotation_deg = round(
                kin.rotation_about_vertical(world, lms.LEFT_SHOULDER, lms.RIGHT_SHOULDER), 1
            )
            features.hip_rotation_deg = round(
                kin.rotation_about_vertical(world, lms.LEFT_HIP, lms.RIGHT_HIP), 1
            )
            features.separation_deg = round(
                abs(features.shoulder_rotation_deg - features.hip_rotation_deg), 1
            )
            stride = world[lms.LEFT_ANKLE] - world[lms.RIGHT_ANKLE]
            features.stride_length_m = round(
                float(np.linalg.norm([stride[0], stride[2]])), 3
            )
        features.spine_lean_deg = round(
            90.0 - lms.line_angle_deg(_tuple(hip_center), _tuple(shoulder_center)), 1
        )
        features.body_alignment_deg = round(
            abs(
                lms.signed_line_angle_deg(
                    _tuple(pixels[lms.LEFT_SHOULDER]), _tuple(pixels[lms.RIGHT_SHOULDER])
                )
                - lms.signed_line_angle_deg(
                    _tuple(pixels[lms.LEFT_HIP]), _tuple(pixels[lms.RIGHT_HIP])
                )
            ),
            1,
        )

        # ---------------- arms / wrists ----------------
        front_arm = (
            (lms.LEFT_SHOULDER, lms.LEFT_ELBOW, lms.LEFT_WRIST)
            if front_side == "left"
            else (lms.RIGHT_SHOULDER, lms.RIGHT_ELBOW, lms.RIGHT_WRIST)
        )
        back_arm = (
            (lms.RIGHT_SHOULDER, lms.RIGHT_ELBOW, lms.RIGHT_WRIST)
            if front_side == "left"
            else (lms.LEFT_SHOULDER, lms.LEFT_ELBOW, lms.LEFT_WRIST)
        )
        features.front_elbow_deg = round(lms.angle_deg(*[pixels[i] for i in front_arm]), 1)
        features.back_elbow_deg = round(lms.angle_deg(*[pixels[i] for i in back_arm]), 1)
        features.wrist_rotation_deg = round(kin.wrist_rotation(pixels, back_side), 1)

        # ---------------- feet orientation ----------------
        features.front_foot_direction_deg = round(
            lms.signed_line_angle_deg(
                _tuple(pixels[lms.LEFT_HEEL if front_side == "left" else lms.RIGHT_HEEL]),
                _tuple(
                    pixels[lms.LEFT_FOOT_INDEX if front_side == "left" else lms.RIGHT_FOOT_INDEX]
                ),
            ),
            1,
        )
        features.back_foot_pivot_deg = round(
            lms.signed_line_angle_deg(
                _tuple(pixels[lms.LEFT_HEEL if back_side == "left" else lms.RIGHT_HEEL]),
                _tuple(pixels[lms.LEFT_FOOT_INDEX if back_side == "left" else lms.RIGHT_FOOT_INDEX]),
            ),
            1,
        )

        features.hands_axis_deg = round(
            lms.signed_line_angle_deg(
                _tuple(pixels[lms.LEFT_WRIST]), _tuple(pixels[lms.RIGHT_WRIST])
            ),
            1,
        )
        features.facing_sign = 1.0 if front_ankle[0] >= hip_center[0] else -1.0

        # ---------------- bat ----------------
        if bat is not None and bat.tip is not None and bat.grip is not None:
            tip = np.array(bat.tip)
            features.bat_tip = bat.tip
            features.bat_grip = bat.grip
            features.bat_angle_deg = round(bat.angle_deg, 1)
            ankle_y = float(max(front_ankle[1], back_ankle[1]))
            if features.body_height_px > 0:
                features.bat_height_ratio = round(
                    float((ankle_y - tip[1]) / features.body_height_px), 3
                )
                features.hands_height_ratio = round(
                    float((ankle_y - bat.grip[1]) / features.body_height_px), 3
                )
            if self._prev_bat_tip is not None and features.px_per_meter > 0:
                step = float(np.linalg.norm(tip - self._prev_bat_tip))
                features.bat_speed_kmh = round(
                    step * self._fps / features.px_per_meter * 3.6, 1
                )
            self._prev_bat_tip = tip
        else:
            self._prev_bat_tip = None

        # ---------------- ball ----------------
        if ball is not None and ball.position is not None:
            position = np.array(ball.position)
            features.ball = ball.position
            if self._prev_ball is not None and features.px_per_meter > 0:
                step = float(np.linalg.norm(position - self._prev_ball))
                features.ball_speed_kmh = round(
                    step * self._fps / features.px_per_meter * 3.6, 1
                )
            self._prev_ball = position
            if features.bat_tip is not None and features.bat_grip is not None:
                features.bat_ball_distance_px = round(
                    _point_segment_distance(position, np.array(features.bat_grip), np.array(features.bat_tip)),
                    2,
                )
            else:
                features.bat_ball_distance_px = float("inf")
        else:
            features.bat_ball_distance_px = float("inf")
            self._prev_ball = None

        # ---------------- balance / stability / motion ----------------
        base_mid = (base_front + base_back) / 2.0
        # A narrow stance (or a front-on camera) collapses the base, so the span
        # is floored relative to body height to keep the ratio meaningful.
        base_span = max(
            float(np.linalg.norm(base_front - base_back)), features.body_height_px * 0.22, 1.0
        )
        lateral_offset = abs(float(com[0]) - float(base_mid[0])) / base_span
        features.balance_score = round(
            float(
                np.clip(
                    100.0 - lateral_offset * 70.0 - abs(features.spine_lean_deg) * 0.9,
                    0.0,
                    100.0,
                )
            ),
            1,
        )

        if self._stance_head is None:
            self._stance_head = head.copy()
        features.head_travel_px = round(float(np.linalg.norm(head - self._stance_head)), 2)

        if self._prev_pixels is not None and self._prev_pixels.shape == pixels.shape:
            features.motion_energy = round(
                float(np.mean(np.linalg.norm(pixels - self._prev_pixels, axis=1))), 3
            )
        self._prev_pixels = pixels.copy()
        return features

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _front_back_side(pixels: np.ndarray) -> Tuple[str, str]:
        """Decide which leg is the front foot.

        The batter looks towards the ball, so the head leans over the front
        foot; the toe of the front foot also points down the pitch.  Both cues
        are combined for robustness.
        """
        nose_x = float(pixels[lms.NOSE][0])
        left_x = float(pixels[lms.LEFT_ANKLE][0])
        right_x = float(pixels[lms.RIGHT_ANKLE][0])
        head_vote = "left" if abs(nose_x - left_x) < abs(nose_x - right_x) else "right"

        left_toe_out = float(pixels[lms.LEFT_FOOT_INDEX][0] - pixels[lms.LEFT_HEEL][0])
        right_toe_out = float(pixels[lms.RIGHT_FOOT_INDEX][0] - pixels[lms.RIGHT_HEEL][0])
        toe_vote = "left" if abs(left_toe_out) > abs(right_toe_out) else "right"

        front = head_vote if head_vote == toe_vote else head_vote
        return front, ("right" if front == "left" else "left")


def _point_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Shortest distance from ``point`` to the segment ``a-b`` (the bat blade)."""
    ab = b - a
    length_sq = float(np.dot(ab, ab))
    if length_sq < 1e-6:
        return float(np.linalg.norm(point - a))
    t = float(np.clip(np.dot(point - a, ab) / length_sq, 0.0, 1.0))
    return float(np.linalg.norm(point - (a + ab * t)))


__all__ = ["FeatureExtractor"]
