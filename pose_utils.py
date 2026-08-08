"""Shared pose-landmark math, drawing, and result types.

Used by both the single-photo path (app.py) and the video path
(video_analysis.py) so weight-transfer, mistake-detection, and
overlay-drawing logic only exists in one place.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import mediapipe as mp
import gdown
from mediapipe.framework.formats import landmark_pb2


def ensure_pose_model(model_path: str) -> None:
    """Download the MediaPipe pose landmarker model if not already present."""
    if not os.path.exists(model_path):
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
        gdown.download(url, model_path, quiet=False)


def get_point_and_vis(landmark, w, h):
    return (int(landmark.x * w), int(landmark.y * h)), float(landmark.visibility)


def wrist_midpoint(landmarks, h):
    """Midpoint of the two wrists, y scaled to pixel space (x stays normalized 0-1)."""
    lw = landmarks[15]
    rw = landmarks[16]
    return ((lw.x + rw.x) / 2.0, (lw.y + rw.y) / 2.0 * h)


# Below this fraction of frame width, front/back foot horizontal separation
# is too small to be a trustworthy weight-transfer signal (see
# compute_weight_transfer).
MIN_STANCE_SPAN_RATIO = 0.05


def compute_weight_transfer(lm, w, h):
    """
    Given a pose-landmark list and frame size, work out which ankle is the
    front foot (using head position), and how far weight has shifted onto
    it as a 0-100% "forward_pct".
    """
    left_ankle_pt, left_ankle_vis = get_point_and_vis(lm[31], w, h)
    right_ankle_pt, right_ankle_vis = get_point_and_vis(lm[32], w, h)
    left_hip_pt, _ = get_point_and_vis(lm[23], w, h)
    right_hip_pt, _ = get_point_and_vis(lm[24], w, h)
    nose_pt, _ = get_point_and_vis(lm[0], w, h)

    # Decide front/back leg with head position
    nose_x = nose_pt[0]
    if abs(nose_x - left_ankle_pt[0]) < abs(nose_x - right_ankle_pt[0]):
        front_ankle_pt = left_ankle_pt
        back_ankle_pt = right_ankle_pt
        back_vis = right_ankle_vis
        front_side = "left"
    else:
        front_ankle_pt = right_ankle_pt
        back_ankle_pt = left_ankle_pt
        back_vis = left_ankle_vis
        front_side = "right"

    front_x = front_ankle_pt[0]
    back_x = back_ankle_pt[0]
    hip_center_x = int((left_hip_pt[0] + right_hip_pt[0]) / 2)

    span = abs(front_x - back_x)
    # When the two feet are nearly on top of each other in the image (e.g.
    # a near-front-on camera angle, where front/back separation barely
    # shows up as a horizontal pixel gap), that gap is noise, not signal -
    # dividing by it swings forward_pct wildly toward 0% or 100% for tiny
    # pixel differences. Below this threshold, treat the reading as
    # unreliable and report a neutral 50/50 instead of a fabricated extreme.
    stance_span_reliable = span >= max(1, MIN_STANCE_SPAN_RATIO * w)

    if stance_span_reliable:
        t = (hip_center_x - back_x) / span
        if front_x < back_x:
            t = 1 - t
        t = max(0.0, min(1.0, t))
    else:
        t = 0.5

    forward_pct = int(round(t * 100))
    back_pct = 100 - forward_pct

    return {
        "forward_pct": forward_pct,
        "back_pct": back_pct,
        "front_ankle_pt": front_ankle_pt,
        "back_ankle_pt": back_ankle_pt,
        "back_vis": back_vis,
        "nose_pt": nose_pt,
        "stance_span_reliable": stance_span_reliable,
        "front_side": front_side,
    }


def detect_mistakes(forward_pct, nose_x, front_x, back_vis, stance_span_reliable=True):
    mistakes = []
    if not stance_span_reliable:
        mistakes.append(
            "Front and back foot are nearly overlapping in this frame, so weight-transfer % isn't reliable "
            "here - try filming more from the side so both feet are clearly separated."
        )
    elif forward_pct < 55:
        mistakes.append("Push more weight onto your front foot at impact.")
    if nose_x > front_x + 20:
        mistakes.append("Try to keep your head over or just in front of your front knee.")
    if back_vis < 0.3:
        mistakes.append("Back foot is not clearly visible. Try a photo where both feet are in the frame.")
    return mistakes


def rate_shot(forward_pct, mistakes):
    if forward_pct >= 70 and len(mistakes) <= 1:
        rating_label = "Excellent Shot"
        encouragement = "Super balance and weight transfer. This is a high-quality shot! 🏏🔥"
    elif forward_pct >= 60:
        rating_label = "Good Shot"
        encouragement = "Good mechanics overall. A bit more forward weight will make it even better."
    elif forward_pct >= 50:
        rating_label = "Okay / Needs Improvement"
        encouragement = "You’re close. Step in stronger and let your weight move more to the front foot."
    else:
        rating_label = "Needs Improvement"
        encouragement = "Most of your weight is staying back. Practise stepping into the ball and driving through the front leg."
    return rating_label, encouragement


def _to_landmark_list_proto(landmarks):
    """
    The Tasks API's PoseLandmarker returns pose_landmarks[0] as a plain
    list of NormalizedLandmark objects, but the legacy
    mp.solutions.drawing_utils.draw_landmarks() expects a protobuf
    NormalizedLandmarkList (with a `.landmark` field) - convert here.
    """
    proto = landmark_pb2.NormalizedLandmarkList()
    proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z, visibility=lm.visibility)
        for lm in landmarks
    ])
    return proto


def draw_pose_overlay(frame, landmarks, w, h):
    """Draw the skeleton + front/back foot markers on frame (in place)."""
    mp.solutions.drawing_utils.draw_landmarks(
        frame, _to_landmark_list_proto(landmarks), mp.solutions.pose.POSE_CONNECTIONS
    )

    wt = compute_weight_transfer(landmarks, w, h)
    cv2.circle(frame, wt["back_ankle_pt"], 6, (0, 0, 255), -1)   # back red
    cv2.circle(frame, wt["front_ankle_pt"], 6, (0, 255, 0), -1)  # front green

    return wt


@dataclass
class ShotAnalysisResult:
    frame: Optional[object] = None
    mistakes: List[str] = field(default_factory=list)
    forward_pct: int = 0
    back_pct: int = 100
    rating_label: str = "No Data"
    encouragement: str = ""
    shot_name: Optional[str] = None
    shot_summary: Optional[str] = None
    field_suggestion: Optional[str] = None
    variations: List[str] = field(default_factory=list)
    master_player: Optional[str] = None
    shot_history: Optional[str] = None
    improvement_summary: Optional[str] = None
    alt_safe: List[str] = field(default_factory=list)
    alt_aggressive: List[str] = field(default_factory=list)
    final_feedback: List[str] = field(default_factory=list)

    # Video-only fields
    is_video: bool = False
    weight_transfer_series: Optional[List[dict]] = None
    phase_markers: Optional[dict] = None

    # Bat/ball tracking (video-only, Phase 2)
    bat_metrics: Optional[dict] = None
    ball_estimate: Optional[dict] = None
    is_calibrated: bool = False

    # Biomechanics (Phase 4) and persistence (Phase 5)
    biomechanics: Optional[dict] = None
    session_id: Optional[int] = None

    # Classification chain (Phase 3): "ml" | "llm" | "rule-based"
    classification_method: str = "rule-based"
    classification_confidence: Optional[float] = None
    classification_reasoning: Optional[str] = None
