"""Video-clip pose sequence extraction, rough swing segmentation,
weight-transfer-over-time analysis, and bat/ball tracking.

This is the video counterpart to app.py's single-photo analysis path.
Segmentation is intentionally a simple heuristic (wrist-speed peak +
highest-wrist-point), not a trained model - see the Phase 1 plan notes
for why, and what a trained model (Phase 3) could improve later. Bat/
ball tracking (Phase 2) is likewise camera-only, via bat_tracking.py
and ball_tracking.py.
"""

import math
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import bat_tracking
import ball_tracking
import biomechanics
import pose_utils
import shot_classifier_llm
import shot_classifier_ml
from pose_utils import ShotAnalysisResult
from shots import SHOT_DB, classify_shot_key, get_shot_info_from_db

MODEL_PATH = "pose_landmarker.task"
MAX_SAMPLES = 120           # spread pose sampling across the whole clip
MIN_VALID_FRAMES = 6        # below this, segmentation is considered unreliable
SPEED_SMOOTHING_WINDOW = 3
LOW_SPEED_THRESHOLD_RATIO = 0.15  # fraction of peak speed considered "still"
EARLY_EXCLUSION_RATIO = 0.08  # ignore this fraction of the clip's start when finding the impact peak

pose_utils.ensure_pose_model(MODEL_PATH)


def _make_video_landmarker():
    """A fresh VIDEO-mode PoseLandmarker per call.

    VIDEO mode is stateful and requires strictly increasing timestamps
    within one instance, so a shared module-level instance (like the
    IMAGE-mode one in app.py) would risk corrupting tracking state if two
    video requests interleaved. A fresh instance per request avoids that
    at the cost of a small per-request model-init overhead.
    """
    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


def extract_landmark_sequence(video_path: str, bat_detector=None, ball_detector=None):
    """
    Sample frames spread across the whole clip and run pose detection
    (plus optional bat/ball detectors) on each - all within a single
    cv2.VideoCapture pass so the clip is only decoded once.

    Returns (sequence, bat_track, ball_track, fps, w, h). `sequence` is
    the Phase 1 pose landmark list ({frame_idx, timestamp_ms, landmarks},
    only for frames where a pose was found). `bat_track` / `ball_track`
    are the analogous lists for whichever detectors were passed in
    (empty if no detector was given, or nothing was detected). Ball
    detection compares each sampled frame to the *previous sampled*
    frame, not necessarily the adjacent video frame - an approximation
    that's fine for slow-mo clips but coarser on long/low-fps ones.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], [], [], 0.0, 0, 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    step = max(1, total_frames // MAX_SAMPLES) if total_frames > 0 else 1

    landmarker = _make_video_landmarker()
    sequence = []
    bat_track = []
    ball_track = []
    last_timestamp_ms = -1
    frame_idx = 0
    prev_sampled_frame = None

    try:
        while True:
            ok = cap.grab()
            if not ok:
                break
            if frame_idx % step == 0:
                ok, frame = cap.retrieve()
                if ok:
                    timestamp_ms = int(round((frame_idx / fps) * 1000))
                    if timestamp_ms <= last_timestamp_ms:
                        timestamp_ms = last_timestamp_ms + 1
                    last_timestamp_ms = timestamp_ms

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    result = landmarker.detect_for_video(mp_image, timestamp_ms)
                    if result.pose_landmarks:
                        sequence.append({
                            "frame_idx": frame_idx,
                            "timestamp_ms": timestamp_ms,
                            "landmarks": result.pose_landmarks[0],
                        })

                    if bat_detector is not None:
                        bat_det = bat_detector.detect(frame)
                        if bat_det is not None:
                            bat_track.append({
                                "frame_idx": frame_idx,
                                "timestamp_ms": timestamp_ms,
                                "center_px": bat_det.center_px,
                                "angle_deg": bat_det.angle_deg,
                            })

                    if ball_detector is not None and prev_sampled_frame is not None:
                        ball_det = ball_detector.detect(prev_sampled_frame, frame)
                        if ball_det is not None:
                            ball_track.append({
                                "frame_idx": frame_idx,
                                "timestamp_ms": timestamp_ms,
                                "center_px": ball_det.center_px,
                                "radius_px": ball_det.radius_px,
                            })

                    prev_sampled_frame = frame
            frame_idx += 1
    finally:
        landmarker.close()
        cap.release()

    return sequence, bat_track, ball_track, fps, w, h


def _smooth(values, window):
    if window <= 1:
        return list(values)
    out = []
    n = len(values)
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


def segment_phases(sequence, h):
    """
    Heuristic swing segmentation from a landmark sequence.

    Returns (phase_markers, impact_idx) where impact_idx indexes into
    `sequence`, or (None, fallback_idx) if the motion signal is too weak
    / too few frames to segment reliably.
    """
    if len(sequence) < MIN_VALID_FRAMES:
        fallback_idx = _most_visible_frame(sequence)
        return None, fallback_idx

    wrist_points = [pose_utils.wrist_midpoint(item["landmarks"], h) for item in sequence]

    raw_speed = [0.0]
    for i in range(1, len(wrist_points)):
        (x0, y0), (x1, y1) = wrist_points[i - 1], wrist_points[i]
        dist = math.hypot(x1 - x0, y1 - y0)
        raw_speed.append(dist / max(1, h))  # normalize by frame height

    speed = _smooth(raw_speed, SPEED_SMOOTHING_WINDOW)

    # Ignore a short leading window when searching for the impact peak.
    # A brief pre-shot movement - settling into stance, tapping the bat
    # down while waiting for the bowler - can register a bigger wrist-speed
    # spike than the real swing, especially in clips that include some
    # dead time before the ball arrives. Real bat-ball impact essentially
    # never happens in a clip's opening frames, so excluding them avoids
    # that spike being mistaken for the shot itself.
    min_search_idx = min(
        int(len(speed) * EARLY_EXCLUSION_RATIO),
        max(0, len(speed) - MIN_VALID_FRAMES),
    )
    search_speed = speed[min_search_idx:]
    peak_speed = max(search_speed)

    if peak_speed < 1e-4:
        fallback_idx = _most_visible_frame(sequence)
        return None, fallback_idx

    impact_idx = min_search_idx + search_speed.index(peak_speed)

    # Highest wrist point (min pixel y) before impact = top of backswing
    search_range = range(0, impact_idx + 1)
    backlift_idx = min(search_range, key=lambda i: wrist_points[i][1], default=0)

    threshold = peak_speed * LOW_SPEED_THRESHOLD_RATIO
    stance_candidates = [i for i in range(0, backlift_idx + 1) if speed[i] < threshold]
    stance_idx = stance_candidates[-1] if stance_candidates else 0

    phase_markers = {
        "stance_ms": sequence[stance_idx]["timestamp_ms"],
        "backlift_ms": sequence[backlift_idx]["timestamp_ms"],
        "downswing_start_ms": sequence[backlift_idx]["timestamp_ms"],
        "impact_ms": sequence[impact_idx]["timestamp_ms"],
        "follow_through_start_ms": sequence[impact_idx]["timestamp_ms"],
    }
    return phase_markers, impact_idx


def _most_visible_frame(sequence):
    if not sequence:
        return 0
    def avg_visibility(item):
        lm = item["landmarks"]
        return sum(p.visibility for p in lm) / len(lm)
    return max(range(len(sequence)), key=lambda i: avg_visibility(sequence[i]))


def _weight_transfer_series(sequence, w, h):
    series = []
    for item in sequence:
        wt = pose_utils.compute_weight_transfer(item["landmarks"], w, h)
        series.append({"t_ms": item["timestamp_ms"], "forward_pct": wt["forward_pct"]})
    return series


def _grab_frame(video_path, frame_idx):
    cap = cv2.VideoCapture(video_path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def _draw_tracks(frame, bat_track, ball_track):
    """Draw the bat (orange) and ball (white) trajectories onto frame in place."""
    if bat_track:
        pts = [tuple(map(int, item["center_px"])) for item in sorted(bat_track, key=lambda i: i["timestamp_ms"])]
        if len(pts) >= 2:
            cv2.polylines(frame, [np.array(pts, dtype=np.int32)], False, (0, 200, 255), 2)
        cv2.circle(frame, pts[-1], 5, (0, 200, 255), -1)
    if ball_track:
        pts = [tuple(map(int, item["center_px"])) for item in sorted(ball_track, key=lambda i: i["timestamp_ms"])]
        if len(pts) >= 2:
            cv2.polylines(frame, [np.array(pts, dtype=np.int32)], False, (255, 255, 255), 2)
        cv2.circle(frame, pts[-1], 5, (255, 255, 255), -1)


def analyze_video(video_path: str, angle: str, cm_per_px: Optional[float] = None) -> ShotAnalysisResult:
    """Full analysis for a short video clip, mirroring analyze_impact_photo."""
    bat_detector = bat_tracking.get_default_bat_detector()
    ball_detector = ball_tracking.get_default_ball_detector()

    sequence, bat_track, ball_track, fps, w, h = extract_landmark_sequence(
        video_path, bat_detector=bat_detector, ball_detector=ball_detector
    )

    if not sequence:
        return ShotAnalysisResult(
            frame=None,
            mistakes=["No pose could be detected in this video. Use a clearer clip with the batter fully in frame."],
            rating_label="No Data",
            encouragement="Pose not detected.",
            is_video=True,
        )

    phase_markers, impact_idx = segment_phases(sequence, h)
    weight_transfer_series = _weight_transfer_series(sequence, w, h)

    impact_item = sequence[impact_idx]
    impact_landmarks = impact_item["landmarks"]

    frame = _grab_frame(video_path, impact_item["frame_idx"])
    if frame is None:
        return ShotAnalysisResult(
            frame=None,
            mistakes=["Could not read the impact frame from this video."],
            rating_label="No Data",
            encouragement="Video not readable.",
            is_video=True,
        )
    original_frame = frame.copy()  # clean copy for the LLM tier, before the overlay is drawn in place

    wt = pose_utils.draw_pose_overlay(frame, impact_landmarks, w, h)
    forward_pct = wt["forward_pct"]
    back_pct = wt["back_pct"]

    mistakes = pose_utils.detect_mistakes(
        forward_pct, wt["nose_pt"][0], wt["front_ankle_pt"][0], wt["back_vis"], wt["stance_span_reliable"]
    )
    rating_label, encouragement = pose_utils.rate_shot(forward_pct, mistakes)

    # Classification chain: local ML model -> LLM fallback -> rule-based.
    # Each tier returns None when it can't/shouldn't produce a result, so
    # this always ends with a usable shot_key even with nothing configured.
    classification_method = "rule-based"
    classification_confidence = None
    classification_reasoning = None

    ml_result = shot_classifier_ml.classify_shot_ml(sequence, w, h)
    if ml_result is not None:
        shot_key, classification_confidence = ml_result
        classification_method = "ml"
    else:
        llm_result = shot_classifier_llm.classify_shot_llm(original_frame, SHOT_DB)
        if llm_result is not None:
            shot_key, classification_confidence, classification_reasoning = llm_result
            classification_method = "llm"
        else:
            shot_key = classify_shot_key(impact_landmarks, w, h, forward_pct, angle)

    (
        shot_name,
        shot_summary,
        field_suggestion,
        variations,
        master_player,
        shot_history,
        improvement_summary,
        alt_safe,
        alt_aggressive,
        final_feedback,
    ) = get_shot_info_from_db(shot_key)

    bat_metrics = bat_tracking.compute_bat_metrics(
        bat_track, phase_markers, impact_landmarks, w, h, cm_per_px=cm_per_px
    )
    ball_estimate = ball_tracking.estimate_line_and_length(ball_track, cm_per_px=cm_per_px)
    _draw_tracks(frame, bat_track, ball_track)

    biomech_metrics = biomechanics.compute_biomechanics(
        impact_landmarks, w, h, wt["front_side"], bat_metrics=bat_metrics, cm_per_px=cm_per_px
    )
    biomechanics_result = biomechanics.compare_to_benchmarks(shot_key, biomech_metrics)

    return ShotAnalysisResult(
        frame=frame,
        mistakes=mistakes,
        forward_pct=forward_pct,
        back_pct=back_pct,
        rating_label=rating_label,
        encouragement=encouragement,
        shot_name=shot_name,
        shot_summary=shot_summary,
        field_suggestion=field_suggestion,
        variations=variations,
        master_player=master_player,
        shot_history=shot_history,
        improvement_summary=improvement_summary,
        alt_safe=alt_safe,
        alt_aggressive=alt_aggressive,
        final_feedback=final_feedback,
        is_video=True,
        weight_transfer_series=weight_transfer_series,
        phase_markers=phase_markers,
        bat_metrics=bat_metrics,
        ball_estimate=ball_estimate,
        is_calibrated=cm_per_px is not None,
        biomechanics=biomechanics_result,
        classification_method=classification_method,
        classification_confidence=classification_confidence,
        classification_reasoning=classification_reasoning,
    )
