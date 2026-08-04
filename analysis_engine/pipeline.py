"""The analysis pipeline facade.

``ShotAnalysisPipeline`` is the only object the web layer needs to know about.
It wires the modular stages together:

``frames -> pose -> bat/ball tracking -> per-frame features -> impact detection
-> temporal swing features -> shot classification -> weight transfer -> technical
metrics -> scores -> coaching -> annotation/report``

Two entry points are provided: :meth:`analyze_image` (single impact photo, used
by the original ``/analyze`` route) and :meth:`analyze_video` (full swing
sequence).  Video decoding runs in a prefetch thread so I/O overlaps inference.
"""

from __future__ import annotations

import logging
import os
import queue
import shutil
import subprocess
import threading
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .classifier import build_swing_features, classify_shot
from .coaching import generate_coaching
from .comparison import compare_with_professional
from .config import CONFIG
from .datatypes import (
    AnalysisResult,
    BallObservation,
    BatObservation,
    FrameFeatures,
    PoseFrame,
)
from .features import FeatureExtractor
from .impact import detect_impact
from .metrics import compute_technical_metrics
from .output.annotate import draw_analysis
from .pose import PoseEstimator
from .scoring import compute_scores
from .tracking import BallTracker, BatTracker, analyse_ball_track, analyse_bat_track
from .weight_transfer import analyse_weight_transfer

LOGGER = logging.getLogger(__name__)

#: Camera-angle codes used by the existing UI.
ANGLE_SIDE, ANGLE_FRONT, ANGLE_DIAGONAL = "A", "B", "C"


class ShotAnalysisPipeline:
    """End-to-end analysis for a photo or a video of a cricket shot."""

    def __init__(self, static_image_mode: bool = False, realtime: bool = False) -> None:
        self.static_image_mode = static_image_mode
        self.pose = PoseEstimator(static_image_mode=static_image_mode, realtime=realtime)
        self.backend = self.pose.backend

    def close(self) -> None:
        self.pose.close()

    # ------------------------------------------------------------------ image
    def analyze_image(
        self,
        frame_bgr: np.ndarray,
        angle: str = ANGLE_SIDE,
        annotated_path: Optional[str] = None,
        graph_path: Optional[str] = None,
        pro_player: Optional[str] = None,
    ) -> AnalysisResult:
        """Analyse a single impact photo.

        Temporal cues are unavailable from one frame, so the confidence is
        automatically damped by the classifier's data-quality factor.
        """
        pose_frame = self.pose.process(frame_bgr, 0, 0.0)
        bat_tracker, ball_tracker = BatTracker(), BallTracker()
        extractor = FeatureExtractor()
        extractor.set_fps(1.0)

        bat = bat_tracker.update(frame_bgr, 0, 0.0, pose_frame)
        ball = ball_tracker.update(frame_bgr, 0, 0.0, pose_frame)
        features = extractor.extract(pose_frame, bat, ball, 0, 0.0)
        result = self._assemble(
            frames_features=[features],
            poses=[pose_frame],
            bat_observations=bat_tracker.observations,
            ball_observations=ball_tracker.observations,
            fps=1.0,
            angle=angle,
            graph_path=graph_path,
            pro_player=pro_player,
            source="image",
        )
        if not features.pose_detected:
            result.warnings.append("Pose not detected. Use a clearer photo with the full body visible.")
        if annotated_path:
            annotated = draw_analysis(
                frame_bgr.copy(),
                pose_frame,
                features if features.pose_detected else None,
                shot=result.shot,
                impact=result.impact,
                bat_path=result.bat_track,
                ball_path=result.ball_track,
                is_impact_frame=True,
            )
            os.makedirs(os.path.dirname(annotated_path) or ".", exist_ok=True)
            cv2.imwrite(annotated_path, annotated)
            result.annotated_image_path = annotated_path
        return result

    # ------------------------------------------------------------------ video
    def analyze_video(
        self,
        video_path: str,
        angle: str = ANGLE_SIDE,
        annotated_video_path: Optional[str] = None,
        annotated_image_path: Optional[str] = None,
        graph_path: Optional[str] = None,
        pro_player: Optional[str] = None,
        max_frames: Optional[int] = None,
    ) -> AnalysisResult:
        """Analyse a full swing from a video file."""
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
        limit = max_frames or CONFIG.max_video_frames

        self.pose.reset()
        bat_tracker, ball_tracker = BatTracker(), BallTracker()
        extractor = FeatureExtractor()
        extractor.set_fps(fps)

        poses: List[Optional[PoseFrame]] = []
        features: List[FrameFeatures] = []
        for index, frame in _frame_reader(capture, limit):
            timestamp = index / fps
            pose_frame = self.pose.process(frame, index, timestamp)
            bat = bat_tracker.update(frame, index, timestamp, pose_frame)
            ball = ball_tracker.update(frame, index, timestamp, pose_frame)
            poses.append(pose_frame)
            features.append(extractor.extract(pose_frame, bat, ball, index, timestamp))
        capture.release()

        if not features:
            raise ValueError("The video contained no readable frames.")

        result = self._assemble(
            frames_features=features,
            poses=poses,
            bat_observations=bat_tracker.observations,
            ball_observations=ball_tracker.observations,
            fps=fps,
            angle=angle,
            graph_path=graph_path,
            pro_player=pro_player,
            source="video",
        )
        if annotated_video_path or annotated_image_path:
            self._render_video(
                video_path,
                result,
                features,
                poses,
                fps,
                annotated_video_path,
                annotated_image_path,
            )
        return result

    # --------------------------------------------------------------- assembly
    def _assemble(
        self,
        frames_features: Sequence[FrameFeatures],
        poses: Sequence[Optional[PoseFrame]],
        bat_observations: Sequence[BatObservation],
        ball_observations: Sequence[BallObservation],
        fps: float,
        angle: str,
        graph_path: Optional[str],
        pro_player: Optional[str],
        source: str,
    ) -> AnalysisResult:
        px_per_meter = float(
            np.median([f.px_per_meter for f in frames_features if f.px_per_meter > 0] or [0.0])
        )
        body_height_px = float(
            np.median([f.body_height_px for f in frames_features if f.body_height_px > 0] or [0.0])
        )
        shoulder_y = next(
            (f.shoulder_center[1] for f in frames_features if f.shoulder_center is not None), None
        )

        impact = detect_impact(frames_features, bat_observations, ball_observations, fps)
        bat_track = analyse_bat_track(
            bat_observations,
            px_per_meter,
            fps,
            impact.frame_index,
            body_height_px,
            shoulder_y,
        )
        ball_track = analyse_ball_track(ball_observations, px_per_meter, fps, impact.frame_index)
        if not impact.bat_speed_kmh:
            impact.bat_speed_kmh = float(bat_track.get("max_speed_kmh", 0.0) or 0.0)
        if not impact.ball_speed_kmh:
            impact.ball_speed_kmh = float(ball_track.get("speed_kmh", 0.0) or 0.0)

        weight = analyse_weight_transfer(frames_features, impact, fps, graph_path)

        vector = build_swing_features(frames_features, impact, bat_track, ball_track)
        vector = _apply_camera_angle(vector, angle)
        pose_frames = sum(1 for f in frames_features if f.pose_detected)
        shot = classify_shot(
            vector,
            frames_analysed=len(frames_features),
            pose_frames=pose_frames,
            ball_detected=bool(ball_track.get("detected")),
            bat_from_image=any(o.from_image_evidence for o in bat_observations),
        )

        metrics = compute_technical_metrics(frames_features, impact, bat_track, ball_track, weight, fps)
        scores = compute_scores(frames_features, impact, weight, metrics, shot)
        coaching = generate_coaching(shot, impact, weight, metrics, scores, frames_features)
        comparison = compare_with_professional(metrics, weight, shot, pro_player)

        metrics["swing_feature_vector"] = {k: round(v, 3) for k, v in vector.items()}
        metrics["pose_backend"] = self.backend
        metrics["pose_frames"] = pose_frames

        return AnalysisResult(
            shot=shot,
            impact=impact,
            weight_transfer=weight,
            metrics=metrics,
            scores=scores,
            coaching=coaching,
            ball_track=list(ball_track.get("path") or []),
            bat_track=list(bat_track.get("path") or []),
            frames_analysed=len(frames_features),
            fps=round(fps, 2),
            source=source,
            graph_path=weight.graph_path,
            comparison=comparison,
        )

    # ---------------------------------------------------------------- render
    def _render_video(
        self,
        video_path: str,
        result: AnalysisResult,
        features: Sequence[FrameFeatures],
        poses: Sequence[Optional[PoseFrame]],
        fps: float,
        annotated_video_path: Optional[str],
        annotated_image_path: Optional[str],
    ) -> None:
        """Second pass: burn the overlays into a video and save the impact frame."""
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            return
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = None
        raw_video_path = None
        if annotated_video_path:
            os.makedirs(os.path.dirname(annotated_video_path) or ".", exist_ok=True)
            raw_video_path = f"{os.path.splitext(annotated_video_path)[0]}.raw.mp4"
            for fourcc in ("mp4v", "avc1", "MJPG"):
                writer = cv2.VideoWriter(
                    raw_video_path, cv2.VideoWriter_fourcc(*fourcc), fps, (width, height)
                )
                if writer.isOpened():
                    break
                writer.release()
                writer = None

        predicted = result.metrics.get("predicted_ball_path") if isinstance(result.metrics, dict) else None
        by_index = {f.frame_index: (f, poses[i] if i < len(poses) else None) for i, f in enumerate(features)}
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            feature, pose_frame = by_index.get(index, (None, None))
            annotated = draw_analysis(
                frame,
                pose_frame,
                feature,
                shot=result.shot,
                impact=result.impact,
                bat_path=result.bat_track,
                ball_path=result.ball_track,
                predicted_path=predicted,
                is_impact_frame=index == result.impact.frame_index,
            )
            if writer is not None:
                writer.write(annotated)
            if annotated_image_path and index == result.impact.frame_index:
                os.makedirs(os.path.dirname(annotated_image_path) or ".", exist_ok=True)
                cv2.imwrite(annotated_image_path, annotated)
                result.annotated_image_path = annotated_image_path
            index += 1
        capture.release()
        if writer is not None:
            writer.release()
            result.annotated_video_path = _finalise_video(raw_video_path, annotated_video_path)
        if annotated_image_path and result.annotated_image_path is None:
            # Impact frame fell outside the decoded range: keep the last frame.
            result.annotated_image_path = None


def _finalise_video(raw_path: Optional[str], final_path: str) -> Optional[str]:
    """Transcode the OpenCV output to browser-friendly H.264 when ffmpeg exists.

    OpenCV can usually only write ``mp4v``/``MJPG``, which most browsers refuse
    to play inline; ffmpeg (already required by OpenCV's video support on most
    installs) is used opportunistically and the raw file is kept otherwise.
    """
    if not raw_path or not os.path.exists(raw_path):
        return None
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        try:
            subprocess.run(
                [
                    ffmpeg, "-y", "-loglevel", "error", "-i", raw_path,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    final_path,
                ],
                check=True,
                timeout=600,
            )
            os.remove(raw_path)
            return final_path
        except Exception as exc:  # pragma: no cover - environment dependent
            LOGGER.warning("H.264 transcode failed (%s); keeping the raw encode.", exc)
    shutil.move(raw_path, final_path)
    return final_path


def _apply_camera_angle(vector: Dict[str, float], angle: str) -> Dict[str, float]:
    """Drop features that the given camera angle cannot measure reliably.

    From the bowler's end (front view) the depth axis is compressed, so how far
    in front of the body contact happened cannot be trusted; the remaining
    features (swing plane, weight transfer, rotation, wrist roll) still separate
    the shots.
    """
    if not vector:
        return vector
    filtered = dict(vector)
    if angle == ANGLE_FRONT:
        filtered.pop("contact_offset", None)
        filtered.pop("stride", None)
    return filtered


def _frame_reader(capture: cv2.VideoCapture, limit: int):
    """Yield ``(index, frame)`` using a background decode thread.

    Decoding overlaps inference, which keeps the CPU busy on the model instead
    of on I/O; the queue is bounded so memory stays flat for long videos.
    """
    frames: "queue.Queue[Optional[Tuple[int, np.ndarray]]]" = queue.Queue(maxsize=16)

    def _worker() -> None:
        index = 0
        try:
            while index < limit:
                ok, frame = capture.read()
                if not ok:
                    break
                frames.put((index, frame))
                index += 1
        finally:
            frames.put(None)

    thread = threading.Thread(target=_worker, name="video-decode", daemon=True)
    thread.start()
    while True:
        item = frames.get()
        if item is None:
            break
        yield item
    thread.join(timeout=1.0)


__all__ = ["ShotAnalysisPipeline", "ANGLE_SIDE", "ANGLE_FRONT", "ANGLE_DIAGONAL"]
