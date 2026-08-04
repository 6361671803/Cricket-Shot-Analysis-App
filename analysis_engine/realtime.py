"""Real-time analysis for webcams, USB cameras and live feeds.

Threading model (keeps the Flask worker responsive and the UI unblocked):

* **capture thread** – reads the camera as fast as it delivers frames and keeps
  only the newest one (dropping stale frames prevents latency build-up);
* **analysis thread** – runs pose + trackers + feature extraction, updates the
  rolling metric window and re-classifies the shot every few frames, then
  publishes an annotated JPEG;
* **request threads** – MJPEG streaming and the metrics endpoint only read the
  latest published result, so HTTP never waits on inference.

An adaptive frame-skip keeps the analysis loop at the configured target FPS on
slower machines; GPU delegation is used when ``CRICKET_USE_GPU=1``.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from .classifier import build_swing_features, classify_shot
from .config import CONFIG, RealtimeConfig
from .datatypes import BallObservation, BatObservation, FrameFeatures, ShotPrediction
from .features import FeatureExtractor
from .impact import detect_impact
from .output.annotate import draw_analysis, draw_live_hud
from .pose import PoseEstimator
from .tracking import BallTracker, BatTracker, analyse_bat_track

LOGGER = logging.getLogger(__name__)

#: How often (in analysed frames) the rolling shot classification is refreshed.
CLASSIFY_EVERY = 5


class RealtimeAnalyzer:
    """Continuous analysis of a live camera feed with a live overlay."""

    def __init__(
        self,
        source: Union[int, str] = 0,
        config: Optional[RealtimeConfig] = None,
        angle: str = "A",
    ) -> None:
        self.config = config or CONFIG.realtime
        self.source = source
        self.angle = angle
        self._capture: Optional[cv2.VideoCapture] = None
        self._pose: Optional[PoseEstimator] = None
        self._bat = BatTracker()
        self._ball = BallTracker()
        self._extractor = FeatureExtractor()
        self._window: Deque[FrameFeatures] = deque(maxlen=90)
        self._bat_window: Deque[BatObservation] = deque(maxlen=90)
        self._ball_window: Deque[BallObservation] = deque(maxlen=90)
        self._shot: Optional[ShotPrediction] = None

        self._lock = threading.Lock()
        self._latest_raw: Optional[Tuple[int, np.ndarray]] = None
        self._latest_jpeg: Optional[bytes] = None
        self._metrics: Dict[str, object] = {"status": "starting"}
        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []
        self._fps = 0.0
        self._skip = 0
        self._last_touch = time.time()
        self.error: Optional[str] = None

    # ---------------------------------------------------------------- control
    def start(self) -> bool:
        """Open the camera and start the capture + analysis threads."""
        if self._threads:
            return True
        capture = cv2.VideoCapture(self.source)
        if not capture.isOpened():
            capture.release()
            self.error = f"Could not open camera source '{self.source}'."
            self._metrics = {"status": "error", "message": self.error}
            return False
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.capture_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.capture_height)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._capture = capture
        self._pose = PoseEstimator(static_image_mode=False, realtime=True)
        self._extractor.set_fps(self.config.target_fps)
        self._stop.clear()
        for target, name in ((self._capture_loop, "rt-capture"), (self._analysis_loop, "rt-analysis")):
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self._threads.append(thread)
        self._metrics = {"status": "running"}
        return True

    def stop(self) -> None:
        """Stop both threads and release the camera."""
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=2.0)
        self._threads.clear()
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        if self._pose is not None:
            self._pose.close()
            self._pose = None
        self._metrics = {"status": "stopped"}

    @property
    def running(self) -> bool:
        return bool(self._threads) and not self._stop.is_set()

    def touch(self) -> None:
        """Mark the feed as still being watched (used for idle shutdown)."""
        self._last_touch = time.time()

    # ------------------------------------------------------------------ loops
    def _capture_loop(self) -> None:
        index = 0
        misses = 0
        while not self._stop.is_set() and self._capture is not None:
            ok, frame = self._capture.read()
            if not ok:
                misses += 1
                # A camera occasionally drops a frame; a file simply ended.
                if misses > self.config.max_read_misses:
                    LOGGER.info("Camera source '%s' stopped delivering frames.", self.source)
                    self._stop.set()
                    break
                time.sleep(0.02)
                continue
            misses = 0
            with self._lock:
                self._latest_raw = (index, frame)
            index += 1
            if time.time() - self._last_touch > self.config.idle_timeout_s:
                LOGGER.info("Real-time feed idle; shutting the camera down.")
                self._stop.set()
                break

    def _analysis_loop(self) -> None:
        target_period = 1.0 / max(1, self.config.target_fps)
        counter = 0
        smoothed_fps = float(self.config.target_fps)
        while not self._stop.is_set():
            started = time.time()
            with self._lock:
                item = self._latest_raw
            if item is None:
                time.sleep(0.01)
                continue
            index, frame = item
            frame = frame.copy()

            if self._skip > 0:
                # Skip analysis but keep the stream alive with the HUD only.
                self._skip -= 1
                self._publish(draw_live_hud(frame, self._metrics))
                continue

            timestamp = index / max(1.0, self.config.target_fps)
            pose_frame = self._pose.process(frame, index, timestamp) if self._pose else None
            bat = self._bat.update(frame, index, timestamp, pose_frame)
            ball = self._ball.update(frame, index, timestamp, pose_frame)
            features = self._extractor.extract(pose_frame, bat, ball, index, timestamp)
            self._window.append(features)
            self._bat_window.append(bat)
            self._ball_window.append(ball)

            counter += 1
            if counter % CLASSIFY_EVERY == 0:
                self._refresh_shot()

            live = self._live_metrics(features, smoothed_fps)
            annotated = draw_analysis(
                frame,
                pose_frame,
                features if features.pose_detected else None,
                shot=self._shot,
                bat_path=[{"frame": o.frame_index, "x": o.tip[0], "y": o.tip[1]} for o in self._bat_window if o.tip],
                ball_path=[
                    {"frame": o.frame_index, "x": o.position[0], "y": o.position[1]}
                    for o in self._ball_window
                    if o.position
                ],
                show_panel=False,
            )
            self._publish(draw_live_hud(annotated, live))

            elapsed = time.time() - started
            instantaneous = 1.0 / max(1e-6, elapsed)
            smoothed_fps = 0.85 * smoothed_fps + 0.15 * instantaneous
            self._fps = smoothed_fps
            if self.config.adaptive_skip and elapsed > target_period * 1.6:
                self._skip = min(self.config.max_skip, self._skip + 1)
            elif elapsed < target_period * 0.7:
                time.sleep(max(0.0, target_period - elapsed))

    # --------------------------------------------------------------- analysis
    def _refresh_shot(self) -> None:
        """Re-run the temporal classifier over the rolling window."""
        window = list(self._window)
        if len(window) < CONFIG.temporal.min_frames_for_sequence:
            return
        bats = list(self._bat_window)
        balls = list(self._ball_window)
        impact = detect_impact(window, bats, balls, self.config.target_fps)
        px_per_meter = float(np.median([f.px_per_meter for f in window if f.px_per_meter > 0] or [0.0]))
        body_height = float(np.median([f.body_height_px for f in window if f.body_height_px > 0] or [0.0]))
        shoulder_y = next((f.shoulder_center[1] for f in window if f.shoulder_center), None)
        bat_track = analyse_bat_track(
            bats, px_per_meter, self.config.target_fps, impact.frame_index, body_height, shoulder_y
        )
        vector = build_swing_features(window, impact, bat_track)
        if not vector:
            return
        if self.angle == "B":
            vector.pop("contact_offset", None)
        self._shot = classify_shot(
            vector,
            frames_analysed=len(window),
            pose_frames=sum(1 for f in window if f.pose_detected),
            ball_detected=any(b.position for b in balls),
            bat_from_image=any(b.from_image_evidence for b in bats),
        )
        self._last_impact = impact
        self._last_bat_track = bat_track

    def _live_metrics(self, features: FrameFeatures, fps: float) -> Dict[str, object]:
        window = list(self._window)
        heads = np.array([f.head for f in window if f.head is not None])
        head_stability = (
            float(np.clip(100.0 - float(np.std(heads, axis=0).mean()) / max(1e-6, features.body_height_px) * 420.0, 0.0, 100.0))
            if len(heads) > 1 and features.body_height_px > 0
            else 100.0
        )
        bat_track = getattr(self, "_last_bat_track", {})
        footwork = (
            "Positive" if features.forward_pct >= 60 else "Neutral" if features.forward_pct >= 45 else "Back foot"
        )
        metrics: Dict[str, object] = {
            "status": "running",
            "shot": self._shot.name if self._shot else "Analysing...",
            "confidence": self._shot.confidence if self._shot else 0.0,
            "reason": self._shot.reason if self._shot else "",
            "forward_pct": features.forward_pct,
            "back_pct": features.back_pct,
            "head_stability": round(head_stability, 1),
            "bat_angle_deg": features.bat_angle_deg,
            "backlift_angle_deg": bat_track.get("backlift_angle_deg", 0.0),
            "front_foot_angle_deg": features.front_foot_direction_deg,
            "swing_speed_kmh": features.bat_speed_kmh,
            "impact_prediction": self._impact_prediction(features),
            "timing": self._timing(features),
            "footwork_quality": footwork,
            "balance": features.balance_score,
            "fps": round(fps, 1),
            "pose_detected": features.pose_detected,
            "frame": features.frame_index,
        }
        self._metrics = metrics
        return metrics

    def _impact_prediction(self, features: FrameFeatures) -> str:
        """Predict how close contact is from the bat/ball closing distance."""
        if features.ball is None or features.bat_tip is None:
            return "Tracking..."
        distance = features.bat_ball_distance_px
        if not np.isfinite(distance):
            return "Tracking..."
        threshold = max(12.0, features.body_height_px * 0.06)
        if distance <= threshold:
            return "IMPACT"
        previous = [f for f in self._window if np.isfinite(f.bat_ball_distance_px)]
        if len(previous) >= 2:
            closing = previous[-2].bat_ball_distance_px - distance
            if closing > 1.0:
                frames_left = max(0.0, (distance - threshold) / closing)
                return f"in {frames_left / max(1.0, self.config.target_fps):.2f}s"
        return "Not imminent"

    def _timing(self, features: FrameFeatures) -> str:
        window = [f for f in self._window if f.pose_detected]
        if len(window) < 4:
            return "-"
        forward = [f.forward_pct for f in window]
        trend = forward[-1] - forward[max(0, len(forward) - 6)]
        if features.bat_speed_kmh > 25 and trend < -4:
            return "Late (weight going back)"
        if features.bat_speed_kmh > 25 and trend > 4:
            return "Well timed"
        return "Set"

    # ---------------------------------------------------------------- outputs
    def _publish(self, frame: np.ndarray) -> None:
        ok, buffer = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.config.jpeg_quality]
        )
        if ok:
            with self._lock:
                self._latest_jpeg = buffer.tobytes()

    def latest_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_jpeg

    def latest_metrics(self) -> Dict[str, object]:
        metrics = dict(self._metrics)
        metrics.setdefault("fps", round(self._fps, 1))
        return metrics

    def mjpeg_stream(self):
        """Multipart MJPEG generator suitable for a Flask ``Response``."""
        boundary = b"--frame\r\n"
        while self.running:
            self.touch()
            payload = self.latest_jpeg()
            if payload is None:
                time.sleep(0.05)
                continue
            yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + payload + b"\r\n"
            time.sleep(1.0 / max(1, self.config.target_fps))


class RealtimeManager:
    """Process-wide registry of live analysers (one per camera source)."""

    def __init__(self) -> None:
        self._analyzers: Dict[str, RealtimeAnalyzer] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _key(source: Union[int, str]) -> str:
        return str(source)

    def get_or_start(self, source: Union[int, str] = 0, angle: str = "A") -> RealtimeAnalyzer:
        key = self._key(source)
        with self._lock:
            analyzer = self._analyzers.get(key)
            if analyzer is not None and analyzer.running:
                analyzer.angle = angle
                analyzer.touch()
                return analyzer
            analyzer = RealtimeAnalyzer(source=source, angle=angle)
            analyzer.start()
            self._analyzers[key] = analyzer
            return analyzer

    def get(self, source: Union[int, str] = 0) -> Optional[RealtimeAnalyzer]:
        return self._analyzers.get(self._key(source))

    def stop(self, source: Union[int, str] = 0) -> None:
        with self._lock:
            analyzer = self._analyzers.pop(self._key(source), None)
        if analyzer is not None:
            analyzer.stop()

    def stop_all(self) -> None:
        with self._lock:
            analyzers = list(self._analyzers.values())
            self._analyzers.clear()
        for analyzer in analyzers:
            analyzer.stop()


REALTIME_MANAGER = RealtimeManager()

__all__ = ["RealtimeAnalyzer", "RealtimeManager", "REALTIME_MANAGER", "CLASSIFY_EVERY"]
