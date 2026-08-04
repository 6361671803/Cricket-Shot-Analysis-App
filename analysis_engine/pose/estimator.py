"""3D pose estimation with automatic MediaPipe backend selection.

Two backends are supported and probed in order:

1. **Pose Landmarker (MediaPipe Tasks API)** – the modern BlazePose GHUM
   ``.task`` bundle.  Gives metric *world* landmarks (3D, in metres) and is the
   only backend available in MediaPipe >= 1.0 where ``mp.solutions`` was
   removed.  The model is downloaded once into ``models/`` on first use.
2. **``mp.solutions.pose`` (legacy solution API)** – used when the Tasks API or
   the model bundle is unavailable (e.g. MediaPipe 0.10 without network access).

Both backends are normalised into :class:`~analysis_engine.datatypes.PoseFrame`,
so the rest of the engine is backend agnostic.
"""

from __future__ import annotations

import logging
import os
import threading
import urllib.request
from typing import Optional

import cv2
import numpy as np

from ..config import CONFIG, PoseConfig
from ..datatypes import PoseFrame
from . import landmarks as lms

LOGGER = logging.getLogger(__name__)

MODEL_URLS = {
    "pose_landmarker_full.task": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
    ),
    "pose_landmarker_lite.task": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    ),
    "pose_landmarker_heavy.task": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
    ),
}

_DOWNLOAD_LOCK = threading.Lock()


def ensure_model(path: str, timeout: float = 60.0) -> Optional[str]:
    """Return a local path to the ``.task`` bundle, downloading it if needed."""
    if path and os.path.exists(path):
        return path
    name = os.path.basename(path or "")
    url = MODEL_URLS.get(name)
    if url is None:
        return None
    with _DOWNLOAD_LOCK:
        if os.path.exists(path):
            return path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = f"{path}.part"
        try:
            LOGGER.info("Downloading pose model %s", name)
            with urllib.request.urlopen(url, timeout=timeout) as response, open(tmp, "wb") as fh:
                fh.write(response.read())
            os.replace(tmp, path)
            return path
        except Exception as exc:  # pragma: no cover - network dependent
            LOGGER.warning("Could not download pose model %s: %s", name, exc)
            if os.path.exists(tmp):
                os.remove(tmp)
            return None


class PoseEstimator:
    """Thread-confined pose estimator producing 2D pixel + 3D world landmarks.

    A single instance is **not** safe to share between threads; create one per
    worker (``RealtimeAnalyzer`` and the video pipeline each own theirs).
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        config: Optional[PoseConfig] = None,
        realtime: bool = False,
    ) -> None:
        self.config = config or CONFIG.pose
        self.static_image_mode = static_image_mode
        self.complexity = (
            self.config.realtime_model_complexity if realtime else self.config.model_complexity
        )
        self.backend = "none"
        self._landmarker = None
        self._legacy = None
        self._mp_image_cls = None
        self._frame_counter = 0
        # The Pose Landmarker's VIDEO mode requires strictly increasing
        # timestamps for the lifetime of the landmarker, so timestamps of
        # successive clips are offset past the previous clip instead of
        # restarting at zero.
        self._stamp_offset_ms = 0
        self._last_stamp_ms = -1
        self._init_backend()

    # ------------------------------------------------------------------ setup
    def _init_backend(self) -> None:
        if self._init_tasks_backend():
            return
        if self._init_legacy_backend():
            return
        LOGGER.error("No MediaPipe pose backend available; pose estimation disabled.")

    def _init_tasks_backend(self) -> bool:
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
        except Exception as exc:  # pragma: no cover - import guard
            LOGGER.debug("Tasks API unavailable: %s", exc)
            return False

        wanted = self.config.landmarker_task_path
        if self.complexity == 0:
            wanted = os.path.join(os.path.dirname(wanted), "pose_landmarker_lite.task")
        elif self.complexity >= 2:
            wanted = os.path.join(os.path.dirname(wanted), "pose_landmarker_heavy.task")
        model_path = ensure_model(wanted) or ensure_model(self.config.landmarker_task_path)
        if model_path is None:
            return False

        try:
            delegate = (
                mp_python.BaseOptions.Delegate.GPU
                if self.config.use_gpu
                else mp_python.BaseOptions.Delegate.CPU
            )
            base = mp_python.BaseOptions(model_asset_path=model_path, delegate=delegate)
            running_mode = (
                mp_vision.RunningMode.IMAGE
                if self.static_image_mode
                else mp_vision.RunningMode.VIDEO
            )
            options = mp_vision.PoseLandmarkerOptions(
                base_options=base,
                running_mode=running_mode,
                num_poses=1,
                min_pose_detection_confidence=self.config.min_detection_confidence,
                min_pose_presence_confidence=self.config.min_detection_confidence,
                min_tracking_confidence=self.config.min_tracking_confidence,
                output_segmentation_masks=False,
            )
            self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)
            self._mp_image_cls = mp.Image
            self._mp_image_format = mp.ImageFormat.SRGB
            self.backend = "pose_landmarker"
            LOGGER.info("Pose backend: MediaPipe Pose Landmarker (%s)", os.path.basename(model_path))
            return True
        except Exception as exc:  # pragma: no cover - runtime guard
            LOGGER.warning("Pose Landmarker init failed (%s); trying legacy backend", exc)
            self._landmarker = None
            return False

    def _init_legacy_backend(self) -> bool:
        try:
            import mediapipe as mp

            solutions = getattr(mp, "solutions", None)
            if solutions is None or not hasattr(solutions, "pose"):
                return False
            self._legacy = solutions.pose.Pose(
                static_image_mode=self.static_image_mode,
                model_complexity=self.complexity,
                enable_segmentation=False,
                min_detection_confidence=self.config.min_detection_confidence,
                min_tracking_confidence=self.config.min_tracking_confidence,
            )
            self.backend = "mp_solutions_pose"
            LOGGER.info("Pose backend: mp.solutions.pose (BlazePose GHUM)")
            return True
        except Exception as exc:  # pragma: no cover - runtime guard
            LOGGER.warning("Legacy pose backend init failed: %s", exc)
            return False

    # ----------------------------------------------------------------- public
    def process(self, frame_bgr: np.ndarray, frame_index: int, timestamp: float) -> PoseFrame:
        """Run pose estimation on one BGR frame."""
        height, width = frame_bgr.shape[:2]
        empty = PoseFrame(
            index=frame_index,
            timestamp=timestamp,
            pixels=np.zeros((0, 2)),
            world=np.zeros((0, 3)),
            visibility=np.zeros((0,)),
            width=width,
            height=height,
        )
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if self._landmarker is not None:
            return self._process_tasks(rgb, frame_index, timestamp, width, height) or empty
        if self._legacy is not None:
            return self._process_legacy(rgb, frame_index, timestamp, width, height) or empty
        return empty

    def reset(self) -> None:
        """Start a new clip on the same estimator.

        In VIDEO mode the landmarker carries tracking state and demands strictly
        increasing timestamps, so it is rebuilt: without this, a second clip
        either inherits the previous clip's tracking or is rejected for going
        back in time.
        """
        self._frame_counter = 0
        self._stamp_offset_ms = 0
        self._last_stamp_ms = -1
        if self.static_image_mode or self._landmarker is None:
            return
        self.close()
        self._init_backend()

    def close(self) -> None:
        for obj in (self._landmarker, self._legacy):
            try:
                if obj is not None:
                    obj.close()
            except Exception:  # pragma: no cover
                pass
        self._landmarker = None
        self._legacy = None

    def __enter__(self) -> "PoseEstimator":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # ---------------------------------------------------------------- helpers
    def _process_tasks(
        self, rgb: np.ndarray, frame_index: int, timestamp: float, width: int, height: int
    ) -> Optional[PoseFrame]:
        image = self._mp_image_cls(image_format=self._mp_image_format, data=rgb)
        try:
            if self.static_image_mode:
                result = self._landmarker.detect(image)
            else:
                self._frame_counter += 1
                stamp_ms = max(
                    int(timestamp * 1000.0) + self._stamp_offset_ms,
                    self._last_stamp_ms + 1,
                )
                self._last_stamp_ms = stamp_ms
                result = self._landmarker.detect_for_video(image, stamp_ms)
        except Exception as exc:  # pragma: no cover - runtime guard
            LOGGER.debug("Pose Landmarker inference failed: %s", exc)
            return None

        if not result or not result.pose_landmarks:
            return None
        norm = result.pose_landmarks[0]
        world_lms = result.pose_world_landmarks[0] if result.pose_world_landmarks else None

        pixels = np.array([[lm.x * width, lm.y * height] for lm in norm], dtype=float)
        visibility = np.array(
            [getattr(lm, "visibility", None) or getattr(lm, "presence", 0.0) or 0.0 for lm in norm],
            dtype=float,
        )
        if world_lms is not None:
            world = np.array([[lm.x, lm.y, lm.z] for lm in world_lms], dtype=float)
        else:
            world = self._world_from_pixels(pixels, norm, width, height)
        return PoseFrame(
            index=frame_index,
            timestamp=timestamp,
            pixels=pixels,
            world=world,
            visibility=visibility,
            width=width,
            height=height,
        )

    def _process_legacy(
        self, rgb: np.ndarray, frame_index: int, timestamp: float, width: int, height: int
    ) -> Optional[PoseFrame]:
        result = self._legacy.process(rgb)
        if not result.pose_landmarks:
            return None
        norm = result.pose_landmarks.landmark
        pixels = np.array([[lm.x * width, lm.y * height] for lm in norm], dtype=float)
        visibility = np.array([float(lm.visibility) for lm in norm], dtype=float)
        if getattr(result, "pose_world_landmarks", None):
            world = np.array(
                [[lm.x, lm.y, lm.z] for lm in result.pose_world_landmarks.landmark], dtype=float
            )
        else:
            world = self._world_from_pixels(pixels, norm, width, height)
        return PoseFrame(
            index=frame_index,
            timestamp=timestamp,
            pixels=pixels,
            world=world,
            visibility=visibility,
            width=width,
            height=height,
        )

    @staticmethod
    def _world_from_pixels(pixels: np.ndarray, norm, width: int, height: int) -> np.ndarray:
        """Approximate metric world landmarks when the backend omits them.

        The batter's pixel height is mapped onto an assumed real body height so
        3D-derived quantities (centre of mass, stride length) stay usable.
        """
        scale_px = max(1.0, lms.body_height_px(pixels))
        scale = CONFIG.biomech.assumed_body_height_m / scale_px
        hip_center = lms.midpoint(pixels, lms.LEFT_HIP, lms.RIGHT_HIP)
        z_norm = np.array([getattr(lm, "z", 0.0) for lm in norm], dtype=float)
        world = np.zeros((len(pixels), 3), dtype=float)
        world[:, 0] = (pixels[:, 0] - hip_center[0]) * scale
        world[:, 1] = (pixels[:, 1] - hip_center[1]) * scale
        world[:, 2] = z_norm * width * scale
        return world


__all__ = ["PoseEstimator", "ensure_model", "MODEL_URLS"]
