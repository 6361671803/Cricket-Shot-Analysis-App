# Advanced Analysis Engine

The Flask app, its pages and its routes are unchanged. Everything that decides
*what happened in the shot* now lives in the `analysis_engine` package, which
replaces the previous single-frame, joint-angle rules.

```
frames ─► pose ─► bat/ball tracking ─► per-frame features ─► impact detection
       ─► temporal swing features ─► shot classification ─► weight transfer
       ─► technical metrics ─► scores ─► coaching ─► annotation / PDF report
```

## Modules

| Module | Responsibility |
| --- | --- |
| `config.py` | Every tunable constant, overridable by environment variable (`CRICKET_*`). |
| `datatypes.py` | Small serialisable value objects (`PoseFrame`, `FrameFeatures`, `ImpactInfo`, `ShotPrediction`, `WeightTransferResult`, `AnalysisResult`). |
| `pose/estimator.py` | 3D pose via the MediaPipe **Pose Landmarker** task (BlazePose GHUM, world landmarks), with the legacy `mp.solutions.pose` API as a fallback. Downloads the `.task` model into `models/` on first use. |
| `pose/landmarks.py` | Landmark indices, skeleton connections and angle/geometry helpers. |
| `tracking/bat_tracker.py` | Bat grip + toe tracking (grip-anchored ROI, Canny + Hough line evidence, anthropometric fallback, temporal smoothing) and swing-path metrics: path, speed, swing plane, arc, backlift, bat face, follow-through, sweet spot. |
| `tracking/ball_tracker.py` | Ball detection (frame differencing, circularity, red/white colour likelihood), constant-velocity prediction, gap interpolation, longest-consistent-tracklet filtering, then speed, bounce frame, impact point, post-impact direction and predicted path. |
| `metrics/kinematics.py` | Pure biomechanics helpers: segment-mass centre of mass, foot-line projection, rotation about the vertical axis, knee flexion, heel lift, wrist rotation, smoothing, velocity series. |
| `features.py` | Turns pose + bat + ball into one `FrameFeatures` record per frame: weight distribution, COM, head/hip/shoulder centres, knee flexion, heel lift, rotations and separation, spine lean, elbow angles, wrist roll, stride, foot direction/pivot, bat angle/speed, ball speed, bat–ball distance, balance, head travel, motion energy. |
| `impact.py` | Impact frame from five fused cues — bat/ball proximity, bat velocity peak, ball direction change, bat deceleration and whole-body motion energy — plus contact quality, sweet-spot offset, impact position and a plain-English reason. |
| `weight_transfer.py` | Centre-of-mass based transfer: per-frame front/back %, front/back/balanced label, early/on-time/late timing relative to impact, quality band, transfer rate, coaching notes and a PNG movement graph. |
| `classifier/shot_definitions.py` | Fuzzy templates for all **20** shots plus the aliases used to look shots up in `shots.json`. |
| `classifier/swing_features.py` | Aggregates the frames around impact into one shot signature (contact height/offset, swing plane and direction, rotation, wrist roll, follow-through, backlift, stride, reverse/switch flags). Features that need a sequence are omitted for single photos. |
| `classifier/shot_classifier.py` | Fuzzy membership → weighted template score → softmax confidence, damped by data quality, with a reason and ranked alternatives. |
| `metrics/technical.py` | The full technical metric set and the ordered rows used by the page and the PDF. |
| `scoring.py` | Ten 0–100 scores: footwork, balance, weight transfer, timing, head position, shot execution, follow-through, power, control, overall technique. |
| `coaching.py` | Strengths, weaknesses, coaching cues, practice drills and a one-line summary. |
| `comparison.py` | Benchmarks against Virat Kohli, Steve Smith, Babar Azam, Joe Root and Kane Williamson, with per-metric gaps and a match percentage. |
| `output/annotate.py` | Skeleton, bat path, ball path, predicted path, impact marker, centre of gravity, weight-transfer arrow, hip/shoulder rotation arcs, shot name/confidence and the live HUD. |
| `output/report.py` | The PDF report (summary, annotated frame, graph, scores, metrics, coaching, comparison). |
| `realtime.py` | Threaded live analysis: capture thread (newest frame only), analysis thread (pose → trackers → rolling classification → annotated JPEG) and lock-free reads for HTTP. Adaptive frame-skip holds the target FPS. |
| `pipeline.py` | The façade: `analyze_image()` and `analyze_video()`. Video decoding runs in a prefetch thread; the annotated video is transcoded to H.264 when ffmpeg is available. |

## Shots recognised

Forward Defence, Back Foot Defence, Straight Drive, Cover Drive, On Drive,
Lofted Drive, Pull Shot, Hook Shot, Cut Shot, Square Cut, Upper Cut, Sweep,
Reverse Sweep, Paddle Sweep, Flick, Glance, Late Cut, Ramp Shot, Switch Hit,
Helicopter Shot.

Every prediction carries a **name**, a **confidence %**, a **reason** and ranked
**alternatives**. `shots.json` gained library entries for the six shots it did
not previously describe; existing entries were not modified.

## How the app uses it

```python
from analysis_engine import ShotAnalysisPipeline

pipeline = ShotAnalysisPipeline(static_image_mode=True)
result = pipeline.analyze_image(frame, angle="A", annotated_path=..., graph_path=...)
result.shot.name, result.shot.confidence, result.scores, result.coaching
```

* `POST /analyze` — unchanged contract; now accepts a **video** in the same field
  as well as a photo, and renders the same page with extra sections appended.
* `POST /update_shot` — unchanged; when an `analysis_id` is posted the measured
  metrics are preserved and only the shot identity changes.
* `GET /report/<id>` — PDF export.
* `GET /api/analysis/<id>`, `GET /api/shots` — JSON.
* `GET /live`, `/live/stream`, `/live/metrics`, `POST /live/stop` — real-time
  camera analysis (webcam index, USB camera index, or any OpenCV source URL).

## Performance notes

* The pose model is built once per process and guarded by a lock (MediaPipe
  graphs are not thread-safe); HTTP threads never block on each other's
  inference.
* Video decode overlaps inference through a bounded prefetch queue; uploads are
  capped by `CRICKET_MAX_VIDEO_FRAMES` (default 900).
* Live mode drops stale frames rather than queueing them, and skips analysis
  frames adaptively to keep the configured 20–30 FPS target.
* `CRICKET_USE_GPU=1` requests the MediaPipe GPU delegate; the engine falls back
  to CPU automatically if it is unavailable.

Measured on the development machine (CPU only): ~0.2 s for a photo and ~32
analysed frames/second for a 30 fps clip.

## Environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `CRICKET_POSE_TASK` | `models/pose_landmarker_full.task` | Pose Landmarker bundle for uploads (live mode uses the `lite` bundle beside it). |
| `CRICKET_POSE_COMPLEXITY` / `_RT` | `1` / `0` | Model complexity for uploads / live mode. |
| `CRICKET_USE_GPU` | `0` | Use the GPU delegate when available. |
| `CRICKET_TARGET_FPS` | `25` | Live-analysis target frame rate. |
| `CRICKET_CAM_WIDTH` / `CRICKET_CAM_HEIGHT` | `960` / `540` | Capture resolution. |
| `CRICKET_MAX_VIDEO_FRAMES` | `900` | Frame cap for uploaded videos. |
| `CRICKET_BODY_HEIGHT_M` | `1.75` | Assumed batter height, used for the pixel→metre scale. |
