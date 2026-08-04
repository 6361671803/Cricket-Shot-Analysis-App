# Cricket-Shot-Analysis-App
🏏 Cricket Shot Analysis &amp; Coaching Assistant — Pose Estimation + AI Insights  This project analyzes a batter’s impact frame during a cricket shot using MediaPipe pose estimation and provides instant coaching feedback based on professional batting techniques.
##video :https://youtu.be/qRNHSJfaOVo

## Run it

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt   # Windows: .venv\Scripts\pip
.venv/bin/python app.py                     # http://127.0.0.1:5000
```

The pose model (~9 MB `.task` bundle) downloads into `models/` on the first run.

## Analysis engine

The UI and routes are unchanged, but the analysis behind them is now the modular
`analysis_engine` package: 3D pose (MediaPipe Pose Landmarker / BlazePose GHUM),
bat and ball tracking, a temporal 20-shot classifier with confidence and
reasons, multi-cue impact-frame detection, centre-of-mass weight transfer with a
movement graph, the full technical metric set, ten 0–100 scores, AI coaching,
professional comparison, video overlays, a PDF report and real-time camera
analysis at `/live`. `POST /analyze` also accepts a video of the shot, not just
the impact photo.

See [docs/ANALYSIS_ENGINE.md](docs/ANALYSIS_ENGINE.md) for the module-by-module
description and the configuration options.
