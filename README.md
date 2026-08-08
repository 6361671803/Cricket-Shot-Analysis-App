# Cricket Shot Analysis App 🏏

A Flask app that analyzes a batter's technique from an ordinary photo or slow-mo phone video — pose estimation, weight transfer, bat/ball tracking, biomechanics scoring, shot classification, and session history — with no specialized hardware, just a camera and free/open-source software.

> Early demo (single-photo version, before most of what's described below existed)

## What it does

- **Photo or video input.** Upload a JPG/PNG/WEBP impact photo, or a short MP4/MOV slow-mo clip; a live camera-capture option is also available for photos.
- **Pose analysis.** MediaPipe pose estimation on the impact frame (or across the whole clip for video), with a skeleton overlay and front/back-foot weight-transfer percentage.
- **Video sequence analysis.** For video, the clip is segmented into stance → backlift → downswing → impact → follow-through using a wrist-speed heuristic, with a weight-transfer timeline chart.
- **Bat & ball tracking (video only, camera-only).** Track the bat via a printed ArUco marker or bright tape, and the ball via motion + Hough-circle detection. Produces backlift/downswing angle, bat speed, point of contact, and a rough ball line/length estimate. An optional two-point calibration step converts pixel measurements to real-world cm/km-h.
- **Shot classification, three tiers, in order:**
  1. **Local ML model** (`shot_classifier_ml.py`) — a RandomForest classifier over hand-crafted pose features, trainable on your own labeled clips via `train_shot_classifier.py`. Ships untrained; the app works fully without it.
  2. **LLM fallback** (`shot_classifier_llm.py`) — Google Gemini vision API, used only when the local model isn't trained/confident. Optional; disabled automatically with no API key configured.
  3. **Rule-based classifier** (`shots.py`) — a hand-written heuristic over bat angle and camera view. Always available, works fully offline, the guaranteed fallback.
- **Biomechanics scoring** (`biomechanics.py`) — head-over-front-knee offset, front-knee flex, elbow extension, shoulder/hip rotation, and bat speed, compared against illustrative technique-archetype benchmark ranges (`benchmarks.json`) with an under/within/over verdict per metric. These are coaching heuristics, not validated sports-science data.
- **Shot library** — a browsable reference of 24 shot types (classical and modern) with summaries, field placement, master players, and coaching tips (`shots.json`).
- **Session history** (`db.py`, SQLite) — every analysis is saved automatically. A History view shows recent sessions, most-played shots, a biomechanics-score trend, and your strongest/weakest shot by average score.
- **Match-context tagging** — optionally tag an analyzed shot with over number, bowler type, line/length, and outcome (runs/dismissal); tagged shots populate an illustrative wagon-wheel shot map.
- **Shot correction** — if a detected shot is wrong, correct it from a dropdown; the correction is logged (`dataset/shot_corrections.jsonl`) and syncs back into session history.

## How it's built

Flask + Jinja templates, vanilla JS/CSS (no frontend framework, no build step). A single-page tabbed UI (`templates/index.html`: Analyze / Biomechanics / History / Shot Library) drives everything through a JSON API (`/api/analyze`, `/api/history`); a server-rendered result page (`templates/result.html`) remains as a working no-JS fallback for the upload form.

| File | Responsibility |
|---|---|
| `app.py` | Flask routes, request handling, wiring everything together |
| `pose_utils.py` | Shared pose-landmark math (weight transfer, mistake detection, overlay drawing) |
| `video_analysis.py` | Video pose-sequence extraction, swing segmentation |
| `bat_tracking.py` / `ball_tracking.py` | Camera-only bat and ball tracking |
| `calibration.py` | Pixel-to-real-world scale conversion |
| `biomechanics.py` / `benchmarks.json` | Biomechanics metrics + benchmark comparison |
| `shots.py` / `shots.json` | Shot database + rule-based classifier |
| `shot_classifier_ml.py` / `train_shot_classifier.py` | Trainable local shot classifier |
| `shot_classifier_llm.py` | Gemini-based LLM classification fallback |
| `db.py` | SQLite session persistence |
| `shot_map.py` | Wagon-wheel shot-map data |
| `label_clips.py` | Interactive tool to turn raw footage into labeled training clips |

## Setup

Requires **Python 3.9+** (the Gemini SDK's floor).

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

`mediapipe` is pinned to `0.10.9` (newer releases drop the drawing API this app's skeleton overlay depends on), and `opencv-python` is capped below `5.x` (untested against the ArUco/Hough-circle/video code here). The MediaPipe pose model (`pose_landmarker.task`) is downloaded automatically on first run.

## Configuration (optional)

Copy `.env.example` to `.env` to enable the LLM classification fallback:

```bash
cp .env.example .env
```

Get a free Gemini API key (no credit card required) at https://aistudio.google.com/app/apikey, and set `GEMINI_API_KEY` in `.env`. Without a key, the app works fully — it just skips straight to the rule-based classifier. See `.env.example` for all available settings (model, call cap, enable/disable toggle).

**Never commit `.env`** — it's gitignored. `.env.example` should only ever contain placeholder values.

## Running

```bash
python app.py
```

Open http://127.0.0.1:5000.

## Training the local classifier

The local ML classifier ships untrained — `data/` starts empty. To train it on your own footage:

1. Record short slow-mo clips of your shots (side-on angle works best — see `data/README.md` for filming and clip-count guidance).
2. Label and trim them into `data/<shot_name>/*.mp4` with the interactive tool:
   ```bash
   python label_clips.py path/to/your_recording.mp4
   ```
3. Train:
   ```bash
   python train_shot_classifier.py
   ```

This saves `models/shot_classifier.joblib`; `classify_shot_ml()` picks it up automatically on the next run.

## Limitations

- Shot classification accuracy depends on which tier fires: the rule-based tier is a crude heuristic (7 of 24 shot names only), the LLM tier depends on a working API key and Google's model availability, and the local model is only as good as the clips you train it on.
- Weight-transfer and biomechanics readings need the front and back foot to be visibly separated in frame (a side-on camera angle) — near-front-on footage is detected and reported as unreliable rather than guessed.
- Biomechanics benchmark ranges are illustrative coaching heuristics grouped by technique archetype, not validated sports-science data.
- The wagon-wheel shot map is an illustrative shot-type-to-field-position mapping, not derived from real ball-tracking/trajectory data.
- Bat/ball tracking requires a visible ArUco marker or bright tape on the bat; ball detection is motion-based and best-effort.
- The LLM fallback sends the analyzed photo to Google's Gemini API — a real privacy trade-off worth being aware of.

## License

MIT — see [LICENSE](LICENSE).
