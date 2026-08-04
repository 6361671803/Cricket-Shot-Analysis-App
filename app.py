"""Cricket Shot Analyzer web app.

The UI, routes and workflow are unchanged; the analysis behind them is now the
modular :mod:`analysis_engine` pipeline (3D pose, bat/ball tracking, temporal
20-shot classifier, multi-cue impact detection, centre-of-mass weight transfer,
technical metrics, 0-100 scores, coaching, professional comparison, PDF report
and real-time camera analysis).
"""

import atexit
import base64
import json
import os
import threading
import uuid
from collections import OrderedDict

import cv2
import numpy as np
from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    send_file,
    url_for,
)
from werkzeug.utils import secure_filename

from analysis_engine import (
    PRO_PLAYERS,
    REALTIME_MANAGER,
    SUPPORTED_SHOTS,
    ShotAnalysisPipeline,
    build_pdf_report,
    score_label,
)
from analysis_engine.classifier import SHOT_DB_ALIASES

# ---------------- Flask setup ----------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
# The page templates live next to app.py in this project, not in templates/.
app = Flask(__name__, template_folder=BASE_DIR)

UPLOAD_FOLDER = os.path.join("static", "uploads")
OUTPUT_FOLDER = os.path.join("static", "outputs")
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm", "m4v"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -------------- Load shots database ----------
with open("shots.json", "r") as f:
    SHOT_DB = json.load(f)

# -------------- Analysis engine --------------
# One pipeline per process (the pose model is expensive to build) guarded by a
# lock, because MediaPipe graphs are not safe to call from several threads.
ENGINE = ShotAnalysisPipeline(static_image_mode=True)
VIDEO_ENGINE = ShotAnalysisPipeline(static_image_mode=False)
ENGINE_LOCK = threading.Lock()


@atexit.register
def _close_engines() -> None:
    """Release the pose graphs while MediaPipe is still usable.

    Left to the garbage collector they are finalised during interpreter
    shutdown, which makes MediaPipe print a confusing teardown traceback.
    """
    ENGINE.close()
    VIDEO_ENGINE.close()


#: Recent analyses, kept so the report/metrics endpoints can be served later.
ANALYSES: "OrderedDict[str, dict]" = OrderedDict()
MAX_CACHED_ANALYSES = 40


def remember_analysis(analysis_id: str, payload: dict) -> None:
    ANALYSES[analysis_id] = payload
    while len(ANALYSES) > MAX_CACHED_ANALYSES:
        ANALYSES.popitem(last=False)


# -------------- Helpers ----------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def is_video_file(filename: str) -> bool:
    return (
        "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS
    )


def shot_db_key(shot_name: str) -> str:
    """Map an engine shot name onto a key that exists in ``shots.json``."""
    if shot_name in SHOT_DB:
        return shot_name
    alias = SHOT_DB_ALIASES.get(shot_name)
    if alias in SHOT_DB:
        return alias
    return "Straight Drive"


def get_shot_info_from_db(shot_key: str):
    """Return shot info fields for templates."""
    data = SHOT_DB.get(shot_key, SHOT_DB["Straight Drive"])

    shot_name = shot_key
    shot_summary = data.get("summary", "")
    field_suggestion = data.get("fields", "")
    variations = data.get("variations", [])
    master_player = data.get("masters", "")
    shot_history = data.get("history", "")
    improvement_summary = data.get("improve", "")
    alt_safe = data.get("alt_safe", [])
    alt_aggressive = data.get("alt_aggressive", [])
    final_feedback = data.get("final_feedback", [])

    return (
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
    )


def rating_from_scores(result) -> tuple:
    """Derive the existing rating label / encouragement from the new scores."""
    overall = int(result.scores.get("overall_technique", 0))
    if overall >= 80:
        return (
            "Excellent Shot",
            "Super balance and weight transfer. This is a high-quality shot! 🏏🔥",
        )
    if overall >= 65:
        return (
            "Good Shot",
            "Good mechanics overall. Tighten the details below and this becomes a top shot.",
        )
    if overall >= 50:
        return (
            "Okay / Needs Improvement",
            "You’re close. Work through the coaching points below and it will click.",
        )
    return (
        "Needs Improvement",
        "Plenty to build on — start with the two biggest weaknesses listed below.",
    )


def static_url(path: str):
    """Turn a ``static/...`` filesystem path into a URL, if it exists."""
    if not path:
        return None
    normalised = path.replace(os.sep, "/")
    marker = "static/"
    if marker in normalised:
        return url_for("static", filename=normalised.split(marker, 1)[1])
    return None


def build_result_context(result, analysis_id: str, angle: str) -> dict:
    """Legacy template variables plus the new, additive analysis detail."""
    db_key = shot_db_key(result.shot.name)
    (
        db_shot_name,
        shot_summary,
        field_suggestion,
        variations,
        master_player,
        shot_history,
        improvement_summary,
        alt_safe,
        alt_aggressive,
        final_feedback,
    ) = get_shot_info_from_db(db_key)

    rating_label, encouragement = rating_from_scores(result)
    mistakes = list(result.coaching.get("weaknesses", [])) + list(result.warnings)
    impact = result.impact
    weight = result.weight_transfer

    return {
        # ---- original template variables (unchanged names/semantics) ----
        "image_url": static_url(result.annotated_image_path),
        "mistakes": mistakes,
        "is_good": rating_label in ["Excellent Shot", "Good Shot"],
        "forward_pct": weight.forward_pct,
        "back_pct": weight.back_pct,
        "rating_label": rating_label,
        "encouragement": encouragement,
        "shot_name": result.shot.name,
        "shot_summary": shot_summary,
        "field_suggestion": field_suggestion,
        "variations": variations,
        "master_player": master_player,
        "shot_history": shot_history,
        "improvement_summary": improvement_summary,
        "alt_safe": alt_safe,
        "alt_aggressive": alt_aggressive,
        "final_feedback": final_feedback,
        "all_shots": all_shot_options(),
        "current_shot": db_shot_name,
        # ---- new, additive detail ----
        "analysis_id": analysis_id,
        "angle": angle,
        "confidence": round(result.shot.confidence, 1),
        "shot_reason": result.shot.reason,
        "shot_alternatives": result.shot.alternatives,
        "impact": impact,
        "weight": weight,
        "scores": result.scores,
        "score_bands": {k: score_label(int(v)) for k, v in result.scores.items()},
        "metric_rows": result.metrics.get("display", []),
        "coaching": result.coaching,
        "comparison": result.comparison,
        "graph_url": static_url(result.graph_path),
        "video_url": static_url(result.annotated_video_path),
        "report_url": url_for("report", analysis_id=analysis_id),
        "frames_analysed": result.frames_analysed,
        "fps": result.fps,
        "source": result.source,
        "pro_players": list(PRO_PLAYERS),
    }


def all_shot_options() -> list:
    """Every shot the engine knows, plus any extra keys already in the database."""
    options = list(SUPPORTED_SHOTS)
    options.extend(key for key in SHOT_DB if key not in options)
    return options


# -------------- Routes -----------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    angle = request.form.get("angle", "A") or "A"
    pro_player = request.form.get("pro_player") or None

    file = request.files.get("image") or request.files.get("video")
    camera_data = request.form.get("image_from_camera")

    analysis_id = uuid.uuid4().hex
    annotated_path = os.path.join(OUTPUT_FOLDER, f"{analysis_id}.jpg")
    graph_path = os.path.join(OUTPUT_FOLDER, f"{analysis_id}_weight.png")
    frame = None
    video_path = None

    # 1) camera photo (base64 from hidden input)
    if camera_data:
        try:
            _, encoded = camera_data.split(",", 1)
            np_arr = np.frombuffer(base64.b64decode(encoded), np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as exc:
            app.logger.warning("Camera decode error: %s", exc)
            frame = None

    # 2) uploaded photo or video
    elif file and file.filename:
        safe_name = secure_filename(f"{analysis_id}_{file.filename}")
        saved_path = os.path.join(UPLOAD_FOLDER, safe_name)
        if allowed_file(file.filename):
            file.save(saved_path)
            frame = cv2.imread(saved_path)
        elif is_video_file(file.filename):
            file.save(saved_path)
            video_path = saved_path

    if frame is None and video_path is None:
        return "No valid image received. Please upload or capture again."

    with ENGINE_LOCK:
        if video_path is not None:
            result = VIDEO_ENGINE.analyze_video(
                video_path,
                angle=angle,
                annotated_video_path=os.path.join(OUTPUT_FOLDER, f"{analysis_id}.mp4"),
                annotated_image_path=annotated_path,
                graph_path=graph_path,
                pro_player=pro_player,
            )
        else:
            result = ENGINE.analyze_image(
                frame,
                angle=angle,
                annotated_path=annotated_path,
                graph_path=graph_path,
                pro_player=pro_player,
            )

    context = build_result_context(result, analysis_id, angle)
    remember_analysis(analysis_id, {"result": result, "context": context})
    return render_template("result.html", **context)


# ---------- shot correction route ----------
@app.route("/update_shot", methods=["POST"])
def update_shot():
    correct_shot = request.form.get("correct_shot")
    image_url = request.form.get("image_url")
    analysis_id = request.form.get("analysis_id")

    if not correct_shot or correct_shot not in all_shot_options():
        correct_shot = "Straight Drive"

    cached = ANALYSES.get(analysis_id or "")
    if cached is not None:
        # Keep every measured metric; only the shot identity was corrected.
        context = dict(cached["context"])
        (
            db_shot_name,
            shot_summary,
            field_suggestion,
            variations,
            master_player,
            shot_history,
            improvement_summary,
            alt_safe,
            alt_aggressive,
            final_feedback,
        ) = get_shot_info_from_db(shot_db_key(correct_shot))
        context.update(
            shot_name=correct_shot,
            current_shot=db_shot_name,
            shot_summary=shot_summary,
            field_suggestion=field_suggestion,
            variations=variations,
            master_player=master_player,
            shot_history=shot_history,
            improvement_summary=improvement_summary,
            alt_safe=alt_safe,
            alt_aggressive=alt_aggressive,
            final_feedback=final_feedback,
            rating_label="Shot Updated ✔",
            encouragement="Shot information updated based on your selection.",
        )
        return render_template("result.html", **context)

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
    ) = get_shot_info_from_db(shot_db_key(correct_shot))

    # We keep weight transfer & mistakes neutral in update mode
    return render_template(
        "result.html",
        image_url=image_url,
        mistakes=[],
        is_good=False,
        forward_pct=0,
        back_pct=0,
        rating_label="Shot Updated ✔",
        encouragement="Shot information updated based on your selection.",
        shot_name=correct_shot,
        shot_summary=shot_summary,
        field_suggestion=field_suggestion,
        variations=variations,
        master_player=master_player,
        shot_history=shot_history,
        improvement_summary=improvement_summary,
        alt_safe=alt_safe,
        alt_aggressive=alt_aggressive,
        final_feedback=final_feedback,
        all_shots=all_shot_options(),
        current_shot=shot_name,
    )


# ---------- NEW: PDF report ----------
@app.route("/report/<analysis_id>", methods=["GET"])
def report(analysis_id: str):
    cached = ANALYSES.get(analysis_id)
    if cached is None:
        return "That analysis is no longer available. Please analyse the shot again.", 404
    pdf_path = os.path.join(OUTPUT_FOLDER, f"{analysis_id}_report.pdf")
    if not os.path.exists(pdf_path):
        result = cached["result"]
        shot_info = SHOT_DB.get(shot_db_key(result.shot.name), {})
        if build_pdf_report(result, pdf_path, shot_info) is None:
            return "PDF export needs the 'reportlab' package installed.", 501
    return send_file(pdf_path, as_attachment=True, download_name=f"shot_report_{analysis_id}.pdf")


# ---------- NEW: JSON analysis result ----------
@app.route("/api/analysis/<analysis_id>", methods=["GET"])
def api_analysis(analysis_id: str):
    cached = ANALYSES.get(analysis_id)
    if cached is None:
        return jsonify({"error": "unknown analysis id"}), 404
    return jsonify(cached["result"].to_dict())


@app.route("/api/shots", methods=["GET"])
def api_shots():
    return jsonify({"shots": all_shot_options(), "players": list(PRO_PLAYERS)})


# ---------- NEW: real-time camera analysis ----------
@app.route("/live", methods=["GET"])
def live():
    source = request.args.get("source", "0")
    angle = request.args.get("angle", "A")
    return render_template("live.html", source=source, angle=angle)


@app.route("/live/stream", methods=["GET"])
def live_stream():
    analyzer = REALTIME_MANAGER.get_or_start(_camera_source(), request.args.get("angle", "A"))
    if not analyzer.running:
        return analyzer.error or "Camera unavailable.", 503
    return Response(
        analyzer.mjpeg_stream(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/live/metrics", methods=["GET"])
def live_metrics():
    analyzer = REALTIME_MANAGER.get_or_start(_camera_source(), request.args.get("angle", "A"))
    analyzer.touch()
    return jsonify(analyzer.latest_metrics())


@app.route("/live/stop", methods=["POST"])
def live_stop():
    REALTIME_MANAGER.stop(_camera_source())
    return jsonify({"status": "stopped"})


def _camera_source():
    """Camera index (``0``, ``1``, ...) or a stream URL / device path."""
    source = request.args.get("source", "0")
    return int(source) if str(source).isdigit() else source


if __name__ == "__main__":
    app.run(debug=True)
