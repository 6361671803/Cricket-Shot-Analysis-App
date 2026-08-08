import os
import uuid
import json
import base64
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, render_template, request, url_for, jsonify, redirect
from werkzeug.utils import secure_filename

import biomechanics
import calibration
import db
import pose_utils
import shot_classifier_llm
import shot_map
import video_analysis
from pose_utils import ShotAnalysisResult
from shots import SHOT_DB, classify_shot_key, get_shot_info_from_db

# ---------------- Flask setup ----------------
app = Flask(__name__)

UPLOAD_FOLDER = os.path.join("static", "uploads")
OUTPUT_FOLDER = os.path.join("static", "outputs")
CORRECTIONS_FILE = os.path.join("dataset", "shot_corrections.jsonl")
ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "mov"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER
# Raised from 10MB so short slow-mo video clips aren't rejected.
app.config["MAX_CONTENT_LENGTH"] = 80 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs("dataset", exist_ok=True)

# -------------- Persistence (Phase 5) --------------
db.init_db(os.path.join(app.instance_path, "sessions.db"))

# -------------- Mediapipe setup (photo path) --------------
model_path = "pose_landmarker.task"
pose_utils.ensure_pose_model(model_path)

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE)

pose = PoseLandmarker.create_from_options(options)


# -------------- Helpers ----------------------
def file_extension(filename: str) -> str:
    return filename.rsplit(".", 1)[1].lower() if "." in filename else ""


def _index_context(error: Optional[str] = None):
    """Shared template context for '/' - keeps index() and index_with_error() in sync."""
    return {
        "error": error,
        "all_shots": SHOT_DB,
        "max_upload_mb": app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024),
        "allowed_image_extensions": sorted(ALLOWED_IMAGE_EXTENSIONS),
        "allowed_video_extensions": sorted(ALLOWED_VIDEO_EXTENSIONS),
    }


def index_with_error(message: str, status: int = 400):
    return render_template("index.html", **_index_context(error=message)), status


def clean_player_name(value: str) -> str:
    """Keep optional, user-provided player labels safe and consistent."""
    return " ".join((value or "").strip().split())[:80]


def parse_calibration(form) -> Optional[float]:
    """
    Build a cm-per-pixel scale from the two optional calibration fields
    (a measured pixel length + its known real-world length in cm).
    Returns None if either field is missing/invalid - video analysis
    then stays in pixel-only units.
    """
    pixel_length_raw = (form.get("calib_pixel_length") or "").strip()
    real_cm_raw = (form.get("calib_real_cm") or "").strip()
    if not pixel_length_raw or not real_cm_raw:
        return None
    try:
        return calibration.calibrate_scale(float(pixel_length_raw), float(real_cm_raw))
    except ValueError:
        return None


def save_correction(image_url: str, shot_name: str, player_name: str, angle: str):
    """Append a verified label for future model training; never guess identity."""
    record = {
        "image_url": image_url,
        "shot_name": shot_name,
        "player_name": player_name or None,
        "camera_angle": angle if angle in {"A", "B", "C"} else None,
    }
    with open(CORRECTIONS_FILE, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


# -------------- Core analysis ----------------
def analyze_impact_photo(frame, angle: str) -> ShotAnalysisResult:
    """
    Full analysis for a single impact photo:
      - pose
      - weight transfer
      - rating + encouragement
      - mistakes
      - shot classification + DB info
    """
    if frame is None:
        return ShotAnalysisResult(
            mistakes=["Image could not be read."],
            rating_label="No Data",
            encouragement="Image not readable.",
        )

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = pose.detect(mp_image)
    if not result.pose_landmarks:
        return ShotAnalysisResult(
            frame=frame,
            mistakes=["Pose not detected. Use a clearer photo with full body visible."],
            rating_label="No Data",
            encouragement="Pose not detected.",
        )

    h, w, _ = frame.shape
    lm = result.pose_landmarks[0]
    original_frame = frame.copy()  # clean copy for the LLM tier, before the overlay is drawn in place

    wt = pose_utils.draw_pose_overlay(frame, lm, w, h)
    forward_pct = wt["forward_pct"]
    back_pct = wt["back_pct"]

    mistakes = pose_utils.detect_mistakes(
        forward_pct, wt["nose_pt"][0], wt["front_ankle_pt"][0], wt["back_vis"], wt["stance_span_reliable"]
    )
    rating_label, encouragement = pose_utils.rate_shot(forward_pct, mistakes)

    # Shot classification -> DB. No pose sequence on the photo path, so the
    # ML tier (video-only) is skipped: LLM fallback -> rule-based.
    classification_method = "rule-based"
    classification_confidence = None
    classification_reasoning = None

    llm_result = shot_classifier_llm.classify_shot_llm(original_frame, SHOT_DB)
    if llm_result is not None:
        shot_key, classification_confidence, classification_reasoning = llm_result
        classification_method = "llm"
    else:
        shot_key = classify_shot_key(lm, w, h, forward_pct, angle)

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

    biomech_metrics = biomechanics.compute_biomechanics(lm, w, h, wt["front_side"])
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
        biomechanics=biomechanics_result,
        classification_method=classification_method,
        classification_confidence=classification_confidence,
        classification_reasoning=classification_reasoning,
    )


def _load_upload_as_frame_or_video(file, camera_data, image_id):
    """
    Resolve an incoming request's media into either:
      - ("photo", frame) for an image (upload or camera capture), or
      - ("video", path) for a video upload
    Returns (kind, payload) or (None, error_message) if nothing usable was given.
    """
    # 1) camera photo (base64 from hidden input) - photo only
    if camera_data:
        try:
            _, encoded = camera_data.split(",", 1)
            img_bytes = base64.b64decode(encoded, validate=True)
            if len(img_bytes) > app.config["MAX_CONTENT_LENGTH"]:
                return None, "The captured photo is too large. Please try again."
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                return None, "We couldn't read that image. Please use a JPG, PNG, or WEBP impact photo."
            return "photo", frame
        except (ValueError, base64.binascii.Error):
            return None, "We couldn't read that image. Please use a JPG, PNG, or WEBP impact photo."

    # 2) uploaded file (image or video)
    if file and file.filename != "":
        ext = file_extension(file.filename)
        if ext in ALLOWED_VIDEO_EXTENSIONS:
            safe_name = secure_filename(f"{image_id}_{file.filename}")
            video_path = os.path.join(UPLOAD_FOLDER, safe_name)
            file.save(video_path)
            return "video", video_path
        if ext in ALLOWED_IMAGE_EXTENSIONS:
            safe_name = secure_filename(f"{image_id}_{file.filename}")
            image_path = os.path.join(UPLOAD_FOLDER, safe_name)
            file.save(image_path)
            frame = cv2.imread(image_path)
            if frame is None:
                return None, "We couldn't read that image. Please use a JPG, PNG, or WEBP impact photo."
            return "photo", frame

    return None, "We couldn't read that upload. Please use a JPG/PNG/WEBP photo or an MP4/MOV video clip."


def run_analysis(file, camera_data, angle: str, cm_per_px: Optional[float] = None):
    """
    Shared entry point for /analyze and /api/analyze: resolves the upload
    (photo or video), runs the matching analysis, and returns
    (ShotAnalysisResult, image_id, video_static_path) - video_static_path
    is the uploaded video's path relative to static/ (for building a
    playable <video> URL), or None for a photo. On failure, returns
    (None, error_message, None).
    """
    image_id = uuid.uuid4().hex
    kind, payload = _load_upload_as_frame_or_video(file, camera_data, image_id)

    if kind is None:
        return None, payload, None

    video_static_path = None
    if kind == "video":
        video_static_path = os.path.relpath(payload, "static").replace(os.sep, "/")
        result = video_analysis.analyze_video(payload, angle, cm_per_px=cm_per_px)
    else:
        result = analyze_impact_photo(payload, angle)

    if result.frame is None:
        return None, (result.mistakes[0] if result.mistakes else "Analysis failed."), None

    return result, image_id, video_static_path


def save_output_image(result: ShotAnalysisResult, image_id: str):
    output_name = f"{image_id}.jpg"
    output_path = os.path.join(OUTPUT_FOLDER, output_name)
    if not cv2.imwrite(output_path, result.frame):
        return None
    return output_name


def save_session_for_result(result: ShotAnalysisResult, player_name: str, angle: str,
                             image_url: str, video_url: Optional[str]) -> int:
    """Persist a completed analysis as a session row (Phase 5). Returns the new row id."""
    biomechanics_score = result.biomechanics.get("overall_score") if result.biomechanics else None
    return db.save_session(
        player_name=player_name,
        shot_key=result.shot_name,
        angle=angle,
        is_video=result.is_video,
        image_url=image_url,
        video_url=video_url,
        forward_pct=result.forward_pct,
        back_pct=result.back_pct,
        biomechanics_score=biomechanics_score,
        metrics=result.biomechanics,
    )


# -------------- Routes -----------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", **_index_context())


@app.route("/analyze", methods=["POST"])
def analyze():
    angle = request.form.get("angle", "A")
    if angle not in {"A", "B", "C"}:
        return index_with_error("Choose the camera angle before analyzing the photo.")

    file = request.files.get("image")
    camera_data = request.form.get("image_from_camera")
    player_name = clean_player_name(request.form.get("player_name"))
    cm_per_px = parse_calibration(request.form)

    result, image_id_or_error, video_static_path = run_analysis(file, camera_data, angle, cm_per_px=cm_per_px)
    if result is None:
        return index_with_error(image_id_or_error)

    output_name = save_output_image(result, image_id_or_error)
    if output_name is None:
        return index_with_error("We couldn't save the analysis image. Please try again.", 500)

    image_url = url_for("static", filename=f"outputs/{output_name}")
    video_url = url_for("static", filename=video_static_path) if video_static_path else None
    session_id = save_session_for_result(result, player_name, angle, image_url, video_url)

    return render_template(
        "result.html",
        image_url=image_url,
        video_url=video_url,
        session_id=session_id,
        biomechanics=result.biomechanics,
        classification_method=result.classification_method,
        classification_confidence=result.classification_confidence,
        classification_reasoning=result.classification_reasoning,
        mistakes=result.mistakes,
        is_good=(result.rating_label in ["Excellent Shot", "Good Shot"]),
        forward_pct=result.forward_pct,
        back_pct=result.back_pct,
        rating_label=result.rating_label,
        encouragement=result.encouragement,
        shot_name=result.shot_name,
        shot_summary=result.shot_summary,
        field_suggestion=result.field_suggestion,
        variations=result.variations,
        master_player=result.master_player,
        shot_history=result.shot_history,
        improvement_summary=result.improvement_summary,
        alt_safe=result.alt_safe,
        alt_aggressive=result.alt_aggressive,
        final_feedback=result.final_feedback,
        is_video=result.is_video,
        weight_transfer_series=result.weight_transfer_series,
        phase_markers=result.phase_markers,
        bat_metrics=result.bat_metrics,
        ball_estimate=result.ball_estimate,
        is_calibrated=result.is_calibrated,
        # NEW: for dropdown
        all_shots=list(SHOT_DB.keys()),
        current_shot=result.shot_name,
        player_name=player_name,
        angle=angle,
    )


# ---- NEW: API endpoint for JSON response ----
@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    angle = request.form.get("angle", "A")
    if angle not in {"A", "B", "C"}:
        return jsonify({"error": "Choose the camera angle before analyzing the photo."}), 400

    file = request.files.get("image")
    camera_data = request.form.get("image_from_camera")
    player_name = clean_player_name(request.form.get("player_name"))
    cm_per_px = parse_calibration(request.form)

    result, image_id_or_error, video_static_path = run_analysis(file, camera_data, angle, cm_per_px=cm_per_px)
    if result is None:
        return jsonify({"error": image_id_or_error}), 400

    output_name = save_output_image(result, image_id_or_error)
    if output_name is None:
        return jsonify({"error": "We couldn't save the analysis image. Please try again."}), 500

    output_url = url_for("static", filename=f"outputs/{output_name}", _external=True)
    video_url = url_for("static", filename=video_static_path, _external=True) if video_static_path else None
    session_id = save_session_for_result(result, player_name, angle, output_url, video_url)

    response_data = {
        "imageUrl": output_url,
        "videoUrl": video_url,
        "sessionId": session_id,
        "analysis": {
            "weightTransfer": {
                "forwardPercent": result.forward_pct,
                "backPercent": result.back_pct,
            },
            "rating": {
                "label": result.rating_label,
                "encouragement": result.encouragement,
            },
            "mistakes": result.mistakes,
            "shotInfo": {
                "name": result.shot_name,
                "summary": result.shot_summary,
                "fieldSuggestion": result.field_suggestion,
                "variations": result.variations,
                "masterPlayer": result.master_player,
                "history": result.shot_history,
                "improvementSummary": result.improvement_summary,
                "alternatives": {
                    "safe": result.alt_safe,
                    "aggressive": result.alt_aggressive,
                },
                "finalFeedback": result.final_feedback,
            },
            "isVideo": result.is_video,
            "weightTransferSeries": result.weight_transfer_series,
            "phaseMarkers": result.phase_markers,
            "batMetrics": result.bat_metrics,
            "ballEstimate": result.ball_estimate,
            "isCalibrated": result.is_calibrated,
            "biomechanics": result.biomechanics,
            "classificationMethod": result.classification_method,
            "classificationConfidence": result.classification_confidence,
            "classificationReasoning": result.classification_reasoning,
        },
        "player": {
            "name": player_name,
            "angle": angle,
        },
        "correction": {
            "allShots": list(SHOT_DB.keys()),
            "currentShot": result.shot_name,
        }
    }
    return jsonify(response_data)


# ---------- NEW: shot correction route ----------
@app.route("/update_shot", methods=["POST"])
def update_shot():
    correct_shot = request.form.get("correct_shot")
    image_url = request.form.get("image_url")
    video_url = request.form.get("video_url") or None
    player_name = clean_player_name(request.form.get("player_name"))
    angle = request.form.get("angle", "")
    try:
        forward_pct = max(0, min(100, int(request.form.get("forward_pct", 0))))
    except (TypeError, ValueError):
        forward_pct = 0
    back_pct = 100 - forward_pct

    if not correct_shot or correct_shot not in SHOT_DB:
        correct_shot = "Straight Drive"

    save_correction(image_url, correct_shot, player_name, angle)

    session_id = None
    try:
        session_id = int(request.form.get("session_id", ""))
        db.update_shot_key(session_id, correct_shot)
    except (TypeError, ValueError):
        session_id = None

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
    ) = get_shot_info_from_db(correct_shot)

    # We keep weight transfer & mistakes neutral in update mode
    return render_template(
        "result.html",
        image_url=image_url,
        video_url=video_url,
        session_id=session_id,
        biomechanics=None,
        classification_method="manual",
        classification_confidence=None,
        classification_reasoning=None,
        mistakes=[],
        is_good=False,
        forward_pct=forward_pct,
        back_pct=back_pct,
        rating_label="Shot Updated ✔",
        encouragement="Shot information updated based on your selection.",
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
        is_video=False,
        weight_transfer_series=None,
        phase_markers=None,
        bat_metrics=None,
        ball_estimate=None,
        is_calibrated=False,
        all_shots=list(SHOT_DB.keys()),
        current_shot=shot_name,
        player_name=player_name,
        angle=angle,
    )


# ---------- NEW: match-context tagging route (Phase 6) ----------
@app.route("/tag_shot", methods=["POST"])
def tag_shot():
    try:
        session_id = int(request.form.get("session_id", ""))
    except (TypeError, ValueError):
        return redirect(request.referrer or url_for("index"))

    over_number = None
    try:
        over_raw = (request.form.get("over_number") or "").strip()
        if over_raw:
            over_number = int(over_raw)
    except ValueError:
        over_number = None

    outcome_runs = None
    try:
        runs_raw = (request.form.get("outcome_runs") or "").strip()
        if runs_raw:
            outcome_runs = int(runs_raw)
    except ValueError:
        outcome_runs = None

    bowler_type = request.form.get("bowler_type") or None
    line_length = request.form.get("line_length") or None
    outcome_dismissal = request.form.get("outcome_dismissal") in {"on", "true", "1"}

    db.save_match_context(session_id, over_number, bowler_type, line_length, outcome_runs, outcome_dismissal)

    return redirect(request.referrer or url_for("index"))


# ---------- NEW: player history route (Phase 5 + 6) ----------
@app.route("/api/history", methods=["GET"])
def api_history():
    player_name = clean_player_name(request.args.get("player_name", ""))

    tagged_sessions = db.get_tagged_sessions(player_name, limit=50)

    return jsonify({
        "playerName": player_name,
        "recentSessions": db.get_recent_sessions(player_name, limit=20),
        "shotCounts": db.get_shot_counts(player_name),
        "trend": db.get_trend(player_name, limit=20),
        "strongestWeakest": db.get_strongest_weakest(player_name),
        "wagonWheel": shot_map.build_wagon_wheel_points(tagged_sessions),
    })


if __name__ == "__main__":
    app.run(debug=True)
