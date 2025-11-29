import os
import uuid
import json
import base64

import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

# ---------------- Flask setup ----------------
app = Flask(__name__)

UPLOAD_FOLDER = os.path.join("static", "uploads")
OUTPUT_FOLDER = os.path.join("static", "outputs")
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -------------- Load shots database ----------
with open("shots.json", "r") as f:
    SHOT_DB = json.load(f)

# -------------- Mediapipe setup --------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


# -------------- Helpers ----------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_point_and_vis(landmark, w, h):
    return (int(landmark.x * w), int(landmark.y * h)), float(landmark.visibility)


def classify_shot_key(lm, w, h, forward_pct, angle: str) -> str:
    """
    Use pose + camera angle to pick a reasonable shot key from SHOT_DB.
    This is a simple rule-based classifier (not perfect, but decent).
    """

    lw = lm[mp_pose.PoseLandmark.LEFT_WRIST]
    rw = lm[mp_pose.PoseLandmark.RIGHT_WRIST]
    ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
    rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    lh = lm[mp_pose.PoseLandmark.LEFT_HIP]
    rh = lm[mp_pose.PoseLandmark.RIGHT_HIP]

    lw_pt = np.array([lw.x * w, lw.y * h])
    rw_pt = np.array([rw.x * w, rw.y * h])
    ls_pt = np.array([ls.x * w, ls.y * h])
    rs_pt = np.array([rs.x * w, rs.y * h])
    lh_pt = np.array([lh.x * w, lh.y * h])
    rh_pt = np.array([rh.x * w, rh.y * h])

    bat_vec = rw_pt - lw_pt
    bat_angle_deg = abs(np.degrees(np.arctan2(bat_vec[1], bat_vec[0])))

    shoulder_rotation = abs(ls_pt[0] - rs_pt[0]) / max(1, abs(lh_pt[0] - rh_pt[0]))

    # Default
    shot_key = "Straight Drive"

    if angle == "A":  # Side view
        if forward_pct >= 60:
            if bat_angle_deg < 30:
                shot_key = "Lofted Drive"
            else:
                shot_key = "Straight Drive"
        else:
            if bat_angle_deg < 35:
                shot_key = "Pull Shot"
            else:
                shot_key = "Square Cut"

    elif angle == "B":  # Front view (bowler end)
        if forward_pct >= 60:
            shot_key = "Cover Drive"
        else:
            if shoulder_rotation > 1.15:
                shot_key = "Pull Shot"
            else:
                shot_key = "Square Cut"

    elif angle == "C":  # Diagonal / 45°
        if forward_pct >= 60:
            if bat_angle_deg < 28:
                shot_key = "Sweep"
            elif bat_angle_deg < 55:
                shot_key = "Straight Drive"
            else:
                shot_key = "Lofted Drive"
        else:
            if bat_angle_deg < 32:
                shot_key = "Pull Shot"
            else:
                shot_key = "Late Cut"

    if shot_key not in SHOT_DB:
        shot_key = "Straight Drive"

    return shot_key


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


# -------------- Core analysis ----------------
def analyze_impact_photo(frame, angle: str):
    """
    Full analysis for a single impact photo:
      - pose
      - weight transfer
      - rating + encouragement
      - mistakes
      - shot classification + DB info
    """
    if frame is None:
        return (
            None,
            ["Image could not be read."],
            0,
            100,
            "No Data",
            "Image not readable.",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            [],
            [],
            [],
        )

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)
    if not result.pose_landmarks:
        return (
            frame,
            ["Pose not detected. Use a clearer photo with full body visible."],
            0,
            100,
            "No Data",
            "Pose not detected.",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            [],
            [],
            [],
        )

    h, w, _ = frame.shape
    lm = result.pose_landmarks.landmark

    # Draw skeleton
    mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Keypoints
    left_ankle_pt, left_ankle_vis = get_point_and_vis(
        lm[mp_pose.PoseLandmark.LEFT_ANKLE], w, h
    )
    right_ankle_pt, right_ankle_vis = get_point_and_vis(
        lm[mp_pose.PoseLandmark.RIGHT_ANKLE], w, h
    )
    left_hip_pt, _ = get_point_and_vis(lm[mp_pose.PoseLandmark.LEFT_HIP], w, h)
    right_hip_pt, _ = get_point_and_vis(lm[mp_pose.PoseLandmark.RIGHT_HIP], w, h)
    nose_pt, _ = get_point_and_vis(lm[mp_pose.PoseLandmark.NOSE], w, h)

    # Decide front/back leg with head position
    nose_x = nose_pt[0]
    if abs(nose_x - left_ankle_pt[0]) < abs(nose_x - right_ankle_pt[0]):
        front_ankle_pt = left_ankle_pt
        back_ankle_pt = right_ankle_pt
        back_vis = right_ankle_vis
    else:
        front_ankle_pt = right_ankle_pt
        back_ankle_pt = left_ankle_pt
        back_vis = left_ankle_vis

    # Draw feet
    cv2.circle(frame, back_ankle_pt, 6, (0, 0, 255), -1)   # back red
    cv2.circle(frame, front_ankle_pt, 6, (0, 255, 0), -1)  # front green

    # Weight transfer
    front_x = front_ankle_pt[0]
    back_x = back_ankle_pt[0]
    hip_center_x = int((left_hip_pt[0] + right_hip_pt[0]) / 2)

    span = max(1, abs(front_x - back_x))
    t = (hip_center_x - back_x) / span
    if front_x < back_x:
        t = 1 - t
    t = max(0.0, min(1.0, t))

    forward_pct = int(round(t * 100))
    back_pct = 100 - forward_pct

    # Feedback / mistakes
    mistakes = []
    if forward_pct < 55:
        mistakes.append("Push more weight onto your front foot at impact.")
    if nose_x > front_x + 20:
        mistakes.append("Try to keep your head over or just in front of your front knee.")
    if back_vis < 0.3:
        mistakes.append("Back foot is not clearly visible. Try a photo where both feet are in the frame.")

    # Rating + encouragement
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

    # Shot classification -> DB
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

    return (
        frame,
        mistakes,
        forward_pct,
        back_pct,
        rating_label,
        encouragement,
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


# -------------- Routes -----------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    angle = request.form.get("angle", "A")

    file = request.files.get("image")
    camera_data = request.form.get("image_from_camera")

    image_id = uuid.uuid4().hex
    frame = None

    # 1) camera photo (base64 from hidden input)
    if camera_data:
        try:
            header, encoded = camera_data.split(",", 1)
            img_bytes = base64.b64decode(encoded)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            print("Camera decode error:", e)
            frame = None

    # 2) uploaded file if camera not used
    elif file and file.filename != "" and allowed_file(file.filename):
        safe_name = secure_filename(f"{image_id}_{file.filename}")
        image_path = os.path.join(UPLOAD_FOLDER, safe_name)
        file.save(image_path)
        frame = cv2.imread(image_path)

    if frame is None:
        return "No valid image received. Please upload or capture again."

    (
        analyzed_frame,
        mistakes,
        forward_pct,
        back_pct,
        rating_label,
        encouragement,
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
    ) = analyze_impact_photo(frame, angle)

    # Save output image
    output_name = f"{image_id}.jpg"
    output_path = os.path.join(OUTPUT_FOLDER, output_name)
    cv2.imwrite(output_path, analyzed_frame)

    return render_template(
        "result.html",
        image_url=url_for("static", filename=f"outputs/{output_name}"),
        mistakes=mistakes,
        is_good=(rating_label in ["Excellent Shot", "Good Shot"]),
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
        # NEW: for dropdown
        all_shots=list(SHOT_DB.keys()),
        current_shot=shot_name,
    )


# ---------- NEW: shot correction route ----------
@app.route("/update_shot", methods=["POST"])
def update_shot():
    correct_shot = request.form.get("correct_shot")
    image_url = request.form.get("image_url")

    if not correct_shot or correct_shot not in SHOT_DB:
        correct_shot = "Straight Drive"

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
        mistakes=[],
        is_good=False,
        forward_pct=0,
        back_pct=0,
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
        all_shots=list(SHOT_DB.keys()),
        current_shot=shot_name,
    )


if __name__ == "__main__":
    app.run(debug=True)
