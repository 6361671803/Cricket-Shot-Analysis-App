"""Train the local ML shot classifier from labeled clips in data/<shot_name>/*.mp4.

Usage:
    python train_shot_classifier.py [--data-root data] [--min-clips-per-class 30]

Reads clips produced by label_clips.py (see data/README.md for the folder
convention and how many clips per class are realistic), extracts pose-
sequence feature vectors via shot_classifier_ml.extract_feature_vector,
trains a RandomForestClassifier, and saves it to
models/shot_classifier.joblib for classify_shot_ml() to pick up.
"""

import argparse
import glob
import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import shot_classifier_ml
import video_analysis
from shots import SHOT_DB

DEFAULT_DATA_ROOT = "data"
DEFAULT_MIN_CLIPS_PER_CLASS = 30


def collect_clips(data_root: str):
    """Returns {shot_name: [clip_paths]} for subfolders that match a known shot."""
    clips_by_shot = {}
    if not os.path.isdir(data_root):
        return clips_by_shot

    for entry in sorted(os.listdir(data_root)):
        shot_dir = os.path.join(data_root, entry)
        if not os.path.isdir(shot_dir):
            continue
        if entry not in SHOT_DB:
            print(f"Skipping '{entry}' - not a known shot in shots.json.")
            continue
        clip_paths = sorted(glob.glob(os.path.join(shot_dir, "*.mp4")))
        if clip_paths:
            clips_by_shot[entry] = clip_paths

    return clips_by_shot


def extract_dataset(clips_by_shot: dict):
    X, y = [], []
    for shot_name, clip_paths in clips_by_shot.items():
        for clip_path in clip_paths:
            sequence, _bat, _ball, _fps, w, h = video_analysis.extract_landmark_sequence(
                clip_path, bat_detector=None, ball_detector=None
            )
            if len(sequence) < video_analysis.MIN_VALID_FRAMES:
                print(f"  Skipping {clip_path} - only {len(sequence)} pose-detected frames.")
                continue
            X.append(shot_classifier_ml.extract_feature_vector(sequence, w, h))
            y.append(shot_name)
    return np.array(X), np.array(y)


def main():
    parser = argparse.ArgumentParser(description="Train the local ML shot classifier.")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--min-clips-per-class", type=int, default=DEFAULT_MIN_CLIPS_PER_CLASS,
                         help="Just a warning threshold - training still runs below this.")
    args = parser.parse_args()

    clips_by_shot = collect_clips(args.data_root)
    if not clips_by_shot:
        print(f"No labeled clips found under '{args.data_root}/<shot_name>/*.mp4'.")
        print("Use label_clips.py to create some first - see data/README.md.")
        return

    print("Clips per class:")
    for shot_name, clip_paths in clips_by_shot.items():
        flag = " (thin - see data/README.md)" if len(clip_paths) < args.min_clips_per_class else ""
        print(f"  {shot_name}: {len(clip_paths)}{flag}")

    if len(clips_by_shot) < 2:
        print("\nNeed at least 2 different shot classes with clips to train a classifier.")
        return

    print("\nExtracting pose features from clips (this runs MediaPipe on every clip)...")
    X, y = extract_dataset(clips_by_shot)
    if len(X) == 0:
        print("No usable clips after filtering - nothing to train on.")
        return

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    if len(X) >= 10 and len(set(y_encoded)) > 1:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
    else:
        X_train, y_train = X, y_encoded
        X_test, y_test = X, y_encoded
        print("\nDataset is small - evaluating on the training set itself (not a real holdout).")

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"\nTrained on {len(X_train)} clips, evaluated on {len(X_test)}. Accuracy: {accuracy:.1%}")

    os.makedirs(os.path.dirname(shot_classifier_ml.MODEL_PATH), exist_ok=True)
    joblib.dump({"model": model, "label_encoder": label_encoder}, shot_classifier_ml.MODEL_PATH)
    print(f"Saved model to {shot_classifier_ml.MODEL_PATH}")


if __name__ == "__main__":
    main()
