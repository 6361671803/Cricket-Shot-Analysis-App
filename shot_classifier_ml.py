"""Trainable local shot classifier (Phase 3, tier 1 of the classification
chain: ML -> LLM -> rule-based).

No model ships with this - `models/shot_classifier.joblib` doesn't exist
until you run `train_shot_classifier.py` against labeled clips in `data/`
(see data/README.md and label_clips.py). Until then, classify_shot_ml()
always returns None and callers fall through to the next tier.
"""

import math
import os
from typing import List, Optional, Tuple

import numpy as np

import biomechanics
import pose_utils

MODEL_PATH = os.path.join("models", "shot_classifier.joblib")

N_KEYFRAMES = 15
FEATURES_PER_FRAME = 5  # forward_pct, knee flex, elbow extension, shoulder/hip rotation, wrist speed
FEATURE_VECTOR_LENGTH = N_KEYFRAMES * FEATURES_PER_FRAME

DEFAULT_CONFIDENCE_THRESHOLD = 0.5


def _resample_indices(n: int, k: int) -> List[int]:
    """k evenly-spaced indices into a sequence of length n (pads with the
    last index if n < k, so short clips still produce a fixed-length output)."""
    if n <= 0:
        return [0] * k
    if n <= k:
        return list(range(n)) + [n - 1] * (k - n)
    return [round(i * (n - 1) / (k - 1)) for i in range(k)]


def extract_feature_vector(sequence: list, w: int, h: int) -> np.ndarray:
    """
    Pose-landmark sequence (video_analysis.extract_landmark_sequence's
    output format) -> a fixed-length feature vector for the classifier.
    Resamples to N_KEYFRAMES points across the whole clip; per keyframe,
    reuses the same math as compute_weight_transfer/biomechanics.py so
    features here are computed identically to what the rest of the app
    already shows the user - not a separate/duplicated model of the pose.
    """
    n = len(sequence)
    if n == 0:
        return np.zeros(FEATURE_VECTOR_LENGTH, dtype=np.float32)

    wrist_points = [pose_utils.wrist_midpoint(item["landmarks"], h) for item in sequence]
    raw_speed = [0.0]
    for i in range(1, n):
        (x0, y0), (x1, y1) = wrist_points[i - 1], wrist_points[i]
        raw_speed.append(math.hypot(x1 - x0, y1 - y0) / max(1, h))

    indices = _resample_indices(n, N_KEYFRAMES)

    features = []
    for idx in indices:
        lm = sequence[idx]["landmarks"]
        wt = pose_utils.compute_weight_transfer(lm, w, h)
        biomech = biomechanics.compute_biomechanics(lm, w, h, wt["front_side"])
        features.extend([
            wt["forward_pct"] / 100.0,
            biomech["front_knee_flex_deg"] / 180.0,
            biomech["elbow_extension_deg"] / 180.0,
            biomech["shoulder_hip_rotation_deg"] / 90.0,
            raw_speed[idx],
        ])
    return np.array(features, dtype=np.float32)


class ShotClassifier:
    """Interface every classifier implementation follows - swappable for a
    future PyTorch/LSTM model without touching call sites."""

    def predict(self, feature_vector: np.ndarray) -> Tuple[str, float]:
        raise NotImplementedError


class RandomForestShotClassifier(ShotClassifier):
    def __init__(self, model, label_encoder):
        self.model = model
        self.label_encoder = label_encoder

    def predict(self, feature_vector: np.ndarray) -> Tuple[str, float]:
        probs = self.model.predict_proba(feature_vector.reshape(1, -1))[0]
        best_idx = int(np.argmax(probs))
        shot_key = self.label_encoder.inverse_transform([best_idx])[0]
        return shot_key, float(probs[best_idx])

    @classmethod
    def load(cls, path: str) -> "RandomForestShotClassifier":
        import joblib
        payload = joblib.load(path)
        return cls(payload["model"], payload["label_encoder"])


_classifier_cache: Optional[RandomForestShotClassifier] = None
_load_attempted = False


def get_ml_classifier() -> Optional[RandomForestShotClassifier]:
    """Lazily load and cache the trained model. Returns None (not an
    exception) if no model file exists yet, or it fails to load."""
    global _classifier_cache, _load_attempted
    if _load_attempted:
        return _classifier_cache
    _load_attempted = True

    if not os.path.exists(MODEL_PATH):
        return None
    try:
        _classifier_cache = RandomForestShotClassifier.load(MODEL_PATH)
    except Exception:
        _classifier_cache = None
    return _classifier_cache


def classify_shot_ml(sequence: list, w: int, h: int,
                      confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD) -> Optional[Tuple[str, float]]:
    """
    Returns (shot_key, confidence) if a trained model exists and predicts
    with at least confidence_threshold, else None so the caller falls
    through to the next tier (LLM, then rule-based).
    """
    classifier = get_ml_classifier()
    if classifier is None or not sequence:
        return None

    feature_vector = extract_feature_vector(sequence, w, h)
    shot_key, confidence = classifier.predict(feature_vector)
    if confidence < confidence_threshold:
        return None
    return shot_key, confidence
