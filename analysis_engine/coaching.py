"""Rule-based AI coaching feedback.

Each rule is a measurable trigger paired with the cue a batting coach would
actually say ("your head is falling outside off stump", "keep the front elbow
higher").  Rules are evaluated against the measured metrics, ranked by severity
and grouped into strengths, weaknesses and improvement drills, so feedback is
specific to the shot that was played rather than generic advice.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .datatypes import FrameFeatures, ImpactInfo, ShotPrediction, WeightTransferResult
from .scoring import score_label


def _num(metrics: Dict[str, object], key: str) -> Optional[float]:
    """Metric as a float, or ``None`` when the engine could not measure it.

    Rules must stay silent about anything that was not measured (a single photo
    cannot show a follow-through) instead of coaching off a zero.
    """
    value = metrics.get(key)
    if value is None:
        return None
    return float(value)


FRONT_FOOT_SHOTS = {
    "Forward Defence",
    "Straight Drive",
    "Cover Drive",
    "On Drive",
    "Lofted Drive",
    "Sweep",
    "Reverse Sweep",
    "Paddle Sweep",
    "Flick",
}


def generate_coaching(
    shot: ShotPrediction,
    impact: ImpactInfo,
    weight: WeightTransferResult,
    metrics: Dict[str, object],
    scores: Dict[str, int],
    features: Optional[Sequence[FrameFeatures]] = None,
) -> Dict[str, object]:
    """Produce strengths, weaknesses, coaching cues, drills and a summary line."""
    strengths: List[str] = []
    weaknesses: List[str] = []
    cues: List[str] = []
    drills: List[str] = []

    if not metrics.get("available"):
        return {
            "strengths": [],
            "weaknesses": ["Pose could not be detected, so no technical feedback is available."],
            "cues": ["Record from side-on with the full body inside the frame."],
            "drills": [],
            "summary": "Analysis incomplete: the batter was not detected clearly.",
        }

    front_foot_shot = shot.name in FRONT_FOOT_SHOTS
    forward = weight.forward_pct

    # ------------------------------------------------------------- head / eyes
    head_stability = _num(metrics, "head_stability")
    if head_stability is not None:
        if head_stability >= 78:
            strengths.append(f"Head is very still through the shot ({head_stability:.0f}/100).")
        elif head_stability < 55:
            weaknesses.append(f"Head moves a lot during the swing ({head_stability:.0f}/100).")
            cues.append("Keep your head still and let your hands do the work.")
            drills.append("Shadow-bat in front of a mirror with a coin balanced on the bat toe.")
    eye_level = _num(metrics, "eye_level_stability")
    if eye_level is not None and eye_level < 60:
        cues.append("Keep your eyes level — your head is dropping as you swing.")
    if float(metrics["spine_lean_deg"]) > 22:
        cues.append("Your head is falling outside the line of off stump; lead with your shoulder, not your head.")

    # ------------------------------------------------------- weight / footwork
    if front_foot_shot and forward < 55:
        weaknesses.append(f"Only {forward}% of your weight reaches the front foot on a front-foot shot.")
        cues.append("Transfer more weight onto the front foot and drive through the front leg.")
        drills.append("Front-foot drive drill off a tee, finishing with the back heel off the ground.")
    elif not front_foot_shot and forward > 62:
        weaknesses.append("You commit forward on a back-foot shot, which cramps you for room.")
        cues.append("Get back and across earlier so you have time on the back foot.")
    elif 55 <= forward <= 82 and front_foot_shot:
        strengths.append(f"Good forward weight transfer ({forward}% on the front foot).")

    if weight.timing_label == "Late Transfer":
        weaknesses.append("Weight transfer arrives after contact.")
        cues.append("Earlier downswing required — start the stride as the bowler releases.")
    elif weight.timing_label == "Early Transfer":
        cues.append("You commit a fraction early; hold your shape until the ball is released.")

    knee = float(metrics["front_knee_flexion_deg"])
    if knee < 10:
        cues.append("Bend the front knee — a locked leg kills balance and weight transfer.")
    elif 15 <= knee <= 55:
        strengths.append("Front knee is nicely flexed over the ball.")

    if float(metrics["stride_length_m"]) < 0.3 and front_foot_shot:
        weaknesses.append("Very short stride into the ball.")
        drills.append("Stride-length markers: place a cone at your ideal front-foot landing point.")

    pivot = _num(metrics, "back_foot_pivot_deg")
    if pivot is not None and pivot < 5 and front_foot_shot:
        cues.append("Let the back foot pivot so your hips can rotate through the shot.")

    # ---------------------------------------------------------------- bat work
    elbow = float(metrics["front_elbow_deg"])
    if elbow < 95:
        cues.append("Keep your front elbow higher through contact.")
        weaknesses.append("Front elbow collapses at impact, closing the bat face.")
    elif elbow >= 130:
        strengths.append("High front elbow — classical shape through the line.")

    face = abs(float(metrics["bat_face_angle_deg"]))
    if face > 28:
        cues.append("Keep your bat face straighter at contact.")
    backlift = _num(metrics, "backlift_height_ratio")
    if backlift is not None and backlift < 0.05:
        cues.append("Lift the bat higher in the backlift to generate free power.")

    follow = str(metrics["follow_through"])
    if follow == "Checked" and shot.name not in {"Forward Defence", "Back Foot Defence", "Glance", "Paddle Sweep"}:
        weaknesses.append("Follow-through is checked, which costs timing and power.")
        cues.append("Complete a full, relaxed follow-through and balance after impact.")
    elif follow == "Full":
        strengths.append("Full, balanced follow-through.")

    # ---------------------------------------------------------------- contact
    if impact.contact_quality_score >= 75:
        strengths.append(f"Ball struck out of the middle ({impact.contact_quality}).")
    elif impact.contact_quality_score > 0 and impact.contact_quality_score < 50:
        weaknesses.append(f"Contact was not clean ({impact.contact_quality}).")
        cues.append("Meet the ball under your eyes, not beside your body.")
    if "Late" in str(metrics["impact_position"]):
        cues.append("You are playing the ball too late — meet it in front of the pad.")

    if float(metrics["balance_score"]) < 55:
        weaknesses.append("Balance is lost through and after impact.")
        cues.append("Improve balance after impact — hold your finishing position for two seconds.")
        drills.append("Freeze-finish drill: hold the follow-through until the ball reaches the fielder.")

    bat_speed = _num(metrics, "bat_speed_kmh")
    if bat_speed and scores.get("power", 0) >= 75:
        strengths.append(f"Strong bat speed ({bat_speed:.0f} km/h peak).")
    elif bat_speed and scores.get("power", 0) < 45:
        weaknesses.append("Low bat speed through the hitting zone.")
        drills.append("Heavy-bat swings (10 x 8) to build swing speed without losing shape.")

    overall = scores.get("overall_technique", 0)
    summary = (
        f"{shot.name} played with {shot.confidence:.0f}% confidence — overall technique "
        f"{overall}/100 ({score_label(overall)}). {weight.label}, {weight.timing_label.lower()}, "
        f"contact: {impact.contact_quality.lower()}."
    )

    return {
        "strengths": _dedupe(strengths) or ["Solid base to build on — keep repeating the shot."],
        "weaknesses": _dedupe(weaknesses) or ["No major technical faults detected in this shot."],
        "cues": _dedupe(cues) or ["Keep repeating this shape; the fundamentals look sound."],
        "drills": _dedupe(drills) or ["Throw-down repetitions of this shot, 3 sets of 12."],
        "summary": summary,
    }


def _dedupe(items: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


__all__ = ["generate_coaching", "FRONT_FOOT_SHOTS"]
