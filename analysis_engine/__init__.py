"""Advanced cricket shot analysis engine.

A modular replacement for the original single-frame, rule-based analysis.  The
stages are independent and individually reusable:

==============================  ==========================================================
module                          responsibility
==============================  ==========================================================
``config``                      all tunable constants (env-overridable)
``datatypes``                   value objects passed between stages
``pose``                        3D pose estimation (BlazePose GHUM / Pose Landmarker)
``tracking.bat_tracker``        bat grip/toe tracking, swing path metrics
``tracking.ball_tracker``       ball detection, trajectory, bounce, prediction
``features``                    per-frame biomechanical feature extraction
``impact``                      multi-cue impact-frame detection
``weight_transfer``             centre-of-mass weight transfer + movement graph
``classifier``                  temporal, explainable 20-shot classifier
``metrics``                     kinematics helpers + full technical metric set
``scoring``                     ten 0-100 technique scores
``coaching``                    strengths / weaknesses / cues / drills
``comparison``                  benchmark comparison against professionals
``output.annotate``             skeleton, trajectories, HUD overlays
``output.report``               PDF report
``realtime``                    threaded live-camera analysis (20-30 FPS target)
``pipeline``                    facade wiring everything together
==============================  ==========================================================

Typical use::

    from analysis_engine import ShotAnalysisPipeline

    pipeline = ShotAnalysisPipeline(static_image_mode=True)
    result = pipeline.analyze_image(frame, angle="A")
    print(result.shot.name, result.shot.confidence, result.impact.frame_index)
"""

from .comparison import PRO_PLAYERS, compare_with_professional
from .config import CONFIG
from .datatypes import AnalysisResult, ImpactInfo, ShotPrediction, WeightTransferResult
from .classifier import SUPPORTED_SHOTS
from .output import build_pdf_report
from .pipeline import ShotAnalysisPipeline
from .realtime import REALTIME_MANAGER, RealtimeAnalyzer
from .scoring import score_label

__all__ = [
    "AnalysisResult",
    "CONFIG",
    "ImpactInfo",
    "PRO_PLAYERS",
    "REALTIME_MANAGER",
    "RealtimeAnalyzer",
    "SUPPORTED_SHOTS",
    "ShotAnalysisPipeline",
    "ShotPrediction",
    "WeightTransferResult",
    "build_pdf_report",
    "compare_with_professional",
    "score_label",
]

__version__ = "2.0.0"
