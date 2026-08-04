"""Rendering outputs: video/frame annotation and PDF reports."""

from .annotate import draw_analysis, draw_live_hud
from .report import build_pdf_report

__all__ = ["build_pdf_report", "draw_analysis", "draw_live_hud"]
