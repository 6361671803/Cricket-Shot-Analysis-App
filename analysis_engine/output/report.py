"""PDF report generation for a completed analysis.

Uses ReportLab (pure Python, no system dependencies) to build a one-file report
containing the shot verdict, impact details, weight transfer, all technique
scores, the full technical-metric table, coaching feedback and — when available
— the annotated impact frame and the weight-transfer graph.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from ..datatypes import AnalysisResult
from ..scoring import score_label

PAGE_MARGIN = 36


def build_pdf_report(
    result: AnalysisResult,
    output_path: str,
    shot_info: Optional[Dict[str, object]] = None,
) -> Optional[str]:
    """Render ``result`` to a PDF at ``output_path``; returns the path or None."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            Image,
            PageBreak,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except Exception:  # pragma: no cover - optional dependency
        return None

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CricketTitle", parent=styles["Title"], textColor=colors.HexColor("#0b6b3a"), fontSize=20
    )
    heading = ParagraphStyle(
        "CricketHeading",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#0b6b3a"),
        spaceBefore=10,
        spaceAfter=4,
    )
    body = ParagraphStyle("CricketBody", parent=styles["BodyText"], fontSize=9.5, leading=13)

    story: List[object] = []
    story.append(Paragraph("Cricket Shot Analysis Report", title_style))
    story.append(Paragraph(str(result.coaching.get("summary", "")), body))
    story.append(Spacer(1, 6))

    def table(rows: List[List[str]], widths: Optional[List[float]] = None) -> Table:
        component = Table(rows, colWidths=widths or [70 * mm, 95 * mm], hAlign="LEFT")
        component.setStyle(
            TableStyle(
                [
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#333333")),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8f5e9")),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c8e6c9")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        return component

    # ---------------------------------------------------------------- summary
    impact = result.impact
    weight = result.weight_transfer
    story.append(Paragraph("Shot Summary", heading))
    story.append(
        table(
            [
                ["Item", "Value"],
                ["Shot", result.shot.name],
                ["Confidence", f"{result.shot.confidence:.1f}%"],
                ["Why", result.shot.reason],
                ["Impact frame", str(impact.frame_index)],
                ["Impact timestamp", f"{impact.timestamp:.2f} s"],
                ["Bat speed", f"{impact.bat_speed_kmh:.1f} km/h"],
                ["Ball speed", f"{impact.ball_speed_kmh:.1f} km/h"],
                ["Contact quality", f"{impact.contact_quality} ({impact.contact_quality_score:.0f}/100)"],
                ["Impact position", impact.impact_position],
                ["Weight transfer", f"{weight.forward_pct}% front / {weight.back_pct}% back"],
                ["Transfer type", f"{weight.label}, {weight.timing_label}"],
                ["Transfer quality", weight.quality_label],
                ["Frames analysed", f"{result.frames_analysed} @ {result.fps:.1f} fps"],
            ]
        )
    )

    # ------------------------------------------------------------------ media
    for path, caption in (
        (result.annotated_image_path, "Annotated impact frame"),
        (result.graph_path or weight.graph_path, "Weight-transfer movement graph"),
    ):
        if path and os.path.exists(path):
            image = _scaled_image(Image, path, 165 * mm, 120 * mm)
            if image is not None:
                story.append(Paragraph(caption, heading))
                story.append(image)

    # ----------------------------------------------------------------- scores
    if result.scores:
        story.append(Paragraph("Technique Scores (0-100)", heading))
        rows = [["Category", "Score", "Band"]]
        for key, value in result.scores.items():
            rows.append([key.replace("_", " ").title(), str(value), score_label(int(value))])
        story.append(table(rows, [70 * mm, 25 * mm, 40 * mm]))

    # ---------------------------------------------------------------- metrics
    display = result.metrics.get("display") if isinstance(result.metrics, dict) else None
    if display:
        story.append(PageBreak())
        story.append(Paragraph("Technical Metrics", heading))
        rows = [["Metric", "Value"]] + [[row["label"], row["value"]] for row in display]
        story.append(table(rows))

    # --------------------------------------------------------------- coaching
    for key, label in (
        ("strengths", "Strengths"),
        ("weaknesses", "Weaknesses"),
        ("cues", "Coaching Feedback"),
        ("drills", "Improvement Suggestions"),
    ):
        items = result.coaching.get(key) or []
        if not items:
            continue
        story.append(Paragraph(label, heading))
        for item in items:
            story.append(Paragraph(f"\u2022 {item}", body))

    # ------------------------------------------------------------- comparison
    comparison = result.comparison or {}
    if comparison.get("rows"):
        story.append(Paragraph(f"Comparison vs {comparison.get('player')}", heading))
        story.append(Paragraph(str(comparison.get("style", "")), body))
        rows = [["Metric", "You", "Pro", "Gap", "Verdict"]]
        for row in comparison["rows"]:
            rows.append([row["metric"], row["you"], row["pro"], row["delta"], row["verdict"]])
        story.append(table(rows, [45 * mm, 28 * mm, 28 * mm, 25 * mm, 40 * mm]))
        story.append(Paragraph(f"Overall match: {comparison.get('match_pct', 0)}%", body))

    # ------------------------------------------------------- shot library info
    if shot_info:
        story.append(Paragraph("Shot Library", heading))
        rows = [["Field", "Detail"]]
        for key in ("summary", "fields", "masters", "history", "improve"):
            if shot_info.get(key):
                rows.append([key.title(), str(shot_info[key])])
        story.append(table(rows))

    document = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=PAGE_MARGIN,
        rightMargin=PAGE_MARGIN,
        topMargin=PAGE_MARGIN,
        bottomMargin=PAGE_MARGIN,
        title="Cricket Shot Analysis Report",
    )
    document.build(story)
    return output_path


def _scaled_image(image_cls, path: str, max_width: float, max_height: float):
    """Embed an image scaled to fit the page while preserving its aspect ratio."""
    try:
        from reportlab.lib.utils import ImageReader

        width, height = ImageReader(path).getSize()
        if width <= 0 or height <= 0:
            return None
        scale = min(max_width / width, max_height / height)
        return image_cls(path, width=width * scale, height=height * scale)
    except Exception:  # pragma: no cover - unreadable image
        return None


__all__ = ["build_pdf_report"]
