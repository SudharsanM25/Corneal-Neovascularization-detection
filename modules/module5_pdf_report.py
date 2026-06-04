"""
Module 5 — PDF Report Generation
==================================
Generates a clinical-style PDF report from all pipeline outputs.

Structure:
  Page 1  — Cover / Patient summary
  Page 2  — Module 1: Eye ROI (preprocessed | raw overlay | morph overlay | morph mask)
  Page 3  — Module 2: Cornea ROI (preprocessed | raw overlay | morph overlay | morph mask)
  Page 4  — Module 3: Vessel Filtering (structure overlay | heatmap | skeleton | binary | colored)
  Page 5  — Module 3: Debug strip (full 4-row diagnostic)
  Page 6  — Module 4: Spatial Tracking (zone map | vessel overlay | tracking overlay | severity heatmap)
  Page 7  — Quantitative Summary (metrics table + zone-density table)
  Page 8  — Zone density heatmap chart

Requires: reportlab
"""

import os
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm, cm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.colors import (
        HexColor, Color, white, black, lightgrey, grey
    )
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image as RLImage, PageBreak, HRFlowable,
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    REPORTLAB_OK = True
except ImportError:
    REPORTLAB_OK = False


# ─────────────────────────────────────────────────────────────
# COLOR PALETTE
# ─────────────────────────────────────────────────────────────
C_DARK    = HexColor("#0D1117")
C_NAVY    = HexColor("#0A1628")
C_TEAL    = HexColor("#00B4D8")
C_ACCENT  = HexColor("#90E0EF")
C_ORANGE  = HexColor("#FF6B35")
C_GREEN   = HexColor("#38B000")
C_RED     = HexColor("#D62839")
C_LIGHT   = HexColor("#CAF0F8")
C_GREY    = HexColor("#8D99AE")
C_WHITE   = white
C_PANEL   = HexColor("#EFF6FF")


# ─────────────────────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────────────────────
def _make_styles():
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "title", fontSize=26, fontName="Helvetica-Bold",
            textColor=C_TEAL, spaceAfter=4, alignment=TA_CENTER,
        ),
        "subtitle": ParagraphStyle(
            "subtitle", fontSize=13, fontName="Helvetica",
            textColor=C_GREY, spaceAfter=8, alignment=TA_CENTER,
        ),
        "section": ParagraphStyle(
            "section", fontSize=13, fontName="Helvetica-Bold",
            textColor=C_TEAL, spaceBefore=8, spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body", fontSize=9, fontName="Helvetica",
            textColor=C_DARK, spaceAfter=3,
        ),
        "caption": ParagraphStyle(
            "caption", fontSize=8, fontName="Helvetica-Oblique",
            textColor=C_GREY, alignment=TA_CENTER, spaceAfter=2,
        ),
        "metric_key": ParagraphStyle(
            "metric_key", fontSize=9, fontName="Helvetica-Bold",
            textColor=C_DARK,
        ),
        "metric_val": ParagraphStyle(
            "metric_val", fontSize=9, fontName="Helvetica",
            textColor=C_DARK, alignment=TA_RIGHT,
        ),
        "tag_high": ParagraphStyle(
            "tag_high", fontSize=9, fontName="Helvetica-Bold",
            textColor=C_RED, alignment=TA_RIGHT,
        ),
        "tag_low": ParagraphStyle(
            "tag_low", fontSize=9, fontName="Helvetica-Bold",
            textColor=C_GREEN, alignment=TA_RIGHT,
        ),
    }
    return styles


# ─────────────────────────────────────────────────────────────
# IMAGE HELPERS
# ─────────────────────────────────────────────────────────────
def _rl_image(path: str, width_mm: float, height_mm: float = None) -> RLImage:
    """Load image and return a ReportLab Image flowable."""
    if not os.path.exists(path):
        return Spacer(1, 5 * mm)
    w = width_mm * mm
    if height_mm:
        return RLImage(path, width=w, height=height_mm * mm)
    # auto height preserving aspect ratio
    img = cv2.imread(path)
    if img is None:
        return Spacer(1, 5 * mm)
    h_img, w_img = img.shape[:2]
    h = w * h_img / w_img
    return RLImage(path, width=w, height=h)


def _image_grid(paths_captions: list, col_widths_mm: list,
                col_height_mm: float = None) -> Table:
    """
    paths_captions : [(img_path, caption_str), ...]
    col_widths_mm  : list of widths matching len(paths_captions)
    """
    styles = _make_styles()
    imgs   = [_rl_image(p, w, col_height_mm)
              for (p, _), w in zip(paths_captions, col_widths_mm)]
    caps   = [Paragraph(c, styles["caption"])
              for _, c in paths_captions]
    t_data = [imgs, caps]
    t = Table(t_data, colWidths=[w * mm for w in col_widths_mm])
    t.setStyle(TableStyle([
        ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",   (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 2),
        ("LEFTPADDING",  (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
    ]))
    return t


# ─────────────────────────────────────────────────────────────
# SECTION HEADER
# ─────────────────────────────────────────────────────────────
def _section_header(title: str, subtitle: str = "") -> list:
    styles = _make_styles()
    items  = [HRFlowable(width="100%", thickness=1, color=C_TEAL, spaceAfter=4)]
    items.append(Paragraph(f"▶  {title}", styles["section"]))
    if subtitle:
        items.append(Paragraph(subtitle, styles["body"]))
    return items


# ─────────────────────────────────────────────────────────────
# METRICS TABLE
# ─────────────────────────────────────────────────────────────
def _metrics_table(rows: list, col_widths_mm: list) -> Table:
    """rows = [[cell, cell, ...], ...]"""
    t = Table(rows, colWidths=[w * mm for w in col_widths_mm])
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), C_NAVY),
        ("TEXTCOLOR",    (0, 0), (-1, 0), C_ACCENT),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_WHITE, C_PANEL]),
        ("GRID",         (0, 0), (-1, -1), 0.3, C_GREY),
        ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
    ]))
    return t


# ─────────────────────────────────────────────────────────────
# SEVERITY BADGE
# ─────────────────────────────────────────────────────────────
def _severity_grade(severity_pct: float) -> tuple:
    if severity_pct < 5:
        return "MINIMAL",  C_GREEN
    elif severity_pct < 15:
        return "MILD",     HexColor("#F4A261")
    elif severity_pct < 30:
        return "MODERATE", C_ORANGE
    else:
        return "SEVERE",   C_RED


# ─────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────
def run(out_pdf_path: str,
        session_dir: str,
        m3_results: dict,
        m4_results: dict,
        image_filename: str = "Unknown",
        patient_id: str = "N/A") -> str:
    """
    Generate the clinical PDF report.

    Parameters
    ----------
    out_pdf_path    : full path to write the PDF
    session_dir     : base session directory (contains module1..4 subdirs)
    m3_results      : dict returned by module3.run()
    m4_results      : dict returned by module4.run()
    image_filename  : original uploaded image name
    patient_id      : optional patient identifier

    Returns
    -------
    str : path to the written PDF
    """
    if not REPORTLAB_OK:
        raise ImportError("reportlab is required. Install with: pip install reportlab")

    os.makedirs(os.path.dirname(out_pdf_path) or ".", exist_ok=True)

    styles  = _make_styles()
    W_PAGE  = A4[0]; H_PAGE = A4[1]
    MARGIN  = 15 * mm
    CONTENT = (W_PAGE - 2 * MARGIN) / mm   # content width in mm

    doc = SimpleDocTemplate(
        out_pdf_path,
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
        title="Corneal Vascularization Report",
        author="CornealAI Pipeline",
    )

    # ── Convenience path resolver ──
    def p(module_subdir, filename):
        return os.path.join(session_dir, module_subdir, filename)

    m3_paths = m3_results.get("paths", {})
    m4_paths = m4_results.get("paths", {})
    summary  = m4_results.get("summary", {})
    vessel_a = m3_results.get("vessel_analysis", {})
    severity = m3_results.get("severity", 0.0)
    grade, grade_color = _severity_grade(severity)
    now      = datetime.now().strftime("%Y-%m-%d  %H:%M")

    story = []

    # ════════════════════════════════════════════════════════
    # PAGE 1 — COVER
    # ════════════════════════════════════════════════════════
    story.append(Spacer(1, 18 * mm))
    story.append(Paragraph("CORNEAL VASCULARIZATION", styles["title"]))
    story.append(Paragraph("Automated Analysis Report", styles["subtitle"]))
    story.append(Spacer(1, 4 * mm))
    story.append(HRFlowable(width="100%", thickness=2, color=C_TEAL))
    story.append(Spacer(1, 6 * mm))

    # Meta info box
    meta_data = [
        ["Parameter", "Value"],
        ["Image File",      image_filename],
        ["Patient ID",      patient_id],
        ["Analysis Date",   now],
        ["Pipeline",        "Eye ROI → Cornea ROI → Vessel Filter → Spatial Track"],
        ["Model Encoder",   "EfficientNet-B5 (UNet++)"],
        ["Vessel Detector", "Multi-scale Frangi + Meijering + Sato + Redness (LAB)"],
        ["Zone Grid",       f"{summary.get('n_rings',3)} rings × {summary.get('n_sectors',12)} sectors = "
                            f"{summary.get('n_rings',3)*summary.get('n_sectors',12)} zones"],
    ]
    story.append(_metrics_table(meta_data, [60, CONTENT - 60]))
    story.append(Spacer(1, 8 * mm))

    # Severity summary
    sev_data = [
        ["Metric", "Value", "Grade"],
        ["Vessel Coverage",
         f"{severity:.2f}%",
         grade],
        ["Vessels Detected",
         str(len(vessel_a)),
         ""],
        ["Affected Clock Hours",
         str(sorted(set(m3_results.get("affected_hours", [])))),
         ""],
        ["Total Vessel Area (px)",
         str(summary.get("total_vessel_px", "N/A")),
         ""],
        ["Cornea Area (px)",
         str(summary.get("cornea_area_px", "N/A")),
         ""],
    ]
    story.append(_metrics_table(sev_data, [70, 50, CONTENT - 120]))
    story.append(Spacer(1, 4 * mm))

    # Show original image on cover
    orig_path = p("module1", "eye_preprocessed.png")
    if not os.path.exists(orig_path):
        # Try to find the original uploaded image
        for ext in [".png", ".jpg", ".jpeg"]:
            cand = os.path.join(session_dir, "input" + ext)
            if os.path.exists(cand):
                orig_path = cand
                break

    if os.path.exists(orig_path):
        story.append(Spacer(1, 3 * mm))
        story.append(_image_grid(
            [(orig_path, "Original Slit-Lamp Image (CLAHE preprocessed)"),
             (m3_paths.get("structure_overlay", ""), "Vessel Structure Overlay [PRIMARY]")],
            [CONTENT / 2, CONTENT / 2],
            col_height_mm=70,
        ))

    story.append(PageBreak())

    # ════════════════════════════════════════════════════════
    # PAGE 2 — MODULE 1: Eye ROI
    # ════════════════════════════════════════════════════════
    story.extend(_section_header(
        "Module 1 — Eye Region of Interest (ROI)",
        "UNet++ / EfficientNet-B5 · CLAHE preprocessing · Fill holes + smooth edges post-processing"
    ))
    story.append(Spacer(1, 3 * mm))
    c = CONTENT / 4
    story.append(_image_grid([
        (p("module1", "eye_preprocessed.png"),  "Preprocessed Input"),
        (p("module1", "eye_raw_mask.png"),       "Raw Predicted Mask"),
        (p("module1", "eye_morph_mask.png"),     "Post-processed Mask"),
        (p("module1", "eye_morph_overlay.png"),  "Final Eye Overlay (green)"),
    ], [c, c, c, c], col_height_mm=55))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(
        "The Eye ROI model localises the eye region. The morphologically "
        "post-processed mask (fill holes + smooth edges) is passed as a "
        "constraint to Module 2, ensuring the cornea model only operates "
        "within the valid eye area.", styles["body"]))
    story.append(PageBreak())

    # ════════════════════════════════════════════════════════
    # PAGE 3 — MODULE 2: Cornea ROI
    # ════════════════════════════════════════════════════════
    story.extend(_section_header(
        "Module 2 — Cornea Region of Interest (ROI)",
        "UNet++ / EfficientNet-B5 · Glare removal → Bilateral filter → CLAHE(L) → Morph clean"
    ))
    story.append(Spacer(1, 3 * mm))
    story.append(_image_grid([
        (p("module2", "cornea_preprocessed.png"),  "Preprocessed Input"),
        (p("module2", "cornea_raw_mask.png"),       "Raw Predicted Mask"),
        (p("module2", "cornea_morph_mask.png"),     "Post-processed Mask"),
        (p("module2", "cornea_morph_overlay.png"),  "Final Cornea Overlay (orange)"),
    ], [c, c, c, c], col_height_mm=55))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(
        "The cornea ROI model uses a richer preprocessing chain to handle "
        "specular reflections and slit-lamp illumination artefacts. The "
        "eye mask from Module 1 is applied before inference (non-eye pixels "
        "zeroed). The resulting cornea mask defines the analysis region for "
        "Modules 3 and 4.", styles["body"]))
    story.append(PageBreak())

    # ════════════════════════════════════════════════════════
    # PAGE 4 — MODULE 3: Vessel Filtering
    # ════════════════════════════════════════════════════════
    story.extend(_section_header(
        "Module 3 — Vessel Filtering & Detection",
        "Frangi + Meijering + Sato ridge filters · LAB redness weighting · Percentile threshold"
    ))
    story.append(Spacer(1, 3 * mm))
    c2 = CONTENT / 2
    story.append(_image_grid([
        (m3_paths.get("structure_overlay", ""), "Structure Overlay [PRIMARY]"),
        (m3_paths.get("structure_heat",    ""), "INFERNO Heatmap"),
    ], [c2, c2], col_height_mm=65))
    story.append(Spacer(1, 2 * mm))
    story.append(_image_grid([
        (m3_paths.get("skeleton",       ""), "Pruned Skeleton"),
        (m3_paths.get("vessel_binary",  ""), "Binary Vessel Mask"),
        (m3_paths.get("vessel_colored", ""), "Individual Vessels (colored)"),
    ], [CONTENT/3, CONTENT/3, CONTENT/3], col_height_mm=50))

    # Per-vessel table
    if vessel_a:
        story.append(Spacer(1, 4 * mm))
        story.append(Paragraph("Per-Vessel Measurements", styles["section"]))
        vrows = [["ID", "Length (px)", "Mean Width (px)", "Tortuosity",
                  "Clock Hours", "Color"]]
        for lid, va in list(vessel_a.items())[:20]:  # cap at 20 rows
            vrows.append([
                str(lid),
                f"{va['length_px']:.0f}",
                f"{va['mean_width_px']:.1f}",
                f"{va['tortuosity']:.2f}",
                str(va["sectors"]),
                va["color_hex"],
            ])
        story.append(_metrics_table(vrows,
            [15, 30, 35, 28, 45, CONTENT - 153]))
    story.append(PageBreak())

    # ════════════════════════════════════════════════════════
    # PAGE 5 — MODULE 3: Debug Strip
    # ════════════════════════════════════════════════════════
    story.extend(_section_header(
        "Module 3 — Diagnostic Strip (4-Row Visualisation)",
        "Row 1: Preprocessing chain  |  Row 2: Ridge filter responses  |  "
        "Row 3: Heatmap / Overlay  |  Row 4: Skeleton / Binary / Overlays"
    ))
    story.append(Spacer(1, 3 * mm))
    debug_path = m3_paths.get("debug_strip", "")
    if os.path.exists(debug_path):
        story.append(_rl_image(debug_path, CONTENT))
    story.append(PageBreak())

    # ════════════════════════════════════════════════════════
    # PAGE 6 — MODULE 4: Spatial Tracking
    # ════════════════════════════════════════════════════════
    story.extend(_section_header(
        "Module 4 — Spatial Zone Tracking",
        f"{summary.get('n_rings',3)} rings × {summary.get('n_sectors',12)} clock sectors = "
        f"{summary.get('n_rings',3)*summary.get('n_sectors',12)} zones  ·  "
        f"Cornea diameter assumed 12 mm"
    ))
    story.append(Spacer(1, 3 * mm))
    story.append(_image_grid([
        (m4_paths.get("zone_map",         ""), "36-Zone Grid"),
        (m4_paths.get("vessel_overlay",   ""), "Vessels by Zone"),
    ], [c2, c2], col_height_mm=65))
    story.append(Spacer(1, 2 * mm))
    story.append(_image_grid([
        (m4_paths.get("tracking_overlay", ""), "Zone Traversal Tracking"),
        (m4_paths.get("severity_heatmap", ""), "Zone Density Heatmap"),
    ], [c2, c2], col_height_mm=65))
    story.append(PageBreak())

    # ════════════════════════════════════════════════════════
    # PAGE 7 — QUANTITATIVE SUMMARY
    # ════════════════════════════════════════════════════════
    story.extend(_section_header("Quantitative Summary"))
    story.append(Spacer(1, 3 * mm))

    # Overall metrics
    ov_rows = [
        ["Metric", "Value"],
        ["Severity Grade",        grade],
        ["Vessel Coverage",       f"{severity:.2f}%"],
        ["Vessels Detected",      str(len(vessel_a))],
        ["Total Vessel Pixels",   str(summary.get("total_vessel_px", "N/A"))],
        ["Cornea Area (px)",      str(summary.get("cornea_area_px", "N/A"))],
        ["Cornea Radius (px)",    str(summary.get("r_eff_px", "N/A"))],
        ["Affected Clock Hours",  str(sorted(set(m3_results.get("affected_hours", []))))],
        ["Analysis Timestamp",    now],
    ]
    story.append(_metrics_table(ov_rows, [80, CONTENT - 80]))
    story.append(Spacer(1, 6 * mm))

    # Zone-density table (top 15 densest zones)
    zone_stats = summary.get("zone_stats", {})
    if zone_stats:
        story.append(Paragraph("Top 15 Zones by Vessel Density", styles["section"]))
        sorted_zones = sorted(zone_stats.items(),
                              key=lambda x: x[1]["density_pct"], reverse=True)[:15]
        zrows = [["Zone", "Ring", "Sector", "Zone Area (px)", "Vessel (px)", "Density %"]]
        for zone_name, zs in sorted_zones:
            zrows.append([
                zone_name,
                str(zs["ring"]),
                str(zs["sector"]),
                str(zs["zone_area_px"]),
                str(zs["vessel_px"]),
                f"{zs['density_pct']:.2f}%",
            ])
        story.append(_metrics_table(zrows,
            [45, 18, 22, 35, 30, CONTENT - 150]))

    story.append(PageBreak())

    # ════════════════════════════════════════════════════════
    # PAGE 8 — DISCLAIMER
    # ════════════════════════════════════════════════════════
    story.extend(_section_header("Disclaimer & Notes"))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(
        "This report is generated automatically by an AI-assisted corneal "
        "vascularization analysis pipeline. All results are intended for "
        "research and clinical support purposes only and must be reviewed "
        "and validated by a qualified ophthalmologist before clinical use. "
        "Vessel measurements are approximate and depend on image quality, "
        "corneal diameter assumptions (12 mm), and model performance.",
        styles["body"]
    ))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(
        f"Report generated: {now}  |  "
        f"Pipeline: Eye ROI (UNet++ EfficientNet-B5) → Cornea ROI (UNet++ EfficientNet-B5) "
        f"→ Vessel Filter (Frangi/Meijering/Sato + LAB redness) → Spatial Tracking (36 zones)",
        styles["caption"]
    ))

    # ── Build PDF ──
    doc.build(story)
    print(f"[Module 5] PDF report written → {out_pdf_path}")
    return out_pdf_path
