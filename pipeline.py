"""
pipeline.py — Master Orchestrator
====================================
Runs all 5 modules in sequence for a single slit-lamp image.

Usage (CLI):
    python pipeline.py --image path/to/image.jpg \
                       --eye_model path/to/eye_model.pth \
                       --cornea_model path/to/cornea_model.pth \
                       --out_dir ./results/session_001

Usage (API / import):
    from pipeline import run_pipeline
    results = run_pipeline(
        image_path   = "image.jpg",
        eye_model    = "eye.pth",
        cornea_model = "cornea.pth",
        out_dir      = "./results/session_001",
    )
"""

import os
import sys
import json
import argparse
import traceback
from pathlib import Path

# ── Module imports ──
from modules import (
    module1_eye_roi,
    module2_cornea_roi,
    module3_vessel_filter,
    module4_spatial_tracking,
    module5_pdf_report,
)


def run_pipeline(
    image_path: str,
    eye_model_path: str,
    cornea_model_path: str,
    out_dir: str,
    sensitivity: int = 50,
    use_tta: bool = False,
    threshold: float = 0.5,
    patient_id: str = "N/A",
    progress_cb=None,        # optional callable(step:int, total:int, msg:str)
) -> dict:
    """
    Run the full corneal vascularization pipeline on a single image.

    Returns
    -------
    dict with keys:
        m1, m2, m3, m4   — per-module result dicts
        pdf_path          — path to generated PDF
        session_dir       — base output directory
        success           — bool
        error             — str or None
    """

    def _progress(step, msg):
        if progress_cb:
            progress_cb(step, 6, msg)
        print(f"[Pipeline] Step {step}/6 — {msg}")

    results = {
        "m1": None, "m2": None, "m3": None, "m4": None,
        "pdf_path": None, "session_dir": out_dir,
        "success": False, "error": None,
    }

    os.makedirs(out_dir, exist_ok=True)

    try:
        # ── STEP 1: Save a copy of the input for reference ──
        import shutil, cv2
        import numpy as np
        img_ext  = Path(image_path).suffix
        inp_copy = os.path.join(out_dir, f"input{img_ext}")
        shutil.copy2(image_path, inp_copy)

        # ── MODULE 1: Eye ROI ──
        _progress(1, "Eye ROI segmentation")
        m1_dir = os.path.join(out_dir, "module1")
        m1     = module1_eye_roi.run(
            image_path=image_path,
            model_path=eye_model_path,
            out_dir=m1_dir,
            use_tta=use_tta,
            threshold=threshold,
        )
        results["m1"] = {k: v for k, v in m1.items() if k != "morph_mask" and k != "raw_mask"}
        eye_mask = m1["morph_mask"]  # H x W, uint8 0/1

        # ── MODULE 2: Cornea ROI ──
        _progress(2, "Cornea ROI segmentation")
        m2_dir = os.path.join(out_dir, "module2")
        m2     = module2_cornea_roi.run(
            image_path=image_path,
            model_path=cornea_model_path,
            out_dir=m2_dir,
            eye_mask=eye_mask,
            use_tta=use_tta,
            threshold=threshold,
        )
        results["m2"] = {k: v for k, v in m2.items() if k != "morph_mask" and k != "raw_mask"}
        cornea_mask = m2["morph_mask"] * 255   # convert 0/1 → 0/255 for module 3/4

        # ── MODULE 3: Vessel Filtering ──
        _progress(3, "Vessel filtering & detection")
        m3_dir = os.path.join(out_dir, "module3")
        m3     = module3_vessel_filter.run(
            image_path=image_path,
            out_dir=m3_dir,
            cornea_mask=cornea_mask,
            sensitivity=sensitivity,
        )
        results["m3"] = {k: v for k, v in m3.items()
                         if k not in ("final_vessels", "skeleton", "structure_map")}

        # ── MODULE 4: Spatial Tracking ──
        _progress(4, "Spatial zone tracking")
        m4_dir = os.path.join(out_dir, "module4")
        m4     = module4_spatial_tracking.run(
            image_path=image_path,
            out_dir=m4_dir,
            cornea_mask=cornea_mask,
            vessel_mask=m3["final_vessels"],
        )
        results["m4"] = m4

        # ── MODULE 5: PDF Report ──
        _progress(5, "Generating PDF report")
        pdf_path = os.path.join(out_dir, "cornea_report.pdf")
        module5_pdf_report.run(
            out_pdf_path=pdf_path,
            session_dir=out_dir,
            m3_results=m3,
            m4_results=m4,
            image_filename=Path(image_path).name,
            patient_id=patient_id,
        )
        results["pdf_path"] = pdf_path

        # ── Save master summary JSON ──
        _progress(6, "Saving summary")
        master = {
            "image":          Path(image_path).name,
            "patient_id":     patient_id,
            "severity_pct":   m3["severity"],
            "vessels":        len(m3["vessel_analysis"]),
            "affected_hours": m3["affected_hours"],
            "coverage_pct":   m4["summary"].get("vessel_coverage_pct", 0),
            "n_zones":        m4["summary"].get("n_rings", 3) * m4["summary"].get("n_sectors", 12),
            "module_paths": {
                "m1": m1["paths"],
                "m2": m2["paths"],
                "m3": m3["paths"],
                "m4": m4["paths"],
                "pdf": pdf_path,
            },
        }
        with open(os.path.join(out_dir, "pipeline_summary.json"), "w") as f:
            json.dump(master, f, indent=2)

        results["success"] = True
        print(f"\n[Pipeline] ✅ Complete → {out_dir}")
        return results

    except Exception as exc:
        tb = traceback.format_exc()
        results["error"] = f"{exc}\n\n{tb}"
        print(f"\n[Pipeline] ❌ Error: {exc}")
        print(tb)
        return results


# ─────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Corneal Vascularization Pipeline")
    parser.add_argument("--image",        required=True,  help="Path to slit-lamp image")
    parser.add_argument("--eye_model",    required=True,  help="Path to eye ROI model .pth")
    parser.add_argument("--cornea_model", required=True,  help="Path to cornea model .pth")
    parser.add_argument("--out_dir",      default="./results/session", help="Output directory")
    parser.add_argument("--sensitivity",  type=int, default=50, help="Vessel sensitivity 0-100")
    parser.add_argument("--tta",          action="store_true",  help="Enable TTA (4-flip)")
    parser.add_argument("--threshold",    type=float, default=0.5)
    parser.add_argument("--patient_id",   default="N/A")
    args = parser.parse_args()

    run_pipeline(
        image_path=args.image,
        eye_model_path=args.eye_model,
        cornea_model_path=args.cornea_model,
        out_dir=args.out_dir,
        sensitivity=args.sensitivity,
        use_tta=args.tta,
        threshold=args.threshold,
        patient_id=args.patient_id,
    )
