"""
app.py — FastAPI Backend
=========================
Endpoints:
  POST /analyze          — upload image + params → start analysis job
  GET  /status/{job_id}  — poll job status & progress
  GET  /result/{job_id}  — get full result JSON
  GET  /download/{job_id}/{filename} — download any output file
  GET  /                 — serve frontend HTML

Run:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import uuid
import shutil
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import pipeline as pl
from huggingface_hub import hf_hub_download
import os

def download_models():
    token = os.environ.get("HF_TOKEN_MODEL")
    
    if not os.path.exists("eye_unetpp_effb5_draft1_morphTrue_ttaFalse.pth"):
        hf_hub_download(
            repo_id="SudharsanM25/corneal-neovascularization-models",
            filename="eye_unetpp_effb5_draft1_morphTrue_ttaFalse.pth",
            local_dir=".",
            token=token
        )
    if not os.path.exists("cornea_unetpp_effb5_draft3_glareTrue_bilTrue_morphTrue_ttaFalse_best_iou.pth"):
        hf_hub_download(
            repo_id="SudharsanM25/corneal-neovascularization-models",
            filename="cornea_unetpp_effb5_draft3_glareTrue_bilTrue_morphTrue_ttaFalse_best_iou.pth",
            local_dir=".",
            token=token
        )

download_models()

# ─────────────────────────────────────────────────────────────
# CONFIG  — update model paths to your actual checkpoint files
# ─────────────────────────────────────────────────────────────
EYE_MODEL_PATH    = os.environ.get(
    "EYE_MODEL_PATH",
    "eye_unetpp_effb5_draft1_morphTrue_ttaFalse.pth"
)
CORNEA_MODEL_PATH = os.environ.get(
    "CORNEA_MODEL_PATH",
    "cornea_unetpp_effb5_draft3_glareTrue_bilTrue_morphTrue_ttaFalse_best_iou.pth"
)
RESULTS_DIR = "./results"
UPLOADS_DIR = "./uploads"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────
app = FastAPI(title="Corneal Vascularization Analyzer", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# Mount static files (frontend assets)
if os.path.isdir("./static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


# ─────────────────────────────────────────────────────────────
# IN-MEMORY JOB STORE
# ─────────────────────────────────────────────────────────────
jobs: dict = {}   # job_id → {status, progress, step, total, message, result, error}


def _update_job(job_id, **kwargs):
    if job_id in jobs:
        jobs[job_id].update(kwargs)


# ─────────────────────────────────────────────────────────────
# BACKGROUND WORKER
# ─────────────────────────────────────────────────────────────
def _run_job(job_id: str, image_path: str, out_dir: str,
             sensitivity: int, use_tta: bool, threshold: float,
             patient_id: str):

    def progress_cb(step, total, msg):
        _update_job(job_id, step=step, total=total, message=msg,
                    progress=int(step / total * 100))

    _update_job(job_id, status="running", progress=0, step=0, total=6)

    result = pl.run_pipeline(
        image_path=image_path,
        eye_model_path=EYE_MODEL_PATH,
        cornea_model_path=CORNEA_MODEL_PATH,
        out_dir=out_dir,
        sensitivity=sensitivity,
        use_tta=use_tta,
        threshold=threshold,
        patient_id=patient_id,
        progress_cb=progress_cb,
    )

    if result["success"]:
        _update_job(job_id, status="done", progress=100,
                    message="Analysis complete!", result=result)
    else:
        _update_job(job_id, status="error", progress=0,
                    message=result.get("error", "Unknown error"),
                    error=result.get("error"))


# ─────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────

@app.post("/analyze")
async def analyze(
    file:        UploadFile = File(...),
    sensitivity: int        = Form(50),
    use_tta:     bool       = Form(False),
    threshold:   float      = Form(0.5),
    patient_id:  str        = Form("N/A"),
):
    """Upload a slit-lamp image and start the analysis pipeline."""
    job_id  = str(uuid.uuid4())[:8]
    ext     = Path(file.filename).suffix.lower()
    allowed = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported file type: {ext}")

    # Save upload
    img_path = os.path.join(UPLOADS_DIR, f"{job_id}{ext}")
    with open(img_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Prepare output dir
    out_dir = os.path.join(RESULTS_DIR, job_id)
    os.makedirs(out_dir, exist_ok=True)

    # Init job record
    jobs[job_id] = {
        "job_id":     job_id,
        "filename":   file.filename,
        "status":     "queued",
        "progress":   0,
        "step":       0,
        "total":      6,
        "message":    "Queued",
        "result":     None,
        "error":      None,
        "out_dir":    out_dir,
        "image_path": img_path,
    }

    # Start background thread
    t = threading.Thread(
        target=_run_job,
        args=(job_id, img_path, out_dir, sensitivity, use_tta, threshold, patient_id),
        daemon=True,
    )
    t.start()

    return JSONResponse({"job_id": job_id, "status": "queued"})


@app.get("/status/{job_id}")
async def status(job_id: str):
    """Poll job status and progress."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    j = jobs[job_id]
    return JSONResponse({
        "job_id":   job_id,
        "status":   j["status"],
        "progress": j["progress"],
        "step":     j["step"],
        "total":    j["total"],
        "message":  j["message"],
    })


@app.get("/result/{job_id}")
async def result(job_id: str):
    """Get full result info for a completed job."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    j = jobs[job_id]
    if j["status"] != "done":
        return JSONResponse({"status": j["status"], "message": j["message"]})

    # Build a clean summary for the frontend
    r       = j["result"]
    m3_r    = r.get("m3", {})
    m4_r    = r.get("m4", {})
    summary = m4_r.get("summary", {})

    def _rel(p):
        """Return path relative to results root for download."""
        if p and os.path.exists(p):
            return f"/download/{job_id}/{Path(p).name}"
        return None

    m3_paths = m3_r.get("paths", {})
    m4_paths = m4_r.get("paths", {})
    all_module_paths = {}

    # Collect all module output paths
    for mod, paths_dict in [
        ("m1", r.get("m1", {}).get("paths", {})),
        ("m2", r.get("m2", {}).get("paths", {})),
        ("m3", m3_paths),
        ("m4", m4_paths),
    ]:
        for key, pth in paths_dict.items():
            if pth and os.path.exists(pth):
                all_module_paths[f"{mod}_{key}"] = f"/download/{job_id}/{Path(pth).name}"

    return JSONResponse({
        "job_id":          job_id,
        "status":          "done",
        "filename":        j["filename"],
        "severity_pct":    m3_r.get("severity", 0),
        "vessels":         len(m3_r.get("vessel_analysis", {})),
        "affected_hours":  m3_r.get("affected_hours", []),
        "coverage_pct":    summary.get("vessel_coverage_pct", 0),
        "pdf_url":         f"/download/{job_id}/cornea_report.pdf",
        "paths":           all_module_paths,
    })


@app.get("/download/{job_id}/{filename}")
async def download(job_id: str, filename: str):
    """Serve any output file from a job's results directory."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    out_dir = jobs[job_id]["out_dir"]
    # Search in out_dir and all subdirectories
    for root, dirs, files in os.walk(out_dir):
        if filename in files:
            fp = os.path.join(root, filename)
            media = "application/pdf" if filename.endswith(".pdf") else None
            return FileResponse(fp, filename=filename, media_type=media)

    # Also check top-level results dir
    fp = os.path.join(out_dir, filename)
    if os.path.exists(fp):
        media = "application/pdf" if filename.endswith(".pdf") else None
        return FileResponse(fp, filename=filename, media_type=media)

    raise HTTPException(404, f"File not found: {filename}")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend."""
    html_path = "./static/index.html"
    if os.path.exists(html_path):
        with open(html_path, encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>Frontend not found. Place index.html in ./static/</h1>")


@app.get("/health")
async def health():
    return {"status": "ok", "eye_model": EYE_MODEL_PATH,
            "cornea_model": CORNEA_MODEL_PATH}
