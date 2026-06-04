Here’s your **complete, clean README (updated with Python 3.11 + environment variables)** — ready to copy 👇

---

# CornealAI — Corneal Vascularization Analyzer

A full clinical-grade pipeline for automated corneal vascularization analysis
from slit-lamp images. Upload one image, get masks, overlays, heatmaps, zone
tracking, and a PDF report — all in one click.

---

## 📁 Project Structure

```
cornea_app/
├── modules/
│   ├── __init__.py
│   ├── module1_eye_roi.py          # Eye ROI segmentation (UNet++ EfficientNet-B5)
│   ├── module2_cornea_roi.py       # Cornea ROI segmentation (glare+bilateral+CLAHE)
│   ├── module3_vessel_filter.py    # Vessel detection (Frangi+Meijering+Sato+redness)
│   ├── module4_spatial_tracking.py # 36-zone spatial analysis & tracking
│   └── module5_pdf_report.py       # 8-page clinical PDF report
├── static/
│   └── index.html                  # Frontend (drag-drop, live progress, gallery)
├── pipeline.py                     # Master pipeline orchestrator
├── app.py                          # FastAPI backend
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 🔹 0. Use Python 3.11 (Important)

This project is tested and optimized for:

* **Python 3.11**

Check your version:

```bash
python --version
```

If needed, install Python 3.11 and ensure it is available in your system PATH.

---

### 🔹 1. Create Virtual Environment

```bash
# Create virtual environment using Python 3.11
python3.11 -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/macOS)
source venv/bin/activate
```

---

### 🔹 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 🔹 3. Set Model Paths

You can either hardcode paths in `app.py`:

```python
EYE_MODEL_PATH    = "path/to/eye_unetpp_effb5_draft1.pth"
CORNEA_MODEL_PATH = "path/to/cornea_unetpp_effb5_draft3_best_loss.pth"
```

Or use environment variables (recommended for clean setup):

#### 🟢 Windows (PowerShell)

```powershell
$env:EYE_MODEL_PATH="C:\path\to\eye_model.pth"
$env:CORNEA_MODEL_PATH="C:\path\to\cornea_model.pth"
```

#### 🟢 Linux / macOS

```bash
export EYE_MODEL_PATH="/path/to/eye_model.pth"
export CORNEA_MODEL_PATH="/path/to/cornea_model.pth"
```

---

### 🔹 4. (Optional) Persistent Environment Variables

#### Windows

```powershell
setx EYE_MODEL_PATH "C:\path\to\eye_model.pth"
setx CORNEA_MODEL_PATH "C:\path\to\cornea_model.pth"
```

#### Linux/macOS

Add to `.bashrc` or `.zshrc`:

```bash
echo 'export EYE_MODEL_PATH="/path/to/eye_model.pth"' >> ~/.bashrc
echo 'export CORNEA_MODEL_PATH="/path/to/cornea_model.pth"' >> ~/.bashrc
source ~/.bashrc
```

---

### 🔹 5. Run the Server

```bash
cd cornea_app
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser at:

```
http://localhost:8000
```

---

## 🔬 Pipeline Modules

| Module | Task          | Model / Method                           | Key Outputs                                         |
| ------ | ------------- | ---------------------------------------- | --------------------------------------------------- |
| **M1** | Eye ROI       | UNet++ / EfficientNet-B5                 | `eye_raw_mask`, `eye_morph_overlay`                 |
| **M2** | Cornea ROI    | UNet++ / EfficientNet-B5 + glare removal | `cornea_morph_mask`, `cornea_morph_overlay`         |
| **M3** | Vessel Filter | Frangi + Meijering + Sato + LAB redness  | `structure_overlay`, `vessel_binary`, `debug_strip` |
| **M4** | Spatial Track | 3 rings × 12 sectors = 36 zones          | `zone_map`, `tracking_overlay`, `severity_heatmap`  |
| **M5** | PDF Report    | ReportLab                                | `cornea_report.pdf` (8 pages)                       |

---

## 🖥️ CLI Usage

```bash
python pipeline.py \
  --image        path/to/slitlamp.jpg \
  --eye_model    path/to/eye.pth \
  --cornea_model path/to/cornea.pth \
  --out_dir      ./results/case_001 \
  --sensitivity  50 \
  --threshold    0.5 \
  --patient_id   "PT-2024-001"
```

---

## 📊 Output Structure (per session)

```
results/{job_id}/
├── input.jpg
├── pipeline_summary.json
├── cornea_report.pdf
├── module1/
│   ├── eye_preprocessed.png
│   ├── eye_raw_mask.png
│   ├── eye_morph_mask.png
│   ├── eye_raw_overlay.png
│   └── eye_morph_overlay.png
├── module2/
│   ├── cornea_preprocessed.png
│   ├── cornea_raw_mask.png
│   ├── cornea_morph_mask.png
│   ├── cornea_raw_overlay.png
│   └── cornea_morph_overlay.png
├── module3/
│   ├── structure_gray.png
│   ├── structure_heat.png
│   ├── structure_overlay.png
│   ├── skeleton.png
│   ├── vessel_binary.png
│   ├── vessel_colored.png
│   └── debug_strip.png
└── module4/
    ├── zone_map.png
    ├── vessel_overlay.png
    ├── tracking_overlay.png
    ├── severity_heatmap.png
    ├── zone_report.csv
    └── summary.json
```

---

## 📄 PDF Report Pages

1. Cover — Patient info, severity grade, summary table, visuals
2. Module 1 — Eye ROI preprocessing + mask + overlay
3. Module 2 — Cornea ROI preprocessing chain
4. Module 3 — Vessel filtering + measurements
5. Module 3 Debug — Diagnostic strip
6. Module 4 Spatial — Zones + tracking + heatmap
7. Quantitative Summary — Metrics + densest zones
8. Disclaimer — Clinical guidance

---

## 🛠️ Tuning Parameters

| Parameter     | Default | Effect                                         |
| ------------- | ------- | ---------------------------------------------- |
| `sensitivity` | 50      | Lower = fewer vessels, Higher = more vessels   |
| `threshold`   | 0.5     | Model confidence cutoff                        |
| `use_tta`     | False   | Test-time augmentation (slower, more accurate) |

---

## 📌 Notes

* Cornea diameter is assumed to be **12 mm** for all measurements
* `structure_overlay` is the **primary clinical visualization**
* Model checkpoints are **not included** — set `.pth` paths before running
* Always activate your virtual environment before running the project

---
