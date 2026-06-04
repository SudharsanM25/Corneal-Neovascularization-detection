"""
Module 2 — Cornea ROI Segmentation
=====================================
UNet++ / EfficientNet-B5 model with richer preprocessing:
  RGB → Glare removal (Telea inpaint) → Bilateral filter
       → CLAHE (LAB L, clip=3) → Morph clean (open+close)

Uses eye mask from Module 1 as ROI constraint (zeros non-eye pixels).
Orange overlay to distinguish from Module 1 green.

Outputs saved to: <session_dir>/module2/
  - cornea_raw_mask.png
  - cornea_morph_mask.png
  - cornea_raw_overlay.png
  - cornea_morph_overlay.png
  - cornea_preprocessed.png
"""

import os
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
class _Cfg:
    IMAGE_SIZE = 512
    ENCODER    = "efficientnet-b5"
    DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
    THRESHOLD  = 0.5
    USE_TTA    = False

    # Glare
    GLARE_THRESHOLD = 240
    GLARE_SAT_MAX   = 30
    GLARE_VAL_MIN   = 220
    GLARE_DILATE_PX = 15
    INPAINT_RADIUS  = 3

    # Bilateral
    BILATERAL_D       = 9
    BILATERAL_SIGMA_C = 75
    BILATERAL_SIGMA_S = 75

    # CLAHE
    CLAHE_CLIP = 3.0
    CLAHE_GRID = (8, 8)

    # Morph
    MORPH_KSIZE = 5


# ─────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────
def _detect_glare(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv  = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    _, s, v = cv2.split(hsv)
    _, bright = cv2.threshold(gray, _Cfg.GLARE_THRESHOLD, 255, cv2.THRESH_BINARY)
    glare_sat = ((s < _Cfg.GLARE_SAT_MAX) & (v > _Cfg.GLARE_VAL_MIN)).astype(np.uint8) * 255
    combined  = cv2.bitwise_or(bright, glare_sat)
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (_Cfg.GLARE_DILATE_PX * 2 + 1, _Cfg.GLARE_DILATE_PX * 2 + 1)
    )
    return cv2.dilate(combined, k)


def _remove_glare(rgb: np.ndarray) -> np.ndarray:
    mask = _detect_glare(rgb)
    bgr  = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    out  = cv2.inpaint(bgr, mask, _Cfg.INPAINT_RADIUS, cv2.INPAINT_TELEA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def _apply_bilateral(rgb: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    out = cv2.bilateralFilter(bgr, _Cfg.BILATERAL_D,
                              _Cfg.BILATERAL_SIGMA_C, _Cfg.BILATERAL_SIGMA_S)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def _apply_clahe(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=_Cfg.CLAHE_CLIP,
                         tileGridSize=_Cfg.CLAHE_GRID).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)


def _apply_morph(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (_Cfg.MORPH_KSIZE, _Cfg.MORPH_KSIZE))
    l = cv2.morphologyEx(l, cv2.MORPH_OPEN,  k)
    l = cv2.morphologyEx(l, cv2.MORPH_CLOSE, k)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)


def _preprocess(rgb: np.ndarray) -> np.ndarray:
    rgb = _remove_glare(rgb)
    rgb = _apply_bilateral(rgb)
    rgb = _apply_clahe(rgb)
    rgb = _apply_morph(rgb)
    return rgb


# ─────────────────────────────────────────────────────────────
# POST-PROCESSING
# ─────────────────────────────────────────────────────────────
def _fill_holes(mask: np.ndarray) -> np.ndarray:
    u8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(u8)
    cv2.drawContours(filled, contours, -1, 255, thickness=-1)
    return (filled > 0).astype(np.uint8)


def _smooth_edges(mask: np.ndarray, ksize=3) -> np.ndarray:
    k = np.ones((ksize, ksize), np.uint8)
    mask = cv2.morphologyEx((mask > 0).astype(np.uint8), cv2.MORPH_CLOSE, k)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)


# ─────────────────────────────────────────────────────────────
# TRANSFORM + TTA
# ─────────────────────────────────────────────────────────────
def _transform():
    return A.Compose([
        A.Resize(_Cfg.IMAGE_SIZE, _Cfg.IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def _tta_predict(model, tensor: torch.Tensor) -> np.ndarray:
    preds = []
    for hf in [False, True]:
        for vf in [False, True]:
            x = tensor.clone()
            if hf: x = torch.flip(x, dims=[3])
            if vf: x = torch.flip(x, dims=[2])
            with torch.no_grad():
                out = model(x)
                if isinstance(out, (list, tuple)): out = out[-1]
                p = torch.sigmoid(out)
            if hf: p = torch.flip(p, dims=[3])
            if vf: p = torch.flip(p, dims=[2])
            preds.append(p.cpu().numpy()[0, 0])
    return np.mean(preds, axis=0)


# ─────────────────────────────────────────────────────────────
# OVERLAY — orange for cornea
# ─────────────────────────────────────────────────────────────
def _make_overlay(bgr: np.ndarray, mask: np.ndarray,
                  alpha=0.35, color=(0, 165, 255)) -> np.ndarray:
    ov = np.zeros_like(bgr)
    ov[mask == 1] = color
    return cv2.addWeighted(bgr, 1 - alpha, ov, alpha, 0)


# ─────────────────────────────────────────────────────────────
# MODEL CACHE
# ─────────────────────────────────────────────────────────────
_model_cache = {}

def _load_model(model_path: str):
    if model_path in _model_cache:
        return _model_cache[model_path]
    model = smp.UnetPlusPlus(
        encoder_name=_Cfg.ENCODER,
        encoder_weights=None,
        in_channels=3,
        classes=1,
        deep_supervision=True,
    ).to(_Cfg.DEVICE)
    ckpt  = torch.load(model_path, map_location=_Cfg.DEVICE)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    _model_cache[model_path] = model
    return model


# ─────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────
def run(image_path: str, model_path: str, out_dir: str,
        eye_mask: np.ndarray = None,
        use_tta: bool = False, threshold: float = 0.5) -> dict:
    """
    Run Module 2 cornea-ROI inference on a single image.

    Parameters
    ----------
    eye_mask : np.ndarray (H x W, 0/1 uint8) from Module 1.
               If None, full image is used.

    Returns
    -------
    dict with keys:
        morph_mask  : np.ndarray (H x W, uint8 0/1)
        paths       : dict of saved file paths
    """
    os.makedirs(out_dir, exist_ok=True)
    _Cfg.USE_TTA   = use_tta
    _Cfg.THRESHOLD = threshold

    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    rgb     = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = bgr.shape[:2]

    # Preprocess
    pp_rgb = _preprocess(rgb.copy())

    # Save preprocessed
    pp_path = os.path.join(out_dir, "cornea_preprocessed.png")
    cv2.imwrite(pp_path, cv2.cvtColor(pp_rgb, cv2.COLOR_RGB2BGR))

    # Apply eye mask ROI (zero-out non-eye pixels)
    if eye_mask is not None:
        eye_resized = cv2.resize(eye_mask, (orig_w, orig_h),
                                 interpolation=cv2.INTER_NEAREST)
        rgb_roi = pp_rgb.copy()
        rgb_roi[eye_resized == 0] = 0
    else:
        eye_resized = np.ones((orig_h, orig_w), dtype=np.uint8)
        rgb_roi = pp_rgb

    # Inference
    model  = _load_model(model_path)
    tf     = _transform()
    tensor = tf(image=rgb_roi)["image"].unsqueeze(0).to(_Cfg.DEVICE)

    if _Cfg.USE_TTA:
        prob = _tta_predict(model, tensor)
    else:
        with torch.no_grad():
            out = model(tensor)
            if isinstance(out, (list, tuple)): out = out[-1]
            prob = torch.sigmoid(out).cpu().numpy()[0, 0]

    # Raw mask → apply eye constraint
    raw_mask = (prob > _Cfg.THRESHOLD).astype(np.uint8)
    raw_mask = cv2.resize(raw_mask, (orig_w, orig_h),
                          interpolation=cv2.INTER_NEAREST)
    raw_mask = raw_mask * eye_resized

    # Morph post-process → apply eye constraint again
    morph_mask = _smooth_edges(_fill_holes(raw_mask.copy()))
    morph_mask = morph_mask * eye_resized

    # Save
    paths = {}
    paths["cornea_raw_mask"]      = os.path.join(out_dir, "cornea_raw_mask.png")
    paths["cornea_morph_mask"]    = os.path.join(out_dir, "cornea_morph_mask.png")
    paths["cornea_raw_overlay"]   = os.path.join(out_dir, "cornea_raw_overlay.png")
    paths["cornea_morph_overlay"] = os.path.join(out_dir, "cornea_morph_overlay.png")
    paths["cornea_preprocessed"]  = pp_path

    cv2.imwrite(paths["cornea_raw_mask"],      raw_mask   * 255)
    cv2.imwrite(paths["cornea_morph_mask"],    morph_mask * 255)
    cv2.imwrite(paths["cornea_raw_overlay"],   _make_overlay(bgr, raw_mask))
    cv2.imwrite(paths["cornea_morph_overlay"], _make_overlay(bgr, morph_mask))

    print(f"[Module 2] Cornea ROI done → {out_dir}")
    return {"morph_mask": morph_mask, "raw_mask": raw_mask, "paths": paths}
