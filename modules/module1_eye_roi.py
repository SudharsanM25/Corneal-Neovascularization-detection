"""
Module 1 — Eye ROI Segmentation
=================================
UNet++ / EfficientNet-B5 model.
Preprocessing: CLAHE (LAB L-channel) + optional morphological clean.
Post-processing: fill_holes → smooth_edges.
TTA: 4-flip average (optional).

Outputs saved to: <session_dir>/module1/
  - eye_raw_mask.png
  - eye_morph_mask.png
  - eye_raw_overlay.png
  - eye_morph_overlay.png
  - preprocessed.png
"""

import os
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ─────────────────────────────────────────────────────────────
# CONFIG  (override via run() kwargs)
# ─────────────────────────────────────────────────────────────
class _Cfg:
    IMAGE_SIZE = 512
    ENCODER    = "efficientnet-b5"
    DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
    THRESHOLD  = 0.5
    USE_CLAHE  = True
    USE_MORPH  = False
    USE_TTA    = False


# ─────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────
def _apply_clahe(image_rgb: np.ndarray, clip=2.0, grid=(8, 8)) -> np.ndarray:
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)


def _apply_morph(image_rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    l = cv2.morphologyEx(l, cv2.MORPH_OPEN,  k)
    l = cv2.morphologyEx(l, cv2.MORPH_CLOSE, k)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)


def _preprocess(image_rgb: np.ndarray) -> np.ndarray:
    if _Cfg.USE_CLAHE:
        image_rgb = _apply_clahe(image_rgb)
    if _Cfg.USE_MORPH:
        image_rgb = _apply_morph(image_rgb)
    return image_rgb


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
# TRANSFORM
# ─────────────────────────────────────────────────────────────
def _transform():
    return A.Compose([
        A.Resize(_Cfg.IMAGE_SIZE, _Cfg.IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# ─────────────────────────────────────────────────────────────
# TTA
# ─────────────────────────────────────────────────────────────
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
# OVERLAY
# ─────────────────────────────────────────────────────────────
def _make_overlay(bgr: np.ndarray, mask: np.ndarray,
                  alpha=0.35, color=(0, 255, 0)) -> np.ndarray:
    ov = np.zeros_like(bgr)
    ov[mask == 1] = color
    return cv2.addWeighted(bgr, 1 - alpha, ov, alpha, 0)


# ─────────────────────────────────────────────────────────────
# LOAD MODEL (cached in module-level dict)
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
        use_tta: bool = False, threshold: float = 0.5) -> dict:
    """
    Run Module 1 eye-ROI inference on a single image.

    Returns
    -------
    dict with keys:
        morph_mask  : np.ndarray  (H x W, uint8 0/1)
        paths       : dict of saved file paths
    """
    os.makedirs(out_dir, exist_ok=True)
    _Cfg.USE_TTA   = use_tta
    _Cfg.THRESHOLD = threshold

    # Load & preprocess
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pp_rgb = _preprocess(rgb.copy())

    # Save preprocessed image
    pp_path = os.path.join(out_dir, "eye_preprocessed.png")
    cv2.imwrite(pp_path, cv2.cvtColor(pp_rgb, cv2.COLOR_RGB2BGR))

    # Model inference
    model  = _load_model(model_path)
    tf     = _transform()
    tensor = tf(image=pp_rgb)["image"].unsqueeze(0).to(_Cfg.DEVICE)

    if _Cfg.USE_TTA:
        prob = _tta_predict(model, tensor)
    else:
        with torch.no_grad():
            out = model(tensor)
            if isinstance(out, (list, tuple)): out = out[-1]
            prob = torch.sigmoid(out).cpu().numpy()[0, 0]

    # Raw mask at original resolution
    raw_mask = (prob > _Cfg.THRESHOLD).astype(np.uint8)
    raw_mask = cv2.resize(raw_mask, (bgr.shape[1], bgr.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    # Morph post-process
    morph_mask = _smooth_edges(_fill_holes(raw_mask.copy()))

    # Save outputs
    paths = {}
    paths["eye_raw_mask"]      = os.path.join(out_dir, "eye_raw_mask.png")
    paths["eye_morph_mask"]    = os.path.join(out_dir, "eye_morph_mask.png")
    paths["eye_raw_overlay"]   = os.path.join(out_dir, "eye_raw_overlay.png")
    paths["eye_morph_overlay"] = os.path.join(out_dir, "eye_morph_overlay.png")
    paths["eye_preprocessed"]  = pp_path

    cv2.imwrite(paths["eye_raw_mask"],      raw_mask   * 255)
    cv2.imwrite(paths["eye_morph_mask"],    morph_mask * 255)
    cv2.imwrite(paths["eye_raw_overlay"],   _make_overlay(bgr, raw_mask))
    cv2.imwrite(paths["eye_morph_overlay"], _make_overlay(bgr, morph_mask))

    print(f"[Module 1] Eye ROI done → {out_dir}")
    return {"morph_mask": morph_mask, "raw_mask": raw_mask, "paths": paths}
