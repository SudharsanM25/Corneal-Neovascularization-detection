"""
Module 3 — Vessel Filtering & Analysis
========================================
Classical multi-scale vessel detection pipeline (vessel_hashv3 logic):

  1. Adaptive CLAHE on green channel (brightness-dependent clip/tile)
  2. Red-suppressed + inverted-red vessel contrast (3-channel fuse)
  3. LAB redness weighting with sigmoid probability
  4. Multi-scale Frangi + Meijering + Sato ridge filtering
  5. Power-law contrast stretch (sensitivity-driven)
  6. Percentile threshold → structure binary
  7. Morphological gap-bridging, skeletonization + pruning
  8. Per-vessel: length, width, tortuosity, clock-hour sectors

Outputs saved to: <session_dir>/module3/
  - structure_gray.png     — raw probability map (uint8)
  - structure_heat.png     — INFERNO colourmap heatmap
  - structure_overlay.png  — heatmap blended on original (PRIMARY visual)
  - skeleton.png           — pruned vessel skeleton
  - vessel_binary.png      — binary vessel mask
  - vessel_colored.png     — each vessel in its own color
  - debug_strip.png        — 4-row diagnostic strip
"""

import os
import itertools
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.cm as cm
import numpy as np
from skimage.color import rgb2lab
from skimage.filters import frangi, meijering, sato
from skimage.morphology import disk, remove_small_objects, skeletonize

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
class _Cfg:
    SENSITIVITY              = 50
    MIN_VESSEL_LEN           = 15
    GAP_CLOSING              = 7
    FRANGI_SIGMAS            = range(1, 10)
    MEIJERING_SIGMAS         = range(1, 10)
    SATO_SIGMAS              = range(1, 10)
    FRANGI_GREEN_SIGMAS      = range(1, 7)
    MIN_VESSEL_AREA_PX       = 30
    STRUCTURE_OVERLAY_ALPHA  = 0.55
    THUMB_W                  = 380
    PANEL_FONT_SCALE         = 0.75
    PANEL_FONT_COLOR         = (0, 255, 255)


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def _get_clock_hour(ys, xs, center):
    angles = np.degrees(np.arctan2(ys - center[1], xs - center[0]))
    values = (angles + 90) / 30.0
    hours  = np.floor(values).astype(int) % 12
    hours[hours == 0] = 12
    return hours


def _color_for_label(label_id: int):
    hue = (37 * label_id) % 180
    hsv = np.uint8([[[hue, 255, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return (int(bgr[2]), int(bgr[1]), int(bgr[0]))


def _normalize(arr: np.ndarray) -> np.ndarray:
    vmin, vmax = arr.min(), arr.max()
    if vmax - vmin < 1e-10:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - vmin) / (vmax - vmin)).astype(np.float32)


# ─────────────────────────────────────────────────────────────
# HEATMAP + OVERLAY
# ─────────────────────────────────────────────────────────────
def _structure_to_heatmap(structure_map: np.ndarray) -> np.ndarray:
    norm  = structure_map.astype(np.float32) / 255.0
    color = (cm.inferno(norm)[:, :, :3] * 255).astype(np.uint8)
    return cv2.cvtColor(color, cv2.COLOR_RGB2BGR)


def _make_structure_overlay(orig_rgb, structure_map, cornea_mask):
    orig_bgr = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)
    heat_bgr = _structure_to_heatmap(structure_map)
    roi      = (cornea_mask > 127)
    blended  = orig_bgr.copy()
    a        = _Cfg.STRUCTURE_OVERLAY_ALPHA
    blended[roi] = (
        (1.0 - a) * orig_bgr[roi].astype(np.float32)
        + a * heat_bgr[roi].astype(np.float32)
    ).astype(np.uint8)
    return blended


# ─────────────────────────────────────────────────────────────
# CORNEA GEOMETRY (center from mask moments)
# ─────────────────────────────────────────────────────────────
def _cornea_center(cornea_mask: np.ndarray):
    M = cv2.moments(cornea_mask)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        h, w = cornea_mask.shape
        cx, cy = w // 2, h // 2
    return (cx, cy)


# ─────────────────────────────────────────────────────────────
# CORE VESSEL ANALYSIS  (vessel_hashv3 logic — unchanged)
# ─────────────────────────────────────────────────────────────
def _analyze_vessels(image_rgb, cornea_mask, center, sensitivity=50,
                     min_vessel_len=15, gap_closing=7):
    sens_t         = sensitivity / 100.0
    power_exp      = 2.2  - 0.8  * sens_t
    redness_center = 0.45 - 0.10 * sens_t
    redness_steep  = 6    + 2    * sens_t

    green = image_rgb[:, :, 1]
    red   = image_rgb[:, :, 0]

    # Adaptive CLAHE
    mean_b = float(np.mean(green[cornea_mask > 0])) if np.count_nonzero(cornea_mask) else 128.0
    if mean_b < 60:
        clip, tile = 4.5, (4, 4)
    elif mean_b < 120:
        clip, tile = 3.5, (6, 6)
    else:
        clip, tile = 2.5, (8, 8)
    clahe_obj      = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    enhanced_green = clahe_obj.apply(green)

    # 3-channel vessel contrast fuse
    red_blurred    = cv2.GaussianBlur(red, (5, 5), 0)
    red_suppressed = cv2.subtract(enhanced_green, red_blurred)
    red_inverted   = cv2.normalize(255.0 - red.astype(np.float32),
                                   None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    enh_red_inv    = clahe_obj.apply(red_inverted)
    vessel_contrast = cv2.normalize(
        0.50 * enhanced_green.astype(np.float32)
        + 0.25 * red_suppressed.astype(np.float32)
        + 0.25 * enh_red_inv.astype(np.float32),
        None, 0, 255, cv2.NORM_MINMAX,
    ).astype(np.uint8)

    # LAB redness weighting
    lab        = rgb2lab(image_rgb)
    l_ch       = lab[:, :, 0]
    a_ch       = lab[:, :, 1]
    validity   = np.ones_like(l_ch, dtype=np.float32)
    validity[l_ch > 94] = 0
    validity[l_ch < 5]  = 0
    min_a      = np.percentile(a_ch, 2)
    max_a      = np.percentile(a_ch, 98)
    redness_p  = np.clip((a_ch - min_a) / (max_a - min_a + 1e-6), 0, 1)
    redness_p  = 1 / (1 + np.exp(-redness_steep * (redness_p - redness_center)))

    # Ridge filters
    f_res   = frangi(vessel_contrast,   sigmas=_Cfg.FRANGI_SIGMAS,      black_ridges=True)
    m_res   = meijering(vessel_contrast,sigmas=_Cfg.MEIJERING_SIGMAS,   black_ridges=True)
    s_res   = sato(vessel_contrast,     sigmas=_Cfg.SATO_SIGMAS,        black_ridges=True)
    f_green = frangi(enhanced_green,    sigmas=_Cfg.FRANGI_GREEN_SIGMAS, black_ridges=True)

    structure_main  = np.power(
        (1.1 * _normalize(f_res) + _normalize(m_res) + 0.9 * _normalize(s_res)) / 3.0,
        power_exp,
    )
    structure_green = np.power(_normalize(f_green), power_exp + 0.3)
    structure       = np.maximum(structure_main, 0.5 * structure_green)

    # Probability map
    prob_map = structure * redness_p * validity
    if prob_map.max() > 0:
        prob_map = cv2.normalize(prob_map, None, 0, 255, cv2.NORM_MINMAX)
    prob_map = prob_map.astype(np.uint8)
    prob_map = cv2.bitwise_and(prob_map, prob_map, mask=cornea_mask)
    prob_map = cv2.GaussianBlur(prob_map, (3, 3), 0)

    skeleton         = np.zeros_like(prob_map)
    structure_binary = np.zeros_like(prob_map)

    valid_px = prob_map[cornea_mask > 0]
    if len(valid_px) > 0 and valid_px.max() > 0:
        nz = valid_px[valid_px > 0]
        if len(nz) > 0:
            pct    = 92 - (17 * sens_t)
            thresh = max(float(np.percentile(nz, pct)), 8.0)
            print(f"    [Vessel thresh] p{pct:.0f} = {thresh:.1f}")
            structure_binary = (prob_map > thresh).astype(np.uint8) * 255
            structure_binary = cv2.bitwise_and(structure_binary, structure_binary, mask=cornea_mask)
            clean            = remove_small_objects(
                structure_binary.astype(bool), min_size=int(120 - 80 * sens_t)
            )
            structure_binary = clean.astype(np.uint8) * 255
            skeleton         = skeletonize(clean).astype(np.uint8) * 255
            if min_vessel_len > 0:
                n_skel, skel_lbl = cv2.connectedComponents(skeleton)
                for sid in range(1, n_skel):
                    if int(np.count_nonzero(skel_lbl == sid)) < min_vessel_len:
                        skeleton[skel_lbl == sid] = 0

    # Width map
    width_map = cv2.distanceTransform(structure_binary, cv2.DIST_L2, 5).astype(np.float32)

    # Per-vessel measurement
    vessel_analysis: Dict = {}
    affected_hours: List  = []
    n_lbl, lbl_map, stats, centroids = cv2.connectedComponentsWithStats(structure_binary)

    for lid in range(1, n_lbl):
        if stats[lid, cv2.CC_STAT_AREA] < _Cfg.MIN_VESSEL_AREA_PX:
            continue
        vmask = lbl_map == lid
        ys, xs = np.where(vmask)
        if len(xs) == 0:
            continue
        hours     = sorted(np.unique(_get_clock_hour(ys, xs, center)).tolist())
        color_rgb = _color_for_label(lid)
        skel_px   = (skeleton > 0) & vmask
        length_px = float(np.count_nonzero(skel_px))
        if np.count_nonzero(skel_px) > 0:
            widths        = width_map[skel_px]
            mean_w        = float(np.mean(widths)) * 2.0
            max_w         = float(np.max(widths))  * 2.0
        else:
            mean_w = max_w = 0.0
        pts  = np.column_stack((xs, ys))
        hull = cv2.convexHull(pts)
        chord = 1.0
        if hull is not None and len(hull) >= 2:
            for p1, p2 in itertools.combinations(hull[:, 0, :], 2):
                d = float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))
                if d > chord: chord = d
        tortuosity = max(round(length_px / chord, 2), 1.0)
        vessel_analysis[lid] = {
            "centroid":      [int(centroids[lid][0]), int(centroids[lid][1])],
            "sectors":       hours,
            "length_px":     round(length_px, 2),
            "mean_width_px": round(mean_w, 2),
            "max_width_px":  round(max_w, 2),
            "tortuosity":    tortuosity,
            "color_rgb":     list(color_rgb),
            "color_hex":     "#{:02X}{:02X}{:02X}".format(*color_rgb),
        }
        affected_hours.extend(hours)

    severity = float(
        np.count_nonzero(structure_binary) / (np.count_nonzero(cornea_mask) + 1e-6) * 100.0
    )

    debug_maps = {
        "frangi":    _normalize(f_res),
        "meijering": _normalize(m_res),
        "sato":      _normalize(s_res),
        "redness":   redness_p.astype(np.float32),
        "validity":  validity,
        "_enhanced_green":    enhanced_green,
        "_vessel_contrast":   vessel_contrast,
        "_structure_binary":  structure_binary,
    }

    return (structure_binary, skeleton, prob_map,
            vessel_analysis, sorted(set(affected_hours)), severity, debug_maps)


# ─────────────────────────────────────────────────────────────
# OVERLAY HELPERS
# ─────────────────────────────────────────────────────────────
def _colored_overlay(image_rgb, final_vessels, vessel_analysis):
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    n_lbl, lbl_map, stats, _ = cv2.connectedComponentsWithStats(final_vessels)
    for lid in range(1, n_lbl):
        if lid in vessel_analysis:
            r, g, b = vessel_analysis[lid]["color_rgb"]
            bgr[lbl_map == lid] = (b, g, r)
        else:
            bgr[lbl_map == lid] = (0, 230, 80)
    return bgr


def _green_overlay(image_rgb, mask):
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    bgr[mask > 0] = (0, 230, 80)
    return bgr


# ─────────────────────────────────────────────────────────────
# DEBUG STRIP HELPERS
# ─────────────────────────────────────────────────────────────
def _make_panel(img_bgr, lbl):
    h, w = img_bgr.shape[:2]
    tw   = _Cfg.THUMB_W
    th   = int(h * tw / w)
    t    = cv2.resize(img_bgr, (tw, th))
    ov   = t.copy()
    cv2.rectangle(ov, (0, 0), (tw, 38), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.55, t, 0.45, 0, t)
    cv2.putText(t, lbl, (8, 26), cv2.FONT_HERSHEY_SIMPLEX,
                _Cfg.PANEL_FONT_SCALE, _Cfg.PANEL_FONT_COLOR, 2, cv2.LINE_AA)
    return t


def _float_bgr(arr):
    u8 = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)


def _mask_bgr(mask):
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


def _hstack_equal(panels):
    max_h = max(p.shape[0] for p in panels)
    out   = []
    for p in panels:
        if p.shape[0] < max_h:
            pad = np.zeros((max_h - p.shape[0], p.shape[1], 3), np.uint8)
            p   = np.vstack([p, pad])
        out.append(p)
    return np.hstack(out)


def _build_debug_strip(orig_rgb, cornea_mask, structure_map,
                        heat_bgr, overlay_bgr, skeleton,
                        final_vessels, vessel_analysis, debug_maps):
    eg  = debug_maps["_enhanced_green"]
    vc  = debug_maps["_vessel_contrast"]
    row1 = _hstack_equal([
        _make_panel(cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR), "1. Original"),
        _make_panel(_mask_bgr(cornea_mask),                    "2. Cornea ROI"),
        _make_panel(cv2.merge([eg, eg, eg]),                   "3. CLAHE green ch"),
        _make_panel(cv2.cvtColor(vc, cv2.COLOR_GRAY2BGR),      "4. Vessel contrast"),
        _make_panel(_float_bgr(debug_maps["validity"]),        "5. Validity mask"),
    ])
    row2 = _hstack_equal([
        _make_panel(_float_bgr(debug_maps["frangi"]),    "6. Frangi"),
        _make_panel(_float_bgr(debug_maps["meijering"]), "7. Meijering"),
        _make_panel(_float_bgr(debug_maps["sato"]),      "8. Sato"),
        _make_panel(_float_bgr(debug_maps["redness"]),   "9. Redness (LAB a*)"),
        _make_panel(_mask_bgr(structure_map),            "10. Structure map"),
    ])
    row3 = _hstack_equal([
        _make_panel(heat_bgr,    "11. Heatmap (INFERNO)"),
        _make_panel(overlay_bgr, "12. Structure overlay [PRIMARY]"),
        _make_panel(cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR), "13. Original (ref)"),
        _make_panel(_mask_bgr(cornea_mask), "14. Cornea ROI (ref)"),
        _make_panel(_mask_bgr(skeleton),    "15. Skeleton (ref)"),
    ])
    row4 = _hstack_equal([
        _make_panel(_mask_bgr(skeleton),      "16. Skeleton (pruned)"),
        _make_panel(_mask_bgr(final_vessels), "17. Binary vessel mask"),
        _make_panel(_colored_overlay(orig_rgb, final_vessels, vessel_analysis), "18. Coloured overlay"),
        _make_panel(_green_overlay(orig_rgb, final_vessels),                    "19. Green overlay"),
    ])
    max_w = max(r.shape[1] for r in [row1, row2, row3, row4])
    def pw(r):
        if r.shape[1] < max_w:
            pad = np.zeros((r.shape[0], max_w - r.shape[1], 3), np.uint8)
            return np.hstack([r, pad])
        return r
    h1 = row1.shape[0]; h2 = h1 + row2.shape[0]; h3 = h2 + row3.shape[0]
    strip = np.vstack([pw(row1), pw(row2), pw(row3), pw(row4)])
    for hh in [h1, h2, h3]:
        cv2.line(strip, (0, hh), (max_w, hh), (80, 80, 80), 2)
    return strip


# ─────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────
def run(image_path: str, out_dir: str,
        cornea_mask: np.ndarray,
        sensitivity: int = 50,
        min_vessel_len: int = 15,
        gap_closing: int = 7) -> dict:
    """
    Run Module 3 vessel filtering on a single image.

    Parameters
    ----------
    cornea_mask : np.ndarray (H x W, uint8 0/255) from Module 2.
    sensitivity : 0–100, higher = more vessels, more noise.

    Returns
    -------
    dict with keys:
        final_vessels    : np.ndarray binary vessel mask
        skeleton         : np.ndarray skeleton
        structure_map    : np.ndarray probability map (uint8)
        vessel_analysis  : dict per-vessel metrics
        affected_hours   : list of clock hours
        severity         : float %
        paths            : dict of saved file paths
    """
    os.makedirs(out_dir, exist_ok=True)

    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Resize cornea mask to match image if needed
    if cornea_mask.shape[:2] != rgb.shape[:2]:
        cornea_mask = cv2.resize(cornea_mask, (rgb.shape[1], rgb.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)

    center = _cornea_center(cornea_mask)

    print(f"[Module 3] Running vessel analysis (sensitivity={sensitivity}) ...")
    (final_vessels, skeleton, structure_map,
     vessel_analysis, affected_hours, severity, debug_maps) = _analyze_vessels(
        rgb, cornea_mask, center,
        sensitivity=sensitivity,
        min_vessel_len=min_vessel_len,
        gap_closing=gap_closing,
    )

    print(f"    → {np.sum(final_vessels > 0):,} vessel px | "
          f"{len(vessel_analysis)} components | severity={severity:.2f}%")
    print(f"    → Clock hours: {affected_hours}")

    heat_bgr    = _structure_to_heatmap(structure_map)
    overlay_bgr = _make_structure_overlay(rgb, structure_map, cornea_mask)
    debug_strip = _build_debug_strip(
        rgb, cornea_mask, structure_map,
        heat_bgr, overlay_bgr, skeleton,
        final_vessels, vessel_analysis, debug_maps,
    )

    paths = {
        "structure_gray":    os.path.join(out_dir, "structure_gray.png"),
        "structure_heat":    os.path.join(out_dir, "structure_heat.png"),
        "structure_overlay": os.path.join(out_dir, "structure_overlay.png"),
        "skeleton":          os.path.join(out_dir, "skeleton.png"),
        "vessel_binary":     os.path.join(out_dir, "vessel_binary.png"),
        "vessel_colored":    os.path.join(out_dir, "vessel_colored.png"),
        "debug_strip":       os.path.join(out_dir, "debug_strip.png"),
    }

    cv2.imwrite(paths["structure_gray"],    structure_map)
    cv2.imwrite(paths["structure_heat"],    heat_bgr)
    cv2.imwrite(paths["structure_overlay"], overlay_bgr)
    cv2.imwrite(paths["skeleton"],          skeleton)
    cv2.imwrite(paths["vessel_binary"],     final_vessels)
    cv2.imwrite(paths["vessel_colored"],    _colored_overlay(rgb, final_vessels, vessel_analysis))
    cv2.imwrite(paths["debug_strip"],       debug_strip)

    print(f"[Module 3] Vessel filtering done → {out_dir}")

    return {
        "final_vessels":   final_vessels,
        "skeleton":        skeleton,
        "structure_map":   structure_map,
        "vessel_analysis": vessel_analysis,
        "affected_hours":  affected_hours,
        "severity":        severity,
        "center":          center,
        "paths":           paths,
    }
