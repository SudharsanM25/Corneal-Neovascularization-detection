"""
Module 4 — Spatial Tracking & Zone Analysis
=============================================
Divides cornea into 3 concentric rings × 12 clock sectors = 36 zones.
Detects and labels individual vessels, tracks their path across zones.
Overlays drawn on the original slit-lamp image.

Outputs saved to: <session_dir>/module4/
  - zone_map.png           — 36-zone colored grid
  - vessel_overlay.png     — vessels colored by zone on original image
  - tracking_overlay.png   — skeleton paths with zone-transition markers
  - zone_report.csv        — per-vessel zone traversal data
  - summary.json           — overall stats
"""

import os
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import cv2
from scipy import ndimage
from skimage import morphology
from matplotlib.colors import hsv_to_rgb


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
NUM_RINGS   = 3
NUM_SECTORS = 12
MIN_VESSEL_PX = 0


# ─────────────────────────────────────────────────────────────
# ZONE HELPERS
# ─────────────────────────────────────────────────────────────
def _cornea_geometry(mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("Cornea mask is empty")
    cx     = float(xs.mean())
    cy     = float(ys.mean())
    r_eff  = float(np.sqrt(mask.sum() / np.pi))
    return cx, cy, r_eff


def _build_zone_map(cornea_mask, cx, cy, r_eff,
                    n_rings=NUM_RINGS, n_sectors=NUM_SECTORS):
    H, W   = cornea_mask.shape
    yy, xx = np.mgrid[0:H, 0:W]
    dx     = (xx - cx).astype(float)
    dy     = (yy - cy).astype(float)
    dist   = np.sqrt(dx**2 + dy**2)
    angle  = np.degrees(np.arctan2(dx, -dy)) % 360   # 0=12 o'clock, clockwise

    ring_edges = np.linspace(0, r_eff, n_rings + 1)
    ring_map   = np.zeros_like(dist, dtype=np.int16)
    for r in range(1, n_rings + 1):
        ring_map[(dist >= ring_edges[r-1]) & (dist < ring_edges[r])] = r
    ring_map[(dist >= ring_edges[-1]) & cornea_mask] = n_rings

    sector_map = (angle / (360.0 / n_sectors)).astype(np.int16) + 1
    zone_map   = np.zeros(cornea_mask.shape, dtype=np.int16)
    inside     = cornea_mask & (ring_map > 0)
    zone_map[inside] = (ring_map[inside] - 1) * n_sectors + sector_map[inside]
    return zone_map, ring_map, sector_map


def _zone_label(zone_id, n_sectors=NUM_SECTORS):
    ring   = (zone_id - 1) // n_sectors + 1
    sector = (zone_id - 1) % n_sectors + 1
    return f"Ring{ring}-Sector{sector}"


def _ring_sector(zone_id, n_sectors=NUM_SECTORS):
    ring   = (zone_id - 1) // n_sectors + 1
    sector = (zone_id - 1) % n_sectors + 1
    return ring, sector


# ─────────────────────────────────────────────────────────────
# VESSEL ZONE TRACKING
# ─────────────────────────────────────────────────────────────
def _label_vessels(vessel_mask, min_size=MIN_VESSEL_PX):
    if min_size > 0:
        vessel_mask = morphology.remove_small_objects(vessel_mask, min_size=min_size)
    labeled, n = ndimage.label(vessel_mask)
    return labeled, n


def _vessel_zones(labeled_vessels, zone_map, vessel_id):
    vmask   = labeled_vessels == vessel_id
    px_len  = int(vmask.sum())
    skel    = morphology.skeletonize(vmask)
    sy, sx  = np.where(skel)
    if len(sy) == 0:
        sy, sx = np.where(vmask)
    cy_v = sy.mean(); cx_v = sx.mean()
    order = np.argsort(np.arctan2(sy - cy_v, sx - cx_v))
    sy = sy[order]; sx = sx[order]
    zones_path = []
    for y, x in zip(sy, sx):
        z = int(zone_map[y, x])
        if z > 0 and (not zones_path or zones_path[-1] != z):
            zones_path.append(z)
    all_zones = sorted(set(
        int(zone_map[y, x]) for y, x in zip(*np.where(vmask)) if zone_map[y, x] > 0
    ))
    return zones_path, all_zones, px_len, skel


def _px_to_mm(px, r_eff_px, cornea_diameter_mm=12.0):
    return round(px * (cornea_diameter_mm / 2.0) / r_eff_px, 3)


# ─────────────────────────────────────────────────────────────
# ZONE COLOR PALETTE
# ─────────────────────────────────────────────────────────────
def _make_zone_palette(n_rings=NUM_RINGS, n_sectors=NUM_SECTORS):
    total  = n_rings * n_sectors
    colors = []
    for i in range(total):
        h = i / total
        s = 0.55 + 0.2 * (i % n_rings) / max(n_rings - 1, 1)
        colors.append(hsv_to_rgb([h, s, 0.9]))
    return (np.array(colors) * 255).astype(np.uint8)

ZONE_PALETTE = _make_zone_palette()


# ─────────────────────────────────────────────────────────────
# DRAWING HELPERS
# ─────────────────────────────────────────────────────────────
def _draw_zone_grid(canvas, cornea_mask, cx, cy, r_eff,
                    n_rings=NUM_RINGS, n_sectors=NUM_SECTORS,
                    line_color=(255, 255, 255), label_color=(255, 255, 255)):
    overlay = canvas.copy()
    for k in range(1, n_rings):
        r = int(r_eff * k / n_rings)
        cv2.circle(overlay, (int(cx), int(cy)), r, line_color, 1, cv2.LINE_AA)
    for s in range(n_sectors):
        a_rad = np.radians(s * (360.0 / n_sectors) - 90)
        x2 = int(cx + r_eff * np.cos(a_rad))
        y2 = int(cy + r_eff * np.sin(a_rad))
        cv2.line(overlay, (int(cx), int(cy)), (x2, y2), line_color, 1, cv2.LINE_AA)
    cntr = cornea_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(cntr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 220, 255), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    H, W = canvas.shape[:2]
    for ring in range(1, n_rings + 1):
        r_mid = r_eff * (ring - 0.5) / n_rings
        for sec in range(1, n_sectors + 1):
            a_rad = np.radians((sec - 0.5) * (360.0 / n_sectors) - 90)
            tx = int(cx + r_mid * np.cos(a_rad))
            ty = int(cy + r_mid * np.sin(a_rad))
            if 0 <= tx < W and 0 <= ty < H:
                lbl = f"R{ring}S{sec}"
                cv2.putText(overlay, lbl, (tx - 12, ty + 4), font, 0.28,
                            (0, 0, 0),    2, cv2.LINE_AA)
                cv2.putText(overlay, lbl, (tx - 12, ty + 4), font, 0.28,
                            label_color,  1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 1.0, canvas, 0.0, 0, canvas)


def _make_zone_map_image(cornea_mask, zone_map, cx, cy, r_eff,
                          n_rings=NUM_RINGS, n_sectors=NUM_SECTORS):
    H, W  = cornea_mask.shape
    img   = np.zeros((H, W, 3), dtype=np.uint8)
    total = n_rings * n_sectors
    for z_id in range(1, total + 1):
        px = zone_map == z_id
        if np.any(px):
            color = ZONE_PALETTE[(z_id - 1) % len(ZONE_PALETTE)]
            img[px] = color
    _draw_zone_grid(img, cornea_mask, cx, cy, r_eff, n_rings, n_sectors)
    return img


def _make_vessel_overlay(orig_bgr, cornea_mask, zone_map,
                          labeled_vessels, vessel_meta,
                          cx, cy, r_eff, n_rings, n_sectors):
    canvas = orig_bgr.copy() if orig_bgr is not None else \
             np.zeros((*cornea_mask.shape, 3), dtype=np.uint8)
    n_vessels = len(vessel_meta)
    for v_id, meta in vessel_meta.items():
        all_zones = meta["all_zones"]
        vmask     = labeled_vessels == v_id
        for z in all_zones:
            z_px   = (zone_map == z) & vmask
            c_idx  = (z - 1) % len(ZONE_PALETTE)
            color  = (int(ZONE_PALETTE[c_idx][2]),
                      int(ZONE_PALETTE[c_idx][1]),
                      int(ZONE_PALETTE[c_idx][0]))
            canvas[z_px] = color
    _draw_zone_grid(canvas, cornea_mask, cx, cy, r_eff, n_rings, n_sectors)
    return canvas


def _make_tracking_overlay(orig_bgr, cornea_mask, labeled_vessels,
                             vessel_meta, skeletons, zone_map,
                             cx, cy, r_eff, n_rings, n_sectors):
    canvas = orig_bgr.copy() if orig_bgr is not None else \
             np.zeros((*cornea_mask.shape, 3), dtype=np.uint8)
    _draw_zone_grid(canvas, cornea_mask, cx, cy, r_eff, n_rings, n_sectors)
    font = cv2.FONT_HERSHEY_SIMPLEX
    H, W = canvas.shape[:2]
    for v_id, meta in vessel_meta.items():
        c_idx   = (v_id - 1) % len(ZONE_PALETTE)
        col     = (int(ZONE_PALETTE[c_idx][2]),
                   int(ZONE_PALETTE[c_idx][1]),
                   int(ZONE_PALETTE[c_idx][0]))
        skel    = skeletons[v_id]
        skel_ys, skel_xs = np.where(skel)
        if len(skel_ys) == 0:
            continue
        skel_pts = list(zip(skel_ys.tolist(), skel_xs.tolist()))
        for y, x in skel_pts:
            canvas[y, x] = col
        prev_z = None
        for y, x in skel_pts:
            z = int(zone_map[y, x]) if (0 <= y < H and 0 <= x < W) else 0
            if z > 0 and z != prev_z:
                r_n, s_n = _ring_sector(z, n_sectors)
                cv2.circle(canvas, (x, y), 5, (255, 255, 255), 1, cv2.LINE_AA)
                lbl = f"R{r_n}S{s_n}"
                cv2.putText(canvas, lbl, (x + 6, y - 4), font, 0.30,
                            (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(canvas, lbl, (x + 6, y - 4), font, 0.30,
                            col, 1, cv2.LINE_AA)
                prev_z = z
        if skel_pts:
            sy, sx = skel_pts[0]
            cv2.putText(canvas, f"V{v_id}", (sx - 4, sy - 10),
                        font, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(canvas, f"V{v_id}", (sx - 4, sy - 10),
                        font, 0.55, col, 1, cv2.LINE_AA)
    return canvas


# ─────────────────────────────────────────────────────────────
# ZONE SEVERITY HEATMAP
# ─────────────────────────────────────────────────────────────
def _make_zone_severity_map(cornea_mask, zone_map, vessel_mask,
                             cx, cy, r_eff, n_rings=NUM_RINGS, n_sectors=NUM_SECTORS):
    """Color each zone by its vessel density (blue=clean → red=dense)."""
    H, W     = cornea_mask.shape
    heat_img = np.zeros((H, W, 3), dtype=np.uint8)
    total    = n_rings * n_sectors
    for z_id in range(1, total + 1):
        zone_px    = (zone_map == z_id)
        zone_area  = np.sum(zone_px)
        if zone_area == 0:
            continue
        vessel_px  = np.sum(zone_px & (vessel_mask > 0))
        density    = vessel_px / zone_area  # 0–1
        # blue → yellow → red (cold-to-hot)
        hue        = int(120 - density * 120)  # 120=green/blue, 0=red
        hsv_color  = np.uint8([[[hue, 220, 220]]])
        bgr        = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0, 0]
        heat_img[zone_px] = bgr
    _draw_zone_grid(heat_img, cornea_mask, cx, cy, r_eff, n_rings, n_sectors,
                    line_color=(255, 255, 255), label_color=(255, 255, 255))
    return heat_img


# ─────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────
def run(image_path: str, out_dir: str,
        cornea_mask: np.ndarray,
        vessel_mask: np.ndarray,
        n_rings: int = NUM_RINGS,
        n_sectors: int = NUM_SECTORS) -> dict:
    """
    Run Module 4 spatial tracking on a single image.

    Parameters
    ----------
    cornea_mask : np.ndarray (H x W, uint8 0/255) from Module 2.
    vessel_mask : np.ndarray (H x W, uint8 0/255) from Module 3.

    Returns
    -------
    dict with keys:
        vessel_meta  : per-vessel zone info
        zone_stats   : per-zone vessel counts
        paths        : saved file paths
        summary      : overall stats dict
    """
    os.makedirs(out_dir, exist_ok=True)

    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # Align masks to image size
    H, W = bgr.shape[:2]
    cornea_mask = cv2.resize(cornea_mask, (W, H), interpolation=cv2.INTER_NEAREST)
    vessel_mask = cv2.resize(vessel_mask, (W, H), interpolation=cv2.INTER_NEAREST)

    # Constrain vessels to cornea
    vessel_mask = vessel_mask & cornea_mask

    cx, cy, r_eff = _cornea_geometry(cornea_mask.astype(bool))
    print(f"[Module 4] centroid=({cx:.1f},{cy:.1f})  r_eff={r_eff:.1f}px")

    zone_map, ring_map, sector_map = _build_zone_map(
        cornea_mask.astype(bool), cx, cy, r_eff, n_rings, n_sectors)

    labeled_vessels, n_vessels = _label_vessels(vessel_mask.astype(bool))
    print(f"[Module 4] Vessels: {n_vessels}")

    vessel_meta = {}
    skeletons   = {}
    rows        = []

    for v_id in range(1, n_vessels + 1):
        zones_path, all_zones, px_len, skel = _vessel_zones(
            labeled_vessels, zone_map, v_id)
        mm_len = _px_to_mm(px_len, r_eff)
        vessel_meta[v_id] = {
            "vessel_id":       v_id,
            "px_length":       px_len,
            "mm_length":       mm_len,
            "zones_path":      zones_path,
            "all_zones":       all_zones,
            "start_zone":      zones_path[0]  if zones_path else None,
            "end_zone":        zones_path[-1] if zones_path else None,
            "n_zones_crossed": len(set(zones_path)),
        }
        skeletons[v_id] = skel
        path_labels = [_zone_label(z, n_sectors) for z in zones_path]
        rows.append({
            "vessel_id":         v_id,
            "px_length":         px_len,
            "mm_length_approx":  mm_len,
            "start_zone":        _zone_label(zones_path[0],  n_sectors) if zones_path else "",
            "end_zone":          _zone_label(zones_path[-1], n_sectors) if zones_path else "",
            "zones_traversed":   " -> ".join(path_labels),
            "unique_zones":      len(set(zones_path)),
            "all_unique_zones":  ", ".join(_zone_label(z, n_sectors) for z in all_zones),
        })

    # Per-zone stats
    total_zones = n_rings * n_sectors
    zone_stats  = {}
    for z_id in range(1, total_zones + 1):
        zone_px   = zone_map == z_id
        zone_area = int(np.sum(zone_px))
        ves_px    = int(np.sum(zone_px & (vessel_mask > 0)))
        density   = round(ves_px / zone_area * 100, 2) if zone_area > 0 else 0.0
        r_n, s_n  = _ring_sector(z_id, n_sectors)
        zone_stats[_zone_label(z_id, n_sectors)] = {
            "ring": r_n, "sector": s_n,
            "zone_area_px": zone_area,
            "vessel_px": ves_px,
            "density_pct": density,
        }

    # Build images
    zone_img = _make_zone_map_image(
        cornea_mask.astype(bool), zone_map, cx, cy, r_eff, n_rings, n_sectors)
    ves_img  = _make_vessel_overlay(
        bgr, cornea_mask.astype(bool), zone_map,
        labeled_vessels, vessel_meta, cx, cy, r_eff, n_rings, n_sectors)
    trk_img  = _make_tracking_overlay(
        bgr, cornea_mask.astype(bool), labeled_vessels,
        vessel_meta, skeletons, zone_map, cx, cy, r_eff, n_rings, n_sectors)
    sev_img  = _make_zone_severity_map(
        cornea_mask.astype(bool), zone_map, vessel_mask,
        cx, cy, r_eff, n_rings, n_sectors)

    # Save images
    paths = {
        "zone_map":         os.path.join(out_dir, "zone_map.png"),
        "vessel_overlay":   os.path.join(out_dir, "vessel_overlay.png"),
        "tracking_overlay": os.path.join(out_dir, "tracking_overlay.png"),
        "severity_heatmap": os.path.join(out_dir, "severity_heatmap.png"),
        "zone_report_csv":  os.path.join(out_dir, "zone_report.csv"),
        "summary_json":     os.path.join(out_dir, "summary.json"),
    }
    cv2.imwrite(paths["zone_map"],         zone_img)
    cv2.imwrite(paths["vessel_overlay"],   ves_img)
    cv2.imwrite(paths["tracking_overlay"], trk_img)
    cv2.imwrite(paths["severity_heatmap"], sev_img)

    # Save CSV
    if rows:
        with open(paths["zone_report_csv"], "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    # Summary
    summary = {
        "n_vessels":           n_vessels,
        "total_vessel_px":     int(np.sum(vessel_mask > 0)),
        "cornea_area_px":      int(np.sum(cornea_mask > 0)),
        "vessel_coverage_pct": round(np.sum(vessel_mask > 0) / (np.sum(cornea_mask > 0) + 1e-6) * 100, 2),
        "r_eff_px":            round(r_eff, 2),
        "cornea_center":       [round(cx, 1), round(cy, 1)],
        "n_rings":             n_rings,
        "n_sectors":           n_sectors,
        "zone_stats":          zone_stats,
        "vessel_meta":         {str(k): v for k, v in vessel_meta.items()},
    }
    with open(paths["summary_json"], "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[Module 4] Spatial tracking done → {out_dir}")

    return {
        "vessel_meta": vessel_meta,
        "zone_stats":  zone_stats,
        "summary":     summary,
        "paths":       paths,
    }
