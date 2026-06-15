"""
ground_truth.py  –  Ground-truth schema, loaders, and (optional) fluorescence
                    -derived scar labels.

WHY THIS EXISTS
───────────────
BS-Detector had no quantitative notion of "correct".  This module defines a
single, source-agnostic ground-truth (GT) schema so that labels coming from
different places all feed the same evaluation harness (`evaluation.py`):

  • Manual annotation by lab mates   (scar position, pole identity, lineage)
  • Carmen's existing length data     (length-only validation)
  • Septin fluorescence              (automatic scar position; see
                                      `septin_scar_position` below)

Every GT field except the cell centroid is OPTIONAL, so a partial annotation
(e.g. "length only" or "scar position only") is valid.  The evaluation harness
silently skips metrics it has no labels for.

COORDINATE CONVENTION
─────────────────────
All coordinates are (y, x) = (row, col) in pixels, matching skimage
`regionprops` centroids and `find_contours` output used throughout the
pipeline.  Keep annotation tools in the same convention.

GROUND-TRUTH CSV SCHEMA
───────────────────────
One row per annotated cell-in-a-frame.  Header (blank cell = "unknown"):

    frame,gt_id,y,x,scar_present,scar_y,scar_x,
    new_pole_y,new_pole_x,old_pole_y,old_pole_x,length,notes

  frame        int    – 0-based frame index                       (required)
  gt_id        str    – annotator's stable name for the cell      (optional,
                        used for lineage/identity metrics)
  y, x         float  – cell centroid (row, col)                  (required;
                        used to match GT cells to detections)
  scar_present 0/1    – whether a birth scar is visible           (optional)
  scar_y,x     float  – birth-scar midpoint                       (optional)
  new_pole_y,x float  – new-pole coordinate                       (optional)
  old_pole_y,x float  – old-pole coordinate                       (optional)
  length       float  – pole-to-pole length [px]                  (optional)
  notes        str    – free text                                 (optional)
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, asdict, field
from typing import Optional

import numpy as np


# ── Schema ────────────────────────────────────────────────────────────────────

@dataclass
class GTRecord:
    """One ground-truth annotation for a single cell in a single frame."""
    frame: int
    y: float
    x: float
    gt_id: Optional[str] = None
    scar_present: Optional[bool] = None
    scar_y: Optional[float] = None
    scar_x: Optional[float] = None
    new_pole_y: Optional[float] = None
    new_pole_x: Optional[float] = None
    old_pole_y: Optional[float] = None
    old_pole_x: Optional[float] = None
    length: Optional[float] = None
    notes: Optional[str] = None

    # ── Convenience accessors (return np.array or None) ──────────────────────
    @property
    def centroid(self) -> np.ndarray:
        return np.array([self.y, self.x], dtype=float)

    @property
    def scar_midpoint(self) -> Optional[np.ndarray]:
        if self.scar_y is None or self.scar_x is None:
            return None
        return np.array([self.scar_y, self.scar_x], dtype=float)

    @property
    def new_pole(self) -> Optional[np.ndarray]:
        if self.new_pole_y is None or self.new_pole_x is None:
            return None
        return np.array([self.new_pole_y, self.new_pole_x], dtype=float)

    @property
    def old_pole(self) -> Optional[np.ndarray]:
        if self.old_pole_y is None or self.old_pole_x is None:
            return None
        return np.array([self.old_pole_y, self.old_pole_x], dtype=float)

    @property
    def has_poles(self) -> bool:
        return self.new_pole is not None and self.old_pole is not None


GT_COLUMNS = [
    'frame', 'gt_id', 'y', 'x', 'scar_present', 'scar_y', 'scar_x',
    'new_pole_y', 'new_pole_x', 'old_pole_y', 'old_pole_x', 'length', 'notes',
]


# ── Parsing helpers ───────────────────────────────────────────────────────────

def _as_float(v) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if s == '' or s.lower() in ('na', 'nan', 'none', 'null'):
        return None
    return float(s)


def _as_int(v) -> Optional[int]:
    f = _as_float(v)
    return int(f) if f is not None else None


def _as_bool(v) -> Optional[bool]:
    if v is None:
        return None
    s = str(v).strip().lower()
    if s == '' or s in ('na', 'nan', 'none', 'null', 'unknown'):
        return None
    if s in ('1', 'true', 't', 'yes', 'y'):
        return True
    if s in ('0', 'false', 'f', 'no', 'n'):
        return False
    raise ValueError(f"Cannot parse boolean from {v!r}")


def _as_str(v) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s if s != '' else None


def _record_from_row(row: dict) -> GTRecord:
    frame = _as_int(row.get('frame'))
    y     = _as_float(row.get('y'))
    x     = _as_float(row.get('x'))
    if frame is None or y is None or x is None:
        raise ValueError(f"GT row missing required frame/y/x: {row!r}")
    return GTRecord(
        frame=frame, y=y, x=x,
        gt_id=_as_str(row.get('gt_id')),
        scar_present=_as_bool(row.get('scar_present')),
        scar_y=_as_float(row.get('scar_y')),
        scar_x=_as_float(row.get('scar_x')),
        new_pole_y=_as_float(row.get('new_pole_y')),
        new_pole_x=_as_float(row.get('new_pole_x')),
        old_pole_y=_as_float(row.get('old_pole_y')),
        old_pole_x=_as_float(row.get('old_pole_x')),
        length=_as_float(row.get('length')),
        notes=_as_str(row.get('notes')),
    )


# ── Public loaders / writers ──────────────────────────────────────────────────

def load_ground_truth(path: str) -> dict[int, list[GTRecord]]:
    """
    Load ground-truth annotations from a CSV or JSON file.

    Returns
    -------
    dict  frame_index -> list[GTRecord]
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Ground-truth file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    records: list[GTRecord] = []

    if ext == '.json':
        with open(path) as f:
            data = json.load(f)
        rows = data['records'] if isinstance(data, dict) else data
        records = [_record_from_row(r) for r in rows]
    else:  # treat everything else as CSV
        with open(path, newline='') as f:
            for row in csv.DictReader(f):
                records.append(_record_from_row(row))

    by_frame: dict[int, list[GTRecord]] = {}
    for r in records:
        by_frame.setdefault(r.frame, []).append(r)
    return by_frame


def write_ground_truth(records, path: str) -> str:
    """Write a list of GTRecord (or dicts) to CSV. Useful for annotation tools."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=GT_COLUMNS, extrasaction='ignore')
        writer.writeheader()
        for r in records:
            row = asdict(r) if isinstance(r, GTRecord) else dict(r)
            if isinstance(row.get('scar_present'), bool):
                row['scar_present'] = int(row['scar_present'])
            writer.writerow(row)
    return path


def write_template(path: str, all_results=None) -> str:
    """
    Write a blank/pre-filled annotation template.

    If *all_results* (output of run_pipeline) is given, one row per detected
    cell is pre-seeded with frame + centroid so an annotator only has to fill
    in the scar/pole/length columns rather than locating cells from scratch.
    """
    records: list[GTRecord] = []
    if all_results is not None:
        for fd in all_results:
            fidx = fd['frame_idx']
            for r in fd['cells']:
                cy, cx = r['centroid']
                records.append(GTRecord(
                    frame=fidx, y=float(cy), x=float(cx),
                    gt_id=r.get('cell_name'),
                ))
    if not records:
        # Emit a header-only template with one example comment row.
        records = [GTRecord(frame=0, y=0.0, x=0.0, notes='example – delete me')]
    return write_ground_truth(records, path)


# ── Fluorescence-derived ground truth (septin channel) ────────────────────────

def septin_scar_position(fluor_frame, center, axis, long_norm, smooth_pts,
                         band_halfwidth=0.04):
    """
    Estimate the birth-scar midpoint from a septin (or other division-site)
    fluorescence channel, for ONE cell.

    Septins localise to the division septum, which is exactly where the birth
    scar forms.  We project fluorescence intensity onto the cell long axis and
    take the brightest transverse band as the septum position; the scar
    midpoint is the point on the cell centreline at that longitudinal position.

    This is the automatic, human-free ground-truth path described in the lab
    notes ("septin at points of change in curvature", "get me the data in both
    brightfield and fluorescence").  It requires brightfield and fluorescence
    to be registered to the same pixel grid (same cell, same coordinates).

    Parameters
    ----------
    fluor_frame  : 2-D fluorescence image (same shape/registration as BF)
    center, axis : cell centre and unit long axis (from compute_pca_axis)
    long_norm    : (N,) normalized longitudinal position of each smooth_pt [0,1]
    smooth_pts   : (N, 2) smoothed contour points (row, col)
    band_halfwidth : half-width (in normalized length) of the transverse band
                     summed at each longitudinal position

    Returns
    -------
    (scar_midpoint_yx, confidence)  or  (None, 0.0) if no clear septum band.
        confidence = peak band intensity / median band intensity.
    """
    fluor = np.asarray(fluor_frame)
    center = np.asarray(center, dtype=float)
    axis   = np.asarray(axis, dtype=float)
    pts    = np.asarray(smooth_pts, dtype=float)

    # Sample longitudinal positions across the cell body.
    sample_norms = np.linspace(0.1, 0.9, 33)
    band_intensity = []
    band_points = []
    for sn in sample_norms:
        mask = np.abs(long_norm - sn) < band_halfwidth
        if mask.sum() < 2:
            band_intensity.append(0.0)
            band_points.append(None)
            continue
        band_pts = pts[mask]
        mid = band_pts.mean(axis=0)            # centreline point at this position
        rr = np.clip(int(round(mid[0])), 0, fluor.shape[0] - 1)
        cc = np.clip(int(round(mid[1])), 0, fluor.shape[1] - 1)
        band_intensity.append(float(fluor[rr, cc]))
        band_points.append(mid)

    band_intensity = np.asarray(band_intensity)
    if not np.any(band_intensity > 0):
        return None, 0.0

    best = int(np.argmax(band_intensity))
    if band_points[best] is None:
        return None, 0.0

    # Confidence = peak band intensity relative to the average signal along the
    # whole cell.  A sharp septin ring gives a large ratio; diffuse cytoplasmic
    # signal (no clear septum) gives a ratio near 1.
    background = float(np.mean(band_intensity))
    confidence = band_intensity[best] / (background + 1e-9)
    return band_points[best], float(confidence)
