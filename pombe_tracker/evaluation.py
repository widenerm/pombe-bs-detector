"""
evaluation.py  –  Quantitative accuracy metrics for BS-Detector.

WHAT IT DOES
────────────
Given detector output and ground-truth annotations (see `ground_truth.py`),
this module:

  1. Matches detected cells to GT cells per frame (Hungarian on centroid).
  2. Computes accuracy metrics:
       • Cell matching      – precision/recall of cell detection
       • Scar detection     – precision/recall/F1 of "is there a scar"
       • Scar localization  – px error between detected and GT scar midpoint
       • Pole assignment    – fraction with correct new/old pole labelling
       • Length agreement   – Bland-Altman bias / limits / MAE / Pearson r
  3. Rolls everything into ONE scalar `objective` (higher = better) so a
     hyperparameter search (Optuna etc.) can optimise directly against it.

DESIGN NOTE
───────────
`evaluate()` accepts EITHER the live `all_results` from `run_pipeline` OR a
detector-export produced by `export_eval_json()`.  The CSV written by
`io_utils.export_csv` does NOT contain centroids or scar/pole coordinates, so
it cannot be used for matching — hence the dedicated JSON export, which also
lets you evaluate offline / in CI without re-running Cellpose.

All coordinates are (y, x) = (row, col), matching the rest of the pipeline.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from .ground_truth import GTRecord, load_ground_truth


# ── Tunable evaluation parameters ─────────────────────────────────────────────

@dataclass
class EvalConfig:
    # Max centroid distance (px) for a detection<->GT match.
    match_max_distance: float = 40.0
    # A detected scar counts as correctly localized if within this distance (px)
    # of the GT scar midpoint.  Used for the "localized F1" metric.
    localization_tolerance: float = 10.0
    # Objective weights (renormalized over whichever components have data).
    w_scar_f1: float = 0.50
    w_localization: float = 0.25
    w_pole: float = 0.15
    w_length: float = 0.10
    # Localization error is mapped to a [0,1] score via exp(-err/scale).
    localization_score_scale: float = 10.0
    # Length error (MAE) is mapped to a [0,1] score via exp(-mae/scale).
    length_score_scale: float = 5.0


# ── Detector normalization ────────────────────────────────────────────────────

@dataclass
class DetCell:
    frame: int
    name: Optional[str]
    centroid: np.ndarray
    scar_detected: bool
    scar_midpoint: Optional[np.ndarray]
    new_pole: Optional[np.ndarray]
    old_pole: Optional[np.ndarray]
    length: Optional[float]
    seg_quality: Optional[str]


def _arr(v):
    return None if v is None else np.asarray(v, dtype=float)


def _normalize_detector(detector) -> dict[int, list[DetCell]]:
    """
    Accept live `all_results` (list of frame dicts with 'cells') OR a list of
    flat detector-export dicts (from export_eval_json).  Return frame -> cells.
    """
    by_frame: dict[int, list[DetCell]] = {}

    # Live all_results: items are dicts with a 'cells' key.
    if detector and isinstance(detector[0], dict) and 'cells' in detector[0]:
        for fd in detector:
            fidx = fd['frame_idx']
            for r in fd['cells']:
                dbg = r.get('debug_info', {})
                by_frame.setdefault(fidx, []).append(DetCell(
                    frame=fidx,
                    name=r.get('cell_name'),
                    centroid=_arr(r['centroid']),
                    scar_detected=bool(r.get('scar_detected')),
                    scar_midpoint=_arr(r.get('scar_midpoint')),
                    new_pole=_arr(dbg.get('new_pole_point')),
                    old_pole=_arr(dbg.get('old_pole_point')),
                    length=r.get('length'),
                    seg_quality=r.get('seg_quality'),
                ))
        return by_frame

    # Flat export: list of dicts already in DetCell shape.
    for r in detector:
        fidx = int(r['frame'])
        by_frame.setdefault(fidx, []).append(DetCell(
            frame=fidx,
            name=r.get('name'),
            centroid=_arr(r['centroid']),
            scar_detected=bool(r.get('scar_detected')),
            scar_midpoint=_arr(r.get('scar_midpoint')),
            new_pole=_arr(r.get('new_pole')),
            old_pole=_arr(r.get('old_pole')),
            length=r.get('length'),
            seg_quality=r.get('seg_quality'),
        ))
    return by_frame


def export_eval_json(all_results, path: str) -> str:
    """
    Serialize the detector fields needed for evaluation to JSON, so accuracy
    can be scored offline / in CI without re-running the pipeline.
    """
    out = []
    for fd in all_results:
        fidx = fd['frame_idx']
        for r in fd['cells']:
            dbg = r.get('debug_info', {})

            def lst(v):
                return None if v is None else np.asarray(v).tolist()

            out.append({
                'frame':         fidx,
                'name':          r.get('cell_name'),
                'centroid':      lst(r['centroid']),
                'scar_detected': bool(r.get('scar_detected')),
                'scar_midpoint': lst(r.get('scar_midpoint')),
                'new_pole':      lst(dbg.get('new_pole_point')),
                'old_pole':      lst(dbg.get('old_pole_point')),
                'length':        (float(r['length']) if r.get('length') is not None else None),
                'seg_quality':   r.get('seg_quality'),
            })

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    return path


# ── Matching ──────────────────────────────────────────────────────────────────

def match_frame(det_cells: list[DetCell], gt_records: list[GTRecord],
                max_dist: float):
    """
    Hungarian match detections to GT records by centroid distance, rejecting
    pairs farther apart than *max_dist*.

    Returns (pairs, unmatched_det, unmatched_gt) where pairs is a list of
    (DetCell, GTRecord, distance).
    """
    if not det_cells or not gt_records:
        return [], list(det_cells), list(gt_records)

    n_d, n_g = len(det_cells), len(gt_records)
    INF = 1e9
    cost = np.full((n_d, n_g), INF)
    for i, d in enumerate(det_cells):
        for j, g in enumerate(gt_records):
            dist = float(np.linalg.norm(d.centroid - g.centroid))
            if dist <= max_dist:
                cost[i, j] = dist

    row_ind, col_ind = linear_sum_assignment(cost)

    pairs = []
    matched_d, matched_g = set(), set()
    for i, j in zip(row_ind, col_ind):
        if cost[i, j] < INF:
            pairs.append((det_cells[i], gt_records[j], float(cost[i, j])))
            matched_d.add(i)
            matched_g.add(j)

    unmatched_det = [d for i, d in enumerate(det_cells) if i not in matched_d]
    unmatched_gt  = [g for j, g in enumerate(gt_records) if j not in matched_g]
    return pairs, unmatched_det, unmatched_gt


# ── Report ────────────────────────────────────────────────────────────────────

@dataclass
class EvalReport:
    # cell matching
    n_matched: int = 0
    n_unmatched_det: int = 0
    n_unmatched_gt: int = 0
    cell_precision: float = 0.0
    cell_recall: float = 0.0

    # scar detection
    scar_tp: int = 0
    scar_fp: int = 0
    scar_fn: int = 0
    scar_tn: int = 0
    scar_precision: float = 0.0
    scar_recall: float = 0.0
    scar_f1: float = 0.0
    scar_evaluable: int = 0

    # scar localization (px), among detection-TPs with GT positions
    loc_n: int = 0
    loc_mean: Optional[float] = None
    loc_median: Optional[float] = None
    loc_p90: Optional[float] = None
    loc_within_tol_frac: Optional[float] = None

    # pole assignment
    pole_n: int = 0
    pole_accuracy: Optional[float] = None

    # length agreement (detector - GT)
    len_n: int = 0
    len_bias: Optional[float] = None
    len_mae: Optional[float] = None
    len_std: Optional[float] = None
    len_loa_low: Optional[float] = None
    len_loa_high: Optional[float] = None
    len_pearson: Optional[float] = None

    # single scalar to optimize (higher = better)
    objective: float = 0.0
    objective_components: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

    def __str__(self):
        return format_report(self)


# ── Core evaluation ───────────────────────────────────────────────────────────

def evaluate(detector, ground_truth, eval_config: Optional[EvalConfig] = None,
             count_unmatched_det_scars_as_fp: bool = False) -> EvalReport:
    """
    Score detector output against ground truth.

    Parameters
    ----------
    detector     : live `all_results`, OR a list from export_eval_json, OR a
                   path to an eval JSON file.
    ground_truth : dict {frame: [GTRecord]}, OR a path to a GT CSV/JSON.
    eval_config  : EvalConfig (defaults used if None).
    count_unmatched_det_scars_as_fp : if True, a detected scar on a cell with
                   no matching GT cell counts as a false positive.  Default
                   False keeps cell-matching errors out of the scar metric.
    """
    cfg = eval_config or EvalConfig()

    if isinstance(detector, str):
        with open(detector) as f:
            detector = json.load(f)
    if isinstance(ground_truth, str):
        ground_truth = load_ground_truth(ground_truth)

    det_by_frame = _normalize_detector(detector)

    rep = EvalReport()
    loc_errors = []
    loc_within = 0
    pole_correct = 0
    pole_total = 0
    len_det, len_gt = [], []

    all_frames = sorted(set(det_by_frame) | set(ground_truth))
    for fidx in all_frames:
        det_cells = det_by_frame.get(fidx, [])
        gt_cells  = ground_truth.get(fidx, [])
        pairs, unmatched_det, unmatched_gt = match_frame(
            det_cells, gt_cells, cfg.match_max_distance)

        rep.n_matched       += len(pairs)
        rep.n_unmatched_det += len(unmatched_det)
        rep.n_unmatched_gt  += len(unmatched_gt)

        # Unmatched detected scars optionally pollute the scar FP count.
        if count_unmatched_det_scars_as_fp:
            rep.scar_fp += sum(1 for d in unmatched_det if d.scar_detected)

        for det, gt, _dist in pairs:
            # ── scar detection ────────────────────────────────────────────
            if gt.scar_present is not None:
                rep.scar_evaluable += 1
                if gt.scar_present and det.scar_detected:
                    rep.scar_tp += 1
                elif gt.scar_present and not det.scar_detected:
                    rep.scar_fn += 1
                elif (not gt.scar_present) and det.scar_detected:
                    rep.scar_fp += 1
                else:
                    rep.scar_tn += 1

            # ── scar localization (only true positives with positions) ────
            if (det.scar_detected and det.scar_midpoint is not None
                    and gt.scar_midpoint is not None
                    and (gt.scar_present is None or gt.scar_present)):
                err = float(np.linalg.norm(det.scar_midpoint - gt.scar_midpoint))
                loc_errors.append(err)
                if err <= cfg.localization_tolerance:
                    loc_within += 1

            # ── pole assignment ───────────────────────────────────────────
            if (gt.has_poles and det.new_pole is not None
                    and det.old_pole is not None):
                pole_total += 1
                agree = (np.linalg.norm(det.new_pole - gt.new_pole)
                         + np.linalg.norm(det.old_pole - gt.old_pole))
                swap  = (np.linalg.norm(det.new_pole - gt.old_pole)
                         + np.linalg.norm(det.old_pole - gt.new_pole))
                if agree <= swap:
                    pole_correct += 1

            # ── length ────────────────────────────────────────────────────
            if gt.length is not None and det.length is not None:
                len_det.append(float(det.length))
                len_gt.append(float(gt.length))

    # ── Roll up cell matching ─────────────────────────────────────────────────
    rep.cell_precision = _safe_div(rep.n_matched, rep.n_matched + rep.n_unmatched_det)
    rep.cell_recall    = _safe_div(rep.n_matched, rep.n_matched + rep.n_unmatched_gt)

    # ── Roll up scar detection ────────────────────────────────────────────────
    rep.scar_precision = _safe_div(rep.scar_tp, rep.scar_tp + rep.scar_fp)
    rep.scar_recall    = _safe_div(rep.scar_tp, rep.scar_tp + rep.scar_fn)
    rep.scar_f1        = _safe_div(2 * rep.scar_precision * rep.scar_recall,
                                   rep.scar_precision + rep.scar_recall)

    # ── Roll up localization ──────────────────────────────────────────────────
    if loc_errors:
        arr = np.asarray(loc_errors)
        rep.loc_n               = len(arr)
        rep.loc_mean            = float(arr.mean())
        rep.loc_median          = float(np.median(arr))
        rep.loc_p90             = float(np.percentile(arr, 90))
        rep.loc_within_tol_frac = loc_within / len(arr)

    # ── Roll up pole ──────────────────────────────────────────────────────────
    if pole_total:
        rep.pole_n        = pole_total
        rep.pole_accuracy = pole_correct / pole_total

    # ── Roll up length (Bland-Altman) ─────────────────────────────────────────
    if len_det:
        d = np.asarray(len_det)
        g = np.asarray(len_gt)
        diff = d - g
        rep.len_n        = len(d)
        rep.len_bias     = float(diff.mean())
        rep.len_mae      = float(np.abs(diff).mean())
        rep.len_std      = float(diff.std(ddof=1)) if len(d) > 1 else 0.0
        rep.len_loa_low  = rep.len_bias - 1.96 * rep.len_std
        rep.len_loa_high = rep.len_bias + 1.96 * rep.len_std
        if len(d) > 1 and d.std() > 0 and g.std() > 0:
            rep.len_pearson = float(np.corrcoef(d, g)[0, 1])

    # ── Objective ─────────────────────────────────────────────────────────────
    rep.objective, rep.objective_components = _objective(rep, cfg)
    return rep


def _objective(rep: EvalReport, cfg: EvalConfig):
    """Weighted scalar in [0,1]; weights renormalized over present components."""
    comps, weights = {}, {}

    if rep.scar_evaluable > 0:
        comps['scar_f1'] = rep.scar_f1
        weights['scar_f1'] = cfg.w_scar_f1

    if rep.loc_n > 0:
        comps['localization'] = float(np.exp(-rep.loc_mean / cfg.localization_score_scale))
        weights['localization'] = cfg.w_localization

    if rep.pole_n > 0:
        comps['pole'] = rep.pole_accuracy
        weights['pole'] = cfg.w_pole

    if rep.len_n > 0:
        comps['length'] = float(np.exp(-rep.len_mae / cfg.length_score_scale))
        weights['length'] = cfg.w_length

    if not weights:
        return 0.0, comps

    total_w = sum(weights.values())
    obj = sum(comps[k] * weights[k] for k in comps) / total_w
    return float(obj), comps


def _safe_div(a, b):
    return float(a) / float(b) if b else 0.0


# ── Pretty printing ───────────────────────────────────────────────────────────

def format_report(rep: EvalReport) -> str:
    def f(v, p=3):
        return '—' if v is None else f'{v:.{p}f}'

    lines = [
        '── BS-Detector Evaluation ─────────────────────────────────',
        f'  Cells     matched={rep.n_matched}  '
        f'unmatched_det={rep.n_unmatched_det}  unmatched_gt={rep.n_unmatched_gt}',
        f'            precision={f(rep.cell_precision)}  recall={f(rep.cell_recall)}',
        f'  Scar      P={f(rep.scar_precision)}  R={f(rep.scar_recall)}  '
        f'F1={f(rep.scar_f1)}   (TP={rep.scar_tp} FP={rep.scar_fp} '
        f'FN={rep.scar_fn} TN={rep.scar_tn}, n={rep.scar_evaluable})',
        f'  Localize  n={rep.loc_n}  mean={f(rep.loc_mean,2)}px  '
        f'median={f(rep.loc_median,2)}px  p90={f(rep.loc_p90,2)}px  '
        f'within_tol={f(rep.loc_within_tol_frac)}',
        f'  Poles     n={rep.pole_n}  accuracy={f(rep.pole_accuracy)}',
        f'  Length    n={rep.len_n}  bias={f(rep.len_bias,2)}px  '
        f'MAE={f(rep.len_mae,2)}px  LoA=[{f(rep.len_loa_low,2)}, '
        f'{f(rep.len_loa_high,2)}]  r={f(rep.len_pearson)}',
        f'  OBJECTIVE {f(rep.objective,4)}   '
        f'{ {k: round(v,3) for k, v in rep.objective_components.items()} }',
        '───────────────────────────────────────────────────────────',
    ]
    return '\n'.join(lines)
