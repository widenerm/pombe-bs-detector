"""
pipeline.py  –  CellProcessor, frame-level helpers, run_pipeline.

Changes vs. previous version
──────────────────────────────
• filter_valid_cells enforces aspect-ratio and circularity thresholds.
• process_cell makes a single detect() call (full-cell, pole hint as hint).
• determine_poles_strategy now accepts lineage_new_pole / lineage_old_pole
  keyword arguments for freshly divided daughters.
• After each frame's tracking pass, _apply_lineage_poles() corrects the pole
  assignment for any daughters produced by a division in that frame, then
  re-selects the best scar candidate using the corrected pole hint.
• Chain cells (both poles touching credible neighbors) are detected in
  determine_new_pole_from_neighbors and handled by skipping directly to
  scar-based or morphology assignment.
"""
import numpy as np
from skimage.measure import find_contours, regionprops

from .geometry import (
    compute_pca_axis,
    compute_smoothed_curvature,
    get_contour_endpoints,
    measure_width_at_position,
    measure_cell_length,
    measure_pole_lengths,
    compute_curvature_fingerprint,
)
from .detection import BirthScarDetector
from .poles import (
    find_pole_to_pole_neighbors,
    determine_new_pole_from_neighbors,
    determine_poles_strategy,
)


class CellProcessor:
    """Process a single cell region and return a result dict."""

    def __init__(self, config):
        self.cfg      = config
        self.detector = BirthScarDetector(config)

    def process_cell(self, region, labels, frame, all_cell_info):
        """
        Full analysis pipeline for one cell.

        Returns a result dict or None if the contour cannot be found.
        """
        cell_mask = (labels == region.label)
        contours  = find_contours(cell_mask, 0.5)
        if not contours:
            return None
        contour = max(contours, key=len)

        # ── Geometry ─────────────────────────────────────────────────────────
        smooth_pts, kappa = compute_smoothed_curvature(
            contour, self.cfg.SMOOTH_FACTOR, self.cfg.N_CONTOUR_POINTS)
        center, axis  = compute_pca_axis(smooth_pts)
        endpoints     = get_contour_endpoints(contour, center, axis)

        rel       = smooth_pts - center
        long_proj = rel @ axis
        long_norm = (long_proj - long_proj.min()) / (long_proj.max() - long_proj.min() + 1e-10)

        fingerprint = compute_curvature_fingerprint(kappa)

        # ── Segmentation quality gate ─────────────────────────────────────────
        seg_quality, seg_reason = check_segmentation_quality(
            contour, kappa, frame.shape, self.cfg)

        # ── Step 1 : tentative new pole from neighbors ────────────────────────
        neighbors = find_pole_to_pole_neighbors(
            region.label, endpoints, all_cell_info,
            self.cfg.POLE_PROXIMITY_THRESHOLD)
        new_pole_idx, neighbor_conf = determine_new_pole_from_neighbors(
            neighbors, endpoints, self.cfg)

        # Only use the neighbor hint for detection when confidence is unambiguous.
        # Chain cells (conf == 'chain') and low-confidence cases get no hint here;
        # the pole will be resolved by scar or lineage after detection.
        tentative_new_pole = None
        if new_pole_idx is not None and neighbor_conf in ('high', 'medium'):
            tentative_new_pole = endpoints[new_pole_idx]

        # ── Step 2 : birth scar detection (single full-cell pass) ────────────
        scar_pair, debug_info = self.detector.detect(
            contour, new_pole_point=tentative_new_pole)

        # ── Step 3 : final pole assignment ───────────────────────────────────
        scar_midpoint = (scar_pair[0] + scar_pair[1]) / 2 if scar_pair is not None else None

        if new_pole_idx is not None and neighbor_conf == 'high':
            new_pole   = endpoints[new_pole_idx]
            old_pole   = endpoints[1 - new_pole_idx]
            method, confidence = 'neighbor_proximity', 'high'
        else:
            new_pole, old_pole, _, method, confidence = determine_poles_strategy(
                region.label, endpoints, center, axis, contour,
                all_cell_info, self.cfg, scar_midpoint)

        # ── Measurements ─────────────────────────────────────────────────────
        cell_length    = measure_cell_length(endpoints[0], endpoints[1])
        width_centroid = measure_width_at_position(
            smooth_pts, center, axis, long_norm, target_norm=0.5)

        # ── Populate debug_info ──────────────────────────────────────────────
        debug_info.update({
            'new_pole_point':        new_pole,
            'old_pole_point':        old_pole,
            'pole_method':           method,
            'pole_confidence':       confidence,
            'curvature_fingerprint': fingerprint,
        })

        result = {
            'label':          region.label,
            'centroid':       region.centroid,
            'area':           region.area,
            'contour':        contour,
            'debug_info':     debug_info,
            'neighbors':      neighbors,
            'length':         cell_length,
            'width_centroid': width_centroid,
            'seg_quality':    seg_quality,
            'seg_reason':     seg_reason,
        }

        if scar_pair is not None:
            new_len, old_len = measure_pole_lengths(scar_midpoint, new_pole, old_pole)
            scar_width = float(np.linalg.norm(scar_pair[0] - scar_pair[1]))

            result.update({
                'scar_detected':  True,
                'scar_points':    scar_pair,
                'scar_midpoint':  scar_midpoint,
                'new_end_length': new_len,
                'old_end_length': old_len,
                'width_scar':     scar_width,
            })
            debug_info.update({
                'recent_scar':    scar_midpoint,
                'new_end_length': new_len,
                'old_end_length': old_len,
            })
        else:
            result.update({
                'scar_detected':  False,
                'width_scar':     None,
                'new_end_length': None,
                'old_end_length': None,
            })

        return result


# ── Lineage-based pole correction ─────────────────────────────────────────────

def _apply_lineage_poles(results, tracker, frame_idx):
    """
    For any daughters produced by a division in *frame_idx*, correct their
    new / old pole assignment using the parent's stored pole coordinates, then
    re-select the best scar candidate using the corrected pole as the hint.

    Called in run_pipeline immediately after tracker.update() assigns names.

    Convention (matches naming scheme):
      X0  – new-end daughter: inherited parent's new-end as its OLD pole;
             its NEW pole is the division site.
      X1  – old-end daughter: inherited parent's old-end as its OLD pole;
             its NEW pole is the division site.

    The division site for each daughter is estimated as the endpoint closest
    to the sibling's centroid.  If parent pole coordinates are available, they
    are used to identify which endpoint of each daughter is the inherited old
    pole (the one closest to the corresponding parent pole).
    """
    name_to_result = {r.get('cell_name'): r for r in results
                      if r.get('cell_name') is not None}

    for ev in tracker.lineage_log:
        if ev['frame'] != frame_idx:
            continue

        d0_name, d1_name = ev['daughters']   # d0 = new-end, d1 = old-end
        d0 = name_to_result.get(d0_name)
        d1 = name_to_result.get(d1_name)

        if d0 is None or d1 is None:
            continue

        d0_centroid = np.array(d0['centroid'])
        d1_centroid = np.array(d1['centroid'])

        # ── Determine the new poles (division site) for each daughter ─────────
        # New pole = the endpoint of the daughter closest to the sibling.
        d0_dbg = d0['debug_info']
        d1_dbg = d1['debug_info']
        d0_ep1 = d0_dbg.get('new_pole_point')
        d0_ep2 = d0_dbg.get('old_pole_point')
        d1_ep1 = d1_dbg.get('new_pole_point')
        d1_ep2 = d1_dbg.get('old_pole_point')

        # Fall back to using the currently assigned poles if the endpoints are
        # stored there; we need two distinct points per daughter.
        if d0_ep1 is None or d0_ep2 is None or d1_ep1 is None or d1_ep2 is None:
            continue

        d0_ep1 = np.array(d0_ep1)
        d0_ep2 = np.array(d0_ep2)
        d1_ep1 = np.array(d1_ep1)
        d1_ep2 = np.array(d1_ep2)

        parent_new_pole = ev.get('parent_new_pole')
        parent_old_pole = ev.get('parent_old_pole')

        if parent_new_pole is not None and parent_old_pole is not None:
            # Use parent pole coordinates to identify inherited old poles.
            # d0 inherited the parent's new-end as its old pole.
            # d1 inherited the parent's old-end as its old pole.
            p_new = np.array(parent_new_pole)
            p_old = np.array(parent_old_pole)

            # d0 old pole = whichever endpoint is closest to parent's new pole
            if np.linalg.norm(d0_ep1 - p_new) < np.linalg.norm(d0_ep2 - p_new):
                d0_new_pole, d0_old_pole = d0_ep2, d0_ep1
            else:
                d0_new_pole, d0_old_pole = d0_ep1, d0_ep2

            # d1 old pole = whichever endpoint is closest to parent's old pole
            if np.linalg.norm(d1_ep1 - p_old) < np.linalg.norm(d1_ep2 - p_old):
                d1_new_pole, d1_old_pole = d1_ep2, d1_ep1
            else:
                d1_new_pole, d1_old_pole = d1_ep1, d1_ep2
        else:
            # No parent pole info — use sibling proximity only.
            # New pole = endpoint closest to sibling centroid.
            if np.linalg.norm(d0_ep1 - d1_centroid) < np.linalg.norm(d0_ep2 - d1_centroid):
                d0_new_pole, d0_old_pole = d0_ep1, d0_ep2
            else:
                d0_new_pole, d0_old_pole = d0_ep2, d0_ep1

            if np.linalg.norm(d1_ep1 - d0_centroid) < np.linalg.norm(d1_ep2 - d0_centroid):
                d1_new_pole, d1_old_pole = d1_ep1, d1_ep2
            else:
                d1_new_pole, d1_old_pole = d1_ep2, d1_ep1

        # ── Apply corrected poles to both daughters ───────────────────────────
        for result, new_pole, old_pole in [
            (d0, d0_new_pole, d0_old_pole),
            (d1, d1_new_pole, d1_old_pole),
        ]:
            dbg = result['debug_info']
            dbg['new_pole_point']  = new_pole
            dbg['old_pole_point']  = old_pole
            dbg['pole_method']     = 'lineage'
            dbg['pole_confidence'] = 'high'

            # Re-select scar candidate using corrected pole hint
            _reselect_scar(result, new_pole)

        print(f"  Lineage poles applied: {d0_name} / {d1_name}")


def _reselect_scar(result, new_pole_point):
    """
    Re-select the best scar candidate from debug_info['scar_candidates']
    using a (possibly corrected) new-pole hint.

    Picks the candidate whose midpoint is closest to new_pole_point,
    then recomputes all derived measurements.
    """
    cands = result['debug_info'].get('scar_candidates', [])
    if not cands:
        return

    np_arr = np.array(new_pole_point)

    def midpoint_dist(c):
        mp = (np.array(c['points'][0]) + np.array(c['points'][1])) / 2.0
        return float(np.linalg.norm(mp - np_arr))

    best = min(cands, key=midpoint_dist)

    pt1    = np.array(best['points'][0])
    pt2    = np.array(best['points'][1])
    new_mp = (pt1 + pt2) / 2.0

    # Snap to original contour
    orig = np.array(result['contour'])
    def closest(target):
        return orig[np.argmin(np.linalg.norm(orig - target, axis=1))]

    scar_pair = (closest(pt1), closest(pt2))
    scar_mp   = (np.array(scar_pair[0]) + np.array(scar_pair[1])) / 2.0

    new_pole = np.array(result['debug_info']['new_pole_point'])
    old_pole = np.array(result['debug_info']['old_pole_point'])
    new_len, old_len = measure_pole_lengths(scar_mp, new_pole, old_pole)

    result['scar_detected']  = True
    result['scar_points']    = scar_pair
    result['scar_midpoint']  = scar_mp
    result['width_scar']     = float(np.linalg.norm(np.array(scar_pair[0]) - np.array(scar_pair[1])))
    result['new_end_length'] = new_len
    result['old_end_length'] = old_len

    dbg = result['debug_info']
    dbg['match_type']     = best['match_type']
    dbg['recent_scar']    = scar_mp
    dbg['new_end_length'] = new_len
    dbg['old_end_length'] = old_len


# ── Frame-level helpers ───────────────────────────────────────────────────────

def check_segmentation_quality(contour, kappa, frame_shape, config):
    """
    Detect Cellpose segmentation artifacts before any analysis is run.

    Returns
    -------
    quality : 'ok' | 'border_clip' | 'septum_fragment'
    reason  : human-readable string
    """
    h, w = frame_shape[:2]

    rows = contour[:, 0]
    cols = contour[:, 1]
    if rows.min() <= 1 or cols.min() <= 1 or rows.max() >= h - 2 or cols.max() >= w - 2:
        return 'border_clip', 'Contour touches image boundary (incomplete cell)'

    max_kappa = float(np.max(np.abs(kappa)))
    threshold = getattr(config, 'CURVATURE_QUALITY_THRESHOLD', 0.1)
    if max_kappa > threshold:
        return 'septum_fragment', (
            f'Max |κ| = {max_kappa:.3f} exceeds threshold {threshold} '
            f'(likely septum or clipped boundary)')

    return 'ok', ''


def filter_valid_cells(regions, frame_shape, config):
    """
    Remove cells that are too small, touch the image border, have wrong
    aspect ratio (too round → debris), or are too circular (not rod-shaped).

    Returns dict  label → region  for all cells that pass all filters.
    """
    h, w     = frame_shape[:2]
    min_area = config.MIN_CELL_AREA
    min_ar   = getattr(config, 'ASPECT_RATIO_MIN', 1.5)
    max_circ = getattr(config, 'MAX_CIRCULARITY',  0.85)
    valid    = {}

    for region in regions:
        if region.area < min_area:
            continue

        r0, c0, r1, c1 = region.bbox
        if r0 == 0 or c0 == 0 or r1 == h or c1 == w:
            continue

        minor = region.minor_axis_length
        if minor < 1e-3:
            continue
        if region.major_axis_length / minor < min_ar:
            continue

        perim = region.perimeter
        if perim < 1e-3:
            continue
        if (4.0 * np.pi * region.area) / (perim ** 2) > max_circ:
            continue

        valid[region.label] = region

    return valid


def prepare_cell_info(valid_regions, labels, config):
    """
    Pre-compute geometry for all cells so neighbor analysis can reference
    any cell without reprocessing.
    """
    all_cell_info = {}
    for region in valid_regions:
        mask     = (labels == region.label)
        contours = find_contours(mask, 0.5)
        if not contours:
            continue
        contour       = max(contours, key=len)
        smooth_pts, _ = compute_smoothed_curvature(
            contour, config.SMOOTH_FACTOR, config.N_CONTOUR_POINTS)
        center, axis  = compute_pca_axis(smooth_pts)
        endpoints     = get_contour_endpoints(contour, center, axis)
        all_cell_info[region.label] = {
            'endpoints': endpoints,
            'centroid':  region.centroid,
            'center':    center,
            'axis':      axis,
        }
    return all_cell_info


def process_frame(frame, segmenter, processor):
    """Segment a frame and process every valid cell."""
    labels  = segmenter.segment(frame)
    regions = regionprops(labels)
    if not regions:
        return []

    valid_cells = filter_valid_cells(regions, frame.shape, processor.cfg)
    if not valid_cells:
        return []

    valid_regions = list(valid_cells.values())
    all_cell_info = prepare_cell_info(valid_regions, labels, processor.cfg)

    results = []
    for region in valid_regions:
        result = processor.process_cell(region, labels, frame, all_cell_info)
        if result is not None:
            results.append(result)
    return results


# ── Top-level pipeline ────────────────────────────────────────────────────────

def run_pipeline(frames, config, tracker=None):
    """
    Run the full detection + tracking pipeline on *frames*.

    Parameters
    ----------
    frames  : list/array of 2-D images
    config  : Config instance
    tracker : optional pre-initialized CellTracker (for resuming)

    Returns
    -------
    all_results : list of  {'frame_idx', 'frame', 'cells'}  dicts
    """
    from .segmentation import CellposeSegmenter
    from .tracking     import CellTracker

    segmenter = CellposeSegmenter(config)
    processor = CellProcessor(config)
    if tracker is None:
        tracker = CellTracker(config)

    all_results = []
    n_frames    = len(frames)

    for frame_idx, frame in enumerate(frames):
        print(f"\n[Frame {frame_idx + 1}/{n_frames}]")
        results = process_frame(frame, segmenter, processor)

        name_map = tracker.update(results, frame_idx=frame_idx)
        for r in results:
            r['cell_name'] = name_map.get(r['label'], '?')

        # After naming, apply lineage-based pole correction for any daughters
        # produced by a division in this frame, then re-select their scars.
        _apply_lineage_poles(results, tracker, frame_idx)

        detected = sum(1 for r in results if r['scar_detected'])
        print(f"  {len(results)} cells  |  {detected} scars detected")
        for r in results:
            name = r['cell_name']
            if r['scar_detected']:
                ni, oi = r['new_end_length'], r['old_end_length']
                print(f"  {name}: ✓  new={ni:.1f}px  old={oi:.1f}px  "
                      f"ratio={ni/oi:.2f}  [{r['debug_info']['pole_method']}]")
            else:
                err = r['debug_info'].get('error', '?')
                print(f"  {name}: ✗  ({err})")

        all_results.append({
            'frame_idx': frame_idx,
            'frame':     frame,
            'cells':     results,
            'tracker':   tracker,
        })

    return all_results
