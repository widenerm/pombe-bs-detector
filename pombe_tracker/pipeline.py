"""
pipeline.py  –  CellProcessor, frame-level helpers, run_pipeline.

Changes vs. previous version
──────────────────────────────
• filter_valid_cells now accepts a config object and enforces aspect-ratio
  and circularity thresholds to reject debris / noise before tracking.
• process_cell makes a single detect() call (the detector always does a
  full-cell search internally; hemisphere preference is passed as a hint).
• All Config access goes through the passed config object (no singleton leaks).
• determine_new_pole_from_neighbors always receives config.
• debug_info contains no ghost keys.
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

        tentative_new_pole = None
        if new_pole_idx is not None and neighbor_conf in ('high', 'medium'):
            tentative_new_pole = endpoints[new_pole_idx]

        # ── Step 2 : birth scar detection (single full-cell pass) ────────────
        # The detector always searches the full contour; new_pole_point is used
        # only as a hemisphere preference for candidate selection.
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
            'seg_quality':    seg_quality,   # 'ok' | 'border_clip' | 'septum_fragment'
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


# ── Frame-level helpers ───────────────────────────────────────────────────────

def check_segmentation_quality(contour, kappa, frame_shape, config):
    """
    Detect Cellpose segmentation artifacts before any analysis is run.

    Two failure modes are caught:

    1. BORDER CLIP  – the contour actually touches the image boundary.
    2. SEPTUM FRAGMENT – extremely high curvature spike, well above anything
       seen on a healthy contour.

    Returns
    -------
    quality : 'ok' | 'border_clip' | 'septum_fragment'
    reason  : human-readable string for the researcher
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

    Parameters
    ----------
    regions     : list of regionprops objects
    frame_shape : (H, W[, ...]) shape of the frame
    config      : Config instance

    Returns
    -------
    dict  label → region  for all cells that pass all filters
    """
    h, w     = frame_shape[:2]
    min_area = config.MIN_CELL_AREA
    min_ar   = getattr(config, 'ASPECT_RATIO_MIN', 1.5)
    max_circ = getattr(config, 'MAX_CIRCULARITY',  0.85)
    valid    = {}

    for region in regions:
        # ── Area ──────────────────────────────────────────────────────────────
        if region.area < min_area:
            continue

        # ── Bounding-box border contact ───────────────────────────────────────
        r0, c0, r1, c1 = region.bbox
        if r0 == 0 or c0 == 0 or r1 == h or c1 == w:
            continue

        # ── Aspect ratio (rejects round blobs) ────────────────────────────────
        minor = region.minor_axis_length
        if minor < 1e-3:
            continue
        ar = region.major_axis_length / minor
        if ar < min_ar:
            continue

        # ── Circularity  (4π × area / perimeter²; circle = 1.0, rod ≈ 0.4–0.7)
        perim = region.perimeter
        if perim < 1e-3:
            continue
        circularity = (4.0 * np.pi * region.area) / (perim ** 2)
        if circularity > max_circ:
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
