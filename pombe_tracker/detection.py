"""
detection.py  –  BirthScarDetector

Strategy
────────
The full cell contour is always searched.  If a new-pole hint is provided,
the candidate whose midpoint is geometrically closest to the new pole is
selected; otherwise the highest-prominence candidate wins.

Two geometric constraints suppress false positives at the poles:

  1. WIDTH   The scar vector must span ≥ MIN_SCAR_WIDTH_RATIO × average
             mid-cell width (not max width — max can be inflated by the scar
             itself).  Average width is computed from seven evenly-spaced
             cross-sections between normalised positions 0.2 and 0.8.

  2. ANGLE   The scar vector must be ⊥ to the long axis (within
             MAX_ANGLE_DEVIATION°).  At the poles the curvature peaks are
             separated along the axis, not across it, so their connecting
             vector is nearly parallel → rejected.

Scoring uses local prominence rather than raw peak height.
Prominence = peak curvature − mean curvature in a ±PROMINENCE_WINDOW index
window around the peak (computed on the periodic contour).  This rewards
peaks that stand out sharply from a flat surrounding baseline — the
signature of a genuine birth-scar ridge — and discards broad, gently-curved
sections that happen to have a slightly elevated absolute curvature.

All valid candidates are stored in debug_info['scar_candidates'] so
postprocessing.py can retroactively enforce a cross-frame consensus position
without re-running the detector.
"""
import numpy as np
from .geometry import compute_smoothed_curvature, compute_pca_axis

# Contour indices on each side of a peak used for the local baseline.
PROMINENCE_WINDOW = 25


class BirthScarDetector:

    def __init__(self, config):
        self.cfg = config

    # ── Public ───────────────────────────────────────────────────────────────

    def detect(self, contour, new_pole_point=None, search_mode='auto'):
        """
        Detect the most recent birth scar on *contour*.

        Parameters
        ----------
        contour        : raw contour from skimage (N, 2)
        new_pole_point : if provided, select the candidate closest to this point
        search_mode    : kept for API compatibility; full-cell is always used

        Returns
        -------
        scar_pair  : (pt1, pt2) on original contour, or None
        debug_info : dict with curvature data, all candidates, and diagnostics
        """
        smooth_pts, kappa = compute_smoothed_curvature(
            contour, self.cfg.SMOOTH_FACTOR, self.cfg.N_CONTOUR_POINTS)
        center, axis = compute_pca_axis(smooth_pts)
        normal_vec   = np.array([-axis[1], axis[0]])

        rel       = smooth_pts - center
        long_proj = rel @ axis
        rng       = long_proj.max() - long_proj.min()
        long_norm = (long_proj - long_proj.min()) / (rng + 1e-10)
        transverse = rel @ normal_vec

        max_thickness = np.percentile(transverse, 98) - np.percentile(transverse, 2)

        # ── Average mid-cell width ────────────────────────────────────────────
        # Sample seven cross-sections between 20 % and 80 % of cell length.
        # Excludes tapered poles; robust to the scar bulge inflating the max.
        sample_norms = np.linspace(0.2, 0.8, 7)
        sample_widths = []
        for sn in sample_norms:
            mask = np.abs(long_norm - sn) < 0.05
            if mask.sum() >= 2:
                t = transverse[mask]
                sample_widths.append(float(t.max() - t.min()))
        avg_width      = float(np.mean(sample_widths)) if sample_widths else max_thickness
        min_scar_width = self.cfg.MIN_SCAR_WIDTH_RATIO * avg_width

        debug_info = {
            'smooth_pts':    smooth_pts,
            'kappa':         kappa,
            'center':        center,
            'axis':          axis,
            'long_norm':     long_norm,
            'avg_width':     avg_width,
            'max_thickness': max_thickness,
            # display_mask is always full-cell for visualization
            'display_mask':  np.ones(len(smooth_pts), dtype=bool),
        }

        peaks = self._find_peaks(kappa)
        debug_info['peaks'] = peaks

        if len(peaks) == 0:
            debug_info['error']           = 'no_peaks'
            debug_info['scar_candidates'] = []
            return None, debug_info

        # ── Collect ALL valid candidates over the full cell ───────────────────
        strict_cands = self._collect_strict_candidates(
            peaks, smooth_pts, kappa, center, axis, normal_vec, min_scar_width, long_norm)
        asym_cands   = self._collect_asymmetric_candidates(
            peaks, smooth_pts, kappa, center, axis, normal_vec, long_norm, min_scar_width)

        for c in strict_cands:
            c['match_type'] = 'strict'
        for c in asym_cands:
            c['match_type'] = 'fallback'

        all_cands = strict_cands + asym_cands

        # Store every valid candidate for consensus postprocessing
        debug_info['scar_candidates'] = [
            {
                'points':     c['points'],
                'score':      c['score'],
                'match_type': c['match_type'],
            }
            for c in all_cands
        ]

        if not all_cands:
            debug_info['error'] = 'no_valid_pairs'
            return None, debug_info

        # ── Select best candidate ─────────────────────────────────────────────
        best = self._select_best_candidate(
            strict_cands, asym_cands, center, axis, new_pole_point)

        debug_info.update(
            match_type = best['match_type'],
            best_pair  = best['indices'],
            best_score = best['score'],
        )

        return self._map_to_original(best['points'], contour), debug_info

    # ── Private helpers ──────────────────────────────────────────────────────

    def _find_peaks(self, kappa):
        """Local maxima of curvature over the full contour, positive only."""
        d_kappa   = np.gradient(kappa)
        sign_diff = np.diff(np.sign(d_kappa))
        all_peaks = np.where(sign_diff < 0)[0]
        return np.array([p for p in all_peaks if kappa[p] > 0], dtype=int)

    def _peak_prominence(self, kappa, peak_idx):
        """
        Local prominence of a curvature peak.

        Prominence = kappa[peak] − mean(kappa in a ±PROMINENCE_WINDOW ring
        neighbourhood, excluding the peak itself).

        Rewards peaks that rise sharply above a flat baseline (birth-scar
        ridge) over broad regions with uniformly elevated curvature.
        """
        n       = len(kappa)
        indices = [(peak_idx + d) % n
                   for d in range(-PROMINENCE_WINDOW, PROMINENCE_WINDOW + 1)
                   if d != 0]
        baseline = float(np.mean(kappa[indices]))
        return max(0.0, float(kappa[peak_idx]) - baseline)

    def _score(self, i1, i2, kappa, long_norm=None):
        """
        Pair quality = sum of individual peak prominences.
        long_norm is accepted for API compatibility but not used.
        """
        return self._peak_prominence(kappa, i1) + self._peak_prominence(kappa, i2)

    def _is_valid_scar_vector(self, pt1, pt2, center, axis, normal_vec, min_width):
        """
        Return (is_valid, scar_width) for a candidate scar pt1 → pt2.
        Checks: opposite sides of cell, minimum width vs average, perpendicularity.
        """
        side1 = np.dot(pt1 - center, normal_vec)
        side2 = np.dot(pt2 - center, normal_vec)
        if side1 * side2 >= 0:
            return False, 0.0

        scar_vec  = pt2 - pt1
        width     = float(np.linalg.norm(scar_vec))
        if width < min_width:
            return False, width

        scar_unit = scar_vec / width
        angle_deg = np.degrees(
            np.arccos(np.clip(np.abs(np.dot(scar_unit, axis)), 0.0, 1.0)))
        if abs(90.0 - angle_deg) >= self.cfg.MAX_ANGLE_DEVIATION:
            return False, width

        return True, width

    def _collect_strict_candidates(self, peaks, smooth_pts, kappa, center, axis,
                                    normal_vec, min_width, long_norm):
        """All valid peak-pair candidates (both endpoints are curvature peaks)."""
        candidates = []
        for i in range(len(peaks)):
            for j in range(i + 1, len(peaks)):
                p1, p2   = peaks[i], peaks[j]
                pt1, pt2 = smooth_pts[p1], smooth_pts[p2]
                valid, _ = self._is_valid_scar_vector(
                    pt1, pt2, center, axis, normal_vec, min_width)
                if valid:
                    candidates.append(dict(
                        indices = (p1, p2),
                        points  = (pt1, pt2),
                        score   = self._score(p1, p2, kappa, long_norm),
                    ))
        return candidates

    def _collect_asymmetric_candidates(self, peaks, smooth_pts, kappa, center, axis,
                                        normal_vec, long_norm, min_width):
        """
        Asymmetric candidates: one strong curvature peak paired with the
        highest-curvature point on the opposite side at the same longitudinal
        position (within ±5 % of normalized cell length).
        """
        n           = len(smooth_pts)
        all_indices = np.arange(n)
        candidates  = []

        for p_strong in peaks:
            pt_strong   = smooth_pts[p_strong]
            side_strong = np.sign(np.dot(pt_strong - center, normal_vec))
            long_pos    = long_norm[p_strong]

            partners = [
                idx for idx in all_indices
                if np.sign(np.dot(smooth_pts[idx] - center, normal_vec)) != side_strong
                and abs(long_norm[idx] - long_pos) <= 0.05
            ]
            if not partners:
                continue

            best_weak = max(partners, key=lambda idx: kappa[idx])
            pt_weak   = smooth_pts[best_weak]
            valid, _  = self._is_valid_scar_vector(
                pt_strong, pt_weak, center, axis, normal_vec, min_width)
            if valid:
                candidates.append(dict(
                    indices = (p_strong, best_weak),
                    points  = (pt_strong, pt_weak),
                    score   = self._score(p_strong, best_weak, kappa, long_norm),
                ))
        return candidates

    def _select_best_candidate(self, strict_cands, asym_cands, center, axis,
                                new_pole_point):
        """
        Select the best candidate.

        When a new-pole hint is available, pick the candidate whose midpoint
        is geometrically closest to the new pole — directly implementing
        "if multiple candidates, pick the one nearest the new end."
        Strict pairs are still tried before asymmetric ones so that a
        weak-endpoint fallback is only used when no peak-pair exists.

        When no hint is available, fall back to highest-prominence score.
        """
        def midpoint(c):
            return (np.array(c['points'][0]) + np.array(c['points'][1])) / 2.0

        if new_pole_point is not None:
            np_arr = np.array(new_pole_point)
            if strict_cands:
                return min(strict_cands,
                           key=lambda c: float(np.linalg.norm(midpoint(c) - np_arr)))
            if asym_cands:
                return min(asym_cands,
                           key=lambda c: float(np.linalg.norm(midpoint(c) - np_arr)))
        else:
            if strict_cands:
                return max(strict_cands, key=lambda x: x['score'])
            if asym_cands:
                return max(asym_cands, key=lambda x: x['score'])

        raise RuntimeError('_select_best_candidate called with no candidates')

    def _map_to_original(self, smooth_points, original_contour):
        """Snap smoothed scar points back onto the original contour."""
        orig = np.array(original_contour)

        def closest(target):
            return orig[np.argmin(np.linalg.norm(orig - target, axis=1))]

        return (closest(smooth_points[0]), closest(smooth_points[1]))
