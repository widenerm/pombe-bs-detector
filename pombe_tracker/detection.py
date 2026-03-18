"""
detection.py  –  BirthScarDetector

Strategy
────────
The full cell contour is always searched.  If a new-pole hint is provided,
it is used to prefer candidates in that hemisphere during selection, but
the full-cell search always runs so that ALL valid candidates are available
for downstream consensus stabilisation in postprocessing.py.

Two geometric constraints suppress false positives at the poles:

  1. WIDTH   The scar vector must span ≥ MIN_SCAR_WIDTH_RATIO × max cell width.
             At the poles the cell is narrow, so pole-tip peak pairs fail.

  2. ANGLE   The scar vector must be ⊥ to the long axis (within
             MAX_ANGLE_DEVIATION°).  At the poles the curvature peaks are
             separated along the axis, not across it, so their connecting
             vector is nearly parallel → rejected.

All valid candidates are stored in debug_info['scar_candidates'] as a list
of {'points', 'score', 'match_type'} dicts.  This lets postprocessing.py
retroactively enforce a cross-frame consensus position without re-running
the detector.
"""
import numpy as np
from .geometry import compute_smoothed_curvature, compute_pca_axis


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
        new_pole_point : if provided, prefer candidates in that hemisphere
        search_mode    : kept for API compatibility; full-cell is always used

        Returns
        -------
        scar_pair  : (pt1, pt2) on original contour, or None
        debug_info : dict with curvature data, all candidates, and diagnostics
        """
        smooth_pts, kappa = compute_smoothed_curvature(
            contour, self.cfg.SMOOTH_FACTOR, self.cfg.N_CONTOUR_POINTS)
        centre, axis = compute_pca_axis(smooth_pts)
        normal_vec = np.array([-axis[1], axis[0]])

        rel       = smooth_pts - centre
        long_proj = rel @ axis
        rng       = long_proj.max() - long_proj.min()
        long_norm = (long_proj - long_proj.min()) / (rng + 1e-10)

        transverse     = rel @ normal_vec
        max_thickness  = np.percentile(transverse, 98) - np.percentile(transverse, 2)
        min_scar_width = self.cfg.MIN_SCAR_WIDTH_RATIO * max_thickness

        debug_info = {
            'smooth_pts':   smooth_pts,
            'kappa':        kappa,
            'centre':       centre,
            'center':       centre,    # alias for backward compat
            'axis':         axis,
            'long_norm':    long_norm,
            # display_mask is always full-cell for visualization
            'display_mask': np.ones(len(smooth_pts), dtype=bool),
        }

        peaks = self._find_peaks(kappa)
        debug_info['peaks'] = peaks

        if len(peaks) == 0:
            debug_info['error']           = 'no_peaks'
            debug_info['scar_candidates'] = []
            return None, debug_info

        # ── Collect ALL valid candidates over the full cell ───────────────────
        strict_cands = self._collect_strict_candidates(
            peaks, smooth_pts, kappa, centre, axis, normal_vec, min_scar_width, long_norm)
        asym_cands   = self._collect_asymmetric_candidates(
            peaks, smooth_pts, kappa, centre, axis, normal_vec, long_norm, min_scar_width)

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

        # ── Select best candidate (hemisphere-aware) ─────────────────────────
        best = self._select_best_candidate(
            strict_cands, asym_cands, centre, axis, new_pole_point)

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

    def _score(self, i1, i2, kappa, long_norm):
        """
        Pair quality score.  Upweight candidates away from the cell centre
        because the most recent scar is typically close to the new pole.
        """
        base    = float(kappa[i1]) + float(kappa[i2])
        avg_pos = (long_norm[i1] + long_norm[i2]) / 2.0
        weight  = 1.0 + 4.0 * abs(avg_pos - 0.5)   # 1.0 at centre, 3.0 at poles
        return base * weight

    def _is_valid_scar_vector(self, pt1, pt2, centre, axis, normal_vec, min_width):
        """
        Return (is_valid, scar_width) for a candidate scar pt1 → pt2.
        Checks: opposite sides of cell, minimum width, perpendicularity.
        """
        side1 = np.dot(pt1 - centre, normal_vec)
        side2 = np.dot(pt2 - centre, normal_vec)
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

    def _collect_strict_candidates(self, peaks, smooth_pts, kappa, centre, axis,
                                    normal_vec, min_width, long_norm):
        """All valid peak-pair candidates (both endpoints are curvature peaks)."""
        candidates = []
        for i in range(len(peaks)):
            for j in range(i + 1, len(peaks)):
                p1, p2   = peaks[i], peaks[j]
                pt1, pt2 = smooth_pts[p1], smooth_pts[p2]
                valid, _ = self._is_valid_scar_vector(
                    pt1, pt2, centre, axis, normal_vec, min_width)
                if valid:
                    candidates.append(dict(
                        indices = (p1, p2),
                        points  = (pt1, pt2),
                        score   = self._score(p1, p2, kappa, long_norm),
                    ))
        return candidates

    def _collect_asymmetric_candidates(self, peaks, smooth_pts, kappa, centre, axis,
                                        normal_vec, long_norm, min_width):
        """
        Asymmetric candidates: one strong curvature peak paired with the
        highest-curvature point on the opposite side at the same longitudinal
        position (within ±5% of normalised cell length).
        """
        n           = len(smooth_pts)
        all_indices = np.arange(n)
        candidates  = []

        for p_strong in peaks:
            pt_strong   = smooth_pts[p_strong]
            side_strong = np.sign(np.dot(pt_strong - centre, normal_vec))
            long_pos    = long_norm[p_strong]

            partners = [
                idx for idx in all_indices
                if np.sign(np.dot(smooth_pts[idx] - centre, normal_vec)) != side_strong
                and abs(long_norm[idx] - long_pos) <= 0.05
            ]
            if not partners:
                continue

            best_weak = max(partners, key=lambda idx: kappa[idx])
            pt_weak   = smooth_pts[best_weak]
            valid, _  = self._is_valid_scar_vector(
                pt_strong, pt_weak, centre, axis, normal_vec, min_width)
            if valid:
                candidates.append(dict(
                    indices = (p_strong, best_weak),
                    points  = (pt_strong, pt_weak),
                    score   = self._score(p_strong, best_weak, kappa, long_norm),
                ))
        return candidates

    def _select_best_candidate(self, strict_cands, asym_cands, centre, axis,
                                new_pole_point):
        """
        Select the best candidate.

        Priority order:
          1. Strict pair in the new-pole hemisphere (if hint is given)
          2. Strict pair (any position)
          3. Asymmetric pair in the new-pole hemisphere (if hint is given)
          4. Asymmetric pair (any position)
        """
        def midpoint_side(c):
            mp = (np.array(c['points'][0]) + np.array(c['points'][1])) / 2
            return np.sign(float(np.dot(mp - centre, axis)))

        if new_pole_point is not None:
            new_side = np.sign(float(np.dot(np.array(new_pole_point) - centre, axis)))

            hemi_strict = [c for c in strict_cands if midpoint_side(c) == new_side]
            if hemi_strict:
                return max(hemi_strict, key=lambda x: x['score'])

            if strict_cands:
                return max(strict_cands, key=lambda x: x['score'])

            hemi_asym = [c for c in asym_cands if midpoint_side(c) == new_side]
            if hemi_asym:
                return max(hemi_asym, key=lambda x: x['score'])

            if asym_cands:
                return max(asym_cands, key=lambda x: x['score'])
        else:
            if strict_cands:
                return max(strict_cands, key=lambda x: x['score'])
            if asym_cands:
                return max(asym_cands, key=lambda x: x['score'])

        # Should not be reached (caller checks all_cands is non-empty)
        raise RuntimeError('_select_best_candidate called with no candidates')

    def _map_to_original(self, smooth_points, original_contour):
        """Snap smoothed scar points back onto the original contour."""
        orig = np.array(original_contour)

        def closest(target):
            return orig[np.argmin(np.linalg.norm(orig - target, axis=1))]

        return (closest(smooth_points[0]), closest(smooth_points[1]))
