"""
detection.py  –  BirthScarDetector

Strategy
────────
No region of the cell is excluded from the search.  Instead, two geometric
constraints naturally suppress false positives at the poles:

  1. WIDTH   The scar vector must span ≥ MIN_SCAR_WIDTH_RATIO × max cell width.
             At the poles the cell is narrow, so pole-tip peak pairs fail.

  2. ANGLE   The scar vector must be ⊥ to the long axis (within MAX_ANGLE_DEVIATION°).
             At the poles the curvature peaks are separated along the axis, not
             across it, so their connecting vector is nearly parallel → rejected.

If no peaks are discernible (e.g. very young cell), the algorithm picks the
region where the cell begins rounding toward the pole, which is biologically the
birth-scar location in a freshly born cell.
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
        contour        : raw contour from skimage (N,2)
        new_pole_point : if provided, restrict initial search to that hemisphere
        search_mode    : 'auto' | 'hemisphere' | 'full'

        Returns
        -------
        scar_pair  : (pt1, pt2) on original contour, or None
        debug_info : dict with curvature data and diagnostic fields
        """
        smooth_pts, kappa = compute_smoothed_curvature(
            contour, self.cfg.SMOOTH_FACTOR, self.cfg.N_CONTOUR_POINTS)
        centre, axis = compute_pca_axis(smooth_pts)
        normal_vec = np.array([-axis[1], axis[0]])

        rel       = smooth_pts - centre
        long_proj = rel @ axis
        rng       = long_proj.max() - long_proj.min()
        long_norm = (long_proj - long_proj.min()) / (rng + 1e-10)

        # Determine search region
        if search_mode == 'auto':
            search_mode = 'hemisphere' if new_pole_point is not None else 'full'

        if search_mode == 'hemisphere' and new_pole_point is not None:
            new_side = np.sign(np.dot(new_pole_point - centre, axis))
            # Include the hemisphere of the new pole plus a 5% band around centre
            hemi_mask    = (np.sign(long_proj) == new_side) | (np.abs(long_norm - 0.5) < 0.05)
            valid_mask   = hemi_mask
            search_region = 'new_pole_hemisphere'
        else:
            valid_mask    = np.ones(len(smooth_pts), dtype=bool)
            search_region = 'full_cell'

        valid_indices = np.where(valid_mask)[0]

        # Cell width (used for min scar width check)
        transverse    = rel @ normal_vec
        max_thickness = np.percentile(transverse, 98) - np.percentile(transverse, 2)
        min_scar_width = self.cfg.MIN_SCAR_WIDTH_RATIO * max_thickness

        debug_info = {
            'smooth_pts':    smooth_pts,
            'kappa':         kappa,
            'centre':        centre,
            'center':        centre,   # alias for backward compat
            'axis':          axis,
            'long_norm':     long_norm,
            'valid_mask':    valid_mask,
            # display_mask is always full-cell. The hemisphere is an internal
            # first-pass hint, not an exclusion zone — showing a restricted
            # region in the curvature plot would be misleading.
            'display_mask':  np.ones(len(smooth_pts), dtype=bool),
            'search_region': search_region,
        }

        peaks = self._find_peaks(kappa, valid_indices)
        debug_info['peaks'] = peaks

        if len(peaks) == 0:
            debug_info['error'] = f'no_peaks_in_{search_region}'
            return None, debug_info

        # ── Stage 1 : strict perpendicular pair (both sides are peaks) ───────
        scar = self._find_strict_pair(
            peaks, smooth_pts, kappa, centre, axis, normal_vec,
            long_proj, min_scar_width, long_norm, search_mode)

        if scar is not None:
            debug_info.update(match_type='strict',
                              best_pair=scar['indices'],
                              best_score=scar['score'])
            return self._map_to_original(scar['points'], contour), debug_info

        # ── Stage 2 : asymmetric fallback (one strong peak, mirror best point)
        scar = self._find_asymmetric_pair(
            peaks, valid_indices, smooth_pts, kappa, centre, axis,
            normal_vec, long_proj, long_norm, min_scar_width, search_mode)

        if scar is not None:
            debug_info.update(match_type='fallback',
                              best_pair=scar['indices'],
                              best_score=scar['score'])
            return self._map_to_original(scar['points'], contour), debug_info

        debug_info['error'] = 'no_valid_pairs'
        return None, debug_info

    # ── Private helpers ──────────────────────────────────────────────────────

    def _score(self, i1, i2, kappa, long_norm, search_mode):
        """
        Pair quality score.
        In 'full' mode we upweight pairs away from the cell centre because
        the most recent scar is the one that is most distal from the centroid.
        """
        base = float(kappa[i1]) + float(kappa[i2])
        if search_mode == 'full':
            avg_pos = (long_norm[i1] + long_norm[i2]) / 2.0
            weight  = 1.0 + 4.0 * abs(avg_pos - 0.5)   # 1.0 at centre, 3.0 at poles
            return base * weight
        return base

    def _find_peaks(self, kappa, valid_indices):
        """Local maxima of curvature within valid_indices, positive only."""
        d_kappa   = np.gradient(kappa)
        sign_diff = np.diff(np.sign(d_kappa))
        all_peaks = np.where(sign_diff < 0)[0]
        valid_set = set(valid_indices.tolist())
        return np.array(
            [p for p in all_peaks if p in valid_set and kappa[p] > 0], dtype=int)

    def _is_valid_scar_vector(self, pt1, pt2, centre, axis, normal_vec, min_width):
        """
        Return (is_valid, scar_width) for a candidate scar pt1→pt2.
        Checks: opposite sides of cell, minimum width, perpendicularity.
        """
        # Must be on opposite sides of the long axis
        side1 = np.dot(pt1 - centre, normal_vec)
        side2 = np.dot(pt2 - centre, normal_vec)
        if side1 * side2 >= 0:
            return False, 0.0

        scar_vec = pt2 - pt1
        width    = float(np.linalg.norm(scar_vec))
        if width < min_width:
            return False, width

        scar_unit = scar_vec / width
        angle_deg = np.degrees(
            np.arccos(np.clip(np.abs(np.dot(scar_unit, axis)), 0.0, 1.0)))
        if abs(90.0 - angle_deg) >= self.cfg.MAX_ANGLE_DEVIATION:
            return False, width

        return True, width

    def _find_strict_pair(self, peaks, smooth_pts, kappa, centre, axis,
                           normal_vec, long_proj, min_width, long_norm, search_mode):
        candidates = []
        for i in range(len(peaks)):
            for j in range(i + 1, len(peaks)):
                p1, p2 = peaks[i], peaks[j]
                pt1, pt2 = smooth_pts[p1], smooth_pts[p2]
                valid, _ = self._is_valid_scar_vector(
                    pt1, pt2, centre, axis, normal_vec, min_width)
                if valid:
                    candidates.append(dict(
                        indices=(p1, p2), points=(pt1, pt2),
                        score=self._score(p1, p2, kappa, long_norm, search_mode)))

        return max(candidates, key=lambda x: x['score']) if candidates else None

    def _find_asymmetric_pair(self, peaks, valid_indices, smooth_pts, kappa,
                               centre, axis, normal_vec, long_proj, long_norm,
                               min_width, search_mode):
        candidates = []
        for p_strong in peaks:
            pt_strong  = smooth_pts[p_strong]
            side_strong = np.sign(np.dot(pt_strong - centre, normal_vec))
            long_pos    = long_norm[p_strong]

            # Candidate mirror points: opposite side, same longitudinal position ±5%
            partners = [
                idx for idx in valid_indices
                if np.sign(np.dot(smooth_pts[idx] - centre, normal_vec)) != side_strong
                and abs(long_norm[idx] - long_pos) <= 0.05
            ]
            if not partners:
                continue

            best_weak = max(partners, key=lambda i: kappa[i])
            pt_weak   = smooth_pts[best_weak]
            valid, _  = self._is_valid_scar_vector(
                pt_strong, pt_weak, centre, axis, normal_vec, min_width)
            if valid:
                candidates.append(dict(
                    indices=(p_strong, best_weak),
                    points=(pt_strong, pt_weak),
                    score=self._score(p_strong, best_weak, kappa, long_norm, search_mode)))

        return max(candidates, key=lambda x: x['score']) if candidates else None

    def _map_to_original(self, smooth_points, original_contour):
        """Snap smoothed scar points back onto the original contour."""
        orig = np.array(original_contour)
        def closest(target):
            return orig[np.argmin(np.linalg.norm(orig - target, axis=1))]
        return (closest(smooth_points[0]), closest(smooth_points[1]))
