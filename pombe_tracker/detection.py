"""
detection.py  –  BirthScarDetector

Strategy
────────
The full cell contour is always searched.  Curvature is accumulated in a
longitudinal window on both sides of the PCA axis, so a scar does not need to
produce two independent local maxima.  The first and last cap regions are
excluded because normal pole rounding otherwise dominates scar-free cells.

Geometric constraints suppress false positives at the poles:

  1. WIDTH   The scar vector must span ≥ MIN_SCAR_WIDTH_RATIO × average
             mid-cell width (seven cross-sections at 20–80 % of cell length).
             Average width is more stable than max width, which can be
             inflated by the scar bulge itself.

  2. CAPS    Windows centred in the first/last SCAR_CAP_EXCLUSION fraction
             of the long axis are never considered.
  3. ANGLE   The scar vector receives a soft perpendicularity weight, allowing
             genuinely diagonal scars without making pole candidates compete.

Scoring
───────
score = windowed_prominence_sum × angle_weight

  windowed_prominence_sum : positive curvature above a local baseline,
                            accumulated over a longitudinal window on both
                            sides of the cell.

  angle_weight   : square-rooted perpendicularity score.  Cap exclusion is
                   applied before scoring, not as a soft penalty.

All valid candidates are stored in debug_info['scar_candidates'] for the
postprocessing consensus pass.
"""
import numpy as np
from .geometry import compute_smoothed_curvature, compute_pca_axis

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

        rel        = smooth_pts - center
        long_proj  = rel @ axis
        rng        = long_proj.max() - long_proj.min()
        long_norm  = (long_proj - long_proj.min()) / (rng + 1e-10)
        transverse = rel @ normal_vec

        max_thickness = np.percentile(transverse, 98) - np.percentile(transverse, 2)

        # ── Average mid-cell width ────────────────────────────────────────────
        sample_norms  = np.linspace(0.2, 0.8, 7)
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
            'display_mask':  np.ones(len(smooth_pts), dtype=bool),
        }

        peaks = self._find_peaks(kappa)
        debug_info['peaks'] = peaks

        if len(peaks) == 0:
            debug_info['error']           = 'no_peaks'
            debug_info['scar_candidates'] = []
            return None, debug_info

        all_cands = self._collect_windowed_candidates(
            smooth_pts, kappa, center, axis, normal_vec, min_scar_width, long_norm)
        for c in all_cands:
            c['match_type'] = 'windowed'

        debug_info['scar_candidates'] = [
            {'points': c['points'], 'score': c['score'], 'match_type': c['match_type']}
            for c in all_cands
        ]

        if not all_cands:
            debug_info['error'] = 'no_valid_pairs'
            return None, debug_info

        best = self._select_best_candidate(all_cands, new_pole_point)

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
        Local prominence = kappa[peak] − mean(kappa in ±PROMINENCE_WINDOW ring).

        Rewards sharp localised bumps (birth-scar ridges) over broad flat
        regions that happen to have a slightly elevated absolute curvature.
        """
        n       = len(kappa)
        indices = [(peak_idx + d) % n
                   for d in range(-PROMINENCE_WINDOW, PROMINENCE_WINDOW + 1)
                   if d != 0]
        baseline = float(np.mean(kappa[indices]))
        return max(0.0, float(kappa[peak_idx]) - baseline)

    def _perp_score(self, pt1, pt2, axis):
        """
        Perpendicularity of the scar vector relative to the cell long axis.
        Returns 1.0 for perfectly perpendicular, 0.0 for parallel.
        """
        scar_vec = np.array(pt2) - np.array(pt1)
        norm     = float(np.linalg.norm(scar_vec))
        if norm < 1e-6:
            return 0.0
        return 1.0 - float(np.abs(np.dot(scar_vec / norm, axis)))

    def _local_curvature_excess(self, kappa):
        """Return positive curvature above a circular local median baseline."""
        n = len(kappa)
        excess = np.zeros(n, dtype=float)
        for idx in range(n):
            offsets = [(idx + d) % n for d in range(-PROMINENCE_WINDOW,
                                                       PROMINENCE_WINDOW + 1)
                       if d != 0]
            baseline = float(np.median(kappa[offsets]))
            excess[idx] = max(0.0, float(kappa[idx]) - baseline)
        return excess

    def _score(self, i1, i2, kappa, smooth_pts, axis, long_norm=None):
        """
        Combined score = prominence_sum × (1 + perp_score).

        The perp bonus rewards the pair that is most perpendicular to the
        cell long axis, providing fine-grained location accuracy on top of
        the coarse prominence-based region selection.
        long_norm is accepted for API compatibility but not used directly.
        """
        prom = self._peak_prominence(kappa, i1) + self._peak_prominence(kappa, i2)
        perp = self._perp_score(smooth_pts[i1], smooth_pts[i2], axis)
        return prom * (1.0 + perp)

    def _collect_windowed_candidates(self, smooth_pts, kappa, center, axis,
                                     normal_vec, min_width, long_norm):
        """Collect candidates by integrating curvature in longitudinal windows.

        Unlike peak pairing, each side contributes independently.  This makes
        a strong feature on only one side sufficient, while the cap exclusion
        prevents ordinary pole curvature from winning on scar-free cells.
        """
        n = len(smooth_pts)
        side = np.sign((smooth_pts - center) @ normal_vec)
        excess = self._local_curvature_excess(kappa)
        window = float(getattr(self.cfg, 'SCAR_CURVATURE_WINDOW', 0.08))
        half_window = window / 2.0
        cap = float(getattr(self.cfg, 'SCAR_CAP_EXCLUSION', 0.12))
        max_offset = float(getattr(self.cfg, 'SCAR_MAX_LONGITUDINAL_OFFSET', 0.08))
        candidates = []

        # Use a modest number of overlapping windows.  This gives the
        # stabilizer several nearby alternatives without emitting one
        # candidate for every contour sample.
        centers = np.linspace(cap, 1.0 - cap, max(12, n // 12))
        for target in centers:
            eligible = ((long_norm >= cap) & (long_norm <= 1.0 - cap))
            near = eligible & (np.abs(long_norm - target) <= half_window)
            if not near.any():
                continue

            side_scores = []
            side_indices = []
            for sign in (-1.0, 1.0):
                members = np.flatnonzero(near & (side == sign))
                if len(members) == 0:
                    side_scores.append(0.0)
                    side_indices.append(None)
                    continue
                # Mean avoids a sampling-density bias; summing the two side
                # means retains the intended "either side can be strong"
                # behavior while keeping scores comparable across windows.
                side_scores.append(float(np.mean(excess[members])))
                side_indices.append(int(members[np.argmax(excess[members])]))

            if side_indices[0] is None or side_indices[1] is None:
                continue
            p1, p2 = side_indices
            if abs(long_norm[p1] - long_norm[p2]) > max_offset:
                continue
            valid, width = self._is_valid_scar_vector(
                smooth_pts[p1], smooth_pts[p2], center, axis, normal_vec,
                min_width, enforce_angle=False)
            if not valid:
                continue

            scar_vec = smooth_pts[p2] - smooth_pts[p1]
            angle_deg = np.degrees(np.arccos(np.clip(
                np.abs(np.dot(scar_vec / np.linalg.norm(scar_vec), axis)),
                0.0, 1.0)))
            angle_deviation = abs(90.0 - angle_deg)
            angle_scale = float(getattr(self.cfg, 'MAX_ANGLE_DEVIATION', 30.0))
            angle_weight = np.exp(-0.5 * (angle_deviation / angle_scale) ** 2)
            score = (side_scores[0] + side_scores[1]) * angle_weight
            # A geometrically valid cross-section is not automatically a
            # scar.  In particular, a smooth scar-free rod should not yield
            # an arbitrary zero-score candidate after cap suppression.
            if score <= 1e-10:
                continue
            candidates.append(dict(
                indices=(p1, p2),
                points=(smooth_pts[p1], smooth_pts[p2]),
                score=float(score),
                window_center=float(target),
                window_curvature=float(side_scores[0] + side_scores[1]),
                angle_weight=float(angle_weight),
                width=float(width),
            ))
        return candidates

    def _is_valid_scar_vector(self, pt1, pt2, center, axis, normal_vec,
                              min_width, enforce_angle=True):
        """
        Return (is_valid, scar_width) for a candidate scar pt1 → pt2.
        Checks: opposite sides, minimum width vs average, perpendicularity.
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
        if enforce_angle and abs(90.0 - angle_deg) >= self.cfg.MAX_ANGLE_DEVIATION:
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
                        score   = self._score(p1, p2, kappa, smooth_pts, axis, long_norm),
                    ))
        return candidates

    def _collect_asymmetric_candidates(self, peaks, smooth_pts, kappa, center, axis,
                                        normal_vec, long_norm, min_width):
        """
        Asymmetric candidates: one strong curvature peak paired with the
        most-perpendicular point on the opposite side at the same longitudinal
        position (within ±5 % of normalized cell length).

        Partner selection uses perpendicularity (not raw κ) so that the
        selected pair is as geometrically accurate as possible.
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

            # Pick partner that makes the most perpendicular scar vector
            def perp_for_partner(idx):
                return self._perp_score(pt_strong, smooth_pts[idx], axis)

            best_weak = max(partners, key=perp_for_partner)
            pt_weak   = smooth_pts[best_weak]
            valid, _  = self._is_valid_scar_vector(
                pt_strong, pt_weak, center, axis, normal_vec, min_width)
            if valid:
                candidates.append(dict(
                    indices = (p_strong, best_weak),
                    points  = (pt_strong, pt_weak),
                    score   = self._score(p_strong, best_weak, kappa, smooth_pts, axis, long_norm),
                ))
        return candidates

    def _select_best_candidate(self, candidates, new_pole_point):
        """
        Select the best candidate.

        With a new-pole hint, spatial proximity remains authoritative. Without
        one, the highest windowed score wins.
        """
        def midpoint(c):
            return (np.array(c['points'][0]) + np.array(c['points'][1])) / 2.0

        if new_pole_point is not None:
            np_arr    = np.array(new_pole_point)
            if candidates:
                return min(candidates,
                           key=lambda c: float(np.linalg.norm(midpoint(c) - np_arr)))
        else:
            if candidates:
                return max(candidates, key=lambda x: x['score'])

        raise RuntimeError('_select_best_candidate called with no candidates')

    def _map_to_original(self, smooth_points, original_contour):
        """Snap smoothed scar points back onto the original contour."""
        orig = np.array(original_contour)

        def closest(target):
            return orig[np.argmin(np.linalg.norm(orig - target, axis=1))]

        return (closest(smooth_points[0]), closest(smooth_points[1]))
