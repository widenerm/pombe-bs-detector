"""
tracking.py  –  Frame-to-frame cell tracking + lineage naming.

Cell naming scheme
──────────────────
Initial cells  :  A, B, C, … Z, AA, AB, … (alphabetical, encoding order of
                  first appearance)
After division :  Parent "X" → daughters "X0" (new-end daughter, kept the
                  division-site pole) and "X1" (old-end daughter, kept the
                  parent's pre-existing old pole).
Subsequent     :  X0 → X00 / X01,  X1 → X10 / X11,  etc.

Tracking cost  (Hungarian assignment)
──────────────────────────────────────
  cost = w_dist   × (Δcentre / MAX_TRACKING_DISTANCE)
       + w_area   × |ΔArea|  / max(area1, area2)
       + w_curv   × ‖fingerprint_diff‖

Matches whose cost exceeds MATCH_THRESHOLD are rejected (cell lost / born).
Division detection: an unmatched previous cell whose 'footprint' is covered
by two unmatched current cells whose areas sum to ≈ parent area.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

MATCH_THRESHOLD = 1.5   # reject assignments above this normalised cost


class CellTracker:

    def __init__(self, config):
        self.cfg            = config
        self._name_counter  = 0
        self.active_tracks  = {}   # seg_label → state_dict
        self.lineage_log    = []   # list of division events
        self._current_frame = 0

    # ── Name generation ──────────────────────────────────────────────────────

    def _next_base_name(self):
        """A … Z, AA, AB … AZ, BA …"""
        idx = self._name_counter
        self._name_counter += 1
        if idx < 26:
            return chr(65 + idx)
        # Two-letter names (supports up to 702 initial cells)
        first  = (idx - 26) // 26
        second = (idx - 26) %  26
        return chr(65 + first) + chr(65 + second)

    @staticmethod
    def daughter_names(parent_name):
        """Return (new_end_name, old_end_name)."""
        return parent_name + '0', parent_name + '1'

    # ── Cost ─────────────────────────────────────────────────────────────────

    def _cost(self, prev, curr):
        cfg = self.cfg

        # Distance (normalised; > 1.0 → exceeds max allowed displacement)
        d = np.linalg.norm(np.array(prev['centroid']) - np.array(curr['centroid']))
        dist_cost = d / (cfg.MAX_TRACKING_DISTANCE + 1e-9)
        if dist_cost >= 1.0:
            return float('inf')   # too far → never match

        # Area change
        a1, a2    = prev['area'], curr['area']
        area_cost = abs(a1 - a2) / (max(a1, a2) + 1e-9)

        # Curvature fingerprint
        f1, f2    = prev.get('fingerprint'), curr.get('fingerprint')
        curv_cost = float(np.linalg.norm(f1 - f2)) if (f1 is not None and f2 is not None) else 0.0

        return (cfg.COST_WEIGHT_DISTANCE  * dist_cost
                + cfg.COST_WEIGHT_AREA    * area_cost
                + cfg.COST_WEIGHT_CURVATURE * curv_cost)

    # ── Public ───────────────────────────────────────────────────────────────

    def update(self, current_results, frame_idx=None):
        """
        Match *current_results* to existing tracks via Hungarian algorithm.

        Returns dict  {seg_label → cell_name}  for this frame.
        """
        if frame_idx is not None:
            self._current_frame = frame_idx

        if not self.active_tracks:
            return self._initialize(current_results)

        prev_labels  = list(self.active_tracks.keys())
        curr_by_lbl  = {r['label']: r for r in current_results}
        curr_labels  = list(curr_by_lbl.keys())
        n_p, n_c     = len(prev_labels), len(curr_labels)

        # Build cost matrix
        INF          = 1e9
        cost_matrix  = np.full((n_p, n_c), INF)
        for i, pl in enumerate(prev_labels):
            for j, cl in enumerate(curr_labels):
                c = self._cost(self.active_tracks[pl], curr_by_lbl[cl])
                if c < INF:
                    cost_matrix[i, j] = c

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assigned_prev  = set()
        assigned_curr  = set()
        new_tracks     = {}

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] >= MATCH_THRESHOLD:
                continue
            pl, cl = prev_labels[r], curr_labels[c]
            name   = self.active_tracks[pl]['name']
            new_tracks[cl] = self._make_state(curr_by_lbl[cl], name)
            assigned_prev.add(r)
            assigned_curr.add(c)

        # Unmatched cells
        unmatched_prev = [prev_labels[r] for r in range(n_p) if r not in assigned_prev]
        unmatched_curr = [curr_labels[c] for c in range(n_c) if c not in assigned_curr]

        # Division detection
        self._handle_divisions(unmatched_prev, unmatched_curr,
                                curr_by_lbl, new_tracks, assigned_curr)

        # Any remaining unmatched current cells are genuinely new
        for cl in curr_labels:
            if cl not in new_tracks:
                new_tracks[cl] = self._make_state(curr_by_lbl[cl], self._next_base_name())

        self.active_tracks = new_tracks
        return {lbl: new_tracks[lbl]['name'] for lbl in new_tracks}

    # ── Private ──────────────────────────────────────────────────────────────

    def _initialize(self, current_results):
        for r in current_results:
            name = self._next_base_name()
            self.active_tracks[r['label']] = self._make_state(r, name)
        return {lbl: st['name'] for lbl, st in self.active_tracks.items()}

    def _make_state(self, cell_result, name):
        dbg = cell_result.get('debug_info', {})
        return {
            'name':        name,
            'centroid':    cell_result.get('centroid', (0, 0)),
            'area':        cell_result.get('area', 0),
            'fingerprint': dbg.get('curvature_fingerprint'),
            'new_pole':    dbg.get('new_pole_point'),
            'old_pole':    dbg.get('old_pole_point'),
        }

    def _handle_divisions(self, unmatched_prev_labels, unmatched_curr_labels,
                           curr_by_lbl, new_tracks, assigned_curr):
        """
        For each unmatched previous cell, test whether two unmatched current
        cells could be its daughters.  The daughter closer to the parent's
        old pole inherits the '1' suffix; the other inherits '0'.
        """
        cfg       = self.cfg
        used_curr = set()

        for pl in unmatched_prev_labels:
            parent       = self.active_tracks[pl]
            p_centroid   = np.array(parent['centroid'])
            p_area       = parent['area']

            # Candidate daughters: nearby and large enough
            cands = [
                cl for cl in unmatched_curr_labels
                if cl not in used_curr
                and np.linalg.norm(
                    np.array(curr_by_lbl[cl].get('centroid', (0,0))) - p_centroid
                ) < cfg.MAX_TRACKING_DISTANCE
                and curr_by_lbl[cl].get('area', 0) / max(p_area, 1) >= cfg.DIVISION_AREA_RATIO
            ]

            if len(cands) < 2:
                continue

            best = self._best_division_pair(cands, parent, curr_by_lbl, p_area)
            if best is None:
                continue

            d0_lbl, d1_lbl = best   # d0 = new-end ('0'), d1 = old-end ('1')
            d0_name, d1_name = self.daughter_names(parent['name'])

            new_tracks[d0_lbl] = self._make_state(curr_by_lbl[d0_lbl], d0_name)
            new_tracks[d1_lbl] = self._make_state(curr_by_lbl[d1_lbl], d1_name)
            assigned_curr.update([d0_lbl, d1_lbl])
            used_curr.update([d0_lbl, d1_lbl])

            self.lineage_log.append({
                'frame':     self._current_frame,
                'parent':    parent['name'],
                'daughters': [d0_name, d1_name],
            })
            print(f"  Division detected: {parent['name']} → {d0_name}, {d1_name}")

    def _best_division_pair(self, candidates, parent, curr_by_lbl, parent_area):
        """
        Among all pairs of candidates, return the pair whose combined area
        is closest to parent_area, broken by old-pole proximity for '0'/'1'.
        """
        best_score = float('inf')
        best_pair  = None

        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                a_lbl, b_lbl = candidates[i], candidates[j]
                a_area = curr_by_lbl[a_lbl].get('area', 0)
                b_area = curr_by_lbl[b_lbl].get('area', 0)
                score  = abs((a_area + b_area) - parent_area)

                if score < best_score:
                    best_score = score
                    # Assign '1' to the daughter closer to parent's old pole
                    old_pole = parent.get('old_pole')
                    if old_pole is not None:
                        op = np.array(old_pole)
                        da = np.linalg.norm(
                            np.array(curr_by_lbl[a_lbl].get('centroid', (0,0))) - op)
                        db = np.linalg.norm(
                            np.array(curr_by_lbl[b_lbl].get('centroid', (0,0))) - op)
                        # closer to old pole → '1', further → '0'
                        best_pair = (b_lbl, a_lbl) if da < db else (a_lbl, b_lbl)
                    else:
                        best_pair = (a_lbl, b_lbl)

        return best_pair
