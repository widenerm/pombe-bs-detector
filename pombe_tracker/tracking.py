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
  cost = w_dist   × (Δcenter / MAX_TRACKING_DISTANCE)
       + w_area   × |ΔArea|  / max(area1, area2)
       + w_curv   × ‖fingerprint_diff‖

Matches whose cost exceeds MATCH_THRESHOLD are rejected (cell lost / born).

Division detection
──────────────────
The Hungarian algorithm greedily matches a dividing parent to one of its
daughters (the closer one), so the parent never appears in `unmatched_prev`
and the second daughter gets a spurious base name.

Fix: after Hungarian matching, scan every accepted match where the current
cell's area is substantially smaller than the previous cell's area
(< DIVISION_SUSPICION_RATIO × prev_area).  If a second unmatched current
cell is also nearby and the two daughters together recover the parent area,
the match is reclassified as a division: the greedy match is reverted and
both daughters are routed through _handle_divisions.

Ghost track matching
────────────────────
When Cellpose produces a bad segmentation for one or two frames, the
Hungarian algorithm cannot match the fragment and the original cell
reappears under a new base name when the correct segmentation returns.

Fix: a _lost_tracks buffer stores state dicts for recently disappeared
tracks for up to GHOST_FRAMES (default 3) frames.  Before minting a new
base name for an unmatched current cell, _match_ghost_tracks() checks
whether any lost track is compatible by centroid proximity, area, and
curvature fingerprint.  If a match is found, the track is resumed under
its original name.

Lineage pole information
────────────────────────
Each division event recorded in lineage_log now includes the parent's last
known new_pole and old_pole coordinates.  The pipeline uses these to
spatially determine the correct new / old poles for each daughter immediately
after division, before any scar-based inference runs.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

MATCH_THRESHOLD = 1.5

DIVISION_SUSPICION_RATIO = 0.75


class CellTracker:

    def __init__(self, config):
        self.cfg            = config
        self._name_counter  = 0
        self.active_tracks  = {}
        self.lineage_log    = []
        self._current_frame = 0
        self._lost_tracks   = []

    # ── Name generation ──────────────────────────────────────────────────────

    def _next_base_name(self):
        """A … Z, AA, AB … AZ, BA …"""
        idx = self._name_counter
        self._name_counter += 1
        if idx < 26:
            return chr(65 + idx)
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

        d = np.linalg.norm(np.array(prev['centroid']) - np.array(curr['centroid']))
        dist_cost = d / (cfg.MAX_TRACKING_DISTANCE + 1e-9)
        if dist_cost >= 1.0:
            return float('inf')

        a1, a2    = prev['area'], curr['area']
        area_cost = abs(a1 - a2) / (max(a1, a2) + 1e-9)

        f1, f2    = prev.get('fingerprint'), curr.get('fingerprint')
        curv_cost = float(np.linalg.norm(f1 - f2)) if (f1 is not None and f2 is not None) else 0.0

        return (cfg.COST_WEIGHT_DISTANCE   * dist_cost
                + cfg.COST_WEIGHT_AREA     * area_cost
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

        # ── Build cost matrix ────────────────────────────────────────────────
        INF          = 1e9
        cost_matrix  = np.full((n_p, n_c), INF)
        for i, pl in enumerate(prev_labels):
            for j, cl in enumerate(curr_labels):
                c = self._cost(self.active_tracks[pl], curr_by_lbl[cl])
                if c < INF:
                    cost_matrix[i, j] = c

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        new_tracks      = {}
        assigned_prev   = set()
        assigned_curr   = set()
        suspicious      = []

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] >= MATCH_THRESHOLD:
                continue
            pl  = prev_labels[r]
            cl  = curr_labels[c]
            prev_area = self.active_tracks[pl]['area']
            curr_area = curr_by_lbl[cl].get('area', 0)
            if prev_area > 0 and curr_area / prev_area < DIVISION_SUSPICION_RATIO:
                suspicious.append((r, c))
            else:
                name = self.active_tracks[pl]['name']
                new_tracks[cl] = self._make_state(curr_by_lbl[cl], name)
                assigned_prev.add(r)
                assigned_curr.add(c)

        # ── Division re-check for suspicious matches ─────────────────────────
        unmatched_curr_so_far = {curr_labels[c] for c in range(n_c)
                                 if c not in assigned_curr}

        for r, c in suspicious:
            pl = prev_labels[r]
            cl = curr_labels[c]
            parent     = self.active_tracks[pl]
            p_area     = parent['area']
            p_centroid = np.array(parent['centroid'])

            partner_candidates = [
                lbl for lbl in unmatched_curr_so_far
                if lbl != cl
                and np.linalg.norm(
                    np.array(curr_by_lbl[lbl].get('centroid', (0, 0))) - p_centroid
                ) < self.cfg.MAX_TRACKING_DISTANCE
                and curr_by_lbl[lbl].get('area', 0) / max(p_area, 1) >= self.cfg.DIVISION_AREA_RATIO
            ]

            d1_area       = curr_by_lbl[cl].get('area', 0)
            found_partner = None
            best_score    = float('inf')
            for partner in partner_candidates:
                d2_area = curr_by_lbl[partner].get('area', 0)
                score   = abs((d1_area + d2_area) - p_area)
                if score < best_score:
                    best_score    = score
                    found_partner = partner

            if found_partner is not None:
                self._record_division(
                    pl, [cl, found_partner], parent, curr_by_lbl,
                    new_tracks, assigned_curr)
                assigned_prev.add(r)
                unmatched_curr_so_far.discard(cl)
                unmatched_curr_so_far.discard(found_partner)
            else:
                name = parent['name']
                new_tracks[cl] = self._make_state(curr_by_lbl[cl], name)
                assigned_prev.add(r)
                assigned_curr.add(c)
                unmatched_curr_so_far.discard(cl)

        unmatched_prev = [prev_labels[r] for r in range(n_p) if r not in assigned_prev]
        unmatched_curr = list(unmatched_curr_so_far)

        self._handle_divisions(unmatched_prev, unmatched_curr,
                               curr_by_lbl, new_tracks, assigned_curr)

        # ── Expire active tracks that were not matched ────────────────────────
        for pl in unmatched_prev:
            state = self.active_tracks[pl]
            self._lost_tracks.append({
                'state':      state,
                'lost_frame': self._current_frame,
            })

        # ── Ghost track matching ──────────────────────────────────────────────
        ghost_frames = getattr(self.cfg, 'GHOST_FRAMES', 3)
        self._lost_tracks = [
            lt for lt in self._lost_tracks
            if self._current_frame - lt['lost_frame'] <= ghost_frames
        ]

        still_unmatched = [cl for cl in curr_labels
                           if cl not in new_tracks and cl not in assigned_curr]
        self._match_ghost_tracks(still_unmatched, curr_by_lbl, new_tracks)

        for cl in curr_labels:
            if cl not in new_tracks:
                new_tracks[cl] = self._make_state(curr_by_lbl[cl], self._next_base_name())

        self.active_tracks = new_tracks
        return {lbl: new_tracks[lbl]['name'] for lbl in new_tracks}

    # ── Ghost track matching ─────────────────────────────────────────────────

    def _match_ghost_tracks(self, unmatched_curr_labels, curr_by_lbl, new_tracks):
        if not unmatched_curr_labels or not self._lost_tracks:
            return

        cfg               = self.cfg
        fp_threshold      = getattr(cfg, 'GHOST_FINGERPRINT_THRESHOLD', 1.0)
        used_ghost_indices = set()

        for cl in unmatched_curr_labels:
            curr     = curr_by_lbl[cl]
            c_center = np.array(curr.get('centroid', (0, 0)))
            c_area   = curr.get('area', 0)
            c_fp     = curr.get('debug_info', {}).get('curvature_fingerprint')

            best_cost = float('inf')
            best_idx  = None

            for gi, lt in enumerate(self._lost_tracks):
                if gi in used_ghost_indices:
                    continue
                state    = lt['state']
                g_center = np.array(state['centroid'])
                g_area   = state['area']
                g_fp     = state.get('fingerprint')

                dist = float(np.linalg.norm(c_center - g_center))
                if dist >= cfg.MAX_TRACKING_DISTANCE:
                    continue

                area_ratio = c_area / max(g_area, 1)
                if (area_ratio < cfg.DIVISION_AREA_RATIO * 2
                        or area_ratio > 1.0 / (cfg.DIVISION_AREA_RATIO * 2)):
                    continue

                dist_cost = dist / (cfg.MAX_TRACKING_DISTANCE + 1e-9)
                area_cost = abs(c_area - g_area) / (max(c_area, g_area) + 1e-9)

                if c_fp is not None and g_fp is not None:
                    fp_dist = float(np.linalg.norm(c_fp - g_fp))
                    if fp_dist > fp_threshold:
                        continue
                    curv_cost = fp_dist
                else:
                    curv_cost = 0.0

                cost = (cfg.COST_WEIGHT_DISTANCE    * dist_cost
                        + cfg.COST_WEIGHT_AREA      * area_cost
                        + cfg.COST_WEIGHT_CURVATURE * curv_cost)

                if cost < best_cost:
                    best_cost = cost
                    best_idx  = gi

            if best_idx is not None and best_cost < MATCH_THRESHOLD:
                resumed_state = self._lost_tracks[best_idx]['state']
                new_tracks[cl] = self._make_state(curr, resumed_state['name'])
                used_ghost_indices.add(best_idx)
                print(f"  Ghost match: resumed '{resumed_state['name']}' "
                      f"(lost {self._current_frame - self._lost_tracks[best_idx]['lost_frame']} "
                      f"frame(s) ago)")

        self._lost_tracks = [lt for i, lt in enumerate(self._lost_tracks)
                             if i not in used_ghost_indices]

    # ── Division helpers ──────────────────────────────────────────────────────

    def _record_division(self, parent_label, daughter_labels, parent,
                         curr_by_lbl, new_tracks, assigned_curr):
        """
        Assign daughter names to two current cells that together constitute
        a division of *parent*.

        The parent's new_pole and old_pole are stored in the lineage_log entry
        so the pipeline can use them to determine the correct pole assignment
        for each daughter immediately after division.
        """
        d0_lbl, d1_lbl = self._assign_daughter_order(
            daughter_labels, parent, curr_by_lbl)
        d0_name, d1_name = self.daughter_names(parent['name'])

        new_tracks[d0_lbl] = self._make_state(curr_by_lbl[d0_lbl], d0_name)
        new_tracks[d1_lbl] = self._make_state(curr_by_lbl[d1_lbl], d1_name)
        assigned_curr.update([d0_lbl, d1_lbl])

        # Store parent pole coordinates so the pipeline can spatially resolve
        # the new / old poles of each daughter without re-running detection.
        parent_old_pole = parent.get('old_pole')
        parent_new_pole = parent.get('new_pole')
        if parent_old_pole is not None:
            parent_old_pole = np.array(parent_old_pole)
        if parent_new_pole is not None:
            parent_new_pole = np.array(parent_new_pole)

        self.lineage_log.append({
            'frame':           self._current_frame,
            'parent':          parent['name'],
            'daughters':       [d0_name, d1_name],
            'parent_old_pole': parent_old_pole,
            'parent_new_pole': parent_new_pole,
        })
        print(f"  Division detected: {parent['name']} → {d0_name}, {d1_name}")

    def _assign_daughter_order(self, candidate_labels, parent, curr_by_lbl):
        """
        Return (new_end_label, old_end_label).
        The daughter closer to the parent's old pole gets the '1' (old-end) suffix.
        Falls back to arbitrary ordering if old pole is unavailable.
        """
        if len(candidate_labels) != 2:
            return candidate_labels[0], candidate_labels[1]

        la, lb   = candidate_labels[0], candidate_labels[1]
        old_pole = parent.get('old_pole')

        if old_pole is not None:
            op  = np.array(old_pole)
            ca  = np.array(curr_by_lbl[la].get('centroid', (0, 0)))
            cb  = np.array(curr_by_lbl[lb].get('centroid', (0, 0)))
            if np.linalg.norm(ca - op) < np.linalg.norm(cb - op):
                return lb, la
            else:
                return la, lb
        return la, lb

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
        cells could be its daughters (area sum ≈ parent area, both nearby).
        """
        cfg       = self.cfg
        used_curr = set()

        for pl in unmatched_prev_labels:
            parent      = self.active_tracks[pl]
            p_centroid  = np.array(parent['centroid'])
            p_area      = parent['area']

            cands = [
                cl for cl in unmatched_curr_labels
                if cl not in used_curr
                and np.linalg.norm(
                    np.array(curr_by_lbl[cl].get('centroid', (0, 0))) - p_centroid
                ) < cfg.MAX_TRACKING_DISTANCE
                and curr_by_lbl[cl].get('area', 0) / max(p_area, 1) >= cfg.DIVISION_AREA_RATIO
            ]

            if len(cands) < 2:
                continue

            best = self._best_division_pair(cands, parent, curr_by_lbl, p_area)
            if best is None:
                continue

            self._record_division(pl, list(best), parent, curr_by_lbl,
                                  new_tracks, assigned_curr)
            used_curr.update(best)

    def _best_division_pair(self, candidates, parent, curr_by_lbl, parent_area):
        best_score = float('inf')
        best_pair  = None

        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                a_lbl  = candidates[i]
                b_lbl  = candidates[j]
                a_area = curr_by_lbl[a_lbl].get('area', 0)
                b_area = curr_by_lbl[b_lbl].get('area', 0)
                score  = abs((a_area + b_area) - parent_area)

                if score < best_score:
                    best_score = score
                    best_pair  = (a_lbl, b_lbl)

        return best_pair
