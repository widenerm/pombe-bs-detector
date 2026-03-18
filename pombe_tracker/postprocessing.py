"""
postprocessing.py  –  Consensus-based temporal scar stabilisation.

Problem
───────
The curvature landscape of a cell changes slightly between frames due to
small segmentation differences and genuine cell shape changes.  The
top-scoring scar candidate can jump between frames even when the true birth
scar is not moving.

Strategy (replaces the old rolling-median approach)
────────────────────────────────────────────────────
All valid scar candidates are now stored per frame by the detector.  After
the full pipeline has run, for each tracked cell:

  1. Collect the normalised scar position of EVERY candidate across ALL frames.
  2. Find the consensus position: the normalised coordinate supported by the
     most candidates across the most frames (vote-count with a distance
     tolerance of SCAR_STABILITY_THRESHOLD).
  3. For each frame:
       a. If the already-selected candidate is within tolerance → 'raw'.
       b. If a different candidate is within tolerance → swap to it → 'corrected'.
       c. If no candidate matches but scar was detected → 'interpolated'.
       d. If scar was not detected at all → also 'interpolated' (if SCAR_INTERPOLATE).
  4. Interpolate all frames marked 'interpolated' by linearly blending
     scar_midpoint from the nearest valid neighbours.
  5. Write scar_source ('raw' | 'corrected' | 'interpolated' | 'no_detection')
     and scar_stable (bool) into each result dict so they appear in the CSV.

scar_stable = False when the fraction of corrected + interpolated frames
exceeds SCAR_INSTABILITY_FRACTION (default 0.30).
"""

import numpy as np


# ── Public entry point ────────────────────────────────────────────────────────

def stabilise_scars(all_results, config):
    """
    Run in-place consensus scar stabilisation on the output of run_pipeline.

    Parameters
    ----------
    all_results : list of frame dicts returned by run_pipeline
    config      : Config object

    Returns
    -------
    all_results          : same list, mutated in-place
    report               : dict  {cell_name: [event_dicts]}
    """
    threshold            = getattr(config, 'SCAR_STABILITY_THRESHOLD', 0.12)
    do_interp            = getattr(config, 'SCAR_INTERPOLATE',          True)
    instability_fraction = getattr(config, 'SCAR_INSTABILITY_FRACTION', 0.30)

    # ── Build per-cell timeline ───────────────────────────────────────────────
    cell_timeline = {}
    for fd in all_results:
        for r in fd['cells']:
            name = r.get('cell_name', '?')
            cell_timeline.setdefault(name, []).append((fd['frame_idx'], r))

    report = {}

    for name, timeline in cell_timeline.items():
        frame_indices = [t[0] for t in timeline]
        results       = [t[1] for t in timeline]

        if len(timeline) < 2:
            # Single-frame cell — nothing to stabilise
            for r in results:
                r.setdefault('scar_source',
                             'raw' if r['scar_detected'] else 'no_detection')
                r.setdefault('scar_stable', True)
            continue

        # ── Collect per-frame candidate pools ────────────────────────────────
        # frame_cands[i] = list of
        #   {'norm_pos': float, 'points': (pt1, pt2), 'score': float, 'match_type': str}
        frame_cands = []
        for r in results:
            dbg       = r.get('debug_info', {})
            raw_cands = dbg.get('scar_candidates', [])
            fc        = []

            for c in raw_cands:
                mp  = _midpoint_from_candidate(c)
                pos = _normalised_position_from_midpoint(r, mp)
                if pos is not None:
                    fc.append({
                        'norm_pos':   pos,
                        'points':     c['points'],
                        'score':      c['score'],
                        'match_type': c['match_type'],
                    })

            # Backward-compat: if scar_candidates absent (old data), use
            # the selected scar as the single candidate.
            if not raw_cands and r['scar_detected'] and r.get('scar_midpoint') is not None:
                pos = _normalised_scar_position(r)
                if pos is not None:
                    fc.append({
                        'norm_pos':   pos,
                        'points':     r.get('scar_points',
                                            (r['scar_midpoint'], r['scar_midpoint'])),
                        'score':      1.0,
                        'match_type': 'raw',
                    })

            frame_cands.append(fc)

        # ── Find consensus position ───────────────────────────────────────────
        all_positions = [fc['norm_pos'] for fcs in frame_cands for fc in fcs]

        if len(all_positions) < 2:
            for r in results:
                r.setdefault('scar_source',
                             'raw' if r['scar_detected'] else 'no_detection')
                r.setdefault('scar_stable', True)
            continue

        consensus_pos = _find_consensus_position(all_positions, threshold)

        if consensus_pos is None:
            # No clear consensus — mark all as raw but flag cell as unstable
            for r in results:
                r['scar_source'] = 'raw' if r['scar_detected'] else 'no_detection'
                r['scar_stable'] = False
            report[name] = [{'frame': fi, 'status': 'no_consensus'}
                             for fi in frame_indices]
            print(f"  Scar stabilisation – {name}: no consensus found, cell flagged unstable")
            continue

        # ── Enforce consensus on each frame ──────────────────────────────────
        cell_report = []
        n_changed   = 0

        for i, (fidx, r) in enumerate(zip(frame_indices, results)):
            fcs = frame_cands[i]

            if r['scar_detected']:
                curr_pos = _normalised_scar_position(r)

                if curr_pos is not None and abs(curr_pos - consensus_pos) <= threshold:
                    # Already consistent with consensus
                    r['scar_source']    = 'raw'
                    r['scar_stability'] = 'stable'
                    continue

                # Current selection deviates — try a better candidate
                near = [fc for fc in fcs
                        if abs(fc['norm_pos'] - consensus_pos) <= threshold]
                if near:
                    best      = max(near, key=lambda x: x['score'])
                    old_ratio = _ratio(r)
                    _apply_candidate(r, best)
                    r['scar_source']    = 'corrected'
                    r['scar_stability'] = 'corrected'
                    n_changed          += 1
                    cell_report.append({
                        'frame':     fidx,
                        'status':    'corrected',
                        'old_ratio': round(old_ratio, 3) if old_ratio is not None else None,
                        'new_ratio': round(_ratio(r),   3) if _ratio(r) is not None else None,
                    })
                else:
                    r['scar_source']    = 'interpolated'
                    r['scar_stability'] = 'suspect'
                    n_changed          += 1
                    cell_report.append({
                        'frame':     fidx,
                        'status':    'pre_interpolate',
                        'old_ratio': round(_ratio(r), 3) if _ratio(r) is not None else None,
                    })

            else:
                # No detection at all — look for any candidate near consensus
                near = [fc for fc in fcs
                        if abs(fc['norm_pos'] - consensus_pos) <= threshold]
                if near:
                    best = max(near, key=lambda x: x['score'])
                    _apply_candidate(r, best)
                    r['scar_source']    = 'corrected'
                    r['scar_stability'] = 'corrected'
                    n_changed          += 1
                    cell_report.append({
                        'frame':     fidx,
                        'status':    'recovered',
                        'new_ratio': round(_ratio(r), 3) if _ratio(r) is not None else None,
                    })
                else:
                    r['scar_source']    = 'interpolated'
                    r['scar_stability'] = 'no_detection'
                    cell_report.append({'frame': fidx, 'status': 'no_candidate'})

        # ── Interpolate remaining marked frames ───────────────────────────────
        if do_interp:
            interp_indices = [i for i, r in enumerate(results)
                              if r.get('scar_source') == 'interpolated']
            if interp_indices:
                _interpolate_frames(results, interp_indices)
                for i in interp_indices:
                    r = results[i]
                    r['scar_stability'] = 'interpolated'
                    # Update report entry status
                    for entry in cell_report:
                        if entry['frame'] == frame_indices[i] and entry['status'] in (
                                'pre_interpolate', 'no_candidate'):
                            entry['status']    = 'interpolated'
                            entry['new_ratio'] = (round(_ratio(r), 3)
                                                  if _ratio(r) is not None else None)

        # ── Cell-level stability flag ─────────────────────────────────────────
        stable = (n_changed / len(results)) < instability_fraction
        for r in results:
            r['scar_stable'] = stable
            r.setdefault('scar_source',
                         'raw' if r['scar_detected'] else 'no_detection')

        if cell_report:
            report[name] = cell_report
            status_counts = {}
            for e in cell_report:
                status_counts[e['status']] = status_counts.get(e['status'], 0) + 1
            print(f"  Scar stabilisation – {name}: {status_counts}"
                  f"  [{'stable' if stable else 'UNSTABLE'}]")

    # ── Ensure every result has scar_source / scar_stable ────────────────────
    for fd in all_results:
        for r in fd['cells']:
            r.setdefault('scar_source',
                         'raw' if r['scar_detected'] else 'no_detection')
            r.setdefault('scar_stable', True)

    return all_results, report


# ── Internal helpers ──────────────────────────────────────────────────────────

def _midpoint_from_candidate(candidate):
    pt1 = np.array(candidate['points'][0])
    pt2 = np.array(candidate['points'][1])
    return (pt1 + pt2) / 2.0


def _normalised_position_from_midpoint(result, midpoint):
    """
    Compute normalised scar position [0, 1] for an arbitrary midpoint,
    using the cell geometry already stored in debug_info.
    """
    dbg    = result.get('debug_info', {})
    centre = dbg.get('centre') or dbg.get('center')
    axis   = dbg.get('axis')
    ep1    = dbg.get('new_pole_point')
    ep2    = dbg.get('old_pole_point')

    if any(x is None for x in [centre, axis, ep1, ep2]) or midpoint is None:
        return None

    def proj(pt):
        return float(np.dot(np.array(pt) - np.array(centre), np.array(axis)))

    p_ep1 = proj(ep1)
    p_ep2 = proj(ep2)
    p_mp  = proj(midpoint)

    lo, hi = min(p_ep1, p_ep2), max(p_ep1, p_ep2)
    span   = hi - lo
    if span < 1e-6:
        return None

    return float(np.clip((p_mp - lo) / span, 0.0, 1.0))


def _normalised_scar_position(result):
    """
    Express the currently selected scar midpoint as a normalised position.
    Delegates to _normalised_position_from_midpoint.
    """
    mp = result.get('scar_midpoint')
    if mp is None:
        return None
    return _normalised_position_from_midpoint(result, np.array(mp))


def _find_consensus_position(positions, threshold):
    """
    Find the normalised scar position with the most support.

    Each position casts a vote for every position within *threshold* of
    itself.  The consensus is the median of the highest-supported cluster.
    Returns None if no cluster has at least 2 members.
    """
    if not positions:
        return None

    best_pos     = None
    best_support = 0

    for candidate in positions:
        supporters = [p for p in positions if abs(p - candidate) <= threshold]
        if len(supporters) > best_support:
            best_support = len(supporters)
            best_pos     = float(np.median(supporters))

    if best_support < 2:
        return None

    # One refinement pass: recompute median around the found centre
    supporters = [p for p in positions if abs(p - best_pos) <= threshold]
    if len(supporters) < 2:
        return None

    return float(np.median(supporters))


def _apply_candidate(result, candidate):
    """
    Replace the current scar detection with *candidate* and recompute
    all derived measurements.
    """
    pt1    = np.array(candidate['points'][0])
    pt2    = np.array(candidate['points'][1])
    new_mp = (pt1 + pt2) / 2.0

    result['scar_detected'] = True
    result['scar_points']   = (pt1, pt2)
    result['scar_midpoint'] = new_mp
    result['width_scar']    = float(np.linalg.norm(pt1 - pt2))

    dbg    = result.get('debug_info', {})
    np_pt  = dbg.get('new_pole_point')
    op_pt  = dbg.get('old_pole_point')

    if np_pt is not None and op_pt is not None:
        from .geometry import measure_pole_lengths
        new_len, old_len = measure_pole_lengths(
            new_mp, np.array(np_pt), np.array(op_pt))
        result['new_end_length'] = new_len
        result['old_end_length'] = old_len
        dbg['new_end_length']    = new_len
        dbg['old_end_length']    = old_len
        dbg['recent_scar']       = new_mp


def _interpolate_frames(results, suspect_indices):
    """
    For each suspect frame, linearly interpolate scar_midpoint between the
    nearest valid neighbours and recompute new/old end lengths.
    """
    n           = len(results)
    suspect_set = set(suspect_indices)

    for i in suspect_indices:
        r = results[i]

        prev_i = next((j for j in range(i - 1, -1, -1)
                       if j not in suspect_set
                       and results[j]['scar_detected']
                       and results[j].get('scar_midpoint') is not None), None)

        next_i = next((j for j in range(i + 1, n)
                       if j not in suspect_set
                       and results[j]['scar_detected']
                       and results[j].get('scar_midpoint') is not None), None)

        if prev_i is None and next_i is None:
            continue

        if prev_i is not None and next_i is not None:
            t       = (i - prev_i) / (next_i - prev_i)
            prev_mp = np.array(results[prev_i]['scar_midpoint'])
            next_mp = np.array(results[next_i]['scar_midpoint'])
            new_mp  = (1 - t) * prev_mp + t * next_mp
        elif prev_i is not None:
            new_mp = np.array(results[prev_i]['scar_midpoint'])
        else:
            new_mp = np.array(results[next_i]['scar_midpoint'])

        dbg   = r.get('debug_info', {})
        np_pt = dbg.get('new_pole_point')
        op_pt = dbg.get('old_pole_point')

        r['scar_midpoint'] = new_mp
        r['scar_detected'] = True

        if np_pt is not None and op_pt is not None:
            from .geometry import measure_pole_lengths
            new_len, old_len = measure_pole_lengths(
                new_mp, np.array(np_pt), np.array(op_pt))
            r['new_end_length']   = new_len
            r['old_end_length']   = old_len
            dbg['new_end_length'] = new_len
            dbg['old_end_length'] = old_len
            dbg['recent_scar']    = new_mp


def _ratio(result):
    ni = result.get('new_end_length')
    oi = result.get('old_end_length')
    if ni is not None and oi is not None and oi > 0:
        return ni / oi
    return None


# ── Pretty-print the stabilisation report ────────────────────────────────────

def print_stability_report(report):
    if not report:
        print("Scar stability: all frames clean.")
        return

    print("\n── Scar Stability Report ──────────────────────────────────")
    for name, events in sorted(report.items()):
        print(f"  {name}:")
        for e in events:
            old = f"{e['old_ratio']:.2f}" if e.get('old_ratio') is not None else "—"
            new = f"{e['new_ratio']:.2f}" if e.get('new_ratio') is not None else "—"
            print(f"    frame {e['frame']:>3}  [{e['status']:>14}]  "
                  f"ratio: {old} → {new}")
    print("───────────────────────────────────────────────────────────\n")
