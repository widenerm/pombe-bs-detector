"""
postprocessing.py  –  Temporal stabilisation of birth scar detections.

Problem
───────
The curvature landscape of a cell changes slightly between frames due to
small segmentation differences and genuine cell shape changes.  This means
the highest-scoring scar candidate can jump between frames even when the
true birth scar is not moving.

Observable in the summary table as outlier ratios like:
  F  frame 1 : new=7.5  old=139.3  ratio=0.05   ← outlier
  F  frame 0 : new=22.8 old=232.6  ratio=0.10   ← expected
  F  frame 2 : new=20.2 old=234.7  ratio=0.09   ← expected

Strategy
────────
For each tracked cell we express the scar midpoint as a normalised position
along the cell's long axis (0 = one pole, 1 = other pole).  This is scale-
and orientation-invariant.

After the full pipeline has run we:

  1. Collect the per-frame normalised scar position for every cell.
  2. Compute a rolling median over a small window (default 3 frames).
  3. Flag frames where the detection deviates > SCAR_STABILITY_THRESHOLD
     from the rolling median as 'suspect'.
  4. For suspect frames, replace the scar midpoint (and derived measurements)
     with an interpolated value from the nearest valid neighbours.
     If interpolation is disabled (SCAR_INTERPOLATE = False), the frame is
     flagged but values are left unchanged.

The result is written back into all_results in-place so the CSV and
visualisations automatically use the corrected values.
"""

import numpy as np


# ── Public entry point ────────────────────────────────────────────────────────

def stabilise_scars(all_results, config):
    """
    Run in-place temporal scar stabilisation on the output of run_pipeline.

    Parameters
    ----------
    all_results : list of frame dicts returned by run_pipeline
    config      : Config object

    Returns
    -------
    all_results : same list, mutated in-place
    report      : dict  {cell_name: [{'frame', 'status', 'old_ratio', 'new_ratio'}]}
    """
    window    = getattr(config, 'SCAR_STABILITY_WINDOW',    3)
    threshold = getattr(config, 'SCAR_STABILITY_THRESHOLD', 0.12)
    do_interp = getattr(config, 'SCAR_INTERPOLATE',         True)

    # ── Build per-cell timeline ───────────────────────────────────────────────
    # cell_timeline[name] = list of (frame_idx, result_dict) in order
    cell_timeline = {}
    for fd in all_results:
        for r in fd['cells']:
            name = r.get('cell_name', '?')
            cell_timeline.setdefault(name, []).append((fd['frame_idx'], r))

    report = {}

    for name, timeline in cell_timeline.items():
        if len(timeline) < 2:
            continue

        frame_indices = [t[0] for t in timeline]
        results       = [t[1] for t in timeline]

        # Normalised scar positions (None if not detected)
        norm_positions = []
        for r in results:
            if r['scar_detected'] and r.get('scar_midpoint') is not None:
                pos = _normalised_scar_position(r)
            else:
                pos = None
            norm_positions.append(pos)

        # Rolling median over valid positions
        medians = _rolling_median(norm_positions, window)

        cell_report = []
        suspect_indices = []

        for i, (fidx, r, pos, med) in enumerate(
                zip(frame_indices, results, norm_positions, medians)):

            if pos is None or med is None:
                r['scar_stability'] = 'no_detection'
                continue

            deviation = abs(pos - med)
            if deviation > threshold:
                r['scar_stability'] = 'suspect'
                suspect_indices.append(i)
                old_ratio = (r['new_end_length'] / r['old_end_length']
                             if r.get('old_end_length') else None)
                cell_report.append({'frame': fidx, 'status': 'suspect',
                                    'deviation': round(deviation, 3),
                                    'old_ratio': round(old_ratio, 3) if old_ratio else None})
            else:
                r['scar_stability'] = 'stable'

        # ── Interpolate suspect frames ────────────────────────────────────────
        if do_interp and suspect_indices:
            _interpolate_suspects(results, norm_positions, suspect_indices)
            # Re-label interpolated frames
            for i in suspect_indices:
                r = results[i]
                if r.get('scar_stability') == 'suspect':
                    r['scar_stability'] = 'interpolated'
                    new_ratio = (r['new_end_length'] / r['old_end_length']
                                 if r.get('old_end_length') else None)
                    for entry in cell_report:
                        if entry['frame'] == frame_indices[i]:
                            entry['status']    = 'interpolated'
                            entry['new_ratio'] = round(new_ratio, 3) if new_ratio else None

        if cell_report:
            report[name] = cell_report
            print(f"  Scar stabilisation – {name}: "
                  f"{len(cell_report)} suspect frame(s) "
                  f"{'→ interpolated' if do_interp else '(flagged only)'}")

    return all_results, report


# ── Internal helpers ──────────────────────────────────────────────────────────

def _normalised_scar_position(result):
    """
    Express the scar midpoint as a normalised scalar [0, 1] along the
    cell's long axis.  Returns None if geometry is unavailable.
    """
    dbg = result.get('debug_info', {})
    centre = dbg.get('centre')
    if centre is None:
        centre = dbg.get('center')
    axis   = dbg.get('axis')
    mp     = result.get('scar_midpoint')

    if centre is None or axis is None or mp is None:
        return None

    ep1 = dbg.get('new_pole_point')
    ep2 = dbg.get('old_pole_point')
    if ep1 is None or ep2 is None:
        return None

    # Project everything onto the long axis
    def proj(pt):
        return float(np.dot(np.array(pt) - np.array(centre), np.array(axis)))

    p_ep1 = proj(ep1)
    p_ep2 = proj(ep2)
    p_mp  = proj(mp)

    lo, hi = min(p_ep1, p_ep2), max(p_ep1, p_ep2)
    span = hi - lo
    if span < 1e-6:
        return None

    return float(np.clip((p_mp - lo) / span, 0.0, 1.0))


def _rolling_median(values, window):
    """
    For each position in *values* (a list of floats or None), compute the
    median of the *window* nearest valid neighbours (excluding the element
    itself when at least one neighbour exists).

    Returns a list of the same length; entries are None where no valid
    neighbours exist.
    """
    n      = len(values)
    result = [None] * n

    for i in range(n):
        half = window // 2
        neighbors = []
        for j in range(max(0, i - half), min(n, i + half + 1)):
            if j != i and values[j] is not None:
                neighbors.append(values[j])
        # Fall back to including i itself if no neighbours found
        if not neighbors and values[i] is not None:
            neighbors.append(values[i])
        result[i] = float(np.median(neighbors)) if neighbors else None

    return result


def _interpolate_suspects(results, norm_positions, suspect_indices):
    """
    For each suspect frame, linearly interpolate the scar midpoint between
    the nearest valid frames before and after it, then recompute the
    new/old end lengths from the interpolated midpoint.
    """
    n            = len(results)
    suspect_set  = set(suspect_indices)

    for i in suspect_indices:
        r = results[i]

        # Find nearest valid frame before i
        prev_i = next((j for j in range(i - 1, -1, -1)
                       if j not in suspect_set
                       and results[j]['scar_detected']
                       and results[j].get('scar_midpoint') is not None), None)

        # Find nearest valid frame after i
        next_i = next((j for j in range(i + 1, n)
                       if j not in suspect_set
                       and results[j]['scar_detected']
                       and results[j].get('scar_midpoint') is not None), None)

        if prev_i is None and next_i is None:
            continue   # no valid neighbours at all

        if prev_i is not None and next_i is not None:
            # Linear interpolation
            t        = (i - prev_i) / (next_i - prev_i)
            prev_mp  = np.array(results[prev_i]['scar_midpoint'])
            next_mp  = np.array(results[next_i]['scar_midpoint'])
            new_mp   = (1 - t) * prev_mp + t * next_mp
        elif prev_i is not None:
            new_mp = np.array(results[prev_i]['scar_midpoint'])
        else:
            new_mp = np.array(results[next_i]['scar_midpoint'])

        # Write corrected midpoint and recompute lengths
        dbg     = r.get('debug_info', {})
        np_pt   = dbg.get('new_pole_point')
        op_pt   = dbg.get('old_pole_point')

        r['scar_midpoint'] = new_mp
        if np_pt is not None and op_pt is not None:
            from .geometry import measure_pole_lengths
            new_len, old_len = measure_pole_lengths(
                new_mp, np.array(np_pt), np.array(op_pt))
            r['new_end_length'] = new_len
            r['old_end_length'] = old_len
            dbg['new_end_length'] = new_len
            dbg['old_end_length'] = old_len


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
            dev = f"{e['deviation']:.3f}"
            print(f"    frame {e['frame']:>3}  deviation={dev}  "
                  f"ratio: {old} → {new}  [{e['status']}]")
    print("───────────────────────────────────────────────────────────\n")
