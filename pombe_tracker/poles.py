"""
poles.py  –  Pole identification via neighbour proximity, birth scar, or morphology.

All public functions accept a *config* argument so there are no hidden Config
singleton references.
"""
import numpy as np
from .geometry import measure_pole_pointiness


# ── Neighbour detection ───────────────────────────────────────────────────────

def find_pole_to_pole_neighbors(cell_label, endpoints, all_cell_info, threshold):
    """
    Return a list of dicts describing which poles of *cell_label* are within
    *threshold* pixels of a pole on another cell.
    """
    neighbors = []
    our_poles  = [np.array(ep) for ep in endpoints]

    for other_label, other_info in all_cell_info.items():
        if other_label == cell_label:
            continue
        if not other_info.get('endpoints'):
            continue
        their_poles = [np.array(ep) for ep in other_info['endpoints']]

        for i, our_p in enumerate(our_poles):
            for j, their_p in enumerate(their_poles):
                dist = float(np.linalg.norm(our_p - their_p))
                if dist < threshold:
                    neighbors.append({
                        'label':        other_label,
                        'our_pole_idx': i,
                        'their_pole_idx': j,
                        'distance':     dist,
                        'our_pole':     our_p,
                        'their_pole':   their_p,
                    })
    return neighbors


def determine_new_pole_from_neighbors(neighbors, endpoints, config):
    """
    Given a list of neighbour records, decide which of the two *endpoints*
    (index 0 or 1) is the new pole.

    Returns (pole_index_or_None, confidence_string).
    """
    if not neighbors:
        return None, 'none'

    p0_min = p1_min = float('inf')
    has_p0 = has_p1 = False

    for n in neighbors:
        d = n['distance']
        if n['our_pole_idx'] == 0:
            has_p0 = True
            p0_min = min(p0_min, d)
        else:
            has_p1 = True
            p1_min = min(p1_min, d)

    hc = config.NEIGHBOR_HIGH_CONFIDENCE_DIST

    if has_p0 and not has_p1:
        return 0, ('high' if p0_min < hc else 'medium')
    if has_p1 and not has_p0:
        return 1, ('high' if p1_min < hc else 'medium')

    # Both poles have neighbours – pick the closer one
    diff   = abs(p0_min - p1_min)
    margin = hc * 0.2
    if p0_min <= p1_min:
        return 0, ('high' if diff > margin else 'low')
    else:
        return 1, ('high' if diff > margin else 'low')


# ── Unified pole strategy ─────────────────────────────────────────────────────

def determine_poles_strategy(cell_label, endpoints, centre, axis, contour,
                              all_cell_info, config, scar_midpoint=None):
    """
    Assign new / old poles using the best available evidence.

    Priority
    --------
    1. Birth scar midpoint  (most reliable – geometric ground truth)
    2. Pole-to-pole neighbour proximity  (reliable shortly after division)
    3. Pole morphology / pointiness  (fallback for isolated cells)

    Returns
    -------
    new_pole, old_pole, new_direction, method_str, confidence_str
    """
    ep1, ep2 = np.array(endpoints[0]), np.array(endpoints[1])

    # ── Strategy 1: scar ─────────────────────────────────────────────────────
    if scar_midpoint is not None:
        sm = np.array(scar_midpoint)
        if np.linalg.norm(sm - ep1) < np.linalg.norm(sm - ep2):
            new_pole, old_pole = ep1, ep2
        else:
            new_pole, old_pole = ep2, ep1
        return new_pole, old_pole, _direction(new_pole, centre, axis), 'scar_based', 'high'

    # ── Strategy 2: neighbour proximity ──────────────────────────────────────
    neighbors = find_pole_to_pole_neighbors(
        cell_label, endpoints, all_cell_info, config.POLE_PROXIMITY_THRESHOLD)
    idx, conf = determine_new_pole_from_neighbors(neighbors, endpoints, config)

    if idx is not None and conf in ('high', 'medium'):
        new_pole = ep1 if idx == 0 else ep2
        old_pole = ep2 if idx == 0 else ep1
        return new_pole, old_pole, _direction(new_pole, centre, axis), 'neighbor_poles', conf

    # ── Strategy 3: morphology ────────────────────────────────────────────────
    pt1 = measure_pole_pointiness(contour, ep1, centre, axis)
    pt2 = measure_pole_pointiness(contour, ep2, centre, axis)

    if pt1 > pt2 * 1.1:                     # ep1 is more pointed → old pole
        new_pole, old_pole = ep2, ep1
        conf = 'medium' if pt1 > pt2 * 1.3 else 'low'
    elif pt2 > pt1 * 1.1:
        new_pole, old_pole = ep1, ep2
        conf = 'medium' if pt2 > pt1 * 1.3 else 'low'
    else:
        new_pole, old_pole = ep1, ep2       # no clear signal
        conf = 'very_low'

    return new_pole, old_pole, _direction(new_pole, centre, axis), 'morphology', conf


# ── Internal ──────────────────────────────────────────────────────────────────

def _direction(new_pole, centre, axis):
    vec = np.array(new_pole) - np.array(centre)
    return -axis if np.dot(vec, axis) < 0 else axis
