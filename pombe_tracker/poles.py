"""
poles.py  –  Pole identification via neighbor proximity, birth scar, or morphology.

All public functions accept a *config* argument so there are no hidden Config
singleton references.

Chain cell handling
───────────────────
When a cell has credible neighbors at BOTH poles (e.g. it is sandwiched
between two other cells in a chain), neighbor proximity cannot distinguish
new from old end.  This case is now detected explicitly and returned as
confidence = 'chain'.  Callers should skip to scar-based or lineage-based
pole assignment rather than using the neighbor result.

The threshold for "credible" is NEIGHBOR_HIGH_CONFIDENCE_DIST: if both poles
have a neighbor within that distance, we cannot trust proximity alone.
"""
import numpy as np
from .geometry import measure_pole_pointiness


# ── Neighbor detection ────────────────────────────────────────────────────────

def find_pole_to_pole_neighbors(cell_label, endpoints, all_cell_info, threshold):
    """
    Return a list of dicts describing which poles of *cell_label* are within
    *threshold* pixels of a pole on another cell.
    """
    neighbors  = []
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
                        'label':          other_label,
                        'our_pole_idx':   i,
                        'their_pole_idx': j,
                        'distance':       dist,
                        'our_pole':       our_p,
                        'their_pole':     their_p,
                    })
    return neighbors


def determine_new_pole_from_neighbors(neighbors, endpoints, config):
    """
    Given a list of neighbor records, decide which of the two *endpoints*
    (index 0 or 1) is the new pole.

    Returns (pole_index_or_None, confidence_string).

    Confidence values:
      'high'   – one pole has a neighbor clearly closer than the other
      'medium' – one pole has a neighbor; the other does not
      'low'    – one pole is closer but the margin is small
      'chain'  – BOTH poles have credible neighbors (within
                 NEIGHBOR_HIGH_CONFIDENCE_DIST); neighbor proximity cannot
                 resolve new vs old end.  Caller should use scar/lineage.
      'none'   – no neighbors at all
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

    # Only one pole has neighbors — unambiguous
    if has_p0 and not has_p1:
        return 0, ('high' if p0_min < hc else 'medium')
    if has_p1 and not has_p0:
        return 1, ('high' if p1_min < hc else 'medium')

    # Both poles have neighbors — check for chain ambiguity first
    if has_p0 and has_p1:
        # If both poles have a credible (close) neighbor, we cannot trust
        # proximity to distinguish new from old end.
        if p0_min < hc and p1_min < hc:
            return None, 'chain'

        # One credible, one not — the credible one is the new pole
        diff   = abs(p0_min - p1_min)
        margin = hc * 0.2
        if p0_min <= p1_min:
            return 0, ('high' if diff > margin else 'low')
        else:
            return 1, ('high' if diff > margin else 'low')

    return None, 'none'


# ── Unified pole strategy ─────────────────────────────────────────────────────

def determine_poles_strategy(cell_label, endpoints, center, axis, contour,
                              all_cell_info, config, scar_midpoint=None,
                              lineage_new_pole=None, lineage_old_pole=None):
    """
    Assign new / old poles using the best available evidence.

    Priority
    --------
    1. Lineage-derived poles  (passed in by the pipeline after division)
    2. Birth scar midpoint    (geometric ground truth)
    3. Pole-to-pole neighbor proximity  (reliable shortly after division,
                                         but skipped for chain cells)
    4. Pole morphology / pointiness     (fallback)

    Parameters
    ----------
    lineage_new_pole, lineage_old_pole : np.array or None
        When provided (freshly divided daughters), bypass all other strategies.

    Returns
    -------
    new_pole, old_pole, new_direction, method_str, confidence_str
    """
    ep1, ep2 = np.array(endpoints[0]), np.array(endpoints[1])

    # ── Strategy 0: lineage (highest priority) ────────────────────────────────
    if lineage_new_pole is not None and lineage_old_pole is not None:
        new_pole = np.array(lineage_new_pole)
        old_pole = np.array(lineage_old_pole)
        # Snap to the actual endpoint that is closest
        if np.linalg.norm(new_pole - ep1) < np.linalg.norm(new_pole - ep2):
            new_pole, old_pole = ep1, ep2
        else:
            new_pole, old_pole = ep2, ep1
        return new_pole, old_pole, _direction(new_pole, center, axis), 'lineage', 'high'

    # ── Strategy 1: scar ─────────────────────────────────────────────────────
    if scar_midpoint is not None:
        sm = np.array(scar_midpoint)
        if np.linalg.norm(sm - ep1) < np.linalg.norm(sm - ep2):
            new_pole, old_pole = ep1, ep2
        else:
            new_pole, old_pole = ep2, ep1
        return new_pole, old_pole, _direction(new_pole, center, axis), 'scar_based', 'high'

    # ── Strategy 2: neighbor proximity (skipped for chain cells) ─────────────
    neighbors = find_pole_to_pole_neighbors(
        cell_label, endpoints, all_cell_info, config.POLE_PROXIMITY_THRESHOLD)
    idx, conf = determine_new_pole_from_neighbors(neighbors, endpoints, config)

    if conf == 'chain':
        # Both poles have credible neighbors — fall through to morphology,
        # but flag that this cell's pole assignment is uncertain.
        pass
    elif idx is not None and conf in ('high', 'medium'):
        new_pole = ep1 if idx == 0 else ep2
        old_pole = ep2 if idx == 0 else ep1
        return new_pole, old_pole, _direction(new_pole, center, axis), 'neighbor_poles', conf

    # ── Strategy 3: morphology ────────────────────────────────────────────────
    pt1 = measure_pole_pointiness(contour, ep1, center, axis)
    pt2 = measure_pole_pointiness(contour, ep2, center, axis)

    if pt1 > pt2 * 1.1:
        new_pole, old_pole = ep2, ep1
        conf = 'medium' if pt1 > pt2 * 1.3 else 'low'
    elif pt2 > pt1 * 1.1:
        new_pole, old_pole = ep1, ep2
        conf = 'medium' if pt2 > pt1 * 1.3 else 'low'
    else:
        new_pole, old_pole = ep1, ep2
        conf = 'very_low'

    method = 'morphology_chain' if conf == 'chain' else 'morphology'
    return new_pole, old_pole, _direction(new_pole, center, axis), method, conf


# ── Internal ──────────────────────────────────────────────────────────────────

def _direction(new_pole, center, axis):
    vec = np.array(new_pole) - np.array(center)
    return -axis if np.dot(vec, axis) < 0 else axis
