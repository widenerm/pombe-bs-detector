"""
geometry.py  –  Low-level geometric computations.

All coordinate conventions: (row, col) → (y, x).
"""
import numpy as np
from scipy.interpolate import splprep, splev


# ── Contour smoothing & curvature ────────────────────────────────────────────

def compute_smoothed_curvature(contour, smooth_factor, n_points=300):
    """
    Fit a periodic B-spline to the contour and return (smooth_pts, kappa).

    smooth_pts : (n_points, 2)  array of (row, col)
    kappa      : (n_points,)    signed curvature; convention: convex is positive
    """
    contour = np.array(contour)
    x, y = contour[:, 1], contour[:, 0]   # col, row

    tck, _ = splprep([x, y], s=smooth_factor, per=True)
    u_new = np.linspace(0, 1, n_points, endpoint=False)
    x_s, y_s = splev(u_new, tck)

    dx  = np.gradient(x_s)
    dy  = np.gradient(y_s)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    denom = (dx**2 + dy**2)**1.5
    kappa = np.where(denom > 1e-10, (dx * ddy - dy * ddx) / denom, 0.0)

    # Ensure convex (outward) curvature is positive
    if np.mean(kappa) < 0:
        kappa = -kappa

    smooth_pts = np.vstack([y_s, x_s]).T   # back to (row, col)
    return smooth_pts, kappa


# ── PCA ──────────────────────────────────────────────────────────────────────

def compute_pca_axis(points):
    """
    Return (centre, unit_axis) where unit_axis is the first principal
    component (long axis) of *points*.
    """
    pts = np.array(points, dtype=float)
    centre = pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts - centre, full_matrices=False)
    axis = Vt[0]
    return centre, axis / np.linalg.norm(axis)


# ── Endpoints ────────────────────────────────────────────────────────────────

def get_contour_endpoints(contour, centre, axis):
    """
    Return the two contour points that are most extreme along *axis*
    (i.e., the two poles of the cell).
    """
    rel  = np.array(contour) - centre
    proj = rel @ axis
    return contour[np.argmin(proj)], contour[np.argmax(proj)]


# ── Width measurements ───────────────────────────────────────────────────────

def measure_width_at_position(smooth_pts, centre, axis, long_norm,
                               target_norm, window=0.05):
    """
    Measure cell width (perpendicular to *axis*) at a normalised position
    along the cell (0 = one pole, 1 = other pole, 0.5 = centre).

    Returns width in pixels.
    """
    normal_vec = np.array([-axis[1], axis[0]])
    near = np.abs(long_norm - target_norm) < window
    if near.sum() < 2:
        return 0.0
    pts_near    = smooth_pts[near]
    transverse  = (pts_near - centre) @ normal_vec
    return float(transverse.max() - transverse.min())


# ── Lengths ──────────────────────────────────────────────────────────────────

def measure_cell_length(ep1, ep2):
    """Pole-to-pole distance."""
    return float(np.linalg.norm(np.array(ep1) - np.array(ep2)))


def measure_pole_lengths(scar_midpoint, new_pole, old_pole):
    """Return (new_end_length, old_end_length) from scar midpoint to each pole."""
    return (float(np.linalg.norm(scar_midpoint - new_pole)),
            float(np.linalg.norm(scar_midpoint - old_pole)))


# ── Pole morphology ──────────────────────────────────────────────────────────

def measure_pole_pointiness(contour, pole_point, centre, axis, search_radius=15):
    """
    Higher score = more tapered (pointed) pole.
    Old poles tend to be slightly more pointed than new poles.
    """
    dists = np.linalg.norm(np.array(contour) - pole_point, axis=1)
    near  = contour[dists < search_radius]
    if len(near) < 5:
        return 0.0
    normal_vec = np.array([-axis[1], axis[0]])
    lateral    = np.abs((near - centre) @ normal_vec)
    return 1.0 / (lateral.mean() + 1e-6)


# ── Curvature fingerprint (for tracking) ─────────────────────────────────────

def compute_curvature_fingerprint(kappa, n_bins=20):
    """
    Normalised histogram of curvature values.
    Used as a soft cell identity signal in the Hungarian tracker.
    """
    hist, _ = np.histogram(kappa, bins=n_bins, range=(-1.0, 1.0))
    norm = np.linalg.norm(hist)
    return hist.astype(float) / (norm + 1e-8)
