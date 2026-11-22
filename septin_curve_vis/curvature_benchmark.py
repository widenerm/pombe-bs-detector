"""Which contour gives the most accurate curvature? A synthetic benchmark.

Draws rod-shaped cells of known shape (plain, constricted, with a scar
ridge, bent), renders them as brightfield images using the mean edge profile
and noise measured on real cells, segments them with Cellpose exactly as the
septin analysis does, and compares each contour method's curvature with the
true curvature.

Truth is placed GFP_EDGE_PX inside where Cellpose puts the edge on real cells,
the half-maximum of diffuse Spn3-GFP measured relative to Cellpose edges.
The absolute membrane position is uncertain by about 2 px, so every method is
also scored offset-free: against the parallel curve at the method's own mean
offset, which isolates whether it gets the shape right.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import tifffile
from matplotlib.path import Path as Polygon
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from skimage.measure import find_contours

from pombe_tracker.config import Config
from pombe_tracker.geometry import compute_smoothed_curvature
from pombe_tracker.scar_profile import _outward_normals, contour_geometry, membrane_offsets

GFP_EDGE_PX = 2.0          # true membrane this far inside the typical Cellpose edge
PROFILE_RANGE = 15.0


# ── Real-image statistics ────────────────────────────────────────────────────

def measure_edge_profile(image_paths, step=0.5, bright_at=-1.25):
    """Brightfield edge profile along contour normals, aligned on the bright halo.

    Averaging around the Cellpose edge directly blurs the profile by the
    edge's own jitter, so each normal is first shifted to put its bright-halo
    peak at bright_at (where the unaligned mean has it). Returns the profile
    vs signed distance from the typical Cellpose edge (+ outside) in
    background-noise units, and the interior texture SD in the same units.
    """
    offs = np.arange(-PROFILE_RANGE - 6, PROFILE_RANGE + 6 + 1e-6, step)
    grid = np.arange(-PROFILE_RANGE, PROFILE_RANGE + 1e-6, step)
    search = (offs >= -5) & (offs <= 2)
    rows, texture = [], []
    for path in image_paths:
        bf = tifffile.imread(path)[0].astype(float)
        cells = tifffile.imread(str(path).replace("_image", "_cells"))
        for lab in np.unique(cells)[1:]:
            m = cells == lab
            if m.sum() < 3000:
                continue
            rr, cc = np.nonzero(m)
            r0, c0 = max(rr.min() - 30, 0), max(cc.min() - 30, 0)
            r1, c1 = rr.max() + 31, cc.max() + 31
            mc, im = m[r0:r1, c0:c1], bf[r0:r1, c0:c1]
            other = (cells[r0:r1, c0:c1] > 0) & ~mc
            sd = ndi.distance_transform_edt(~mc) - ndi.distance_transform_edt(mc)
            far = ndi.distance_transform_edt(~other) >= 16
            bg = im[far & (sd > 10) & (sd <= PROFILE_RANGE)]
            if bg.size < 50:
                continue
            noise = 1.4826 * np.median(np.abs(bg - np.median(bg)))
            z = (im - np.median(bg)) / noise
            g = contour_geometry(max(find_contours(mc.astype(float), 0.5), key=len), 0.15)
            q = g["pts"][:, None, :] + offs[None, :, None] * _outward_normals(g["pts"])[:, None, :]
            prof = ndi.map_coordinates(z, [q[..., 0], q[..., 1]], order=1, mode="nearest")
            blocked = ndi.map_coordinates((~far).astype(float), [q[..., 0], q[..., 1]],
                                          order=0, mode="nearest").any(axis=1)
            peak = offs[search][np.argmax(ndi.gaussian_filter1d(prof[:, search], 1, axis=1), axis=1)]
            for r, p0 in zip(prof[~blocked], peak[~blocked]):
                rows.append(np.interp(grid + p0 - bright_at, offs, r))
            texture.append(np.std(z[mc & (sd < -10)]))
    return grid, np.mean(rows, axis=0), float(np.median(texture))


# ── Synthetic cells ──────────────────────────────────────────────────────────

def make_cell(rng, kind):
    """Dense true outline (local x along the axis, y across) with feature flags."""
    L, W = rng.uniform(120, 260), rng.uniform(48, 60)
    R = W / 2
    straight = L / 2 - R
    ss = 8
    xs = np.arange(-L / 2 - 6, L / 2 + 6, 1 / ss)
    ys = np.arange(-W, W, 1 / ss)
    X, Y = np.meshgrid(xs, ys)
    w = np.where(np.abs(X) <= straight, R,
                 np.sqrt(np.clip(R ** 2 - (np.abs(X) - straight) ** 2, 0, None)))
    yc = np.zeros_like(X)
    feature = None
    if kind == "constriction":
        x0, s = rng.uniform(-0.1, 0.1) * L, rng.uniform(3, 7)
        w = w * (1 - rng.uniform(0.15, 0.45) * np.exp(-(X - x0) ** 2 / (2 * s ** 2)))
        feature = (x0, 3 * s)
    elif kind == "scar":
        s = rng.uniform(3, 5)
        xb = rng.choice([-1, 1]) * (straight - rng.uniform(0.2, 0.6) * W)
        w = w + (w > 0) * rng.uniform(1.0, 2.5) * np.exp(-(X - xb) ** 2 / (2 * s ** 2))
        feature = (xb, 3 * s)
    elif kind == "bent":
        yc = rng.uniform(3, 8) / (L / 2) ** 2 * X ** 2
    inside = np.abs(Y - yc) < w
    c = max(find_contours(inside.astype(float), 0.5), key=len)
    pts = np.c_[xs[0] + c[:, 1] / ss, ys[0] + c[:, 0] / ss]          # x, y
    # Resample to 0.1 px and smooth 1.5 px so the truth has well-defined curvature.
    seg = np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]), axis=0), axis=1)
    arc = np.r_[0, np.cumsum(seg)]
    n = int(arc[-1] / 0.1)
    t = np.arange(n) * arc[-1] / n
    closed = np.vstack([pts, pts[:1]])
    dense = np.c_[np.interp(t, arc, closed[:, 0]), np.interp(t, arc, closed[:, 1])]
    dense = ndi.gaussian_filter1d(dense, 15, axis=0, mode="wrap")
    on_feature = (np.abs(dense[:, 0] - feature[0]) < feature[1]) if feature else \
        np.zeros(len(dense), bool)
    return dict(local=dense, kind=kind, feature=on_feature, R=R)


def curvature_of_dense(pts):
    """Signed curvature of a closed, finely sampled curve; convex positive."""
    d1 = (np.roll(pts, -1, 0) - np.roll(pts, 1, 0)) / 2
    d2 = np.roll(pts, -1, 0) - 2 * pts + np.roll(pts, 1, 0)
    k = (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / np.linalg.norm(d1, axis=1) ** 3
    return k if np.mean(k) > 0 else -k


def place(cell, theta, center):
    """Rotate/translate the local outline into image (row, col) coordinates."""
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    xy = cell["local"] @ rot.T + center
    pts = np.c_[xy[:, 1], xy[:, 0]]
    return dict(cell, pts=pts, kappa=curvature_of_dense(pts), normals=_outward_normals(pts))


def render(cells, shape, profile, texture_sd, rng):
    """Brightfield-like image: real edge profile around each true outline, plus noise."""
    d_cp, z_cp = profile
    z = np.zeros(shape)
    rows, cols = np.mgrid[:shape[0], :shape[1]]
    for cell in cells:
        lo = np.maximum(np.floor(cell["pts"].min(0)).astype(int) - 25, 0)
        hi = np.minimum(np.ceil(cell["pts"].max(0)).astype(int) + 26, shape)
        rr, cc = rows[lo[0]:hi[0], lo[1]:hi[1]], cols[lo[0]:hi[0], lo[1]:hi[1]]
        q = np.c_[rr.ravel(), cc.ravel()].astype(float)
        dist, _ = cKDTree(cell["pts"]).query(q)
        inside = Polygon(cell["pts"]).contains_points(q)
        sd = np.where(inside, -dist, dist).reshape(rr.shape)
        prof = np.interp(sd - GFP_EDGE_PX, d_cp, z_cp)
        z[lo[0]:hi[0], lo[1]:hi[1]] = np.where(sd < 25, prof, z[lo[0]:hi[0], lo[1]:hi[1]])
        cell["truth_mask_box"] = (lo, hi, sd < 0)
    texture = ndi.gaussian_filter(rng.normal(size=shape), 1.5)
    texture *= texture_sd / texture.std()
    interior = np.zeros(shape)
    for cell in cells:
        lo, hi, m = cell["truth_mask_box"]
        interior[lo[0]:hi[0], lo[1]:hi[1]] = np.maximum(
            interior[lo[0]:hi[0], lo[1]:hi[1]],
            ndi.gaussian_filter((ndi.distance_transform_edt(m) > 8).astype(float), 2))
    z = z + texture * interior + rng.normal(size=shape)
    return np.clip(2000 + 110 * z, 0, 65535).astype(np.float32)


# ── Contour methods ──────────────────────────────────────────────────────────

def method_curves(mask, image):
    """(name, points (row, col), curvature) for every method from one cell mask."""
    contour = max(find_contours(mask.astype(float), 0.5), key=len)
    out = []
    for s in (10.0, 40.0):
        pts, k = compute_smoothed_curvature(contour, s, 300)
        out.append((f"mask_spline_s{s:g}", pts, k))
    g = contour_geometry(contour, 0.15)
    out.append(("mask_profile_geometry", g["pts"], g["kappa"]))
    off = membrane_offsets(g, image)
    traced = g["pts"] + off[:, None] * _outward_normals(g["pts"])
    for s in (0.15, 0.5, 1.0, 2.0):
        tg = contour_geometry(traced, s)
        out.append((f"membrane_s{s:g}", tg["pts"], tg["kappa"]))
    return out


def score(cell, pts, k):
    """Errors of a contour's curvature against the true outline."""
    j = cKDTree(cell["pts"]).query(pts)[1]
    h = np.sum((pts - cell["pts"][j]) * cell["normals"][j], axis=1)     # + outside truth
    kt = cell["kappa"][j]
    hbar = float(h.mean())
    denom = 1 + kt * hbar
    ok = denom > 0.3
    k_par = kt / np.where(ok, denom, 1)
    feat = cell["feature"][j] & ok
    row = dict(kind=cell["kind"], offset_px=hbar, wiggle_px=float(np.std(h - hbar)),
               rmse=float(np.sqrt(np.mean((k - kt) ** 2))),
               rmse_shape=float(np.sqrt(np.mean((k[ok] - k_par[ok]) ** 2))),
               spearman=float(spearmanr(k, kt).correlation),
               pole_ratio=float(np.percentile(k, 95) / np.percentile(kt, 95)))
    if feat.sum() >= 5:
        row["rmse_feature_shape"] = float(np.sqrt(np.mean((k[feat] - k_par[feat]) ** 2)))
        if cell["kind"] == "constriction":
            row["feature_peak_ratio"] = float(k[feat].min() / k_par[feat].min())
        else:
            row["feature_peak_ratio"] = float((k[feat].max() - np.median(k[feat])) /
                                              (k_par[feat].max() - np.median(k_par[feat])))
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-fields", type=Path, required=True,
                        help="Folder of *_image.tif / *_cells.tif to calibrate the edge profile")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--images", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    d_cp, z_cp, texture_sd = measure_edge_profile(sorted(args.real_fields.glob("*_image.tif"))[:8])
    print(f"Edge profile from real cells; interior texture SD {texture_sd:.2f}", flush=True)
    rng = np.random.default_rng(args.seed)
    from pombe_tracker.segmentation import CellposeSegmenter
    segmenter = CellposeSegmenter(Config())
    kinds = ["plain", "constriction", "scar", "bent"]
    rows, shape = [], (1024, 1024)
    for i in range(args.images):
        cells = []
        for slot in range(6):
            # (x, y) = (column, row) of the cell centre, 3 x 2 grid
            center = np.array([170 + 340 * (slot % 3), 256 + 512 * (slot // 3)], float)
            center += rng.uniform(-20, 20, 2)
            cells.append(place(make_cell(rng, kinds[(i * 6 + slot) % 4]),
                               rng.uniform(0, np.pi), center))
        image = render(cells, shape, (d_cp, z_cp), texture_sd, rng)
        labels = segmenter.segment(image)
        for n, cell in enumerate(cells):
            lo, hi, m = cell["truth_mask_box"]
            truth = np.zeros(shape, bool)
            truth[lo[0]:hi[0], lo[1]:hi[1]] = m
            hit = labels[truth]
            lab = np.bincount(hit[hit > 0]).argmax() if np.any(hit > 0) else 0
            iou = (np.sum((labels == lab) & truth) / np.sum((labels == lab) | truth)) if lab else 0
            sources = [("ideal_mask", truth)] + ([("cellpose", labels == lab)] if iou > 0.7 else [])
            for source, mask in sources:
                for name, pts, k in method_curves(mask, image):
                    rows.append(dict(image=i, cell=n, mask=source, method=name, iou=iou,
                                     **score(cell, pts, k)))
        print(f"image {i + 1}/{args.images}: Cellpose IoU "
              f"{[round(r['iou'], 2) for r in rows if r['method'] == 'mask_spline_s40' and r['mask'] == 'cellpose' and r['image'] == i]}",
              flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
