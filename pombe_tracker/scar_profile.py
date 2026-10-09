"""
scar_profile.py  –  Interpretable axial-profile birth-scar localizer.

The scar is a ring around the cell, so it is located by its distance d from a
pole along the PCA axis. Every 1-px cross-section perpendicular to the axis,
from each pole, is a candidate; the prediction is the midpoint of the best
cross-section. Each candidate gets a linear score built from named terms
(all lengths in units of the mid-cell width W):

  distance prior   -(u - mu)^2 / (2 sd^2),  u = d/W,
                   mu = SCAR_PRIOR_OFFSET + SCAR_PRIOR_SLOPE * (L/W)
                   (longer cells carry the scar further from the new pole)
  neighbor hint    +1 at the end nearest the neighbor-derived new pole,
                   -1 at the other end, 0 without a hint
  membrane width   image-refined width of the cross-section: mask width plus
                   the outward offsets of the dark membrane line traced in
                   the brightfield image on both flanks. The scar ridge is
                   the widest point of the new end.
  membrane ridge   negative 2nd derivative of that width profile (a ridge or
                   corner on both flanks)

Weights were fitted once on Carmen's 80 annotated wild-type cells (see
validation/reports/scar_v3/REPORT.md); they are plain numbers in Config and may be
re-tuned by hand or re-fitted for other strains with validation/scar_fit.py.

Confidence = softmax probability mass of candidates within
SCAR_CONFIDENCE_RADIUS px of the chosen point. Cells below
SCAR_CONFIDENCE_FLAG are marked for manual review.
"""
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates

CROP_MARGIN = 20


# ── Contour geometry ─────────────────────────────────────────────────────────

def contour_geometry(contour, smooth_per_point=0.15, step=1.0):
    """Periodic spline resampled uniformly by arc length, with PCA axis."""
    c = np.asarray(contour, float)
    if np.allclose(c[0], c[-1]):
        c = c[:-1]
    tck, _ = splprep([c[:, 1], c[:, 0]], s=smooth_per_point * len(c), per=True)
    dense = np.linspace(0, 1, 20 * len(c) + 1)
    xy = np.array(splev(dense, tck)).T
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    n = int(arc[-1] / step)
    u = np.interp(np.arange(n) * arc[-1] / n, arc, dense)
    x, y = splev(u, tck)
    dx, dy = splev(u, tck, der=1)
    ddx, ddy = splev(u, tck, der=2)
    kappa = (dx * ddy - dy * ddx) / np.maximum((dx * dx + dy * dy) ** 1.5, 1e-12)
    if np.mean(kappa) < 0:
        kappa = -kappa
    pts = np.c_[y, x]
    center = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - center, full_matrices=False)
    axis = vt[0] / np.linalg.norm(vt[0])
    normal = np.array([-axis[1], axis[0]])
    return dict(pts=pts, kappa=kappa, center=center, axis=axis, normal=normal,
                proj=(pts - center) @ axis, trans=(pts - center) @ normal, ds=arc[-1] / n)


def _cross_section(g, pos):
    proj, trans = g['proj'], g['trans']
    b, tb = np.roll(proj, -1), np.roll(trans, -1)
    cross = np.flatnonzero(((proj <= pos) & (b > pos)) | ((b <= pos) & (proj > pos)))
    if len(cross) < 2:
        return None
    f = (pos - proj[cross]) / (b[cross] - proj[cross])
    ys = trans[cross] + f * (tb[cross] - trans[cross])
    lo, hi = np.argmin(ys), np.argmax(ys)
    base = g['center'] + pos * g['axis']
    return (base + ys[lo] * g['normal'], base + ys[hi] * g['normal'],
            ys[hi] - ys[lo], cross[lo] + f[lo], cross[hi] + f[hi])


def end_profiles(g, step=1.0, max_frac=0.5):
    """1-px cross-sections from each pole inward, up to max_frac of the length."""
    lo, hi = g['proj'].min(), g['proj'].max()
    L = hi - lo
    out = []
    for end in (0, 1):
        rows = []
        for d in np.arange(1.0, L * max_frac, step):
            s = _cross_section(g, lo + d if end == 0 else hi - d)
            if s is not None:
                rows.append((d,) + s)
        out.append(dict(end=end, d=np.array([r[0] for r in rows]),
                        p1=np.array([r[1] for r in rows]), p2=np.array([r[2] for r in rows]),
                        w=np.array([r[3] for r in rows]),
                        i1=np.array([r[4] for r in rows]), i2=np.array([r[5] for r in rows])))
    return out, L


def _sample_periodic(arr, idx):
    n = len(arr)
    i0 = np.floor(idx).astype(int) % n
    f = idx - np.floor(idx)
    return arr[i0] * (1 - f) + arr[(i0 + 1) % n] * f


# ── Membrane tracing in the brightfield image ────────────────────────────────

def _outward_normals(pts):
    t = np.roll(pts, -1, 0) - np.roll(pts, 1, 0)
    t /= np.linalg.norm(t, axis=1)[:, None]
    nrm = np.c_[-t[:, 1], t[:, 0]]
    if np.mean(np.sum((pts - pts.mean(0)) * nrm, 1)) < 0:
        nrm = -nrm
    return nrm


def _viterbi(cost, lam, maxjump=2):
    """Path through (N, K) costs minimizing sum(cost) + lam*|jump|; circular via 3 copies."""
    N, K = cost.shape
    C = np.vstack([cost, cost, cost])
    acc = C[0].copy()
    back = np.zeros((len(C), K), int)
    ks = np.arange(K)
    for t in range(1, len(C)):
        best = np.full(K, np.inf)
        arg = np.zeros(K, int)
        for dk in range(-maxjump, maxjump + 1):
            src = ks - dk
            valid = (src >= 0) & (src < K)
            v = np.full(K, np.inf)
            v[valid] = acc[src[valid]] + lam * abs(dk)
            better = v < best
            best[better] = v[better]
            arg[better] = src[better]
        acc = best + C[t]
        back[t] = arg
    path = np.zeros(len(C), int)
    path[-1] = int(np.argmin(acc))
    for t in range(len(C) - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path[N:2 * N]


def membrane_offsets(g, image, half=6.0, lam=0.3, sigma_img=1.0, avg_tan=1.0):
    """Outward offset (px) of the membrane from each contour point.

    Mean of two smoothly traced edges: the darkest line and the strongest
    dark-to-bright (membrane-to-halo) transition, searched within ±half px.
    """
    pts = g['pts']
    lo = np.maximum(0, np.floor(pts.min(0)).astype(int) - CROP_MARGIN)
    hi = np.minimum(image.shape, np.ceil(pts.max(0)).astype(int) + CROP_MARGIN + 1)
    im = gaussian_filter(image[lo[0]:hi[0], lo[1]:hi[1]].astype(float), sigma_img)
    im = (im - np.median(im)) / (np.std(im) + 1e-9)
    local = pts - lo
    nrm = _outward_normals(local)
    offs = np.arange(-half, half + 1e-6, 0.5)
    q = local[:, None, :] + offs[None, :, None] * nrm[:, None, :]
    prof = map_coordinates(im, [q[..., 0], q[..., 1]], order=1, mode='nearest')
    if avg_tan > 0:
        prof = gaussian_filter1d(prof, avg_tan / g['ds'], axis=0, mode='wrap')
    traced = []
    for cost in (-np.gradient(prof, axis=1), prof):
        cost = (cost - cost.mean(1, keepdims=True)) / (cost.std(1, keepdims=True) + 1e-9)
        traced.append(offs[_viterbi(cost, lam)])
    return 0.5 * (traced[0] + traced[1])


# ── Scoring ──────────────────────────────────────────────────────────────────

def profile_candidates(contour, image, hint_point, cfg):
    """All cross-section candidates with their named score terms."""
    g = contour_geometry(contour, cfg.SCAR_PROFILE_SMOOTHING)
    profiles, L = end_profiles(g)
    proj = g['proj']
    mid_mask = np.abs(proj - (proj.min() + proj.max()) / 2) < 0.2 * L
    W = (np.percentile(g['trans'][mid_mask], 95) - np.percentile(g['trans'][mid_mask], 5)
         if mid_mask.sum() > 4 else 50.0)
    offsets = (membrane_offsets(g, image, cfg.SCAR_MEMBRANE_SEARCH_PX, cfg.SCAR_MEMBRANE_SMOOTHNESS)
               if image is not None else np.zeros(len(g['pts'])))
    poles = [g['center'] + proj.min() * g['axis'], g['center'] + proj.max() * g['axis']]
    hint = np.zeros(2)
    if hint_point is not None:
        i = int(np.argmin([np.linalg.norm(p - np.asarray(hint_point)) for p in poles]))
        hint[i], hint[1 - i] = 1.0, -1.0
    mu = cfg.SCAR_PRIOR_OFFSET + cfg.SCAR_PRIOR_SLOPE * L / W
    clip = cfg.SCAR_MEMBRANE_CLIP_PX
    cands = []
    for p in profiles:
        if len(p['d']) == 0:
            continue
        u = p['d'] / W
        a = _sample_periodic(offsets, p['i1'])
        b = _sample_periodic(offsets, p['i2'])
        a = np.clip(a - np.median(a), -clip, clip)
        b = np.clip(b - np.median(b), -clip, clip)
        raw = p['w'] / W + (a + b) / W
        width = gaussian_filter1d(raw, cfg.SCAR_WIDTH_SMOOTH_PX)
        ridge = -gaussian_filter1d(raw, cfg.SCAR_RIDGE_SCALE_PX, order=2) * W
        terms = dict(prior=-0.5 * ((u - mu) / cfg.SCAR_PRIOR_SD) ** 2 * cfg.SCAR_PRIOR_WEIGHT,
                     hint=cfg.SCAR_HINT_WEIGHT * hint[p['end']] * np.ones_like(u),
                     membrane_width=cfg.SCAR_MEMBRANE_WIDTH_WEIGHT * width,
                     membrane_ridge=cfg.SCAR_MEMBRANE_RIDGE_WEIGHT * ridge)
        score = sum(terms.values())
        keep = (u >= cfg.SCAR_MIN_DISTANCE_W) & (u <= cfg.SCAR_MAX_DISTANCE_W)
        for j in np.flatnonzero(keep):
            cands.append(dict(points=(p['p1'][j], p['p2'][j]), score=float(score[j]),
                              end=int(p['end']), distance_px=float(p['d'][j]),
                              distance_w=float(u[j]), width=float(p['w'][j]),
                              terms={k: float(v[j]) for k, v in terms.items()},
                              features=dict(u=float(u[j]), aspect=float(L / W), hint=float(hint[p['end']]),
                                            membrane_width=float(width[j]), membrane_ridge=float(ridge[j])),
                              match_type='profile'))
    info = dict(length=float(L), mid_width=float(W), prior_mean_w=float(mu),
                smooth_pts=g['pts'], kappa=g['kappa'], center=g['center'], axis=g['axis'],
                long_norm=(proj - proj.min()) / max(L, 1e-10), avg_width=float(W),
                membrane_offsets=offsets, used_image=image is not None)
    return cands, info


def confidence(cands, best, radius):
    """Softmax probability mass within *radius* px of the chosen candidate."""
    s = np.array([c['score'] for c in cands])
    p = np.exp(s - s.max())
    p /= p.sum()
    mids = np.array([np.mean(c['points'], axis=0) for c in cands])
    chosen = np.mean(best['points'], axis=0)
    return float(p[np.linalg.norm(mids - chosen, axis=1) <= radius].sum())


# ── Bilateral windowed membrane-curvature mode ───────────────────────────────
#
# Alternative scorer (SCAR_DETECTOR_MODE = 'bilateral'): at each cross-section,
# average the curvature of the image-traced membrane over an axial window on
# each flank, then combine the two flanks (SCAR_BILATERAL_COMBINE: 'min' =
# both flanks must bulge; 'sum' = mean of both flanks, tolerant of lopsided
# scars; 'max' = either flank). Curvature of the Cellpose mask alone does not
# locate the scar; the membrane traced in the image does.

def darkest_membrane_offsets(g, image, half=5.0, sigma_img=1.0, median=5):
    """Sub-pixel offset (px, + outward) of the darkest pixel along each normal."""
    from scipy.ndimage import median_filter
    pts = g['pts']
    lo = np.maximum(0, np.floor(pts.min(0)).astype(int) - CROP_MARGIN)
    hi = np.minimum(image.shape, np.ceil(pts.max(0)).astype(int) + CROP_MARGIN + 1)
    im = gaussian_filter(image[lo[0]:hi[0], lo[1]:hi[1]].astype(float), sigma_img)
    im = (im - np.median(im)) / (np.std(im) + 1e-9)
    local = pts - lo
    nrm = _outward_normals(local)
    step = 0.5
    offs = np.arange(-half, half + 1e-6, step)
    q = local[:, None, :] + offs[None, :, None] * nrm[:, None, :]
    prof = map_coordinates(im, [q[..., 0], q[..., 1]], order=1, mode='nearest')
    j = np.clip(np.argmin(prof, axis=1), 1, len(offs) - 2)
    r = np.arange(len(j))
    a, b, c = prof[r, j - 1], prof[r, j], prof[r, j + 1]
    den = a - 2 * b + c
    delta = np.where(np.abs(den) > 1e-9, 0.5 * (a - c) / den, 0.0)
    off = offs[j] + np.clip(delta, -1, 1) * step
    return median_filter(off, median, mode='wrap') if median > 1 else off


def membrane_kink(g, off, scale=4.0, base=10.0, highpass=15.0, clip=3.0):
    """Curvature of the image-traced membrane minus the smooth cell-shape curvature."""
    ds, k = g['ds'], g['kappa']
    offc = np.clip(off - gaussian_filter1d(off, highpass / ds, mode='wrap'), -clip, clip)
    h = gaussian_filter1d(offc, scale / ds, mode='wrap')
    hpp = (np.roll(h, -1) - 2 * h + np.roll(h, 1)) / ds ** 2
    return gaussian_filter1d(k, scale / ds, mode='wrap') - hpp - gaussian_filter1d(k, base / ds, mode='wrap')


def bilateral_candidates(contour, image, hint_point, cfg):
    from scipy.ndimage import uniform_filter1d
    g = contour_geometry(contour, cfg.SCAR_PROFILE_SMOOTHING)
    profiles, L = end_profiles(g)
    proj = g['proj']
    tw = g['trans'][np.abs(proj - (proj.min() + proj.max()) / 2) < 0.2 * L]
    W = np.percentile(tw, 95) - np.percentile(tw, 5) if len(tw) > 4 else 50.0
    off = (darkest_membrane_offsets(g, image, cfg.SCAR_BILATERAL_SEARCH_PX) if image is not None
           else np.zeros(len(g['pts'])))
    kink = membrane_kink(g, off, cfg.SCAR_BILATERAL_SCALE_PX, cfg.SCAR_BILATERAL_BASELINE_PX)
    poles = [g['center'] + proj.min() * g['axis'], g['center'] + proj.max() * g['axis']]
    hint_end = None
    if hint_point is not None:
        hint_end = int(np.argmin([np.linalg.norm(p - np.asarray(hint_point)) for p in poles]))
    mu = cfg.SCAR_BILATERAL_PRIOR_OFFSET + cfg.SCAR_BILATERAL_PRIOR_SLOPE * L / W
    combine = cfg.SCAR_BILATERAL_COMBINE
    cands = []
    for p in profiles:
        if len(p['d']) == 0 or (hint_end is not None and p['end'] != hint_end):
            continue
        d = p['d']
        flanks = np.stack([_sample_periodic(kink, p['i1']), _sample_periodic(kink, p['i2'])])
        win = max(1, int(round(cfg.SCAR_BILATERAL_WINDOW_W * W)))
        if win > 1:
            flanks = uniform_filter1d(flanks, win, axis=1, mode='nearest')
        if combine == 'min':
            signal = flanks.min(0)
        elif combine == 'max':
            signal = flanks.max(0)
        elif combine == 'sum':
            signal = flanks.mean(0)
        else:
            raise ValueError(f'Unknown SCAR_BILATERAL_COMBINE: {combine}')
        keep = (d >= cfg.SCAR_BILATERAL_MIN_DISTANCE_W * W) & (d <= 0.5 * L)
        if not keep.any():
            continue
        z = (signal - signal[keep].mean()) / (signal[keep].std() + 1e-12)
        prior = -cfg.SCAR_BILATERAL_PRIOR_WEIGHT * 0.5 * ((d / W - mu) / cfg.SCAR_BILATERAL_PRIOR_SD) ** 2
        blunt = np.zeros_like(d)
        if hint_end is None and cfg.SCAR_BILATERAL_BLUNTNESS_WEIGHT > 0:
            full = p['w'] / W >= 0.95
            knee = d[np.argmax(full)] if full.any() else d[-1]
            blunt[:] = -cfg.SCAR_BILATERAL_BLUNTNESS_WEIGHT * knee / (0.1 * W)
        score = z + prior + blunt
        for j in np.flatnonzero(keep):
            cands.append(dict(points=(p['p1'][j], p['p2'][j]), score=float(score[j]), end=int(p['end']),
                              distance_px=float(d[j]), distance_w=float(d[j] / W), width=float(p['w'][j]),
                              terms=dict(bilateral_kink=float(z[j]), prior=float(prior[j]),
                                         end_bluntness=float(blunt[j])),
                              match_type='bilateral'))
    info = dict(length=float(L), mid_width=float(W), prior_mean_w=float(mu), hint_end=hint_end,
                smooth_pts=g['pts'], kappa=g['kappa'], center=g['center'], axis=g['axis'],
                long_norm=(proj - proj.min()) / max(L, 1e-10), avg_width=float(W),
                membrane_offsets=off, membrane_kink=kink, used_image=image is not None)
    return cands, info


def bilateral_confidence(cands, best, W, separation_w, hinted):
    """Best score + margin over the best alternative at least separation_w*W away
    (or at the other end), + 1 if a neighbor hint fixed the end. In z units."""
    others = [c['score'] for c in cands
              if c['end'] != best['end'] or abs(c['distance_px'] - best['distance_px']) >= separation_w * W]
    margin = best['score'] - max(others) if others else 3.0
    return float(best['score'] + margin + (1.0 if hinted else 0.0))
