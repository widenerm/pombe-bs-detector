"""Fit / re-fit the profile-mode scar weights from annotated cells.

Use this to recalibrate for a new strain or mutant: annotate a few dozen
cells (same protocol as GROUND_TRUTH_PROTOCOL.md), build records with
validation/scar_cells.py, then

    python -m validation.scar_fit validation/runs/scar_v3/cells_h200.pkl

It prints cross-validated accuracy (leave-one-video-out and
leave-one-package-out) and the fitted values as Config lines to paste into
pombe_tracker/config.py (or set on a Config instance).

Model: conditional logit over each cell's cross-section candidates,
    score = a*u + b*u^2 + c*u*(L/W) + hint_w*hint + width_w*width + ridge_w*ridge
fitted with soft labels (Gaussian, sigma 2 px, around the annotated point)
and light L2. The u terms are reported as the equivalent Gaussian prior
(SCAR_PRIOR_OFFSET, SCAR_PRIOR_SLOPE, SCAR_PRIOR_SD).
"""
import argparse
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize

from validation.scar_eval import hint_for, load, make_config
from pombe_tracker.scar_profile import profile_candidates

NAMES = ['u', 'u2', 'u_x_aspect', 'hint', 'membrane_width', 'membrane_ridge']


def design(records, cfg):
    data = []
    for rec in records:
        if rec['status'] != 'ok':
            continue
        o = np.asarray(rec['crop_origin'], float)
        hint = hint_for(rec)
        cands, _ = profile_candidates(rec['contour'] - o, rec['crop'], None if hint is None else hint - o, cfg)
        f = [c['features'] for c in cands]
        X = np.array([[d['u'], d['u'] ** 2, d['u'] * d['aspect'], d['hint'], d['membrane_width'],
                       d['membrane_ridge']] for d in f])
        mids = np.array([np.mean(c['points'], axis=0) for c in cands]) + o
        data.append(dict(rec=rec, X=X, mids=mids))
    return data


def fit(data, l2=0.01, sigma=2.0):
    Xs = np.vstack([d['X'] for d in data])
    mu, sd = Xs.mean(0), Xs.std(0) + 1e-9
    ys = []
    for d in data:
        dist = np.linalg.norm(d['mids'] - d['rec']['gt'], axis=1)
        y = np.exp(-0.5 * (dist / sigma) ** 2)
        ys.append(y / y.sum() if y.sum() > 0 else np.full(len(y), 1 / len(y)))

    def nll(w):
        tot, grad = 0.0, np.zeros(len(w))
        for d, y in zip(data, ys):
            Z = (d['X'] - mu) / sd
            s = Z @ w
            s -= s.max()
            p = np.exp(s)
            p /= p.sum()
            tot -= np.sum(y * np.log(p + 1e-300))
            grad -= Z.T @ (y - p)
        return tot + l2 * w @ w, grad + 2 * l2 * w

    w = minimize(nll, np.zeros(Xs.shape[1]), jac=True, method='L-BFGS-B').x
    return dict(zip(NAMES, w / sd))  # raw-unit weights (offset is irrelevant for argmax)


def predict(weights, d):
    s = d['X'] @ np.array([weights[n] for n in NAMES])
    return d['mids'][int(np.argmax(s))]


def as_config(w):
    """Express raw weights as the named Config parameters."""
    if w['u2'] >= 0:
        raise ValueError('Fitted distance prior is not concave; collect more annotated cells')
    sd = np.sqrt(-1 / (2 * w['u2']))
    return dict(SCAR_PRIOR_OFFSET=-w['u'] / (2 * w['u2']), SCAR_PRIOR_SLOPE=-w['u_x_aspect'] / (2 * w['u2']),
                SCAR_PRIOR_SD=sd, SCAR_PRIOR_WEIGHT=1.0, SCAR_HINT_WEIGHT=w['hint'],
                SCAR_MEMBRANE_WIDTH_WEIGHT=w['membrane_width'], SCAR_MEMBRANE_RIDGE_WEIGHT=w['membrane_ridge'])


def cross_validate(data, key, **kw):
    groups = defaultdict(list)
    for i, d in enumerate(data):
        groups[d['rec'][key]].append(i)
    err = np.zeros(len(data))
    for g, test in groups.items():
        train = [data[i] for i in range(len(data)) if i not in test]
        w = fit(train, **kw)
        for i in test:
            err[i] = np.linalg.norm(predict(w, data[i]) - data[i]['rec']['gt'])
    return err


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('cells')
    p.add_argument('--l2', type=float, default=0.01)
    p.add_argument('--sigma', type=float, default=2.0)
    a = p.parse_args()
    data = design(load(a.cells), make_config({'SCAR_DETECTOR_MODE': 'profile'}))
    for key in ('video', 'package'):
        if len({d['rec'][key] for d in data}) > 1:
            e = cross_validate(data, key, l2=a.l2, sigma=a.sigma)
            print(f'CV by {key}: within 5 px {(e <= 5).sum()}/{len(e)}  within 3 px {(e <= 3).sum()}  median {np.median(e):.2f}')
    w = fit(data, l2=a.l2, sigma=a.sigma)
    e = np.array([np.linalg.norm(predict(w, d) - d['rec']['gt']) for d in data])
    print(f'in-sample: within 5 px {(e <= 5).sum()}/{len(e)}')
    for k, v in as_config(w).items():
        print(f'    {k} = {v:.3f}')


if __name__ == '__main__':
    main()
