"""Score a detector configuration on frozen cell records (see scar_cells.py).

GT is used only after detection, to measure error. The evaluator mirrors the
production call in CellProcessor.process_cell: the neighbor-derived pole hint
is passed only for 'high'/'medium' neighbor confidence, and the frame image
is passed for detectors that use it.

    python -m validation.scar_eval validation/runs/scar_v3/cells_cuda.pkl \
        --overrides '{"SCAR_DETECTOR_MODE": "legacy"}' --per-cell
"""
import argparse
import json
import pickle
from collections import defaultdict

import numpy as np

from pombe_tracker.config import Config
from pombe_tracker.detection import BirthScarDetector


def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def make_config(overrides=None):
    cfg = Config()
    for k, v in (overrides or {}).items():
        if not hasattr(cfg, k):
            raise ValueError(f'Unknown configuration: {k}')
        setattr(cfg, k, v)
    return cfg


def hint_for(rec):
    if rec.get('new_pole_idx') is not None and rec.get('neighbor_conf') in ('high', 'medium'):
        return rec['endpoints'][rec['new_pole_idx']]
    return None


def detect_record(rec, cfg, detector_cls=BirthScarDetector):
    """Run the detector in crop coordinates; return (midpoint, debug, origin)."""
    origin = np.asarray(rec['crop_origin'], dtype=float)
    hint = hint_for(rec)
    detector = detector_cls(cfg)
    pair, dbg = detector.detect(rec['contour'] - origin,
                                new_pole_point=None if hint is None else hint - origin,
                                image=rec['crop'])
    mid = None if pair is None else np.mean(pair, axis=0) + origin
    return mid, dbg, origin, detector


def evaluate(records, overrides=None, detector_cls=BirthScarDetector):
    cfg = make_config(overrides)
    rows = []
    for rec in records:
        row = dict(key=rec['key'], package=rec['package'], video=rec['video'], frame=rec['frame'],
                   cell=rec['cell'], status=rec['status'], error=None, oracle=None)
        if rec['status'] == 'ok':
            mid, dbg, origin, detector = detect_record(rec, cfg, detector_cls)
            gt = rec['gt']
            if mid is not None:
                row['error'] = float(np.linalg.norm(mid - gt))
                row['pred'] = mid
            cands = dbg.get('scar_candidates', [])
            if cands:
                mids = [np.mean(detector._map_to_original(c['points'], rec['contour'] - origin), axis=0) + origin
                        for c in cands]
                row['oracle'] = float(min(np.linalg.norm(m - gt) for m in mids))
            row['n_candidates'] = len(cands)
            row['hint'] = hint_for(rec) is not None
            row['debug'] = dbg
        rows.append(row)
    return rows


def summarize(rows):
    usable = [r for r in rows if r['status'] != 'no cached mask']
    n = len(usable)
    errors = np.array([np.inf if r['error'] is None else r['error'] for r in usable])
    finite = errors[np.isfinite(errors)]
    oracle = np.array([np.inf if r['oracle'] is None else r['oracle'] for r in usable])
    return dict(n=n, within_3=int((errors <= 3).sum()), within_5=int((errors <= 5).sum()),
                within_10=int((errors <= 10).sum()), frac_5=float((errors <= 5).mean()) if n else None,
                median=float(np.median(finite)) if len(finite) else None,
                mean=float(finite.mean()) if len(finite) else None,
                p90=float(np.percentile(finite, 90)) if len(finite) else None,
                oracle_5=int((oracle <= 5).sum()))


def report(rows, per_cell=False, groups=('package', 'video')):
    lines = []
    s = summarize(rows)
    lines.append(f"ALL      n={s['n']:3d}  <=5px {s['within_5']:3d} ({s['frac_5']:.1%})  <=3 {s['within_3']}  "
                 f"<=10 {s['within_10']}  median {s['median']:.2f}  mean {s['mean']:.2f}  p90 {s['p90']:.2f}  "
                 f"oracle<=5 {s['oracle_5']}")
    for g in groups:
        by = defaultdict(list)
        for r in rows:
            by[r[g]].append(r)
        for k in sorted(by):
            s = summarize(by[k])
            if s['n']:
                lines.append(f"  {g}={k[-22:]:22s} n={s['n']:3d} <=5px {s['within_5']:3d} ({s['frac_5']:.0%}) "
                             f"median {s['median']:.2f} mean {s['mean']:.2f}")
    if per_cell:
        for r in sorted(rows, key=lambda r: -1 if r['error'] is None else r['error'], reverse=True):
            if r['status'] == 'no cached mask':
                continue
            e = 'None' if r['error'] is None else f"{r['error']:.2f}"
            o = 'None' if r['oracle'] is None else f"{r['oracle']:.2f}"
            lines.append(f"    {r['package']} {r['video'][-18:]} f{r['frame']:<4d} {r['cell']:5s} err {e:>7s} "
                         f"oracle {o:>6s} hint {r.get('hint')} {r['status'] if r['status'] != 'ok' else ''}")
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('cells')
    p.add_argument('--overrides', default='{}')
    p.add_argument('--per-cell', action='store_true')
    a = p.parse_args()
    rows = evaluate(load(a.cells), json.loads(a.overrides))
    print(report(rows, a.per_cell))


if __name__ == '__main__':
    main()
