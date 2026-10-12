"""Compare manual cell tracks (validation/packet_neto) with BS-Detector output.

Inputs
  * one or more CSVs written by BS_track_cell.ijm (or the manual template):
    per cell and frame, the scar midpoint and the new and old pole tips in
    image pixels (x = column, y = row)
  * the pipeline's measurements CSV for the same movie (export_csv), which
    carries scar_midpoint, new_pole_point and old_pole_point as [row, col]

Each annotated frame is matched to the pipeline cell in that frame whose two
pole tips lie closest to the annotated tips (either orientation), so the
comparison does not depend on the pipeline's lineage names. Lengths use the
pipeline's convention: straight line from the scar midpoint to each pole tip.

New end take-off (NETO) is estimated per cell by fitting
    L_new(t) = a                 for t < t0
             = a + b (t - t0)    for t >= t0
to the new-end length series (least squares over candidate t0), separately for
the annotation and for the pipeline.

    python -m validation.compare_compartments ANNOTATION_CSV_OR_DIR \
        --measurements measurements.csv --frame-minutes 5 --out results/neto
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _point_xy(row, prefix):
    x, y = _num(row.get(f'{prefix}_x')), _num(row.get(f'{prefix}_y'))
    return None if x is None or y is None else np.array([y, x])  # -> (row, col)


def load_annotations(path):
    files = sorted(Path(path).glob('*.csv')) if Path(path).is_dir() else [Path(path)]
    rows = []
    for f in files:
        for r in csv.DictReader(open(f, newline='')):
            if r.get('movie', '').startswith('EXAMPLE') or not r.get('cell_id'):
                continue
            rows.append(dict(file=f.name, movie=r['movie'], cell=r['cell_id'], scorer=r.get('scorer', ''),
                             frame=int(float(r['frame_index'])), status=r.get('status', 'ok'),
                             notes=r.get('notes', ''), scar=_point_xy(r, 'scar'),
                             new=_point_xy(r, 'new_pole'), old=_point_xy(r, 'old_pole')))
    return rows


def _parse_point(text):
    if text in (None, '', 'None'):
        return None
    p = np.asarray(json.loads(text), dtype=float)
    return p if p.shape == (2,) and np.isfinite(p).all() else None


def load_measurements(path):
    by_frame = defaultdict(list)
    for r in csv.DictReader(open(path, newline='')):
        new, old = _parse_point(r.get('new_pole_point')), _parse_point(r.get('old_pole_point'))
        if new is None or old is None:
            continue
        by_frame[int(float(r['frame']))].append(dict(
            name=r.get('cell_name'), new=new, old=old, scar=_parse_point(r.get('scar_midpoint')),
            new_len=_num(r.get('new_end_length')), old_len=_num(r.get('old_end_length')),
            confidence=_num(r.get('scar_confidence')), review=r.get('scar_needs_review')))
    if not by_frame:
        raise ValueError('Measurements CSV has no new_pole_point/old_pole_point columns; '
                         're-export with the current export_csv defaults')
    return by_frame


def match(ann, candidates, max_pole_px):
    """Pipeline cell whose poles best match the annotated poles; None if too far."""
    best, best_d, same_orientation = None, np.inf, True
    for c in candidates:
        d_same = max(np.linalg.norm(ann['new'] - c['new']), np.linalg.norm(ann['old'] - c['old']))
        d_swap = max(np.linalg.norm(ann['new'] - c['old']), np.linalg.norm(ann['old'] - c['new']))
        d = min(d_same, d_swap)
        if d < best_d:
            best, best_d, same_orientation = c, d, d_same <= d_swap
    if best is None or best_d > max_pole_px:
        return None, best_d, None
    return best, best_d, same_orientation


def fit_neto(t, L, min_points=6):
    """Hinge fit; returns dict(t0, a, slope, sse, sse_line) or None."""
    t, L = np.asarray(t, float), np.asarray(L, float)
    ok = np.isfinite(t) & np.isfinite(L)
    t, L = t[ok], L[ok]
    if len(t) < min_points:
        return None
    order = np.argsort(t)
    t, L = t[order], L[order]
    best = None
    for t0 in t[1:-2]:   # at least one point before and two after the hinge
        X = np.c_[np.ones_like(t), np.maximum(0.0, t - t0)]
        coef, *_ = np.linalg.lstsq(X, L, rcond=None)
        sse = float(np.sum((X @ coef - L) ** 2))
        if coef[1] > 0 and (best is None or sse < best['sse']):
            best = dict(t0=float(t0), a=float(coef[0]), slope=float(coef[1]), sse=sse)
    if best is None:
        return None
    X = np.c_[np.ones_like(t), t]
    coef, *_ = np.linalg.lstsq(X, L, rcond=None)
    best['sse_line'] = float(np.sum((X @ coef - L) ** 2))
    return best


def compare(annotations, measurements, frame_minutes=None, max_pole_px=20.0):
    frames = []
    for a in annotations:
        rec = dict(cell=a['cell'], movie=a['movie'], frame=a['frame'], status=a['status'], notes=a['notes'],
                   matched=False, pole_match_px=None, same_pole_labels=None,
                   ann_new_len=None, ann_old_len=None, det_new_len=None, det_old_len=None,
                   scar_error_px=None, new_len_error_px=None, det_cell=None)
        if a['new'] is None or a['old'] is None:
            frames.append(rec)
            continue
        if a['scar'] is not None:
            rec['ann_new_len'] = float(np.linalg.norm(a['scar'] - a['new']))
            rec['ann_old_len'] = float(np.linalg.norm(a['scar'] - a['old']))
        det, d, same = match(a, measurements.get(a['frame'], []), max_pole_px)
        rec['pole_match_px'] = None if not np.isfinite(d) else round(float(d), 2)
        if det is not None:
            rec.update(matched=True, same_pole_labels=same, det_cell=det['name'],
                       det_confidence=det['confidence'], det_review=det['review'])
            if det['scar'] is not None:
                # Lengths measured against the annotator's new/old pole identity.
                new_tip, old_tip = (det['new'], det['old']) if same else (det['old'], det['new'])
                rec['det_new_len'] = float(np.linalg.norm(det['scar'] - new_tip))
                rec['det_old_len'] = float(np.linalg.norm(det['scar'] - old_tip))
                if a['scar'] is not None:
                    rec['scar_error_px'] = float(np.linalg.norm(det['scar'] - a['scar']))
                    rec['new_len_error_px'] = rec['det_new_len'] - rec['ann_new_len']
        frames.append(rec)

    cells = []
    for cell in sorted({f['cell'] for f in frames}):
        fs = sorted([f for f in frames if f['cell'] == cell], key=lambda f: f['frame'])
        t = np.array([f['frame'] for f in fs], float)
        scale = frame_minutes or 1.0
        ann = fit_neto(t, [np.nan if f['ann_new_len'] is None else f['ann_new_len'] for f in fs])
        det = fit_neto(t, [np.nan if f['det_new_len'] is None else f['det_new_len'] for f in fs])
        errs = [f['new_len_error_px'] for f in fs if f['new_len_error_px'] is not None]
        scar = [f['scar_error_px'] for f in fs if f['scar_error_px'] is not None]
        matched = [f for f in fs if f['matched']]
        cells.append(dict(
            cell=cell, frames=len(fs), first_frame=int(t.min()), last_frame=int(t.max()),
            ended=fs[-1]['status'], matched=len(matched),
            same_pole_labels=sum(bool(f['same_pole_labels']) for f in matched),
            scar_within_5px=sum(e <= 5 for e in scar), scar_compared=len(scar),
            scar_median_px=round(float(np.median(scar)), 2) if scar else None,
            new_len_mae_px=round(float(np.mean(np.abs(errs))), 2) if errs else None,
            new_len_bias_px=round(float(np.mean(errs)), 2) if errs else None,
            neto_annotated=None if ann is None else round(ann['t0'] * scale, 1),
            neto_detector=None if det is None else round(det['t0'] * scale, 1),
            neto_difference=None if ann is None or det is None else round((det['t0'] - ann['t0']) * scale, 1),
            growth_annotated=None if ann is None else round(ann['slope'] / scale, 3),
            growth_detector=None if det is None else round(det['slope'] / scale, 3)))
    return frames, cells


def write_outputs(frames, cells, out, frame_minutes=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (('per_frame.csv', frames), ('per_cell.csv', cells)):
        with open(out / name, 'w', newline='') as f:
            w = csv.DictWriter(f, list(dict.fromkeys(k for r in rows for k in r)))
            w.writeheader()
            w.writerows(rows)
    unit = 'min' if frame_minutes else 'frames'
    scar = [f['scar_error_px'] for f in frames if f['scar_error_px'] is not None]
    errs = [f['new_len_error_px'] for f in frames if f['new_len_error_px'] is not None]
    matched = [f for f in frames if f['matched']]
    lines = ['# Compartment lengths through time: detector vs manual tracks', '',
             f'- Annotated cell-frames: {len(frames)}; matched to a detector cell: {len(matched)}',
             f'- Detector used the same new/old pole labels in {sum(bool(f["same_pole_labels"]) for f in matched)}'
             f'/{len(matched)} matched frames']
    if scar:
        lines.append(f'- Scar midpoint within 5 px: {sum(e <= 5 for e in scar)}/{len(scar)} '
                     f'(median {np.median(scar):.1f} px)')
    if errs:
        lines.append(f'- New-end length error: mean absolute {np.mean(np.abs(errs)):.1f} px, '
                     f'bias {np.mean(errs):+.1f} px (detector minus manual)')
    lines += ['', f'| Cell | Frames | Same pole labels | Scar ≤5 px | New-length MAE px | '
                  f'NETO manual ({unit}) | NETO detector ({unit}) | Difference |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for c in cells:
        lines.append(f"| {c['cell']} | {c['frames']} | {c['same_pole_labels']}/{c['matched']} | "
                     f"{c['scar_within_5px']}/{c['scar_compared']} | {c['new_len_mae_px']} | "
                     f"{c['neto_annotated']} | {c['neto_detector']} | {c['neto_difference']} |")
    (out / 'SUMMARY.md').write_text('\n'.join(lines) + '\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n = len(cells)
    if n:
        cols = min(3, n)
        fig, axs = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(5 * cols, 3.4 * np.ceil(n / cols)), squeeze=False)
        scale = frame_minutes or 1.0
        for ax, c in zip(axs.flat, cells):
            fs = sorted([f for f in frames if f['cell'] == c['cell']], key=lambda f: f['frame'])
            t = np.array([f['frame'] for f in fs]) * scale
            for key, label, col in (('ann_new_len', 'manual', 'tab:red'), ('det_new_len', 'detector', 'tab:blue')):
                ax.plot(t, [np.nan if f[key] is None else f[key] for f in fs], 'o-', ms=3, color=col, label=label)
            for key, col in (('neto_annotated', 'tab:red'), ('neto_detector', 'tab:blue')):
                if c[key] is not None:
                    ax.axvline(c[key], color=col, ls=':')
            ax.set(title=f"cell {c['cell']}", xlabel=f'time ({unit})', ylabel='new-end length (px)')
            ax.legend(fontsize=8)
        for ax in list(axs.flat)[n:]:
            ax.axis('off')
        fig.tight_layout()
        fig.savefig(out / 'new_end_length.png', dpi=120)
        plt.close(fig)
    return out / 'SUMMARY.md'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('annotations', help='CSV from BS_track_cell.ijm, or a folder of them')
    p.add_argument('--measurements', required=True, help='pipeline measurements CSV for the same movie')
    p.add_argument('--frame-minutes', type=float, default=None, help='minutes between frames (for NETO in minutes)')
    p.add_argument('--max-pole-px', type=float, default=20.0, help='max pole distance to accept a match')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    frames, cells = compare(load_annotations(a.annotations), load_measurements(a.measurements),
                            a.frame_minutes, a.max_pole_px)
    print(write_outputs(frames, cells, a.out, a.frame_minutes).read_text())


if __name__ == '__main__':
    main()
