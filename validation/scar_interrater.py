"""Second-scorer comparison: two scorers on the same cells, and the detector against both.

The first scorer's annotations come from the frozen cell records (scar_cells.py);
the second scorer's from their filled-in ground-truth templates. Rows match on
(video_id, frame, cell_name). Pole A/B letters are resolved to physical cell
ends by matching each scorer's two compartment lengths to the Cellpose pole
endpoints, so a scorer who labels a horizontal cell's poles left/right is not
counted as disagreeing.

    python -m validation.scar_interrater validation/runs/scar_v3/cells_h200.pkl \
        validation/annotations/dev1_annotator_b.csv \
        validation/annotations/dev2_annotator_b.csv \
        --names 'Annotator A' 'Annotator B' --out validation/reports/scar_v3/per_cell_interrater.csv
"""
import argparse
import csv
from collections import defaultdict

import numpy as np
from scipy.stats import mannwhitneyu

from validation.scar_batch import write_csv
from validation.scar_eval import evaluate, load, make_config
from validation.scar_fit import design, fit, predict, as_config

MODES = {'profile': {}, 'bilateral': {'SCAR_DETECTOR_MODE': 'bilateral'}, 'legacy': {'SCAR_DETECTOR_MODE': 'legacy'}}


def read_second(paths):
    rows = {}
    for path in paths:
        for r in csv.DictReader(open(path)):
            if r['scorer_initials'].upper() == 'EXAMPLE':
                continue
            scar = (np.array([float(r['gt_scar_mid_y_px']), float(r['gt_scar_mid_x_px'])])
                    if r['gt_scar_mid_x_px'] else None)
            rows[(r['video_id'], int(r['frame']), r['cell_name'])] = dict(
                gt=scar, notes=r['notes'], gt_area=float(r['gt_area_px2']),
                gt_new_pole_guess=r['gt_new_pole_guess'],
                gt_poleA=float(r['gt_poleA_length_px'] or 'nan'), gt_poleB=float(r['gt_poleB_length_px'] or 'nan'))
    return rows


def axis_frame(rec):
    """Cell centre, unit long axis and unit normal from the Cellpose pole endpoints."""
    e = rec['endpoints']
    a = (e[1] - e[0]) / np.linalg.norm(e[1] - e[0])
    return (e[0] + e[1]) / 2, a, np.array([-a[1], a[0]])


def along_across(rec, p):
    c, a, n = axis_frame(rec)
    return float((p - c) @ a), float((p - c) @ n)


def off_midline(rec, p):
    """Distance of p from the midpoint of the cell's cross-section through p."""
    c, a, n = axis_frame(rec)
    s, t = along_across(rec, p)
    q = rec['contour'] - c
    band = np.abs(q @ a - s) < 2
    return abs(t - ((q @ n)[band].max() + (q @ n)[band].min()) / 2) if band.sum() > 1 else np.nan


def ends(rec, ann):
    """Map a scorer's pole A/B to Cellpose endpoints; return new-end index and per-end lengths."""
    if ann['gt'] is None or np.isnan(ann['gt_poleA']):
        return None
    d = [np.linalg.norm(ann['gt'] - e) for e in rec['endpoints']]
    la, lb = ann['gt_poleA'], ann['gt_poleB']
    a = 0 if abs(d[0] - la) + abs(d[1] - lb) <= abs(d[1] - la) + abs(d[0] - lb) else 1
    new = a if ann['gt_new_pole_guess'].strip().upper() == 'A' else 1 - a
    return dict(new=new, length={a: la, 1 - a: lb}, upper_is_a=a == int(np.argmin(rec['endpoints'][:, 0])),
                new_is_nearer=new == int(np.argmin(d)))


def dist(a, b):
    return np.inf if a is None or b is None else float(np.linalg.norm(a - b))


def tally(e):
    e = np.asarray(e, float)
    return (f"{(e <= 5).sum():2d}/{len(e)} ({(e <= 5).mean():4.0%})  <=3 {(e <= 3).sum():2d}  "
            f"<=10 {(e <= 10).sum():2d}  median {np.median(e):5.2f}")


def auc(bad, score):
    bad = np.asarray(bad)
    return mannwhitneyu(score[bad], score[~bad]).statistic / (bad.sum() * (~bad).sum())


def cell_resample(rng, keys):
    """Bootstrap indices that resample physical cells, keeping each cell's frames together."""
    groups = defaultdict(list)
    for i, k in enumerate(keys):
        groups[(k[0], k[2])].append(i)
    groups = list(groups.values())
    return np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])


def held_out_predictions(data, key):
    out = {}
    for g in {d['rec'][key] for d in data}:
        w = fit([d for d in data if d['rec'][key] != g])
        out.update({d['rec']['key']: predict(w, d) for d in data if d['rec'][key] == g})
    return out


def gallery(path, cases, names):
    """Crops of disputed cells with both scorers' points and the detector's."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import textwrap
    from matplotlib.lines import Line2D
    styles = ((names[0], dict(marker='+', color='#ff8c1a', ms=14, mew=2)),
              (names[1], dict(marker='x', color='#33ccff', ms=11, mew=2)),
              ('detector', dict(marker='o', mfc='none', color='#ffe44d', ms=12, mew=1.5)))
    cols = 4
    rows = -(-len(cases) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4.5 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis('off')
    for ax, (rec, p1, p2, pd, title) in zip(axes.flat, cases):
        o, crop = np.asarray(rec['crop_origin'], float), rec['crop']
        lo, hi = np.percentile(crop, [1, 99.5])
        ax.imshow(crop, cmap='gray', vmin=lo, vmax=hi,
                  extent=(o[1] - .5, o[1] + crop.shape[1] - .5, o[0] + crop.shape[0] - .5, o[0] - .5))
        ax.plot(rec['contour'][:, 1], rec['contour'][:, 0], color='white', lw=.5, alpha=.5)
        for pt, (_, style) in zip((p1, p2, pd), styles):
            if pt is not None:
                ax.plot(pt[1], pt[0], ls='none', **style)
        ax.set_title('\n'.join(textwrap.fill(line, 52) for line in title.split('\n')), fontsize=8)
    axes.flat[0].legend([Line2D([], [], ls='none', **s) for _, s in styles], [n for n, _ in styles],
                        loc='lower left', fontsize=7, framealpha=.6)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('cells', help='frozen records holding the first scorer (scar_cells.py)')
    p.add_argument('second', nargs='+', help="second scorer's ground-truth CSVs")
    p.add_argument('--names', nargs=2, default=['first', 'second'])
    p.add_argument('--out', help='per-cell CSV')
    p.add_argument('--gallery', help='PNG of cells where the scorers disagree by >10 px, '
                                     'one left the scar blank, or the detector is >5 px from both')
    a = p.parse_args()
    n1, n2 = a.names
    recs = [r for r in load(a.cells) if r['status'] == 'ok']
    second = read_second(a.second)
    missing = [r['key'] for r in recs if r['key'] not in second]
    if missing:
        raise ValueError(f'{len(missing)} cells have no second-scorer row, e.g. {missing[0]}')
    first = {r['key']: dict(gt=r['gt'], notes=r['notes'], gt_area=r['gt_area'], gt_new_pole_guess=r['gt_new_pole_guess'],
                            gt_poleA=r['gt_poleA'], gt_poleB=r['gt_poleB']) for r in recs}
    second = {r['key']: second[r['key']] for r in recs}
    rec_by = {r['key']: r for r in recs}
    keys = [r['key'] for r in recs]
    both = [k for k in keys if second[k]['gt'] is not None]
    mean = {k: (first[k]['gt'] + second[k]['gt']) / 2 for k in both}
    d12 = {k: dist(first[k]['gt'], second[k]['gt']) for k in both}
    print(f'{len(keys)} cells; {n2} left {len(keys) - len(both)} scars blank: '
          + ', '.join(f'{k[2]} f{k[1]} ({second[k]["notes"]})' for k in keys if k not in both))

    print(f'\n## Scar midpoint agreement, {len(both)} cells both scored (within 5 px)')
    print(f'{n1} vs {n2}'.ljust(26), tally([d12[k] for k in both]))
    preds = {}
    for mode, overrides in MODES.items():
        rows = {r['key']: r for r in evaluate(recs, overrides)}
        preds[mode] = {k: (rows[k].get('pred'), (rows[k].get('debug') or {}).get('scar_confidence')) for k in keys}
        P = {k: v[0] for k, v in preds[mode].items()}
        for name, ref in ((n1, {k: first[k]['gt'] for k in both}), (n2, {k: second[k]['gt'] for k in both}),
                          ('mean of both', mean)):
            print(f'{mode} vs {name}'.ljust(26), tally([dist(P[k], ref[k]) for k in both]))
    P = {k: v[0] for k, v in preds['profile'].items()}
    conf = {k: v[1] for k, v in preds['profile'].items()}
    e1 = np.array([dist(P[k], first[k]['gt']) for k in both])
    e2 = np.array([dist(P[k], second[k]['gt']) for k in both])
    e12 = np.array([d12[k] for k in both])
    rng = np.random.default_rng(0)
    boot = [(lambda i: (e2[i] <= 5).mean() - (e12[i] <= 5).mean())(cell_resample(rng, both)) for _ in range(5000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f'{len({(k[0], k[2]) for k in both})} physical cells. Within 5 px, profile vs {n2} minus {n1} vs {n2}: '
          f'{(e2 <= 5).mean() - (e12 <= 5).mean():+.0%} (95% CI {lo:+.0%} to {hi:+.0%}, bootstrap over cells)')
    agree = [k for k in both if d12[k] <= 5]
    miss = [k for k in agree if dist(P[k], first[k]['gt']) > 5 and dist(P[k], second[k]['gt']) > 5]
    print(f'Scorers within 5 px of each other on {len(agree)} cells; profile >5 px from both on {len(miss)}: '
          + ', '.join(f'{k[2]} f{k[1]}' for k in miss))

    print('\n## Systematic offsets (cells where scorers are within 10 px)')
    close = [k for k in both if d12[k] <= 10]
    for name, x, y in ((f'{n2} - {n1}', second, first), (f'profile - {n1}', None, first), (f'profile - {n2}', None, second)):
        toward = []
        for k in close:
            s_ref = along_across(rec_by[k], y[k]['gt'])[0]
            s = along_across(rec_by[k], P[k] if x is None else x[k]['gt'])[0]
            toward.append((s - s_ref) * np.sign(s_ref))
        toward = np.array(toward)
        print(f'{name:18s} along the axis, + toward the pole: median {np.median(toward):+.2f} px, '
              f'|along| median {np.median(np.abs(toward)):.2f}')
    for name, pts in ((n1, {k: first[k]['gt'] for k in close}), (n2, {k: second[k]['gt'] for k in close}),
                      ('profile', {k: P[k] for k in close})):
        off = np.array([off_midline(rec_by[k], pts[k]) for k in close])
        print(f'{name:8s} distance from the cell midline: median {np.nanmedian(off):.2f} px, p90 {np.nanpercentile(off, 90):.2f}')

    print('\n## Per-measurer noise along the axis (three-cornered hat)')
    data = design(recs, make_config({'SCAR_DETECTOR_MODE': 'profile'}))
    held = held_out_predictions(data, 'video')
    core = [k for k in both if max(d12[k], dist(held[k], first[k]['gt']), dist(held[k], second[k]['gt'])) <= 10]
    for comp, label in ((0, 'along'), (1, 'across')):
        x = {w: np.array([along_across(rec_by[k], pt[k])[comp] for k in core]) for w, pt in
             ((n1, {k: first[k]['gt'] for k in core}), (n2, {k: second[k]['gt'] for k in core}), ('detector', held))}

        def hat(idx):
            v = {(i, j): np.var(x[i][idx] - x[j][idx]) for i in x for j in x if i < j}
            def pair(i, j): return v[(min(i, j), max(i, j))]
            names = list(x)
            return {w: (pair(w, o1) + pair(w, o2) - pair(o1, o2)) / 2
                    for w in names for o1, o2 in [[n for n in names if n != w]]}
        h = hat(np.arange(len(core)))
        boot = [hat(cell_resample(rng, core)) for _ in range(2000)]
        ci = {w: np.sqrt(np.clip(np.percentile([b[w] for b in boot], [2.5, 97.5]), 0, None)) for w in x}
        print(f'{label:6s} SD ({len(core)} cells, all three within 10 px; detector held out by video): '
              + '  '.join(f'{w} {np.sqrt(max(h[w], 0)):.2f} [{ci[w][0]:.1f}-{ci[w][1]:.1f}]' for w in x))
        if comp == 0:
            sd_along = {w: np.sqrt(max(h[w], 0)) for w in x}
            ci_along = ci
        else:
            sd_across = {w: np.sqrt(max(h[w], 0)) for w in x}
    z = np.random.default_rng(1).normal(size=(2, 200000))
    for w in (n1, n2):
        within = [(((z[0] * a) ** 2 + (z[1] * sd_across[w]) ** 2) <= 25).mean() for a in (*ci_along[w], sd_along[w])]
        print(f'A detector exactly on the true scar would be within 5 px of {w} in {within[2]:.0%} of cells '
              f'(range {min(within[:2]):.0%}-{max(within[:2]):.0%} over the along-axis CI; ignores choosing a different scar)')

    print('\n## Refitting the profile weights on each label set (within 5 px)')
    label_sets = {n1: {k: first[k]['gt'] for k in keys}, n2: {k: second[k]['gt'] for k in both},
                  'mean': {k: mean[k] for k in both if d12[k] <= 10}}

    def relabel(name):
        out = []
        for d in data:
            if d['rec']['key'] in label_sets[name]:
                out.append(dict(d, rec=dict(d['rec'], gt=label_sets[name][d['rec']['key']])))
        return out

    def score(train, test, key):
        tr, te = relabel(train), relabel(test)
        err = []
        for g in sorted({d['rec'][key] for d in te}) if key else [None]:
            w = fit([d for d in tr if key is None or d['rec'][key] != g])
            err += [dist(predict(w, d), d['rec']['gt']) for d in te if key is None or d['rec'][key] == g]
        return tally(err)
    # Not held out by packet: the same physical cells appear in both packets.
    print('train -> test labels'.ljust(22), '| in-sample'.ljust(46), '| held-out video')
    for tr, te in ((n1, n1), (n2, n2), ('mean', 'mean'), (n1, n2), (n2, n1), (n1, 'mean'), (n2, 'mean')):
        print(f'{tr} -> {te}'.ljust(22), '|', score(tr, te, None), '|', score(tr, te, 'video'))
    for name in label_sets:
        c = as_config(fit(relabel(name)))
        print(f'fitted on {name}:'.ljust(18), '  '.join(f"{k[5:]}={v:.3f}" for k, v in c.items() if k != 'SCAR_PRIOR_WEIGHT'))

    print('\n## Confidence (profile scar_confidence; AUC for flagging the cells that are >5 px off)')
    c = np.array([conf[k] for k in both])
    for name, err in ((f'detector vs {n1}', e1), (f'detector vs {n2}', e2),
                      ('detector vs mean', np.array([dist(P[k], mean[k]) for k in both])), (f'{n1} vs {n2}', e12)):
        print(f'{name:22s} AUC {auc(err > 5, -c):.2f}')
    flag = c < make_config().SCAR_CONFIDENCE_FLAG
    for name, err in ((n1, e1), (n2, e2)):
        print(f'vs {name}: {flag.sum()} flagged; within 5 px on the rest {(err[~flag] <= 5).sum()}/{(~flag).sum()} '
              f'({(err[~flag] <= 5).mean():.0%}), on the flagged {(err[flag] <= 5).sum()}/{flag.sum()}')

    print('\n## Repeat annotations of one cell <=5 frames apart (distance between the two points, px)')
    by_cell = defaultdict(list)
    for k in keys:
        by_cell[(k[0], k[2])].append(k)
    for cell_keys in by_cell.values():
        cell_keys.sort(key=lambda k: k[1])
        for ka, kb in zip(cell_keys, cell_keys[1:]):
            if kb[1] - ka[1] <= 5:
                fmt = lambda x, y: 'blank' if np.isinf(dist(x, y)) else f'{dist(x, y):.1f}'
                print(f'{ka[0][-3:]} {ka[2]:4s} f{ka[1]}-f{kb[1]}: {n1} {fmt(first[ka]["gt"], first[kb]["gt"])}  '
                      f'{n2} {fmt(second[ka]["gt"], second[kb]["gt"])}  profile {fmt(P[ka], P[kb])}  '
                      f'(cell centre moved {np.linalg.norm(rec_by[ka]["endpoints"].mean(0) - rec_by[kb]["endpoints"].mean(0)):.1f})')

    print('\n## Area, lengths and new-pole identity')
    def pct(x, y):
        d = (x - y) / ((x + y) / 2) * 100
        return f'bias {d.mean():+.1f}%, |diff| median {np.median(np.abs(d)):.1f}%, r={np.corrcoef(x, y)[0, 1]:.3f}'
    a1 = np.array([first[k]['gt_area'] for k in keys]); a2 = np.array([second[k]['gt_area'] for k in keys])
    am = np.array([rec_by[k]['area'] for k in keys])
    print(f'Area {n2} vs {n1}: {pct(a2, a1)}; Cellpose vs {n1}: {pct(am, a1)}; Cellpose vs {n2}: {pct(am, a2)}')
    e_1 = {k: ends(rec_by[k], first[k]) for k in keys}; e_2 = {k: ends(rec_by[k], second[k]) for k in keys}
    for name, e in ((n1, e_1), (n2, e_2)):
        ok = [k for k in keys if e[k]]
        print(f'{name}: pole A is not the upper pole in ' + (', '.join(f'{k[2]} f{k[1]}' for k in ok if not e[k]['upper_is_a']) or 'none')
              + f'; new pole is the end nearer their scar in {sum(e[k]["new_is_nearer"] for k in ok)}/{len(ok)}')
    lk = [k for k in keys if e_1[k] and e_2[k]]
    same_end = [k for k in lk if e_1[k]['new'] == e_2[k]['new']]
    letters = sum(first[k]['gt_new_pole_guess'] == second[k]['gt_new_pole_guess'] for k in lk)
    print(f'New pole, physical end: same in {len(same_end)}/{len(lk)} (A/B letters match in {letters}/{len(lk)}); different: '
          + (', '.join(f'{k[2]} f{k[1]}' for k in lk if k not in same_end) or 'none'))
    sib = [k for k in lk if rec_by[k]['new_pole_idx'] is not None and rec_by[k]['neighbor_conf'] in ('high', 'medium')]
    print(f"Detector's sibling-contact new pole ({len(sib)} cells with high/medium confidence): matches {n1} in "
          f"{sum(rec_by[k]['new_pole_idx'] == e_1[k]['new'] for k in sib)}, {n2} in {sum(rec_by[k]['new_pole_idx'] == e_2[k]['new'] for k in sib)}")
    dn = np.array([e_2[k]['length'][e_2[k]['new']] - e_1[k]['length'][e_1[k]['new']] for k in same_end])
    dt = np.array([sum(e_2[k]['length'].values()) - sum(e_1[k]['length'].values()) for k in lk])
    print(f'New-end compartment length, {n2} - {n1}: bias {dn.mean():+.2f} px, |diff| median {np.median(np.abs(dn)):.2f}, '
          f'p90 {np.percentile(np.abs(dn), 90):.2f}; total length bias {dt.mean():+.2f} px')
    for name, e in ((n1, e_1), (n2, e_2)):
        det = np.array([np.linalg.norm(P[k] - rec_by[k]['endpoints'][e[k]['new']]) - e[k]['length'][e[k]['new']] for k in lk])
        print(f'New-end length from the profile scar to the Cellpose pole, minus {name}\'s measurement: '
              f'|diff| median {np.median(np.abs(det)):.2f} px')

    if a.out:
        rows = []
        for k in keys:
            r, f1, f2 = rec_by[k], first[k], second[k]
            pr = P[k]
            rows.append(dict(package=r['package'], video=k[0], frame=k[1], cell=k[2],
                             first_row=f1['gt'][0], first_col=f1['gt'][1],
                             second_row=None if f2['gt'] is None else f2['gt'][0],
                             second_col=None if f2['gt'] is None else f2['gt'][1],
                             scorer_distance_px=None if f2['gt'] is None else round(d12[k], 2),
                             profile_row=None if pr is None else round(pr[0], 1),
                             profile_col=None if pr is None else round(pr[1], 1),
                             profile_err_first=round(dist(pr, f1['gt']), 2),
                             profile_err_second=None if f2['gt'] is None else round(dist(pr, f2['gt']), 2),
                             profile_confidence=round(conf[k], 3),
                             same_new_end=None if not (e_1[k] and e_2[k]) else e_1[k]['new'] == e_2[k]['new'],
                             first_area=f1['gt_area'], second_area=f2['gt_area'], cellpose_area=r['area'],
                             first_notes=f1['notes'], second_notes=f2['notes']))
        write_csv(a.out, rows)
        print(f'\nwrote {a.out}')

    if a.gallery:
        cases = []
        for k in keys:
            f1, f2, pr = first[k]['gt'], second[k]['gt'], P[k]
            if f2 is not None and d12[k] <= 10 and min(dist(pr, f1), dist(pr, f2)) <= 5:
                continue
            fmt = lambda x: 'blank' if np.isinf(x) else f'{x:.0f}'
            notes = '; '.join(f'{n}: {s[k]["notes"]}' for n, s in ((n1, first), (n2, second)) if s[k]['notes'])
            title = (f'{k[0][-3:]} {k[2]} f{k[1]}   {n1[0]}-{n2[0]} {fmt(dist(f1, f2))}  det-{n1[0]} {fmt(dist(pr, f1))}  '
                     f'det-{n2[0]} {fmt(dist(pr, f2))} px' + (f'\n{notes}' if notes else ''))
            cases.append((rec_by[k], f1, f2, pr, title))
        gallery(a.gallery, cases, a.names)
        print(f'wrote {a.gallery} ({len(cases)} cells)')


if __name__ == '__main__':
    main()
