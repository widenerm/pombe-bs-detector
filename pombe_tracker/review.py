"""Which cells to check, and what the detector did to them.

review.csv (one row per cell, written by `bsd track` and `bsd review`) lists the frames worth a
look and why. A frame is flagged when:

    unsure        the detector's own review flag: low confidence, or the scar is nearer the end it
                  called old
    jump          whole-cell, new-end or old-end length is more than JUMP_UM from the cell's
                  running median over WINDOW frames (a "crazy number": usually the scar mark
                  jumping, or a bad outline)
    off-axis      new end + old end is over OFF_AXIS x the whole-cell length (scar far from the
                  cell's midline)
    interpolated  no scar was found in this frame; the position was filled in from neighbours
    no scar       no scar at all
    outline       the segmentation was flagged (e.g. a septum fragment)

Frames already corrected, excluded or marked checked (see corrections.py) are not counted as
"to check". Cells used in the growth curves are listed first.

show_cell() draws one cell: its lengths through time with flagged and corrected frames marked,
and the cell at chosen frames with its outline, scar and poles. Coordinates are in the original
movie's pixels, so they match Fiji.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .corrections import corrected_measurements, frames_text, parse_frames, point, to_movie

JUMP_UM = 0.5
WINDOW = 7
OFF_AXIS = 1.10
REASONS = ['unsure', 'jump', 'off-axis', 'interpolated', 'no scar', 'outline']
WHOLE, OLD, NEW = '#2a78d6', '#c8457e', '#008300'
INK, MUTED, FLAG, GRID = '#1a1a1a', '#5a5a5a', '#f2c14e', '#e6e6e6'


def _truthy(s):
    return s.astype(str).str.strip().str.lower().isin({'true', '1', 'yes'})


def frame_flags(d):
    """Boolean columns, one per reason, plus to_check and a text column `reasons`, aligned with d."""
    d = d.sort_values(['cell_name', 'frame'])
    excluded = d['excluded'] if 'excluded' in d else pd.Series(False, index=d.index)
    f = pd.DataFrame(index=d.index)
    f['unsure'] = _truthy(d.get('scar_needs_review', pd.Series('', index=d.index)))
    jump = pd.Series(False, index=d.index)
    for col in ('length_um', 'new_end_length_um', 'old_end_length_um'):
        if col not in d:
            continue
        v = d[col].where(~excluded)
        med = v.groupby(d.cell_name).transform(lambda x: x.rolling(WINDOW, center=True, min_periods=1).median())
        jump |= (v - med).abs() > JUMP_UM
    f['jump'] = jump
    total = d.get('new_end_length_um', np.nan) + d.get('old_end_length_um', np.nan)
    f['off-axis'] = (total > OFF_AXIS * d.get('length_um', np.nan)).fillna(False)
    f['interpolated'] = d.get('scar_source', pd.Series('', index=d.index)).astype(str).eq('interpolated')
    f['no scar'] = ~_truthy(d.get('scar_detected', pd.Series('True', index=d.index)))
    f['outline'] = d.get('seg_quality', pd.Series('ok', index=d.index)).astype(str).ne('ok')
    done = excluded.copy()
    if 'checked' in d:
        done |= d['checked'].astype(bool)
    if 'corrected' in d:
        done |= d['corrected'].fillna('').astype(str).ne('')
    f['to_check'] = f[REASONS].any(axis=1) & ~done
    f['reasons'] = f[REASONS].apply(lambda r: ', '.join(k for k in REASONS if r[k]), axis=1)
    return f.reindex(d.index)


def review_table(run, d=None, frame_minutes=None):
    """One row per cell, cells in the growth curves first, then most frames to check."""
    from .growth import complete_cycles
    if d is None:
        d, _ = corrected_measurements(run, log=lambda *_: None)
    f = frame_flags(d)
    fm = frame_minutes or run.frame_minutes or 2.0
    kept = d[~d['excluded']] if 'excluded' in d else d
    try:
        cycles = set(complete_cycles(kept, fm, {}))
    except Exception:
        cycles = set()
    rows = []
    for name, g in d.groupby('cell_name', sort=False):
        fg = f.loc[g.index]
        check = g[fg.to_check]
        where = check.iloc[0] if len(check) else g.iloc[0]
        npole, opole = point(where.get('new_pole_point')), point(where.get('old_pole_point'))
        x = y = None
        if npole is not None and opole is not None:
            x, y = to_movie(run, *((npole + opole) / 2))
        counts = {k: int((fg[k] & fg.to_check).sum()) for k in REASONS}
        rows.append(dict(
            cell_name=name, in_growth_curves=name in cycles,
            first_frame=int(g.frame.min()), last_frame=int(g.frame.max()), n_frames=len(g),
            frames_to_check=int(fg.to_check.sum()),
            why=', '.join(f'{k} {v}' for k, v in counts.items() if v),
            check_frames=frames_text(check.frame) if len(check) else '',
            x=round(x) if x is not None else None, y=round(y) if y is not None else None,
            at_frame=int(where.frame), fiji_slice=int(where.frame) + 1,
            corrected=int(g['corrected'].fillna('').astype(str).ne('').sum()) if 'corrected' in g else 0,
            excluded=int(g['excluded'].sum()) if 'excluded' in g else 0,
        ))
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    return t.sort_values(['in_growth_curves', 'frames_to_check'], ascending=[False, False]).reset_index(drop=True)


def write_review(run, log=print):
    t = review_table(run)
    out = run.path / 'review.csv'
    t.to_csv(out, index=False)
    if not t.empty:
        n = int((t.frames_to_check > 0).sum())
        g = t[t.in_growth_curves]
        log(f'  review.csv: {n} of {len(t)} cells have frames to check'
            + (f'; of the {len(g)} cells in the growth curves, {int((g.frames_to_check > 0).sum())} do' if len(g) else ''))
    return t


# ── drawing a cell ────────────────────────────────────────────────────────────

def _label_for(labels, row):
    lab = row.get('label')
    if lab is not None and not (isinstance(lab, float) and np.isnan(lab)) and int(lab) > 0 and (labels == int(lab)).any():
        return int(lab)
    npole, opole = point(row.get('new_pole_point')), point(row.get('old_pole_point'))
    if npole is None or opole is None:
        return 0
    r, c = np.round((npole + opole) / 2).astype(int)
    r, c = np.clip(r, 0, labels.shape[0] - 1), np.clip(c, 0, labels.shape[1] - 1)
    if labels[r, c]:
        return int(labels[r, c])
    win = labels[max(r - 10, 0):r + 11, max(c - 10, 0):c + 11]
    vals = win[win > 0]
    return int(np.bincount(vals).argmax()) if vals.size else 0


def cell_view(run, row, frames_src, pad=14):
    """Everything needed to draw one cell in one frame, in original-movie pixel coordinates."""
    frame = int(row['frame'])
    image = np.asarray(frames_src[frame])
    with np.load(run.masks / f'{frame:04d}.npz') as z:
        labels = z['labels']
    lab = _label_for(labels, row)
    pts = {k: point(row.get(k)) for k in ('scar_midpoint', 'new_pole_point', 'old_pole_point', 'scar_end_1', 'scar_end_2')}
    if lab:
        rr, cc = np.nonzero(labels == lab)
    else:
        known = np.array([p for p in pts.values() if p is not None])
        rr, cc = known[:, 0], known[:, 1]
    r0, r1 = max(int(rr.min()) - pad, 0), min(int(rr.max()) + pad + 1, image.shape[0])
    c0, c1 = max(int(cc.min()) - pad, 0), min(int(cc.max()) + pad + 1, image.shape[1])
    crop = image[r0:r1, c0:c1].astype(float)
    outline = []
    if lab:
        from skimage.measure import find_contours
        for cnt in find_contours((labels[r0:r1, c0:c1] == lab).astype(float), 0.5):
            outline.append(np.array([to_movie(run, r + r0, c + c0) for r, c in cnt]))
    x0, y0 = to_movie(run, r0 - 0.5, c0 - 0.5)
    x1, y1 = to_movie(run, r1 - 0.5, c1 - 0.5)
    xy = {k: (to_movie(run, *p) if p is not None else None) for k, p in pts.items()}
    return dict(frame=frame, crop=crop, extent=(x0, x1, y1, y0), outline=outline, points=xy, label=lab)


def draw_cell_view(ax, v, row, flags_text='', title=True):
    lo, hi = np.percentile(v['crop'], [1, 99])
    ax.imshow(v['crop'], cmap='gray', vmin=lo, vmax=hi, extent=v['extent'], interpolation='nearest')
    for o in v['outline']:
        ax.plot(o[:, 0], o[:, 1], color='#7fd3ff', lw=0.8)
    p = v['points']
    manual = str(row.get('scar_source')) == 'manual'
    if p['scar_end_1'] is not None and p['scar_end_2'] is not None and not manual:
        ax.plot([p['scar_end_1'][0], p['scar_end_2'][0]], [p['scar_end_1'][1], p['scar_end_2'][1]],
                color='#ffd400', lw=1.6)
    for key, kw in (('new_pole_point', dict(marker='o', color=NEW, ms=7)),
                    ('old_pole_point', dict(marker='D', color=OLD, ms=6)),
                    ('scar_midpoint', dict(marker='*' if manual else 'o', color='white', ms=10 if manual else 6))):
        if p[key] is not None:
            ax.plot(*p[key], linestyle='none', markeredgecolor=INK, markeredgewidth=0.8, **kw)
    ax.set_xlim(v['extent'][0], v['extent'][1])
    ax.set_ylim(v['extent'][2], v['extent'][3])
    ax.tick_params(labelsize=6, length=2, colors=MUTED)
    if title:
        ne, oe = row.get('new_end_length_um'), row.get('old_end_length_um')
        lens = f'new {ne:.2f}, old {oe:.2f} µm' if pd.notna(ne) and pd.notna(oe) else 'no scar'
        ax.set_title(f'frame {v["frame"]} (Fiji {v["frame"] + 1})\n{lens}' + (f'\n{flags_text}' if flags_text else ''),
                     fontsize=7, color=INK, loc='left')


def _pick_frames(g, f, frames, max_panels):
    have = list(g.frame.astype(int))
    if frames is not None:
        chosen = [x for x in parse_frames(frames) if x in set(have)] if isinstance(frames, str) else [x for x in frames if x in set(have)]
    else:
        flagged = list(g.frame[f.loc[g.index, 'to_check']].astype(int))
        pool = flagged or have
        chosen = pool if len(pool) <= max_panels else [pool[i] for i in np.linspace(0, len(pool) - 1, max_panels).round().astype(int)]
    return sorted(dict.fromkeys(chosen))[:max(max_panels, 1) if frames is None else 24]


def show_cell(run, cell, frames=None, max_panels=8):
    """Figure: the cell's lengths through time, then the cell at chosen frames (default: the frames to
    check, or evenly spaced if none). `frames` is e.g. '40-60' or [40, 52]; at most 24 panels."""
    import matplotlib
    import matplotlib.pyplot as plt
    d, _ = corrected_measurements(run, log=lambda *_: None)
    g = d[d.cell_name.astype(str) == str(cell)].sort_values('frame')
    if g.empty:
        near = sorted(d.cell_name.astype(str).unique())
        raise ValueError(f'no cell {cell!r} in {run.path.name}; cells: {", ".join(near[:40])}{" ..." if len(near) > 40 else ""}')
    f = frame_flags(d)
    shown = _pick_frames(g, f, frames, max_panels)
    ncol = min(4, max(len(shown), 1))
    nrow = int(np.ceil(len(shown) / ncol)) if shown else 0
    fig = plt.figure(figsize=(2.3 * ncol + 0.6, 2.6 + 2.5 * nrow))
    gs = fig.add_gridspec(1 + nrow, ncol, height_ratios=[1.1] + [1.25] * nrow, hspace=0.75, wspace=0.35)

    ax = fig.add_subplot(gs[0, :])
    excl = g['excluded'] if 'excluded' in g else pd.Series(False, index=g.index)
    for fr in g.frame[f.loc[g.index, 'to_check']]:
        ax.axvspan(fr - 0.5, fr + 0.5, color=FLAG, alpha=0.35, lw=0)
    for col, color, name in (('length_um', WHOLE, 'whole cell'), ('old_end_length_um', OLD, 'old end'),
                             ('new_end_length_um', NEW, 'new end')):
        ax.plot(g.frame, g[col].where(~excl), color=color, lw=1.4, label=name)
    corr = g[g['corrected'].fillna('').astype(str).ne('')] if 'corrected' in g else g.iloc[:0]
    if len(corr):
        ax.plot(corr.frame, corr.new_end_length_um, linestyle='none', marker='o', mfc='none', mec=INK, ms=5,
                label='corrected')
    for fr in shown:
        ax.axvline(fr, color=MUTED, lw=0.6, ls=':')
    ax.set_xlabel('frame', fontsize=8)
    ax.set_ylabel('µm', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(axis='y', color=GRID, lw=0.6)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    n_check = int(f.loc[g.index, 'to_check'].sum())
    ax.set_title(f'{cell}: frames {int(g.frame.min())}–{int(g.frame.max())}; {n_check} to check (shaded)',
                 fontsize=9, loc='left', color=INK)
    ax.legend(fontsize=7, frameon=False, ncol=4, loc='upper left', bbox_to_anchor=(0, -0.28))

    with run.frames() as src:
        for k, fr in enumerate(shown):
            row = g[g.frame == fr].iloc[0]
            axk = fig.add_subplot(gs[1 + k // ncol, k % ncol])
            v = cell_view(run, row, src)
            why = f.loc[row.name, 'reasons'] if f.loc[row.name, 'to_check'] else ''
            if 'corrected' in g and str(row.get('corrected') or ''):
                why = ('excluded' if row.get('excluded') else 'corrected')
            draw_cell_view(axk, v, row, why)
    fig.text(0.01, 0.005, 'yellow: birth scar   white: scar midpoint (star: placed by hand)   green: new pole   '
             'magenta: old pole   axes: movie pixels (as in Fiji)', fontsize=6.5, color=MUTED)
    return fig
