"""Growth curves of complete cell cycles from a pipeline measurements CSV.

A complete cycle is a cell born in the movie (its parent is tracked) that also
divides in it (a daughter is tracked), lasts at least MIN_CYCLE_MIN and never
collapses mid-cycle (a missed division). Frames whose length is more than JUMP_UM
from the cell's 7-frame running median are dropped as segmentation errors
(typically a partial mask while the septum forms); the data CSV keeps them,
marked kept=False. Compartment lengths jitter when the scar position jumps
between frames, so panel B draws a running median over the raw trace. NETO is
not estimated here: on these cells an automatic hinge fit is not yet reliable.

Change per frame is length at frame n minus length at frame n-1, computed only
when both frames were kept. It is about as noisy as it is large (whole cell:
~0.07 um per 2-min frame, frame-to-frame SD 0.05-0.1 um; compartments far noisier
because the scar mark shifts), so growth_deltas draws each raw change faintly and
a DELTA_SMOOTH-frame running median in bold.

Growth since birth is length at frame n minus length at birth, the quantity the
lab's hand-tracked growth curves plot ("average delta length"). Birth length is the
median of the first 3 kept frames; each end uses its 7-frame running median, so
its baseline is the median of its first 4 frames. growth_since_birth puts the whole
cell and both ends on one set of axes (mean +/- SE across cells, plus the
featured cells) so slope changes line up; a hand-tracked reference adds its means
there and writes growth_since_birth_vs_hand (one panel per end). Reference tables
have columns TYPE, CELLNUM, END (TOT/OE/NE), LENGTH, TIME; hand_tracking.py makes
one from the lab's growth workbook.

    bsd growth RUN_DIR [--hand WORKBOOK.xlsx --hand-type y_wt]        (normal use)
    python -m pombe_tracker.growth MEASUREMENTS.csv --out DIR [--cells A0 B0 C0] [--featured A0 C0]
                                   [--frame-minutes 2] [--reference REF.csv --reference-type y_wt]

Writes growth_curves.pdf/.png (A: length of every cycle; B: whole cell, old end
and new end for the --featured cells), growth_deltas.pdf/.png (the same layout
for change per frame), growth_since_birth.pdf/.png, growth_curves_data.csv (every
plotted point) and growth_curves_summary.csv.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

JUMP_UM = 0.5
SMOOTH_FRAMES = 7
DELTA_SMOOTH = 15     # frames (30 min at 2 min/frame)
GROWN_COLS = {'length_um': ('grown_um', 'TOT', 'Whole cell'), 'old_end_length_um': ('grown_old_um', 'OE', 'Old end'),
              'new_end_length_um': ('grown_new_um', 'NE', 'New end')}
REF_INK = '#3d3d3d'
DELTA_COLS = {'length_um': 'd_length_um', 'old_end_length_um': 'd_old_end_um', 'new_end_length_um': 'd_new_end_um'}
WHOLE, OLD, NEW = '#2a78d6', '#c8457e', '#008300'
INK, MUTED, GRID = '#1a1a1a', '#5a5a5a', '#e6e6e6'


def real_division(d, parent, ratio=0.75):
    """True if the parent's daughters are each clearly shorter than the parent was at the end.

    A false division (a mask fragment split off for a frame or two) leaves one
    "daughter" as long as the mother; real daughters are born at about half.
    """
    end = d[d.cell_name == parent].sort_values('frame').length_um.tail(3).median()
    kids = [k for k in (parent + '0', parent + '1') if (d.cell_name == k).any()]
    return bool(kids) and all(
        d[d.cell_name == k].sort_values('frame').length_um.head(3).median() < ratio * end for k in kids)


MIN_CYCLE_MIN = 60     # S. pombe cycles take 2-4 h; a "cycle" under an hour is a mask merge or split
COLLAPSE = 0.75        # a running-median drop below this fraction mid-cycle is a division the tracker missed


def missed_division(c, ratio=COLLAPSE, w=5):
    """True if the cell's length falls to under `ratio` of itself within the trace (5-frame medians either side)."""
    L = c.sort_values('frame').length_um.reset_index(drop=True)
    before = L.rolling(w).median()
    after = L[::-1].rolling(w).median()[::-1].shift(-1)
    return bool(((after / before) < ratio).any())


def complete_cycles(d, frame_minutes=2.0, rejected=None):
    """Cells born by a real division in the movie that also divide in it.

    Root cells (letters only, e.g. 'AF') were present at the first frame or
    appeared later; lineage names add a digit per division (tracking.py).
    A cycle is also rejected if it lasts under MIN_CYCLE_MIN (a merged or split
    mask that "divides" again within minutes) or its length collapses part-way
    (two cycles joined by a missed division). Reasons go into `rejected` if given.
    """
    names = set(d.cell_name)
    born = [n for n in names if n[-1] in '01' and n[:-1] in names]
    out = []
    for n in sorted(born):
        if not (real_division(d, n[:-1]) and real_division(d, n)):
            continue
        c = d[d.cell_name == n]
        minutes = (c.frame.max() - c.frame.min() + 1) * frame_minutes
        why = (f'lasts {minutes:.0f} min' if minutes < MIN_CYCLE_MIN else
               'length collapses mid-cycle (missed division)' if missed_division(c) else None)
        if why and rejected is not None:
            rejected[n] = why
        if not why:
            out.append(n)
    return out


def cycle_table(d, cells, frame_minutes):
    rows = []
    for n in cells:
        c = d[d.cell_name == n].sort_values('frame').copy()
        c['time_min'] = (c.frame - c.frame.min()) * frame_minutes
        med = c.length_um.rolling(7, center=True, min_periods=4).median()
        c['kept'] = (c.length_um - med).abs() <= JUMP_UM
        both = c.kept & c.kept.shift(fill_value=False) & (c.frame.diff() == 1)
        for col, dcol in DELTA_COLS.items():
            c[dcol] = c[col].diff().where(both)   # length at frame n minus length at frame n-1
        k = c.kept
        c['grown_um'] = (c.length_um - c.length_um[k].head(3).median()).where(k)   # length at n minus length at birth
        for col in ('old_end_length_um', 'new_end_length_um'):
            sm = c[col].where(k).rolling(SMOOTH_FRAMES, center=True, min_periods=1).median()
            c[GROWN_COLS[col][0]] = (sm - sm[k].iloc[0]).where(k)
        rows.append(c[['cell_name', 'frame', 'time_min', 'length_um', 'new_end_length_um', 'old_end_length_um',
                       *DELTA_COLS.values(), *(v[0] for v in GROWN_COLS.values()), 'scar_needs_review', 'seg_quality', 'kept']])
    return pd.concat(rows, ignore_index=True)


def smooth_delta(c, dcol):
    """Running median of the change per frame over DELTA_SMOOTH frames, skipping missing changes."""
    return c[dcol].rolling(DELTA_SMOOTH, center=True, min_periods=DELTA_SMOOTH // 2).median()


def summarize(t, frame_minutes):
    out = []
    for n, c in t[t.kept].groupby('cell_name', sort=False):
        cycle = (c.time_min.max() + frame_minutes)
        early = c[c.time_min <= 0.75 * cycle]
        rate = np.polyfit(early.time_min, early.length_um, 1)[0] * 60
        dl = t[t.cell_name == n].d_length_um.dropna()
        out.append(dict(cell=n, birth_length_um=c.length_um.head(3).median(), division_length_um=c.length_um.tail(5).median(),
                        cycle_min=cycle, elongation_um_per_h=rate,
                        mean_change_um_per_frame=dl.mean(), sd_change_um_per_frame=dl.std(),
                        frames_dropped=int((~t[t.cell_name == n].kept).sum())))
    return pd.DataFrame(out)


def _span(lo, hi, values, step=2):
    """Axis limits: the default [lo, hi], widened to the next step if the data go past it."""
    v = np.asarray(pd.Series(values).dropna(), float)
    if v.size:
        lo, hi = min(lo, step * np.floor(v.min() / step)), max(hi, step * np.ceil(v.max() / step))
    return lo, hi


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
                         'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                         'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    return plt


def figure(t, featured, path):
    """A: length of every cycle, one color. B: whole cell, old end and new end for the featured cells."""
    plt = _plt()
    cells = list(dict.fromkeys(t.cell_name))
    k = len(featured)
    tmax = t.time_min.max()
    ticks = np.arange(0, tmax + 1, 60)
    fig = plt.figure(figsize=(7.2, 6.4))
    gs = fig.add_gridspec(2, k, height_ratios=[1.1, 1], hspace=0.75, wspace=0.12)

    def style(ax):
        ax.grid(axis='y', color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlim(0, tmax + 4)
        ax.set_xticks(ticks)

    ax = fig.add_subplot(gs[0, :])
    for n in cells:
        c = t[(t.cell_name == n) & t.kept]
        ax.plot(c.time_min, c.length_um, color=WHOLE, lw=1.3, alpha=0.85, solid_capstyle='round')
    style(ax)
    ax.set_ylim(*_span(6, 16, t[t.kept].length_um))
    ax.set_xlabel('Time since birth (min)')
    ax.set_ylabel('Cell length (µm)')
    ax.set_title(f'A   Cell length through one cycle (all {len(cells)} complete cycles)', loc='left',
                 fontsize=10, fontweight='bold', color=INK, pad=10)

    axes = [fig.add_subplot(gs[1, j]) for j in range(k)]
    for j, (ax2, n) in enumerate(zip(axes, featured)):
        c = t[(t.cell_name == n) & t.kept]
        ax2.plot(c.time_min, c.length_um, color=WHOLE, lw=1.5)
        for col, color in (('old_end_length_um', OLD), ('new_end_length_um', NEW)):
            ax2.plot(c.time_min, c[col], color=color, lw=0.6, alpha=0.3)
            ax2.plot(c.time_min, c[col].rolling(SMOOTH_FRAMES, center=True, min_periods=1).median(), color=color, lw=1.5)
        style(ax2)
        lo2, hi2 = _span(0, 16, t[t.kept].length_um, 4)
        ax2.set_ylim(lo2, hi2)
        if hi2 > 16:
            ax2.set_yticks(np.arange(0, hi2 + 0.1, 5))
        ax2.tick_params(labelsize=7.5)
        ax2.set_title(n, fontsize=8.5, color=INK)
        if j:
            ax2.tick_params(labelleft=False)
        else:
            ax2.set_ylabel('Length (µm)')
        if j == 0 and k <= 3:                      # room for end labels; otherwise a legend below
            last = c.iloc[-1]
            ax2.text(last.time_min + 4, last.length_um, 'whole\ncell', fontsize=7.5, color=INK, va='center', linespacing=0.9)
            for col, label in (('old_end_length_um', 'old\nend'), ('new_end_length_um', 'new\nend')):
                y = c[col].tail(SMOOTH_FRAMES).median()
                ax2.text(last.time_min + 4, y, label, fontsize=7.5, color=INK, va='center', linespacing=0.9)
    box = axes[0].get_position()
    fig.text(box.x0 - 0.06, box.y1 + 0.075, 'B   Whole cell and the compartments either side of the birth scar',
             fontsize=10, fontweight='bold', color=INK)
    if k > 3:
        h = [plt.Line2D([], [], color=cl, lw=1.5) for cl in (WHOLE, OLD, NEW)]
        fig.legend(h, ['whole cell', 'old end', 'new end'], loc='lower right', ncol=3, frameon=False, fontsize=8,
                   bbox_to_anchor=(axes[-1].get_position().x1, box.y1 + 0.03), handlelength=1.6, columnspacing=1.2)
    fig.text(0.5, box.y0 - 0.085, 'Time since birth (min)', ha='center', fontsize=9, color=INK)
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def delta_figure(t, featured, path, frame_minutes):
    """Change in length per frame. A: whole cell, every cycle. B: whole cell, old end and new end for the featured cells."""
    plt = _plt()
    cells = list(dict.fromkeys(t.cell_name))
    k = len(featured)
    tmax = t.time_min.max()
    ticks = np.arange(0, tmax + 1, 60)
    fig = plt.figure(figsize=(7.2, 6.4))
    gs = fig.add_gridspec(2, k, height_ratios=[1.1, 1], hspace=0.75, wspace=0.12)
    unit = f'µm per {frame_minutes:g}-min frame'

    def style(ax, lo, hi):
        ax.grid(axis='y', color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
        ax.set_xlim(0, tmax + 4)
        ax.set_xticks(ticks)
        ax.set_ylim(lo, hi)

    ax = fig.add_subplot(gs[0, :])
    for n in cells:
        c = t[t.cell_name == n]
        ax.plot(c.time_min, c.d_length_um, 'o', ms=1.6, color=WHOLE, alpha=0.18, mec='none')
    for n in cells:
        c = t[t.cell_name == n]
        ax.plot(c.time_min, smooth_delta(c, 'd_length_um'), color=WHOLE, lw=1.4, alpha=0.9, solid_capstyle='round')
    lo, hi = -0.3, 0.4
    style(ax, lo, hi)
    ax.set_xlabel('Time since birth (min)')
    ax.set_ylabel(f'Change in length\n({unit})')
    ax.set_title(f'A   Whole-cell growth per frame (all {len(cells)} complete cycles)', loc='left',
                 fontsize=10, fontweight='bold', color=INK, pad=10)
    hidden = int(((t.d_length_um < lo) | (t.d_length_um > hi)).sum())
    if hidden:
        ax.text(1, 1.01, f'{hidden} raw changes beyond the axis', transform=ax.transAxes, ha='right', va='bottom',
                fontsize=7, color=MUTED)

    series = (('d_old_end_um', OLD, 'old end'), ('d_new_end_um', NEW, 'new end'), ('d_length_um', WHOLE, 'whole cell'))
    axes = [fig.add_subplot(gs[1, j]) for j in range(k)]
    for j, (ax2, n) in enumerate(zip(axes, featured)):
        c = t[t.cell_name == n]
        for dcol, color, _ in series:
            ax2.plot(c.time_min, c[dcol], 'o', ms=1.4, color=color, alpha=0.3, mec='none')
        for dcol, color, label in series:
            ax2.plot(c.time_min, smooth_delta(c, dcol), color=color, lw=1.5, label=label)
        style(ax2, lo, hi)
        ax2.tick_params(labelsize=7.5)
        out = int(sum(((c[x] < lo) | (c[x] > hi)).sum() for x, _, _ in series))
        ax2.set_title(n, fontsize=8.5, color=INK)
        if out:
            ax2.text(0.97, 0.03, f'{out} off axis', transform=ax2.transAxes, ha='right', va='bottom', fontsize=6.5, color=MUTED)
        if j:
            ax2.tick_params(labelleft=False)
        else:
            ax2.set_ylabel(f'Change in length\n({unit})')
    h, l = axes[0].get_legend_handles_labels()
    order = [2, 0, 1]
    box = axes[0].get_position()
    fig.text(box.x0 - 0.06, box.y1 + 0.075, 'B   Growth per frame of the whole cell and each side of the birth scar',
             fontsize=10, fontweight='bold', color=INK)
    fig.legend([h[i] for i in order], [l[i] for i in order], loc='lower right', ncol=3, frameon=False, fontsize=8,
               bbox_to_anchor=(axes[-1].get_position().x1, box.y1 + 0.03), handlelength=1.6, columnspacing=1.2)
    fig.text(0.5, box.y0 - 0.085, 'Time since birth (min)', ha='center', fontsize=9, color=INK)
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def mean_se(df, x, y, min_n=3):
    """Mean and standard error across cells at each time, where at least min_n cells contribute."""
    m = df.dropna(subset=[y]).groupby(x)[y].agg(['mean', 'std', 'count'])
    m = m[m['count'] >= min_n]
    return m.index.to_numpy(float), m['mean'].to_numpy(), (m['std'] / np.sqrt(m['count'])).to_numpy()


def read_reference(path, kind):
    r = pd.read_csv(path).dropna(how='all')
    r = r[r.TYPE == kind]
    if r.empty:
        raise ValueError(f'no rows with TYPE == {kind!r} in {path}')
    return r


def since_birth_figure(t, featured, path, ref=None, ref_label=None):
    """Growth since birth with the whole cell, old end and new end on one set of axes, so slope changes line up.

    A: mean +/- SE across cells (solid), with the hand-tracked means dashed if given. B: the featured cells.
    """
    plt = _plt()
    ends = [(gcol, end, title.lower(), {'TOT': WHOLE, 'OE': OLD, 'NE': NEW}[end]) for gcol, end, title in GROWN_COLS.values()]
    cells = list(dict.fromkeys(t.cell_name))
    k = len(featured)
    tmax = max(t.time_min.max(), ref.TIME.max() if ref is not None else 0)
    ticks = np.arange(0, tmax + 1, 40)
    fig = plt.figure(figsize=(7.2, 6.4))
    gs = fig.add_gridspec(2, k, height_ratios=[1.25, 1], hspace=0.7, wspace=0.12)

    top = _span(0, 8, pd.concat([t[g] for g, *_ in ends] + ([ref.LENGTH] if ref is not None else [])))[1]

    def style(ax):
        ax.grid(axis='y', color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlim(0, tmax + 4)
        ax.set_xticks(ticks)
        ax.set_ylim(0, top)
        ax.set_yticks(np.arange(0, top + 0.1, 2))

    ax = fig.add_subplot(gs[0, :])
    for gcol, end, label, color in ends:
        x, m, se = mean_se(t, 'time_min', gcol)
        ax.fill_between(x, m - se, m + se, color=color, alpha=0.18, lw=0)
        ax.plot(x, m, color=color, lw=2)
        ax.text(x[-1] + 3, m[-1], label, fontsize=8, color=INK, va='center')
        if ref is not None:
            r = ref[ref.END == end]
            rx, rm, _ = mean_se(r, 'TIME', 'LENGTH')
            ax.plot(rx, rm, color=color, lw=1.3, ls=(0, (3, 2)))
    style(ax)
    ax.set_xlabel('Minutes after cell separation')
    ax.set_ylabel('Δ length since separation (µm)')
    ax.set_title(f'A   Growth since birth, mean of {len(cells)} complete cycles', loc='left',
                 fontsize=10, fontweight='bold', color=INK, pad=22)
    if ref is not None:
        n_ref = ref.CELLNUM.nunique()
        h = [plt.Line2D([], [], color=MUTED, lw=2), plt.Line2D([], [], color=MUTED, lw=1.3, ls=(0, (3, 2)))]
        ax.legend(h, [f'Automated pipeline, mean ± SE (n = {len(cells)})', f'{ref_label}, mean (n = {n_ref})'],
                  loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=7.5, borderaxespad=0.2)

    axes = [fig.add_subplot(gs[1, j]) for j in range(k)]
    for j, (ax2, n) in enumerate(zip(axes, featured)):
        c = t[t.cell_name == n]
        for gcol, end, label, color in ends:
            kc = c[c[gcol].notna()]                      # join across dropped frames
            ax2.plot(kc.time_min, kc[gcol], color=color, lw=1.5)
        style(ax2)
        if k > 3:
            ax2.set_xticks(np.arange(0, tmax + 1, 80))
        ax2.tick_params(labelsize=7.5)
        ax2.set_title(n, fontsize=8.5, color=INK)
        if j:
            ax2.tick_params(labelleft=False)
        else:
            ax2.set_ylabel('Δ length since\nseparation (µm)')
    box = axes[0].get_position()
    fig.text(box.x0 - 0.06, box.y1 + 0.06, 'B   Single cells', fontsize=10, fontweight='bold', color=INK)
    fig.text(0.5, box.y0 - 0.085, 'Minutes after cell separation', ha='center', fontsize=9, color=INK)
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def versus_hand_figure(t, path, ref, ref_label):
    """Growth since birth, one panel per end: pipeline cells faint, pipeline mean +/- SE bold, hand-tracked mean +/- SE dashed."""
    plt = _plt()
    cells = list(dict.fromkeys(t.cell_name))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0), sharey=True, gridspec_kw=dict(wspace=0.12))
    tmax = max(t.time_min.max(), ref.TIME.max() if ref is not None else 0)
    for ax, (col, (gcol, end, title)) in zip(axes, GROWN_COLS.items()):
        color = {'TOT': WHOLE, 'OE': OLD, 'NE': NEW}[end]
        for n in cells:
            c = t[(t.cell_name == n) & t[gcol].notna()]
            ax.plot(c.time_min, c[gcol], color=color, lw=0.6, alpha=0.3)
        x, m, se = mean_se(t, 'time_min', gcol)
        ax.fill_between(x, m - se, m + se, color=color, alpha=0.18, lw=0)
        ax.plot(x, m, color=color, lw=2, label=f'Automated pipeline (n = {len(cells)} cells)')
        if ref is not None:
            r = ref[ref.END == end]
            rx, rm, rse = mean_se(r, 'TIME', 'LENGTH')
            ax.fill_between(rx, rm - rse, rm + rse, color=REF_INK, alpha=0.12, lw=0)
            ax.plot(rx, rm, color=REF_INK, lw=1.6, ls=(0, (4, 2)), label=f'{ref_label} (n = {r.CELLNUM.nunique()} cells)')
        ax.grid(axis='y', color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlim(0, tmax + 4)
        ax.set_xticks(np.arange(0, tmax + 1, 40))
        top = _span(0, 9, pd.concat([t[c] for c, *_ in GROWN_COLS.values()] + [ref.LENGTH]))[1]
        ax.set_ylim(0, top)
        ax.set_yticks(np.arange(0, top + 0.1, 2))
        ax.tick_params(labelsize=7.5)
        ax.set_title(title, fontsize=9, color=INK)
    axes[0].set_ylabel('Δ length since cell separation (µm)')
    fig.supxlabel('Minutes after cell separation', fontsize=9, color=INK, y=-0.04)
    h, l = axes[0].get_legend_handles_labels()
    if ref is not None:
        h = [plt.Line2D([], [], color=MUTED, lw=2), plt.Line2D([], [], color=REF_INK, lw=1.6, ls=(0, (4, 2)))]
    fig.legend(h, l, loc='lower left', bbox_to_anchor=(0.1, 0.97), ncol=2, frameon=False, fontsize=7.5)
    fig.text(0.1, 1.07, 'Lines: mean ± SE across cells where ≥3 cells contribute; faint lines: single pipeline cells.',
             fontsize=7, color=MUTED, ha='left')
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def make_growth(measurements, out, cells=None, featured=None, frame_minutes=2.0, reference=None,
                reference_label='Hand-tracked', verbose=True, fig_dir=None):
    """Find complete cycles in a measurements table; write the tables to *out* and, if fig_dir is
    given, every growth figure (PNG and PDF) to fig_dir.

    measurements : DataFrame or path to a pipeline measurements CSV
    reference    : optional DataFrame of hand-tracked curves (TYPE, CELLNUM, END, LENGTH, TIME), one TYPE
    Returns (cycle names, summary DataFrame, {rejected cycle: reason}).
    """
    d = pd.read_csv(measurements) if not isinstance(measurements, pd.DataFrame) else measurements
    rejected = {}
    found = complete_cycles(d, frame_minutes, rejected)
    cells = cells or found
    missing = set(cells) - set(found)
    if missing:
        raise ValueError(f'not complete cycles: {sorted(missing)}; complete cycles are {found}')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if not cells:
        if verbose:
            print('no complete cell cycles in this movie (a cell must be born and divide while tracked); '
                  'no growth curves')
            for n, why in rejected.items():
                print(f'  rejected {n}: {why}')
        return [], pd.DataFrame(), rejected
    t = cycle_table(d, cells, frame_minutes)
    s = summarize(t, frame_minutes)
    t.to_csv(out / 'growth_curves_data.csv', index=False)
    s.round(3).to_csv(out / 'growth_curves_summary.csv', index=False)
    featured = featured or cells
    if set(featured) - set(cells):
        raise ValueError(f'featured cells must be among the plotted cells {cells}')
    if fig_dir is not None:
        fig_dir = Path(fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)
        figure(t, featured, fig_dir / 'growth_curves')
        delta_figure(t, featured, fig_dir / 'growth_deltas', frame_minutes)
        since_birth_figure(t, featured, fig_dir / 'growth_since_birth', reference, reference_label)
        if reference is not None:
            versus_hand_figure(t, fig_dir / 'growth_since_birth_vs_hand', reference, reference_label)
    if verbose:
        print(f'{len(found)} complete cycles: {found}')
        for n, why in rejected.items():
            print(f'  rejected {n}: {why}')
        print(s.round(2).to_string(index=False))
    return cells, s, rejected


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('measurements')
    p.add_argument('--out', required=True)
    p.add_argument('--cells', nargs='+', help='cells for panel A (default: every complete cycle)')
    p.add_argument('--featured', nargs='+', help='cells for panel B, chosen from --cells (default: all of them)')
    p.add_argument('--frame-minutes', type=float, default=2.0)
    p.add_argument('--reference', help='hand-tracked growth curves (TYPE, CELLNUM, END, LENGTH, TIME)')
    p.add_argument('--reference-type', default='y_wt', help='TYPE to plot from --reference (default y_wt: wild type in YE)')
    p.add_argument('--reference-label', default='Hand-tracked')
    a = p.parse_args()
    ref = read_reference(a.reference, a.reference_type) if a.reference else None
    make_growth(a.measurements, a.out, a.cells, a.featured, a.frame_minutes, ref, a.reference_label, fig_dir=a.out)


if __name__ == '__main__':
    main()
