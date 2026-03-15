"""
visualization.py  –  All plotting functions.

Each function returns the matplotlib Figure so the caller can save or display
it.  None of the functions call plt.show() themselves.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


# ══════════════════════════════════════════════════════════════════════════════
# 1. Frame overview
# ══════════════════════════════════════════════════════════════════════════════

def plot_frame_overview(frame, results, frame_idx, config=None):
    """
    One panel showing all cells in the frame.
    Green outline = scar detected; red dashed = not detected.
    Cyan dotted lines = pole-to-pole neighbour connections.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(frame, cmap='gray', vmin=np.percentile(frame, 1), vmax=np.percentile(frame, 99))
    ax.set_title(f'Frame {frame_idx}  –  {len(results)} cells', fontsize=13, fontweight='bold')
    ax.axis('off')

    for r in results:
        contour = r['contour']
        name    = r.get('cell_name', str(r['label']))
        cx, cy  = contour[:, 1].mean(), contour[:, 0].mean()

        if r['scar_detected']:
            ax.plot(contour[:, 1], contour[:, 0], color='lime', lw=1.5, alpha=0.9)
            ax.text(cx, cy, name, color='white', fontsize=7, fontweight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.2', fc='green', alpha=0.7, lw=0))
        else:
            ax.plot(contour[:, 1], contour[:, 0], color='tomato',
                    lw=1.2, alpha=0.7, ls='--')
            ax.text(cx, cy, name, color='white', fontsize=7, fontweight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.2', fc='firebrick', alpha=0.7, lw=0))

        # Orange warning for bad segmentations (drawn on top)
        if r.get('seg_quality', 'ok') != 'ok':
            ax.plot(contour[:, 1], contour[:, 0], color='orange',
                    lw=2.5, alpha=0.95, zorder=4)
            ax.text(cx, cy - 10, f"⚠ {r['seg_quality'].replace('_', ' ')}",
                    color='orange', fontsize=6, ha='center', va='bottom',
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.1', fc='black', alpha=0.6, lw=0))

        for nb in r.get('neighbors', []):
            op, tp = nb['our_pole'], nb['their_pole']
            ax.plot([op[1], tp[1]], [op[0], tp[0]], color='cyan',
                    lw=0.8, ls=':', alpha=0.6)

    legend = [
        mpatches.Patch(color='lime',    label='Scar detected'),
        mpatches.Patch(color='tomato',  label='Not detected'),
        mpatches.Patch(color='orange',  label='⚠ Bad segmentation'),
        Line2D([0], [0], color='cyan',  ls=':', lw=1, label='Pole neighbour'),
    ]
    ax.legend(handles=legend, loc='upper right', fontsize=8, framealpha=0.8)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 2. Individual cell panels
# ══════════════════════════════════════════════════════════════════════════════

def plot_individual_cells(frame, results, frame_idx, config=None):
    """
    One subplot per cell showing the birth scar, poles, and measurement lines.
    """
    n     = len(results)
    ncols = min(n, 4)
    nrows = max(1, (n + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for idx, (ax, r) in enumerate(zip(axes_flat, results)):
        contour  = r['contour']
        dbg      = r['debug_info']
        name     = r.get('cell_name', str(r['label']))

        # Crop region
        pad   = 20
        r0    = max(0, int(contour[:, 0].min()) - pad)
        r1    = min(frame.shape[0], int(contour[:, 0].max()) + pad)
        c0    = max(0, int(contour[:, 1].min()) - pad)
        c1    = min(frame.shape[1], int(contour[:, 1].max()) + pad)
        crop  = frame[r0:r1, c0:c1]
        ax.imshow(crop, cmap='gray',
                  vmin=np.percentile(crop, 1), vmax=np.percentile(crop, 99))

        # Helper: shift to local coords
        def loc(pt):
            return np.array(pt) - np.array([r0, c0])

        local_c = contour.copy()
        local_c[:, 0] -= r0;  local_c[:, 1] -= c0
        ax.plot(local_c[:, 1], local_c[:, 0], 'cyan', lw=1.5, alpha=0.8)

        if r['scar_detected']:
            sp   = r['scar_points']
            mp   = loc(r['scar_midpoint'])
            np_  = loc(dbg['new_pole_point'])
            op_  = loc(dbg['old_pole_point'])
            s1   = loc(sp[0]);  s2 = loc(sp[1])

            # Scar line
            ax.plot([s1[1], s2[1]], [s1[0], s2[0]], 'yellow', lw=2.5, zorder=5,
                    label='Birth scar')
            ax.plot(s1[1], s1[0], 'o', color='yellow', ms=6,
                    mec='black', mew=1.5, zorder=6)
            ax.plot(s2[1], s2[0], 'o', color='yellow', ms=6,
                    mec='black', mew=1.5, zorder=6)
            ax.plot(mp[1], mp[0], 'o', color='white', ms=9,
                    mec='black', mew=1.5, zorder=7, label='Scar midpoint')

            # Poles
            ax.plot(np_[1], np_[0], 'X', color='lime', ms=11,
                    mec='black', mew=1.5, zorder=7, label='New pole')
            ax.plot(op_[1], op_[0], 'D', color='magenta', ms=8,
                    mec='black', mew=1.5, zorder=7, label='Old pole')

            # Measurement lines
            ni, oi = r['new_end_length'], r['old_end_length']
            ax.plot([mp[1], np_[1]], [mp[0], np_[0]], color='lime',
                    lw=1.8, ls='--', label=f'New {ni:.0f}px')
            ax.plot([mp[1], op_[1]], [mp[0], op_[0]], color='magenta',
                    lw=1.8, ls='--', label=f'Old {oi:.0f}px')

            pm  = dbg.get('pole_method', '?')
            pc  = dbg.get('pole_confidence', '?')
            mt  = dbg.get('match_type', '?')
            title = (f'{name}  ✓  [{mt}]\n'
                     f'N={ni:.0f}  O={oi:.0f}  R={ni/max(oi,1e-3):.2f}\n'
                     f'{pm} ({pc})')
            ax.set_title(title, fontsize=8, fontweight='bold', color='green')
        else:
            err = dbg.get('error', '?')
            # Still draw poles if available
            for key, marker, color in [('new_pole_point', 'X', 'lime'),
                                        ('old_pole_point', 'D', 'magenta')]:
                if dbg.get(key) is not None:
                    p = loc(dbg[key])
                    ax.plot(p[1], p[0], marker, color=color, ms=9,
                            mec='black', mew=1.5, alpha=0.6)
            pm = dbg.get('pole_method', '?')
            pc = dbg.get('pole_confidence', '?')
            ax.set_title(f'{name}  ✗  {err}\n{pm} ({pc})',
                         fontsize=8, fontweight='bold', color='firebrick')

        ax.legend(loc='upper right', fontsize=6, framealpha=0.8)
        ax.set_aspect('equal')
        ax.axis('off')

    # Hide empty panels
    for ax in axes_flat[n:]:
        ax.axis('off')

    fig.suptitle(f'Frame {frame_idx} – Individual cells', fontsize=12, fontweight='bold')
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 3. Curvature heatmaps
# ══════════════════════════════════════════════════════════════════════════════

def plot_curvature_heatmaps(frame, results, frame_idx, config=None):
    """Curvature colour-mapped onto the smoothed contour for each cell."""
    n     = len(results)
    ncols = min(n, 4)
    nrows = max(1, (n + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes_flat  = axes.flatten()

    for ax, r in zip(axes_flat, results):
        dbg  = r['debug_info']
        name = r.get('cell_name', str(r['label']))

        if 'smooth_pts' not in dbg or 'kappa' not in dbg:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(name); ax.axis('off'); continue

        sp     = dbg['smooth_pts']
        kappa  = dbg['kappa']

        pad  = 20
        r0   = max(0, int(sp[:, 0].min()) - pad)
        r1   = min(frame.shape[0], int(sp[:, 0].max()) + pad)
        c0   = max(0, int(sp[:, 1].min()) - pad)
        c1   = min(frame.shape[1], int(sp[:, 1].max()) + pad)
        crop = frame[r0:r1, c0:c1]
        ax.imshow(crop, cmap='gray',
                  vmin=np.percentile(crop, 1), vmax=np.percentile(crop, 99))

        local_sp = sp.copy()
        local_sp[:, 0] -= r0;  local_sp[:, 1] -= c0

        vlim = np.percentile(np.abs(kappa), 98) if len(kappa) else 0.1
        sc   = ax.scatter(local_sp[:, 1], local_sp[:, 0], c=kappa,
                          cmap='RdBu_r', s=8, vmin=-vlim, vmax=vlim, zorder=3)
        plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, label='κ')

        # Mark detected scar if present
        if r['scar_detected']:
            for pt in r['scar_points']:
                lp = np.array(pt) - np.array([r0, c0])
                ax.plot(lp[1], lp[0], 'y*', ms=10, mec='black', mew=1, zorder=5)

        # Mark peaks
        if 'peaks' in dbg and len(dbg['peaks']) > 0:
            pk = dbg['peaks']
            ax.plot(local_sp[pk, 1], local_sp[pk, 0], 'ko', ms=3, zorder=4,
                    label=f'{len(pk)} peaks')

        ax.set_title(f'{name}  {"✓" if r["scar_detected"] else "✗"}',
                     fontsize=9, color='green' if r['scar_detected'] else 'firebrick')
        ax.axis('off')

    for ax in axes_flat[n:]:
        ax.axis('off')

    fig.suptitle(f'Frame {frame_idx} – Curvature heatmaps', fontsize=12, fontweight='bold')
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 4. Curvature profiles
# ══════════════════════════════════════════════════════════════════════════════

def plot_curvature_profiles(frame_or_results, results_or_frame_idx, frame_idx_or_config=None, config=None):
    """
    κ vs. contour-index plots for every cell in the frame.

    Accepts two calling conventions so it works both standalone and inside
    the single-cell inspect loop alongside plot_individual_cells /
    plot_curvature_heatmaps (which all take frame as their first argument):

        plot_curvature_profiles(frame, results, frame_idx, config)   ← 4-arg
        plot_curvature_profiles(results, frame_idx, config)          ← 3-arg
    """
    if hasattr(frame_or_results, 'shape'):   # numpy image array → 4-arg form
        results   = results_or_frame_idx
        frame_idx = frame_idx_or_config
        # config already bound via keyword
    else:                                    # list of result dicts → 3-arg form
        results   = frame_or_results
        frame_idx = results_or_frame_idx
        config    = frame_idx_or_config
    n     = len(results)
    ncols = min(n, 3)
    nrows = max(1, (n + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows), squeeze=False)
    axes_flat  = axes.flatten()

    for ax, r in zip(axes_flat, results):
        dbg  = r['debug_info']
        name = r.get('cell_name', str(r['label']))

        if 'kappa' not in dbg:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(name); continue

        kappa      = dbg['kappa']
        # display_mask is always all-True (full cell searched).
        # valid_mask is the internal hemisphere hint — not shown here because
        # the hemisphere is not an exclusion zone.
        display_mask = dbg.get('display_mask', np.ones(len(kappa), dtype=bool))
        idx_arr    = np.arange(len(kappa))

        ax.plot(idx_arr, kappa, 'steelblue', lw=1, alpha=0.7)
        ax.axhline(0, color='k', ls='--', lw=0.5, alpha=0.4)
        ax.fill_between(idx_arr, 0, kappa, where=display_mask,
                        color='steelblue', alpha=0.08, label='Full cell (searched)')

        # Show the quality threshold so researchers can see what trips the QC flag
        if config is not None:
            thresh = getattr(config, 'CURVATURE_QUALITY_THRESHOLD', 0.10)
            ax.axhline( thresh, color='orange', ls=':', lw=1, alpha=0.7,
                       label=f'QC threshold (±{thresh})')
            ax.axhline(-thresh, color='orange', ls=':', lw=1, alpha=0.7)

        if 'peaks' in dbg and len(dbg['peaks']) > 0:
            pk = dbg['peaks']
            ax.plot(pk, kappa[pk], 'ro', ms=4, label=f'{len(pk)} peaks')

        if 'best_pair' in dbg:
            p1, p2 = dbg['best_pair']
            ax.plot([p1, p2], [kappa[p1], kappa[p2]], 'g^', ms=8,
                    mec='black', mew=1, label='Selected pair', zorder=5)

        ax.set_title(f'{name}', fontsize=9)
        ax.set_xlabel('Contour index', fontsize=8)
        ax.set_ylabel('κ', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)
        if ax is axes_flat[0]:
            ax.legend(fontsize=7)

    for ax in axes_flat[n:]:
        ax.axis('off')

    fig.suptitle(f'Frame {frame_idx} – Curvature profiles', fontsize=12, fontweight='bold')
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 5. Lineage tree (Gantt style)
# ══════════════════════════════════════════════════════════════════════════════

def plot_lineage_tree(all_results, config=None):
    """
    Horizontal Gantt chart with one row per unique cell name.
    Branch points mark observed division events.
    """
    # Collect timeline data
    timeline = {}       # name → {'first': int, 'last': int}
    divisions = {}      # parent_name → [(frame, [d0, d1])]

    for fd in all_results:
        fidx = fd['frame_idx']
        for cell in fd['cells']:
            name = cell.get('cell_name', '?')
            if name not in timeline:
                timeline[name] = {'first': fidx, 'last': fidx}
            else:
                timeline[name]['last'] = fidx

    # Collect division events from tracker lineage log
    if all_results and 'tracker' in all_results[-1]:
        tracker = all_results[-1]['tracker']
        for ev in tracker.lineage_log:
            p = ev['parent']
            if p not in divisions:
                divisions[p] = []
            divisions[p].append((ev['frame'], ev['daughters']))

    if not timeline:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'No tracking data', ha='center', va='center', fontsize=14)
        return fig

    # Sort names: base names alphabetically, then by depth
    def sort_key(n):
        depth = len(n) - len(n.lstrip('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
        return (len(n), n)

    names   = sorted(timeline.keys(), key=sort_key)
    n_cells = len(names)
    y_map   = {name: i for i, name in enumerate(names)}
    n_frames = max(fd['frame_idx'] for fd in all_results) + 1

    fig, ax  = plt.subplots(figsize=(max(10, n_frames * 0.6), max(4, n_cells * 0.5)))

    palette = plt.cm.tab20.colors
    color_map = {name: palette[i % len(palette)] for i, name in enumerate(names)}

    for name, span in timeline.items():
        y     = y_map[name]
        color = color_map[name]
        ax.barh(y, span['last'] - span['first'] + 1, left=span['first'],
                height=0.6, color=color, alpha=0.8, edgecolor='k', lw=0.5)
        ax.text(span['first'] - 0.3, y, name, va='center', ha='right', fontsize=8)

    # Division branch markers
    for parent, events in divisions.items():
        if parent not in y_map:
            continue
        for frame, daughters in events:
            py = y_map[parent]
            ax.axvline(frame, color='gray', lw=0.8, ls='--', alpha=0.5)
            for d in daughters:
                if d in y_map:
                    dy = y_map[d]
                    ax.annotate('', xy=(frame, dy), xytext=(frame, py),
                                arrowprops=dict(arrowstyle='->', color='gray',
                                                lw=1.0))

    ax.set_xlabel('Frame', fontsize=10)
    ax.set_yticks([])
    ax.set_xlim(-1, n_frames)
    ax.set_ylim(-0.8, n_cells - 0.2)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.2)
    ax.set_title('Cell Lineage', fontsize=13, fontweight='bold')
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Convenience wrapper
# ══════════════════════════════════════════════════════════════════════════════

def visualize_all(all_results, config, save_dir=None):
    """
    Generate all enabled visualisations for every frame.

    Parameters
    ----------
    config   : Config object (SHOW_* flags control which plots are made)
    save_dir : if provided, figures are saved as PNGs here
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for fd in all_results:
        fidx  = fd['frame_idx']
        frame = fd['frame']
        cells = fd['cells']

        figs = {}

        if config.SHOW_CELL_OVERVIEW:
            figs['overview'] = plot_frame_overview(frame, cells, fidx, config)

        if config.SHOW_INDIVIDUAL_CELLS:
            figs['cells'] = plot_individual_cells(frame, cells, fidx, config)

        if config.SHOW_CURVATURE_HEATMAPS:
            figs['heatmaps'] = plot_curvature_heatmaps(frame, cells, fidx, config)

        if config.SHOW_CURVATURE_PROFILES:
            figs['profiles'] = plot_curvature_profiles(cells, fidx, config)

        for name, fig in figs.items():
            plt.figure(fig.number)
            plt.show()
            if save_dir and config.SAVE_FIGURES:
                path = os.path.join(save_dir, f'frame{fidx:04d}_{name}.png')
                fig.savefig(path, dpi=150, bbox_inches='tight')
                print(f"  Saved {path}")
            plt.close(fig)

    if config.SHOW_LINEAGE_TREE:
        fig = plot_lineage_tree(all_results, config)
        plt.show()
        if save_dir and config.SAVE_FIGURES:
            path = os.path.join(save_dir, 'lineage_tree.png')
            fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
