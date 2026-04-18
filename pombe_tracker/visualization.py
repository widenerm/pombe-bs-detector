"""
visualization.py  –  All plotting functions.

Each function returns the matplotlib Figure so the caller can save or display
it.  None of the functions call plt.show() themselves.

Poster-ready defaults
─────────────────────
Pass  poster=True  to any public function to activate:
  • Higher base font sizes (suitable for 300 DPI print at poster scale)
  • Thicker lines and larger markers
  • Suppressed debug tags (scar_source, match_type) from titles
  • Tight, clean axis labels for a non-specialist audience

The  save_poster_figure()  helper at the bottom enforces 300 DPI and a
transparent background so figures drop cleanly onto a dark poster layout.
"""
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe


# ── Poster style context ──────────────────────────────────────────────────────

def _poster_rc(poster=False):
    """Return a dict of rcParam overrides for poster mode."""
    if not poster:
        return {}
    return {
        'font.size':        14,
        'axes.titlesize':   15,
        'axes.labelsize':   13,
        'xtick.labelsize':  11,
        'ytick.labelsize':  11,
        'legend.fontsize':  11,
        'lines.linewidth':  2.0,
        'lines.markersize': 9,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 1. Frame overview
# ══════════════════════════════════════════════════════════════════════════════

def plot_frame_overview(frame, results, frame_idx, config=None, poster=False):
    """
    One panel showing all cells in the frame.
    Green outline  = scar detected
    Red dashed     = not detected
    Orange overlay = segmentation quality flag
    Cyan dotted    = pole-to-pole neighbor connections
    """
    rc = _poster_rc(poster)
    with matplotlib.rc_context(rc):
        fig, ax = plt.subplots(figsize=(14, 11) if poster else (10, 8))
        ax.imshow(frame, cmap='gray',
                  vmin=np.percentile(frame, 1), vmax=np.percentile(frame, 99))

        title = f'Frame {frame_idx}  –  {len(results)} cells'
        ax.set_title(title, fontsize=rc.get('axes.titlesize', 13), fontweight='bold')
        ax.axis('off')

        lw_cell   = 2.5 if poster else 1.5
        lw_nb     = 1.2 if poster else 0.8
        fs_label  = rc.get('font.size', 7) - 1

        for r in results:
            contour = r['contour']
            name    = r.get('cell_name', str(r['label']))
            cx, cy  = contour[:, 1].mean(), contour[:, 0].mean()

            if r['scar_detected']:
                ax.plot(contour[:, 1], contour[:, 0],
                        color='lime', lw=lw_cell, alpha=0.9)
                ax.text(cx, cy, name, color='white',
                        fontsize=fs_label, fontweight='bold',
                        ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.2', fc='green',
                                  alpha=0.7, lw=0))
            else:
                ax.plot(contour[:, 1], contour[:, 0],
                        color='tomato', lw=lw_cell * 0.8, alpha=0.7, ls='--')
                ax.text(cx, cy, name, color='white',
                        fontsize=fs_label, fontweight='bold',
                        ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.2', fc='firebrick',
                                  alpha=0.7, lw=0))

            if r.get('seg_quality', 'ok') != 'ok':
                ax.plot(contour[:, 1], contour[:, 0],
                        color='orange', lw=lw_cell + 1, alpha=0.95, zorder=4)
                ax.text(cx, cy - 10,
                        f"\u26a0 {r['seg_quality'].replace('_', ' ')}",
                        color='orange', fontsize=max(fs_label - 1, 5),
                        ha='center', va='bottom', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.1', fc='black',
                                  alpha=0.6, lw=0))

            for nb in r.get('neighbors', []):
                op, tp = nb['our_pole'], nb['their_pole']
                ax.plot([op[1], tp[1]], [op[0], tp[0]],
                        color='cyan', lw=lw_nb, ls=':', alpha=0.6)

        legend = [
            mpatches.Patch(color='lime',   label='Birth scar detected'),
            mpatches.Patch(color='tomato', label='No scar detected'),
            mpatches.Patch(color='orange', label='\u26a0 Segmentation flag'),
            Line2D([0], [0], color='cyan', ls=':', lw=1.5,
                   label='Pole-to-pole neighbor'),
        ]
        ax.legend(handles=legend, loc='upper right',
                  fontsize=rc.get('legend.fontsize', 8), framealpha=0.85)
        fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 2. Individual cell panels
# ══════════════════════════════════════════════════════════════════════════════

def plot_individual_cells(frame, results, frame_idx, config=None, poster=False):
    """
    One subplot per cell showing the birth scar, poles, and measurement lines.

    poster=True  suppresses debug tags (scar_source, match_type) and uses
    larger fonts / markers suitable for 300 DPI print.
    """
    rc = _poster_rc(poster)
    with matplotlib.rc_context(rc):
        n     = len(results)
        ncols = min(n, 4)
        nrows = max(1, (n + ncols - 1) // ncols)
        fw    = (6 * ncols) if poster else (5 * ncols)
        fh    = (6 * nrows) if poster else (5 * nrows)
        fig, axes = plt.subplots(nrows, ncols, figsize=(fw, fh), squeeze=False)
        axes_flat = axes.flatten()

        ms_scar  = 10 if poster else 6
        ms_pole  = 14 if poster else 11
        ms_mid   = 12 if poster else 9
        lw_cont  = 2.0 if poster else 1.5
        lw_meas  = 2.2 if poster else 1.8
        lw_scar  = 3.0 if poster else 2.5

        for idx, (ax, r) in enumerate(zip(axes_flat, results)):
            contour = r['contour']
            dbg     = r['debug_info']
            name    = r.get('cell_name', str(r['label']))

            pad  = 25 if poster else 20
            r0   = max(0, int(contour[:, 0].min()) - pad)
            r1   = min(frame.shape[0], int(contour[:, 0].max()) + pad)
            c0   = max(0, int(contour[:, 1].min()) - pad)
            c1   = min(frame.shape[1], int(contour[:, 1].max()) + pad)
            crop = frame[r0:r1, c0:c1]
            ax.imshow(crop, cmap='gray',
                      vmin=np.percentile(crop, 1), vmax=np.percentile(crop, 99))

            def loc(pt):
                return np.array(pt) - np.array([r0, c0])

            local_c = contour.copy()
            local_c[:, 0] -= r0
            local_c[:, 1] -= c0
            ax.plot(local_c[:, 1], local_c[:, 0], 'cyan', lw=lw_cont, alpha=0.8)

            if r['scar_detected']:
                sp  = r.get('scar_points')
                mp  = loc(r['scar_midpoint'])
                np_ = loc(dbg['new_pole_point'])
                op_ = loc(dbg['old_pole_point'])

                if sp is not None:
                    s1 = loc(sp[0])
                    s2 = loc(sp[1])
                    if np.linalg.norm(np.array(s1) - np.array(s2)) > 1.0:
                        ax.plot([s1[1], s2[1]], [s1[0], s2[0]],
                                'yellow', lw=lw_scar, zorder=5,
                                label='Birth scar')
                        ax.plot(s1[1], s1[0], 'o', color='yellow',
                                ms=ms_scar, mec='black', mew=1.5, zorder=6)
                        ax.plot(s2[1], s2[0], 'o', color='yellow',
                                ms=ms_scar, mec='black', mew=1.5, zorder=6)

                ax.plot(mp[1], mp[0], 'o', color='white',
                        ms=ms_mid, mec='black', mew=1.5, zorder=7,
                        label='Scar midpoint')
                ax.plot(np_[1], np_[0], 'X', color='lime',
                        ms=ms_pole, mec='black', mew=1.5, zorder=7,
                        label='New pole')
                ax.plot(op_[1], op_[0], 'D', color='magenta',
                        ms=ms_pole * 0.75, mec='black', mew=1.5, zorder=7,
                        label='Old pole')

                ni, oi = r['new_end_length'], r['old_end_length']
                ax.plot([mp[1], np_[1]], [mp[0], np_[0]],
                        color='lime', lw=lw_meas, ls='--',
                        label=f'New end  {ni:.0f} px')
                ax.plot([mp[1], op_[1]], [mp[0], op_[0]],
                        color='magenta', lw=lw_meas, ls='--',
                        label=f'Old end  {oi:.0f} px')

                pm = dbg.get('pole_method', '?')
                pc = dbg.get('pole_confidence', '?')
                if poster:
                    title = (f'{name}   N={ni:.0f} px   O={oi:.0f} px'
                             f'   ratio={ni/max(oi, 1e-3):.2f}')
                else:
                    mt  = dbg.get('match_type', '?')
                    src = r.get('scar_source', 'raw')
                    title = (f'{name}  \u2713  [{mt}] [{src}]\n'
                             f'N={ni:.0f}  O={oi:.0f}  R={ni/max(oi,1e-3):.2f}\n'
                             f'{pm} ({pc})')
                ax.set_title(title,
                             fontsize=rc.get('axes.titlesize', 8),
                             fontweight='bold', color='green')
            else:
                err = dbg.get('error', '?')
                for key, marker, color in [('new_pole_point', 'X', 'lime'),
                                            ('old_pole_point', 'D', 'magenta')]:
                    if dbg.get(key) is not None:
                        p = loc(dbg[key])
                        ax.plot(p[1], p[0], marker, color=color,
                                ms=ms_pole, mec='black', mew=1.5, alpha=0.6)
                pm = dbg.get('pole_method', '?')
                pc = dbg.get('pole_confidence', '?')
                if poster:
                    ax.set_title(f'{name}   no scar detected',
                                 fontsize=rc.get('axes.titlesize', 8),
                                 fontweight='bold', color='firebrick')
                else:
                    ax.set_title(f'{name}  \u2717  {err}\n{pm} ({pc})',
                                 fontsize=rc.get('axes.titlesize', 8),
                                 fontweight='bold', color='firebrick')

            ax.legend(loc='upper right',
                      fontsize=rc.get('legend.fontsize', 6),
                      framealpha=0.85)
            ax.set_aspect('equal')
            ax.axis('off')

        for ax in axes_flat[n:]:
            ax.axis('off')

        if not poster:
            fig.suptitle(f'Frame {frame_idx} \u2013 Individual cells',
                         fontsize=12, fontweight='bold')
        fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 3. Curvature heatmaps
# ══════════════════════════════════════════════════════════════════════════════

def plot_curvature_heatmaps(frame, results, frame_idx, config=None, poster=False):
    """Curvature color-mapped onto the smoothed contour for each cell."""
    rc = _poster_rc(poster)
    with matplotlib.rc_context(rc):
        n     = len(results)
        ncols = min(n, 4)
        nrows = max(1, (n + ncols - 1) // ncols)
        fw    = (6 * ncols) if poster else (5 * ncols)
        fh    = (5 * nrows) if poster else (4 * nrows)
        fig, axes = plt.subplots(nrows, ncols, figsize=(fw, fh), squeeze=False)
        axes_flat = axes.flatten()

        ms_peak = 5 if poster else 3
        ms_scar = 14 if poster else 10

        for ax, r in zip(axes_flat, results):
            dbg  = r['debug_info']
            name = r.get('cell_name', str(r['label']))

            if 'smooth_pts' not in dbg or 'kappa' not in dbg:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.set_title(name)
                ax.axis('off')
                continue

            sp    = dbg['smooth_pts']
            kappa = dbg['kappa']

            pad  = 25 if poster else 20
            r0   = max(0, int(sp[:, 0].min()) - pad)
            r1   = min(frame.shape[0], int(sp[:, 0].max()) + pad)
            c0   = max(0, int(sp[:, 1].min()) - pad)
            c1   = min(frame.shape[1], int(sp[:, 1].max()) + pad)
            crop = frame[r0:r1, c0:c1]
            ax.imshow(crop, cmap='gray',
                      vmin=np.percentile(crop, 1), vmax=np.percentile(crop, 99))

            local_sp = sp.copy()
            local_sp[:, 0] -= r0
            local_sp[:, 1] -= c0

            vlim = np.percentile(np.abs(kappa), 98) if len(kappa) else 0.1
            sc   = ax.scatter(local_sp[:, 1], local_sp[:, 0], c=kappa,
                              cmap='RdBu_r', s=12 if poster else 8,
                              vmin=-vlim, vmax=vlim, zorder=3)
            cb = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
            cb.set_label('Curvature  κ',
                         fontsize=rc.get('axes.labelsize', 8))
            cb.ax.tick_params(labelsize=rc.get('xtick.labelsize', 7))

            if r['scar_detected']:
                sp_pts = r.get('scar_points')
                if sp_pts is not None:
                    pt1 = np.array(sp_pts[0])
                    pt2 = np.array(sp_pts[1])
                    if np.linalg.norm(pt1 - pt2) > 1.0:
                        for pt in [pt1, pt2]:
                            lp = pt - np.array([r0, c0])
                            ax.plot(lp[1], lp[0], 'y*', ms=ms_scar,
                                    mec='black', mew=1, zorder=5)
                if r.get('scar_midpoint') is not None:
                    mp = np.array(r['scar_midpoint']) - np.array([r0, c0])
                    ax.plot(mp[1], mp[0], 'w*', ms=ms_scar * 0.8,
                            mec='black', mew=0.8, zorder=5,
                            label='Scar midpoint')

            if 'peaks' in dbg and len(dbg['peaks']) > 0:
                pk = dbg['peaks']
                ax.plot(local_sp[pk, 1], local_sp[pk, 0], 'ko',
                        ms=ms_peak, zorder=4,
                        label=f'{len(pk)} curvature peaks')

            detected = r['scar_detected']
            if poster:
                label = f'{name}   {"Birth scar detected" if detected else "No scar detected"}'
            else:
                src     = r.get('scar_source', '')
                src_tag = f' [{src}]' if src and src != 'raw' else ''
                label   = f'{name}  {"\u2713" if detected else "\u2717"}{src_tag}'

            ax.set_title(label,
                         fontsize=rc.get('axes.titlesize', 9),
                         color='green' if detected else 'firebrick')
            if poster and any('peaks' in dbg for dbg in [dbg]):
                ax.legend(loc='upper right',
                          fontsize=rc.get('legend.fontsize', 8),
                          framealpha=0.85)
            ax.axis('off')

        for ax in axes_flat[n:]:
            ax.axis('off')

        if not poster:
            fig.suptitle(f'Frame {frame_idx} \u2013 Curvature heatmaps',
                         fontsize=12, fontweight='bold')
        fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 4. Curvature profiles
# ══════════════════════════════════════════════════════════════════════════════

def plot_curvature_profiles(frame_or_results, results_or_frame_idx,
                             frame_idx_or_config=None, config=None,
                             poster=False):
    """
    Contour curvature (κ) vs. contour index for every cell in the frame.

    Accepts two calling conventions:
        plot_curvature_profiles(frame, results, frame_idx, config)   ← 4-arg
        plot_curvature_profiles(results, frame_idx, config)          ← 3-arg
    """
    if hasattr(frame_or_results, 'shape'):
        results   = results_or_frame_idx
        frame_idx = frame_idx_or_config
    else:
        results   = frame_or_results
        frame_idx = results_or_frame_idx
        config    = frame_idx_or_config

    rc = _poster_rc(poster)
    with matplotlib.rc_context(rc):
        n     = len(results)
        ncols = min(n, 3)
        nrows = max(1, (n + ncols - 1) // ncols)
        fw    = (6 * ncols) if poster else (5 * ncols)
        fh    = (4 * nrows) if poster else (3 * nrows)
        fig, axes = plt.subplots(nrows, ncols, figsize=(fw, fh), squeeze=False)
        axes_flat = axes.flatten()

        for ax, r in zip(axes_flat, results):
            dbg  = r['debug_info']
            name = r.get('cell_name', str(r['label']))

            if 'kappa' not in dbg:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.set_title(name)
                continue

            kappa        = dbg['kappa']
            display_mask = dbg.get('display_mask',
                                   np.ones(len(kappa), dtype=bool))
            idx_arr      = np.arange(len(kappa))

            lw = 1.5 if poster else 1.0
            ax.plot(idx_arr, kappa, color='steelblue', lw=lw, alpha=0.85)
            ax.axhline(0, color='k', ls='--', lw=0.6, alpha=0.4)
            ax.fill_between(idx_arr, 0, kappa, where=display_mask,
                            color='steelblue', alpha=0.10,
                            label='Contour (full search)')

            if config is not None:
                thresh = getattr(config, 'CURVATURE_QUALITY_THRESHOLD', 0.10)
                ax.axhline(thresh, color='orange', ls=':', lw=1.2, alpha=0.8,
                           label=f'QC threshold (±{thresh})')
                ax.axhline(-thresh, color='orange', ls=':', lw=1.2, alpha=0.8)

            if 'peaks' in dbg and len(dbg['peaks']) > 0:
                pk = dbg['peaks']
                ax.plot(pk, kappa[pk], 'ro',
                        ms=6 if poster else 4,
                        label=f'{len(pk)} curvature peaks')

            if 'best_pair' in dbg:
                p1, p2 = dbg['best_pair']
                ax.plot([p1, p2], [kappa[p1], kappa[p2]], 'g^',
                        ms=11 if poster else 8,
                        mec='black', mew=1.2,
                        label='Selected scar pair', zorder=5)

            ax.set_title(f'Cell {name}' if poster else name,
                         fontsize=rc.get('axes.titlesize', 9))
            ax.set_xlabel('Contour position (index)',
                          fontsize=rc.get('axes.labelsize', 8))
            ax.set_ylabel('Contour curvature  κ',
                          fontsize=rc.get('axes.labelsize', 8))
            ax.tick_params(labelsize=rc.get('xtick.labelsize', 7))
            ax.grid(True, alpha=0.2)
            if ax is axes_flat[0]:
                ax.legend(fontsize=rc.get('legend.fontsize', 7),
                          framealpha=0.85)

        for ax in axes_flat[n:]:
            ax.axis('off')

        if not poster:
            fig.suptitle(f'Frame {frame_idx} \u2013 Curvature profiles',
                         fontsize=12, fontweight='bold')
        fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 5. Lineage tree (Gantt style)
# ══════════════════════════════════════════════════════════════════════════════

def plot_lineage_tree(all_results, config=None, poster=False):
    """
    Horizontal Gantt chart with one row per unique cell name.
    Branch points mark observed division events.
    """
    rc = _poster_rc(poster)
    with matplotlib.rc_context(rc):
        timeline  = {}
        divisions = {}

        for fd in all_results:
            fidx = fd['frame_idx']
            for cell in fd['cells']:
                name = cell.get('cell_name', '?')
                if name not in timeline:
                    timeline[name] = {'first': fidx, 'last': fidx}
                else:
                    timeline[name]['last'] = fidx

        if all_results and 'tracker' in all_results[-1]:
            tracker = all_results[-1]['tracker']
            for ev in tracker.lineage_log:
                p = ev['parent']
                if p not in divisions:
                    divisions[p] = []
                divisions[p].append((ev['frame'], ev['daughters']))

        if not timeline:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, 'No tracking data',
                    ha='center', va='center', fontsize=14)
            return fig

        def sort_key(n):
            return (len(n), n)

        names    = sorted(timeline.keys(), key=sort_key)
        n_cells  = len(names)
        y_map    = {name: i for i, name in enumerate(names)}
        n_frames = max(fd['frame_idx'] for fd in all_results) + 1

        fw = max(12, n_frames * 0.8) if poster else max(10, n_frames * 0.6)
        fh = max(5,  n_cells  * 0.7) if poster else max(4,  n_cells  * 0.5)
        fig, ax = plt.subplots(figsize=(fw, fh))

        palette   = plt.cm.tab20.colors
        color_map = {name: palette[i % len(palette)] for i, name in enumerate(names)}

        bar_h = 0.7 if poster else 0.6

        for name, span in timeline.items():
            y     = y_map[name]
            color = color_map[name]
            ax.barh(y, span['last'] - span['first'] + 1, left=span['first'],
                    height=bar_h, color=color, alpha=0.85,
                    edgecolor='k', lw=0.6)
            ax.text(span['first'] - 0.3, y, name,
                    va='center', ha='right',
                    fontsize=rc.get('font.size', 8),
                    fontweight='bold' if poster else 'normal')

        for parent, events in divisions.items():
            if parent not in y_map:
                continue
            for frame, daughters in events:
                py = y_map[parent]
                ax.axvline(frame, color='gray', lw=1.0, ls='--', alpha=0.5)
                for d in daughters:
                    if d in y_map:
                        dy = y_map[d]
                        ax.annotate('', xy=(frame, dy), xytext=(frame, py),
                                    arrowprops=dict(arrowstyle='->',
                                                    color='gray', lw=1.2))

        ax.set_xlabel('Frame', fontsize=rc.get('axes.labelsize', 10))
        ax.set_yticks([])
        ax.set_xlim(-1, n_frames)
        ax.set_ylim(-0.8, n_cells - 0.2)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.2)
        title = 'Cell Lineage Tree' if poster else 'Cell Lineage'
        ax.set_title(title,
                     fontsize=rc.get('axes.titlesize', 13),
                     fontweight='bold')
        fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 6. Single-cell pipeline overview  (poster figure)
# ══════════════════════════════════════════════════════════════════════════════

def plot_pipeline_overview(frame, result, config=None, poster=True):
    """
    Multi-panel figure showing each stage of the BS-Detector pipeline for a
    single cell, designed as a poster hero figure.

    Panels (left → right):
      1. Raw brightfield crop
      2. Cellpose segmentation mask
      3. Smoothed contour + PCA long axis
      4. Curvature heatmap on contour
      5. Curvature profile with selected scar pair
      6. Detection result: scar, poles, measurement lines

    Parameters
    ----------
    frame   : 2-D numpy array (the full frame)
    result  : single cell result dict from run_pipeline
    config  : Config object (optional, used for QC threshold line)
    poster  : if True, uses larger fonts/markers
    """
    rc = _poster_rc(poster)
    with matplotlib.rc_context(rc):
        dbg     = result['debug_info']
        contour = result['contour']

        # Crop geometry
        pad = 30 if poster else 20
        r0  = max(0, int(contour[:, 0].min()) - pad)
        r1  = min(frame.shape[0], int(contour[:, 0].max()) + pad)
        c0  = max(0, int(contour[:, 1].min()) - pad)
        c1  = min(frame.shape[1], int(contour[:, 1].max()) + pad)
        crop = frame[r0:r1, c0:c1]

        def loc(pt):
            return np.array(pt) - np.array([r0, c0])

        local_c = contour.copy()
        local_c[:, 0] -= r0
        local_c[:, 1] -= c0

        n_panels = 6
        fw = 5 * n_panels if poster else 4 * n_panels
        fh = 5.5 if poster else 4.5
        fig = plt.figure(figsize=(fw, fh))
        gs  = GridSpec(1, n_panels, figure=fig,
                       wspace=0.08, left=0.02, right=0.98,
                       top=0.82, bottom=0.05)

        vmin = np.percentile(crop, 1)
        vmax = np.percentile(crop, 99)

        panel_titles = [
            'Brightfield',
            'Segmentation\n(Cellpose)',
            'Contour + Long Axis\n(PCA)',
            'Curvature Heatmap',
            'Curvature Profile',
            'Birth Scar Detection',
        ]
        title_fs  = rc.get('axes.titlesize', 13) if poster else 10
        label_fs  = rc.get('axes.labelsize',  11) if poster else 8
        lw_cont   = 2.5 if poster else 1.8
        ms_marker = 10 if poster else 7

        # ── Panel 1: raw brightfield ──────────────────────────────────────────
        ax1 = fig.add_subplot(gs[0])
        ax1.imshow(crop, cmap='gray', vmin=vmin, vmax=vmax)
        ax1.set_title(panel_titles[0], fontsize=title_fs, fontweight='bold', pad=8)
        ax1.axis('off')

        # ── Panel 2: segmentation mask ────────────────────────────────────────
        ax2 = fig.add_subplot(gs[1])
        ax2.imshow(crop, cmap='gray', vmin=vmin, vmax=vmax)
        # Draw filled mask as a semi-transparent overlay
        from skimage.draw import polygon as sk_polygon
        mask_overlay = np.zeros((*crop.shape[:2], 4), dtype=float)
        rr, cc = sk_polygon(local_c[:, 0], local_c[:, 1], crop.shape[:2])
        mask_overlay[rr, cc] = [0.2, 0.8, 0.2, 0.35]   # green RGBA
        ax2.imshow(mask_overlay)
        ax2.plot(local_c[:, 1], local_c[:, 0],
                 color='lime', lw=lw_cont, alpha=0.9)
        ax2.set_title(panel_titles[1], fontsize=title_fs, fontweight='bold', pad=8)
        ax2.axis('off')

        # ── Panel 3: smoothed contour + PCA axis ──────────────────────────────
        ax3 = fig.add_subplot(gs[2])
        ax3.imshow(crop, cmap='gray', vmin=vmin, vmax=vmax)

        sp   = dbg['smooth_pts']
        lsp  = sp.copy()
        lsp[:, 0] -= r0
        lsp[:, 1] -= c0
        ax3.plot(lsp[:, 1], lsp[:, 0],
                 color='cyan', lw=lw_cont, alpha=0.9,
                 label='Smoothed contour')

        center = dbg['center']
        axis   = dbg['axis']
        lc     = loc(center)

        # Draw the long axis as a line through the cell
        ep1_pt = dbg.get('new_pole_point')
        ep2_pt = dbg.get('old_pole_point')
        if ep1_pt is not None and ep2_pt is not None:
            le1 = loc(ep1_pt)
            le2 = loc(ep2_pt)
            ax3.plot([le1[1], le2[1]], [le1[0], le2[0]],
                     color='yellow', lw=2.0, ls='--', alpha=0.9,
                     label='Long axis (PCA)')
            ax3.plot(le1[1], le1[0], 'o', color='white',
                     ms=ms_marker, mec='black', mew=1.5, zorder=5)
            ax3.plot(le2[1], le2[0], 'o', color='white',
                     ms=ms_marker, mec='black', mew=1.5, zorder=5)
        ax3.plot(lc[1], lc[0], '+', color='yellow',
                 ms=ms_marker + 2, mew=2.5, zorder=6, label='Cell center')
        ax3.set_title(panel_titles[2], fontsize=title_fs, fontweight='bold', pad=8)
        ax3.axis('off')

        # ── Panel 4: curvature heatmap ─────────────────────────────────────────
        ax4 = fig.add_subplot(gs[3])
        ax4.imshow(crop, cmap='gray', vmin=vmin, vmax=vmax)

        kappa = dbg['kappa']
        vlim  = np.percentile(np.abs(kappa), 98) if len(kappa) else 0.1
        sc    = ax4.scatter(lsp[:, 1], lsp[:, 0], c=kappa,
                            cmap='RdBu_r', s=14 if poster else 8,
                            vmin=-vlim, vmax=vlim, zorder=3)

        if 'peaks' in dbg and len(dbg['peaks']) > 0:
            pk = dbg['peaks']
            ax4.plot(lsp[pk, 1], lsp[pk, 0], 'ko',
                     ms=6 if poster else 4, zorder=4,
                     label='Curvature peaks')

        cb = plt.colorbar(sc, ax=ax4, fraction=0.046, pad=0.03,
                          shrink=0.85)
        cb.set_label('κ', fontsize=label_fs)
        cb.ax.tick_params(labelsize=label_fs - 2)
        ax4.set_title(panel_titles[3], fontsize=title_fs, fontweight='bold', pad=8)
        ax4.axis('off')

        # ── Panel 5: curvature profile ────────────────────────────────────────
        ax5 = fig.add_subplot(gs[4])
        idx_arr = np.arange(len(kappa))
        lw_line = 1.8 if poster else 1.2

        ax5.plot(idx_arr, kappa, color='steelblue', lw=lw_line, alpha=0.9)
        ax5.axhline(0, color='gray', ls='--', lw=0.8, alpha=0.5)
        ax5.fill_between(idx_arr, 0, kappa,
                         color='steelblue', alpha=0.12)

        if config is not None:
            thresh = getattr(config, 'CURVATURE_QUALITY_THRESHOLD', 0.10)
            ax5.axhline(thresh, color='orange', ls=':', lw=1.2, alpha=0.8)
            ax5.axhline(-thresh, color='orange', ls=':', lw=1.2, alpha=0.8)

        if 'peaks' in dbg and len(dbg['peaks']) > 0:
            pk = dbg['peaks']
            ax5.plot(pk, kappa[pk], 'ro',
                     ms=7 if poster else 5,
                     label=f'{len(pk)} peaks')

        if 'best_pair' in dbg:
            p1, p2 = dbg['best_pair']
            ax5.plot([p1, p2], [kappa[p1], kappa[p2]], 'g^',
                     ms=12 if poster else 9,
                     mec='black', mew=1.2,
                     label='Selected scar pair', zorder=5)

        ax5.set_xlabel('Contour position', fontsize=label_fs)
        ax5.set_ylabel('Curvature  κ', fontsize=label_fs)
        ax5.tick_params(labelsize=label_fs - 1)
        ax5.grid(True, alpha=0.2)
        ax5.legend(fontsize=label_fs - 1, framealpha=0.85)
        ax5.set_title(panel_titles[4], fontsize=title_fs, fontweight='bold', pad=8)

        # ── Panel 6: final detection result ───────────────────────────────────
        ax6 = fig.add_subplot(gs[5])
        ax6.imshow(crop, cmap='gray', vmin=vmin, vmax=vmax)
        ax6.plot(local_c[:, 1], local_c[:, 0], 'cyan', lw=lw_cont, alpha=0.7)

        if result['scar_detected']:
            sp_pts = result.get('scar_points')
            mp     = loc(result['scar_midpoint'])
            np_pt  = loc(dbg['new_pole_point'])
            op_pt  = loc(dbg['old_pole_point'])

            if sp_pts is not None:
                s1 = loc(sp_pts[0])
                s2 = loc(sp_pts[1])
                if np.linalg.norm(np.array(s1) - np.array(s2)) > 1.0:
                    ax6.plot([s1[1], s2[1]], [s1[0], s2[0]],
                             'yellow', lw=3.5 if poster else 2.5, zorder=5,
                             label='Birth scar')
                    ax6.plot(s1[1], s1[0], 'o', color='yellow',
                             ms=ms_marker, mec='black', mew=1.5, zorder=6)
                    ax6.plot(s2[1], s2[0], 'o', color='yellow',
                             ms=ms_marker, mec='black', mew=1.5, zorder=6)

            ax6.plot(mp[1], mp[0], 'o', color='white',
                     ms=ms_marker + 2, mec='black', mew=1.5, zorder=7,
                     label='Scar midpoint')
            ax6.plot(np_pt[1], np_pt[0], 'X', color='lime',
                     ms=ms_marker + 4 if poster else ms_marker + 2,
                     mec='black', mew=1.5, zorder=7, label='New pole')
            ax6.plot(op_pt[1], op_pt[0], 'D', color='magenta',
                     ms=ms_marker + 1, mec='black', mew=1.5, zorder=7,
                     label='Old pole')

            ni = result['new_end_length']
            oi = result['old_end_length']
            ax6.plot([mp[1], np_pt[1]], [mp[0], np_pt[0]],
                     color='lime', lw=2.2 if poster else 1.8, ls='--',
                     label=f'New end  {ni:.0f} px')
            ax6.plot([mp[1], op_pt[1]], [mp[0], op_pt[0]],
                     color='magenta', lw=2.2 if poster else 1.8, ls='--',
                     label=f'Old end  {oi:.0f} px')

        ax6.legend(loc='upper right',
                   fontsize=label_fs - 1 if poster else 6,
                   framealpha=0.85)
        ax6.set_title(panel_titles[5], fontsize=title_fs, fontweight='bold', pad=8)
        ax6.axis('off')

        # ── Shared super-title ────────────────────────────────────────────────
        name = result.get('cell_name', '')
        suptitle = (f'BS-Detector Pipeline  –  Cell {name}'
                    if name else 'BS-Detector Pipeline')
        fig.suptitle(suptitle,
                     fontsize=rc.get('axes.titlesize', 14) + 2,
                     fontweight='bold', y=0.97)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Poster export helper
# ══════════════════════════════════════════════════════════════════════════════

def save_poster_figure(fig, path, dpi=300):
    """
    Save *fig* at print resolution with a transparent background.

    Parameters
    ----------
    fig  : matplotlib Figure
    path : output file path (.png or .pdf recommended)
    dpi  : dots per inch (300 is standard for poster printing)
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight',
                facecolor='none', transparent=True)
    print(f'Saved poster figure: {path}')


# ══════════════════════════════════════════════════════════════════════════════
# Convenience wrapper
# ══════════════════════════════════════════════════════════════════════════════

def visualize_all(all_results, config, save_dir=None):
    """
    Generate all enabled visualizations for every frame.

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
                print(f'  Saved {path}')
            plt.close(fig)

    if config.SHOW_LINEAGE_TREE:
        fig = plot_lineage_tree(all_results, config)
        plt.show()
        if save_dir and config.SAVE_FIGURES:
            path = os.path.join(save_dir, 'lineage_tree.png')
            fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
