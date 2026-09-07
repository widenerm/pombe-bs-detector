"""
visualize_results.py  –  Generate performance metric plots and scar overlay
examples for the BS-Detector vs. Carmen ground-truth comparison.

Outputs (all written to --out dir):
  metrics_dashboard.png   - CDF, histograms, Bland-Altman, pole identity
  scar_overlays.png       - example cells showing det vs GT scar placement
  results.html            - self-contained HTML embedding both images + stats
"""
import argparse
import ast
import base64
import csv
import io
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import tifffile
from PIL import Image


# ── helpers ──────────────────────────────────────────────────────────────────

def _parse_point(s):
    if not s or s == 'None':
        return None
    return tuple(ast.literal_eval(s))

def _to_float(s):
    if s is None or s == '' or s == 'None':
        return None
    return float(s)

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ── data loading ─────────────────────────────────────────────────────────────

def load_comparison(path):
    rows = []
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def load_gt(path):
    rows = {}
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            key = (r['video_id'], int(r['frame']), r['cell_name'])
            rows[key] = r
    return rows


def load_det_csvs(paths):
    rows = {}
    for path in paths:
        with open(path, newline='') as f:
            for r in csv.DictReader(f):
                key = (r.get('video_id',''), int(r['frame']), r['cell_name'])
                rows[key] = r
    return rows


# ── palette ──────────────────────────────────────────────────────────────────

BG      = '#0f1117'
PANEL   = '#1a1d27'
ACCENT  = '#4f8ef7'   # detector blue
GT_COL  = '#f7c44f'   # Carmen gold
ERR_COL = '#e05c5c'   # error red
TEXT    = '#e8eaf0'
GRID    = '#2a2d3a'
GREEN   = '#4ecb8d'
ORANGE  = '#f7944f'

plt.rcParams.update({
    'figure.facecolor': BG,
    'axes.facecolor':   PANEL,
    'axes.edgecolor':   GRID,
    'axes.labelcolor':  TEXT,
    'xtick.color':      TEXT,
    'ytick.color':      TEXT,
    'text.color':       TEXT,
    'grid.color':       GRID,
    'grid.linewidth':   0.5,
    'axes.grid':        True,
    'font.family':      'sans-serif',
    'font.size':        9,
})


# ── metrics dashboard ─────────────────────────────────────────────────────────

def plot_metrics_dashboard(rows):
    scar_dists   = [float(r['scar_dist_px']) for r in rows if r.get('scar_dist_px')]
    area_errs    = [float(r['area_error'])    for r in rows if r.get('area_error')]
    area_gt      = [float(r['area_gt'])       for r in rows if r.get('area_gt')]
    poleA_errs   = [float(r['poleA_error'])   for r in rows if r.get('poleA_error')]
    poleA_gt     = [float(r['poleA_gt'])      for r in rows if r.get('poleA_gt')]
    poleB_errs   = [float(r['poleB_error'])   for r in rows if r.get('poleB_error')]
    poleB_gt     = [float(r['poleB_gt'])      for r in rows if r.get('poleB_gt')]
    pole_agree   = [r['new_pole_agreement']   for r in rows if r.get('new_pole_agreement')]
    n_agree      = sum(1 for v in pole_agree if v.lower() == 'true')

    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    fig.suptitle('BS-Detector Performance vs. Carmen Ground Truth  (n=40 cells)',
                 color=TEXT, fontsize=13, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.38)

    # ── 1. Scar distance CDF ──────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, :2])
    sorted_d = np.sort(scar_dists)
    cdf = np.arange(1, len(sorted_d)+1) / len(sorted_d)
    ax.plot(sorted_d, cdf * 100, color=ACCENT, linewidth=2)
    ax.fill_between(sorted_d, cdf * 100, alpha=0.15, color=ACCENT)

    # threshold lines
    for thresh, label in [(15, '15 px'), (25, '25 px'), (40, '40 px')]:
        pct = np.interp(thresh, sorted_d, cdf * 100)
        ax.axvline(thresh, color=GRID, linewidth=1, linestyle='--')
        ax.text(thresh + 0.5, 5, f'{pct:.0f}%\n@ {label}',
                color=TEXT, fontsize=7.5, va='bottom')

    ax.set_xlabel('Scar localization error (px)')
    ax.set_ylabel('Cumulative % of cells')
    ax.set_title('Scar Location Error — CDF', color=TEXT, fontweight='bold')
    ax.set_xlim(0, max(sorted_d) * 1.05)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0f}%'))

    # ── 2. Scar distance histogram ────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    bins = np.arange(0, max(scar_dists) + 10, 8)
    ax2.hist(scar_dists, bins=bins, color=ACCENT, edgecolor=BG, linewidth=0.5)
    ax2.axvline(np.median(scar_dists), color=GT_COL, linewidth=1.5,
                linestyle='--', label=f'Median {np.median(scar_dists):.1f} px')
    ax2.axvline(np.mean(scar_dists), color=ORANGE, linewidth=1.5,
                linestyle=':', label=f'Mean {np.mean(scar_dists):.1f} px')
    ax2.set_xlabel('Scar error (px)')
    ax2.set_ylabel('Cell count')
    ax2.set_title('Scar Error Distribution', color=TEXT, fontweight='bold')
    ax2.legend(fontsize=7.5, framealpha=0.3)

    # ── 3. Pole identity pie ──────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 3])
    n_total = len(pole_agree)
    n_wrong = n_total - n_agree
    wedges, texts, autotexts = ax3.pie(
        [n_agree, n_wrong],
        labels=['Correct', 'Wrong'],
        autopct='%1.0f%%',
        colors=[GREEN, ERR_COL],
        startangle=90,
        wedgeprops={'edgecolor': BG, 'linewidth': 2},
        textprops={'color': TEXT, 'fontsize': 9},
    )
    for at in autotexts:
        at.set_color(BG)
        at.set_fontweight('bold')
    ax3.set_title('New/Old Pole\nIdentity Agreement', color=TEXT, fontweight='bold')

    # ── 4. Area Bland-Altman ──────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, :2])
    area_mean = [(g + (g + e)) / 2 for g, e in zip(area_gt, area_errs)]
    ax4.scatter(area_mean, area_errs, color=ACCENT, s=30, alpha=0.8, zorder=3)
    mean_bias = np.mean(area_errs)
    sd        = np.std(area_errs)
    ax4.axhline(mean_bias,           color=GT_COL, linewidth=1.5,
                linestyle='--', label=f'Bias {mean_bias:+.0f} px²')
    ax4.axhline(mean_bias + 1.96*sd, color=ERR_COL, linewidth=1,
                linestyle=':', label=f'+1.96σ {mean_bias+1.96*sd:+.0f}')
    ax4.axhline(mean_bias - 1.96*sd, color=ERR_COL, linewidth=1,
                linestyle=':', label=f'−1.96σ {mean_bias-1.96*sd:+.0f}')
    ax4.axhline(0, color=GRID, linewidth=0.8)
    ax4.set_xlabel('Mean of Detector + GT area (px²)')
    ax4.set_ylabel('Detector − GT area (px²)')
    ax4.set_title('Cell Area Bland-Altman  (Detector always overestimates)', color=TEXT, fontweight='bold')
    ax4.legend(fontsize=7.5, framealpha=0.3)

    # ── 5. Compartment length scatter ────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    all_gt  = poleA_gt  + poleB_gt
    all_det = [g + e for g, e in zip(poleA_gt, poleA_errs)] + \
              [g + e for g, e in zip(poleB_gt, poleB_errs)]
    lim = max(max(all_gt), max(all_det)) * 1.05
    ax5.scatter(poleA_gt, [g + e for g, e in zip(poleA_gt, poleA_errs)],
                color=ACCENT, s=24, alpha=0.8, label='Pole A')
    ax5.scatter(poleB_gt, [g + e for g, e in zip(poleB_gt, poleB_errs)],
                color=ORANGE, s=24, alpha=0.8, marker='^', label='Pole B')
    ax5.plot([0, lim], [0, lim], color=GRID, linewidth=0.8, linestyle='--')
    ax5.set_xlabel('GT compartment length (px)')
    ax5.set_ylabel('Detector compartment length (px)')
    ax5.set_title('Compartment Lengths', color=TEXT, fontweight='bold')
    ax5.legend(fontsize=7.5, framealpha=0.3)
    ax5.set_xlim(0, lim); ax5.set_ylim(0, lim)

    # ── 6. Per-pole error bar chart ───────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 3])
    labels = ['Pole A', 'Pole B']
    maes   = [np.mean(np.abs(poleA_errs)), np.mean(np.abs(poleB_errs))]
    biases = [np.mean(poleA_errs), np.mean(poleB_errs)]
    x = np.array([0, 1])
    ax6.bar(x - 0.18, maes,   width=0.32, color=ACCENT,  label='MAE',  zorder=3)
    ax6.bar(x + 0.18, biases, width=0.32, color=ORANGE,  label='Bias', zorder=3)
    ax6.axhline(0, color=GRID, linewidth=0.8)
    ax6.set_xticks(x); ax6.set_xticklabels(labels)
    ax6.set_ylabel('Pixels')
    ax6.set_title('Compartment Length Error', color=TEXT, fontweight='bold')
    ax6.legend(fontsize=7.5, framealpha=0.3)

    return fig


# ── scar overlay examples ────────────────────────────────────────────────────

def crop_around(img_arr, row, col, half=120):
    r0 = max(0, int(row) - half)
    r1 = min(img_arr.shape[0], int(row) + half)
    c0 = max(0, int(col) - half)
    c1 = min(img_arr.shape[1], int(col) + half)
    return img_arr[r0:r1, c0:c1], r0, c0


def plot_scar_overlays(comparison_rows, gt_lookup, det_lookup, raw_frame_dir, run_dirs):
    """Show 6 example cells ordered by scar error: 2 good, 2 median, 2 bad."""
    rows_with_scar = [r for r in comparison_rows if r.get('scar_dist_px')]
    rows_with_scar.sort(key=lambda r: float(r['scar_dist_px']))
    n = len(rows_with_scar)

    # pick indices: 2 from bottom third, 2 from middle, 2 from top
    def pick(start_frac, end_frac, count=2):
        lo = int(start_frac * n)
        hi = int(end_frac * n)
        idxs = np.linspace(lo, hi - 1, count, dtype=int)
        return [rows_with_scar[i] for i in idxs]

    examples = pick(0, 0.3) + pick(0.35, 0.65) + pick(0.7, 1.0)
    tier_labels = ['Good', 'Good', 'Median', 'Median', 'Bad', 'Bad']

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), facecolor=BG)
    fig.suptitle('Scar Localization Examples: Detector (blue ×) vs. Carmen GT (gold ●)',
                 color=TEXT, fontsize=12, fontweight='bold', y=0.99)

    for ax, row, tier in zip(axes.flat, examples, tier_labels):
        video_id  = row['video_id']
        frame     = int(row['frame'])
        cell_name = row['cell_name']
        scar_err  = float(row['scar_dist_px'])
        key       = (video_id, frame, cell_name)

        # locate raw frame
        raw_name = f"{video_id}_frame{frame:04d}_raw.tif"
        raw_path = os.path.join(raw_frame_dir, raw_name)
        if not os.path.exists(raw_path):
            # fall back: find it in run dirs
            raw_path = None
            for rd in run_dirs:
                candidate = os.path.join(rd, raw_name)
                if os.path.exists(candidate):
                    raw_path = candidate
                    break

        if raw_path is None or not os.path.exists(raw_path):
            ax.text(0.5, 0.5, 'frame not found', ha='center', va='center',
                    transform=ax.transAxes, color=ERR_COL)
            ax.axis('off')
            continue

        raw = tifffile.imread(raw_path).astype(np.float32)
        # robust contrast stretch
        lo, hi = np.percentile(raw, (1, 99.5))
        raw_disp = np.clip((raw - lo) / (hi - lo + 1e-9), 0, 1)

        # GT scar position (x=col, y=row in FIJI convention)
        gt_row = gt_lookup.get(key, {})
        gt_x   = _to_float(gt_row.get('gt_scar_mid_x_px'))  # col
        gt_y   = _to_float(gt_row.get('gt_scar_mid_y_px'))  # row

        # Detector scar position: scar_midpoint is [row, col]
        det_row_data = det_lookup.get(key, {})
        det_scar = _parse_point(det_row_data.get('scar_midpoint'))
        det_r, det_c = (det_scar if det_scar else (gt_y, gt_x))

        # crop centred on midpoint between the two markers
        if gt_y is not None and gt_x is not None:
            cen_r = (gt_y + det_r) / 2
            cen_c = (gt_x + det_c) / 2
        else:
            cen_r, cen_c = det_r, det_c

        half = 130
        crop, r0, c0 = crop_around(raw_disp, cen_r, cen_c, half=half)

        ax.imshow(crop, cmap='gray', vmin=0, vmax=1, interpolation='bilinear')

        # detector scar (blue X)
        ax.scatter([det_c - c0], [det_r - r0],
                   marker='x', s=160, linewidths=2.5, color=ACCENT, zorder=5,
                   label='Detector')
        # GT scar (gold dot)
        if gt_x is not None and gt_y is not None:
            ax.scatter([gt_x - c0], [gt_y - r0],
                       marker='o', s=80, color=GT_COL, zorder=6,
                       edgecolors='white', linewidths=0.8, label='Carmen GT')
            # error line
            ax.plot([det_c - c0, gt_x - c0], [det_r - r0, gt_y - r0],
                    color=ERR_COL, linewidth=1.2, linestyle='--', alpha=0.7, zorder=4)

        tier_color = GREEN if tier == 'Good' else (GT_COL if tier == 'Median' else ERR_COL)
        notes = gt_row.get('notes', '')
        note_str = f'  [{notes}]' if notes else ''
        ax.set_title(
            f'[{tier}]  {cell_name}  fr{frame}  |  err={scar_err:.1f} px{note_str}',
            color=tier_color, fontsize=8.5, fontweight='bold', pad=4
        )
        ax.axis('off')

    # shared legend
    handles = [
        mpatches.Patch(color=ACCENT,   label='Detector scar (×)'),
        mpatches.Patch(color=GT_COL,   label='Carmen GT scar (●)'),
        mpatches.Patch(color=ERR_COL,  label='Error line'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3,
               framealpha=0.3, fontsize=9,
               bbox_to_anchor=(0.5, 0.01))

    fig.tight_layout(rect=[0, 0.05, 1, 0.98])
    return fig


# ── HTML report ───────────────────────────────────────────────────────────────

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>BS-Detector Validation Report</title>
<style>
  :root {{
    --bg: #0f1117; --panel: #1a1d27; --text: #e8eaf0;
    --accent: #4f8ef7; --gt: #f7c44f; --err: #e05c5c;
    --green: #4ecb8d; --grid: #2a2d3a;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: -apple-system, sans-serif;
          font-size: 14px; padding: 24px; }}
  h1 {{ font-size: 22px; font-weight: 700; margin-bottom: 6px; color: var(--accent); }}
  .sub {{ color: #888; font-size: 12px; margin-bottom: 28px; }}
  h2 {{ font-size: 15px; font-weight: 600; color: var(--text); margin: 28px 0 10px; }}
  .stat-row {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 24px; }}
  .stat {{ background: var(--panel); border-radius: 10px; padding: 14px 20px;
           flex: 1; min-width: 160px; border-top: 3px solid var(--accent); }}
  .stat .val {{ font-size: 26px; font-weight: 700; color: var(--accent); }}
  .stat .lbl {{ font-size: 11px; color: #888; margin-top: 2px; }}
  .stat.good  .val {{ color: var(--green); }}
  .stat.warn  .val {{ color: var(--gt); }}
  .stat.bad   .val {{ color: var(--err); }}
  .cutoff-tool {{ background: var(--panel); border-radius: 10px; padding: 20px 24px;
                  margin: 20px 0; border-left: 4px solid var(--accent); }}
  .cutoff-tool h3 {{ font-size: 14px; margin-bottom: 12px; color: var(--accent); }}
  .slider-row {{ display: flex; align-items: center; gap: 14px; margin-bottom: 8px; }}
  input[type=range] {{ flex: 1; accent-color: var(--accent); }}
  .result-badge {{ background: var(--accent); color: #fff; border-radius: 6px;
                   padding: 4px 14px; font-weight: 700; font-size: 15px; white-space: nowrap; }}
  img.plot {{ width: 100%; border-radius: 10px; margin-bottom: 20px;
              border: 1px solid var(--grid); }}
  .notes-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
  .note-card {{ background: var(--panel); border-radius: 8px; padding: 12px 16px;
                border-left: 3px solid var(--grid); font-size: 12px; }}
  .note-card.bad  {{ border-color: var(--err); }}
  .note-card.good {{ border-color: var(--green); }}
  .note-card b {{ font-size: 13px; color: var(--text); }}
</style>
</head>
<body>
<h1>BS-Detector Validation Report</h1>
<p class="sub">Ground truth: Carmen (cr), 40 cells, 4 videos &nbsp;|&nbsp; 2025-03 experiments</p>

<div class="stat-row">
  <div class="stat good">
    <div class="val">92.5%</div>
    <div class="lbl">New/old pole identity agreement</div>
  </div>
  <div class="stat warn">
    <div class="val">{med_scar:.1f} px</div>
    <div class="lbl">Median scar localization error</div>
  </div>
  <div class="stat warn">
    <div class="val">{mae_scar:.1f} px</div>
    <div class="lbl">Mean scar localization error (MAE)</div>
  </div>
  <div class="stat warn">
    <div class="val">{mae_length:.1f} px</div>
    <div class="lbl">Mean compartment length MAE (pooled)</div>
  </div>
  <div class="stat bad">
    <div class="val">+{area_bias:.0f} px²</div>
    <div class="lbl">Area bias (always overestimates ~{area_pct:.0f}%)</div>
  </div>
</div>

<div class="cutoff-tool">
  <h3>Scar Detection Threshold Explorer</h3>
  <div class="slider-row">
    <span style="min-width:70px">Threshold:</span>
    <input type="range" id="thresh" min="5" max="100" value="25" step="1"
           oninput="updateBadge()">
    <span id="thresh-lbl" style="min-width:60px;font-weight:700;color:var(--accent)">25 px</span>
    <div class="result-badge" id="badge">??%  of cells</div>
  </div>
  <div style="color:#888;font-size:11px">Move the slider to set a pixel tolerance. The badge shows what fraction of the 40 cells had their scar detected within that distance of Carmen's label.</div>
</div>

<h2>Performance Metrics Dashboard</h2>
<img class="plot" src="data:image/png;base64,{metrics_b64}" alt="metrics dashboard">

<h2>Scar Localization Examples</h2>
<img class="plot" src="data:image/png;base64,{overlay_b64}" alt="scar overlays">

<h2>Notable cells</h2>
<div class="notes-grid">
  <div class="note-card bad">
    <b>Worst: xy3 fr118 F1 (89.8 px)</b><br>
    "horizontal cell" — curvature detector struggles when the long axis is nearly horizontal relative to image orientation. Scar ridges on top/bottom edges rather than left/right.
  </div>
  <div class="note-card bad">
    <b>xy2 fr151 H0 (67.4 px) — pole flip</b><br>
    One of the 3 pole identity errors. Scar found ~68 px off Carmen's placement, and the detector inverted which end is new vs. old.
  </div>
  <div class="note-card bad">
    <b>xy3 fr174 E0 (70.4 px)</b><br>
    Large scar error + Pole B overestimated by 74 px. Suggests the scar midpoint was placed near the wrong curvature extremum.
  </div>
  <div class="note-card good">
    <b>Typical good case: ~3–18 px error</b><br>
    23 of 40 cells (57%) are within 25 px. The bulk of the distribution is tight — the tail comes from a small number of challenging cells (horizontal, faint BS, ring-maturing).
  </div>
</div>

<script>
const dists = {dists_json};
function updateBadge() {{
  const t = parseInt(document.getElementById('thresh').value);
  document.getElementById('thresh-lbl').textContent = t + ' px';
  const pct = (dists.filter(d => d <= t).length / dists.length * 100).toFixed(0);
  document.getElementById('badge').textContent = pct + '% of cells within ' + t + ' px';
}}
updateBadge();
</script>
</body>
</html>
"""


def main():
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument('--comparison', required=True)
    parser.add_argument('--gt-csv', required=True)
    parser.add_argument('--raw-frames', required=True,
                        help='directory containing raw TIF frames for the package')
    parser.add_argument('--det-csv', nargs='+', required=True)
    parser.add_argument('--run-dirs', nargs='*', default=[],
                        help='extra dirs to search for raw frames (run measure dirs)')
    parser.add_argument('--out', default='.')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    comp_rows = load_comparison(args.comparison)
    gt_lookup  = load_gt(args.gt_csv)
    det_lookup = load_det_csvs(args.det_csv)

    # ── metrics dashboard ─────────────────────────────────────────────────
    print('Rendering metrics dashboard...')
    fig_m = plot_metrics_dashboard(comp_rows)
    fig_m.savefig(os.path.join(args.out, 'metrics_dashboard.png'),
                  dpi=150, bbox_inches='tight', facecolor=BG)
    metrics_b64 = fig_to_b64(fig_m)
    plt.close(fig_m)

    # ── scar overlays ─────────────────────────────────────────────────────
    print('Rendering scar overlays...')
    fig_o = plot_scar_overlays(comp_rows, gt_lookup, det_lookup,
                               args.raw_frames, args.run_dirs)
    fig_o.savefig(os.path.join(args.out, 'scar_overlays.png'),
                  dpi=150, bbox_inches='tight', facecolor=BG)
    overlay_b64 = fig_to_b64(fig_o)
    plt.close(fig_o)

    # ── stats for HTML ────────────────────────────────────────────────────
    scar_dists = [float(r['scar_dist_px']) for r in comp_rows if r.get('scar_dist_px')]
    area_errs  = [float(r['area_error'])   for r in comp_rows if r.get('area_error')]
    area_gts   = [float(r['area_gt'])      for r in comp_rows if r.get('area_gt')]
    pA = [float(r['poleA_error']) for r in comp_rows if r.get('poleA_error')]
    pB = [float(r['poleB_error']) for r in comp_rows if r.get('poleB_error')]
    mae_length  = np.mean(np.abs(pA + pB))
    area_bias   = np.mean(area_errs)
    area_pct    = area_bias / np.mean(area_gts) * 100

    html = HTML_TEMPLATE.format(
        med_scar   = np.median(scar_dists),
        mae_scar   = np.mean(scar_dists),
        mae_length = mae_length,
        area_bias  = area_bias,
        area_pct   = area_pct,
        metrics_b64= metrics_b64,
        overlay_b64= overlay_b64,
        dists_json = json.dumps(scar_dists),
    )

    html_path = os.path.join(args.out, 'results.html')
    with open(html_path, 'w') as f:
        f.write(html)
    print(f'Done. HTML report: {html_path}')


if __name__ == '__main__':
    main()
