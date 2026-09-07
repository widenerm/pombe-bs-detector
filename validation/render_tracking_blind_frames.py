"""
render_tracking_blind_frames.py  –  Generate tracking-blind reference images
for manual lineage / division validation.

Runs the *full* pipeline from frame 0 (so the tracker's real behavior —
ghost-track matching, division detection, everything — is exercised exactly
as it would be in a real analysis), then renders one contiguous *run* of
frames with cells labeled by a **frame-local index**: renumbered fresh every
single frame (sorted top-to-bottom, left-to-right by centroid), instead of
BS-Detector's persistent `cell_name`.

This is deliberately blind to the algorithm's own tracking identity: a
wet-lab scorer has to independently decide, frame to frame, which numbered
blob is the same physical cell — and where real divisions happen — without
being able to just read off whether a label persisted. That's the point:
showing them the real cell_name would leak the answer to the exact question
being validated.

The frame-local-id -> real cell_name mapping is written to a `*_KEY_do_not_share.csv`
file. That file is for compare_tracking_to_ground_truth.py only — never send
it to the scorer.

A run should be a short, contiguous span (roughly 10-20 frames) so continuous
manual tracking is actually tractable. Sample a few runs from different
points in the movie rather than one giant span, so the resulting error rates
aren't biased toward whatever part of the movie you happened to pick.

Usage
-----
    python -m validation.render_tracking_blind_frames \\
        /path/to/experiment.h5 out_dir/ run0 10 25
"""
import csv
import os
import sys

import matplotlib.pyplot as plt

from pombe_tracker.config import Config
from pombe_tracker.io_utils import load_h5_data
from pombe_tracker.pipeline import run_pipeline
from pombe_tracker.tracking import CellTracker
from pombe_tracker.postprocessing import stabilize_scars


def plot_tracking_blind(frame, results, frame_idx):
    """Cell outline + a frame-local index (NOT cell_name), renumbered every frame."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(frame, cmap='gray')
    ax.set_title(f'Frame {frame_idx}  –  {len(results)} cells (tracking-blind)',
                 fontsize=13, fontweight='bold')
    ax.axis('off')

    ordered = sorted(results, key=lambda r: (r['centroid'][0], r['centroid'][1]))
    local_id_map = {}

    for i, r in enumerate(ordered, start=1):
        contour = r['contour']
        cx, cy = contour[:, 1].mean(), contour[:, 0].mean()
        ax.plot(contour[:, 1], contour[:, 0], color='deepskyblue', lw=1.5, alpha=0.9)
        ax.text(cx, cy, str(i), color='white', fontsize=9, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', fc='navy', alpha=0.7, lw=0))
        local_id_map[i] = r.get('cell_name', str(r['label']))

    fig.tight_layout()
    return fig, local_id_map


def main(h5_path, out_dir, run_id, frame_start, frame_end, num_frames=None):
    cfg = Config()
    cfg.H5_FILE_PATH = h5_path
    cfg.NUM_FRAMES = num_frames

    os.makedirs(out_dir, exist_ok=True)

    frames = load_h5_data(cfg.H5_FILE_PATH, cfg.H5_DATASET_KEY)
    if num_frames:
        frames = frames[:num_frames]

    tracker = CellTracker(cfg)
    results = run_pipeline(frames, cfg, tracker=tracker)
    results, _report = stabilize_scars(results, cfg)

    key_rows = []
    for fd in results:
        fidx = fd['frame_idx']
        if fidx < frame_start or fidx > frame_end:
            continue
        fig, local_id_map = plot_tracking_blind(fd['frame'], fd['cells'], fidx)
        path = os.path.join(out_dir, f"{run_id}_frame{fidx:04d}_trackingblind.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {path}")
        for local_id, cell_name in local_id_map.items():
            key_rows.append({'run_id': run_id, 'frame': fidx,
                              'frame_local_id': local_id, 'cell_name': cell_name})

    key_path = os.path.join(out_dir, f'{run_id}_KEY_do_not_share.csv')
    with open(key_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['run_id', 'frame', 'frame_local_id', 'cell_name'])
        writer.writeheader()
        writer.writerows(key_rows)
    print(f"Key file (internal use only — do NOT send to scorer): {key_path}")

    # Full lineage log (not restricted to the run) so
    # compare_tracking_to_ground_truth.py can check every division
    # BS-Detector reported, regardless of which run it falls in.
    lineage_path = os.path.join(out_dir, 'lineage_log.csv')
    with open(lineage_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['frame', 'parent', 'daughter0', 'daughter1'])
        writer.writeheader()
        for ev in tracker.lineage_log:
            writer.writerow({
                'frame': ev['frame'],
                'parent': ev['parent'],
                'daughter0': ev['daughters'][0],
                'daughter1': ev['daughters'][1],
            })
    print(f"Lineage log saved: {lineage_path}")


if __name__ == '__main__':
    if len(sys.argv) < 6:
        print("Usage: python -m validation.render_tracking_blind_frames "
              "<h5_path> <out_dir> <run_id> <frame_start> <frame_end> [num_frames]")
        sys.exit(1)

    h5_arg = sys.argv[1]
    out_arg = sys.argv[2]
    run_arg = sys.argv[3]
    fs_arg = int(sys.argv[4])
    fe_arg = int(sys.argv[5])
    n_arg = int(sys.argv[6]) if len(sys.argv) > 6 else None
    main(h5_arg, out_arg, run_arg, fs_arg, fe_arg, n_arg)
