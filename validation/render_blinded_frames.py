"""
render_blinded_frames.py  –  Generate blinded reference images for manual
ground-truth collection.

Runs the normal BS-Detector pipeline (segmentation + tracking) so every cell
gets a stable `cell_name`, but renders only the cell outline and its ID label
— no birth scar, no pole markers, no scar_detected color coding. This keeps
the wet-lab scorer blind to the algorithm's own answer while still letting
them identify "which cell is which" against BS-Detector's output.

Usage
-----
    python -m validation.render_blinded_frames /path/to/experiment.h5 out_dir/

Also exports a companion CSV with the extra pixel-coordinate columns
(scar_midpoint, new_pole_point, old_pole_point) needed by
compare_to_ground_truth.py, in addition to the normal measurement columns.
"""
import os
import sys

import matplotlib.pyplot as plt

from pombe_tracker.config import Config
from pombe_tracker.io_utils import load_h5_data, export_csv
from pombe_tracker.pipeline import run_pipeline
from pombe_tracker.tracking import CellTracker
from pombe_tracker.postprocessing import stabilize_scars


def plot_blinded_overview(frame, results, frame_idx):
    """Cell outline + cell_name label only — no scar/pole information."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(frame, cmap='gray')
    ax.set_title(f'Frame {frame_idx}  –  {len(results)} cells (blinded)',
                 fontsize=13, fontweight='bold')
    ax.axis('off')

    for r in results:
        contour = r['contour']
        name = r.get('cell_name', str(r['label']))
        cx, cy = contour[:, 1].mean(), contour[:, 0].mean()

        ax.plot(contour[:, 1], contour[:, 0], color='deepskyblue', lw=1.5, alpha=0.9)
        ax.text(cx, cy, name, color='white', fontsize=8, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', fc='navy', alpha=0.7, lw=0))

    fig.tight_layout()
    return fig


def main(h5_path, out_dir, num_frames=None, video_id=None):
    cfg = Config()
    cfg.H5_FILE_PATH = h5_path
    cfg.NUM_FRAMES = num_frames

    # cell_name is only unique *within* one video's tracker. Once results from
    # multiple videos are pooled for sampling, (frame, cell_name) can collide
    # across videos — video_id disambiguates them everywhere downstream.
    if video_id is None:
        video_id = os.path.splitext(os.path.basename(h5_path))[0]

    os.makedirs(out_dir, exist_ok=True)

    frames = load_h5_data(cfg.H5_FILE_PATH, cfg.H5_DATASET_KEY)
    if num_frames:
        frames = frames[:num_frames]

    tracker = CellTracker(cfg)
    results = run_pipeline(frames, cfg, tracker=tracker)
    results, _report = stabilize_scars(results, cfg)

    for fd in results:
        for r in fd['cells']:
            r['video_id'] = video_id

        fig = plot_blinded_overview(fd['frame'], fd['cells'], fd['frame_idx'])
        path = os.path.join(out_dir, f"{video_id}_frame{fd['frame_idx']:04d}_blinded.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {path}")

    # Export the standard measurement columns plus the extra pixel coordinates
    # (scar_midpoint, new_pole_point, old_pole_point) needed for comparing
    # birth-scar location and for resolving pole A/B against new/old identity,
    # plus video_id to disambiguate cell_name across multiple videos.
    extra_cols = ['video_id'] + cfg.CSV_COLUMNS + ['scar_midpoint', 'new_pole_point', 'old_pole_point']
    csv_path = os.path.join(out_dir, f'{video_id}_bs_detector_measurements.csv')
    export_csv(results, csv_path, columns=extra_cols)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python -m validation.render_blinded_frames <h5_path> <out_dir> [num_frames]")
        sys.exit(1)

    h5_arg = sys.argv[1]
    out_arg = sys.argv[2]
    n_arg = int(sys.argv[3]) if len(sys.argv) > 3 else None
    main(h5_arg, out_arg, n_arg)
