"""
sample_for_scientists.py  –  Randomly sample cells from one or more completed
BS-Detector runs and assemble a ready-to-send package for manual FIJI
ground-truth collection.

Rationale: BS-Detector should always be run on complete, contiguous frame
sequences (tracking, ghost-track matching, and scar-consensus stabilization
all depend on full temporal context — subsampling frames beforehand would
mean testing a different, weaker mode of the algorithm than what's actually
used). So the right order is: process every frame of every video in full,
then randomly sample cells/frames from the complete results afterward for
whatever you send to scientists.

This script does that sampling step: it pools every `*_bs_detector_measurements.csv`
found in the given run directories, keeps only seg_quality == 'ok' rows,
randomly samples N of them, and assembles two things scientists need:

  - raw_frames/  : the ACTUAL frame each sampled cell came from, exported at
                   full native resolution/bit-depth straight from the source
                   .h5 (16-bit TIFF). This is what gets opened and measured
                   in FIJI. Pixel-for-pixel identical to what BS-Detector
                   itself measured on.
  - images/      : the blinded reference PNGs (cell outline + ID only) used
                   purely to identify *which* cell is which. These are
                   matplotlib renders at a DIFFERENT resolution than the raw
                   data (annotated, resampled to the figure's dpi) — do not
                   measure on them, they will not be to the same pixel scale.

ground_truth_template.csv ties them together, with video_id / raw_frame_file
/ image_file / frame / cell_name already filled in.

Making more packages for additional scientists
------------------------------------------------
Re-run this script again with the same --run arguments. By default a fresh
random sample is drawn each time, which will mostly (but not guaranteed to)
avoid repeats. For an explicit guarantee, pick one:

  --exclude prior_package/ground_truth_template.csv [...]
      Draws a fresh sample guaranteed to contain NONE of the cells in the
      given prior package(s) — use this to maximize total distinct coverage
      across scientists.

  --include-only prior_package/ground_truth_template.csv
      Reproduces the EXACT same cells as a prior package (n_samples is
      ignored) — use this to give a second scientist the same cells as a
      first, so you can measure inter-rater agreement (how much scientists
      disagree with each other, which sets a noise floor for interpreting
      BS-Detector's error).

Usage
-----
    python -m validation.sample_for_scientists out_dir/ 40 \\
        --run validation/runs/xy1/measure data/experiments/xy1.h5 \\
        --run validation/runs/xy2/measure data/experiments/xy2.h5 \\
        [--exclude other_package/ground_truth_template.csv]
        [--include-only other_package/ground_truth_template.csv]
"""
import argparse
import csv
import glob
import os
import random

import h5py
import tifffile

from pombe_tracker.config import Config

TEMPLATE_FIELDS = [
    'video_id', 'raw_frame_file', 'image_file', 'frame', 'cell_name', 'scorer_initials',
    'gt_area_px2', 'gt_scar_mid_x_px', 'gt_scar_mid_y_px',
    'gt_poleA_length_px', 'gt_poleB_length_px', 'gt_new_pole_guess', 'notes',
]


def load_run_rows(run_dir, h5_path):
    matches = glob.glob(os.path.join(run_dir, '*_bs_detector_measurements.csv'))
    if not matches:
        print(f"  Warning: no *_bs_detector_measurements.csv found in {run_dir}, skipping")
        return []
    rows = []
    with open(matches[0], newline='') as f:
        for row in csv.DictReader(f):
            row['_run_dir'] = run_dir
            row['_h5_path'] = h5_path
            rows.append(row)
    return rows


def export_raw_frames(sampled, raw_dir):
    """Pull the exact native-resolution frame for each (h5_path, frame) needed
    directly from the source .h5, so FIJI measurements are pixel-for-pixel
    comparable to BS-Detector's own output."""
    needed = {}
    for r in sampled:
        key = (r['_h5_path'], int(r['frame']))
        needed.setdefault(key, []).append(r)

    by_h5 = {}
    for (h5_path, frame_idx) in needed:
        by_h5.setdefault(h5_path, set()).add(frame_idx)

    written = {}
    for h5_path, frame_indices in by_h5.items():
        with h5py.File(h5_path, 'r') as f:
            dataset = f[Config.H5_DATASET_KEY]
            for frame_idx in sorted(frame_indices):
                raw = dataset[frame_idx]
                video_id = None
                for r in needed[(h5_path, frame_idx)]:
                    video_id = r.get('video_id', '')
                    break
                name = f"{video_id}_frame{frame_idx:04d}_raw.tif"
                path = os.path.join(raw_dir, name)
                tifffile.imwrite(path, raw)
                written[(h5_path, frame_idx)] = name
                print(f"  Saved {path}")

    return written


def _keys_from_csv(path):
    keys = set()
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            if not row.get('cell_name', '').strip():
                continue
            keys.add((row.get('video_id', ''), int(row['frame']), row['cell_name']))
    return keys


def main(out_dir, n_samples, runs, exclude_paths=None, include_only_path=None):
    """runs: list of (run_dir, h5_path) pairs.

    exclude_paths     : prior ground_truth_template.csv(s) whose cells should
                        be excluded — use this to give a second scientist a
                        package with NO overlap, maximizing total coverage.
    include_only_path : a prior ground_truth_template.csv whose EXACT cells
                        should be reproduced (n_samples is ignored) — use
                        this to give a second scientist the SAME cells as an
                        earlier package, to measure inter-rater agreement.
    """
    all_rows = []
    for run_dir, h5_path in runs:
        all_rows.extend(load_run_rows(run_dir, h5_path))

    eligible = [r for r in all_rows if r.get('seg_quality') == 'ok']
    print(f"Pooled {len(all_rows)} row(s) across {len(runs)} run(s); "
          f"{len(eligible)} eligible (seg_quality == 'ok')")

    if include_only_path:
        keep = _keys_from_csv(include_only_path)
        sampled = [r for r in eligible
                   if (r.get('video_id', ''), int(r['frame']), r['cell_name']) in keep]
        found = {(r.get('video_id', ''), int(r['frame']), r['cell_name']) for r in sampled}
        missing = keep - found
        print(f"  --include-only: reproducing {len(sampled)}/{len(keep)} cell(s) from {include_only_path}")
        if missing:
            print(f"  {len(missing)} cell(s) from that file were not found in this pool "
                  f"(different run set?): {sorted(missing)[:10]}")
    else:
        if exclude_paths:
            exclude_keys = set()
            for path in exclude_paths:
                exclude_keys |= _keys_from_csv(path)
            before = len(eligible)
            eligible = [r for r in eligible
                        if (r.get('video_id', ''), int(r['frame']), r['cell_name']) not in exclude_keys]
            print(f"  --exclude: removed {before - len(eligible)} already-sampled cell(s) "
                  f"({len(eligible)} remain eligible)")

        if n_samples > len(eligible):
            print(f"  Requested {n_samples} but only {len(eligible)} eligible — "
                  f"sampling all of them")
            n_samples = len(eligible)

        sampled = random.sample(eligible, n_samples)

    sampled.sort(key=lambda r: (r.get('video_id', ''), int(r['frame']), r['cell_name']))

    images_dir = os.path.join(out_dir, 'images')
    raw_dir = os.path.join(out_dir, 'raw_frames')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    raw_frame_names = export_raw_frames(sampled, raw_dir)

    template_rows = []
    missing_images = []

    for r in sampled:
        video_id = r.get('video_id', '')
        frame = int(r['frame'])
        image_name = f"{video_id}_frame{frame:04d}_blinded.png"
        src = os.path.join(r['_run_dir'], image_name)
        dst = os.path.join(images_dir, image_name)

        if not os.path.exists(dst):
            if os.path.exists(src):
                import shutil
                shutil.copy2(src, dst)
            else:
                missing_images.append(src)

        raw_frame_name = raw_frame_names.get((r['_h5_path'], frame), '')

        template_rows.append({
            'video_id': video_id,
            'raw_frame_file': raw_frame_name,
            'image_file': image_name,
            'frame': frame,
            'cell_name': r['cell_name'],
            'scorer_initials': '',
            'gt_area_px2': '', 'gt_scar_mid_x_px': '', 'gt_scar_mid_y_px': '',
            'gt_poleA_length_px': '', 'gt_poleB_length_px': '',
            'gt_new_pole_guess': '', 'notes': '',
        })

    template_path = os.path.join(out_dir, 'ground_truth_template.csv')
    with open(template_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=TEMPLATE_FIELDS)
        writer.writeheader()
        writer.writerows(template_rows)

    print(f"\nSampled {len(sampled)} cell(s) from {len({r.get('video_id') for r in sampled})} video(s)")
    print(f"Raw measurement frames: {raw_dir}")
    print(f"Blinded ID reference images: {images_dir}")
    print(f"Pre-filled template: {template_path}")
    if missing_images:
        print(f"\n  {len(missing_images)} blinded image(s) referenced but not found on disk "
              f"(re-render that run?):")
        for p in missing_images:
            print(f"    {p}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('out_dir')
    parser.add_argument('n_samples', type=int)
    parser.add_argument('--run', nargs=2, action='append', required=True,
                        metavar=('RUN_DIR', 'H5_PATH'),
                        help='a completed run directory and the .h5 file it was generated from; repeatable')
    parser.add_argument('--exclude', action='append', default=[], metavar='PRIOR_TEMPLATE_CSV',
                        help='exclude cells already sampled into a prior package; repeatable')
    parser.add_argument('--include-only', metavar='PRIOR_TEMPLATE_CSV',
                        help='reproduce the exact cells from a prior package (ignores n_samples)')
    args = parser.parse_args()

    main(args.out_dir, args.n_samples, args.run,
         exclude_paths=args.exclude, include_only_path=args.include_only)
