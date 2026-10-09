"""Build frozen per-cell records for fast scar-detector tuning.

Each annotated cell from Carmen's packages is located in a cached full-frame
Cellpose mask. Identity uses the historical pole-midpoint anchor and must agree
with the label containing the annotated scar; disagreements are kept and
flagged, never silently resolved. The record stores everything the detector
sees in production (contour, frame crop, neighbor-derived pole hint) plus the
ground truth, so detector variants can be scored without re-segmenting.

    python -m validation.scar_cells --cache validation/runs/scar_v2/cache_cuda \
        --out validation/runs/scar_v3/cells_cuda.pkl
"""
import argparse
import csv
import glob
import hashlib
import json
import pickle
from pathlib import Path

import h5py
import numpy as np
from skimage.measure import find_contours, regionprops

from pombe_tracker.config import Config
from pombe_tracker.geometry import compute_pca_axis, compute_smoothed_curvature, get_contour_endpoints
from pombe_tracker.pipeline import filter_valid_cells, prepare_cell_info
from pombe_tracker.poles import determine_new_pole_from_neighbors, find_pole_to_pole_neighbors

REPO = Path(__file__).resolve().parents[1]
PACKAGES = {
    'cr_v1': (REPO / 'validation/h5_for_drive/carmen_package_1',
              REPO / 'validation/ground_truth_template_cr-v1.csv'),
    'cr_v2': (REPO / 'validation/h5_for_drive/carmen_package_2',
              REPO / 'validation/package_for_carmen_2/ground_truth_template_cr_v2.csv'),
}
CROP_MARGIN = 40


def _point(text):
    return np.array(json.loads(text), dtype=float)


def historical_anchors():
    """(video, frame, cell) -> midpoint of the historical pole endpoints (row, col)."""
    anchors = {}
    for path in sorted(glob.glob(str(REPO / 'validation/runs/experiment_*/measure/*_bs_detector_measurements.csv'))):
        for r in csv.DictReader(open(path)):
            anchors[(r['video_id'], int(r['frame']), r['cell_name'])] = (
                _point(r['new_pole_point']) + _point(r['old_pole_point'])) / 2
    return anchors


def _label_at(labels, point):
    y, x = np.rint(point).astype(int)
    if not (0 <= y < labels.shape[0] and 0 <= x < labels.shape[1]):
        return 0
    return int(labels[y, x])


def build_cells(cache_dir, packages=PACKAGES, cfg=None):
    cfg = cfg or Config()
    anchors = historical_anchors()
    records = []
    for package, (h5_dir, gt_path) in packages.items():
        gt_rows = [r for r in csv.DictReader(open(gt_path)) if r['scorer_initials'].upper() != 'EXAMPLE']
        for h5 in sorted(Path(h5_dir).glob('*.h5')):
            video = h5.stem
            with h5py.File(h5) as f:
                numbers = [int(n) for n in f['frame_numbers'][:]]
                for i, number in enumerate(numbers):
                    rows = [r for r in gt_rows if r['video_id'] == video and int(r['frame']) == number]
                    if not rows:
                        continue
                    image = f['frames'][i]
                    cache = Path(cache_dir) / video / f'{number:04d}.npz'
                    if not cache.exists():
                        records.extend(dict(key=(video, number, r['cell_name']), package=package,
                                            video=video, frame=number, cell=r['cell_name'],
                                            status='no cached mask') for r in rows)
                        continue
                    with np.load(cache) as z:
                        if str(z['image_sha256']) != hashlib.sha256(image.tobytes()).hexdigest():
                            raise ValueError(f'Stale cache {cache}')
                        labels = z['labels']
                        provenance = json.loads(str(z['provenance']))
                    valid = filter_valid_cells(regionprops(labels), image.shape, cfg)
                    info = prepare_cell_info(list(valid.values()), labels, cfg)
                    for r in rows:
                        gt = np.array([float(r['gt_scar_mid_y_px']), float(r['gt_scar_mid_x_px'])])
                        anchor = anchors[(video, number, r['cell_name'])]
                        label, label_gt = _label_at(labels, anchor), _label_at(labels, gt)
                        status = 'ok'
                        if label == 0 or label not in valid:
                            status = 'segmentation failure'
                        elif label != label_gt:
                            status = 'anchor/GT label disagreement'
                        rec = dict(key=(video, number, r['cell_name']), package=package, video=video,
                                   frame=number, cell=r['cell_name'], gt=gt, anchor=anchor, label=label,
                                   label_gt=label_gt, status=status, notes=r['notes'],
                                   gt_area=float(r['gt_area_px2']), gt_new_pole_guess=r['gt_new_pole_guess'],
                                   gt_poleA=float(r['gt_poleA_length_px'] or 'nan'),
                                   gt_poleB=float(r['gt_poleB_length_px'] or 'nan'),
                                   provenance=provenance)
                        if status == 'ok':
                            contour = max(find_contours(labels == label, 0.5), key=len)
                            smooth_pts, _ = compute_smoothed_curvature(contour, cfg.SMOOTH_FACTOR, cfg.N_CONTOUR_POINTS)
                            center, axis = compute_pca_axis(smooth_pts)
                            endpoints = get_contour_endpoints(contour, center, axis)
                            neighbors = find_pole_to_pole_neighbors(label, endpoints, info, cfg.POLE_PROXIMITY_THRESHOLD)
                            new_pole_idx, conf = determine_new_pole_from_neighbors(neighbors, endpoints, cfg)
                            lo = np.maximum(0, np.floor(contour.min(axis=0)).astype(int) - CROP_MARGIN)
                            hi = np.minimum(image.shape, np.ceil(contour.max(axis=0)).astype(int) + CROP_MARGIN + 1)
                            rec.update(contour=contour, endpoints=np.array(endpoints), neighbors=neighbors,
                                       new_pole_idx=new_pole_idx, neighbor_conf=conf,
                                       area=float((labels == label).sum()),
                                       crop=image[lo[0]:hi[0], lo[1]:hi[1]].copy(),
                                       label_crop=labels[lo[0]:hi[0], lo[1]:hi[1]].copy(),
                                       crop_origin=lo)
                        records.append(rec)
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cache', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    records = build_cells(a.cache)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'wb') as f:
        pickle.dump(records, f)
    bad = [r for r in records if r['status'] != 'ok']
    print(f'{len(records)} cells, {len(bad)} not usable')
    for r in bad:
        print('  ', r['key'], r['status'], 'anchor label', r.get('label'), 'gt label', r.get('label_gt'))


if __name__ == '__main__':
    main()
