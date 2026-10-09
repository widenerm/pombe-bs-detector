"""Build frozen per-cell records for fast scar-detector tuning.

Each annotated cell from Carmen's packages is located in a cached full-frame
Cellpose mask. Identity uses the historical pole-midpoint anchor and must agree
with the label containing the annotated scar; disagreements are kept and
flagged, never silently resolved. The record stores everything the detector
sees in production (contour, frame crop, neighbor-derived pole hint) plus the
ground truth, so detector variants can be scored without re-segmenting.

    python -m validation.scar_cells --cache validation/runs/scar_v3/cache_h200_bf16 \
        --out validation/runs/scar_v3/cells_h200.pkl

On Colab, point --h5-root at the Drive folder holding carmen_package_1/ and
carmen_package_2/ and add --segment-missing to run Cellpose on any frame
without a cached mask (masks are cached, so later runs are fast).
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
ANCHORS_CSV = REPO / 'validation/reports/scar_v3/cell_anchors.csv'
PINNED_MODEL_SHA256 = 'e1440429eb384f95afe32bcba6510f90d518eaedc917ede549bed6804004abe2'


def _point(text):
    return np.array(json.loads(text), dtype=float)


def historical_anchors(path=ANCHORS_CSV):
    """(video, frame, cell) -> midpoint of the historical pole endpoints (row, col).

    Read from the committed anchors CSV; falls back to the original (git-ignored)
    historical measurement tables if it is absent."""
    if Path(path).exists():
        return {(r['video_id'], int(r['frame']), r['cell_name']):
                np.array([float(r['anchor_y']), float(r['anchor_x'])]) for r in csv.DictReader(open(path))}
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


def packages_under(h5_root):
    """Same GT files, H5 packages read from another folder (e.g. Google Drive)."""
    return {k: (Path(h5_root) / Path(h5).name, gt) for k, (h5, gt) in PACKAGES.items()}


def segment_missing(cache_dir, packages=PACKAGES, precision='bfloat16'):
    """Run Cellpose-SAM on every package frame that has no cached mask."""
    from importlib.metadata import version
    from cellpose import core, models
    model = None
    for h5_dir, _ in packages.values():
        for h5 in sorted(Path(h5_dir).glob('*.h5')):
            with h5py.File(h5) as f:
                for i, number in enumerate(int(n) for n in f['frame_numbers'][:]):
                    cache = Path(cache_dir) / h5.stem / f'{number:04d}.npz'
                    if cache.exists():
                        continue
                    if model is None:
                        model = models.CellposeModel(gpu=core.use_gpu(), use_bfloat16=precision == 'bfloat16')
                        weights = Path(model.pretrained_model)
                        prov = dict(cellpose=version('cellpose'), torch=version('torch'), model=str(weights),
                                    model_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                                    diameter=None, channels=[0, 0], batch_size=1,
                                    device=str(model.device), dtype=precision)
                        if prov['model_sha256'] != PINNED_MODEL_SHA256:
                            raise ValueError('Cellpose weights differ from the validated cpsam model')
                    image = f['frames'][i]
                    labels, _, _ = model.eval(image, diameter=None, channels=[0, 0], batch_size=1)
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    tmp = cache.with_suffix('.partial.npz')
                    np.savez_compressed(tmp, labels=labels, image_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                                        provenance=json.dumps(prov, sort_keys=True))
                    tmp.replace(cache)
                    print(f'segmented {h5.stem} frame {number}', flush=True)


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
    p.add_argument('--h5-root', default=None, help='folder with carmen_package_1/ and carmen_package_2/')
    p.add_argument('--segment-missing', action='store_true')
    p.add_argument('--precision', choices=['bfloat16', 'float32'], default='bfloat16')
    a = p.parse_args()
    packages = packages_under(a.h5_root) if a.h5_root else PACKAGES
    if a.segment_missing:
        segment_missing(a.cache, packages, a.precision)
    records = build_cells(a.cache, packages)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'wb') as f:
        pickle.dump(records, f)
    bad = [r for r in records if r['status'] != 'ok']
    print(f'{len(records)} cells, {len(bad)} not usable')
    for r in bad:
        print('  ', r['key'], r['status'], 'anchor label', r.get('label'), 'gt label', r.get('label_gt'))


if __name__ == '__main__':
    main()
