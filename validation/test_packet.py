"""Held-out test packet: single frames of cells from movies never used in development.

Two steps, because ND2 reading and the pinned Cellpose live in different
environments:

    # 1. pick frames and copy them out of the ND2 movies (needs the nd2 package)
    python -m validation.test_packet extract MOVIE.nd2 [...] --h5 validation/runs/scar_test/h5

    # 2. segment with the pinned cpsam weights, sample cells, write the packet
    python -m validation.test_packet build --h5 validation/runs/scar_test/h5 \
        --cache validation/runs/scar_test/cache --out validation/packets/test \
        --key validation/test_set/test_cells.csv

Frames are spaced SPACING frames apart (longer than a cell cycle), so no cell
is annotated twice. Cells are drawn at random, the same number per movie, from
every object the pipeline would analyse (size, border, aspect ratio and
circularity filters only). The segmentation-quality flags are NOT applied, so
bad outlines reach the scorer and count as failures. The scar detector is not
run here: nothing about the test cells is seen before the annotations return.
"""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SPACING = 100          # frames (200 min at 2 min/frame), longer than a WT cycle
FIRST = (20, 60)       # first frame drawn uniformly from this range
COLUMNS = ['video_id', 'raw_frame_file', 'image_file', 'frame', 'cell_name', 'scorer_initials', 'status',
           'gt_area_px2', 'gt_scar_mid_x_px', 'gt_scar_mid_y_px', 'gt_poleA_length_px', 'gt_poleB_length_px',
           'gt_new_pole_guess', 'notes']


def pick_frames(n_frames, rng, spacing=SPACING, first=FIRST):
    start = int(rng.integers(first[0], first[1] + 1))
    return list(range(start, n_frames, spacing))


def extract(movies, h5_dir, seed):
    import h5py
    import nd2
    rng = np.random.default_rng(seed)
    h5_dir = Path(h5_dir)
    h5_dir.mkdir(parents=True, exist_ok=True)
    for movie in sorted(Path(m) for m in movies):
        with nd2.ND2File(movie) as f:
            if set(f.sizes) != {'T', 'Y', 'X'}:
                raise ValueError(f'{movie.name}: expected a single-channel T,Y,X movie, got {dict(f.sizes)}')
            frames = pick_frames(f.sizes['T'], rng)
            data = np.stack([f.read_frame(i) for i in frames])
            out = h5_dir / f'{movie.stem}.h5'
            with h5py.File(out, 'w') as h:
                h.create_dataset('frames', data=data, compression='gzip')
                h.create_dataset('frame_numbers', data=np.asarray(frames, dtype=np.int64))
                h.attrs.update(video_id=movie.stem, source=str(movie), source_bytes=movie.stat().st_size,
                               n_frames_in_movie=f.sizes['T'], pixel_size_um=f.voxel_size().x, seed=seed,
                               spacing=SPACING)
        print(f'{movie.stem}: frames {frames}')


def anchor_inside(mask):
    """A pixel inside the mask, the one nearest its centroid (centroids of bent cells can fall outside)."""
    ys, xs = np.nonzero(mask)
    cy, cx = ys.mean(), xs.mean()
    i = int(np.argmin((ys - cy) ** 2 + (xs - cx) ** 2))
    return int(ys[i]), int(xs[i])


def render(frame, labels, cells, title, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from skimage.measure import find_contours
    lo, hi = np.percentile(frame, [0.5, 99.8])
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(frame, cmap='gray', vmin=lo, vmax=hi)
    for c in cells:
        contour = max(find_contours(labels == c['label'], 0.5), key=len)
        ax.plot(contour[:, 1], contour[:, 0], color='deepskyblue', lw=1.2)
        ax.text(c['anchor_x'], c['anchor_y'], c['cell_name'], color='white', fontsize=10, fontweight='bold',
                ha='center', va='center', path_effects=[pe.withStroke(linewidth=2.5, foreground='black')])
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def build(h5_dir, cache, out, key, n_per_movie, batches, seed):
    import h5py
    import tifffile
    from skimage.measure import regionprops
    from pombe_tracker.config import Config
    from pombe_tracker.pipeline import filter_valid_cells
    from validation.scar_cells import segment_missing
    cfg = Config()
    h5_dir, cache, out, key = Path(h5_dir), Path(cache), Path(out), Path(key)
    segment_missing(cache, {'test': (h5_dir, None)}, precision='float32')
    rng = np.random.default_rng(seed)
    cells, frames = [], {}
    for h5 in sorted(h5_dir.glob('*.h5')):
        video = h5.stem
        pool = []
        with h5py.File(h5) as f:
            for i, number in enumerate(int(n) for n in f['frame_numbers'][:]):
                image = f['frames'][i]
                with np.load(cache / video / f'{number:04d}.npz') as z:
                    if str(z['image_sha256']) != hashlib.sha256(image.tobytes()).hexdigest():
                        raise ValueError(f'Stale mask for {video} frame {number}')
                    labels = z['labels']
                    provenance = json.loads(str(z['provenance']))
                frames[(video, number)] = dict(image=image, labels=labels, provenance=provenance,
                                               sha=hashlib.sha256(image.tobytes()).hexdigest())
                pool += [(number, int(lab)) for lab in filter_valid_cells(regionprops(labels), image.shape, cfg)]
        take = rng.choice(len(pool), size=min(n_per_movie, len(pool)), replace=False)
        print(f'{video}: {len(pool)} eligible objects in {len({p[0] for p in pool})} frames, sampled {len(take)}')
        for j in take:
            number, lab = pool[j]
            mask = frames[(video, number)]['labels'] == lab
            ay, ax = anchor_inside(mask)
            cells.append(dict(video_id=video, frame=number, label=lab, anchor_y=ay, anchor_x=ax,
                              area_px=int(mask.sum())))

    # Batches: deal the shuffled cells round-robin movie by movie, continuing the deal across movies,
    # so every batch spans all movies and batch sizes differ by at most one.
    order = rng.permutation(len(cells))
    by_video = {}
    for i in order:
        by_video.setdefault(cells[i]['video_id'], []).append(cells[i])
    dealt = 0
    for vcells in by_video.values():
        for c in vcells:
            c['batch'] = dealt % batches + 1
            dealt += 1
    # Names: c01, c02, ... top to bottom within each frame.
    for (video, number) in frames:
        fc = sorted([c for c in cells if (c['video_id'], c['frame']) == (video, number)],
                    key=lambda c: (c['anchor_y'], c['anchor_x']))
        for k, c in enumerate(fc, 1):
            c['cell_name'] = f'c{k:02d}'

    (out / 'raw_frames').mkdir(parents=True, exist_ok=True)
    (out / 'images').mkdir(parents=True, exist_ok=True)
    used = sorted({(c['video_id'], c['frame']) for c in cells})
    for video, number in used:
        fr = frames[(video, number)]
        fc = [c for c in cells if (c['video_id'], c['frame']) == (video, number)]
        stem = f'{video}_frame{number:04d}'
        tifffile.imwrite(out / 'raw_frames' / f'{stem}_raw.tif', fr['image'])
        render(fr['image'], fr['labels'], fc,
               f'{video}  frame {number}  -  annotate the {len(fc)} outlined cells (blinded)',
               out / 'images' / f'{stem}_blinded.png')
        for c in fc:
            c.update(raw_frame_file=f'{stem}_raw.tif', image_file=f'{stem}_blinded.png', image_sha256=fr['sha'])

    for b in range(1, batches + 1):
        rows = sorted([c for c in cells if c['batch'] == b], key=lambda c: (c['video_id'], c['frame'], c['cell_name']))
        with open(out / f'ground_truth_template_batch{b}.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, COLUMNS)
            w.writeheader()
            for c in rows:
                w.writerow({k: c.get(k, '') for k in ('video_id', 'raw_frame_file', 'image_file', 'frame', 'cell_name')})
    shutil.copy(REPO / 'validation/GROUND_TRUTH_PROTOCOL.md', out / 'GROUND_TRUTH_PROTOCOL.md')

    key.parent.mkdir(parents=True, exist_ok=True)
    fields = ['batch', 'video_id', 'frame', 'cell_name', 'label', 'anchor_y', 'anchor_x', 'area_px',
              'raw_frame_file', 'image_sha256']
    with open(key, 'w', newline='') as f:
        w = csv.DictWriter(f, fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(sorted(cells, key=lambda c: (c['batch'], c['video_id'], c['frame'], c['cell_name'])))
    prov = {f'{v}/{n}': frames[(v, n)]['provenance'] for v, n in used}
    (key.parent / 'segmentation_provenance.json').write_text(json.dumps(prov, indent=1, sort_keys=True))
    print(f'{len(cells)} cells in {len(used)} frames; batches ' +
          ', '.join(f'{b}: {sum(c["batch"] == b for c in cells)}' for b in range(1, batches + 1)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('extract')
    e.add_argument('movies', nargs='+')
    e.add_argument('--h5', required=True)
    e.add_argument('--seed', type=int, default=20261103)
    b = sub.add_parser('build')
    b.add_argument('--h5', required=True)
    b.add_argument('--cache', required=True)
    b.add_argument('--out', required=True)
    b.add_argument('--key', required=True)
    b.add_argument('--n-per-movie', type=int, default=19)
    b.add_argument('--batches', type=int, default=3)
    b.add_argument('--seed', type=int, default=20261103)
    a = p.parse_args()
    if a.cmd == 'extract':
        extract(a.movies, a.h5, a.seed)
    else:
        build(a.h5, a.cache, a.out, a.key, a.n_per_movie, a.batches, a.seed)


if __name__ == '__main__':
    main()
