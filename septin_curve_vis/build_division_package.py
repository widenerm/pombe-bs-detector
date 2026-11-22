"""Build the dividing / not-dividing labelling package from the septin stills.

For every brightfield/fluorescence ND2 pair this writes, under
<output>/<split>/<date>/:

  <date>_<field>_image.tif    2 channels: brightfield, Sad1-mCherry max projection
  <date>_<field>_cells.tif    cell mask: pixel value = cell ID, 0 = no cell
  <date>_<field>_preview.png  numbered outlines, for finding a cell while labelling

plus cells.csv (one row per cell) and labels_template.csv at the top level.

The masks come from segment_fields.py and keep only the cells the septin
analysis measures, with the same IDs, so a label joins to measurements.csv
on cell_id. Spn3-GFP is left out on purpose: the classifier decides which
cells enter the septin measurement, so it must not see septin.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import nd2
import numpy as np
import tifffile
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries

from pombe_tracker.config import Config
from pombe_tracker.pipeline import CellProcessor
from septin_curve_vis.run_analysis import (analysis_cells, image_sha256, load_mask,
                                           mask_key, pair_nd2_files, project_stack,
                                           read_nd2_frames)

CELL_COLUMNS = ["split", "date", "field", "cell", "cell_id", "image_file", "mask_file",
                "centroid_x_px", "centroid_y_px",
                "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1", "area_px"]
LABEL_COLUMNS = ["split", "date", "field", "cell", "cell_id", "label", "notes"]


def single_frame(path: Path):
    frames = list(read_nd2_frames(path))
    if len(frames) != 1:
        raise ValueError(f"{path.name}: expected one time point, found {len(frames)}")
    return frames[0][0]


def pixel_size_um(path: Path) -> float:
    with nd2.ND2File(path) as f:
        return float(f.voxel_size().x)


def channel_index(path: Path, name: str) -> int:
    with nd2.ND2File(path) as f:
        names = [c.channel.name for c in f.metadata.channels]
    matches = [i for i, n in enumerate(names) if name.lower() in n.lower()]
    if len(matches) != 1:
        raise ValueError(f"{path.name}: channel {name!r} not found once in {names}")
    return matches[0]


def sad1_max_projection(fl_path: Path, key: str, cache: Path, channel_name: str):
    """Max projection over Z of the Sad1 channel, cached so the stack is read once."""
    size = fl_path.stat().st_size
    with h5py.File(cache, "a") as f:
        if key in f and f[key].attrs["source_bytes"] == size:
            return f[key][()]
    ch = channel_index(fl_path, channel_name)
    stack = single_frame(fl_path)                      # Z, C, Y, X
    projection = stack[:, ch].max(axis=0)
    with h5py.File(cache, "a") as f:
        if key in f:
            del f[key]
        ds = f.create_dataset(key, data=projection, compression="gzip", compression_opts=4)
        ds.attrs["source"] = str(fl_path)
        ds.attrs["source_bytes"] = size
        ds.attrs["channel"] = channel_name
        ds.attrs["n_planes"] = stack.shape[0]
    return projection


def grey_lut():
    ramp = np.arange(256, dtype=np.uint8)
    return np.stack([ramp, ramp, ramp])


def magenta_lut():
    ramp = np.arange(256, dtype=np.uint8)
    return np.stack([ramp, np.zeros_like(ramp), ramp])


def write_tiffs(image_path, mask_path, bf, sad1, cells, px_um):
    resolution = (1.0 / px_um, 1.0 / px_um)
    tifffile.imwrite(image_path, np.stack([bf, sad1]).astype(np.uint16), imagej=True,
                     resolution=resolution,
                     metadata={"axes": "CYX", "unit": "um", "mode": "composite",
                               "Labels": ["Brightfield", "Sad1-mCherry max projection"],
                               "LUTs": [grey_lut(), magenta_lut()]})
    tifffile.imwrite(mask_path, cells.astype(np.uint16), imagej=True,
                     resolution=resolution, metadata={"axes": "YX", "unit": "um"})


def stretch(image, lo, hi):
    a, b = np.percentile(image, [lo, hi])
    return np.clip((image.astype(np.float32) - a) / max(b - a, 1e-6), 0, 1)


def save_preview(path, title, bf, sad1, cells):
    """Brightfield with numbered outlines | Sad1 with outlines, at half size.

    Numbers sit only on the brightfield panel so they never hide Sad1 dots.
    """
    half = (slice(None, None, 2), slice(None, None, 2))
    edges = find_boundaries(cells, mode="inner")
    edges = edges.reshape(edges.shape[0] // 2, 2, edges.shape[1] // 2, 2).any(axis=(1, 3))
    overlay = np.zeros(edges.shape + (4,), dtype=np.float32)
    overlay[edges] = (1.0, 0.85, 0.0, 1.0)
    h, w = edges.shape
    dpi = 100
    fig = plt.figure(figsize=(2 * w / dpi, h / dpi), dpi=dpi)
    for i, (img, cmap, lo, hi) in enumerate([(bf, "gray", 1, 99.5),
                                             (sad1, "magma", 50, 99.95)]):
        ax = fig.add_axes([i / 2, 0, 0.5, 1])
        ax.imshow(stretch(img[half], lo, hi), cmap=cmap, interpolation="nearest")
        ax.imshow(overlay, interpolation="nearest")
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(h - 0.5, -0.5)
        ax.axis("off")
        if i == 0:
            for region in regionprops(cells):
                r, c = region.centroid
                ax.text(c / 2, r / 2, str(region.label), color="yellow", fontsize=13,
                        ha="center", va="center", weight="bold",
                        path_effects=[pe.withStroke(linewidth=3, foreground="black")])
        ax.text(10, 25, title if i == 0 else "Sad1-mCherry (max projection)",
                color="white", fontsize=16,
                path_effects=[pe.withStroke(linewidth=3, foreground="black")])
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def build_pair(bf_path, fl_path, split, out, masks, cache, cfg, processor, args):
    date, field = bf_path.stem.split("_")[-2:]
    key = mask_key(bf_path, 0)
    bf = project_stack(single_frame(bf_path), fluorescence=False)   # uint16
    sad1 = sad1_max_projection(fl_path, key, cache, args.sad1_channel)
    if bf.shape != sad1.shape:
        raise ValueError(f"{bf_path.name}: brightfield {bf.shape} vs Sad1 {sad1.shape}")
    px_um = pixel_size_um(bf_path)
    if not np.isclose(px_um, pixel_size_um(fl_path)):
        raise ValueError(f"{bf_path.name}: pixel size differs from {fl_path.name}")

    bf32 = bf.astype(np.float32)
    labels = load_mask(masks, bf_path, 0, bf32)
    keep = sorted(r["label"] for r in analysis_cells(labels, bf32, cfg, processor))
    cells = np.where(np.isin(labels, keep), labels, 0).astype(np.uint16)

    folder = out / split / date
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{date}_{field}"
    image_path = folder / f"{stem}_image.tif"
    mask_path = folder / f"{stem}_cells.tif"
    write_tiffs(image_path, mask_path, bf, sad1, cells, px_um)
    save_preview(folder / f"{stem}_preview.png", f"{date} field {field}", bf, sad1, cells)

    rows = []
    for region in regionprops(cells):
        r0, c0, r1, c1 = region.bbox
        r, c = region.centroid
        rows.append({
            "split": split, "date": date, "field": field, "cell": region.label,
            "cell_id": f"{key}_l{region.label}",
            "image_file": str(image_path.relative_to(out)),
            "mask_file": str(mask_path.relative_to(out)),
            "centroid_x_px": round(c, 1), "centroid_y_px": round(r, 1),
            "bbox_x0": c0, "bbox_y0": r0, "bbox_x1": c1, "bbox_y1": r1,
            "area_px": int(region.area),
        })
    print(f"{split}/{stem}: {len(rows)} cells ({int(labels.max())} Cellpose objects), "
          f"{px_um:.5f} um/px", flush=True)
    return rows, px_um


def write_csv(rows, path, columns):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, nargs="+", required=True,
                        help="ND2 pair directories for training (read only)")
    parser.add_argument("--test", type=Path, nargs="+", required=True,
                        help="ND2 pair directories held out for testing (read only)")
    parser.add_argument("--masks", type=Path, required=True,
                        help="Masks HDF5 from segment_fields.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sad1-cache", type=Path, default=None,
                        help="HDF5 cache of Sad1 projections (default: beside --masks)")
    parser.add_argument("--sad1-channel", default="Texas Red",
                        help="ND2 channel name of Sad1-mCherry")
    args = parser.parse_args()
    cache = args.sad1_cache or args.masks.with_name("sad1_max_projections.h5")

    cfg = Config()
    processor = CellProcessor(cfg)
    rows = []
    for split, roots in (("train", args.train), ("test", args.test)):
        for root in roots:
            for bf_path, fl_path in pair_nd2_files(root):
                pair_rows, _ = build_pair(bf_path, fl_path, split, args.output,
                                          args.masks, cache, cfg, processor, args)
                rows.extend(pair_rows)
    write_csv(rows, args.output / "cells.csv", CELL_COLUMNS)
    write_csv([dict(r, label="", notes="") for r in rows],
              args.output / "labels_template.csv", LABEL_COLUMNS)
    for split in ("train", "test"):
        n_fields = len({(r["date"], r["field"]) for r in rows if r["split"] == split})
        print(f"{split}: {n_fields} fields, {sum(r['split'] == split for r in rows)} cells")


if __name__ == "__main__":
    main()
