"""Segment every brightfield/fluorescence ND2 pair once and save the masks.

The septin analysis and the division-classifier package both read these
label images, so a cell keeps the same ID everywhere it appears. Each frame is
stored as one HDF5 dataset named like the analysis cell_id prefix
(`Spn3-GFP_Sad1-mCh_080724_001_f0000`), with the brightfield SHA-256 and the
Cellpose provenance as attributes. Re-running skips frames already saved for
the same image, so an interrupted run resumes where it stopped and a saved
mask (and the cell IDs other files refer to) is never replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import h5py
import numpy as np

from pombe_tracker.config import Config
from pombe_tracker.segmentation import CellposeSegmenter
from septin_curve_vis.run_analysis import (image_sha256, mask_key, pair_nd2_files,
                                           project_stack, read_nd2_frames)


def brightfield_frames(bf_path: Path):
    """Yield (frame_idx, float32 brightfield) exactly as run_analysis builds it."""
    for frame_idx, (stack, _) in enumerate(read_nd2_frames(bf_path)):
        yield frame_idx, project_stack(stack, fluorescence=False).astype(np.float32)


def segment(inputs, output: Path, log=print, **segmenter_kw):
    """Segment every brightfield file of the ND2 pairs in `inputs` into the masks HDF5 `output`."""
    cfg = Config()
    pairs = [bf for root in inputs for bf, _ in pair_nd2_files(Path(root))]
    output.parent.mkdir(parents=True, exist_ok=True)
    segmenter, prov = None, None
    t_all = time.time()
    for k, bf_path in enumerate(pairs, 1):
        for frame_idx, bf in brightfield_frames(bf_path):
            key = mask_key(bf_path, frame_idx)
            digest = image_sha256(bf)
            if output.exists():
                with h5py.File(output, "r") as f:
                    if key in f and f[key].attrs["image_sha256"] == digest:
                        log(f"[{k}/{len(pairs)}] {key}: already segmented")
                        continue
            if segmenter is None:
                segmenter = CellposeSegmenter(cfg, **segmenter_kw)
                prov = segmenter.provenance
                log(json.dumps(prov))
            t0 = time.time()
            labels = segmenter.segment(bf)
            assert labels.max() < 2 ** 16, "too many objects for uint16"
            with h5py.File(output, "a") as f:
                if key in f:
                    del f[key]
                ds = f.create_dataset(key, data=labels.astype(np.uint16),
                                      compression="gzip", compression_opts=4)
                ds.attrs["image_sha256"] = digest
                ds.attrs["source"] = bf_path.name
                ds.attrs["source_bytes"] = bf_path.stat().st_size
                for name, value in prov.items():
                    ds.attrs[name] = json.dumps(value) if isinstance(value, list) else \
                        ("" if value is None else value)
            log(f"[{k}/{len(pairs)}] {key}: {int(labels.max())} objects, {time.time() - t0:.0f}s")
    log(f"Done in {time.time() - t_all:.0f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True,
                        help="Directories of ND2 pairs (read only)")
    parser.add_argument("--output", type=Path, required=True,
                        help="HDF5 file the masks are added to")
    args = parser.parse_args()
    segment(args.input, args.output)


if __name__ == "__main__":
    main()
