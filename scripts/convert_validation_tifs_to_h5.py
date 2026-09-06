#!/usr/bin/env python3
"""Convert sparse validation TIFF frames into HDF5 files.

The validation packages contain only manually annotated frames, not complete
movies. Each output H5 therefore stores the selected images in ``frames`` and
the original movie indices in ``frame_numbers`` so ground-truth CSV rows keep
their original frame values.

Examples
--------
python scripts/convert_validation_tifs_to_h5.py \
    validation/package_for_carmen \
    --output validation/h5_for_drive/package_for_carmen

python scripts/convert_validation_tifs_to_h5.py \
    validation/carmen_validation_package_2.zip \
    --output validation/h5_for_drive/package_for_carmen_2
"""
import argparse
import csv
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

import h5py
import numpy as np
import tifffile


FRAME_RE = re.compile(r"^(?P<video>.+)_frame(?P<frame>\d+)_raw\.(?:tif|tiff)$",
                      re.IGNORECASE)


def _prepare_input(source):
    """Return (package_root, cleanup_dir) for a folder or ZIP source."""
    source = Path(source).resolve()
    if source.is_dir():
        return source, None
    if source.suffix.lower() != '.zip':
        raise ValueError(f'Input must be a package directory or ZIP: {source}')
    cleanup = Path(tempfile.mkdtemp(prefix='bs_validation_package_'))
    with zipfile.ZipFile(source) as archive:
        archive.extractall(cleanup)
    # Packages may contain an extra top-level ``validation/`` directory.
    # Locate the directory that owns raw_frames instead of assuming a fixed
    # archive layout.
    raw_dirs = list(cleanup.rglob('raw_frames'))
    if len(raw_dirs) == 1:
        return raw_dirs[0].parent, cleanup
    roots = [p for p in cleanup.iterdir() if p.is_dir()]
    return (roots[0] if len(roots) == 1 else cleanup), cleanup


def _find_raw_frames(root):
    grouped = {}
    for path in (root / 'raw_frames').glob('*'):
        match = FRAME_RE.match(path.name)
        if match:
            grouped.setdefault(match.group('video'), []).append(
                (int(match.group('frame')), path))
    if not grouped:
        raise ValueError(f'No raw validation TIFFs found under {root / "raw_frames"}')
    return {video: sorted(items) for video, items in grouped.items()}


def _write_video_h5(video_id, frames, output_dir, source_root):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{video_id}.h5'
    first = tifffile.imread(frames[0][1])
    if first.ndim != 2:
        raise ValueError(f'{frames[0][1]} is not a 2-D grayscale TIFF')
    shape, dtype = first.shape, first.dtype

    with h5py.File(output_path, 'w') as h5:
        dataset = h5.create_dataset(
            'frames', shape=(len(frames), *shape), dtype=dtype,
            compression='gzip', compression_opts=4, shuffle=True)
        frame_numbers = h5.create_dataset(
            'frame_numbers', data=np.asarray([n for n, _ in frames], dtype=np.int64))
        h5.attrs['video_id'] = video_id
        h5.attrs['source_package'] = str(source_root)
        for i, (_, path) in enumerate(frames):
            image = tifffile.imread(path)
            if image.shape != shape or image.ndim != 2:
                raise ValueError(f'Inconsistent TIFF shape in {path}: {image.shape}')
            if image.dtype != dtype:
                image = image.astype(dtype, copy=False)
            dataset[i] = image
    return output_path


def convert(source, output_dir):
    root, cleanup = _prepare_input(source)
    try:
        groups = _find_raw_frames(root)
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = []
        for video_id, frames in sorted(groups.items()):
            path = _write_video_h5(video_id, frames, output_dir, root)
            for local_index, (frame_number, tif_path) in enumerate(frames):
                manifest.append({
                    'video_id': video_id,
                    'local_index': local_index,
                    'frame': frame_number,
                    'source_tif': tif_path.name,
                    'h5_file': path.name,
                })
            print(f'{video_id}: {len(frames)} frame(s) -> {path}')

        with (output_dir / 'frame_manifest.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(manifest[0]))
            writer.writeheader()
            writer.writerows(manifest)
        print(f'Manifest -> {output_dir / "frame_manifest.csv"}')
        return output_dir
    finally:
        if cleanup is not None:
            shutil.rmtree(cleanup)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', help='package directory or ZIP file')
    parser.add_argument('--output', required=True, help='directory for H5 files')
    args = parser.parse_args()
    convert(args.source, args.output)


if __name__ == '__main__':
    main()
