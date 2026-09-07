"""Brightfield segmentation + fluorescence curvature analysis.

This is deliberately frame-local: it does not load an entire experiment into
RAM and it does not require the BS-Detector lineage tracker. It reuses the
repository's Cellpose segmentation and B-spline curvature implementation.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from scipy.ndimage import map_coordinates

from pombe_tracker.config import Config
from pombe_tracker.geometry import compute_smoothed_curvature
from pombe_tracker.pipeline import CellProcessor, filter_valid_cells, prepare_cell_info
from pombe_tracker.segmentation import CellposeSegmenter
from skimage.measure import find_contours, regionprops


def natural_key(path: Path):
    return [int(x) if x.isdigit() else x.lower()
            for x in re.split(r"(\d+)", path.name)]


def pair_nd2_files(root: Path):
    files = sorted(root.glob("*.nd2"), key=natural_key)
    if len(files) < 2 or len(files) % 2:
        raise ValueError(f"Expected an even number of ND2 files, found {len(files)}")
    pairs = []
    for left, right in zip(files[::2], files[1::2]):
        bf, fl = sorted((left, right), key=lambda p: p.stat().st_size)
        pairs.append((bf, fl))
    return pairs


def nd2_axes(path: Path):
    """Return an nd2.ND2File object, keeping the optional dependency isolated."""
    try:
        import nd2
    except ImportError as exc:
        raise RuntimeError(
            "ND2 support requires the `nd2` package. Install requirements.txt first."
        ) from exc
    return nd2.ND2File(path)


def read_nd2_frames(path: Path):
    """Yield 2-D frames and metadata for common nd2 layouts.

    The `nd2` package exposes an ndarray with named axes. Z is summed for
    fluorescence by the caller; brightfield is reduced to its first plane.
    """
    with nd2_axes(path) as f:
        sizes = dict(f.sizes)
        arr = f.asarray()
    axes = list(sizes)
    arr = np.asarray(arr)
    unsupported = set(axes) - set("TZCYX")
    if unsupported:
        raise ValueError(f"Unsupported ND2 axes in {path.name}: {sorted(unsupported)}")
    # Normalize to T,Z,C,Y,X. Missing T/Z/C dimensions are inserted.
    present = list(axes)
    for axis in "TZCYX":
        if axis not in present:
            arr = np.expand_dims(arr, axis=0)
            present.insert(0, axis)
    arr = np.moveaxis(arr, [present.index(a) for a in "TZCYX"], range(5))
    for t in range(arr.shape[0]):
        yield arr[t], sizes


def project_stack(stack, fluorescence: bool, channel: int = 0):
    stack = np.asarray(stack)
    # read_nd2_frames returns Z,C,Y,X. Select GFP by default, then sum Z.
    if stack.ndim == 4:
        if channel >= stack.shape[1]:
            raise ValueError(f"Requested channel {channel}, but only {stack.shape[1]} exist")
        stack = stack[:, channel]
    if stack.ndim == 3:
        return np.sum(stack, axis=0, dtype=np.float32) if fluorescence else stack[0]
    if stack.ndim != 2:
        raise ValueError(f"Expected 2-D or 3-D image, got shape {stack.shape}")
    return stack.astype(np.float32, copy=False)


def sample_band(image, points_rc, normals_rc, half_width=2.0):
    """Mean fluorescence in a narrow normal-direction band at each contour point."""
    values = []
    offsets = np.arange(-half_width, half_width + 0.01, 1.0)
    for point, normal in zip(points_rc, normals_rc):
        coords = np.asarray(point)[:, None] + np.asarray(normal)[:, None] * offsets
        sampled = map_coordinates(image, coords, order=1, mode="nearest")
        values.append(float(np.mean(sampled)))
    return np.asarray(values)


def contour_normals(points):
    tangent = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-8
    return np.column_stack((-tangent[:, 1], tangent[:, 0]))


def save_overlay(image, points, curvature, path):
    finite = np.abs(curvature[np.isfinite(curvature)])
    limit = float(np.percentile(finite, 98)) if finite.size else 1.0
    limit = max(limit, 1e-6)
    # The PNG must remain pixel-registered with the source image. Do not use
    # bbox_inches="tight" or a colorbar here; either changes the canvas size.
    height, width = image.shape[:2]
    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    norm = Normalize(vmin=-limit, vmax=limit)
    sc = ax.scatter(points[:, 1], points[:, 0], c=curvature,
                    cmap="RdBu_r", norm=norm, s=5, linewidths=0)
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)
    ax.axis("off")
    fig.savefig(path, transparent=True, pad_inches=0)
    plt.close(fig)


def save_crop_composite(image, points, curvature, path, pad=30):
    """Save a fluorescence crop with the curvature heatmap overlaid."""
    h, w = image.shape[:2]
    r0 = max(0, int(np.floor(points[:, 0].min())) - pad)
    r1 = min(h, int(np.ceil(points[:, 0].max())) + pad + 1)
    c0 = max(0, int(np.floor(points[:, 1].min())) - pad)
    c1 = min(w, int(np.ceil(points[:, 1].max())) + pad + 1)
    crop = image[r0:r1, c0:c1]
    local = points - np.array([r0, c0])
    finite = np.abs(curvature[np.isfinite(curvature)])
    limit = max(float(np.percentile(finite, 98)) if finite.size else 1.0, 1e-6)
    dpi = 150
    fig = plt.figure(figsize=(crop.shape[1] / dpi, crop.shape[0] / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(crop, cmap="gray", vmin=np.percentile(crop, 1),
              vmax=np.percentile(crop, 99.5))
    ax.scatter(local[:, 1], local[:, 0], c=curvature,
               cmap="RdBu_r", norm=Normalize(vmin=-limit, vmax=limit),
               s=8, linewidths=0)
    ax.set_xlim(-0.5, crop.shape[1] - 0.5)
    ax.set_ylim(crop.shape[0] - 0.5, -0.5)
    ax.axis("off")
    fig.savefig(path, dpi=dpi, pad_inches=0)
    plt.close(fig)


def save_frame_overlay(image, tracks, path):
    """Save one transparent full-frame overlay containing all cell contours."""
    height, width = image.shape[:2]
    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    for points, curvature in tracks:
        finite = np.abs(curvature[np.isfinite(curvature)])
        limit = max(float(np.percentile(finite, 98)) if finite.size else 1.0, 1e-6)
        ax.scatter(points[:, 1], points[:, 0], c=curvature,
                   cmap="RdBu_r", norm=Normalize(vmin=-limit, vmax=limit),
                   s=5, linewidths=0)
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)
    ax.axis("off")
    fig.savefig(path, transparent=True, pad_inches=0)
    plt.close(fig)


def save_profile(curvature, intensity, path, title):
    x = np.linspace(0, 1, len(curvature), endpoint=False)
    fig, ax1 = plt.subplots(figsize=(8, 3), dpi=150)
    ax1.plot(x, curvature, color="tab:blue", label="curvature")
    ax1.axhline(0, color="0.5", lw=0.6)
    ax1.set_xlabel("normalized contour position")
    ax1.set_ylabel("signed curvature", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(x, intensity, color="tab:red", label="septin intensity")
    ax2.set_ylabel("background-subtracted fluorescence", color="tab:red")
    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def analyze_pair(bf_path, fl_path, output, segmenter, cfg, args):
    bf_frames = read_nd2_frames(bf_path)
    fl_frames = read_nd2_frames(fl_path)
    processor = CellProcessor(cfg)
    measurements, summaries = [], []
    frame_tracks = {}
    for frame_idx, (bf_stack, _) in enumerate(bf_frames):
        try:
            fl_stack, _ = next(fl_frames)
        except StopIteration:
            break
        bf = project_stack(bf_stack, fluorescence=False).astype(np.float32)
        fl = project_stack(fl_stack, fluorescence=True,
                           channel=args.fluor_channel).astype(np.float32)
        if bf.shape != fl.shape:
            raise ValueError(
                f"Brightfield/fluorescence shape mismatch in frame {frame_idx}: "
                f"{bf.shape} vs {fl.shape}. Check channel registration/cropping."
            )
        labels = segmenter.segment(bf)
        regions = filter_valid_cells(regionprops(labels), bf.shape, cfg)
        if not regions:
            continue
        valid_regions = list(regions.values())
        cell_info = prepare_cell_info(valid_regions, labels, cfg)
        for region in valid_regions:
            result = processor.process_cell(region, labels, bf, cell_info)
            if result is None or result.get("seg_quality") == "border_clip":
                continue
            dbg = result["debug_info"]
            points = np.asarray(dbg["smooth_pts"], dtype=float)
            curvature = np.asarray(dbg["kappa"], dtype=float)
            intensity = sample_band(fl, points, contour_normals(points), args.band_width)
            background = float(np.percentile(fl, 10))
            intensity = intensity - background
            cell_id = f"{bf_path.stem}_f{frame_idx:04d}_l{result['label']}"
            overlay = output / "overlays" / f"{cell_id}.png"
            profile = output / "profiles" / f"{cell_id}.png"
            overlay.parent.mkdir(parents=True, exist_ok=True)
            profile.parent.mkdir(parents=True, exist_ok=True)
            crop_dir = output / "cell_crops"
            crop_dir.mkdir(parents=True, exist_ok=True)
            save_overlay(fl, points, curvature, overlay)
            save_crop_composite(fl, points, curvature,
                                crop_dir / f"{cell_id}.png")
            save_profile(curvature, intensity, profile, cell_id)
            frame_tracks.setdefault(frame_idx, []).append((points, curvature))
            for i, (point, kappa, fluor) in enumerate(zip(points, curvature, intensity)):
                measurements.append({
                    "pair": bf_path.stem,
                    "frame": frame_idx,
                    "label": result["label"],
                    "cell_id": cell_id,
                    "contour_index": i,
                    "contour_position": i / len(points),
                    "row": point[0],
                    "col": point[1],
                    "curvature": kappa,
                    "fluorescence_intensity": fluor,
                })
            summaries.append({
                "pair": bf_path.stem, "frame": frame_idx,
                "label": result["label"], "cell_id": cell_id,
                "n_contour_points": len(points),
                "mean_curvature": float(np.mean(curvature)),
                "max_abs_curvature": float(np.max(np.abs(curvature))),
                "mean_fluorescence": float(np.mean(intensity)),
                "max_fluorescence": float(np.max(intensity)),
            })
        frame_dir = output / "frame_overlays"
        frame_dir.mkdir(parents=True, exist_ok=True)
        tracks = frame_tracks.get(frame_idx, [])
        if tracks:
            save_frame_overlay(fl, tracks,
                               frame_dir / f"{bf_path.stem}_f{frame_idx:04d}.png")
    return measurements, summaries


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--band-width", type=float, default=2.0)
    parser.add_argument("--fluor-channel", type=int, default=0,
                        help="ND2 fluorescence channel to analyze (default: 0, Spn3-GFP)")
    parser.add_argument("--smooth-factor", type=float, default=40.0)
    parser.add_argument("--contour-points", type=int, default=300)
    parser.add_argument("--limit-pairs", type=int, default=None,
                        help="Only process the first N pairs (useful for validation)")
    args = parser.parse_args()
    cfg = Config()
    cfg.SMOOTH_FACTOR = args.smooth_factor
    cfg.N_CONTOUR_POINTS = args.contour_points
    pairs = pair_nd2_files(args.input)
    if args.limit_pairs is not None:
        pairs = pairs[:args.limit_pairs]
    print(f"Found {len(pairs)} brightfield/fluorescence pair(s)")
    segmenter = CellposeSegmenter(cfg)
    all_measurements, all_summaries = [], []
    for bf, fl in pairs:
        print(f"Analyzing {bf.name} + {fl.name}")
        measurements, summaries = analyze_pair(bf, fl, args.output, segmenter, cfg, args)
        all_measurements.extend(measurements)
        all_summaries.extend(summaries)
    write_csv(all_measurements, args.output / "measurements.csv")
    write_csv(all_summaries, args.output / "cell_summary.csv")
    print(f"Wrote {len(all_measurements)} contour samples and {len(all_summaries)} cells")


if __name__ == "__main__":
    main()
