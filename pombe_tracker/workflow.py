"""The movie workflow behind `bsd`: one run folder per movie (and position).

    RUN/
      run.json           input, pixel size, frame interval, versions, segmentation provenance, step times
      frames.h5          the frames the pipeline sees (absent if the input was already a 0.0647 um/px H5)
      masks/0000.npz ... one saved Cellpose mask per frame (resumable)
      measurements.csv   one row per cell per frame (lengths in px and um)
      qc/                frame overviews and the lineage tree, for a quick visual check
      growth/            growth curves of complete cell cycles

Steps, each resumable and each reading only what the previous one wrote:

    prepare  : read the movie, resample to 0.0647 um/px if needed, write frames.h5 and run.json
    segment  : Cellpose-SAM masks for every frame (or a block of frames, for array jobs)
    track    : tracking, poles, birth scars, scar stabilisation -> measurements.csv
    growth   : complete cycles -> growth figures and tables
"""
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import movie_io

REPO = Path(__file__).resolve().parent.parent


def _git_commit():
    """Commit of the code that ran: from a git checkout, or from pip's record of a git install."""
    if (REPO / '.git').exists():
        try:
            out = subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=5)
            if out.returncode == 0:
                dirty = subprocess.run(['git', '-C', str(REPO), 'status', '--porcelain', '--untracked-files=no'],
                                       capture_output=True, text=True, timeout=5).stdout.strip()
                return out.stdout.strip() + ('-dirty' if dirty else '')
        except Exception:
            pass
    try:                                  # pip install git+https://...  records the commit in direct_url.json
        from importlib.metadata import distribution
        info = json.loads(distribution('bs-detector').read_text('direct_url.json') or '{}')
        return info.get('vcs_info', {}).get('commit_id')
    except Exception:
        return None


def _version():
    try:
        from importlib.metadata import version
        return version('bs-detector')
    except Exception:
        return None


class Run:
    """A run folder and its run.json."""

    def __init__(self, path):
        self.path = Path(path)
        self.json_path = self.path / 'run.json'
        self.info = json.loads(self.json_path.read_text()) if self.json_path.exists() else {}

    def save(self):
        self.path.mkdir(parents=True, exist_ok=True)
        tmp = self.json_path.with_suffix('.partial')
        tmp.write_text(json.dumps(self.info, indent=1, default=str))
        tmp.replace(self.json_path)

    @property
    def masks(self):
        return self.path / 'masks'

    @property
    def measurements(self):
        return self.path / 'measurements.csv'

    @property
    def frames_path(self):
        p = self.info.get('frames', {}).get('path')
        if p is None:
            raise FileNotFoundError(f'{self.path} has no prepared frames; run `bsd segment MOVIE --out {self.path}` first')
        p = Path(p)
        return p if p.is_absolute() else self.path / p

    @property
    def frame_minutes(self):
        return self.info.get('frame_minutes')

    def step(self, name, **details):
        self.info.setdefault('steps', {})[name] = dict(
            finished=datetime.now(timezone.utc).isoformat(timespec='seconds'), bsd=_version(), git_commit=_git_commit(),
            **details)
        self.save()


def run_dirs(path):
    """A run folder, or every run folder directly inside `path` (one per position)."""
    path = Path(path)
    if (path / 'run.json').exists():
        return [path]
    subs = sorted(p.parent for p in path.glob('*/run.json'))
    if not subs:
        raise FileNotFoundError(f'{path} is not a run folder (no run.json); run `bsd segment MOVIE --out {path}` first')
    return subs


def _input_signature(movie_path, position, channel, z):
    st = Path(movie_path).stat()
    return dict(path=str(Path(movie_path).resolve()), size_bytes=st.st_size, mtime=int(st.st_mtime),
                position=position, channel=channel, z=z)


def prepare(movie_path, run_path, position=None, channel=None, z=None, um_per_px=None, frame_minutes=None, log=print):
    """Read the movie into RUN/frames.h5 (resampled to 0.0647 um/px if needed). Skipped if already done."""
    run = Run(run_path)
    sig = _input_signature(movie_path, position, channel, z)
    if run.info.get('input', {}).get('signature') == sig and run.frames_path.exists():
        log(f'{run.path.name}: frames already prepared ({run.info["frames"]["n"]} frames)')
        return run
    if run.info.get('input') and run.info['input'].get('signature', {}).get('path') != sig['path']:
        raise ValueError(f'{run.path} was made from {run.info["input"]["signature"]["path"]}; '
                         f'use a new --out folder for a different movie')
    movie = movie_io.open_movie(movie_path, position, channel, z, um_per_px, frame_minutes)
    try:
        shape, scale = movie_io.resample_plan(movie.shape, movie.um_per_px)
        log(f'{Path(movie_path).name}: {movie.n_frames} frames, {movie.shape[1]}x{movie.shape[0]} px, '
            f'{movie.um_per_px:.4f} um/px, '
            + (f'{movie.frame_minutes:g} min/frame' if movie.frame_minutes else 'frame interval not recorded'))
        reuse = movie.format == 'h5' and scale is None
        if reuse:
            import h5py
            with h5py.File(movie_path, 'r') as f:
                reuse = movie_io.h5_frames_key(f) == 'frames' and f['frames'].dtype == np.uint16
        if reuse:
            frames = dict(path=str(Path(movie_path).resolve()), n=movie.n_frames, shape=list(shape), copied=False)
            pix = dict(source=movie.um_per_px, pipeline=movie.um_per_px, resampled=False, scale_yx=None)
            log('  using the H5 file directly (already 0.0647 um/px)')
        else:
            info = movie_io.write_frames(movie, run.path / 'frames.h5', log=log)
            frames = dict(path='frames.h5', n=movie.n_frames, shape=list(shape), copied=True)
            pix = dict(source=movie.um_per_px, pipeline=info['um_per_px'], resampled=info['resampled'],
                       scale_yx=info['scale_yx'], method=info['resample'])
        run.info.update(
            created=run.info.get('created') or datetime.now(timezone.utc).isoformat(timespec='seconds'),
            input=dict(signature=sig, **movie.describe()), frames=frames, pixel_size_um=pix,
            frame_minutes=movie.frame_minutes,
            environment=dict(python=platform.python_version(), platform=platform.platform()))
        run.step('prepare')
    finally:
        movie.close()
    return run


def segment(run_path, start=0, stop=None, batch_size=8, allow_other_weights=False, log=print):
    """Cellpose masks for frames [start, stop) of a prepared run."""
    import h5py
    from .segmentation import segment_movie
    run = Run(run_path)
    with h5py.File(run.frames_path, 'r') as f:
        frames = f['frames']
        n = segment_movie(frames, run.masks, start, stop, log=log, batch_size=batch_size,
                          allow_other_weights=allow_other_weights)
        missing = sum(not (run.masks / f'{i:04d}.npz').exists() for i in range(frames.shape[0]))
    if missing == 0:
        run.step('segment')
    else:
        log(f'{missing} frames still have no mask (other blocks, or rerun to finish)')
    return n


def _qc(results, cfg, out, log):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .visualization import plot_frame_overview, plot_lineage_tree
    out.mkdir(parents=True, exist_ok=True)
    n = len(results)
    picks = sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1}) if n else []
    for i in picks:
        fd = results[i]
        fig = plot_frame_overview(fd['frame'], fd['cells'], fd['frame_idx'], cfg)
        fig.savefig(out / f'frame_{fd["frame_idx"]:04d}.png', dpi=110, bbox_inches='tight')
        plt.close(fig)
    try:
        fig = plot_lineage_tree(results, cfg)
        fig.savefig(out / 'lineage_tree.png', dpi=130, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:                       # a tree is a convenience, never a reason to fail the run
        log(f'  lineage tree not drawn: {e}')
    log(f'  QC figures in {out}')


def track(run_path, qc=True, log=print):
    """Tracking, pole identity, birth scars and scar stabilisation from saved masks -> measurements.csv."""
    import h5py
    from .config import Config
    from .io_utils import export_csv
    from .pipeline import run_pipeline
    from .postprocessing import stabilize_scars
    from .segmentation import SavedMasks
    from .tracking import CellTracker
    run = Run(run_path)
    cfg = Config()
    if abs(run.info['pixel_size_um']['pipeline'] / cfg.PIXEL_SIZE_UM - 1) > movie_io.RESAMPLE_TOLERANCE:
        raise ValueError('frames are not at the pipeline pixel size; prepare the run again')
    t0 = time.time()
    with h5py.File(run.frames_path, 'r') as f:
        frames = f['frames'][:]
    masks = SavedMasks(run.masks, len(frames))
    log(f'{run.path.name}: tracking {len(frames)} frames')
    results = run_pipeline(frames, cfg, tracker=CellTracker(cfg), segmenter=masks)
    results, report = stabilize_scars(results, cfg)
    export_csv(results, run.measurements, columns=cfg.CSV_COLUMNS, pixel_size_um=cfg.PIXEL_SIZE_UM)
    import pandas as pd
    d = pd.read_csv(run.measurements)
    summary = dict(rows=len(d), cells=int(d.cell_name.nunique()),
                   frames_needing_review=int(d.scar_needs_review.fillna(False).astype(bool).sum())
                   if 'scar_needs_review' in d else None)
    run.info['segmentation'] = [json.loads(s) for s in sorted(masks.provenance)]
    run.step('track', minutes=round((time.time() - t0) / 60, 1), **summary)
    log(f'  {summary["cells"]} tracked cells, {summary["rows"]} cell-frames -> {run.measurements}')
    if qc:
        _qc(results, cfg, run.path / 'qc', log)
    return run


def growth(run_path, hand=None, hand_type=None, frame_minutes=None, cells=None, featured=None, log=print):
    """Growth curves of complete cycles -> RUN/growth/. `hand` adds hand-tracked curves (.xlsx workbook or CSV)."""
    from .growth import make_growth
    from .hand_tracking import load_reference
    run = Run(run_path)
    if not run.measurements.exists():
        raise FileNotFoundError(f'{run.measurements} not found; run `bsd track {run.path}` first')
    fm = frame_minutes or run.frame_minutes
    if fm is None:
        fm = 2.0
        log('  frame interval not recorded in the movie; assuming 2 min (set --frame-minutes)')
    ref = load_reference(hand, hand_type, fm) if hand else None
    found, summary, rejected = make_growth(run.measurements, run.path / 'growth', cells, featured, fm, ref)
    run.step('growth', frame_minutes=fm, complete_cycles=found, rejected=rejected,
             hand=str(hand) if hand else None, hand_type=hand_type)
    return found
