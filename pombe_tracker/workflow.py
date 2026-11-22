"""The movie workflow behind `bsd`: one run folder per movie (and position).

    RUN/
      run.json                    input, pixel size, frame interval, versions, segmentation provenance, steps
      measurements.csv            detector output, one row per cell per frame (never edited)
      review.csv                  one row per cell: which frames to check, and why
      corrections/edits.csv       a scientist's fixes (also Fiji CSVs dropped in corrections/)
      measurements_corrected.csv  measurements with the fixes applied (written when there are fixes)
      growth/                     growth tables of complete cell cycles
      figures/                    QC and growth figures, only when asked for (--figures-dir)
      working/masks/0000.npz ...  one saved Cellpose mask per frame (resumable; needed to inspect cells)
      working/frames.h5           only when the movie had to be resampled (or --frames-dir elsewhere)

Steps, each resumable and each reading only what the previous one wrote:

    prepare  : read the movie; resample to 0.0647 um/px only if needed
    segment  : Cellpose-SAM masks for every frame (or a block of frames, for array jobs)
    track    : tracking, poles, birth scars, scar stabilisation -> measurements.csv, review.csv
    growth   : apply corrections, then complete cycles -> growth tables (and figures if asked)
"""
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import movie_io

REPO = Path(__file__).resolve().parent.parent
# Written next to the detector's own columns so cells can be drawn and corrected later.
EXTRA_COLUMNS = ['label', 'scar_end_1', 'scar_end_2']
DIRECT_FORMATS = ('nd2', 'tif', 'h5')    # read in place when no resampling is needed


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


class FrameSource:
    """Frames as the pipeline sees them, indexable like an array: frames[i] -> 2-D uint16."""

    def __init__(self, reader, n, close):
        self._read, self._n, self.close = reader, n, close

    def __len__(self):
        return self._n

    def __getitem__(self, i):
        if isinstance(i, slice):
            return np.stack([self._read(k) for k in range(*i.indices(self._n))])
        return np.asarray(self._read(int(i)))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


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
        old = self.path / 'masks'                  # run folders made before working/ existed
        return old if old.is_dir() and not (self.path / 'working' / 'masks').exists() else self.path / 'working' / 'masks'

    @property
    def measurements(self):
        return self.path / 'measurements.csv'

    @property
    def corrections_dir(self):
        return self.path / 'corrections'

    @property
    def frame_minutes(self):
        return self.info.get('frame_minutes')

    @property
    def um_per_px(self):
        return self.info.get('pixel_size_um', {}).get('pipeline') or movie_io.PIPELINE_UM_PER_PX

    @property
    def scale_yx(self):
        """Pipeline pixels per original-movie pixel, (sy, sx); (1, 1) if the movie was not resampled."""
        s = self.info.get('pixel_size_um', {}).get('scale_yx')
        return tuple(s) if s else (1.0, 1.0)

    def frames(self):
        """Open the frames the pipeline sees. A converted copy that has gone missing (e.g. a new Colab
        session) is rebuilt from the original movie."""
        f = self.info.get('frames')
        if f is None:
            raise FileNotFoundError(f'{self.path} has no prepared frames; run `bsd prepare MOVIE --out {self.path}` first')
        if f.get('copied'):
            path = Path(f['path'])
            path = path if path.is_absolute() else self.path / path
            if not path.exists():
                print(f'{path} is missing; rebuilding it from the movie')
                inp = self.info['input']
                m = movie_io.open_movie(inp['signature']['path'], inp['signature']['position'],
                                        inp['signature']['channel'], inp['signature']['z'],
                                        inp.get('um_per_px'), inp.get('frame_minutes'))
                try:
                    movie_io.write_frames(m, path)
                finally:
                    m.close()
            import h5py
            h = h5py.File(path, 'r')
            ds = h['frames']
            return FrameSource(lambda i: ds[i], ds.shape[0], h.close)
        inp = self.info['input']
        m = movie_io.open_movie(f['path'], inp['signature']['position'], inp['signature']['channel'],
                                inp['signature']['z'], inp.get('um_per_px'), inp.get('frame_minutes'))
        return FrameSource(m.read, m.n_frames, m.close)

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


def prepare(movie_path, run_path, position=None, channel=None, z=None, um_per_px=None, frame_minutes=None,
            frames_dir=None, log=print):
    """Record the movie in run.json. Frames are read from the movie itself unless it has to be resampled
    (or is in a format read slowly), in which case a copy goes to frames_dir (default RUN/working)."""
    run = Run(run_path)
    sig = _input_signature(movie_path, position, channel, z)
    if run.info.get('input', {}).get('signature') == sig:
        log(f'{run.path.name}: already prepared ({run.info["frames"]["n"]} frames)')
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
        direct = scale is None and movie.format in DIRECT_FORMATS and movie.read(0).dtype == np.uint16
        if direct:
            frames = dict(path=str(Path(movie_path).resolve()), n=movie.n_frames, shape=list(shape), copied=False)
            pix = dict(source=movie.um_per_px, pipeline=movie.um_per_px, resampled=False, scale_yx=None)
            log('  frames are read from the movie itself (no copy)')
        else:
            dest = (Path(frames_dir) if frames_dir else run.path / 'working') / 'frames.h5'
            info = movie_io.write_frames(movie, dest, log=log)
            frames = dict(path=str(dest.resolve()), n=movie.n_frames, shape=list(shape), copied=True)
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
    from .segmentation import segment_movie
    run = Run(run_path)
    with run.frames() as frames:
        n = segment_movie(frames, run.masks, start, stop, log=log, batch_size=batch_size,
                          allow_other_weights=allow_other_weights)
        missing = sum(not (run.masks / f'{i:04d}.npz').exists() for i in range(len(frames)))
    if missing == 0:
        run.step('segment')
    else:
        log(f'{missing} frames still have no mask (other blocks, or rerun to finish)')
    return n


def _qc_figures(results, cfg, out, log):
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
        fig.savefig(out / f'qc_frame_{fd["frame_idx"]:04d}.png', dpi=110, bbox_inches='tight')
        plt.close(fig)
    try:
        fig = plot_lineage_tree(results, cfg)
        fig.savefig(out / 'qc_lineage_tree.png', dpi=130, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:                       # a tree is a convenience, never a reason to fail the run
        log(f'  lineage tree not drawn: {e}')
    log(f'  QC figures in {out}')


def _add_export_fields(results):
    """Mask label and scar endpoints as plain [row, col] lists, for drawing and correcting cells later."""
    for fd in results:
        for r in fd['cells']:
            pts = r.get('scar_points')
            if pts is not None and len(pts) == 2 and pts[0] is not None:
                r['scar_end_1'] = [round(float(v), 3) for v in np.asarray(pts[0]).ravel()[:2]]
                r['scar_end_2'] = [round(float(v), 3) for v in np.asarray(pts[1]).ravel()[:2]]


def track(run_path, figures_dir=None, log=print):
    """Tracking, pole identity, birth scars and scar stabilisation from saved masks -> measurements.csv,
    plus review.csv. QC figures are drawn only if figures_dir is given."""
    from .config import Config
    from .io_utils import export_csv
    from .pipeline import run_pipeline
    from .postprocessing import stabilize_scars
    from .review import write_review
    from .corrections import ensure_edits_file
    from .segmentation import SavedMasks
    from .tracking import CellTracker
    run = Run(run_path)
    cfg = Config()
    if abs(run.um_per_px / cfg.PIXEL_SIZE_UM - 1) > movie_io.RESAMPLE_TOLERANCE:
        raise ValueError('frames are not at the pipeline pixel size; prepare the run again')
    t0 = time.time()
    with run.frames() as src:
        frames = src[:]
    masks = SavedMasks(run.masks, len(frames))
    log(f'{run.path.name}: tracking {len(frames)} frames')
    results = run_pipeline(frames, cfg, tracker=CellTracker(cfg), segmenter=masks)
    results, report = stabilize_scars(results, cfg)
    _add_export_fields(results)
    export_csv(results, run.measurements, columns=cfg.CSV_COLUMNS + EXTRA_COLUMNS, pixel_size_um=cfg.PIXEL_SIZE_UM)
    import pandas as pd
    d = pd.read_csv(run.measurements)
    summary = dict(rows=len(d), cells=int(d.cell_name.nunique()),
                   frames_needing_review=int(d.scar_needs_review.fillna(False).astype(bool).sum())
                   if 'scar_needs_review' in d else None)
    run.info['segmentation'] = [json.loads(s) for s in sorted(masks.provenance)]
    run.step('track', minutes=round((time.time() - t0) / 60, 1), **summary)
    log(f'  {summary["cells"]} tracked cells, {summary["rows"]} cell-frames -> {run.measurements}')
    ensure_edits_file(run)
    review = write_review(run, log=log)
    if figures_dir:
        _qc_figures(results, cfg, Path(figures_dir), log)
    return run


def growth(run_path, hand=None, hand_type=None, frame_minutes=None, cells=None, featured=None,
           figures_dir=None, log=print):
    """Apply any corrections, then growth tables of complete cycles -> RUN/growth/ (figures only if
    figures_dir is given). `hand` adds hand-tracked curves (.xlsx workbook or CSV)."""
    from .corrections import corrected_measurements
    from .growth import make_growth
    from .hand_tracking import load_reference
    run = Run(run_path)
    if not run.measurements.exists():
        raise FileNotFoundError(f'{run.measurements} not found; run `bsd track {run.path}` first')
    fm = frame_minutes or run.frame_minutes
    if fm is None:
        fm = 2.0
        log('  frame interval not recorded in the movie; assuming 2 min (set --frame-minutes)')
    d, applied = corrected_measurements(run, log=log)
    if applied:
        d = d[~d.excluded].drop(columns=['excluded', 'checked', 'corrected'])
    ref = load_reference(hand, hand_type, fm) if hand else None
    found, summary, rejected = make_growth(d, run.path / 'growth', cells, featured, fm, ref,
                                           fig_dir=Path(figures_dir) if figures_dir else None)
    run.step('growth', frame_minutes=fm, complete_cycles=found, rejected=rejected, corrections_applied=applied,
             hand=str(hand) if hand else None, hand_type=hand_type)
    return found
