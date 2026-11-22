"""`bsd`: the BS-Detector command line.

Movies (birth scars, poles, lineages, growth curves):

    bsd info    MOVIE                          what is in the file (positions, channels, pixel size, frames)
    bsd run     MOVIE --out RUN                prepare, segment, track and growth, in order
    bsd prepare MOVIE --out RUN                record the movie (resampled to 0.0647 um/px only if needed)
    bsd segment MOVIE --out RUN | RUN          Cellpose masks, resumable; --block K/N for one of N parallel jobs
    bsd track   RUN                            tracking, poles, scars -> measurements.csv, review.csv
    bsd growth  RUN [--hand WORKBOOK.xlsx]     apply corrections; complete cell cycles -> growth/

Checking and correcting:

    bsd review  RUN                            which cells and frames to check, and why (review.csv)
    bsd show    RUN --cell A01 [--frames 40-60]   draw the cell: lengths over time, scar and poles
    bsd fix     RUN --cell A01 --frames 52 --scar X Y | --lengths NEW OLD | --swap-poles | --exclude | --checked

Septin x curvature (paired brightfield + fluorescence ND2 stills):

    bsd septin run     INPUT_DIR --out RUN [--exclude LABELS.csv] [--min-length-um 8.5]
    bsd septin segment INPUT_DIR... --out RUN  one saved segmentation (RUN/masks.h5), so cell IDs never change
    bsd septin measure INPUT_DIR --out RUN     curvature + intensity along each outline
    bsd septin stats   RUN [--exclude ...]     association tests -> RUN/statistics/

Figures are saved to RUN/figures/ unless --no-figures (or --figures-dir DIR elsewhere). A
multi-position file gives one run folder per position inside --out. Every step skips work
already done, so rerunning a command after an interruption picks up where it stopped. See
docs/USER_GUIDE.md and docs/SEPTIN.md.
"""
import argparse
import re
import sys
from pathlib import Path


def log(msg):
    print(msg, flush=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def _slug(position):
    return 'position_' + re.sub(r'[^A-Za-z0-9]+', '_', str(position)).strip('_').lower().replace('position_', '')


def _movie_runs(a):
    """[(position, run_dir)] for a movie: one entry, or one per position in a multi-position file."""
    from .movie_io import positions
    if a.position is not None:
        return [(a.position, Path(a.out))]
    pos = positions(a.movie)
    if pos == [None]:
        return [(None, Path(a.out))]
    log(f'{Path(a.movie).name} has {len(pos)} positions; one run folder each under {a.out}/')
    return [(p, Path(a.out) / _slug(p)) for p in pos]


def _prepare(a, position, run_dir):
    from .workflow import prepare
    return prepare(a.movie, run_dir, position, a.channel, a.z, a.pixel_size, a.frame_minutes,
                   frames_dir=_frames_dir(a, run_dir), log=log)


def _frames_dir(a, run_dir):
    fd = getattr(a, 'frames_dir', None)
    return Path(fd) / Path(run_dir).name if fd else None


def _figures(a, run_dir):
    if getattr(a, 'no_figures', False):
        return None
    fd = getattr(a, 'figures_dir', None)
    return Path(fd) / Path(run_dir).name if fd else Path(run_dir) / 'figures'


def _block(a, n_frames):
    if getattr(a, 'frames', None):
        lo, _, hi = a.frames.partition(':')
        return int(lo or 0), int(hi) if hi else n_frames
    if getattr(a, 'block', None):
        k, _, n = a.block.partition('/')
        k, n = int(k), int(n)
        if not 0 <= k < n:
            raise SystemExit('--block K/N needs 0 <= K < N (K is the 0-based task number)')
        size = -(-n_frames // n)
        return k * size, min((k + 1) * size, n_frames)
    return 0, n_frames


def _segment(a, run_dir):
    from .workflow import Run, segment
    run = Run(run_dir)
    lo, hi = _block(a, run.info['frames']['n'])
    segment(run_dir, lo, hi, a.batch_size, a.allow_other_weights, log=log)


def _is_run(path):
    return (Path(path) / 'run.json').exists() or any(Path(path).glob('*/run.json'))


# ── movie commands ────────────────────────────────────────────────────────────

def cmd_info(a):
    from .movie_io import ChoiceNeeded, open_movie, positions, resample_plan
    pos = positions(a.movie)
    if pos != [None]:
        log(f'positions: {pos}')
    try:
        m = open_movie(a.movie, a.position if a.position is not None else (pos[0] if pos != [None] else None),
                       a.channel, a.z)
    except ChoiceNeeded as e:
        log(str(e))
        return
    try:
        log(f'format: {m.format}\nframes: {m.n_frames}\nsize: {m.shape[1]} x {m.shape[0]} px')
        log(f'pixel size: {m.um_per_px} um/px' if m.um_per_px else 'pixel size: not recorded (pass --pixel-size)')
        log(f'frame interval: {m.frame_minutes} min' if m.frame_minutes else 'frame interval: not recorded')
        if m.channel is not None:
            log(f'channel: {m.channel}')
        if m.um_per_px:
            shape, scale = resample_plan(m.shape, m.um_per_px)
            log('no resampling needed' if scale is None else f'will be resampled to {shape[1]} x {shape[0]} px at 0.0647 um/px')
    finally:
        m.close()


def cmd_prepare(a):
    for p, run_dir in _movie_runs(a):
        _prepare(a, p, run_dir)


def cmd_segment(a):
    if Path(a.movie).is_dir() and _is_run(a.movie):
        from .workflow import run_dirs
        for run_dir in run_dirs(a.movie):
            _segment(a, run_dir)
        return
    if not a.out:
        raise SystemExit('give --out RUN_FOLDER (or pass an existing run folder instead of a movie)')
    for p, run_dir in _movie_runs(a):
        _prepare(a, p, run_dir)
        _segment(a, run_dir)


def cmd_track(a):
    from .workflow import run_dirs, track
    for run_dir in run_dirs(a.run):
        track(run_dir, figures_dir=_figures(a, run_dir), log=log)


def cmd_growth(a):
    from .workflow import growth, run_dirs
    for run_dir in run_dirs(a.run):
        growth(run_dir, a.hand, a.hand_type, a.frame_minutes, a.cells, a.featured,
               figures_dir=_figures(a, run_dir), log=log)


def cmd_review(a):
    import pandas as pd
    from .review import write_review
    from .workflow import Run, run_dirs
    for run_dir in run_dirs(a.run):
        t = write_review(Run(run_dir), log=log)
        if not t.empty:
            cols = ['cell_name', 'in_growth_curves', 'frames_to_check', 'why', 'x', 'y', 'at_frame']
            with pd.option_context('display.width', 160, 'display.max_colwidth', 50):
                log(t[cols].head(a.top).to_string(index=False))
            log(f'(all cells: {Path(run_dir) / "review.csv"})')


def cmd_show(a):
    import matplotlib
    matplotlib.use('Agg')
    from .review import show_cell
    from .workflow import Run
    run = Run(a.run)
    fig = show_cell(run, a.cell, a.frames, a.max_panels)
    out = Path(a.out) if a.out else run.path / 'figures' / f'cell_{a.cell}.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    log(f'saved {out}')


def cmd_fix(a):
    from .corrections import add_edit, parse_frames
    from .workflow import Run
    run = Run(a.run)
    fields = {}
    if a.scar:
        fields.update(scar_x=a.scar[0], scar_y=a.scar[1])
    if a.lengths:
        fields.update(new_end_length_um=a.lengths[0], old_end_length_um=a.lengths[1])
    for flag in ('swap_poles', 'exclude', 'checked'):
        if getattr(a, flag):
            fields[flag] = 'yes'
    if not fields:
        raise SystemExit('say what to change: --scar X Y, --lengths NEW OLD, --swap-poles, --exclude or --checked')
    if a.frames is None and (a.scar or a.lengths):
        raise SystemExit('--scar and --lengths need --frames (one frame, or a list)')
    parse_frames(a.frames)                       # fail early on a typo
    row = add_edit(run, a.cell, a.frames or '', note=a.note or '', who=a.who or '', **fields)
    log(f'recorded in {run.corrections_dir / "edits.csv"}: ' + ', '.join(f'{k}={v}' for k, v in row.items() if v))
    log(f'run `bsd growth {run.path}` to apply it (or `bsd show {run.path} --cell {a.cell}` to look)')


def cmd_run(a):
    from .workflow import growth, track
    for p, run_dir in _movie_runs(a):
        _prepare(a, p, run_dir)
        _segment(a, run_dir)
        track(run_dir, figures_dir=_figures(a, run_dir), log=log)
        growth(run_dir, a.hand, a.hand_type, a.frame_minutes, figures_dir=_figures(a, run_dir), log=log)
        log(f'done: {run_dir}')


# ── septin commands ───────────────────────────────────────────────────────────

def _septin_masks(a):
    return Path(a.masks) if a.masks else Path(a.out) / 'masks.h5'


def cmd_septin_segment(a):
    from septin_curve_vis.segment_fields import segment
    segment([Path(d) for d in a.input], _septin_masks(a), log=log)


def _septin_figs(a, default):
    if getattr(a, 'no_figures', False):
        return None
    return a.figures_dir if getattr(a, 'figures_dir', None) else default


def cmd_septin_measure(a):
    from septin_curve_vis.run_analysis import measure
    masks = _septin_masks(a)
    if not masks.exists():
        raise SystemExit(f'{masks} not found; run `bsd septin segment {a.input} --out {a.out}` first, '
                         'or point --masks at an existing masks file')
    measure(Path(a.input), Path(a.out), masks, a.band_width, a.fluor_channel, a.smooth_factor,
            a.contour_points, a.limit_pairs, figures_dir=_septin_figs(a, 'output'), log=log)


def cmd_septin_stats(a):
    from septin_curve_vis.analyze_statistics import run
    run(Path(a.run), a.exclude, a.min_length_um, a.max_length_um, a.qc_threshold, a.primary_sigma,
        a.permutations, log=log, figures_dir=_septin_figs(a, 'out'))


def cmd_septin_run(a):
    from septin_curve_vis.analyze_statistics import run
    from septin_curve_vis.run_analysis import measure
    from septin_curve_vis.segment_fields import segment
    masks = _septin_masks(a)
    segment([Path(a.input)], masks, log=log)
    measure(Path(a.input), Path(a.out), masks, a.band_width, a.fluor_channel, a.smooth_factor,
            a.contour_points, a.limit_pairs, figures_dir=_septin_figs(a, 'output'), log=log)
    run(Path(a.out), a.exclude, a.min_length_um, a.max_length_um, a.qc_threshold, a.primary_sigma,
        a.permutations, log=log, figures_dir=_septin_figs(a, 'out'))
    log(f'done: {a.out}')


# ── parser ────────────────────────────────────────────────────────────────────

def _movie_args(p, out_required=True):
    p.add_argument('movie', help='.nd2, SlideBook .sldy, .tif or .h5 movie (read only)')
    p.add_argument('--out', required=out_required, help='run folder to write')
    p.add_argument('--position', help='position in a multi-position file (ND2: 1, 2, ...; SlideBook: name or number). '
                                      'Default: every position, each in its own folder')
    p.add_argument('--channel', help='brightfield channel (name or 0-based number) if the file has several')
    p.add_argument('--z', type=int, help='focal plane (0-based) if the file has a Z stack')
    p.add_argument('--pixel-size', type=float, help='um per pixel, if the file does not record it (or records it wrong)')
    p.add_argument('--frame-minutes', type=float, help='minutes between frames, if the file does not record it')
    p.add_argument('--frames-dir', help='where a converted copy of the movie goes, if one is needed '
                                        '(default RUN/working; e.g. a local disk when RUN is on Google Drive)')


def _seg_args(p):
    p.add_argument('--batch-size', type=int, default=8, help='GPU batch (speed/memory only; masks are the same)')
    p.add_argument('--allow-other-weights', action='store_true',
                   help='segment even if the Cellpose weights are not the validated cpsam weights')


def _figure_args(p):
    p.add_argument('--figures-dir', help='save figures here instead of RUN/figures')
    p.add_argument('--no-figures', action='store_true', help="don't draw figures (tables only)")


def _hand_args(p):
    p.add_argument('--hand', help="hand-tracked growth workbook (.xlsx) or converted CSV, to plot alongside")
    p.add_argument('--hand-type', help='which cells in --hand, e.g. y_wt (YE wild type); listed if omitted and ambiguous')


def _septin_args(p, stats=True, measure=True):
    p.add_argument('--figures-dir', help='save figures here instead of next to the tables')
    p.add_argument('--no-figures', action='store_true', help="don't draw figures (tables only)")
    if measure:
        p.add_argument('--masks', help='masks HDF5 (default RUN/masks.h5); point at an existing one to keep its cell IDs')
        p.add_argument('--fluor-channel', type=int, default=0, help='fluorescence channel (default 0, Spn3-GFP)')
        p.add_argument('--band-width', type=float, default=2.0, help='half-width (px) of the intensity band')
        p.add_argument('--smooth-factor', type=float, default=40.0, help='contour B-spline smoothing (benchmarked)')
        p.add_argument('--contour-points', type=int, default=300)
        p.add_argument('--limit-pairs', type=int, help='only the first N file pairs (quick test)')
    if stats:
        p.add_argument('--exclude', type=Path, help='CSV with cell_id (and optionally label) of cells to leave out, '
                                                    'e.g. the dividing-cell labels')
        p.add_argument('--min-length-um', type=float, help='also report only cells at least this long')
        p.add_argument('--max-length-um', type=float, help='also report only cells at most this long')
        p.add_argument('--qc-threshold', type=float, default=0.10, help='drop cells with max |curvature| above this')
        p.add_argument('--primary-sigma', type=float, default=3.0, help='smoothing, in contour samples')
        p.add_argument('--permutations', type=int, default=1000)


def parser():
    ap = argparse.ArgumentParser(prog='bsd', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True, metavar='COMMAND')

    p = sub.add_parser('info', help='what is in a movie file')
    p.add_argument('movie')
    p.add_argument('--position')
    p.add_argument('--channel')
    p.add_argument('--z', type=int)
    p.set_defaults(fn=cmd_info)

    p = sub.add_parser('run', help='prepare, segment, track and growth in one go')
    _movie_args(p)
    _seg_args(p)
    _hand_args(p)
    _figure_args(p)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser('prepare', help='read a movie into a run folder (once, before parallel segment jobs)')
    _movie_args(p)
    p.set_defaults(fn=cmd_prepare)

    p = sub.add_parser('segment', help='Cellpose masks for every frame (resumable)')
    _movie_args(p, out_required=False)
    _seg_args(p)
    g = p.add_mutually_exclusive_group()
    g.add_argument('--frames', help='only frames A:B (0-based, B excluded)')
    g.add_argument('--block', help='only block K of N, as K/N with K from 0 (e.g. $SLURM_ARRAY_TASK_ID/4)')
    p.set_defaults(fn=cmd_segment)

    p = sub.add_parser('track', help='tracking, poles and birth scars -> measurements.csv')
    p.add_argument('run', help='run folder (or a folder of per-position run folders)')
    _figure_args(p)
    p.set_defaults(fn=cmd_track)

    p = sub.add_parser('growth', help='growth curves of complete cell cycles')
    p.add_argument('run')
    _hand_args(p)
    p.add_argument('--frame-minutes', type=float, help='override the movie\'s frame interval')
    p.add_argument('--cells', nargs='+', help='only these complete cycles (default: all)')
    p.add_argument('--featured', nargs='+', help='cells for the per-cell panels (default: all plotted)')
    _figure_args(p)
    p.set_defaults(fn=cmd_growth)

    p = sub.add_parser('review', help='which cells and frames to check, and why')
    p.add_argument('run')
    p.add_argument('--top', type=int, default=20, help='how many cells to print (all are in review.csv)')
    p.set_defaults(fn=cmd_review)

    p = sub.add_parser('show', help='draw one cell: lengths over time, outline, scar and poles')
    p.add_argument('run')
    p.add_argument('--cell', required=True, help='lineage name, e.g. A01')
    p.add_argument('--frames', help='e.g. 52 or 40-60 (default: the frames to check, else evenly spaced)')
    p.add_argument('--max-panels', type=int, default=8)
    p.add_argument('--out', help='PNG to write (default RUN/figures/cell_<name>.png)')
    p.set_defaults(fn=cmd_show)

    p = sub.add_parser('fix', help='record a correction in RUN/corrections/edits.csv')
    p.add_argument('run')
    p.add_argument('--cell', required=True)
    p.add_argument('--frames', help='e.g. 52, 40-60, 40,42; omit for every frame of the cell')
    p.add_argument('--scar', nargs=2, type=float, metavar=('X', 'Y'),
                   help='scar midpoint in movie pixels (as Fiji shows them); end lengths are recomputed')
    p.add_argument('--lengths', nargs=2, type=float, metavar=('NEW_UM', 'OLD_UM'), help='your own end lengths in µm')
    p.add_argument('--swap-poles', action='store_true', help='the detector called the ends the wrong way round')
    p.add_argument('--exclude', action='store_true', help='leave these frames out of growth curves')
    p.add_argument('--checked', action='store_true', help='looked at; the detector is right')
    p.add_argument('--note')
    p.add_argument('--who')
    p.set_defaults(fn=cmd_fix)

    sp = sub.add_parser('septin', help='septin intensity vs outline curvature on paired ND2 stills')
    ssub = sp.add_subparsers(dest='septin_cmd', required=True, metavar='STEP')
    p = ssub.add_parser('run', help='segment, measure and stats in one go')
    p.add_argument('input', help='folder of brightfield/fluorescence ND2 pairs (read only)')
    p.add_argument('--out', required=True)
    _septin_args(p)
    p.set_defaults(fn=cmd_septin_run)
    p = ssub.add_parser('segment', help='one saved segmentation for one or more folders of ND2 pairs')
    p.add_argument('input', nargs='+')
    p.add_argument('--out', required=True)
    p.add_argument('--masks', help='masks HDF5 to add to (default RUN/masks.h5)')
    p.set_defaults(fn=cmd_septin_segment)
    p = ssub.add_parser('measure', help='curvature and intensity along each outline')
    p.add_argument('input')
    p.add_argument('--out', required=True)
    _septin_args(p, stats=False)
    p.set_defaults(fn=cmd_septin_measure)
    p = ssub.add_parser('stats', help='association tests (all cells, and filtered cells if asked)')
    p.add_argument('run', help='folder with measurements.csv and cell_summary.csv')
    _septin_args(p, measure=False)
    p.set_defaults(fn=cmd_septin_stats)
    return ap


def main(argv=None):
    import os
    a = parser().parse_args(argv)
    try:
        a.fn(a)
    except (FileNotFoundError, ValueError) as e:      # input problems: say what to fix, not a traceback
        if os.environ.get('BSD_DEBUG'):
            raise
        sys.exit(f'bsd: {e}\n(set BSD_DEBUG=1 for the full traceback)')


if __name__ == '__main__':
    main()
