"""Read brightfield time-lapse movies (.nd2, SlideBook .sldy, .tif, .h5) into the pipeline's frame layout.

The detector's settings are in pixels at 0.0647 um/px (Nikon Ti2, 100x/1.49, 1x1
binning). A movie at another pixel size is resampled to 0.0647 um/px, cubic and
centre-aligned, before segmentation, so the validated settings apply unchanged.
This is how a SlideBook movie at 0.11 um/px was analysed for validation
(validation/reports/neto_annotator_b/).

The source file is only read. Frames are written to an H5 file with one dataset,
`frames` (T, Y, X, uint16), chunked per frame:

    movie = open_movie('movie.nd2', position=1)
    write_frames(movie, 'run/frames.h5')

open_movie raises an error that lists the choices when a file holds several
positions or channels and none was picked.
"""
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PIPELINE_UM_PER_PX = 0.0647
RESAMPLE_TOLERANCE = 0.01          # resample if the pixel size differs from 0.0647 by more than 1%
BRIGHTFIELD_NAMES = ('bf', 'bright', 'dia', 'tl', 'trans', 'dic', 'phase', 'ph')


@dataclass
class Movie:
    """Frames of one position and channel. `read(i)` returns frame i as a 2-D array at the source pixel size."""
    path: Path
    format: str
    n_frames: int
    shape: tuple
    um_per_px: float = None
    frame_minutes: float = None
    position: object = None
    channel: object = None
    read: object = field(default=None, repr=False)
    close: object = field(default=lambda: None, repr=False)

    def describe(self):
        return dict(path=str(self.path), format=self.format, n_frames=self.n_frames, shape=list(self.shape),
                    um_per_px=self.um_per_px, frame_minutes=self.frame_minutes,
                    position=self.position, channel=self.channel)


class ChoiceNeeded(ValueError):
    """The file holds several positions/channels/planes and the caller has to pick one."""

    def __init__(self, what, choices, flag):
        self.what, self.choices, self.flag = what, choices, flag
        super().__init__(f'this file has {len(choices)} {what}s: {choices}. Pick one with {flag}')


# ── ND2 (Nikon NIS-Elements) ──────────────────────────────────────────────────

def nd2_positions(path):
    import nd2
    with nd2.ND2File(path) as f:
        return list(range(1, f.sizes.get('P', 1) + 1))


def _nd2_frame_minutes(f):
    try:
        for loop in f.experiment:
            if 'time' in type(loop).__name__.lower() or getattr(loop, 'type', '') in ('TimeLoop', 'NETimeLoop'):
                period = loop.parameters.periodMs
                if period:
                    return round(float(period) / 60000, 4)
    except Exception:
        pass
    return None


def _open_nd2(path, position=None, channel=None, z=None):
    import nd2
    f = nd2.ND2File(path)
    sizes = dict(f.sizes)
    axes = list(sizes)
    if not {'Y', 'X'} <= set(axes) or set(axes) - {'T', 'P', 'C', 'Z', 'Y', 'X'}:
        f.close()
        raise ValueError(f'unsupported ND2 layout {sizes}; expected T, Y, X with optional P, C, Z')
    pick = {}
    if sizes.get('P', 1) > 1:
        if position is None:
            f.close()
            raise ChoiceNeeded('position', list(range(1, sizes['P'] + 1)), '--position')
        if not 1 <= int(position) <= sizes['P']:
            f.close()
            raise ValueError(f'--position must be 1..{sizes["P"]}')
        pick['P'] = int(position) - 1
    names = None
    if 'C' in sizes:
        names = [c.channel.name for c in f.metadata.channels]
        if sizes['C'] > 1:
            if channel is None:
                bf = [i for i, n in enumerate(names) if any(k == w for w in re.split(r'[^a-z]+', n.lower()) for k in BRIGHTFIELD_NAMES)]
                if len(bf) != 1:
                    f.close()
                    raise ChoiceNeeded('channel', names, '--channel (name or number from 0)')
                channel = bf[0]
            if isinstance(channel, str) and not channel.isdigit():
                if channel not in names:
                    f.close()
                    raise ValueError(f'no channel {channel!r}; channels are {names}')
                channel = names.index(channel)
        pick['C'] = int(channel or 0)
    if sizes.get('Z', 1) > 1:
        if z is None:
            f.close()
            raise ChoiceNeeded('focal plane', list(range(sizes['Z'])), '--z')
        pick['Z'] = int(z)
    elif 'Z' in sizes:
        pick['Z'] = 0
    d = f.to_dask()
    n = sizes.get('T', 1)

    def read(i):
        idx = tuple(i if a == 'T' else pick.get(a, slice(None)) for a in axes)
        return np.asarray(d[idx])

    return Movie(Path(path), 'nd2', n, (sizes['Y'], sizes['X']), float(f.voxel_size().x),
                 _nd2_frame_minutes(f), position if 'P' in pick else None,
                 (names[pick['C']] if names else pick['C']) if 'C' in pick else None, read, f.close)


# ── SlideBook 7 (.sldy + .dir) ────────────────────────────────────────────────

def _sldy_dir(path):
    path = Path(path)
    if path.suffix == '.dir':
        return path
    d = path.with_suffix('.dir')
    if not d.is_dir():
        raise FileNotFoundError(f'SlideBook data folder {d} not found next to {path.name}')
    return d


def _yaml_field(text, key):
    return [v.strip() for v in re.findall(rf'^\s*{key}:\s*(.+)$', text, re.M)]


def sldy_positions(path):
    out = []
    for d in sorted(_sldy_dir(path).glob('*.imgdir')):
        rec = (d / 'ImageRecord.yaml').read_text(errors='replace')
        out += [n.replace('_#32;', ' ') for n in _yaml_field(rec, 'mName')[:1]]
    return out


def _open_sldy(path, position=None, channel=None, z=None):
    root = _sldy_dir(path)
    records = []
    for d in sorted(root.glob('*.imgdir')):
        rec = (d / 'ImageRecord.yaml').read_text(errors='replace')
        records.append((d, [n.replace('_#32;', ' ') for n in _yaml_field(rec, 'mName')], rec))
    names = [r[1][0] if r[1] else r[0].name for r in records]
    if position is None:
        if len(records) != 1:
            raise ChoiceNeeded('position', names, '--position (name, e.g. "Position 2", or number from 1)')
        hit = records[0]
    elif str(position).isdigit():
        hit = next((r for r in records if any(n.endswith(f'Position {position}') for n in r[1])), None)
    else:
        hit = next((r for r in records if any(n.endswith(str(position)) for n in r[1])), None)
    if hit is None:
        raise ValueError(f'no position {position!r}; positions are {names}')
    imgdir, _, rec = hit
    um = float(_yaml_field(rec, 'mMicronPerPixel')[0])
    ch = int(channel or 0)
    files = sorted(imgdir.glob(f'ImageData_Ch{ch}_TP*.npy'), key=lambda p: int(p.stem.split('TP')[1]))
    if not files:
        raise ValueError(f'no frames for channel {ch} in {imgdir}')
    for i, p in enumerate(files):
        if int(p.stem.split('TP')[1]) != i:
            raise ValueError(f'frame files are not numbered 0..{len(files) - 1}: {p.name}')
    first = np.load(files[0])

    def read(i):
        a = np.load(files[i])
        return a[int(z or 0)] if a.ndim == 3 else a

    return Movie(Path(path), 'sldy', len(files), first.shape[-2:], um, None, hit[1][0] if hit[1] else position, ch, read)


# ── TIFF and H5 ───────────────────────────────────────────────────────────────

def _tif_pixel_size(tf):
    try:
        page = tf.pages[0]
        ij = tf.imagej_metadata or {}
        unit = str(ij.get('unit', '')).lower().replace('\\u00b5', 'u')
        if unit in ('micron', 'um', 'µm', 'microns', 'u'):
            num, den = page.tags['XResolution'].value
            return den / num
    except Exception:
        pass
    return None


def _open_tif(path, position=None, channel=None, z=None):
    import tifffile
    tf = tifffile.TiffFile(path)
    arr = tf.asarray(out='memmap') if tf.pages[0].is_memmappable else tf.asarray()
    if arr.ndim == 2:
        arr = arr[None]
    if arr.ndim != 3:
        tf.close()
        raise ValueError(f'expected a (T, Y, X) or (Y, X) TIFF, got shape {arr.shape}; save one channel and plane per file')
    ij = tf.imagej_metadata or {}
    fm = float(ij['finterval']) / 60 if ij.get('finterval') else None
    return Movie(Path(path), 'tif', arr.shape[0], arr.shape[1:], _tif_pixel_size(tf), fm, None, None,
                 lambda i: np.asarray(arr[i]), tf.close)


def h5_frames_key(f):
    if 'frames' in f:
        return 'frames'
    keys = [k for k in f if getattr(f[k], 'ndim', 0) == 3]
    if len(keys) != 1:
        raise ValueError(f'no "frames" dataset and {len(keys)} other 3-D datasets ({keys})')
    return keys[0]


def _open_h5(path, position=None, channel=None, z=None):
    import h5py
    f = h5py.File(path, 'r')
    ds = f[h5_frames_key(f)]
    um = next((float(f.attrs[k]) for k in ('um_per_px', 'pixel_size_um') if k in f.attrs), None)
    if um is None:
        # H5 is the pipeline's own format; the lab's older H5 files carry no pixel size and are all
        # Nikon Ti2 100x frames at 0.0647 um/px.
        print(f'{Path(path).name} records no pixel size; assuming {PIPELINE_UM_PER_PX} um/px '
              '(pass --pixel-size if that is wrong)')
        um = PIPELINE_UM_PER_PX
    fm = float(f.attrs['frame_minutes']) if 'frame_minutes' in f.attrs else None
    return Movie(Path(path), 'h5', ds.shape[0], ds.shape[1:], um, fm, None, None, lambda i: ds[i], f.close)


OPENERS = {'.nd2': _open_nd2, '.sldy': _open_sldy, '.dir': _open_sldy, '.tif': _open_tif, '.tiff': _open_tif,
           '.h5': _open_h5, '.hdf5': _open_h5}


def open_movie(path, position=None, channel=None, z=None, um_per_px=None, frame_minutes=None):
    """Open a movie for reading. um_per_px / frame_minutes override (or supply) the file's metadata."""
    path = Path(path)
    opener = OPENERS.get(path.suffix.lower())
    if opener is None:
        raise ValueError(f'unsupported file type {path.suffix!r}; use .nd2, .sldy, .tif or .h5')
    if not path.exists():
        raise FileNotFoundError(path)
    m = opener(path, position, channel, z)
    if um_per_px is not None:
        m.um_per_px = float(um_per_px)
    if frame_minutes is not None:
        m.frame_minutes = float(frame_minutes)
    return m


def positions(path):
    """Positions in a multi-position file (1-based numbers for ND2, names for SlideBook), or [None]."""
    suffix = Path(path).suffix.lower()
    if suffix == '.nd2':
        p = nd2_positions(path)
        return p if len(p) > 1 else [None]
    if suffix in ('.sldy', '.dir'):
        p = sldy_positions(path)
        return p if len(p) > 1 else [None]
    return [None]


# ── Resampling and writing ────────────────────────────────────────────────────

def resample_plan(shape, um_per_px, target=PIPELINE_UM_PER_PX):
    """(out_shape, (sy, sx)) to bring a frame to `target` um/px, or (shape, None) if within tolerance."""
    if um_per_px is None:
        raise ValueError('the file does not record its pixel size; give it with --pixel-size (um per pixel)')
    if abs(um_per_px / target - 1) <= RESAMPLE_TOLERANCE:
        return tuple(shape), None
    s = um_per_px / target
    oh, ow = int(round(shape[0] * s)), int(round(shape[1] * s))
    return (oh, ow), (oh / shape[0], ow / shape[1])


def resample(frame, scale):
    """Cubic, centre-aligned (x_out = (x_in + 0.5) * s - 0.5), as for the validated SlideBook movie."""
    from scipy.ndimage import zoom
    z = zoom(np.asarray(frame, dtype=np.float32), scale, order=3, mode='nearest', grid_mode=True)
    return np.clip(np.rint(z), 0, 65535).astype(np.uint16)


def write_frames(movie, out, target=PIPELINE_UM_PER_PX, log=print):
    """Write `movie` to `out` (H5, dataset 'frames', uint16), resampled if needed. Returns a description dict.

    Writes OUT.partial first and renames it, so a half-written file is never taken for a finished one.
    """
    import h5py
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shape, scale = resample_plan(movie.shape, movie.um_per_px, target)
    if scale:
        log(f'pixel size {movie.um_per_px:.4f} um/px -> resampling {movie.shape[1]}x{movie.shape[0]} to '
            f'{shape[1]}x{shape[0]} at {target} um/px')
    tmp = out.with_suffix('.partial')
    t0 = time.time()
    with h5py.File(tmp, 'w') as h:
        ds = h.create_dataset('frames', shape=(movie.n_frames, *shape), dtype='uint16',
                              chunks=(1, min(512, shape[0]), min(512, shape[1])), compression='gzip')
        for i in range(movie.n_frames):
            a = movie.read(i)
            if scale:
                a = resample(a, scale)
            elif a.dtype != np.uint16:
                if np.issubdtype(a.dtype, np.floating) or a.min() < 0 or a.max() > 65535:
                    raise ValueError(f'frame {i} is {a.dtype} with values outside uint16; convert the movie first')
                a = a.astype(np.uint16)
            ds[i] = a
            if i % 25 == 0 or i == movie.n_frames - 1:
                log(f'  frame {i + 1}/{movie.n_frames}  {time.time() - t0:.0f} s')
        info = dict(source=movie.describe(), um_per_px=target if scale else movie.um_per_px,
                    resampled=bool(scale), scale_yx=list(scale) if scale else None,
                    resample='scipy.ndimage.zoom order=3 grid_mode=True mode=nearest' if scale else None)
        h.attrs['um_per_px'] = info['um_per_px']
        if movie.frame_minutes:
            h.attrs['frame_minutes'] = movie.frame_minutes
        h.attrs['bsd_source'] = json.dumps(info)
    tmp.replace(out)
    return info
