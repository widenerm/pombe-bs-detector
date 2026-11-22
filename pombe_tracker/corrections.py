"""Scientist corrections to the detector's measurements.

measurements.csv is the detector's output and is never edited. Fixes live in CSV files in
RUN/corrections/ and are applied on top of it, giving measurements_corrected.csv. Two file
layouts are read, in file-name order, rows in order (a later row wins):

1. edits.csv (written by `bsd fix`, the Colab notebook, or by hand). One row per fix:

   cell_name           lineage name, e.g. A01
   frames              e.g. 52, 40-60, 40,42,50-55, or empty for all of the cell's frames
   scar_x, scar_y      where the birth-scar midpoint really is, in the ORIGINAL movie's pixels
                       (what Fiji shows); both end lengths are recomputed from it and the detector's poles
   new_end_length_um,  your own measurements (e.g. Fiji line tool); they replace the detector's
   old_end_length_um
   swap_poles          yes: the detector called the ends the wrong way round
   exclude             yes: leave these frames out of growth curves (bad outline, wrong cell, ...)
   checked             yes: looked at, the detector is right (removes the frames from review.csv)
   note, who, when     free text

2. CSVs from the lab's Fiji macro (fiji/BS_check_cell.ijm, same layout as the tracking packet):
   one row per clicked frame with scar_x/y and optionally new_pole_x/y and old_pole_x/y. If both poles
   were clicked they replace the detector's, and whole-cell length becomes the pole-to-pole distance.
   Type the BS-Detector cell name (e.g. A01) as the macro's cell label.
"""
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

EDIT_COLUMNS = ['cell_name', 'frames', 'scar_x', 'scar_y', 'new_end_length_um', 'old_end_length_um',
                'swap_poles', 'exclude', 'checked', 'note', 'who', 'when']
FIJI_COLUMNS = {'cell_id', 'frame_index', 'scar_x', 'scar_y'}
YES = {'yes', 'y', 'true', '1', 'x'}


# ── small helpers ─────────────────────────────────────────────────────────────

def parse_frames(text):
    """'52' -> [52]; '40-60' -> [40..60]; '40,42,50-55' -> ...; '' or 'all' -> None (every frame)."""
    text = '' if text is None or (isinstance(text, float) and np.isnan(text)) else str(text).strip()
    if text.lower() in ('', 'all', 'nan'):
        return None
    out = set()
    for part in text.replace(' ', '').split(','):
        if not part:
            continue
        if '-' in part:
            lo, hi = (int(float(v)) for v in part.split('-', 1))
            out.update(range(min(lo, hi), max(lo, hi) + 1))
        else:
            out.add(int(float(part)))
    return sorted(out)


def frames_text(frames):
    """[40, 41, 42, 50] -> '40-42, 50'."""
    frames = sorted(set(int(f) for f in frames))
    parts, i = [], 0
    while i < len(frames):
        j = i
        while j + 1 < len(frames) and frames[j + 1] == frames[j] + 1:
            j += 1
        parts.append(str(frames[i]) if i == j else f'{frames[i]}-{frames[j]}')
        i = j + 1
    return ', '.join(parts)


def _yes(v):
    return str(v).strip().lower() in YES if v is not None and not (isinstance(v, float) and np.isnan(v)) else False


def _num(v):
    try:
        v = float(v)
        return None if np.isnan(v) else v
    except (TypeError, ValueError):
        return None


def to_pipeline(run, x, y):
    """Original-movie pixel (x, y) -> pipeline (row, col). Centre-aligned, like the resampling."""
    sy, sx = run.scale_yx
    return np.array([(y + 0.5) * sy - 0.5, (x + 0.5) * sx - 0.5])


def to_movie(run, row, col):
    """Pipeline (row, col) -> original-movie pixel (x, y)."""
    sy, sx = run.scale_yx
    return (col + 0.5) / sx - 0.5, (row + 0.5) / sy - 0.5


def point(v):
    """'[row, col]' from measurements.csv -> np.array, or None."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (list, tuple, np.ndarray)):
        return np.asarray(v, float)
    try:
        return np.asarray(json.loads(str(v)), float)
    except (ValueError, TypeError):
        return None


def _fmt(p):
    return json.dumps([round(float(p[0]), 4), round(float(p[1]), 4)])


# ── reading and writing correction files ─────────────────────────────────────

def ensure_edits_file(run):
    """corrections/edits.csv with just the header, so there is an obvious place to type fixes."""
    path = run.corrections_dir / 'edits.csv'
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(','.join(EDIT_COLUMNS) + '\n')
    return path


def add_edit(run, cell_name, frames='', **fields):
    """Append one fix to corrections/edits.csv. fields: any of EDIT_COLUMNS."""
    unknown = set(fields) - set(EDIT_COLUMNS)
    if unknown:
        raise ValueError(f'unknown correction fields {sorted(unknown)}; use {EDIT_COLUMNS}')
    path = ensure_edits_file(run)
    row = {c: '' for c in EDIT_COLUMNS}
    row.update({k: ('' if v is None else v) for k, v in fields.items()})
    row.update(cell_name=cell_name, frames=frames if isinstance(frames, str) else frames_text(frames),
               when=row['when'] or datetime.now().isoformat(timespec='minutes'))
    with path.open('a', newline='') as f:
        csv.DictWriter(f, fieldnames=EDIT_COLUMNS).writerow(row)
    return row


def read_corrections(run):
    """Every fix in RUN/corrections/*.csv, as dicts in a common layout, in the order they apply."""
    rows = []
    folder = run.corrections_dir
    if not folder.exists():
        return rows
    for path in sorted(folder.glob('*.csv')):
        t = pd.read_csv(path, dtype=str, keep_default_na=False)
        if t.empty:
            continue
        cols = set(t.columns)
        for i, r in t.iterrows():
            src = f'{path.name} row {i + 2}'
            if FIJI_COLUMNS <= cols:
                fx = dict(source=src, cell_name=str(r['cell_id']).strip(), frames=[int(float(r['frame_index']))],
                          scar=(_num(r.get('scar_x')), _num(r.get('scar_y'))),
                          new_pole=(_num(r.get('new_pole_x')), _num(r.get('new_pole_y'))),
                          old_pole=(_num(r.get('old_pole_x')), _num(r.get('old_pole_y'))),
                          note=r.get('notes', ''), who=r.get('scorer', ''))
                if None in fx['scar'] and (None in fx['new_pole'] or None in fx['old_pole']):
                    continue                    # nothing clicked in this frame
                rows.append(fx)
            elif 'cell_name' in cols:
                if not str(r['cell_name']).strip():
                    continue
                rows.append(dict(source=src, cell_name=str(r['cell_name']).strip(), frames=parse_frames(r.get('frames')),
                                 scar=(_num(r.get('scar_x')), _num(r.get('scar_y'))),
                                 new_end=_num(r.get('new_end_length_um')), old_end=_num(r.get('old_end_length_um')),
                                 swap=_yes(r.get('swap_poles')), exclude=_yes(r.get('exclude')),
                                 checked=_yes(r.get('checked')), note=r.get('note', ''), who=r.get('who', '')))
            else:
                raise ValueError(f'{path}: not a corrections file (needs cell_name, or the Fiji macro columns)')
    return rows


# ── applying them ─────────────────────────────────────────────────────────────

def apply_corrections(d, fixes, run):
    """Return (corrected copy of d, number of fixes applied, list of problems).

    Adds columns: corrected (what changed, ';'-separated), excluded, checked.
    """
    d = d.copy()
    d['corrected'] = ''
    d['excluded'] = False
    d['checked'] = False
    px = run.um_per_px
    problems, applied = [], 0
    for fx in fixes:
        rows = d.index[d.cell_name.astype(str) == fx['cell_name']]
        if len(rows) == 0:
            problems.append(f'{fx["source"]}: no cell named {fx["cell_name"]!r}')
            continue
        if fx['frames'] is not None:
            want = set(fx['frames'])
            rows = rows[d.loc[rows, 'frame'].isin(want)]
            missing = want - set(d.loc[rows, 'frame'])
            if missing:
                problems.append(f'{fx["source"]}: {fx["cell_name"]} has no frame(s) {frames_text(missing)}')
            if len(rows) == 0:
                continue
        what = []
        for i in rows:
            new_p, old_p = point(d.at[i, 'new_pole_point']), point(d.at[i, 'old_pole_point'])
            if fx.get('exclude'):
                d.at[i, 'excluded'] = True
            if fx.get('swap'):
                for a, b in (('new_end_length', 'old_end_length'), ('new_end_length_um', 'old_end_length_um'),
                             ('new_pole_point', 'old_pole_point')):
                    d.at[i, a], d.at[i, b] = d.at[i, b], d.at[i, a]
                new_p, old_p = old_p, new_p
            npole, opole = fx.get('new_pole', (None, None)), fx.get('old_pole', (None, None))
            if None not in npole and None not in opole:
                new_p, old_p = to_pipeline(run, *npole), to_pipeline(run, *opole)
                d.at[i, 'new_pole_point'], d.at[i, 'old_pole_point'] = _fmt(new_p), _fmt(old_p)
                d.at[i, 'length'] = float(np.linalg.norm(new_p - old_p))
                d.at[i, 'length_um'] = round(d.at[i, 'length'] * px, 4)
            if None not in fx['scar']:
                scar = to_pipeline(run, *fx['scar'])
                if new_p is None or old_p is None:
                    problems.append(f'{fx["source"]}: {fx["cell_name"]} frame {d.at[i, "frame"]} has no poles to measure from')
                else:
                    d.at[i, 'scar_midpoint'] = _fmt(scar)
                    d.at[i, 'new_end_length'] = float(np.linalg.norm(scar - new_p))
                    d.at[i, 'old_end_length'] = float(np.linalg.norm(scar - old_p))
                    d.at[i, 'new_end_length_um'] = round(d.at[i, 'new_end_length'] * px, 4)
                    d.at[i, 'old_end_length_um'] = round(d.at[i, 'old_end_length'] * px, 4)
                    d.at[i, 'scar_detected'] = True
                    d.at[i, 'scar_source'] = 'manual'
            for col in ('new_end', 'old_end'):
                v = fx.get(col)
                if v is not None:
                    d.at[i, f'{col}_length_um'] = v
                    d.at[i, f'{col}_length'] = v / px
            if fx.get('checked'):
                d.at[i, 'checked'] = True
        for flag, word in (('exclude', 'excluded'), ('swap', 'ends swapped'), ('checked', 'checked')):
            if fx.get(flag):
                what.append(word)
        if None not in fx.get('new_pole', (None,)) and None not in fx.get('old_pole', (None,)):
            what.append('poles moved')
        if None not in fx['scar']:
            what.append('scar moved')
        if fx.get('new_end') is not None or fx.get('old_end') is not None:
            what.append('lengths entered')
        label = ', '.join(what) + (f' ({fx["note"]})' if fx.get('note') else '') + f' [{fx["source"]}]'
        for i in rows:
            d.at[i, 'corrected'] = (d.at[i, 'corrected'] + '; ' if d.at[i, 'corrected'] else '') + label
        applied += 1
    return d, applied, problems


def corrected_measurements(run, log=print):
    """measurements.csv with every correction applied. Writes measurements_corrected.csv if there are
    any; returns (table with corrected/excluded/checked columns, number of fixes applied)."""
    d = pd.read_csv(run.measurements)
    fixes = read_corrections(run)
    d, applied, problems = apply_corrections(d, fixes, run)
    for p in problems:
        log(f'  correction not applied: {p}')
    out = run.path / 'measurements_corrected.csv'
    if applied:
        d.to_csv(out, index=False)
        changed = d.corrected.ne('')
        log(f'  {applied} correction(s) applied to {int(changed.sum())} cell-frames '
            f'({int(d.excluded.sum())} excluded) -> {out.name}')
    elif out.exists():
        out.unlink()                       # corrections were removed: don't leave a stale corrected table
    return d, applied
