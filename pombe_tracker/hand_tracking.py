"""Turn the lab's hand-tracked growth workbook into the long table growth.py reads.

Each sheet is one cell, named like "(ye) rgf1DEL gpd1DEL cell 5" or "(s)wt cell 2"
((ye) = YE, (s) = sorbitol). It holds three blocks of ImageJ line measurements,
old end, new end and whole cell, each with a frame column and a length column (µm).
The sheets are laid out differently, so blocks are found by their labels and the
length column by its header. Growth since birth is length minus the first frame's
length (the workbook's "change in l to og"); time is (frame - first frame) x frame minutes.

    python -m pombe_tracker.hand_tracking WORKBOOK.xlsx OUT.csv [--frame-minutes 2]

`bsd growth RUN --hand WORKBOOK.xlsx --hand-type TYPE` does this conversion itself.

Writes TYPE (y_/s_ + strain), CELLNUM, END (TOT/OE/NE), LENGTH (growth since
birth), TIME, plus ABS_LENGTH, FRAME and SHEET for tracing values back. Needs
openpyxl. The workbook is only read.

Sheets number frames from 1 per cell and carry no positions, so a comparison with
the pipeline is between populations of cells, not matched cells.
"""
import argparse
import re
from pathlib import Path

import pandas as pd

ENDS = (('OE', ('old end',)), ('NE', ('new end',)), ('TOT', ('whole cell', 'total len', 'total cell')))
LENGTH_WORDS = ('length', 'leangth', 'lemgth', 'lenght')     # header spellings seen in the workbook


def _text(v):
    return v.strip().lower() if isinstance(v, str) else ''


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def sheet_type(name):
    """'(ye) rgf1DEL gpd1DEL cell 5' -> ('y_rgf1delgpd1del', 5)."""
    n = name.lower().replace('-', ' ')
    medium = 'y' if n.startswith('(ye)') else 's' if n.startswith('(s)') else None
    if medium is None:
        return None, None
    strain = ('rgf1del' if 'rgf1' in n else '') + ('gpd1del' if 'gpd1' in n else '') or ('wt' if 'wt' in n else None)
    cell = re.findall(r'cell\s*(\d+)', n)
    return (f'{medium}_{strain}' if strain else None), (int(cell[-1]) if cell else None)


def _is_length(t):
    return any(w in t for w in LENGTH_WORDS) and not any(w in t for w in ('change', 'cahnge'))


def read_sheet(ws, frame_minutes=2.0):
    """One sheet -> long table for whichever of the three blocks it has (None if none).

    A block starts at its label ("cell 5 - old end", "OLD END", "total length", ...).
    Its header row is the label's row or one of the next two, whichever has a length
    header to the right of the label; its frame column is a "Frame" header in that
    row, else the label's own column (which holds frame numbers in the data rows).
    """
    rows = [list(r) for r in ws.iter_rows(values_only=True)]
    width = max((len(r) for r in rows), default=0)
    rows = [r + [None] * (width - len(r)) for r in rows]
    anchors = {}
    for i, r in enumerate(rows[:12]):
        for j, v in enumerate(r):
            t = _text(v)
            if 'length (' in t:            # a column header like "Length (Old end tip to birthscar)", not a label
                continue
            for end, keys in ENDS:
                if end not in anchors and any(k in t for k in keys):
                    anchors[end] = (i, j)
    order = sorted(anchors.items(), key=lambda kv: kv[1][1])
    out = []
    for k, (end, (ai, aj)) in enumerate(order):
        stop = order[k + 1][1][1] if k + 1 < len(order) else width
        found = None
        for hi in range(ai, min(ai + 3, len(rows))):
            cols = [j for j in range(aj + 1, stop) if _is_length(_text(rows[hi][j]))]
            if cols:
                found = hi, cols[0]
                break
        if found is None:
            continue
        hi, length = found
        frame = next((j for j in range(aj, stop) if _text(rows[hi][j]) == 'frame'), aj)
        for r in rows[hi + 1:]:
            if _num(r[frame]) and _num(r[length]):
                out.append(dict(END=end, FRAME=int(r[frame]), ABS_LENGTH=float(r[length])))
    d = pd.DataFrame(out)
    if d.empty:
        return None
    res = []
    for end, e in d.groupby('END'):
        e = e.drop_duplicates('FRAME').sort_values('FRAME')
        e['LENGTH'] = e.ABS_LENGTH - e.ABS_LENGTH.iloc[0]
        e['TIME'] = (e.FRAME - e.FRAME.iloc[0]) * frame_minutes
        res.append(e)
    return pd.concat(res)


def read_workbook(path, frame_minutes=2.0):
    import openpyxl
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    parts, skipped = [], []
    for ws in wb.worksheets:
        typ, cell = sheet_type(ws.title)
        d = read_sheet(ws, frame_minutes) if typ else None
        if d is None:
            skipped.append(ws.title)
            continue
        parts.append(d.assign(TYPE=typ, CELLNUM=cell, SHEET=ws.title))
    t = pd.concat(parts, ignore_index=True)
    partial = {sh: sorted(set(('OE', 'NE', 'TOT')) - set(g.END)) for sh, g in t.groupby('SHEET') if g.END.nunique() < 3}
    return t[['TYPE', 'CELLNUM', 'END', 'LENGTH', 'TIME', 'ABS_LENGTH', 'FRAME', 'SHEET']], skipped, partial


def load_reference(path, kind=None, frame_minutes=2.0):
    """Hand-tracked curves for one TYPE from a workbook (.xlsx) or an already converted CSV.

    With kind=None the file must hold a single TYPE; otherwise the error lists the TYPEs it has.
    """
    path = Path(path)
    t = read_workbook(path, frame_minutes)[0] if path.suffix.lower() in ('.xlsx', '.xlsm') else pd.read_csv(path)
    t = t.dropna(how='all')
    kinds = sorted(t.TYPE.dropna().unique())
    if kind is None:
        if len(kinds) != 1:
            raise ValueError(f'{path.name} holds several cell types; pick one with --hand-type: {kinds}')
        kind = kinds[0]
    r = t[t.TYPE == kind]
    if r.empty:
        raise ValueError(f'no {kind!r} cells in {path.name}; types present: {kinds}')
    return r


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('workbook')
    p.add_argument('out')
    p.add_argument('--frame-minutes', type=float, default=2.0)
    a = p.parse_args()
    t, skipped, partial = read_workbook(a.workbook, a.frame_minutes)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(a.out, index=False)
    print(t.groupby('TYPE').CELLNUM.nunique().to_string())
    if skipped:
        print('skipped sheets:', skipped)
    for sheet, missing in partial.items():
        print(f'{sheet!r}: no {"/".join(missing)} block')


if __name__ == '__main__':
    main()
