"""
compare_tracking_to_ground_truth.py  –  Compare BS-Detector's tracking and
division detection against manually re-tracked ground truth.

Inputs
------
key_dir           : directory containing the `*_KEY_do_not_share.csv` files
                     written by render_tracking_blind_frames.py (one per run)
                     and the `lineage_log.csv` it also writes.
tracking_csv       : the scorer's filled-in lineage_tracking_template.csv
                     (EXAMPLE row removed).
divisions_csv      : the scorer's filled-in division_events_template.csv
                     (EXAMPLE row removed).

Two failure modes are quantified:

  fragmentation   : ground truth says a physical cell persists from one
                    appearance to the next, but BS-Detector gave it a
                    different cell_name with no division to explain the
                    change — i.e. the cell dropped out of tracking and came
                    back under a new label.

  missed_division : ground truth observed a real division that BS-Detector's
                    lineage_log has no matching record of (it just kept
                    tracking one daughter under the parent's name).

  false_division  : BS-Detector's lineage_log records a division the ground
                    truth run doesn't corroborate.

Usage
-----
    python -m validation.compare_tracking_to_ground_truth \\
        key_dir/ lineage_tracking_template.csv division_events_template.csv \\
        [--out report_dir]
"""
import argparse
import csv
import glob
import os


def load_key(key_dir):
    """Load every *_KEY_do_not_share.csv in key_dir into {(run_id, frame, frame_local_id): cell_name}."""
    key = {}
    for path in glob.glob(os.path.join(key_dir, '*_KEY_do_not_share.csv')):
        with open(path, newline='') as f:
            for row in csv.DictReader(f):
                k = (row['run_id'], int(row['frame']), int(row['frame_local_id']))
                key[k] = row['cell_name']
    return key


def load_lineage_log(key_dir):
    """Load lineage_log.csv into a list of (frame, parent, {daughter0, daughter1})."""
    path = os.path.join(key_dir, 'lineage_log.csv')
    events = []
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            events.append({
                'frame': int(row['frame']),
                'parent': row['parent'],
                'daughters': frozenset([row['daughter0'], row['daughter1']]),
            })
    return events


def load_tracking_rows(path):
    rows = []
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            if row.get('scorer_initials', '').strip().upper() == 'EXAMPLE':
                continue
            if not row.get('true_track_id', '').strip():
                continue
            rows.append(row)
    return rows


def load_division_rows(path):
    rows = []
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            if row.get('scorer_initials', '').strip().upper() == 'EXAMPLE':
                continue
            if not row.get('division_frame', '').strip():
                continue
            rows.append(row)
    return rows


def resolve_name(key, run_id, frame, frame_local_id, missing):
    name = key.get((run_id, frame, frame_local_id))
    if name is None:
        missing.append((run_id, frame, frame_local_id))
    return name


def check_continuity(tracking_rows, key):
    """Group by (run_id, true_track_id), check consecutive appearances for
    identity continuity in BS-Detector's own cell_name."""
    groups = {}
    for row in tracking_rows:
        gk = (row['run_id'], row['true_track_id'])
        groups.setdefault(gk, []).append(row)

    missing = []
    matches, fragmentations = [], []

    for gk, rows in groups.items():
        rows.sort(key=lambda r: int(r['frame']))
        for a, b in zip(rows, rows[1:]):
            run_id = a['run_id']
            fa = int(a['frame']); fb = int(b['frame'])
            name_a = resolve_name(key, run_id, fa, int(a['frame_local_id']), missing)
            name_b = resolve_name(key, run_id, fb, int(b['frame_local_id']), missing)
            if name_a is None or name_b is None:
                continue
            record = {
                'run_id': run_id, 'true_track_id': gk[1],
                'frame_a': fa, 'frame_b': fb,
                'cell_name_a': name_a, 'cell_name_b': name_b,
            }
            if name_a == name_b:
                matches.append(record)
            else:
                fragmentations.append(record)

    return matches, fragmentations, missing


def check_divisions(division_rows, key, lineage_events):
    missing = []
    confirmed, missed = [], []

    for row in division_rows:
        run_id = row['run_id']
        div_frame = int(row['division_frame'])
        parent_name = resolve_name(key, run_id, div_frame - 1,
                                   int(row['parent_frame_local_id']), missing)
        d1_name = resolve_name(key, run_id, div_frame,
                               int(row['daughter1_frame_local_id']), missing)
        d2_name = resolve_name(key, run_id, div_frame,
                               int(row['daughter2_frame_local_id']), missing)
        if None in (parent_name, d1_name, d2_name):
            continue

        record = {
            'run_id': run_id, 'division_frame': div_frame,
            'parent': parent_name, 'daughters': frozenset([d1_name, d2_name]),
        }
        found = any(ev['frame'] == div_frame and ev['parent'] == parent_name
                    and ev['daughters'] == record['daughters']
                    for ev in lineage_events)
        (confirmed if found else missed).append(record)

    confirmed_keys = {(r['division_frame'], r['parent'], r['daughters']) for r in confirmed}
    false_divisions = [
        ev for ev in lineage_events
        if (ev['frame'], ev['parent'], ev['daughters']) not in confirmed_keys
        and _plausible_run_match(ev, division_rows, key)
    ]

    return confirmed, missed, false_divisions, missing


def _plausible_run_match(ev, division_rows, key):
    """Only count a lineage_log event as a 'false division' candidate if it
    falls inside a frame range the scorer actually reviewed, otherwise we
    have no ground truth to judge it against."""
    reviewed_frames = {int(row['division_frame']) for row in division_rows}
    reviewed_frames |= {f for (_, f, _) in key.keys()}
    return ev['frame'] in reviewed_frames


def print_report(matches, fragmentations, confirmed, missed, false_divisions, missing):
    n_cont = len(matches) + len(fragmentations)
    print("\n── Tracking & Division Comparison ──────────────────────────────")
    if n_cont:
        frag_rate = len(fragmentations) / n_cont
        print(f"  Track continuity checked: {n_cont}  "
              f"(fragmentations: {len(fragmentations)}, rate={frag_rate*100:.1f}%)")
    else:
        print("  Track continuity checked: 0 (no comparable rows)")

    n_div_gt = len(confirmed) + len(missed)
    if n_div_gt:
        recall = len(confirmed) / n_div_gt
        print(f"  Ground-truth divisions: {n_div_gt}  "
              f"(detected: {len(confirmed)}, missed: {len(missed)}, recall={recall*100:.1f}%)")
    else:
        print("  Ground-truth divisions: 0")

    print(f"  False divisions (algorithm reported, not corroborated): {len(false_divisions)}")

    if missing:
        print(f"\n  {len(missing)} ground-truth row(s) referenced a "
              f"(run_id, frame, frame_local_id) missing from the key file(s):")
        for k in missing[:20]:
            print(f"    {k}")
        if len(missing) > 20:
            print(f"    ... and {len(missing) - 20} more")
    print("─────────────────────────────────────────────────────────────────\n")


def save_reports(out_dir, fragmentations, missed, false_divisions):
    os.makedirs(out_dir, exist_ok=True)

    def _write(name, rows, fieldnames):
        path = os.path.join(out_dir, name)
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {path} ({len(rows)} row(s))")

    _write('fragmentations.csv', fragmentations,
           ['run_id', 'true_track_id', 'frame_a', 'frame_b', 'cell_name_a', 'cell_name_b'])
    _write('missed_divisions.csv',
           [{'run_id': r['run_id'], 'division_frame': r['division_frame'],
             'parent': r['parent'], 'daughters': set(r['daughters'])}
            for r in missed],
           ['run_id', 'division_frame', 'parent', 'daughters'])
    _write('false_divisions.csv',
           [{'frame': ev['frame'], 'parent': ev['parent'], 'daughters': set(ev['daughters'])}
            for ev in false_divisions],
           ['frame', 'parent', 'daughters'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('key_dir')
    parser.add_argument('tracking_csv')
    parser.add_argument('divisions_csv')
    parser.add_argument('--out', default='.', help='Output directory for detailed CSV logs')
    args = parser.parse_args()

    key = load_key(args.key_dir)
    lineage_events = load_lineage_log(args.key_dir)
    tracking_rows = load_tracking_rows(args.tracking_csv)
    division_rows = load_division_rows(args.divisions_csv)

    matches, fragmentations, missing_a = check_continuity(tracking_rows, key)
    confirmed, missed, false_divisions, missing_b = check_divisions(
        division_rows, key, lineage_events)

    print_report(matches, fragmentations, confirmed, missed, false_divisions,
                 missing_a + missing_b)
    save_reports(args.out, fragmentations, missed, false_divisions)


if __name__ == '__main__':
    main()
