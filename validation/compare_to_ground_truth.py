"""
compare_to_ground_truth.py  –  Compare BS-Detector's output against manually
collected FIJI ground-truth measurements.

Inputs
------
bs_detector_csvs : one or more CSVs produced by render_blinded_frames.py (or
                   export_csv(..., columns=['video_id'] + Config.CSV_COLUMNS +
                   ['scar_midpoint', 'new_pole_point', 'old_pole_point'])).
                   Must include the extra coordinate + video_id columns, not
                   just the default CSV_COLUMNS. Pass one per video — cell
                   names are only unique within a single video's tracker.
ground_truth_csv : the filled-in validation/ground_truth_template.csv from
                   the wet-lab scorer(s), with the EXAMPLE row removed.

Rows are matched on (video_id, frame, cell_name). Pole identity is resolved
positionally — Pole A is whichever pole has the smaller row (y) coordinate —
so compartment-length comparisons never depend on the algorithm's or the
scorer's new/old labeling being right. New/old pole *identity* agreement is
reported separately, as its own diagnostic.

Usage
-----
    python -m validation.compare_to_ground_truth \\
        xy1_bs_detector_measurements.csv xy2_bs_detector_measurements.csv \\
        ground_truth_template.csv [--out report_dir]
"""
import argparse
import ast
import csv
import math
import os


def _parse_point(s):
    """Parse a stringified [row, col] list back into a (row, col) tuple."""
    if s is None or s == '' or s == 'None':
        return None
    return tuple(ast.literal_eval(s))


def _to_float(s):
    if s is None or s == '' or s == 'None':
        return None
    return float(s)


def load_detector_csvs(paths):
    """Load and merge one or more per-video detector CSVs, keyed by
    (video_id, frame, cell_name) so cell names that repeat across videos
    (e.g. every video has an 'A') don't collide."""
    rows = {}
    for path in paths:
        with open(path, newline='') as f:
            for row in csv.DictReader(f):
                video_id = row.get('video_id', '')
                key = (video_id, int(row['frame']), row['cell_name'])
                rows[key] = row
    return rows


def load_ground_truth_csv(path):
    rows = []
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            if row.get('scorer_initials', '').strip().upper() == 'EXAMPLE':
                continue
            if not row.get('cell_name', '').strip():
                continue
            rows.append(row)
    return rows


def resolve_pole_a_b(new_pole_point, old_pole_point, new_end_length, old_end_length):
    """
    Positionally assign Pole A (smaller row) / Pole B (larger row), and report
    which one the detector calls 'new'.

    Returns (poleA_length, poleB_length, det_new_pole_label) or None if pole
    points are unavailable.
    """
    if new_pole_point is None or old_pole_point is None:
        return None
    if new_end_length is None or old_end_length is None:
        return None

    if new_pole_point[0] <= old_pole_point[0]:
        return new_end_length, old_end_length, 'A'
    return old_end_length, new_end_length, 'B'


def compare(detector_rows, gt_rows):
    """Join ground-truth rows against detector rows and compute per-cell errors."""
    joined = []
    unmatched = []

    for gt in gt_rows:
        key = (gt.get('video_id', ''), int(gt['frame']), gt['cell_name'])
        det = detector_rows.get(key)
        if det is None:
            unmatched.append(key)
            continue

        rec = {
            'video_id': key[0],
            'frame': key[1],
            'cell_name': key[2],
            'notes': gt.get('notes', ''),
        }

        # ── Area ──────────────────────────────────────────────────────────
        gt_area = _to_float(gt.get('gt_area_px2'))
        det_area = _to_float(det.get('area'))
        if gt_area is not None and det_area is not None:
            rec['area_gt'] = gt_area
            rec['area_det'] = det_area
            rec['area_error'] = det_area - gt_area

        # ── Birth scar location ──────────────────────────────────────────
        gt_x = _to_float(gt.get('gt_scar_mid_x_px'))
        gt_y = _to_float(gt.get('gt_scar_mid_y_px'))
        det_scar = _parse_point(det.get('scar_midpoint'))
        if gt_x is not None and gt_y is not None and det_scar is not None:
            det_row, det_col = det_scar
            rec['scar_dist_px'] = math.hypot(det_col - gt_x, det_row - gt_y)

        # ── Compartment lengths (positional Pole A / Pole B) ─────────────
        gt_a = _to_float(gt.get('gt_poleA_length_px'))
        gt_b = _to_float(gt.get('gt_poleB_length_px'))
        new_pole_pt = _parse_point(det.get('new_pole_point'))
        old_pole_pt = _parse_point(det.get('old_pole_point'))
        new_len = _to_float(det.get('new_end_length'))
        old_len = _to_float(det.get('old_end_length'))
        resolved = resolve_pole_a_b(new_pole_pt, old_pole_pt, new_len, old_len)

        if resolved is not None:
            det_a, det_b, det_new_label = resolved
            rec['det_new_pole_label'] = det_new_label
            if gt_a is not None:
                rec['poleA_gt'] = gt_a
                rec['poleA_det'] = det_a
                rec['poleA_error'] = det_a - gt_a
            if gt_b is not None:
                rec['poleB_gt'] = gt_b
                rec['poleB_det'] = det_b
                rec['poleB_error'] = det_b - gt_b

            gt_new_guess = (gt.get('gt_new_pole_guess') or '').strip().upper()
            if gt_new_guess in ('A', 'B'):
                rec['new_pole_agreement'] = (gt_new_guess == det_new_label)

        joined.append(rec)

    return joined, unmatched


def _stats(values):
    n = len(values)
    if n == 0:
        return None
    mae = sum(abs(v) for v in values) / n
    rmse = math.sqrt(sum(v * v for v in values) / n)
    bias = sum(values) / n
    return {'n': n, 'mean_bias': bias, 'mae': mae, 'rmse': rmse}


def summarize(joined):
    summary = {}
    for label, key in [('Cell area (px^2)', 'area_error'),
                        ('Birth scar location (px)', 'scar_dist_px'),
                        ('Pole A compartment length (px)', 'poleA_error'),
                        ('Pole B compartment length (px)', 'poleB_error')]:
        values = [r[key] for r in joined if key in r]
        summary[label] = _stats(values)

    agreements = [r['new_pole_agreement'] for r in joined if 'new_pole_agreement' in r]
    if agreements:
        summary['New/old pole identity agreement'] = {
            'n': len(agreements),
            'agreement_rate': sum(agreements) / len(agreements),
        }
    return summary


def print_report(summary, unmatched):
    print("\n── Ground-Truth Comparison ─────────────────────────────────────")
    for label, stats in summary.items():
        if stats is None:
            print(f"  {label}: no comparable rows")
            continue
        if 'agreement_rate' in stats:
            print(f"  {label}: {stats['agreement_rate']*100:.1f}% "
                  f"(n={stats['n']})")
        else:
            print(f"  {label}: n={stats['n']}  "
                  f"bias={stats['mean_bias']:+.2f}  "
                  f"MAE={stats['mae']:.2f}  RMSE={stats['rmse']:.2f}")
    if unmatched:
        print(f"\n  {len(unmatched)} ground-truth row(s) had no matching "
              f"detector row (video_id, frame, cell_name):")
        for key in unmatched:
            print(f"    {key}")
    print("─────────────────────────────────────────────────────────────────\n")


def save_joined_csv(joined, out_path):
    if not joined:
        return
    fieldnames = sorted({k for r in joined for k in r.keys()})
    lead = ['video_id', 'frame', 'cell_name']
    fieldnames = lead + [f for f in fieldnames if f not in lead]
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(joined)
    print(f"Per-cell comparison saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='+',
                        help='one or more per-video detector CSVs, followed by the ground-truth CSV last')
    parser.add_argument('--out', default='.', help='Output directory for the joined report CSV')
    args = parser.parse_args()

    if len(args.paths) < 2:
        parser.error('need at least one detector CSV and one ground-truth CSV')

    *detector_csv_paths, ground_truth_csv = args.paths

    detector_rows = load_detector_csvs(detector_csv_paths)
    gt_rows = load_ground_truth_csv(ground_truth_csv)

    joined, unmatched = compare(detector_rows, gt_rows)
    summary = summarize(joined)
    print_report(summary, unmatched)

    os.makedirs(args.out, exist_ok=True)
    save_joined_csv(joined, os.path.join(args.out, 'ground_truth_comparison.csv'))


if __name__ == '__main__':
    main()
