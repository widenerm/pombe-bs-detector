"""
apply_lineage_corrections.py  –  Post-hoc fix for track fragmentation found
by compare_tracking_to_ground_truth.py.

This does NOT change the detection/tracking algorithm — it merges cell_name
values in an already-exported measurements CSV so a physical cell that got
split across two or more algorithm identities is treated as one lineage in
downstream analysis. Use it once you've validated (via the tracking ground
truth) which identity pairs are genuinely the same cell.

Two steps
---------
1. suggest  – read fragmentations.csv (from compare_tracking_to_ground_truth.py)
              and propose a corrections.csv mapping each fragmented
              cell_name to one canonical name. Chained fragmentations
              (A -> B -> C) are merged transitively into a single group.

              This is a STARTING POINT, not a verdict — review corrections.csv
              by hand (e.g. delete rows for cases you don't actually believe
              are the same cell) before applying it.

2. apply    – rewrite a BS-Detector measurements CSV's cell_name column
              according to a (possibly hand-edited) corrections.csv.

Usage
-----
    python -m validation.apply_lineage_corrections suggest \\
        fragmentations.csv corrections.csv

    python -m validation.apply_lineage_corrections apply \\
        measurements.csv corrections.csv measurements_corrected.csv
"""
import csv
import sys


def _find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _union(parent, a, b):
    ra, rb = _find(parent, a), _find(parent, b)
    if ra != rb:
        parent[ra] = rb


def suggest(fragmentations_csv, out_path):
    pairs = []
    with open(fragmentations_csv, newline='') as f:
        for row in csv.DictReader(f):
            pairs.append((row['cell_name_a'], row['cell_name_b']))

    names = sorted({n for pair in pairs for n in pair})
    parent = {n: n for n in names}
    for a, b in pairs:
        _union(parent, a, b)

    groups = {}
    for n in names:
        groups.setdefault(_find(parent, n), []).append(n)

    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['original_cell_name', 'canonical_cell_name'])
        writer.writeheader()
        for members in groups.values():
            canonical = sorted(members)[0]
            for name in sorted(members):
                writer.writerow({'original_cell_name': name, 'canonical_cell_name': canonical})

    print(f"Suggested corrections saved: {out_path}")
    print("Review before applying — each group is fragmented identities the "
          "validation flagged as likely the same physical cell.")
    for members in groups.values():
        print(f"  {sorted(members)} -> {sorted(members)[0]}")


def apply(measurements_csv, corrections_csv, out_csv):
    corrections = {}
    with open(corrections_csv, newline='') as f:
        for row in csv.DictReader(f):
            corrections[row['original_cell_name']] = row['canonical_cell_name']

    with open(measurements_csv, newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    n_changed = 0
    for row in rows:
        original = row.get('cell_name')
        canonical = corrections.get(original)
        if canonical is not None and canonical != original:
            row['cell_name'] = canonical
            n_changed += 1

    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

    print(f"Corrected CSV saved: {out_csv}  ({n_changed} row(s) relabeled)")


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in ('suggest', 'apply'):
        print(__doc__)
        sys.exit(1)

    mode = sys.argv[1]
    if mode == 'suggest':
        if len(sys.argv) != 4:
            print("Usage: python -m validation.apply_lineage_corrections suggest "
                  "<fragmentations.csv> <corrections_out.csv>")
            sys.exit(1)
        suggest(sys.argv[2], sys.argv[3])
    else:
        if len(sys.argv) != 5:
            print("Usage: python -m validation.apply_lineage_corrections apply "
                  "<measurements.csv> <corrections.csv> <measurements_corrected.csv>")
            sys.exit(1)
        apply(sys.argv[2], sys.argv[3], sys.argv[4])
