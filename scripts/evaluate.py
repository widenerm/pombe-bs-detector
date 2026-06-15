#!/usr/bin/env python3
"""
evaluate.py  –  Score a detector run against ground truth from the command line.

Usage
─────
    # 1. After running the pipeline, export the detector output:
    #       from pombe_tracker.evaluation import export_eval_json
    #       export_eval_json(all_results, 'detector.json')
    #
    # 2. Score it against a ground-truth CSV (see ground_truth.py for schema):
    python scripts/evaluate.py detector.json ground_truth.csv

    # Optional: write the full report as JSON
    python scripts/evaluate.py detector.json ground_truth.csv --out report.json
"""
import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pombe_tracker.evaluation import evaluate, EvalConfig, format_report


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('detector', help='detector eval JSON (from export_eval_json)')
    ap.add_argument('ground_truth', help='ground-truth CSV or JSON')
    ap.add_argument('--out', help='write full report dict as JSON to this path')
    ap.add_argument('--match-distance', type=float, default=40.0)
    ap.add_argument('--localization-tolerance', type=float, default=10.0)
    args = ap.parse_args()

    cfg = EvalConfig(match_max_distance=args.match_distance,
                     localization_tolerance=args.localization_tolerance)
    report = evaluate(args.detector, args.ground_truth, eval_config=cfg)

    print(format_report(report))

    if args.out:
        with open(args.out, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f'\nFull report written to {args.out}')


if __name__ == '__main__':
    main()
