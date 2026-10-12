"""Checks for the through-time compartment comparison (validation/packet_neto)."""
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from validation.compare_compartments import (compare, fit_neto, load_annotations, load_measurements,
                                             write_outputs)

HEADER = ['movie', 'cell_id', 'scorer', 'frame_slider', 'frame_index', 'scar_x', 'scar_y',
          'new_pole_x', 'new_pole_y', 'old_pole_x', 'old_pole_y', 'status', 'notes']


def synthetic_cell(t0=10, n=30, rate=1.5, noise=0.0, seed=0):
    """Vertical cell: scar fixed; new pole static until t0 then grows downward."""
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(n):
        scar = np.array([500.0, 300.0])
        new = scar + [25.0 + rate * max(0, f - t0), 0.0]
        old = scar - [100.0 + 2.0 * f, 0.0]
        jitter = lambda p: p + rng.normal(0, noise, 2)
        rows.append((f, jitter(scar), jitter(new), jitter(old)))
    return rows


def write_annotation(path, rows, status_last='divided'):
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        for i, (f, scar, new, old) in enumerate(rows):
            w.writerow(['movie.nd2', 'c1', 'cr', f + 1, f, scar[1], scar[0], new[1], new[0], old[1], old[0],
                        status_last if i == len(rows) - 1 else 'ok', ''])


def write_measurements(path, rows, swap_frames=()):
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['cell_name', 'frame', 'new_end_length', 'old_end_length', 'scar_midpoint',
                    'new_pole_point', 'old_pole_point'])
        for f, scar, new, old in rows:
            n, o = (old, new) if f in swap_frames else (new, old)
            w.writerow(['A', f, np.linalg.norm(scar - n), np.linalg.norm(scar - o),
                        str(scar.tolist()), str(n.tolist()), str(o.tolist())])
            # an unrelated neighbour that must not be matched
            w.writerow(['B', f, 10, 90, str((scar + 400).tolist()), str((new + 400).tolist()), str((old + 400).tolist())])


class CompareTests(unittest.TestCase):
    def test_hinge_fit_recovers_take_off(self):
        t = np.arange(30.0)
        L = 25 + 1.5 * np.maximum(0, t - 12) + np.random.default_rng(1).normal(0, 0.5, 30)
        fit = fit_neto(t, L)
        self.assertAlmostEqual(fit['t0'], 12, delta=1.5)
        self.assertAlmostEqual(fit['slope'], 1.5, delta=0.2)
        self.assertLess(fit['sse'], fit['sse_line'])

    def test_matching_lengths_and_pole_label_check(self):
        truth = synthetic_cell()
        noisy = synthetic_cell(noise=1.0, seed=3)
        with tempfile.TemporaryDirectory() as d:
            ann, meas = Path(d) / 'movie_cell-c1_cr.csv', Path(d) / 'm.csv'
            write_annotation(ann, noisy)
            write_measurements(meas, truth, swap_frames={4, 5})
            frames, cells = compare(load_annotations(d), load_measurements(meas), frame_minutes=5)
            self.assertTrue(all(f['matched'] and f['det_cell'] == 'A' for f in frames))
            self.assertEqual(sum(not f['same_pole_labels'] for f in frames), 2)
            # lengths are compared against the annotator's pole identity even when swapped
            self.assertLess(max(abs(f['new_len_error_px']) for f in frames), 5)
            c = cells[0]
            self.assertEqual(c['ended'], 'divided')
            self.assertAlmostEqual(c['neto_detector'], 50, delta=5)    # frame 10 x 5 min
            self.assertAlmostEqual(c['neto_annotated'], 50, delta=10)
            summary = write_outputs(frames, cells, Path(d) / 'out', 5).read_text()
            self.assertIn('| c1 |', summary)
            self.assertTrue((Path(d) / 'out' / 'new_end_length.png').exists())

    def test_far_cells_are_not_matched_and_missing_scar_is_kept(self):
        rows = synthetic_cell(n=8)
        with tempfile.TemporaryDirectory() as d:
            ann, meas = Path(d) / 'a.csv', Path(d) / 'm.csv'
            write_annotation(ann, [(f, s + 200, n + 200, o + 200) for f, s, n, o in rows])
            write_measurements(meas, rows)
            frames, _ = compare(load_annotations(ann), load_measurements(meas))
            self.assertFalse(any(f['matched'] for f in frames))
            with open(ann, 'w', newline='') as fh:
                w = csv.writer(fh); w.writerow(HEADER)
                f, s, n, o = rows[0]
                w.writerow(['movie', 'c1', 'cr', 1, 0, '', '', n[1], n[0], o[1], o[0], 'ok', 'scar not visible'])
            frames, _ = compare(load_annotations(ann), load_measurements(meas))
            self.assertTrue(frames[0]['matched'])
            self.assertIsNone(frames[0]['ann_new_len'])
            self.assertIsNotNone(frames[0]['det_new_len'])


if __name__ == '__main__':
    unittest.main()
