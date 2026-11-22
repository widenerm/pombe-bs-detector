"""Checks for the measurements CSV export, including the optional µm columns."""
import csv
import tempfile
import unittest
from pathlib import Path

from pombe_tracker.io_utils import export_csv

COLUMNS = ['cell_name', 'frame', 'length', 'new_end_length', 'area', 'scar_detected']


def results():
    return [{'frame_idx': 0, 'cells': [
        {'cell_name': 'A', 'length': 200.0, 'new_end_length': 50.0, 'area': 10000,
         'scar_detected': True, 'seg_quality': 'ok'},
        {'cell_name': 'B', 'length': 150.0, 'new_end_length': None, 'area': 8000,
         'scar_detected': False, 'seg_quality': 'ok'},
        {'cell_name': 'C', 'length': 90.0, 'area': 4000, 'seg_quality': 'border_clip'},
    ]}]


def read(path):
    with open(path, newline='') as fh:
        reader = csv.DictReader(fh)
        return reader.fieldnames, list(reader)


class ExportTests(unittest.TestCase):
    def test_pixels_only_by_default(self):
        with tempfile.TemporaryDirectory() as d:
            header, rows = read(export_csv(results(), Path(d) / 'm.csv', columns=COLUMNS))
            self.assertEqual(header, ['frame', 'cell_name', 'length', 'new_end_length', 'area',
                                      'scar_detected'])
            self.assertEqual([r['cell_name'] for r in rows], ['A', 'B'])   # border_clip dropped

    def test_micron_columns_follow_their_pixel_columns(self):
        with tempfile.TemporaryDirectory() as d:
            header, rows = read(export_csv(results(), Path(d) / 'm.csv', columns=COLUMNS,
                                           pixel_size_um=0.0647))
            self.assertEqual(header, ['frame', 'cell_name', 'length', 'length_um', 'new_end_length',
                                      'new_end_length_um', 'area', 'area_um2', 'scar_detected'])
            a, b = rows
            self.assertAlmostEqual(float(a['length_um']), 12.94)
            self.assertAlmostEqual(float(a['new_end_length_um']), 3.235)
            self.assertAlmostEqual(float(a['area_um2']), 41.8609)
            self.assertEqual(b['new_end_length_um'], '')                    # no scar stays blank


if __name__ == '__main__':
    unittest.main()
