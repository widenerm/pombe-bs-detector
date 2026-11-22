"""Cell selection for the septin statistics: division labels and length cut-offs."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pombe_tracker.hand_tracking import load_reference
from septin_curve_vis.analyze_statistics import select_cells


class SeptinFilters(unittest.TestCase):

    def setUp(self):
        self.summary = pd.DataFrame(dict(cell_id=['a', 'b', 'c', 'd'], length_um=[7.0, 9.0, 12.0, 15.0],
                                         max_abs_curvature=[0.05] * 4))
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_labels_exclude_only_dividing_cells(self):
        f = self.d / 'labels.csv'
        pd.DataFrame(dict(cell_id=['a', 'b', 'd', 'zz'], label=['not dividing', 'dividing', 'Dividing', 'dividing'])).to_csv(f, index=False)
        kept, notes = select_cells(self.summary, exclude=f)
        self.assertEqual(list(kept.cell_id), ['a', 'c'])
        self.assertIn('excluded 2 cells', notes[0])
        self.assertIn('1 listed IDs are not in this dataset', notes[0])

    def test_plain_list_and_length(self):
        f = self.d / 'drop.csv'
        pd.DataFrame(dict(cell_id=['c'])).to_csv(f, index=False)
        kept, notes = select_cells(self.summary, exclude=f, min_length_um=8.5)
        self.assertEqual(list(kept.cell_id), ['b', 'd'])
        self.assertEqual(len(notes), 2)

    def test_length_needs_column(self):
        with self.assertRaises(ValueError):
            select_cells(self.summary.drop(columns='length_um'), max_length_um=14)


class HandTracking(unittest.TestCase):

    def test_load_reference_picks_one_type(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / 'ref.csv'
            pd.DataFrame(dict(TYPE=['y_wt', 'y_wt', 's_wt'], CELLNUM=[1, 1, 1], END=['TOT'] * 3,
                              LENGTH=[0, 1, 0], TIME=[0, 2, 0])).to_csv(f, index=False)
            self.assertEqual(len(load_reference(f, 'y_wt')), 2)
            with self.assertRaises(ValueError) as e:
                load_reference(f)
            self.assertIn('s_wt', str(e.exception))


if __name__ == '__main__':
    unittest.main()
