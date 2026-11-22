"""Checking and correcting: review flags, corrections (edits.csv and Fiji CSVs), frame storage, cell drawing."""
import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pombe_tracker.colab import click_to_movie
from pombe_tracker.corrections import (add_edit, apply_corrections, corrected_measurements, frames_text,
                                       parse_frames, read_corrections, to_movie, to_pipeline)
from pombe_tracker.review import frame_flags, review_table
from pombe_tracker.workflow import Run

REPO = Path(__file__).resolve().parents[1]


def _table():
    """One cell, 6 frames; frame 3 has a scar jump, frame 4 is flagged by the detector."""
    rows = []
    for f in range(6):
        new = 30.0 if f != 3 else 60.0                  # px
        rows.append(dict(frame=f, cell_name='A0', length=150.0, length_um=150 * 0.0647,
                         new_end_length=new, new_end_length_um=new * 0.0647,
                         old_end_length=150 - new, old_end_length_um=(150 - new) * 0.0647,
                         scar_detected=True, scar_source='raw', seg_quality='ok',
                         scar_needs_review=(f == 4), scar_pole_conflict=False,
                         scar_midpoint=json.dumps([100.0 + new, 200.0]),
                         new_pole_point=json.dumps([100.0, 200.0]), old_pole_point=json.dumps([250.0, 200.0])))
    return pd.DataFrame(rows)


class FakeRun:
    def __init__(self, path, scale=(1.0, 1.0)):
        self.path = Path(path)
        self.corrections_dir = self.path / 'corrections'
        self.measurements = self.path / 'measurements.csv'
        self.scale_yx = scale
        self.um_per_px = 0.0647
        self.frame_minutes = 2.0


class Corrections(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run = FakeRun(self.tmp.name)
        _table().to_csv(self.run.measurements, index=False)

    def tearDown(self):
        self.tmp.cleanup()

    def test_frame_ranges(self):
        self.assertEqual(parse_frames('40-42, 50'), [40, 41, 42, 50])
        self.assertIsNone(parse_frames(''))
        self.assertIsNone(parse_frames('all'))
        self.assertEqual(frames_text([5, 1, 2, 3, 9]), '1-3, 5, 9')

    def test_coordinates_round_trip_with_resampling(self):
        run = FakeRun(self.tmp.name, scale=(1.7, 1.7))
        r, c = to_pipeline(run, 120.0, 80.0)
        self.assertAlmostEqual(r, 80.5 * 1.7 - 0.5)
        x, y = to_movie(run, r, c)
        self.assertAlmostEqual(x, 120.0)
        self.assertAlmostEqual(y, 80.0)

    def test_scar_move_recomputes_both_ends(self):
        add_edit(self.run, 'A0', '3', scar_x=200, scar_y=130)        # 30 px from the new pole, 120 from the old
        d, n = corrected_measurements(self.run, log=lambda *_: None)
        r = d[d.frame == 3].iloc[0]
        self.assertEqual(n, 1)
        self.assertAlmostEqual(r.new_end_length, 30.0)
        self.assertAlmostEqual(r.old_end_length, 120.0)
        self.assertAlmostEqual(r.new_end_length_um, round(30 * 0.0647, 4))
        self.assertEqual(r.scar_source, 'manual')
        self.assertIn('scar moved', r.corrected)
        self.assertTrue((self.run.path / 'measurements_corrected.csv').exists())
        self.assertEqual(pd.read_csv(self.run.measurements).new_end_length[3], 60.0)   # detector output untouched

    def test_exclude_swap_lengths_checked(self):
        add_edit(self.run, 'A0', '0-1', exclude='yes')
        add_edit(self.run, 'A0', '2', swap_poles='yes')
        add_edit(self.run, 'A0', '5', new_end_length_um=2.0, old_end_length_um=7.5)
        add_edit(self.run, 'A0', '4', checked='yes')
        d, n = corrected_measurements(self.run, log=lambda *_: None)
        self.assertEqual(n, 4)
        self.assertEqual(list(d.excluded), [True, True, False, False, False, False])
        self.assertAlmostEqual(d.new_end_length[2], 120.0)
        self.assertEqual(json.loads(d.new_pole_point[2]), [250.0, 200.0])
        self.assertAlmostEqual(d.new_end_length_um[5], 2.0)
        self.assertAlmostEqual(d.new_end_length[5], 2.0 / 0.0647)
        self.assertTrue(d.checked[4])

    def test_later_fix_wins_and_problems_reported(self):
        add_edit(self.run, 'A0', '3', scar_x=200, scar_y=120)
        add_edit(self.run, 'A0', '3', scar_x=200, scar_y=130)
        add_edit(self.run, 'ZZ', '1', exclude='yes')
        add_edit(self.run, 'A0', '99', exclude='yes')
        fixes = read_corrections(self.run)
        d, n, problems = apply_corrections(pd.read_csv(self.run.measurements), fixes, self.run)
        self.assertAlmostEqual(d.new_end_length[3], 30.0)
        self.assertEqual(n, 2)
        self.assertEqual(len(problems), 2)

    def test_fiji_macro_csv(self):
        self.run.corrections_dir.mkdir(exist_ok=True)
        (self.run.corrections_dir / 'fiji_movie_cell-A0.csv').write_text(
            'movie,cell_id,scorer,frame_slider,frame_index,scar_x,scar_y,new_pole_x,new_pole_y,old_pole_x,old_pole_y,status,notes\n'
            'm.nd2,A0,xw,4,3,200,130,200,100,200,260,ok,\n'
            'm.nd2,A0,xw,5,4,,,,,,,ok,nothing clicked\n')
        d, n = corrected_measurements(self.run, log=lambda *_: None)
        r = d[d.frame == 3].iloc[0]
        self.assertEqual(n, 1)                                  # the empty row is ignored
        self.assertAlmostEqual(r.new_end_length, 30.0)
        self.assertAlmostEqual(r.old_end_length, 130.0)         # clicked old pole
        self.assertAlmostEqual(r.length, 160.0)                 # pole to pole, from the clicked poles

    def test_no_corrections_leaves_no_corrected_file(self):
        d, n = corrected_measurements(self.run, log=lambda *_: None)
        self.assertEqual(n, 0)
        self.assertFalse((self.run.path / 'measurements_corrected.csv').exists())


class Review(unittest.TestCase):

    def test_flags_and_corrections_clear_them(self):
        d = _table()
        f = frame_flags(d)
        self.assertEqual(list(f.jump), [False, False, False, True, False, False])
        self.assertEqual(list(f.unsure), [False, False, False, False, True, False])
        self.assertEqual(int(f.to_check.sum()), 2)
        d['corrected'] = ['', '', '', 'scar moved', '', '']
        d['excluded'] = False
        d['checked'] = [False, False, False, False, True, False]
        self.assertEqual(int(frame_flags(d).to_check.sum()), 0)

    def test_review_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = FakeRun(tmp)
            _table().to_csv(run.measurements, index=False)
            t = review_table(run)
            self.assertEqual(t.cell_name[0], 'A0')
            self.assertEqual(t.frames_to_check[0], 2)
            self.assertEqual(t.check_frames[0], '3-4')
            self.assertIn('jump 1', t.why[0])

    def test_click_mapping(self):
        extent = (10.0, 110.0, 250.0, 50.0)                    # x0, x1, y_bottom, y_top
        self.assertEqual(click_to_movie(extent, 0, 0), (10.0, 50.0))
        self.assertEqual(click_to_movie(extent, 0.5, 0.25), (60.0, 100.0))


class SampleRun(unittest.TestCase):
    """The real pipeline on the sample movie: frames read in place, masks in working/, then a cell drawn."""

    def test_track_review_show(self):
        import matplotlib
        matplotlib.use('Agg')
        from pombe_tracker.review import show_cell
        from pombe_tracker.workflow import prepare, track
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / 'run'
            (run_dir / 'working').mkdir(parents=True)
            shutil.copytree(REPO / 'data/sample/masks', run_dir / 'working' / 'masks')
            with contextlib.redirect_stdout(io.StringIO()):
                prepare(REPO / 'data/sample/sample.h5', run_dir, log=lambda *_: None)
                track(run_dir, log=lambda *_: None)
            run = Run(run_dir)
            self.assertFalse(run.info['frames']['copied'])            # no copy of the movie
            self.assertTrue((run_dir / 'review.csv').exists())
            self.assertTrue((run_dir / 'corrections' / 'edits.csv').exists())
            self.assertFalse((run_dir / 'figures').exists())           # no figures unless asked
            m = pd.read_csv(run.measurements)
            self.assertTrue({'label', 'scar_end_1', 'scar_end_2'} <= set(m.columns))
            fig = show_cell(run, 'F')
            self.assertGreaterEqual(len(fig.axes), 6)


class ResampledFrames(unittest.TestCase):

    def test_converted_copy_is_rebuilt_if_missing(self):
        import tifffile
        from pombe_tracker.workflow import prepare
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            stack = np.random.default_rng(1).integers(100, 4000, (2, 30, 40)).astype(np.uint16)
            tifffile.imwrite(tmp / 'm.tif', stack, imagej=True, resolution=(1 / 0.11, 1 / 0.11),
                             metadata={'axes': 'TYX', 'unit': 'um'})
            with contextlib.redirect_stdout(io.StringIO()):
                run = prepare(tmp / 'm.tif', tmp / 'run', frames_dir=tmp / 'local', log=lambda *_: None)
                self.assertTrue(run.info['frames']['copied'])
                with run.frames() as f:
                    first = f[0].copy()
                (tmp / 'local' / 'frames.h5').unlink()                 # e.g. a new Colab session
                with Run(tmp / 'run').frames() as f:
                    np.testing.assert_array_equal(f[0], first)
            self.assertEqual(first.shape, (51, 68))


if __name__ == '__main__':
    unittest.main()
