"""Reading movies, resampling to the pipeline pixel size, and the run-folder workflow pieces."""
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from pombe_tracker import movie_io
from pombe_tracker.cli import _block, _slug, parser


def _tif(path, stack, um_per_px=None, finterval=None):
    import tifffile
    kw = {}
    meta = {'axes': 'TYX'}
    if finterval:
        meta['finterval'] = finterval
    if um_per_px:
        kw['resolution'] = (1 / um_per_px, 1 / um_per_px)
        meta['unit'] = 'um'
    tifffile.imwrite(path, stack, imagej=True, metadata=meta, **kw)


class MovieIO(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        rng = np.random.default_rng(0)
        self.stack = rng.integers(100, 4000, size=(3, 40, 60)).astype(np.uint16)

    def tearDown(self):
        self.tmp.cleanup()

    def test_tif_metadata_and_resampling(self):
        _tif(self.d / 'm.tif', self.stack, um_per_px=0.11, finterval=120)
        m = movie_io.open_movie(self.d / 'm.tif')
        self.assertEqual(m.n_frames, 3)
        self.assertAlmostEqual(m.um_per_px, 0.11, places=4)
        self.assertAlmostEqual(m.frame_minutes, 2.0)
        shape, scale = movie_io.resample_plan(m.shape, m.um_per_px)
        self.assertEqual(shape, (68, 102))                         # 40 * 1.7, 60 * 1.7
        info = movie_io.write_frames(m, self.d / 'f.h5', log=lambda *_: None)
        m.close()
        self.assertTrue(info['resampled'])
        with h5py.File(self.d / 'f.h5') as f:
            self.assertEqual(f['frames'].shape, (3, 68, 102))
            self.assertEqual(f['frames'].dtype, np.uint16)
            self.assertAlmostEqual(f.attrs['um_per_px'], 0.0647)
            # a flat image stays flat after cubic resampling
        flat = np.full((10, 10), 500, np.uint16)
        self.assertTrue(np.all(movie_io.resample(flat, (1.7, 1.7)) == 500))

    def test_pipeline_pixel_size_is_not_resampled(self):
        _tif(self.d / 'm.tif', self.stack, um_per_px=0.0647)
        m = movie_io.open_movie(self.d / 'm.tif')
        info = movie_io.write_frames(m, self.d / 'f.h5', log=lambda *_: None)
        m.close()
        self.assertFalse(info['resampled'])
        with h5py.File(self.d / 'f.h5') as f:
            np.testing.assert_array_equal(f['frames'][:], self.stack)

    def test_missing_pixel_size_needs_flag(self):
        _tif(self.d / 'm.tif', self.stack)
        m = movie_io.open_movie(self.d / 'm.tif')
        with self.assertRaises(ValueError):
            movie_io.resample_plan(m.shape, m.um_per_px)
        m.close()
        m = movie_io.open_movie(self.d / 'm.tif', um_per_px=0.0647)
        self.assertEqual(movie_io.resample_plan(m.shape, m.um_per_px)[1], None)
        m.close()

    def test_prepare_reuses_pipeline_h5(self):
        from pombe_tracker.workflow import Run, prepare
        src = self.d / 'movie.h5'
        with h5py.File(src, 'w') as f:
            f['frames'] = self.stack
            f.attrs['um_per_px'] = 0.0647
        run = prepare(src, self.d / 'run', log=lambda *_: None)
        self.assertFalse(run.info['frames']['copied'])
        self.assertEqual(Run(self.d / 'run').info['frames']['path'], str(src.resolve()))
        prepare(src, self.d / 'run', log=lambda *_: None)          # second call: nothing to do
        other = self.d / 'other.h5'
        with h5py.File(other, 'w') as f:
            f['frames'] = self.stack
        with self.assertRaises(ValueError):                        # a run folder belongs to one movie
            prepare(other, self.d / 'run', log=lambda *_: None)

    def test_unsupported_suffix(self):
        (self.d / 'm.avi').write_bytes(b'')
        with self.assertRaises(ValueError):
            movie_io.open_movie(self.d / 'm.avi')


class Cli(unittest.TestCase):

    def test_blocks_cover_every_frame_once(self):
        p = parser()
        frames = []
        for k in range(4):
            a = p.parse_args(['segment', 'run', '--block', f'{k}/4'])
            lo, hi = _block(a, 241)
            frames += list(range(lo, hi))
        self.assertEqual(frames, list(range(241)))
        a = p.parse_args(['segment', 'run', '--frames', '10:20'])
        self.assertEqual(_block(a, 241), (10, 20))

    def test_position_folder_names(self):
        self.assertEqual(_slug(1), 'position_1')
        self.assertEqual(_slug('Position 2'), 'position_2')

    def test_every_command_parses(self):
        p = parser()
        for argv in (['run', 'm.nd2', '--out', 'r', '--hand', 'w.xlsx', '--hand-type', 'y_wt'],
                     ['prepare', 'm.nd2', '--out', 'r', '--position', '2'],
                     ['track', 'r'], ['growth', 'r', '--frame-minutes', '3'], ['info', 'm.nd2'],
                     ['septin', 'run', 'in', '--out', 'r', '--min-length-um', '8.5'],
                     ['septin', 'segment', 'a', 'b', '--out', 'r'],
                     ['septin', 'measure', 'in', '--out', 'r', '--masks', 'm.h5'],
                     ['septin', 'stats', 'r', '--exclude', 'labels.csv']):
            self.assertTrue(callable(p.parse_args(argv).fn), argv)


if __name__ == '__main__':
    unittest.main()
