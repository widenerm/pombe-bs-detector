"""End-to-end regression: the full pipeline on the bundled 5-frame sample, from its saved masks.

The masks (data/sample/masks/) are the validated Cellpose-SAM masks of those frames, so this
runs without Cellpose or a GPU. Any change to tracking, pole assignment, the scar detector or
scar stabilisation that moves a measurement shows up here. The detector is frozen for the
blind test (validation/test_set/README.md): if this fails, the change needs a new test set.
"""
import math
import unittest
from pathlib import Path

import h5py
import pandas as pd

from pombe_tracker.config import Config
from pombe_tracker.io_utils import export_csv
from pombe_tracker.pipeline import run_pipeline
from pombe_tracker.postprocessing import stabilize_scars
from pombe_tracker.segmentation import SavedMasks
from pombe_tracker.tracking import CellTracker

REPO = Path(__file__).resolve().parents[1]
EXPECTED = REPO / 'tests/data/sample_expected.csv'


class SampleRegression(unittest.TestCase):

    def test_sample_measurements_unchanged(self):
        import contextlib
        import io
        import tempfile
        cfg = Config()
        with h5py.File(REPO / 'data/sample/sample.h5', 'r') as f:
            frames = f['frames'][:]
        with contextlib.redirect_stdout(io.StringIO()):
            res = run_pipeline(frames, cfg, tracker=CellTracker(cfg), segmenter=SavedMasks(REPO / 'data/sample/masks', 5))
            res, _ = stabilize_scars(res, cfg)
            with tempfile.TemporaryDirectory() as d:
                out = Path(d) / 'm.csv'
                export_csv(res, out, columns=cfg.CSV_COLUMNS, pixel_size_um=cfg.PIXEL_SIZE_UM)
                got = pd.read_csv(out)
        want = pd.read_csv(EXPECTED)
        self.assertEqual(list(got.columns), list(want.columns))
        self.assertEqual(len(got), len(want))
        for col in want.columns:
            for i, (g, w) in enumerate(zip(got[col], want[col])):
                if isinstance(w, float) and isinstance(g, float):
                    if math.isnan(w):
                        self.assertTrue(math.isnan(g), f'{col} row {i}')
                    else:
                        self.assertTrue(math.isclose(g, w, rel_tol=1e-6, abs_tol=1e-6), f'{col} row {i}: {g} != {w}')
                elif isinstance(w, str) and w.startswith('['):
                    gv = [float(x) for x in g.strip('[]').split(',')]
                    wv = [float(x) for x in w.strip('[]').split(',')]
                    for a, b in zip(gv, wv):
                        self.assertTrue(math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6), f'{col} row {i}')
                else:
                    self.assertEqual(g, w, f'{col} row {i}')


if __name__ == '__main__':
    unittest.main()
