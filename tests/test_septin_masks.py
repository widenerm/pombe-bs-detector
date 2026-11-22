"""Checks for saved septin masks and the division-classifier package files."""
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import tifffile

from pombe_tracker.config import Config
from pombe_tracker.pipeline import CellProcessor
from septin_curve_vis.build_division_package import write_tiffs
from septin_curve_vis.run_analysis import analysis_cells, image_sha256, load_mask, mask_key


def rod_labels():
    """Two horizontal rods, one fully inside the frame and one cut by the border."""
    labels = np.zeros((120, 200), dtype=np.uint16)
    yy, xx = np.mgrid[:120, :200]
    labels[((xx - 80) / 50.0) ** 2 + ((yy - 40) / 12.0) ** 2 <= 1] = 3
    labels[((xx - 10) / 50.0) ** 2 + ((yy - 95) / 12.0) ** 2 <= 1] = 7
    return labels


class SavedMaskTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.masks = Path(self.dir.name) / 'masks.h5'
        self.bf_path = Path('Spn3-GFP_Sad1-mCh_080724_001.nd2')
        self.bf = np.random.default_rng(0).random((120, 200)).astype(np.float32)
        with h5py.File(self.masks, 'w') as f:
            ds = f.create_dataset(mask_key(self.bf_path, 0), data=rod_labels())
            ds.attrs['image_sha256'] = image_sha256(self.bf)

    def tearDown(self):
        self.dir.cleanup()

    def test_key_matches_cell_id_prefix(self):
        self.assertEqual(mask_key(self.bf_path, 0), 'Spn3-GFP_Sad1-mCh_080724_001_f0000')

    def test_load_returns_saved_labels(self):
        labels = load_mask(self.masks, self.bf_path, 0, self.bf)
        np.testing.assert_array_equal(labels, rod_labels())

    def test_load_rejects_a_different_brightfield(self):
        with self.assertRaises(ValueError):
            load_mask(self.masks, self.bf_path, 0, self.bf + 1)

    def test_load_reports_missing_frame(self):
        with self.assertRaises(KeyError):
            load_mask(self.masks, self.bf_path, 1, self.bf)

    def test_analysis_keeps_whole_cells_only(self):
        cfg = Config()
        kept = [r['label'] for r in analysis_cells(rod_labels(), self.bf, cfg, CellProcessor(cfg))]
        self.assertEqual(kept, [3])


class PackageTiffTests(unittest.TestCase):
    def test_channels_mask_and_pixel_size_round_trip(self):
        bf = np.full((40, 60), 1000, dtype=np.uint16)
        sad1 = np.full((40, 60), 7, dtype=np.uint16)
        cells = rod_labels()[:40, :60]
        with tempfile.TemporaryDirectory() as d:
            image, mask = Path(d) / 'image.tif', Path(d) / 'cells.tif'
            write_tiffs(image, mask, bf, sad1, cells, 0.0647)
            stack = tifffile.imread(image)
            self.assertEqual(stack.shape, (2, 40, 60))
            np.testing.assert_array_equal(stack[0], bf)
            np.testing.assert_array_equal(stack[1], sad1)
            np.testing.assert_array_equal(tifffile.imread(mask), cells)
            with tifffile.TiffFile(image) as tif:
                num, den = tif.pages[0].tags['XResolution'].value
                self.assertAlmostEqual(den / num, 0.0647, places=6)
                self.assertEqual(tif.imagej_metadata['unit'], 'um')


if __name__ == '__main__':
    unittest.main()
