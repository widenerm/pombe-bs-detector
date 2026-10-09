"""Synthetic checks of the profile-mode scar localizer and stabilizer fixes."""
import unittest

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.measure import find_contours

from pombe_tracker.config import Config
from pombe_tracker.detection import BirthScarDetector
from pombe_tracker.postprocessing import _interpolate_frames, _normalized_position_from_midpoint

ROW, C0, C1, R = 100.0, 70.0, 250.0, 30.0   # capsule: axis along columns


def capsule(ridge_col=None, ridge_height=0.0, shape=(200, 320)):
    rr, cc = np.mgrid[:shape[0], :shape[1]].astype(float)
    nearest = np.clip(cc, C0, C1)
    dist = np.hypot(rr - ROW, cc - nearest)
    radius = np.full(shape, R)
    if ridge_col is not None:
        radius += ridge_height * np.exp(-0.5 * ((cc - ridge_col) / 5.0) ** 2)
    return dist <= radius


def rod(ridge_col):
    """Mask without the ridge (Cellpose smooths it away); image shows it as a
    dark membrane line with an outward ridge on both flanks."""
    mask = capsule()
    bumped = capsule(ridge_col, 2.0)
    edge = distance_transform_edt(bumped) + distance_transform_edt(~bumped)  # distance to boundary
    rng = np.random.default_rng(0)
    image = 1000 - 600 * np.exp(-0.5 * (edge / 1.2) ** 2) + rng.normal(0, 10, mask.shape)
    contour = max(find_contours(mask.astype(float), 0.5), key=len)
    return contour, image


def flat_prior_config():
    cfg = Config()
    cfg.SCAR_DETECTOR_MODE = 'profile'
    cfg.SCAR_PRIOR_SD = 100.0     # effectively no distance prior
    cfg.SCAR_HINT_WEIGHT = 0.0
    return cfg


class ProfileDetectorTests(unittest.TestCase):
    def test_finds_membrane_ridge_invisible_in_mask(self):
        pole = C0 - R
        for d in (25.0, 40.0):
            with self.subTest(d=d):
                contour, image = rod(pole + d)
                pair, dbg = BirthScarDetector(flat_prior_config()).detect(contour, image=image)
                mid = np.mean(pair, axis=0)
                self.assertLess(np.linalg.norm(mid - [ROW, pole + d]), 3.0)
                self.assertTrue(0.0 <= dbg['scar_confidence'] <= 1.0)
                self.assertIn('membrane_width', dbg['scar_terms'])

    def test_ridge_at_other_end_switches_end(self):
        pole = C1 + R
        contour, image = rod(pole - 30)
        pair, _ = BirthScarDetector(flat_prior_config()).detect(contour, image=image)
        self.assertLess(abs(np.mean(pair, axis=0)[1] - (pole - 30)), 3.0)

    def test_distance_prior_is_tunable(self):
        contour, image = rod(None)    # no ridge: the prior decides
        cfg = flat_prior_config()
        cfg.SCAR_PRIOR_SD = 0.05
        cfg.SCAR_MEMBRANE_WIDTH_WEIGHT = cfg.SCAR_MEMBRANE_RIDGE_WEIGHT = 0.0
        W = 2 * R
        for target in (0.4, 0.7):
            with self.subTest(target=target):
                cfg.SCAR_PRIOR_SLOPE = 0.0
                cfg.SCAR_PRIOR_OFFSET = target
                _, dbg = BirthScarDetector(cfg).detect(contour, image=image)
                self.assertAlmostEqual(dbg['scar_distance_px'] / W, target, delta=0.03)

    def test_hint_selects_end_when_evidence_is_symmetric(self):
        contour, image = rod(None)
        cfg = flat_prior_config()
        cfg.SCAR_HINT_WEIGHT = 2.0
        for pole in ([ROW, C0 - R], [ROW, C1 + R]):
            pair, _ = BirthScarDetector(cfg).detect(contour, new_pole_point=np.array(pole), image=image)
            self.assertLess(abs(np.mean(pair, axis=0)[1] - pole[1]), (C1 - C0) / 2)

    def test_works_without_image(self):
        contour, _ = rod(None)
        pair, dbg = BirthScarDetector(Config()).detect(contour)
        self.assertIsNotNone(pair)
        self.assertFalse(dbg['used_image'])


class BilateralModeTests(unittest.TestCase):
    def config(self, combine):
        cfg = Config()
        cfg.SCAR_DETECTOR_MODE = 'bilateral'
        cfg.SCAR_BILATERAL_COMBINE = combine
        cfg.SCAR_BILATERAL_PRIOR_SD = 100.0
        cfg.SCAR_BILATERAL_BLUNTNESS_WEIGHT = 0.0
        return cfg

    def test_each_combine_rule_finds_bilateral_ridge(self):
        pole = C0 - R
        contour, image = rod(pole + 40)
        for combine in ('min', 'sum', 'max'):
            with self.subTest(combine=combine):
                pair, dbg = BirthScarDetector(self.config(combine)).detect(contour, image=image)
                self.assertLess(np.linalg.norm(np.mean(pair, axis=0) - [ROW, pole + 40]), 4.0)
                self.assertEqual(dbg['detector_mode'], 'bilateral')
                self.assertIn('bilateral_kink', dbg['scar_terms'])

    def test_hint_restricts_search_to_that_end(self):
        contour, image = rod(C0 - R + 40)     # ridge at the left end
        pair, _ = BirthScarDetector(self.config('min')).detect(
            contour, new_pole_point=np.array([ROW, C1 + R]), image=image)
        self.assertGreater(np.mean(pair, axis=0)[1], (C0 + C1) / 2)

    def test_unknown_combine_rule_rejected(self):
        contour, image = rod(None)
        with self.assertRaises(ValueError):
            BirthScarDetector(self.config('median')).detect(contour, image=image)


class StabilizerTests(unittest.TestCase):
    @staticmethod
    def result(new_pole, old_pole, mid, axis_sign=1):
        return dict(scar_detected=True, scar_midpoint=np.array(mid, float),
                    debug_info=dict(center=np.array([0., 0.]), axis=np.array([0., axis_sign], float),
                                    new_pole_point=np.array(new_pole, float),
                                    old_pole_point=np.array(old_pole, float)))

    def test_position_measured_from_new_pole_regardless_of_axis_sign(self):
        a = self.result([0, -100], [0, 100], [0, -70], axis_sign=1)
        b = self.result([0, -100], [0, 100], [0, -70], axis_sign=-1)
        self.assertAlmostEqual(_normalized_position_from_midpoint(a, a['scar_midpoint']), 0.15)
        self.assertAlmostEqual(_normalized_position_from_midpoint(b, b['scar_midpoint']), 0.15)

    def test_interpolation_follows_the_moving_cell(self):
        frames = [self.result([0, -100 + s], [0, 100 + s], [0, -70 + s]) for s in (0, 50, 100)]
        frames[1]['scar_midpoint'] = np.array([0., 80.])   # outlier frame
        _interpolate_frames(frames, [1])
        np.testing.assert_allclose(frames[1]['scar_midpoint'], [0, -20], atol=1e-6)


if __name__ == '__main__':
    unittest.main()
