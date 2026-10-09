"""Mechanistic tests independent of Carmen's annotations."""
import unittest
import numpy as np
from pombe_tracker.config import Config
from validation.scar_ablations import AblationDetector, periodic_geometry


class ScoringTests(unittest.TestCase):
    def test_production_experimental_selection_uses_bounded_hint(self):
        from pombe_tracker.detection import BirthScarDetector, bounded_pole_prior
        cfg = Config(); cfg.SCAR_DETECTOR_MODE = 'experimental'
        candidates = [dict(points=([0., 0.], [2., 0.]), score=.1),
                      dict(points=([0., 50.], [2., 50.]), score=1.)]
        detector = BirthScarDetector(cfg)
        self.assertIs(detector._select_best_candidate(candidates, np.array([1., 0.])), candidates[1])
        self.assertAlmostEqual(candidates[0]['selection_score'], .12)
        self.assertAlmostEqual(candidates[1]['selection_score'], 1.)
        np.testing.assert_array_equal(bounded_pole_prior([[1, 0], [1, 50]]), [1, 1])
        with self.assertRaises(ValueError):
            bounded_pole_prior([[0, 0]], [0, 0], strength=2)

    def test_weak_hint_cannot_override_tenfold_evidence(self):
        candidates=[dict(points=(np.array([0.,0.]),np.array([2.,0.])),score=.1),
                    dict(points=(np.array([0.,50.]),np.array([2.,50.])),score=1.)]
        detector=AblationDetector(Config(),'weak_neighbor_hint')
        self.assertIs(detector._select_best_candidate(candidates,np.array([1.,0.])),candidates[1])
        self.assertTrue(all(1 <= c['hint_weight'] <= 1.2 for c in candidates))

    def test_legacy_comparison_retains_hard_hint(self):
        candidates=[dict(points=(np.array([0.,0.]),np.array([2.,0.])),score=.1),
                    dict(points=(np.array([0.,50.]),np.array([2.,50.])),score=1.)]
        detector=AblationDetector(Config(),'legacy')
        self.assertIs(detector._select_best_candidate(candidates,np.array([1.,0.])),candidates[0])

    def test_excess_rejects_constant_curvature(self):
        for mechanism in ('legacy','normalized_excess','smooth_cap_baseline'):
            with self.subTest(mechanism=mechanism):
                k=np.ones(300)*.04
                np.testing.assert_allclose(AblationDetector(Config(),mechanism)._local_curvature_excess(k),0,atol=1e-12)

    def test_normalized_excess_scale_invariance_above_floor(self):
        k=.02+.01*np.sin(np.linspace(0,8*np.pi,300,endpoint=False));k[50]+=.08
        detector=AblationDetector(Config(),'normalized_excess')
        np.testing.assert_allclose(detector._local_curvature_excess(k),detector._local_curvature_excess(k*3),atol=1e-10)

    def test_periodic_circle_curvature_and_uniform_sampling(self):
        theta=np.linspace(0,2*np.pi,501)
        contour=np.c_[20*np.sin(theta),20*np.cos(theta)]
        pts,k=periodic_geometry(contour,0,300)
        np.testing.assert_allclose(k,1/20,rtol=.002)
        steps=np.linalg.norm(np.roll(pts,-1,axis=0)-pts,axis=1)
        self.assertLess(steps.std()/steps.mean(),1e-5)
        _,reversed_k=periodic_geometry(contour[::-1],0,300)
        np.testing.assert_allclose(reversed_k,k,rtol=.002)


if __name__=='__main__':unittest.main()
