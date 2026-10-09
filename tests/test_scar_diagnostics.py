"""Regression coverage for exporting current and legacy candidate records."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
from pombe_tracker.config import Config
from validation.scar_diagnostics import gallery


class GalleryTests(unittest.TestCase):
    def test_export_preserves_candidate_ids_and_handles_legacy_candidates(self):
        case = dict(record=dict(video_id='video', frame=7, cell_name='A',
                                error_px=8., failure='wrong-candidate selection'),
                    gt=np.array([1., 2.]),
                    result=dict(contour=np.array([[0., 0.], [1., 1.]]),
                                debug_info=dict(scar_candidates=[
                                    dict(candidate_id=12, score=1., points=(np.array([0., 0.]), np.array([1., 1.]))),
                                    dict(score=.5)])))
        with tempfile.TemporaryDirectory() as temp:
            with patch('validation.scar_diagnostics.plot_case', side_effect=lambda *args: plt.figure()):
                gallery([case], temp, Config())
            out = Path(temp)
            details = json.loads((out/'video_7_A.json').read_text())
            self.assertEqual([c['candidate_id'] for c in details['candidates']], [12, 1])
            self.assertEqual(details['candidates'][0]['points'], [[0., 0.], [1., 1.]])
            self.assertTrue((out/'video_7_A.png').exists())
            self.assertIn('video_7_A.json', (out/'index.html').read_text())


if __name__ == '__main__':
    unittest.main()
