"""Tests for the evaluation harness (matching, metrics, objective)."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pombe_tracker.ground_truth import GTRecord
from pombe_tracker.evaluation import (
    evaluate, match_frame, EvalConfig, DetCell, export_eval_json,
)


def _det(frame, y, x, scar=False, scar_mp=None, new_pole=None, old_pole=None,
         length=None, name=None):
    return DetCell(frame=frame, name=name, centroid=np.array([y, x], float),
                   scar_detected=scar,
                   scar_midpoint=None if scar_mp is None else np.array(scar_mp, float),
                   new_pole=None if new_pole is None else np.array(new_pole, float),
                   old_pole=None if old_pole is None else np.array(old_pole, float),
                   length=length, seg_quality='ok')


def test_match_frame_basic():
    dets = [_det(0, 10, 10), _det(0, 50, 50)]
    gts  = [GTRecord(frame=0, y=11, x=11), GTRecord(frame=0, y=52, x=49)]
    pairs, ud, ug = match_frame(dets, gts, max_dist=40.0)
    assert len(pairs) == 2 and not ud and not ug


def test_match_frame_rejects_far():
    dets = [_det(0, 10, 10)]
    gts  = [GTRecord(frame=0, y=200, x=200)]
    pairs, ud, ug = match_frame(dets, gts, max_dist=40.0)
    assert not pairs and len(ud) == 1 and len(ug) == 1


def test_scar_detection_metrics():
    # frame dict form for live-results path
    all_results = [{
        'frame_idx': 0,
        'cells': [
            {'centroid': (10, 10), 'scar_detected': True, 'cell_name': 'A',
             'length': 20.0, 'debug_info': {}},   # TP
            {'centroid': (50, 50), 'scar_detected': False, 'cell_name': 'B',
             'length': 18.0, 'debug_info': {}},   # FN
            {'centroid': (90, 90), 'scar_detected': True, 'cell_name': 'C',
             'length': 19.0, 'debug_info': {}},   # FP
        ],
    }]
    gt = {0: [
        GTRecord(frame=0, y=10, x=10, scar_present=True),
        GTRecord(frame=0, y=50, x=50, scar_present=True),
        GTRecord(frame=0, y=90, x=90, scar_present=False),
    ]}
    rep = evaluate(all_results, gt)
    assert rep.scar_tp == 1 and rep.scar_fn == 1 and rep.scar_fp == 1
    assert abs(rep.scar_precision - 0.5) < 1e-9
    assert abs(rep.scar_recall - 0.5) < 1e-9
    assert abs(rep.scar_f1 - 0.5) < 1e-9


def test_localization_error():
    all_results = [{
        'frame_idx': 0,
        'cells': [{'centroid': (10, 10), 'scar_detected': True,
                   'scar_midpoint': (13, 14), 'cell_name': 'A',
                   'debug_info': {}}],
    }]
    gt = {0: [GTRecord(frame=0, y=10, x=10, scar_present=True,
                       scar_y=10, scar_x=10)]}
    rep = evaluate(all_results, gt)
    assert rep.loc_n == 1
    assert abs(rep.loc_mean - 5.0) < 1e-9   # 3-4-5 triangle


def test_pole_accuracy():
    # detector new/old poles swapped relative to GT on the second cell
    all_results = [{
        'frame_idx': 0,
        'cells': [
            {'centroid': (10, 10), 'scar_detected': False, 'cell_name': 'A',
             'debug_info': {'new_pole_point': (0, 0), 'old_pole_point': (20, 20)}},
            {'centroid': (50, 50), 'scar_detected': False, 'cell_name': 'B',
             'debug_info': {'new_pole_point': (60, 60), 'old_pole_point': (40, 40)}},
        ],
    }]
    gt = {0: [
        GTRecord(frame=0, y=10, x=10, new_pole_y=0, new_pole_x=0,
                 old_pole_y=20, old_pole_x=20),                       # correct
        GTRecord(frame=0, y=50, x=50, new_pole_y=40, new_pole_x=40,
                 old_pole_y=60, old_pole_x=60),                       # swapped
    ]}
    rep = evaluate(all_results, gt)
    assert rep.pole_n == 2
    assert abs(rep.pole_accuracy - 0.5) < 1e-9


def test_length_bland_altman():
    all_results = [{
        'frame_idx': 0,
        'cells': [
            {'centroid': (10, 10), 'scar_detected': False, 'length': 22.0,
             'cell_name': 'A', 'debug_info': {}},
            {'centroid': (50, 50), 'scar_detected': False, 'length': 18.0,
             'cell_name': 'B', 'debug_info': {}},
        ],
    }]
    gt = {0: [
        GTRecord(frame=0, y=10, x=10, length=20.0),
        GTRecord(frame=0, y=50, x=50, length=20.0),
    ]}
    rep = evaluate(all_results, gt)
    assert rep.len_n == 2
    assert abs(rep.len_bias - 0.0) < 1e-9     # (+2, -2) -> mean 0
    assert abs(rep.len_mae - 2.0) < 1e-9


def test_objective_in_unit_interval():
    all_results = [{
        'frame_idx': 0,
        'cells': [{'centroid': (10, 10), 'scar_detected': True,
                   'scar_midpoint': (10, 10), 'length': 20.0, 'cell_name': 'A',
                   'debug_info': {'new_pole_point': (0, 0),
                                  'old_pole_point': (20, 20)}}],
    }]
    gt = {0: [GTRecord(frame=0, y=10, x=10, scar_present=True, scar_y=10, scar_x=10,
                       new_pole_y=0, new_pole_x=0, old_pole_y=20, old_pole_x=20,
                       length=20.0)]}
    rep = evaluate(all_results, gt)
    assert 0.0 <= rep.objective <= 1.0
    assert rep.objective > 0.9               # near-perfect detection
    assert set(rep.objective_components) == {'scar_f1', 'localization', 'pole', 'length'}


def test_export_and_reload_json(tmp_path):
    all_results = [{
        'frame_idx': 0,
        'cells': [{'centroid': (10, 10), 'scar_detected': True,
                   'scar_midpoint': (10, 10), 'length': 20.0, 'cell_name': 'A',
                   'debug_info': {'new_pole_point': (0, 0),
                                  'old_pole_point': (20, 20)}}],
    }]
    path = str(tmp_path / 'detector.json')
    export_eval_json(all_results, path)
    gt = {0: [GTRecord(frame=0, y=10, x=10, scar_present=True)]}
    rep = evaluate(path, gt)                  # evaluate accepts a JSON path
    assert rep.scar_tp == 1


def test_empty_inputs_do_not_crash():
    rep = evaluate([], {})
    assert rep.objective == 0.0
    assert rep.n_matched == 0
