"""Tests for ground_truth schema, loaders, and septin scar derivation."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pombe_tracker.ground_truth import (
    GTRecord, load_ground_truth, write_ground_truth, write_template,
    septin_scar_position,
)


def test_record_accessors():
    r = GTRecord(frame=0, y=10.0, x=20.0, scar_y=11.0, scar_x=21.0,
                 new_pole_y=0.0, new_pole_x=0.0, old_pole_y=30.0, old_pole_x=40.0)
    assert np.allclose(r.centroid, [10.0, 20.0])
    assert np.allclose(r.scar_midpoint, [11.0, 21.0])
    assert r.has_poles
    r2 = GTRecord(frame=0, y=1.0, x=1.0)
    assert r2.scar_midpoint is None
    assert not r2.has_poles


def test_csv_roundtrip(tmp_path):
    recs = [
        GTRecord(frame=0, y=5.0, x=6.0, scar_present=True, scar_y=5.5, scar_x=6.5,
                 length=22.0, gt_id='A'),
        GTRecord(frame=1, y=7.0, x=8.0, scar_present=False),
    ]
    path = str(tmp_path / 'gt.csv')
    write_ground_truth(recs, path)
    loaded = load_ground_truth(path)
    assert set(loaded.keys()) == {0, 1}
    assert loaded[0][0].scar_present is True
    assert loaded[0][0].length == 22.0
    assert loaded[0][0].gt_id == 'A'
    assert loaded[1][0].scar_present is False


def test_blank_fields_parse_as_none(tmp_path):
    path = str(tmp_path / 'gt.csv')
    with open(path, 'w') as f:
        f.write('frame,y,x,scar_present,length\n')
        f.write('0,1.0,2.0,,\n')          # blank scar_present + length
    loaded = load_ground_truth(path)
    rec = loaded[0][0]
    assert rec.scar_present is None
    assert rec.length is None


def test_write_template_from_results(tmp_path):
    all_results = [{
        'frame_idx': 0,
        'cells': [{'centroid': (10.0, 20.0), 'cell_name': 'A'}],
    }]
    path = str(tmp_path / 'template.csv')
    write_template(path, all_results)
    loaded = load_ground_truth(path)
    assert loaded[0][0].y == 10.0
    assert loaded[0][0].gt_id == 'A'


def test_septin_scar_position_finds_bright_band():
    # Synthetic horizontal cell; bright fluorescence band at x=30 (the septum).
    H, W = 60, 80
    fluor = np.zeros((H, W), dtype=float)
    fluor[:, 29:32] = 100.0

    # Contour points spanning x in [10, 50] at y=30 (centreline), two rails.
    xs = np.linspace(10, 50, 100)
    top = np.stack([np.full_like(xs, 25.0), xs], axis=1)
    bot = np.stack([np.full_like(xs, 35.0), xs], axis=1)
    smooth_pts = np.vstack([top, bot])
    center = smooth_pts.mean(axis=0)
    axis = np.array([0.0, 1.0])  # long axis along x
    long_proj = (smooth_pts - center) @ axis
    long_norm = (long_proj - long_proj.min()) / (long_proj.max() - long_proj.min())

    mid, conf = septin_scar_position(fluor, center, axis, long_norm, smooth_pts)
    assert mid is not None
    assert abs(mid[1] - 30.0) < 3.0     # found the septum near x=30
    assert conf > 2.0                    # clearly above background
