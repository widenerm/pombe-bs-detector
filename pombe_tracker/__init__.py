"""
Pombe Birth Scar Tracker
Automated detection of birth scars and pole identity in S. pombe.

The full pipeline (run_pipeline, CellTracker, …) depends on heavy, optional
packages (scikit-image, h5py, and — at runtime — Cellpose/PyTorch).  Those are
imported defensively so that the lightweight, dependency-free evaluation and
ground-truth modules remain importable in a minimal environment (e.g. fast CI
that only scores a detector-export JSON against ground truth).
"""
from .config import Config

# Always-available, lightweight (numpy + scipy only):
from .ground_truth import GTRecord, load_ground_truth, write_template
from .evaluation import evaluate, EvalConfig, export_eval_json

__all__ = [
    'Config',
    'GTRecord', 'load_ground_truth', 'write_template',
    'evaluate', 'EvalConfig', 'export_eval_json',
]

# Heavy pipeline (needs scikit-image / h5py installed):
try:
    from .pipeline import run_pipeline
    from .io_utils import load_h5_data, export_csv
    from .tracking import CellTracker
    from .postprocessing import stabilize_scars, print_stability_report
    __all__ += [
        'run_pipeline', 'load_h5_data', 'export_csv',
        'CellTracker', 'stabilize_scars', 'print_stability_report',
    ]
except ImportError as _e:  # pragma: no cover
    import warnings
    warnings.warn(
        f"pombe_tracker: pipeline modules unavailable ({_e}). "
        "Install scikit-image and h5py for the full pipeline; the evaluation "
        "harness works without them.",
        stacklevel=2,
    )
