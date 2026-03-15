"""
io_utils.py  –  Data I/O: HDF5 loading and CSV export.
"""
import os
import csv
import numpy as np
import h5py


def load_h5_data(h5_path, dataset_key):
    """Load the image stack from an HDF5 file."""
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"H5 file not found: {h5_path}")
    with h5py.File(h5_path, 'r') as f:
        if dataset_key not in f:
            available = list(f.keys())
            raise KeyError(f"Key '{dataset_key}' not found.  Available: {available}")
        data = f[dataset_key][:]
    print(f"Loaded: {data.shape}  (frames × height × width)")
    return data


def export_csv(all_results, output_path, columns=None):
    """
    Write per-frame, per-cell measurements to a CSV file.

    Parameters
    ----------
    all_results : list returned by run_pipeline
    output_path : destination file path (.csv)
    columns     : list of column names; if None, uses a sensible default set
    """
    default_cols = [
        'cell_name', 'frame', 'length', 'width_centroid', 'width_scar',
        'new_end_length', 'old_end_length', 'area',
        'scar_detected', 'pole_method', 'pole_confidence',
    ]
    columns = columns or default_cols

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    rows = []
    for fd in all_results:
        fidx = fd['frame_idx']
        for cell in fd['cells']:
            row = {}
            for col in columns:
                # Check cell dict first, then debug_info
                if col in cell:
                    val = cell[col]
                elif col in cell.get('debug_info', {}):
                    val = cell['debug_info'][col]
                elif col == 'frame':
                    val = fidx
                else:
                    val = None

                # Convert numpy scalars to Python types for clean CSV output
                if isinstance(val, np.ndarray):
                    val = val.tolist()
                elif isinstance(val, np.floating):
                    val = float(val)
                elif isinstance(val, np.integer):
                    val = int(val)

                row[col] = val

            # Always include frame even if not in columns list
            row.setdefault('frame', fidx)
            rows.append(row)

    if not rows:
        print("No data to export.")
        return None

    # Collect all unique keys (in specified order)
    all_keys = list(dict.fromkeys(['frame'] + columns))

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV saved: {output_path}  ({len(rows)} rows)")
    return output_path
