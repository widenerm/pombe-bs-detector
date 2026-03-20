# Sample Dataset

`sample.h5` contains a short excerpt of brightfield time-lapse microscopy of
*Schizosaccharomyces pombe* suitable for testing and demonstrating BS-Detector.

## Contents

| Property | Value |
|---|---|
| Frames | 5 |
| Dataset key | `frames` |
| Shape | `(5, H, W)` — frames × height × width |
| Dtype | uint16 |
| Pixel size | ~65 nm / px (0.65 µm / px) |

## Usage in Colab

The notebook auto-detects this file when `cfg.H5_FILE_PATH` is set to
`'sample'` (the default when running from a fresh clone).  No edits needed
for a quick test run.

To use your own data, update `cfg.H5_FILE_PATH` in Step 3 to point to your
HDF5 file on Google Drive.

## Format notes

BS-Detector expects a single HDF5 dataset with shape `(N, H, W)` where:
- `N` = number of frames
- `H`, `W` = image height and width in pixels
- Values are single-channel (grayscale) brightfield intensity

If your H5 file has a different dataset key than `frames`, update
`cfg.H5_DATASET_KEY` accordingly.  Use `h5py` to inspect available keys:

```python
import h5py
with h5py.File('your_file.h5', 'r') as f:
    print(list(f.keys()))
```
