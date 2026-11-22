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
| Pixel size | 0.0647 µm / px (64.7 nm): Nikon Ti2, 100×/1.49 objective, Fusion camera, 1×1 binning |
| Source | First 5 frames of `03272025_wt_ymd494__xy1` (wild type, 2-min frame interval) |

`masks/0000.npz` ... `0004.npz` are the validated Cellpose-SAM masks of these frames
(Cellpose 4.0.9, pinned `cpsam` weights, float32, CUDA). The regression test
`tests/test_sample_regression.py` runs the pipeline on them without a GPU.

## Usage

```bash
bsd run data/sample/sample.h5 --out runs/sample
```

The Colab notebook downloads this file when no movie is given. Five frames are too
few for a complete cell cycle, so the run gives measurements and QC figures but no
growth curves.

## Format

Any `.h5` with one 3-D dataset (`frames`, or the only 3-D dataset) of shape
`(frames, height, width)`, single-channel brightfield. An attribute `um_per_px` gives
the pixel size; without it 0.0647 µm/px is assumed. `bsd` also reads `.nd2`, SlideBook
`.sldy` and `.tif` directly, so converting to H5 by hand is not needed.
