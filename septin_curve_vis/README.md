# Septin curvature visualization

This workflow analyzes paired brightfield and fluorescence ND2 files. It
segments cells from brightfield, fits the same periodic B-spline used by
BS-Detector, computes signed curvature, and samples z-projected fluorescence
along the same contour.

## Run

From the repository root:

```bash
python -m pip install -r requirements.txt
python -m septin_curve_vis.run_analysis \
  --input /Users/xwidener/Documents/pombe-bs-detector/septin_curve_vis/080724 \
  --output /Users/xwidener/Documents/pombe-bs-detector/septin_curve_vis/080724_results
```

The input directory should contain the downloaded ND2 files. Files are sorted
by their natural filename order and paired consecutively. Within each pair,
the smaller file is treated as brightfield, matching the acquisition note.

Outputs include:

- `overlays/`: transparent PNG curvature overlays on the fluorescence image;
- `profiles/`: diagnostic plots of curvature and fluorescence intensity;
- `measurements.csv`: one row per contour sample;
- `cell_summary.csv`: one row per analyzed cell/frame;
- `projections/`: optional summed fluorescence TIFFs.

The script processes one ND2 pair at a time, so the full 10 GB experiment is
not loaded into memory. The exported overlay PNGs have the same pixel canvas
as the source fluorescence image and can be placed directly over it.
