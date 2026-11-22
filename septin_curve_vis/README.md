# septin_curve_vis

Septin intensity vs outline curvature on paired brightfield/fluorescence ND2 stills.
Run it with `bsd septin ...`; the full guide (inputs, outputs, statistics, cell
filters, the contour benchmark and open questions) is [docs/SEPTIN.md](../docs/SEPTIN.md).

| Module | Step |
|---|---|
| `segment_fields.py` | one saved Cellpose segmentation for all fields (`bsd septin segment`) |
| `run_analysis.py` | curvature and fluorescence along each outline (`bsd septin measure`) |
| `analyze_statistics.py` | within-cell association tests and cell filters (`bsd septin stats`) |
| `build_division_package.py` | dividing / not-dividing labelling package (brightfield + Sad1 only) |
| `curvature_benchmark.py` | synthetic benchmark that chose the contour method |
