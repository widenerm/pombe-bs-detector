# Septin localization vs outline curvature

Do septins enrich where the cell outline is more curved? This module measures, for every
cell in a set of still images, the curvature of the outline and the fluorescence just
under it at the same 300 points, then tests the association within each cell. It reuses
BS-Detector's Cellpose segmentation, cell filters and contour fitting. Code:
[`septin_curve_vis/`](../septin_curve_vis/).

## Input

A folder of Nikon `.nd2` stills, in **pairs**: one brightfield file and one
fluorescence file per field (for example `..._001.nd2`, `..._002.nd2`, ...).

- Files are sorted by name (numbers in natural order) and paired consecutively. Within a
  pair, the **smaller file is taken as brightfield**. `bsd septin` prints every pair
  before measuring, so **check the printed pairs** the first time you run a new
  dataset.
- Brightfield: first Z plane. Fluorescence: sum over Z of channel `--fluor-channel`
  (default 0, Spn3-GFP in the 7/25, 8/7 and 8/9/2024 stills), minus the image's 10th
  percentile as background.
- Pixel size should be 0.0647 µm/px. The cell filters are in those pixels, and a warning
  is printed otherwise.

## Run

```bash
pip install "bs-detector @ git+https://github.com/widenerm/pombe-bs-detector"
bsd septin run path/to/080724 --out runs/septin_080724
```

Or step by step:

```bash
bsd septin segment path/to/080724 --out runs/septin_080724      # -> runs/septin_080724/masks.h5
bsd septin measure path/to/080724 --out runs/septin_080724      # -> measurements.csv, cell_summary.csv, figures
bsd septin stats   runs/septin_080724                           # -> statistics/
```

Segmentation takes about a minute per field on a laptop GPU. The [Colab
notebook](../notebooks/Septin_Curvature_Colab.ipynb) runs the same commands on Drive
folders.

### One segmentation, fixed cell IDs

A cell's ID (`<brightfield file>_f0000_l<label>`) comes from its Cellpose mask, and
Cellpose can number cells differently on another run. Anything that refers to cells by ID,
such as division labels or hand annotations, must come from **the same saved masks**.
So:

- `bsd septin segment` never recomputes a field already in the masks file. Rerunning it
  only adds new fields.
- To share IDs across several dates, segment them all into one file and point every later
  step at it:

  ```bash
  bsd septin segment path/to/072524 path/to/080724 path/to/080924 --out runs/septin --masks masks/septin_masks.h5
  bsd septin run path/to/080724 --out runs/septin_080724 --masks masks/septin_masks.h5
  ```

Masks made before November 2026 used Cellpose's default bfloat16 precision; newer ones use
float32, like the movie pipeline. Existing masks are kept as they are, so their IDs don't
change.

## Output

| File | Contents |
|---|---|
| `measurements.csv` | one row per outline point: `cell_id`, `contour_index`, position, `curvature`, `fluorescence_intensity` |
| `cell_summary.csv` | one row per cell: mean/max curvature and fluorescence, `length_px`, `length_um` |
| `statistics/statistics_report.md` | the results in words and tables (start here) |
| `statistics/association_summary.csv` | per curvature metric: number of cells, mean within-cell Spearman ρ with 95% CI, permutation p, Wilcoxon p, cells significant after FDR |
| `statistics/cell_level_associations.csv` | per cell and metric: ρ, circular-shift p, FDR q |
| `statistics/*smoothing_sensitivity.*` | how the association changes with smoothing |
| `overlays/`, `profiles/`, `cell_crops/`, `frame_overlays/` | per-cell and per-field figures for checking outlines and profiles |

## The statistics

- **Quality control:** cells whose outline has any curvature above 0.10 px⁻¹
  (`--qc-threshold`) are dropped. These are usually segmentation artifacts.
- **Smoothing:** curvature and fluorescence are smoothed around the outline with a
  circular Gaussian of 3 points (1% of the perimeter; `--primary-sigma`). Sensitivity to
  this choice is reported separately.
- **Three curvature metrics:** signed (convex positive), positive part only (the
  hypothesis that septins enrich at convex regions), and absolute.
- **Cells are the unit, not points.** Points along one outline are autocorrelated, so
  each cell gets one Spearman ρ, tested against a null made by rotating its fluorescence
  profile around the outline (which keeps the autocorrelation). Cell effects are then
  combined: mean ρ with a bootstrap CI, a pooled rotation-permutation test, a Wilcoxon
  test across cells, and Benjamini–Hochberg FDR across cells.
- **Enrichment:** mean fluorescence at each cell's top 10% positive-curvature points
  minus its middle 20%.

## Leaving out dividing cells and other subsets

A dividing cell moves septin to its middle whatever the outline does, so dividing cells
confound the curvature association. `bsd septin stats` can leave cells out and
reports the result next to the all-cells result:

```bash
bsd septin stats runs/septin_080724 --exclude division_labels.csv
bsd septin stats runs/septin_080724 --min-length-um 8.5      # e.g. leave out newly separated cells
```

- `--exclude` takes a CSV with a `cell_id` column. If it also has a `label` column, as the
  labelling template does, only cells labelled `dividing` (or `1`, `yes`) are left out.
  Otherwise every listed cell is.
- `--min-length-um` / `--max-length-um` filter on `length_um` in `cell_summary.csv`. Rerun
  `bsd septin measure` on results made before November 2026, which have no lengths.
- With any filter, the filtered analysis goes to `statistics_filtered/`, and
  `statistics_comparison.md` puts the two side by side. The all-cells analysis is always
  run and never changes with filters.

**Dividing-cell labels.** `python -m septin_curve_vis.build_division_package` exports,
for each field, a 2-channel TIFF (brightfield and Sad1-mCherry max projection; Spn3-GFP
is left out so labels can't be chosen by septin), a mask whose pixel values are the
analysis cell IDs, a numbered preview and a `labels_template.csv`. The working rule is:
a cell is *dividing* from spindle-pole-body separation (two Sad1 dots) until the
daughters physically separate, so septated but still attached pairs count.

## Which outline? (curvature benchmark)

`python -m septin_curve_vis.curvature_benchmark` draws rod-shaped cells of known shape
(plain, bent, with a scar ridge, constricted) and renders them as brightfield using the
edge profile and noise of real cells. It segments them with Cellpose and scores each
outline method's curvature against the truth. "Shape error" is the RMS curvature error
against the true outline moved to the method's own mean offset, so it ignores where the
edge sits (48 cells, October 2026):

| Outline | Shape error, Cellpose masks (px⁻¹) | Shape error, perfect masks | Mean edge offset |
|---|---:|---:|---:|
| **Mask B-spline, s = 40, 300 points (used)** | **0.0084** | **0.0088** | +4.2 px (outside) |
| Scar detector's mask spline | 0.0087 | 0.0112 | +4.2 px |
| Membrane traced in brightfield, refit | 0.0104–0.0117 | 0.0140–0.0172 | +0.9 px |

- **The s = 40 spline wins on shape.** It had the lower error in 91% of cells against
  the traced membrane, and in 83% against the scar detector's spline. Tracing puts the
  edge closer to the true membrane but adds about 1 px of wiggle.
- **Constrictions are never measured.** Cellpose split all 12 constricted cells in two.
  This is another reason to leave dividing cells out.

## Open question: where the intensity is sampled

Brightfield outlines include the bright halo, so they sit about 2–4 px outside the
membrane. Fluorescence is sampled in a ±2 px band (`--band-width`) centred on the
outline, which may lie partly outside the cortex. This affects absolute enrichment
values more than within-cell rank correlations. Sampling about 2 px further in, or at
the traced membrane, has not been tested yet.
