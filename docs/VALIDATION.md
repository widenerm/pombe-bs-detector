# Validation summary

Everything BS-Detector claims, the evidence for it, and what is not yet shown. The
annotations, scripts and full reports are in [`validation/`](../validation/README.md).
Two lab experts did the annotations. They appear here as **Annotator A** and **Annotator B**,
and they work independently: neither sees the other's points until both are finished.
Images are 0.0647 µm/px, so 5 px = 0.32 µm.

## 1. Birth-scar position on single frames (development set)

80 cells in 4 wild-type movies (two imaging days), annotated by Annotator A, then again
by Annotator B, who left 3 obscured scars blank.
[Full report](../validation/reports/scar_v3/INTERRATER.md).

| 77 annotations, 61 cells | Within 5 px | Within 3 px | Median |
|---|---:|---:|---:|
| Annotator A vs Annotator B | 61 (79%) | 30 | 3.4 px |
| **BS-Detector vs Annotator B** | **61 (79%)** | 43 | 2.8 px |
| BS-Detector vs Annotator B, each movie scored with weights fitted on the other three | 59 (77%) | 42 | 2.8 px |
| BS-Detector vs Annotator A | 56 (73%) | 29 | 3.5 px |
| First version (outline curvature) vs Annotator B | 18 (23%) | 10 | 8.0 px |

- **Why the jump from the first version.** Cellpose outlines smooth the scar away, so
  curvature of the outline finds it rarely. The current detector traces the membrane in
  the brightfield image itself
  ([design and ablations](../validation/reports/scar_v3/REPORT.md)).
- **The detector agrees with Annotator B as often as Annotator A does.** The 95% interval
  on that difference is ±13 percentage points, so the two are similar, not provably equal.
- **These numbers are optimistic.** The six detector weights were fitted on Annotator A's
  labels for these cells, and design choices were made while looking at them. Annotator B's
  labels remove the fit to one person's clicking, not the fit to these images. Refitting
  with each movie held out costs 2–3 cells.
- **Pole identity:** the two annotators put the new pole at the same physical end in 76 of
  77 annotations.

## 2. Blind test (pending)

Frozen before anyone ran the detector on these cells ([protocol](../validation/test_set/README.md)):

- 135 cells from 8 movies on two imaging days never used in development
  (3/14/2025 and 3/19/2025, four fields each).
- Frames and cells were drawn at random (seed 20261103) from every object that passes
  the size and shape filters, before the segmentation-quality flags, so bad outlines are
  scored too.
- The detector and its settings are fixed at release v1.0.0. They are scored once when
  the annotations return, and every result is reported, including cells with bad outlines
  or no visible scar. Changing the detector afterwards would need new movies.
- `tests/test_sample_regression.py` fails if any code change moves a single measurement
  on the sample movie.

## 3. Through time: new-end length and NETO (one movie, Annotator B)

Annotator B followed 6 wild-type cells from birth to division, every third frame (184
cell-frames), on a different microscope and camera (SlideBook, 0.11 µm/px, resampled to
0.0647). [Report](../validation/reports/neto_annotator_b/README.md).

| | |
|---|---|
| Annotated cell-frames matched to a detector cell | 179 / 184 |
| Same new/old pole labels | 177 / 179 |
| Scar midpoint within 5 px | 121 / 179 (68%); 87% in the first 30 min after birth |
| New-end length, mean absolute error | 0.30 µm (0.37 µm including 2 wrong-end frames) |

**NETO timing is not yet a fair test.** A flat-then-linear fit is unstable on traces
this noisy for both sources: dropping the first 10 minutes moves the annotator's own
estimate by up to 48 min. BS-Detector does not report NETO times. Annotator A's
cells for this comparison are still being annotated.

## 4. Growth curves vs hand tracking

The hand-tracked cells come from Annotator A's growth workbook. They are different cells,
from different movies of the same strain and medium, so this compares populations.
[Report](../validation/reports/growth_curves/README.md).

Mean growth since birth, µm, pipeline / hand-tracked:

| Wild type, YE (6 vs 5 cycles) | 60 min | 120 min | 150 min |
|---|---:|---:|---:|
| Whole cell | 2.18 / 2.16 | 4.83 / 4.64 | 5.85 / 5.87 |
| Old end | 1.31 / 1.90 | 3.01 / 3.91 | 3.40 / 4.68 |
| New end | 0.59 / 0.74 | 1.51 / 1.51 | 2.09 / 1.80 |

| *rgf1*Δ *gpd1*Δ, YE (5 vs 5 cycles) | 40 min | 80 min | 120 min |
|---|---:|---:|---:|
| Whole cell | 1.12 / 1.23 | 3.09 / 3.24 | 5.17 / 5.58 |
| Old end | 0.88 / 1.29 | 2.64 / 3.22 | 4.14 / 5.19 |
| New end | 0.07 / 0.80 | 0.29 / 0.90 | 0.78 / 1.09 |

- **Whole-cell growth matches** within one standard error at every time, in both strains.
- **The old end grows 1–1.3 µm less than in hand tracking** by late in the cycle. In wild
  type about half of that is because the hand-tracked old and new ends add up to more than
  the whole cell (0.9 µm on average at division).
- **Hand-tracked new ends gain about 0.6–0.8 µm in the first 30–40 min, then stay flat.**
  The pipeline's rise more slowly. Annotator B's through-time tracks show the same early
  jump right after birth, when the new pole rounds up. The pipeline misses most birth
  frames, because the daughters are not yet separate objects then.

## 5. Reproducibility

- **Pinned segmentation.** Every number uses Cellpose 4.0.9 with one set of weights,
  checked by SHA-256 before segmenting, in float32. Masks from CUDA GPUs and an Apple GPU
  were pixel-identical on most frames checked, and differed by 1 pixel on the rest.
- **Everything is recorded.** Each run writes the input, pixel size, software versions,
  weights and device to `run.json`. Each mask is tied to its frame by a checksum.
- **The refactor changed nothing.** `bsd track` reproduced an earlier movie's
  measurements CSV byte for byte, and `bsd growth` reproduced the growth tables byte for
  byte.
