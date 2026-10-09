# Birth-scar detector v3: profile mode

Validation against Carmen's two packets (cr_v1 + cr_v2, 80 annotated cells,
78 frames, 4 videos). Error = distance from the predicted scar midpoint to
Carmen's point.

## Result

| Detector | Within 5 px | Within 3 px | Within 10 px | Median |
|---|---:|---:|---:|---:|
| Previous default (legacy windowed curvature) | 16/80 (20%) | 11 | 45 | 8.9 px |
| **Profile mode, in-sample** | **57/80 (71%)** | 29 | 69 | 3.5 px |
| Profile mode, held out (leave-one-video-out) | 54/80 (68%) | 30 | – | 3.6 px |
| Profile mode, held out (leave-one-packet-out) | 54/80 (68%) | 31 | – | 3.4 px |

- Identical (57/80) on a second, independent segmentation (float32 V100 masks
  vs bfloat16 H200 masks), so the result is not tied to one mask set.
- The 90% target was **not** reached. See "Ceiling" below for why further
  tuning on these 80 cells is unlikely to get there.

## How it works (pombe_tracker/scar_profile.py)

Every 1-px cross-section from each pole is a candidate; the prediction is the
midpoint of the best one. Score = sum of named terms, lengths in units of the
mid-cell width W (so they transfer to other morphologies):

| Config parameter | Value | Meaning |
|---|---:|---|
| SCAR_PRIOR_OFFSET, SCAR_PRIOR_SLOPE | −0.740, 0.374 | expected scar distance from the new pole, d/W = offset + slope·(L/W) |
| SCAR_PRIOR_SD | 0.194 | spread of that expectation, in W |
| SCAR_HINT_WEIGHT | 2.01 | bonus for the end touching the sibling (neighbor-derived new pole) |
| SCAR_MEMBRANE_WIDTH_WEIGHT | 44.2 | scar ridge = widest point of the end, using the membrane traced in the brightfield image |
| SCAR_MEMBRANE_RIDGE_WEIGHT | 18.3 | ridge/corner in that width profile |
| SCAR_CONFIDENCE_FLAG | 0.5 | below this, `scar_needs_review = True` |

The mask boundary alone does not carry the scar: Cellpose smooths it away.
The decisive signal is the membrane line traced in the image along each
contour normal.

Outputs added to the CSV: `scar_confidence`, `scar_needs_review`,
`scar_pole_conflict` (scar nearer the assigned old pole than the new pole).

## Alternative: bilateral windowed curvature (`SCAR_DETECTOR_MODE = 'bilateral'`)

Averages the curvature of the image-traced membrane over an axial window
(SCAR_BILATERAL_WINDOW_W) on each flank and combines the flanks with
SCAR_BILATERAL_COMBINE. Within 5 px on the 80 cells (in-sample with fixed
defaults; nested held-out for 'min' is 52/80 by video, 50/80 by packet):

| SCAR_BILATERAL_COMBINE | Within 5 px |
|---|---:|
| 'min' (both flanks must bulge, default) | 56/80 |
| 'sum' (mean of flanks, tolerant of lopsided scars) | 44/80 |
| 'max' (either flank) | 36/80 |

Confidence is in z units (score + margin); SCAR_BILATERAL_CONFIDENCE_FLAG = 1.5
flags about 20% of cells. Not covered by scar_fit.py (parameters were chosen
by grid search).

## Recalibrating for mutants

1. Annotate a few dozen cells with GROUND_TRUTH_PROTOCOL.md.
2. `python -m validation.scar_cells --cache <masks> --out cells.pkl`
3. `python -m validation.scar_fit cells.pkl` prints held-out accuracy and the
   Config values to use. It reproduces the values above on Carmen's data.
4. `python -m validation.scar_eval cells.pkl --per-cell` scores any setting.

## What was tried (held-out within 5 px, out of 80)

| Approach | Result |
|---|---:|
| Summed windowed curvature across both flanks, mask contour (user's idea as stated) | 6 |
| Same bilateral window on the image-traced membrane, flanks combined by min | 52–56 |
| 6-term linear model (shipped) | 54 |
| Linear ranker, 37 features | 43–46 |
| 1D CNN on per-position signals | 45–47 |
| 2D CNN on straightened image strips (GPU, cluster) | 44–46 |
| Shipped model + multi-frame evidence (±1–3 frames) | 55–58 (in-sample) |

Black-box image models gave no advantage over the interpretable model.

## Ceiling

- Remaining misses are systematic, not frame noise: combining evidence over
  ±3 frames changes within-5 px by at most one cell.
- Annotation repeatability: the same cell annotated in adjacent frames differs
  by 0.8–2.3 px, and once by 4.3 px (cr_v1 f55 vs cr_v2 f54, same cell). A
  5 px tolerance leaves little room.
- Of the 23 in-sample misses, 5 are wrong-end calls (G f77 "septum maturing;
  IDK", AG10 f227 "used older BS", H0 f100, C10 f91, E0 f174); 18 are on the
  correct end, mostly 6–12 px off, with no systematic bias (median offset 0 px).
- Confidence is only moderately informative (AUC ≈ 0.65): flagging the
  least-confident 30% raises accuracy on the rest to ~77%.

## Suggested next steps

- Have a second scorer re-annotate the same 80 cells. This gives the true
  noise ceiling for a 5 px criterion.
- Ask Carmen to re-check H0 f100 (scar 42 px from the pole of a 122 px
  newborn cell) and the wrong-end cases above.
- More annotated cells, especially post-growth cells with scars 40–65 px from
  the pole, where most axial misses occur.

## Pipeline fixes made along the way

- Temporal stabilizer voted with every candidate instead of each frame's
  chosen scar, and measured positions along the PCA axis, whose sign is
  arbitrary per frame. It now uses one vote per frame, measured from the new
  pole. Before the fix it moved a correctly detected scar (20 px from the new
  pole in all 7 frames) to 84 px.
- Interpolated frames copied absolute coordinates from other frames. They now
  interpolate the position along each frame's own pole-to-pole line.

Per-cell results: per_cell_profile.csv, per_cell_legacy.csv.
