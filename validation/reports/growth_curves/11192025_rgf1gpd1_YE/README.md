# Growth curves: rgf1Δ gpd1Δ in YE, 11/19/2025

Movie `11_19_2025_rgf1DEL_gpd1DEL_YE_.nd2` (lab storage, read only): one
brightfield field, 2304 × 2304, 271 frames at 2 min (9 h), 0.0647 µm/px. Copied
frame for frame to `data/experiments/experiment_11192025_rgf1DEL_gpd1DEL_YE.h5`,
segmented with the pinned Cellpose-SAM weights (float32), and run through the
unchanged pipeline (`bsd track`, frozen scar detector).

Hand-tracked comparison: Annotator A's workbook
`rawdata-growthrate-rgf1del-gpd1del-sorbitol.xlsx` (Annotator A),
converted with `pombe_tracker/hand_tracking.py` to `hand_tracked_growth.csv` (all
strains, YE and sorbitol). The figures use the five **YE** rgf1Δ gpd1Δ cells
(`y_rgf1delgpd1del`), the same condition as the movie. Their values match
Annotator A's own R input (`growthcurves_reorganized_forR.csv`) exactly. The sheets
number frames from 1 per cell and give no positions, so this is a comparison of
populations, not of matched cells.

## Figures

- `growth_since_birth.pdf/.png`: growth since birth (length minus length at
  birth, Annotator A's "Δ length"), whole cell, old end and new end on one axis.
  (A) pipeline mean ± SE (solid) and hand-tracked mean (dashed); (B) each cycle.
- `growth_since_birth_vs_hand.pdf/.png`: one panel per end, SE on both.
- `growth_curves.pdf/.png`: length; `growth_deltas.pdf/.png`: change per frame.

## Numbers

Five complete cycles: CF1, EC1, H1, H10, K0.

| | pipeline, median (range) | hand-tracked, median (range) |
|---|---:|---:|
| Length at birth (µm) | 8.8 (7.9–9.9) | 10.5 (8.2–11.1) |
| Length at division / last tracked (µm) | 15.0 (14.2–18.1) | 15.5 |
| Cycle / time tracked (min) | 188 (134–210) | 132 (108–158) |
| Elongation, first 75% of cycle (µm/h) | 2.79 (2.59–3.57) | |

Growth since birth, mean ± SE (pipeline vs hand):

| | 40 min | 80 min | 120 min |
|---|---:|---:|---:|
| Whole cell | 1.12 ± 0.20 vs 1.23 ± 0.14 | 3.09 ± 0.39 vs 3.24 ± 0.33 | 5.17 ± 0.46 vs 5.58 ± 0.68 |
| Old end | 0.88 ± 0.22 vs 1.29 ± 0.30 | 2.64 ± 0.39 vs 3.22 ± 0.45 | 4.14 ± 0.44 vs 5.19 ± 0.65 |
| New end | 0.07 ± 0.06 vs 0.80 ± 0.10 | 0.29 ± 0.14 vs 0.90 ± 0.04 | 0.78 ± 0.37 vs 1.09 ± 0.12 |

- **Whole cell matches** hand tracking within one SE at every time.
- **Almost all growth is at the old end** in both, with the new end adding
  ~1 µm over the cycle.
- **Old end runs ~0.5–1 µm below hand tracking** by 80–120 min, as in wild type,
  where part of the gap was Annotator A's old + new end summing to more than the
  whole cell.
- **New end:** the hand-tracked new end gains ~0.8 µm in the first 40 min and then
  stays flat; the pipeline's stays near zero. The same early difference appears
  in wild type (`../README.md`).
- **Cycles differ in length:** the hand-tracked cells are born longer (10.5 vs
  8.8 µm) and are tracked for less time (132 vs 188 min). Whether that is the
  cells chosen or where tracking starts and stops (separation vs septation) is a
  question for Annotator A.
- **Compartments are not validated** against hand tracking frame by frame. The
  steps in panel B (CF1 near 175 min, EC1 and H10 near 120 min) are the scar mark
  jumping between frames. The review flag is on for 64–67% of frames in CF1 and
  EC1, 48–49% in H1 and H10, and 1% in K0.

## Cycle selection

Eight lineage cycles passed the real-division rule; three were tracking errors
and are rejected by two rules added to `growth_curves.complete_cycles`:

- EG0 (12 min) and HM0 (16 min): a mask merges two cells for a few frames and then
  "divides". Rule: a cycle must last at least 60 min.
- G1: grows to 15 µm, drops to 9.4 µm and grows again; the tracker missed a
  division. Rule: the 5-frame running median must never fall below 75% of itself.

The rules leave all six 3/27 wild-type cycles unchanged.

## Segmentation

Masks came from four devices: frames 0–63 and 197–203 on the laptop GPU (Apple M2,
MPS), the rest on cluster GPUs (Tesla V100 ×2 types, T4); see
`segmentation_provenance.json`. Same weights (sha256 e144…abe2), float32, no
diameter. Cross-checks: on 3/27 frames 60 and 120 the laptop masks are
pixel-identical to the cluster's; on frames 197–203 of this movie, five are
identical and two differ by one pixel each (of 5.3 M).

## Reproduce

```bash
bsd segment 11_19_2025_rgf1DEL_gpd1DEL_YE_.nd2 --out runs/rgf1gpd1_YE     # GPU; see docs/HPC.md for a cluster array job
bsd track runs/rgf1gpd1_YE
bsd growth runs/rgf1gpd1_YE --hand rawdata-growthrate-rgf1del-gpd1del-sorbitol.xlsx --hand-type y_rgf1delgpd1del
```

These figures were made with the scripts `bsd` replaced. From the same measurements,
`bsd growth` reproduces the tables byte for byte (and the figures, with the same
matplotlib version); from the same masks,
`bsd track` reproduced the 020123 movie's measurements CSV byte for byte.
