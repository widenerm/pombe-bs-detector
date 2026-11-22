# NETO packet: Annotator B vs BS-Detector through time

**Scorer:** Annotator B, 6 cells (c1–c6), each followed from birth to division,
every 3rd frame (6 min), 28–35 frames per cell, 184 cell-frames. Raw CSVs:
`validation/annotations/neto_annotator_b/`.
Annotator A's NETO cells are not finished and are not included.

**Movie:** `8_hr_timelapse_wt_020123.sldy` (lab storage),
Position 2: wild type, 241 frames at 2 min, 640 × 952 px at 0.11 µm/px, a different
microscope and camera from the Nikon development movies (SlideBook; much noisier and
dimmer). Not a development or test movie.

**Run:** frames resampled to the pipeline's 0.0647 µm/px (cubic, centre-aligned;
`pombe_tracker/movie_io.py`) so the frozen detector's pixel settings apply unchanged;
Annotator B's clicks mapped the same way (`validation/annotations/neto_annotator_b_resampled/`).
Cellpose-SAM masks on cluster V100s, then `bsd track` and
`python -m validation.compare_compartments --frame-minutes 2`.
All px below are resampled px (0.0647 µm). The detector was not changed.

## Results

| | |
|---|---|
| Annotated cell-frames matched to a detector cell | 179 / 184 |
| Same new/old pole labels | 177 / 179 (99%) |
| Scar midpoint within 5 px (0.32 µm) | 121 / 179 (68%), median 3.9 px (0.25 µm) |
| New-end length error | mean absolute 0.37 µm, bias +0.23 µm (detector longer) |
| ...without the 2 wrong-end frames | mean absolute 0.30 µm, bias +0.16 µm |
| NETO time, detector minus Annotator B | mean absolute 19 min (6–30 min) |

For comparison, on single frames of the development movies the detector was within
5 px of Annotator B's scar mark on 79% of annotations (median 2.8 px).

| Cell | Cycle (min) | Same pole labels | Scar ≤ 5 px | New-end bias (µm) | NETO since birth, Annotator B / detector (min) |
|---|---:|---:|---:|---:|---:|
| c1 | 168 | 29/29 | 20/29 | −0.12 | 78 / 48 |
| c2 | 162 | 27/27 | 23/27 | 0.00 | 6 / 30 |
| c3 | 204 | 35/35 | 27/35 | +0.26 | 66 / 84 |
| c4 | 174 | 27/29 | 16/29 | +0.36* | 30 / 12 |
| c5 | 180 | 29/29 | 19/29 | −0.09 | 12 / 30 |
| c6 | 180 | 30/30 | 16/30 | +0.49 | 6 / 12 |

\* without frames 151 and 154, where the detector put the scar at the wrong end
(~100 px from Annotator B's mark) and swapped the pole labels.

## Reading it

- **Pole identity is solid** (99%), and the new-end length is within ~0.3 µm on
  average, on a microscope the detector has never seen.
- **Scar position is a little worse than on the development movies** (68% vs 79%
  within 0.32 µm). It is better in the first 30 min after birth (87% within 5 px)
  than later (64%). c6's detector trace jumps between frames (bias +0.49 µm).
- **NETO is not yet a fair test.** The hinge fit (flat, then linear) is unstable on
  these traces for both sources: dropping the first 10 min after birth moves
  Annotator B's own estimate by up to 48 min (c6: 6 → 54 min) and leaves the mean
  detector–Annotator B difference at 19 min. With 6-min sampling and this noise, NETO
  needs a better definition (for example the scorer calling the take-off frame,
  or a fit with an uncertainty) before the detector can be judged on it.
- **Annotator B's new end lengthens 0.3–1.1 µm in the first few frames after birth**,
  then levels off: the new pole rounding up after separation. The detector misses
  most birth frames (4 of the 5 unmatched frames are birth frames, where the
  daughters are not yet separate cells in the pipeline), so its traces start after
  this jump. This likely explains the early new-end rise in hand-tracked growth
  curves that the pipeline does not show (`../growth_curves/README.md`).
- n = 6 cells from one movie and one scorer; Annotator B's clicks were made at
  0.11 µm/px, so click precision is coarser than on the Nikon movies.

## Files

`SUMMARY.md`, `per_cell.csv`, `per_frame.csv` (script output, px and movie time),
`new_end_length.png` (new-end length per cell, Annotator B red, detector blue; dotted
lines are each source's hinge NETO).
