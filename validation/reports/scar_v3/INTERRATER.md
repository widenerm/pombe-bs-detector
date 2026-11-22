# Second scorer: Annotator B vs Annotator A

Annotator B re-annotated the same 80 cells as Annotator A (`annotations/dev1_annotator_b.csv`
and `dev2_annotator_b.csv`, the same rows as `dev1_annotator_a.csv` and `dev2_annotator_a.csv`).
The two annotators work independently: neither sees the other's points until both are finished. Annotator B left 3
scars blank, so scar comparisons use 77 annotations:

- C0 f144 and AG10 f227: "obscured by neighboring cells". Annotator A had marked
  both as faint and used an older scar on AG10.
- C101 f149: "scar unclear".

The 77 annotations come from 61 physical cells; some cells were annotated in
2–3 frames. Both annotators' CSVs are in `validation/annotations/`. Error is the distance between scar midpoints, in
pixels.

## What is and isn't held out

Nothing was refit for the headline numbers. The detector is the shipped
profile mode, and both its parts were developed on these same 80 cells:

- **Weights:** its 6 weights were fitted on all 80 of Annotator A's labels.
- **Design:** the membrane tracer, candidate cross-sections, features, sibling
  hint and confidence flag were chosen while scoring against Annotator A's labels
  on these cells.

Annotator B's labels are therefore new, but the cells are not. Scoring against
Annotator B removes any fit to Annotator A's per-cell clicking noise. It does not remove
the fit to these particular images. Every number below is a development-set
result and likely optimistic for new experiments.

The "held out by video" refits hold out only the 6 weights, not the design.
Holding out by packet would leak, because 8 of the physical cells appear in
both packets, so this report doesn't use it. The leave-one-packet-out row in
REPORT.md has the same problem.

## Result

| Comparison (77 annotations, 61 cells) | Within 5 px | Within 3 px | Median |
|---|---:|---:|---:|
| **Annotator A vs Annotator B** | **61 (79%)** | 30 | 3.4 px |
| Detector vs Annotator A (labels it was fitted on) | 56 (73%) | 29 | 3.5 px |
| **Detector vs Annotator B** | **61 (79%)** | 43 | 2.8 px |
| Detector vs Annotator B, weights fitted on Annotator A's other videos | 59 (77%) | 42 | 2.8 px |
| Detector vs mean of both scorers | 61 (79%) | 45 | 2.6 px |
| Bilateral mode vs Annotator B | 51 (66%) | 29 | 3.5 px |
| Legacy mode vs Annotator B | 18 (23%) | 10 | 8.0 px |

- **On these cells the detector agrees with Annotator B as often as Annotator A does.**
  The 95% interval on the difference is −13 to +13 percentage points
  (bootstrap over cells, keeping each cell's frames together). So the data
  shows only that the two are similar, not identical: a gap of about 10 cells
  either way is still consistent with it.
- **The two comparisons aren't fully symmetric:**
  - Annotator A–Annotator B compares two independent people.
  - The detector was built on these cells.
  - Annotator B skipped the 3 hardest scars, but the detector always answers. Of
    those 3, it is 65 px from Annotator A on AG10 f227 and 11 px on C101 f149.
- **The profile mode remains the best mode.** Bilateral mode's settings were
  also grid-searched on these cells.

## How much of the 5 px error is the label?

With three measurers of the same point (Annotator A, Annotator B, and the detector held
out by video), each one's own scatter can be separated out (the
"three-cornered hat"). This uses the 66 annotations where all three are within
10 px of each other, so it excludes choosing a different scar. The intervals
are bootstrapped over cells.

| SD, px (95% CI) | Annotator A | Annotator B | Detector |
|---|---:|---:|---:|
| Along the cell axis | 2.9 (2.2–3.6) | 2.2 (1.4–2.8) | 2.1 (1.2–2.8) |
| Across the cell | 1.4 (1.2–1.6) | 0.7 (0.0–1.0) | 1.1 (0.8–1.4) |

- **A perfect detector would not reach 90% against Annotator A.** Placed exactly on
  the true scar, it would be within 5 px of Annotator A's point in about 90% of
  cells (82–97% over the interval). Against Annotator B it would be about 97%
  (92–100%). That is before counting cells where a scorer picks a different
  scar. So the earlier 90% target against Annotator A's labels was at or near
  what's achievable.
- **The method assumes the measurers' errors are independent, and they
  aren't.** The detector learned Annotator A's conventions, so the shared error
  makes Annotator A and the detector look less noisy than they are, and Annotator B
  more so. The finding that Annotator A scatters more than Annotator B survives this
  bias. The detector's SD may be optimistic. The selection to cells where all
  three agree within 10 px also shrinks every SD.
- **There is no clear systematic offset along the axis** between any two
  measurers (median within 0.9 px).
- **Distance from the cell midline:** Annotator B's points are 1.2 px off it
  (median) and Annotator A's 1.4 px. The detector's are on the midline by
  construction (0.2 px).

## Cells where the scorers disagree by more than 10 px

| Cell | What happened |
|---|---|
| H0 f100 | **Annotator B marks the opposite end, where the detector put it** (4.6 px from Annotator B). This was one of the detector's five "wrong-end" calls against Annotator A, and REPORT.md had already flagged Annotator A's point for a re-check. |
| F10 f170 | Annotator B: "closest birth scar obscured, used middle birth scar". Annotator A and the detector agree on the nearer scar (6.7 px apart). |
| E0 f174 | Both scorers mark the same end, 25 px apart. The detector is at the other end, which is a real detector error. |

The five wrong-end calls in REPORT.md now break down as follows:

- **Confirmed by both scorers (3):** G f77, C10 f91 and E0 f174.
- **Overturned by Annotator B (1):** H0 f100.
- **Unresolved (1):** AG10 f227. Annotator A used an older scar and Annotator B
  couldn't see one.

## Where the detector misses both scorers

The scorers are within 5 px of each other on 61 annotations. On 6 of those,
the detector is more than 5 px from both:

- C11 f225 and B11 f182: 9–17 px off.
- G0 f159: 9 px off.
- C10 f91 and G f77: wrong end.
- AJ f185: 25 px off.

C11 f109 behaves the same way, but the scorers are 5.1 px apart there. Annotator A
noted "septum maturing" on G f77 and AJ f185, and the image of G f77 shows a
septum across the cell. These two cells are dividing, so they may be better
excluded than scored. The `interrater_disputed.png` gallery shows every cell
in this section and the one above.

## Refitting the weights on Annotator B's labels

| Train labels → test labels | In-sample | Held-out video |
|---|---:|---:|
| Annotator A → Annotator A (shipped config) | 57/80 | 54/80 |
| Annotator B → Annotator B | 62/77 | 60/77 |
| Mean of both → mean of both | 62/74 | 61/74 |
| Annotator A → Annotator B | 61/77 | 59/77 |
| Annotator B → Annotator A | 54/80 | 53/80 |
| Annotator A → mean | 61/74 | 59/74 |

The mean-of-both rows use the 74 annotations where the scorers are within
10 px of each other.

- **Which scorer the model is trained on barely matters (≤ 2 cells).** The
  test set's scorer is what changes the number.
- **So the config stays as shipped.** Fitting on the mean of both scorers
  gains 2 cells held out by video, which is within noise.
- **The image terms are stable across label sets:**
  - sibling hint: 2.0–2.5;
  - membrane width: 44–60;
  - membrane ridge: 18–25.
- **The distance prior is not pinned down by 80 cells.** Fitted on Annotator B it
  becomes offset −3.5, slope 1.2, SD 0.33 W. Fitted on Annotator A it is −0.74,
  0.37, 0.19 W.

## Confidence

- **`scar_confidence` predicts misses against Annotator B** (AUC 0.74) better than
  against Annotator A (0.67).
- **It also predicts where the two scorers disagree** (AUC 0.64). The cells
  the detector is unsure about are the ones people are unsure about.
- **With the 0.5 flag:** 18 cells are flagged, and the remaining 59 are within
  5 px of Annotator B in 52 (88%) and of Annotator A in 48 (81%).
- **The 0.5 threshold was chosen on these cells.**

## Other measurements

- **New pole:**
  - Letters: the A/B labels differ in 4 cells, all horizontal cells that were
    labelled left/right instead of "A = upper pole" (Annotator A on F11 f146 and
    f151, Annotator B on F11 f133).
  - Physical ends: mapping A/B onto the cell's actual ends, the scorers choose
    the same new end in 76/77. The only real disagreement is H0 f100.
  - Detector: its new pole from sibling contact matches both scorers in 50/51
    cells (high/medium confidence). Both scorers disagree with it on P f134.
- **Area:**
  - Annotator B vs Annotator A: +4.8% (median |diff| 4.5%, r = 0.98).
  - Cellpose: +12% vs Annotator A and +7.5% vs Annotator B, consistent with the mask
    including the bright halo.
- **New-end compartment length:**
  - Annotator B vs Annotator A: Annotator B reads +2.6 px longer (median |diff| 3.3 px).
  - Detector-based (scar → Cellpose pole): median |diff| 2.7 px vs Annotator B's
    measurement and 4.1 px vs Annotator A's.
  - Part of this difference is the Cellpose pole sitting outside the scorer's
    pole: the total cell length is 3.7–7.3 px longer.
- **Repeatability:** this compares the same cell marked 1–5 frames apart, in
  5 pairs of frames.
  - Annotator A: 0.8–5.7 px.
  - Annotator B: 1.0–5.7 px.
  - Detector: 0.7–2.6 px.
  - The cells moved 2–4 px between frames, so these are upper bounds.

## A fair test

Freeze the current config. Then have both scorers annotate a new packet of
about 60–100 cells from videos and days that weren't used here, and score the
detector once.

## Files and rerun

- `per_cell_interrater.csv`: both scorers' points, the detector's point,
  errors, confidence, areas and notes, one row per annotation.
- `interrater_disputed.png`: the 13 annotations where the scorers disagree by
  more than 10 px, a scar is blank, or the detector is more than 5 px from
  both.

```bash
python -m validation.scar_interrater validation/runs/scar_v3/cells_h200.pkl \
    validation/annotations/dev1_annotator_b.csv \
    validation/annotations/dev2_annotator_b.csv \
    --names 'Annotator A' 'Annotator B' --out validation/reports/scar_v3/per_cell_interrater.csv \
    --gallery validation/reports/scar_v3/interrater_disputed.png
```
