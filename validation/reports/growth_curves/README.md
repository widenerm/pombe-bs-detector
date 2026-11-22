# Growth curves

Complete cell cycles from one wild-type movie, `experiment_03272025_wt_ymd494__xy1`
(full 2304 × 2304 frame, 241 frames at 2 min, 0.0647 µm/px), run through the
current pipeline from saved Cellpose masks. This movie is not one of the held-out
test movies (3/14, 3/19).

**Figure:** `03272025_xy1/growth_curves.pdf` (vector) and `.png` (300 dpi).

## Draft caption

**Growth of wild-type *S. pombe* through one cell cycle.** (A) Cell length from
birth to division for all six complete cycles in one time-lapse movie, i.e. cells
born and dividing within the 8 h movie. (B) Three of those cells: whole-cell
length (blue) and the length from the birth scar to the old end (magenta) and to
the new end (green). Faint lines are per-frame values; bold compartment lines are a
14-min running median. The three were chosen because the scar position is the most
stable across frames. The other three cycles are in `03272025_xy1_all_cells/`.

## Growth since birth (Annotator A's format)

**Figure:** `03272025_xy1/growth_since_birth.pdf` / `.png`. Each value is length at
time *t* minus length at birth, the quantity in Annotator A's hand-tracked growth plots
("Average Δ length", y 0–9 µm, x minutes after cell separation). (A) puts the whole
cell, old end and new end on one set of axes so changes in slope line up: pipeline
mean ± SE across the six cycles (solid, shaded) and Annotator A's hand-tracked wild type
in YE (dashed; type `y_wt` in the lab's hand-tracked growth table, 5 cells, a different
movie). (B) the same three curves for single cells.
`growth_since_birth_vs_hand.pdf` / `.png` gives one panel per end, with SE on both.
Columns `grown_um`, `grown_old_um`, `grown_new_um` in `growth_curves_data.csv`.

| Mean Δ since birth (µm) | 60 min | 120 min | 150 min |
|---|---:|---:|---:|
| Whole cell, pipeline / hand | 2.18 / 2.16 | 4.83 / 4.64 | 5.85 / 5.87 |
| Old end, pipeline / hand | 1.31 / 1.90 | 3.01 / 3.91 | 3.40 / 4.68 |
| New end, pipeline / hand | 0.59 / 0.74 | 1.51 / 1.51 | 2.09 / 1.80 |

- **Whole cell and new end match hand tracking.** Within one SE at every time
  above, except the new end before ~50 min, where the hand-tracked new end rises
  about 0.6 µm in the first 30 min and then stays flat; the pipeline's rises slowly
  from the start.
- **The pipeline's old end grows ~1.3 µm less by 150 min.** About half of that is
  how the hand measurements add up: Annotator A's old + new end growth exceeds their
  whole-cell growth by 0.06–1.96 µm at division (mean 0.9 µm), while the
  pipeline's ends sum to the whole cell (within 0.1 µm at division). The rest is
  unexplained: different cells and movies, and the compartments are not validated.
- **Not sorbitol.** 3/27/2025 is a YE control movie; the lab's 1.3 M / 0.4 M sorbitol
  movies are separate (June and September 2025). Hand-tracked wild type in sorbitol
  (`s_wt`) still grows 4.4–6.5 µm per cycle (1.9–2.3 µm/h, vs 2.1–2.8 in YE).

The hand-tracked table is not public; `bsd growth RUN --hand TABLE --hand-type y_wt`
regenerates these figures from it.

## Change per frame

**Figure:** `03272025_xy1/growth_deltas.pdf` / `.png` (six-cell version in
`03272025_xy1_all_cells/`). Same layout as the length figure, but each value is
length at frame *n* minus length at frame *n*−1 (2 min apart), computed only when
both frames were kept. Columns `d_length_um`, `d_old_end_um`, `d_new_end_um` in
`growth_curves_data.csv`.

**Draft caption.** **Growth rate of wild-type *S. pombe* through one cell cycle.**
(A) Change in cell length between consecutive 2-min frames for all six complete
cycles. Dots are single frames; lines are a 30-min (15-frame) running median.
(B) The same for the whole cell (blue) and the length from the birth scar to the
old end (magenta) and to the new end (green) in three cells. Single-frame changes
outside the axis are counted in each panel.

- **Read the running median, not the dots.** A whole cell grows about 0.07 µm per
  frame (= 2.6 µm/h), and single-frame changes scatter with an SD of 0.05–0.11 µm,
  so one frame's change is mostly measurement noise. The median shows growth
  speeding up from about 0.07 to 0.10–0.12 µm per frame mid-cycle, then dropping to
  near zero late in the cycle, consistent with growth stopping around mitosis.
- **Compartment changes are mostly scar-mark jitter.** Their single-frame SD is
  0.3–0.5 µm in five of six cells (E0: 0.08 µm), several times the growth signal.
  The off-axis counts in panel B (16 for C100, 42 for C11) are scar jumps. The
  compartment medians are not yet a measure of how fast each end grows.
- Per-cell mean ± SD of whole-cell change per frame is in
  `growth_curves_summary.csv` (`mean_change_um_per_frame`, `sd_change_um_per_frame`).

## Numbers (six cycles)

| | median | range |
|---|---:|---:|
| Length at birth (µm) | 8.0 | 7.8–8.8 |
| Length at division (µm) | 14.0 | 13.1–14.7 |
| Cycle (min) | 174 | 142–208 |
| Elongation, first 75% of cycle (µm/h) | 2.58 | 2.41–2.92 |
| Mean change per 2-min frame (µm) | 0.07 | 0.07–0.08 |

Per cell: `03272025_xy1/growth_curves_summary.csv`. Every plotted point:
`growth_curves_data.csv`.

## What to say if asked

- **Length is solid; compartments are not yet validated.** Whole-cell length
  comes from the Cellpose mask and pole detection. The old/new-end split depends on
  where the scar detector puts the scar. The detector agrees with hand scorers
  on single frames (`../scar_v3/`), and on Annotator B's through-time tracks
  (`../neto_annotator_b/`) the new-end length is within about 0.3 µm on average,
  but that is 6 cells from one movie. The steps in C11 at about 100 and 150 min are the
  scar position moving between frames, not growth.
- **The detector knows which frames are uncertain.** It flags low-confidence
  scars for review. The flag rate tracks how clean each trace looks: E0 is flagged
  on 29% of frames, C100 on 49%, C11 on 72%.
- **No NETO time is reported.** The flat-then-rising new end (C11, C100) looks
  like NETO, but a hinge fit on these traces gave implausible times, so none is
  claimed.
- **Cleaning.** Frames where length is more than 0.5 µm from the cell's 7-frame
  running median were dropped as segmentation errors: 8 of 525 frames, at most 4
  per cell (`kept` column). Birth length is the median of the first 3 frames;
  division length is the median of the last 5.
- **Cycle selection.** A cycle counts if the cell's birth and its own division are
  both real divisions, meaning each daughter is under 75% of the mother's final
  length. This rejected two tracker false divisions (C → C1 at frame 38, V1 → V11 at
  frame 193), where a one-frame mask fragment split off and the "daughter"
  carried on at full length.

## Reproduce

```bash
# masks: Cellpose 4.0.9 cpsam (pinned sha256), float32, cluster GPU (provenance in segmentation_provenance.json)
bsd segment experiment_03272025_wt_ymd494__xy1.h5 --out runs/0327_xy1
bsd track runs/0327_xy1
bsd growth runs/0327_xy1 --featured C100 C11 E0 --hand HAND_TRACKED_TABLE.csv --hand-type y_wt
```

`03202025_xy1_window/` is an earlier draft from a 768 × 768 window of another
movie (five cycles). It is superseded by the full-frame figure above.
