# Held-out test set for the birth-scar detector

Created 2026-11-03, before anyone ran the scar detector on these cells.

## Rules

- **Detector frozen.** The detector was frozen on 2026-11-03, before anyone ran it on these cells,
  and is published unchanged as release `v1.0.0`. Since the freeze, the only changes to the
  detector's code are one docstring and an optional argument that lets `run_pipeline` take saved
  masks. The default segmenter now uses the same pinned float32 setup these cells' masks were
  made with. `tests/test_sample_regression.py` fails if any change moves a measurement. Score
  `v1.0.0` once, when the annotations return.
- **No tuning on these movies.** Don't tune anything (weights, thresholds, preprocessing) on them,
  not even after scoring.
- **Report all of it.** Report the results whatever they are, including cells with bad outlines and
  cells without a visible scar.
- **Further rounds.** If the detector changes later, the next test needs new movies.

## Data

- **Movies:** wild-type YMD494 in YE, brightfield, Nikon Ti2 100×/1.49, 0.0647 µm/px, 2 min per
  frame. These are imaging days never used in development:
  - 3/14/2025, fields xy1–xy4 (363 frames each);
  - 3/19/2025, fields xy1–xy4 (241 frames each).
  The originals are in the lab's wild-type YE growth-movie collection (not public).
- **Frames:** for each movie, the first frame was drawn uniformly from 20–60, then every 100th
  frame after it (200 min, longer than a wild-type cell cycle), so no cell appears twice. This gave
  26 frames, and seed 20261103 picks both the frames and the cells.
- **Masks:** segmented with Cellpose 4.0.9 and the pinned `cpsam` weights (sha256 `e1440429…`, the
  same weights as the development masks), float32 on the laptop GPU (MPS). See
  `segmentation_provenance.json`. The masks are cached in `validation/runs/scar_test/cache/`
  (gitignored). **Keep them**: the cell IDs in the packet refer to these masks.
- **Cells:** 19 per movie, drawn at random from every object passing the pipeline's
  size/border/aspect/circularity filters, before the segmentation-quality flags, so bad outlines
  reach the scorer and are counted. Three sparse 3/19 fields had fewer eligible objects (7, 17
  and 16), so all of them were taken. That gives 135 cells in 25 frames (one of the 26 frames
  drew no cells), dealt into three batches of 45 that each span all eight movies. The sample
  includes two specks of debris (399 and 640 px) that pass the 150 px `MIN_CELL_AREA` filter.

## Files

- **`test_cells.csv`:** one row per cell, with its batch, video, frame, packet name (`cell_name`),
  mask label and a pixel inside the cell (`anchor_y`, `anchor_x`), for matching annotations back to
  the masks.
- **`segmentation_provenance.json`:** Cellpose version, weights, device and dtype for each frame.
- **The packet itself:** `validation/packets/test/` (raw frames and images are
  gitignored). It was built by `python -m validation.test_packet` (see that file for the
  commands).

## Scoring (when annotations return)

1. **Build records.** Build frozen cell records from these masks and the returned CSVs, as
   `validation/scar_cells.py` does for the development packets, using `test_cells.csv` for the
   anchors.
2. **Score the detector.** Run release `v1.0.0` and report the agreement table from
   `validation/reports/scar_v3/INTERRATER.md` once.
3. **Report counts separately:** cells with status `bad_outline`, `dividing` and `no_scar`.
