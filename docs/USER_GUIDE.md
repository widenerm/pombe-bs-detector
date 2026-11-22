# BS-Detector user guide

How to go from a brightfield time-lapse movie to birth scars, pole identity, lineages
and growth curves. For septin intensity vs curvature, see [SEPTIN.md](SEPTIN.md).

## 1. Choose where to run it

| | Good for | Segmentation speed (2304 × 2304 frame) |
|---|---|---|
| **Google Colab** ([notebook](../notebooks/BS_Detector_Colab.ipynb)) | most people; no install, free GPU | ~15–40 s per frame |
| **Your own computer** (`pip install`) | Macs with Apple silicon, Linux/Windows with an NVIDIA GPU | ~90 s per frame on an Apple M2; much slower on a CPU only |
| **A SLURM cluster** ([HPC.md](HPC.md)) | many movies, or long movies in a hurry | split across several GPUs |

Segmentation is the only slow step, and it is resumable. Tracking a 9-hour movie takes
5–60 minutes on a laptop CPU, depending on how many cells it holds.

**Work on copies.** Copy the movie from lab storage to your computer or Google Drive.
BS-Detector only ever reads the movie, but a copy keeps the shared drive out of it.

## 2. Install (own computer)

Python 3.10 or newer:

```bash
python -m venv bsd-env && source bsd-env/bin/activate      # Windows: bsd-env\Scripts\activate
pip install "bs-detector @ git+https://github.com/widenerm/pombe-bs-detector"
bsd --help
```

The first segmentation downloads the Cellpose-SAM weights (~1 GB) once. BS-Detector
checks they are the exact weights it was validated with and stops if they are not.

## 3. Look at the movie first

```bash
bsd info my_movie.nd2
```

This prints the positions, channels, frame count, pixel size and frame interval.
BS-Detector needs a **single brightfield channel** and a **single focal plane**:

- **Several positions** (multipoint ND2, SlideBook): `bsd run` makes one run folder per
  position, or pick one with `--position 2`.
- **Several channels:** a channel named BF, Brightfield, DIA, TL, Trans, DIC or Phase is
  picked automatically; otherwise pass `--channel NAME` (or its number from 0).
- **A Z stack:** pass `--z N` for the in-focus plane.
- **Pixel size:** read from the file. Movies not at 0.0647 µm/px (Nikon Ti2, 100×/1.49,
  1×1 binning) are resampled to it, because the detector's settings are in those pixels.
  A SlideBook movie at 0.11 µm/px was analysed this way for validation. If a TIFF has no
  pixel size, pass `--pixel-size`. Older lab `.h5` files with no pixel size are assumed
  to be 0.0647 µm/px.
- **Frame interval:** read from ND2 and ImageJ TIFF files; otherwise pass
  `--frame-minutes` (used for growth-curve time axes; defaults to 2 with a warning).

## 4. Run it

```bash
bsd run my_movie.nd2 --out runs/my_movie
```

Or step by step. Each step skips what is already done, so if anything is interrupted,
run the same command again:

```bash
bsd prepare my_movie.nd2 --out runs/my_movie   # read (and resample) the movie
bsd segment runs/my_movie                      # Cellpose masks; the slow, GPU step
bsd track   runs/my_movie                      # tracking, poles, scars
bsd growth  runs/my_movie                      # growth curves
```

`bsd segment` and `bsd track` also accept a folder of per-position run folders.

## 5. What you get

```
runs/my_movie/
  run.json            the input file, pixel size, frame interval, software versions,
                      Cellpose weights and device, and when each step ran
  frames.h5           the frames as the pipeline saw them (only if converted or resampled)
  masks/0000.npz ...  one Cellpose mask per frame, tied to its frame by a checksum
  measurements.csv    one row per cell per frame
  qc/                 five frame overviews and the lineage tree: look at these first
  growth/             growth figures and tables (if the movie has complete cell cycles)
```

### measurements.csv

Lengths are in pixels and, in the `_um` columns, in µm; area is in px² and µm².

| Column | Meaning |
|---|---|
| `frame`, `cell_name` | frame (from 0) and lineage name: `A` divides into `A0` (new-end daughter) and `A1` (old-end daughter) |
| `length`, `width_centroid`, `area` | pole-to-pole length, width at the centre, area |
| `new_end_length`, `old_end_length` | birth-scar midpoint to the new pole and to the old pole |
| `width_scar` | distance between the two scar endpoints (0 on `interpolated` frames, which have no measured scar) |
| `scar_detected`, `scar_source` | whether a scar was found; `raw`, `corrected` or `interpolated` after stabilisation |
| `scar_confidence` | how concentrated the detector's score is on the chosen position (0–1) |
| `scar_needs_review` | **check this cell by hand**: low confidence, or the scar is nearer the end called old |
| `scar_pole_conflict` | the scar sits nearer the end called old |
| `scar_stable` | false if more than 30% of the cell's frames needed correcting |
| `seg_quality` | `ok` or `septum_fragment` (cells clipped by the image edge are left out) |
| `pole_method`, `pole_confidence` | how the new pole was decided: `lineage`, `scar_based`, `neighbor_proximity`, `neighbor_poles`, `morphology` |
| `scar_midpoint`, `new_pole_point`, `old_pole_point` | `[row, col]` in pixels of the pipeline's frames |

**Whole-cell length is the most reliable number.** The two end lengths depend on where
the scar is placed. Filter or check rows with `scar_needs_review = True`. In the movies
analysed so far, a cell has anywhere from 1% to over 70% of its frames flagged, and the
flag rate tracks how jumpy its end-length trace looks.

### growth/

A **complete cycle** is a cell born in the movie (from a real division: each daughter
under 75% of the mother's length) that also divides in it, lasts at least 60 min, and
never drops to under 75% of its length mid-cycle (which would mean a missed division).

| File | Contents |
|---|---|
| `growth_since_birth.png/.pdf` | length minus length at birth, for the whole cell, old end and new end: mean ± SE across cycles, plus single cells |
| `growth_curves.png/.pdf` | length through each cycle |
| `growth_deltas.png/.pdf` | change per frame (noisy; read the running median) |
| `growth_curves_summary.csv` | per cycle: `birth_length_um`, `division_length_um`, `cycle_min`, `elongation_um_per_h` (first 75% of the cycle), mean and SD of change per frame, frames dropped |
| `growth_curves_data.csv` | every plotted point, with `kept` = false for frames dropped as segmentation glitches (more than 0.5 µm from the 7-frame running median) |

`bsd growth` prints which cycles it rejected and why. A movie may have no complete
cycles (too short, or cells leave the field). Then `measurements.csv` is still useful,
but there are no growth figures.

## 6. Compare with hand tracking

The lab's hand-tracked growth workbook (one sheet per cell, named like
`(ye) wt cell 2` or `(s) rgf1DEL gpd1DEL cell 5`, with old end, new end and whole cell
blocks) can be plotted next to the pipeline:

```bash
bsd growth runs/my_movie --hand growth_workbook.xlsx --hand-type y_wt
```

`--hand-type` is `y_` (YE) or `s_` (sorbitol) plus the strain, e.g. `y_wt`, `s_wt`,
`y_rgf1delgpd1del`. If you leave it out, the error lists the types in the workbook.
This adds hand-tracked means to `growth_since_birth` and writes
`growth_since_birth_vs_hand` (one panel per end). The sheets don't record which movie
or position each cell came from, so this compares populations of cells, not the same
cells.

## 7. Other strains, mutants and microscopes

- The scar detector's six weights were fitted on wild-type cells in YE. On other strains,
  look at `qc/` and the `scar_needs_review` rate before trusting end lengths.
- For a strain where it does poorly, a few dozen hand-annotated cells are enough to refit
  the weights (`python -m validation.scar_fit`; see
  [validation/reports/scar_v3/REPORT.md](../validation/reports/scar_v3/REPORT.md)).
  A refit model is a new model: say so, and don't quote this repository's accuracy for it.
- Size filters and tracking distances are in pixels at 0.0647 µm/px; resampling takes care
  of other pixel sizes.

## 8. Troubleshooting

| Message | What to do |
|---|---|
| `this file has N positions/channels` | pass `--position`, `--channel` or `--z` as the message says |
| `does not record its pixel size` | pass `--pixel-size` (µm per pixel) |
| `weights ... are not the validated cpsam weights` | `pip install cellpose==4.0.9`; delete `~/.cellpose/models/cpsam` so it downloads again |
| `no GPU: expect several minutes per frame` | use Colab or a cluster, or leave it running overnight |
| `N frames have no saved mask` | `bsd segment` didn't finish; run it again |
| `mask ... was made from a different image` | the movie changed after segmentation; use a new `--out` folder |
| `no complete cell cycles` | normal for short movies; see the `rejected` lines for why |

Set `BSD_DEBUG=1` to see the full Python traceback of an error.
