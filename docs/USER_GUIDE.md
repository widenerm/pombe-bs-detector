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
bsd prepare my_movie.nd2 --out runs/my_movie   # record the movie (resample only if needed)
bsd segment runs/my_movie                      # Cellpose masks; the slow, GPU step
bsd track   runs/my_movie                      # tracking, poles, scars; list of cells to check
bsd growth  runs/my_movie                      # apply your corrections; growth curves
```

`bsd segment` and `bsd track` also accept a folder of per-position run folders.

**Figures** (frame overviews, growth curves) go to `RUN/figures/`. Use `--figures-dir DIR` to
put them somewhere else, or `--no-figures` for tables only. The Colab notebook keeps them on
the Colab machine and shows them, unless you tick *SAVE_FIGURES*.

## 5. What you get

```
runs/my_movie/
  run.json                    the input file, pixel size, frame interval, software versions,
                              Cellpose weights and device, and when each step ran
  measurements.csv            the detector's output: one row per cell per frame (never edited)
  review.csv                  one row per cell: which frames to check, and why (section 6)
  corrections/edits.csv       your fixes (section 6); Fiji CSVs go in this folder too
  measurements_corrected.csv  measurements with your fixes applied (only once there are fixes)
  growth/                     growth tables (if the movie has complete cell cycles)
  figures/                    frame overviews, lineage tree, growth curves (unless --no-figures)
  working/masks/              one Cellpose mask per frame, tied to its frame by a checksum
  working/frames.h5           only if the movie had to be resampled
```

The movie itself is read in place: a copy is made only when it has to be resampled. Keep
`working/` while you are still checking cells; `bsd show` and the fixes need the masks.

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
| `scar_end_1`, `scar_end_2`, `label` | the two scar endpoints, and the cell's number in that frame's mask (for drawing it) |

**Whole-cell length is the most reliable number.** The two end lengths depend on where
the scar is placed. Section 6 covers checking and fixing them.

### growth/

A **complete cycle** is a cell born in the movie (from a real division: each daughter
under 75% of the mother's length) that also divides in it, lasts at least 60 min, and
never drops to under 75% of its length mid-cycle (which would mean a missed division).

| File | Contents |
|---|---|
| `figures/growth_since_birth.png/.pdf` | length minus length at birth, for the whole cell, old end and new end: mean ± SE across cycles, plus single cells |
| `figures/growth_curves.png/.pdf` | length through each cycle |
| `figures/growth_deltas.png/.pdf` | change per frame (noisy; read the running median) |
| `growth_curves_summary.csv` | per cycle: `birth_length_um`, `division_length_um`, `cycle_min`, `elongation_um_per_h` (first 75% of the cycle), mean and SD of change per frame, frames dropped |
| `growth_curves_data.csv` | every plotted point, with `kept` = false for frames dropped as segmentation glitches (more than 0.5 µm from the 7-frame running median) |

`bsd growth` prints which cycles it rejected and why. A movie may have no complete
cycles (too short, or cells leave the field). Then `measurements.csv` is still useful,
but there are no growth curves. Growth curves always use your corrections, if there are any.

## 6. Check and correct

**Which cells to look at.** `bsd track` writes `review.csv` (`bsd review RUN` rewrites it and
prints the top). It has one row per cell; cells used in the growth curves come first, then
those with the most frames to check. A frame is flagged when:

| Reason | Meaning |
|---|---|
| `unsure` | the detector's own flag: low confidence, or the scar is nearer the end it called old |
| `jump` | whole-cell, new-end or old-end length more than 0.5 µm off the cell's 7-frame running median: the "crazy numbers", usually the scar mark jumping or a bad outline |
| `off-axis` | new end + old end is more than 1.1 × the whole-cell length (scar far from the midline) |
| `interpolated` | no scar found in this frame; the position was filled in from the frames around it |
| `no scar`, `outline` | no scar at all; the segmentation was flagged (e.g. a septum fragment) |

`why` counts the flagged frames by reason, `check_frames` lists them, and `x`, `y`, `at_frame`
and `fiji_slice` say where to find the cell. Coordinates are in the movie's pixels; the Fiji
slice is frame + 1. The detector's `unsure` flag fires often on noisy movies, so start with
cells in the growth curves and with `jump` frames.

**Look at a cell.**

```bash
bsd show runs/my_movie --cell A01                 # the frames to check (or evenly spaced)
bsd show runs/my_movie --cell A01 --frames 40-60
```

This draws the cell's whole-cell, old-end and new-end lengths through time, with flagged
frames shaded. Below that, it shows the cell at each chosen frame: outline (blue), birth scar
(yellow line, white dot), new pole (green), old pole (magenta), with axes in movie pixels.
In Colab this is step 6, shown in the notebook.

**Fix it.** The detector's output (`measurements.csv`) is never changed. Fixes go into
`corrections/`, and the next `bsd growth` applies them. It writes
`measurements_corrected.csv`, with a `corrected` column saying what changed and where the
fix came from, and redraws the growth curves. There are three ways to record a fix:

1. **Command line:**

   ```bash
   bsd fix runs/my_movie --cell A01 --frames 52 --scar 812 1440        # scar is here (movie x, y)
   bsd fix runs/my_movie --cell A01 --frames 40-45 --exclude --note "outline cut the cell in half"
   bsd fix runs/my_movie --cell A01 --swap-poles                       # ends called the wrong way round
   bsd fix runs/my_movie --cell A01 --frames 60 --lengths 2.1 7.9      # your own new/old end lengths (µm)
   bsd fix runs/my_movie --cell A01 --checked                          # looked; the detector is right
   ```

2. **Colab:** step 6 of the notebook. Choose the cell, the frames and an action. For
   *click the scar*, the cell is shown at each frame and you click the scar midpoint.

3. **Fiji:** open the movie exactly as analysed (all frames, the same position, not cropped
   or resampled) and go to the frame to fix. Run [`fiji/BS_check_cell.ijm`](../fiji/BS_check_cell.ijm),
   type the BS-Detector cell name, and click the scar on each frame. Click the poles too
   only if they are wrong. Save the CSV into the run's `corrections/` folder.

Or edit `corrections/edits.csv` directly. Use one row per fix, with `cell_name`, `frames`
(e.g. `52`, `40-60`, or empty for all frames), and any of:
- `scar_x`, `scar_y`: the scar midpoint in movie pixels;
- `new_end_length_um`, `old_end_length_um`;
- `swap_poles`, `exclude`, `checked`: `yes` to apply;
- `note`, `who`.

**What each fix does:**
- **A moved scar:** both end lengths are recomputed from it and the detector's poles, which
  is how the detector measures them. If you also clicked the poles in Fiji, those are used,
  and the whole-cell length becomes pole to pole.
- **Typed lengths:** replace the detector's.
- **Exclude:** takes the frames out of the growth curves.
- **Swap:** exchanges the new and old ends.
- **Checked:** changes nothing but removes the frames from `review.csv`.
- **Two fixes for the same frame:** the later one wins.

## 7. Compare with hand tracking

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

## 8. Other strains, mutants and microscopes

- The scar detector's six weights were fitted on wild-type cells in YE. On other strains,
  look at the frame overviews and `review.csv` before trusting end lengths.
- For a strain where it does poorly, a few dozen hand-annotated cells are enough to refit
  the weights (`python -m validation.scar_fit`; see
  [validation/reports/scar_v3/REPORT.md](../validation/reports/scar_v3/REPORT.md)).
  A refit model is a new model: say so, and don't quote this repository's accuracy for it.
- Size filters and tracking distances are in pixels at 0.0647 µm/px; resampling takes care
  of other pixel sizes.

## 9. Troubleshooting

| Message | What to do |
|---|---|
| `this file has N positions/channels` | pass `--position`, `--channel` or `--z` as the message says |
| `does not record its pixel size` | pass `--pixel-size` (µm per pixel) |
| `weights ... are not the validated cpsam weights` | `pip install cellpose==4.0.9`; delete `~/.cellpose/models/cpsam` so it downloads again |
| `no GPU: expect several minutes per frame` | use Colab or a cluster, or leave it running overnight |
| `N frames have no saved mask` | `bsd segment` didn't finish; run it again |
| `correction not applied: ...` | the cell name or frame in a fix doesn't exist; check `review.csv` for names |
| `mask ... was made from a different image` | the movie changed after segmentation; use a new `--out` folder |
| `no complete cell cycles` | normal for short movies; see the `rejected` lines for why |

Set `BSD_DEBUG=1` to see the full Python traceback of an error.
