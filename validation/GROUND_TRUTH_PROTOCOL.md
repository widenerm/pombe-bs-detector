# Manual Ground-Truth Measurement Protocol (FIJI)

Purpose: collect independent, by-eye measurements of the same three quantities
BS-Detector reports — **cell area**, **compartment lengths**, and **birth scar
location** — so they can be compared automatically against the tool's output.

To keep this a fair test, cell identities are given to you via **blinded
reference images**: each cell is outlined and labeled with an ID, but the
birth scar and pole assignment the algorithm found are *not* shown. Please
measure independently, by eye, without trying to guess what the tool
detected.

**Important — two different image sets, two different purposes:**

- `raw_frames/*.tif` — the actual microscopy frame, at full original
  resolution, unmodified. **This is what you open in FIJI and measure on.**
- `images/*_blinded.png` — an annotated reference picture, resized/rendered
  for legibility, with each cell outlined and labeled. **Do not measure on
  this one** — it's a different pixel scale than the raw data, so any FIJI
  measurement taken on it won't match BS-Detector's output at all. Use it
  only to see which outlined blob corresponds to which `cell_name`, then go
  find that same cell in the corresponding raw TIFF to actually measure.

Each row in `ground_truth_template.csv` names both files (`raw_frame_file`
and `image_file`) for a given cell.

---

## 0. Before you start: set FIJI to pixel units

BS-Detector has no physical (µm) calibration anywhere in its code — every
length and area it reports is in **raw image pixels**. If your FIJI
measurements come out in µm (because the image has a spatial calibration
baked in), the numbers won't be comparable at all.

For every image you measure:

1. `Image > Properties…` (⌘/Ctrl+Shift+P)
2. Set **Pixel width** and **Pixel height** to `1.0000`
3. Set **Unit of length** to `pixel`
4. Click OK

(Equivalently: `Analyze > Set Scale…` → `Click to Remove Scale`.)

Do this once per image before measuring. All values you record should be in
pixels / pixels², not microns.

---

## 1. Files you'll receive

- `raw_frames/<video_id>_frame####_raw.tif` — the real frame, full
  resolution, unmodified. **Open and measure on this file.**
- `images/<video_id>_frame####_blinded.png` — the same frame, annotated: each
  cell outlined in a neutral color with its `cell_name` ID printed at the
  centroid (e.g. `A`, `A0`, `A01`). Use this only to see which cell is which
  — **do not measure on it** (see the note above). If cells from more than
  one video/field of view are included, the `video_id` prefix (e.g.
  `experiment_03202025_wt_ymd494__xy1`) is how they're told apart — cell
  names repeat across videos (every video has an "A"), so both the ID *and*
  which video it came from matter for matching your rows back to
  BS-Detector's output. Always copy both exactly as printed/named.
- `ground_truth_template.csv` — one row per cell per frame. Delete the
  `EXAMPLE` row before filling in real data, and return the file with the
  same column headers.

---

## 2. What a birth scar looks like

*S. pombe* divides by forming a septum across the middle of the cell. After
division, the site where the septum was leaves a faint **ridge / dent** on
each side of the cell wall, roughly perpendicular to the long axis, at the
same position on opposite edges. In brightfield this shows up as a small,
localized bump/kink in the cell outline — not a full stripe across the cell.
If you genuinely can't identify one on a given cell, leave those fields
blank and note it (see Section 6).

---

## 3. Measurements to record, per cell

Work through each cell labeled in the blinded reference image and fill in
one spreadsheet row per cell per frame.

### (a) Cell area — `gt_area_px2`

Trace the cell outline with the **Freehand** or **Polygon** selection tool,
then `Analyze > Measure` (⌘/Ctrl+M). Record the **Area** value (pixels²).

### (b) Birth scar location — `gt_scar_mid_x_px`, `gt_scar_mid_y_px`

Click the **Point** tool on the visual midpoint between the two scar
ridges (i.e., the point on the cell's centerline directly between the two
opposing bumps). `Analyze > Measure` and record the **X** and **Y** columns
exactly as FIJI reports them (X = column, Y = row, both in pixels).

If you can clearly see both individual ridge points and want to be more
precise, click one, measure, then the other, measure, and record the
midpoint of the two X's and two Y's yourself — either approach is fine as
long as it's consistent per scorer.

### (c) Compartment lengths — `gt_poleA_length_px`, `gt_poleB_length_px`

Identify the cell's two poles (the rounded tips at each end of the long
axis). To avoid any ambiguity about which pole is which, always call:

- **Pole A** = whichever pole appears **higher up in the image** (smaller Y
  / row value)
- **Pole B** = the other pole

Using the **Straight Line** tool (not segmented, not freehand — a single
straight chord), draw a line from the birth scar midpoint (from step b) to
Pole A. `Analyze > Measure`, record the **Length** as `gt_poleA_length_px`.
Repeat for Pole B → `gt_poleB_length_px`.

This must be a straight line directly from the scar midpoint to the pole
tip, *not* a line traced along the curved cell outline — that's the
convention the algorithm uses internally, and tracing the outline instead
will look like a big discrepancy that isn't actually one.

### (d) Which pole is the new end — `gt_new_pole_guess`

Independent of the algorithm, use your own judgment of *S. pombe*
morphology (e.g., the new end is typically less mature/more recently formed
right after division) to guess whether **Pole A** or **Pole B** is the new
end. Enter `A`, `B`, or `unsure` if you're not confident. This lets us
separately check whether any discrepancy in compartment length is due to
the scar/pole positions being off, versus the algorithm mislabeling which
pole is "new" vs "old."

---

## 4. Filling in the spreadsheet

One row per cell per frame. Columns:

| Column | What to enter |
|---|---|
| `video_id` | the video this cell came from — the prefix on the filenames, e.g. `experiment_03202025_wt_ymd494__xy1` |
| `raw_frame_file` | filename of the raw TIFF you actually measured on, e.g. `experiment_03202025_wt_ymd494__xy1_frame0003_raw.tif` |
| `image_file` | filename of the blinded PNG you used to identify the cell (reference only), e.g. `experiment_03202025_wt_ymd494__xy1_frame0003_blinded.png` |
| `frame` | frame index, e.g. `3` |
| `cell_name` | the ID label printed on that cell in the blinded PNG, e.g. `A01` |
| `scorer_initials` | your initials |
| `gt_area_px2` | from step (a) |
| `gt_scar_mid_x_px` | from step (b) |
| `gt_scar_mid_y_px` | from step (b) |
| `gt_poleA_length_px` | from step (c) |
| `gt_poleB_length_px` | from step (c) |
| `gt_new_pole_guess` | `A`, `B`, or `unsure`, from step (d) |
| `notes` | anything ambiguous — poor focus, cell mid-division, scar not visible, etc. |

---

## 5. If a cell doesn't have a clear scar

Still record `gt_area_px2` (area is independent of the scar). Leave the scar
and compartment-length fields blank and write a short note, e.g. `no scar
visible` or `cell just divided, scar not yet distinct`. Don't guess a
location just to fill the cell in — blank + note is more useful than a
low-confidence number recorded as if it were solid.

---

## 6. How many cells / frames

Aim to cover a range of cell-cycle stages (just-divided, mid-cycle, about to
divide) across several frames, from more than one field of view if possible,
rather than exhaustively measuring every cell in every frame. A few dozen
well-distributed cells is far more useful than one frame measured
completely.

---

## 7. Returning results

Send back the completed `ground_truth_template.csv` (same headers, `EXAMPLE`
row removed) along with your initials. That's it — the comparison against
BS-Detector's output is automated from there.
