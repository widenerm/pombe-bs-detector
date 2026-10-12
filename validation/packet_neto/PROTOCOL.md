# Compartment lengths through time: tracking protocol (FIJI)

**Goal:** follow individual cells from birth to division and record, in each
sampled frame, where the birth scar and the two pole tips are. From these
points we compute the **new-end compartment length** (scar to new pole) over
time, find when the new end starts growing (**new end take-off, NETO**), and
compare both with BS-Detector on the same movie.

This works with any time-lapse movie. Nothing has to be run through
BS-Detector first.

---

## What you need

- FIJI, with the movie open **exactly as it will be analysed**: every frame,
  not cropped, rotated, binned or re-sampled. Frame numbers and pixel
  positions must line up with the original file. If FIJI asks how to open an
  `.nd2`/`.czi` (Bio-Formats), use the defaults and open all time points.
- `BS_track_cell.ijm` (this folder).
- Somewhere to save the CSV files.

Pixel units: the macro records raw pixel positions, so the image's µm
calibration does not matter and you do not need to change it.

---

## 1. Choose the cells (before clicking anything)

Pick **6–10 cells** that:

- are **born during the movie** (you can see the division that creates them)
  and **divide again before the movie ends**, so you can follow one complete
  cell cycle;
- stay in focus and away from the image edge for the whole cycle;
- are not buried in a dense clump (touching one or two neighbours is fine);
- are spread across the field of view, ideally including both daughters of
  one division.

Give each cell your own label: `c1`, `c2`, …

**How often to annotate:** aim for **20–30 time points per cell cycle**. If a
cycle spans about 30 frames, annotate every frame (N = 1); about 60 frames,
every 2nd frame (N = 2). Denser sampling around the first third of the cycle
is where NETO usually happens, but keep N constant within a cell.

---

## 2. Run the macro for one cell

1. Move the time slider to the cell's **birth frame**: the first frame in
   which the two daughters are separate cells (septum split).
2. Plugins › Macros › Run… › `BS_track_cell.ijm`.
3. Enter the cell label, your initials, the birth frame (pre-filled with the
   current slider position) and N.
4. Choose a folder for the CSV. One file is written per cell, named after the
   movie and the cell label. It is saved after every frame.
5. For each frame the macro asks for three clicks, one at a time. Click with
   the point tool, then press **OK**:
   1. **Birth scar midpoint:** the point on the cell's centerline midway
      between the two scar ridges (the same definition as the previous
      packets).
   2. **New pole tip:** the tip of the end formed at the division that
      created this cell. At birth it faces the sister cell.
   3. **Old pole tip:** the tip of the other end.

   You can zoom and pan freely while the prompt is open. If you click more
   than once, the last click counts.
6. After the three clicks choose:
   - **Save and go to the next frame**: moves N frames ahead.
   - **Save; the cell DIVIDED in this frame**: use in the first frame where a
     septum is complete or the cell has visibly split. This ends the cell.
   - **Save; cell lost or ambiguous**: ends the cell early (out of focus,
     leaves the field, can no longer tell which cell it is).
   - **Redo this frame**: discards this frame's clicks and asks again.
7. Repeat from step 1 for the next cell.

---

## 3. Rules that keep the data consistent

- **The new pole never changes identity during a cycle.** Decide which end is
  new at birth (it touches the sister) and follow that same end in every
  frame, even after it starts growing.
- **Scar not visible yet?** Right after division the scar can be hard to see.
  Skip only the scar click (press OK with nothing selected) and still click
  both poles. Write "scar not visible" in the notes.
- **Unsure about a point?** Click your best estimate and say so in the notes
  (for example "scar faint", "pole blurred"). A note is more useful than a
  skipped point.
- **Pole tip** means the outermost point of the rounded end, on the cell's
  long axis, at the edge of the cell (the same edge you would trace for area).
- Keep N the same for the whole cell.

---

## 4. If the macro does not work

Use FIJI's Point tool and `Analyze › Measure` (with `Image › Properties` set
to 1 pixel = 1 unit), and fill in `manual_tracking_template.csv` by hand: one
row per cell per frame, delete the EXAMPLE row. `frame_slider` is the frame
number FIJI shows; `frame_index` is that number minus 1. `status` is `ok`,
`divided` or `lost`.

---

## 5. Return

Send back the folder of CSV files (one per cell) and the name of the movie
file they belong to. Time per cell: roughly 20–40 minutes for a full cycle.
