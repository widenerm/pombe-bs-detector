# Analysing a returned NETO packet

Carmen returns one CSV per tracked cell (from `BS_track_cell.ijm`) plus the
movie file name.

1. **Run BS-Detector on the same movie, all frames.** Convert it to H5 if
   needed (`frames` dataset, frames × height × width, brightfield channel) and
   run the main Colab notebook or `run_pipeline` + `stabilize_scars` +
   `export_csv`. The default CSV now includes `scar_midpoint`,
   `new_pole_point` and `old_pole_point`, which the comparison needs. Frame
   numbering must match: the macro's `frame_index` is FIJI's frame minus 1,
   and the pipeline's `frame` starts at 0. Do not drop leading frames.

2. **Compare:**

   ```bash
   python -m validation.compare_compartments path/to/carmen_csvs \
       --measurements path/to/measurements.csv --frame-minutes 5 --out results/neto
   ```

   `--frame-minutes` is the acquisition interval (omit it to report NETO in
   frames).

3. **Read `results/neto/`:**
   - `SUMMARY.md`: per cell, how often the detector's new/old pole labels
     agree with Carmen's, scar position within 5 px, mean error of the
     new-end length, and NETO time from each source.
   - `new_end_length.png`: new-end length through time per cell, manual in
     red and detector in blue, with each NETO estimate as a dotted line.
   - `per_frame.csv`, `per_cell.csv`: every number behind the summary.

## How matching and NETO work

- Each annotated frame is matched to the detector's cell in that frame whose
  two pole tips lie within `--max-pole-px` (default 20 px) of Carmen's, in
  either orientation. Lineage names are not used, so tracking or naming
  errors don't break the comparison. Wrong new/old labels are counted
  separately (`same_pole_labels`) and lengths are compared against Carmen's
  pole identity.
- Lengths follow the detector's convention: straight line from the scar
  midpoint to each pole tip.
- NETO is the hinge time t0 of the best fit of
  `L_new = a` (before t0) and `a + b (t - t0)` (after), searched over the
  annotated times, for each source separately. A cell needs at least 6 frames
  with both the scar and the new pole; frames where Carmen skipped the scar
  are left out of their series.
