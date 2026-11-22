# Birth-scar test packet

Thank you for doing another round. The task is the same as in the two earlier
packets: for each outlined cell, measure the area, the birth scar midpoint,
the two pole lengths and which pole you think is new. The full instructions
are in `GROUND_TRUTH_PROTOCOL.md`.

These cells come from movies we have never used before: the wild-type YE
movies from 3/14/2025 and 3/19/2025, four fields each. That makes this the
fair test of the detector, which is now frozen. It also means a few details
differ from last time.

## What's different

**1. Three batches.** There are three templates,
`ground_truth_template_batch1.csv` to `batch3.csv`, with 45 cells each.
Every batch covers all eight movies and is a random sample, so a finished
batch is useful on its own. Please do batch 1 first and send it back; do
batches 2 and 3 whenever you can.

**2. A new `status` column.** Enter one of these for every cell:

| status | When to use it |
|---|---|
| `ok` | A whole, single cell. Measure everything as usual. |
| `bad_outline` | The blue outline is not one whole cell: two cells merged, part of a cell, or debris. Leave the other fields blank. |
| `dividing` | A septum is visible or the cell is splitting. Measure what you can. |
| `no_scar` | A whole cell, but you can't find its birth scar. Record the area; leave the scar and pole lengths blank. |

Some cells will have bad outlines on purpose. They are a random sample of
everything the software tries to analyze, so we can count how often the
outline itself is wrong.

**3. Mark the most recent birth scar,** the one nearest the new end. If it is
hidden, leave the scar fields blank and say so in `notes`. Please don't use an
older scar instead.

**4. Pole A on horizontal cells.** Pole A is still the upper pole. If both
poles are at about the same height, call the **left** pole A.

**5. Only the outlined cells.** Each image in `images/` outlines only the
cells to measure, labelled `c01`, `c02`, … from top to bottom. Names repeat in
different frames, so always copy `video_id` and `frame` exactly as given in
the template.

**6. Please work independently.** Don't look at earlier annotations, or at
anyone else's, for these cells.

## Files

- `raw_frames/*_raw.tif`: the original 16-bit frames. **Measure on these**, in
  pixel units (protocol section 0).
- `images/*_blinded.png`: use these only to find which cell is which. They are
  at a different scale, so don't measure on them.
- `ground_truth_template_batch*.csv`: one row per cell, with `video_id`,
  `raw_frame_file`, `image_file`, `frame` and `cell_name` filled in. `frame`
  is the frame number in the original movie (counting from 0); it is only an
  identifier.

## Sending it back

Send the filled-in batch CSV(s), with your initials in `scorer_initials`.
That's all we need.
