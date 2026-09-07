# Manual Tracking & Division Ground-Truth Protocol (FIJI)

Purpose: independently re-track a handful of short frame sequences by eye so
we can measure two specific failure modes:

1. **Missed or false division detection** — the algorithm fails to notice a
   cell dividing, or claims a division happened when it didn't.
2. **Track fragmentation** — a cell temporarily drops out of tracking (e.g.
   a bad segmentation frame or two) and comes back under a brand-new
   identity instead of resuming its original one.

This is a separate exercise from the area / compartment-length / birth-scar
measurements in `GROUND_TRUTH_PROTOCOL.md` and uses different reference
images.

---

## Why the labels are different this time

For the other measurements, cells were labeled with their real tracking ID
so you could tell which cell was which. Here, that would defeat the point —
if you could just read off "was the label the same in the next frame," you
wouldn't be doing independent tracking, you'd be grading the algorithm's
homework using its own answer key.

So instead, each frame in this set is labeled with a **number that is
reassigned fresh every single frame** (cells are just numbered 1, 2, 3…
top-to-bottom in that frame). The same physical cell will likely have a
*different* number in the next frame — you have to figure out which blob is
which yourself, by eye, using position, size, and shape.

---

## 1. Files you'll receive

For one or more short **runs** (a run = a contiguous stretch of ~10-20
frames, e.g. frames 10-25):

- `<run_id>_frame####_trackingblind.png` — one image per frame in the run,
  cells outlined and numbered (freshly, per frame).
- The corresponding raw frames/movie if you want to view it as a time-lapse
  rather than flipping between stills.

You will **not** receive the `*_KEY_do_not_share.csv` file — that's the
answer key, kept on our side for scoring.

---

## 2. What to do, per run

Step through the run's frames **in order**. For each cell you can
confidently follow from one frame to the next:

### (a) Track identity — `lineage_tracking_template.csv`

Assign your own tracking ID (any label you like, e.g. `1`, `2`, `A`, `B` —
just needs to be consistent within this run). Every time you see a cell you
believe is the *same physical cell* as one you already gave an ID, write a
new row for it in this run with the *same* `true_track_id`. Every time you
see a cell that's new (just entered the frame, or you can't confidently
match it to anything earlier), give it a **new** `true_track_id`.

**Important — divisions are the one exception:** if a cell divides, do
**not** carry its `true_track_id` forward to either daughter. Both daughters
get **brand-new** `true_track_id`s, and the division itself gets recorded
separately (see below). This keeps "same identity" and "division happened"
unambiguous — if you're not sure whether something is a division or just one
cell continuing, err on the side of calling it a division if you see the
cell physically separate into two.

One row per (frame, cell) you're confident about:

| Column | What to enter |
|---|---|
| `run_id` | the run name, e.g. `run0` |
| `frame` | frame index |
| `frame_local_id` | the number printed on that blob in that frame's image |
| `true_track_id` | your own consistent ID for that physical cell |
| `scorer_initials` | your initials |
| `notes` | anything uncertain |

Skip frames/cells you're not confident about rather than guessing — a
missing row is fine, a wrong guess corrupts the comparison.

### (b) Division events — `division_events_template.csv`

Every time you observe a real division within the run, add one row:

| Column | What to enter |
|---|---|
| `run_id` | the run name |
| `division_frame` | the **first frame** you see two separate cells instead of one |
| `parent_frame_local_id` | the parent's number in the **frame before** `division_frame` |
| `daughter1_frame_local_id` | one daughter's number in `division_frame` |
| `daughter2_frame_local_id` | the other daughter's number in `division_frame` |
| `scorer_initials` | your initials |
| `notes` | anything uncertain |

---

## 3. How many runs

3-5 runs of ~10-20 frames each, sampled from different points across the
movie (start, middle, end; different regions of the field of view if
crowding varies) gives a much more honest error rate than one long run or a
run cherry-picked because it looked hard. Don't specifically seek out
"confusing" frames to test the algorithm — a representative sample is what
lets us say "this happens X% of the time," not just "this can happen."

---

## 4. Returning results

Send back the completed `lineage_tracking_template.csv` and
`division_events_template.csv` (same headers, `EXAMPLE` rows removed).
