# Evaluation & Ground Truth

BS-Detector ships with a quantitative accuracy harness so you can measure how
well it performs against ground truth, compare runs objectively, and (later)
drive automatic hyperparameter optimization.

The harness is intentionally **lightweight** — it depends only on `numpy` and
`scipy`, so you can score a saved detector run offline or in CI without
installing Cellpose/PyTorch or re-running segmentation.

---

## 1. The ground-truth schema

All labels — whether typed by a lab mate, pulled from Carmen's measurements, or
derived automatically from a septin fluorescence channel — share one schema
(`pombe_tracker/ground_truth.py`). One CSV row per annotated cell-in-a-frame:

```
frame,gt_id,y,x,scar_present,scar_y,scar_x,new_pole_y,new_pole_x,old_pole_y,old_pole_x,length,notes
```

| Column | Required | Meaning |
|---|---|---|
| `frame` | ✅ | 0-based frame index |
| `y`, `x` | ✅ | cell centroid (row, col) — used to match GT to detections |
| `gt_id` | | annotator's stable name (enables identity/lineage metrics) |
| `scar_present` | | `1`/`0` — is a birth scar visible |
| `scar_y`, `scar_x` | | birth-scar midpoint |
| `new_pole_y/x`, `old_pole_y/x` | | pole coordinates |
| `length` | | pole-to-pole length [px] |
| `notes` | | free text |

**Everything except `frame`/`y`/`x` is optional.** A blank cell means
"unknown" and that metric is silently skipped. So a length-only file (Carmen's
data) and a scar-position-only file (septin) are both valid.

> **Coordinates are `(y, x) = (row, col)`** throughout — matching skimage
> `regionprops` and the pipeline. Keep annotation tools in the same convention.

### Generating a pre-seeded template

So annotators don't have to locate cells from scratch, seed a template with one
row per detected cell (frame + centroid + lineage name pre-filled):

```python
from pombe_tracker.ground_truth import write_template
write_template('to_annotate.csv', all_results)   # all_results from run_pipeline
```

---

## 2. Scoring a run

```python
from pombe_tracker.evaluation import evaluate, export_eval_json

# After running the pipeline, dump the fields the scorer needs:
export_eval_json(all_results, 'detector.json')

# Score against ground truth (CSV path or in-memory dict both work):
report = evaluate('detector.json', 'ground_truth.csv')
print(report)
```

or from the command line:

```bash
python scripts/evaluate.py detector.json ground_truth.csv --out report.json
```

### Metrics produced

| Group | Metrics |
|---|---|
| **Cell matching** | precision / recall of cell detection (Hungarian on centroid) |
| **Scar detection** | precision / recall / F1 (TP/FP/FN/TN) |
| **Scar localization** | mean / median / p90 pixel error; fraction within tolerance |
| **Pole assignment** | fraction of cells with correct new/old labelling |
| **Length agreement** | Bland–Altman bias, MAE, limits of agreement, Pearson r |
| **Objective** | single scalar in `[0, 1]` (higher = better) |

The **objective** is a weighted blend of whichever components have labels
(weights renormalize over what's present), so it's directly usable as the value
an optimizer maximizes. Tune weights and matching/localization tolerances via
`EvalConfig`.

---

## 3. Ground-truth sources

### Manual annotation (lab mates)
Use `write_template(...)` to pre-seed, hand the CSV to an annotator to fill in
scar/pole columns, then `evaluate(...)`. Best for pole identity and lineage,
which fluorescence can't fully resolve.

### Carmen's length data
Drop measured lengths into the `length` column (centroid still required for
matching). The Bland–Altman block validates detector lengths against the manual
gold standard.

### Septin fluorescence (automatic, no human in the loop)
Septins localize to the division septum = the birth scar. With **registered**
brightfield + fluorescence of the same cells, `septin_scar_position(...)`
projects fluorescence onto the cell long axis and returns the brightest
transverse band as the scar midpoint — turning every fluorescence frame into
automatic ground truth.

```python
from pombe_tracker.ground_truth import septin_scar_position, GTRecord
# per cell, using geometry already computed by the pipeline:
mp, conf = septin_scar_position(fluor_frame, center, axis, long_norm, smooth_pts)
```

> Requires paired BF + fluorescence registered to the same pixel grid. Until
> that data exists, manual annotation + Carmen's lengths drive the harness; the
> septin path then plugs in with no changes to `evaluate()`.

---

## 4. Running the tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -q
```

The tests use synthetic data only (no Cellpose, no skimage, no real images), so
they run in seconds and are safe for CI.
