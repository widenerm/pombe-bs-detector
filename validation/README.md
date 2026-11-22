# validation/

The evidence behind every number in [docs/VALIDATION.md](../docs/VALIDATION.md): how
annotations were collected, the anonymized annotations themselves, the scripts that
scored the detector, and the reports. Raw images, masks and caches stay out of git
(`validation/runs/`, `*.h5`, `*.tif`), so the scripts that segment need the lab's
movies to rerun.

## Reports (start here)

| Folder | What it shows |
|---|---|
| [`reports/scar_v3/`](reports/scar_v3/) | the current scar detector: design, ablations and held-out refits (`REPORT.md`), and the two-annotator comparison (`INTERRATER.md`) |
| [`reports/neto_annotator_b/`](reports/neto_annotator_b/) | through-time comparison with one annotator's tracks of 6 cells: pole labels, scar position, new-end length, NETO |
| [`reports/growth_curves/`](reports/growth_curves/) | growth curves from whole movies vs hand tracking (wild type; `11192025_rgf1gpd1_YE/` for *rgf1*Δ *gpd1*Δ) |
| [`reports/scar_v2/`](reports/scar_v2/) | an earlier audit of the comparison design (frame-only matching, caches, failure lists) |
| [`reports/scar_v1/`](reports/scar_v1/) | the first comparison, of the original outline-curvature detector |
| [`test_set/`](test_set/) | the frozen blind test: rules, the 135 sampled cells, segmentation provenance |

## Annotation protocols and packets

- [`GROUND_TRUTH_PROTOCOL.md`](GROUND_TRUTH_PROTOCOL.md): how annotators mark the
  newest birth scar and the poles on single frames.
- [`TRACKING_PROTOCOL.md`](TRACKING_PROTOCOL.md): lineage and division annotation.
- [`packets/`](packets/): what annotators received. `dev1`, `dev2` are the development
  packets, `test` the blind test, `neto` the through-time tracking packet (protocol and
  ImageJ macro `BS_track_cell.ijm`), and `template` is a blank packet. Images are not
  in git.
- Templates: `ground_truth_template.csv`, `lineage_tracking_template.csv`,
  `division_events_template.csv`.

## Annotations

[`annotations/`](annotations/) holds the returned annotations, with annotators identified
only as A and B:

| File | Contents |
|---|---|
| `dev1_annotator_a.csv`, `dev2_annotator_a.csv` | 80 cells, scar midpoint and pole lengths (the labels the detector weights were fitted on) |
| `dev1_annotator_b.csv`, `dev2_annotator_b.csv` | the same 80 cells, annotated independently |
| `neto_annotator_b/` | 6 cells tracked from birth to division (original 0.11 µm/px pixels) |
| `neto_annotator_b_resampled/` | the same clicks mapped to the resampled 0.0647 µm/px movie |

## Scripts

Run from the repository root as `python -m validation.<name>`.

| Script | Purpose |
|---|---|
| `scar_cells.py` | build frozen per-cell records (crops, contours, annotation) from cached masks |
| `scar_eval.py` | score a detector configuration on those records |
| `scar_fit.py` | fit the profile detector's six weights; held-out refits |
| `scar_ablations.py` | one-mechanism-at-a-time ablations |
| `scar_interrater.py` | two annotators vs each other and vs the detector, with a gallery of disputed cells |
| `scar_batch.py`, `scar_report.py`, `scar_diagnostics.py` | frame-only batch scoring, metric tables and failure galleries |
| `compare_compartments.py` | through-time tracks vs pipeline output (pole labels, scar, new-end length, NETO) |
| `test_packet.py` | built the blind-test packet (random frames and cells, fixed seed) |
| `render_blinded_frames.py`, `render_tracking_blind_frames.py`, `sample_for_scientists.py` | make annotation packets |
| `compare_to_ground_truth.py`, `compare_tracking_to_ground_truth.py`, `apply_lineage_corrections.py`, `visualize_results.py`, `build_colab_handoff.py` | earlier comparison and hand-off tools (v1/v2) |

`h5_for_drive/` describes the sparse H5 files the validation notebooks read; the files
themselves are not in git.
