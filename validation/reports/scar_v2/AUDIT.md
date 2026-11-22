# Audit before detector changes

Reviewed detection.py, geometry.py, segmentation.py, pipeline.py, poles.py usage,
visualization.py, config.py, io_utils.py, both notebooks, comparison/visualization
scripts, v2 annotation CSV, measurement protocol and sparse frame manifest.

- v2: 40 annotated positives, 39 frames, four 2304 × 2304 sparse H5 packages.
  Preserve `frame_numbers`; local H5 indices are not original movie frame IDs.
- Current detector already uses positive curvature minus a circular local median
  (±25 samples), NOT raw curvature. Windows sum two side means; a strong feature
  on only one side is already allowed, but endpoints are independently maximized
  on both sides. The opposite weak endpoint can therefore shift the midpoint.
- Window candidates are only 25 longitudinal centers with 300 contour samples.
  Their spacing scales with cell length, which matters for a fixed 5-pixel goal.
- Spline is periodic but curvature uses nonperiodic finite differences, creating
  seam errors. Samples are uniform in original chord-length spline parameter,
  not in smoothed arc length. Smoothing uses absolute residual budget 40.
- Fixed exclusion removes both candidate centers AND individual samples within
  12% of poles. This can suppress annotated near-pole scars.
- Score multiplies curvature by angle, a Gaussian pole-distance prior and a
  width weight. Width is not merely diagnostic. Documentation is stale.
- Most importantly, when a neighbor pole hint exists, `_select_best_candidate`
  ignores scores and chooses the closest midpoint to that pole. This is a hard
  positional selection, not the documented soft prior.
- Pipeline obtains neighbor hints before detection; lineage reassignment and
  temporal stabilization can replace detections afterward. Sparse detection
  evaluation must stop before tracking and stabilization.
- Existing notebook cache keys depend only on local index, so switching videos
  or models can silently reuse wrong labels. Sparse name association depends on
  GT scar containment and cannot independently identify scar-free cells.
- Existing notebook recall/precision measure prediction presence irrespective of
  distance. Localization fraction excludes missing predictions. Blank scars are
  treated as negatives even though the protocol allows uncertainty.
- All v2 rows have coordinates. Population detection precision is unidentifiable
  without annotated negatives or exhaustive annotation. Report annotated-cell
  localization precision at 5 px and detection recall separately.
- Existing plots show raw curvature and summed windows, but not all score
  components, every candidate, the local baseline or the selection override.

Validation protocol: cache full-frame segmentation with image hashes and model
provenance, detect independently of GT, then associate by mask containment.
Failures are assigned in order: segmentation/association, no candidate, wrong
selection if a snapped candidate is within 5 px, otherwise localization error.
This last label is operational, not a claim that segmentation is biologically
correct. Visual review must identify gross contour errors separately.

## Further validation findings

The original package sampler admits only `seg_quality == 'ok'` cells, so these
annotations are conditional on the historical segmenter's quality filter.
They cannot establish unconditional segmentation recall in a complete movie.
Historical measurement tables contain pole endpoints for all 40 named cells.
`cell_anchors.csv` records their midpoint, source path and source SHA-256. These
provide an independent spatial check of the GT-containment association, without
rerunning or modifying tracking and without using historical scar predictions.

Score-component ablations are compared against a **score-only control**, because
the legacy hard neighbor override otherwise bypasses those components. The
sampling-density ablation holds the number of candidate windows at 25 and doubles
the baseline radius with the contour sample count; changing all three together
would not isolate contour sampling.

The E10/frame198 diagnostic adds a distinct failure: a smooth-looking old cap
wins by score (92.01 px error), even though a generated candidate is only 0.50 px
from the annotation. No neighbor hint is supplied here. This motivates testing
concave-only and absolute local curvature residuals in addition to convex-only
excess; a normal convex cap should not automatically outrank an indentation.
It also motivates separating analytic derivatives from the combined analytic
plus arc-length resampling ablation. These are mechanism tests, not parameter
optimization. Final conclusions require the complete four-video comparison.
