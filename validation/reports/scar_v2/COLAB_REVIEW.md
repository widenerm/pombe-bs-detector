# Four-video Colab scientific review

The 90% within-5-pixel target was not reached. The legacy detector localized 12/40 annotations (30%); the experimental weak-neighbor-hint, width-diagnostic detector localized 14/40 (35%). The experimental change worsened median error, MAE and p90, so it should remain opt-in and must not replace the legacy default. The best prespecified ablation localized 16/40 (40%), still far short of the required 36/40.

## Verification and scope

The review contains 40 unique video/frame/cell annotation keys in 39 frames across four videos. Recall and MAE were independently recomputed from per-cell rows and agree with the saved metrics. Baseline and experimental cache manifests are byte-identical. Both use Cellpose 4.0.9 and the same SHA-256-verified cpsam weights: 19 masks were produced on MPS and 20 on CUDA, in bfloat16 (the original MPS metadata predates an explicit dtype field). Device arithmetic may differ, but comparisons use frozen identical masks. No tracking or temporal stabilization was evaluated or changed.

All 40 annotated cells received a prediction. Prediction-presence recall is therefore 100%, but this is not successful scar detection: localization recall is 30%/35%. Population precision is not estimable because these are positive annotations and unannotated cells are not negatives. The reported 5-pixel precision is conditional on the annotated cells. Historical sampling favored segmentation-quality-ok cells; these results do not establish performance on arbitrary microscopy frames. Repeated cells and related lineages also prevent treating all rows as independent biological replicates.

## Failure analysis

The automated taxonomy finds no missing/filtered segmentation, association, or no-candidate failures. Legacy failures are 25 wrong-candidate selections and 3 localization errors; experimental failures are 23 wrong selections and the same 3 localization errors. This does not prove contours are perfect: it means the independent anchor association and candidate tests passed.

A generated candidate lies within 5 pixels in 37/40 cases (92.5%). Ranking is therefore the largest measured bottleneck. This oracle uses manual coordinates and many spatial windows; it is a candidate-coverage upper bound, not evidence that curvature can identify 92.5% of scars without labels.

I reconstructed all 40 experimental diagnostic figures using the exported contour/score arrays and original H5 frames. I visually inspected the two largest errors, a large experimental regression, and all three localization failures, including raw contour, spline, PCA, pole endpoints, curvature/baseline/excess, cap exclusion, window scores, and selection components:

- March 20 xy1 / frame 198 / E10: 92.01-pixel error although candidate 2 is 0.50 pixels from GT. No neighbor hint is available. The selected opposite-end candidate has roughly eight times the oracle candidate's score. The contour broadly follows the cell; stronger non-scar curvature near the other end dominates.
- March 27 xy1 / frame 221 / S0: 101.40-pixel error; candidate 20 is within 3.02 pixels but ranks seventh by score. No hint is available. The strongest window score is at the opposite end. Large cap curvature near the annotated end is mostly excluded; the no-cap and smooth-baseline ablations nevertheless do not solve the population-level problem.
- March 27 xy3 / frame 161 / C01: the experimental result is 54.23 pixels away, versus legacy 7.07 pixels. Its correct-side candidate has only about one quarter of the winning score. Weakening a useful neighbor hint allows a central contour feature to win; the proposed weak hint is not a generally successful replacement.
- March 27 xy1 / frame 185 / AJ: nearest candidate is 7.16 pixels away and the dense contour-cross-section diagnostic is 6.01 pixels away. The irregular contour/axis geometry or annotation correspondence needs manual review; ranking alone cannot bring this case within tolerance with these candidates.
- March 27 xy1 / frame 236 / S1: candidate oracle 5.32 pixels, dense cross-section oracle 3.30 pixels. This supports a candidate-coordinate refinement experiment rather than only a rank change.
- March 27 xy3 / frame 147 / E0: candidate oracle 8.54 pixels, dense cross-section oracle 0.00 pixels. Available contour geometry permits the location, but the current paired-peak midpoint construction does not propose it accurately. The dense oracle includes GT longitudinal position and is diagnostic only.

## Targeted ablations

Except for the legacy comparison, mechanisms use the score-only selection control (ignore_neighbor_hint), which reaches 14/40. Compare mechanisms to that control, not only to the 12/40 legacy hard-hint result. These are development comparisons, not independent validation after model selection.

| Mechanism | Within 5 px | Fraction | MAE px | Candidate oracle ≤5 px |
|---|---:|---:|---:|---:|
| legacy | 12/40 | 30.0% | 12.82 | 37/40 |
| ignore_neighbor_hint | 14/40 | 35.0% | 15.55 | 37/40 |
| weak_neighbor_hint | 14/40 | 35.0% | 15.55 | 37/40 |
| raw_curvature | 10/40 | 25.0% | 35.41 | 38/40 |
| normalized_excess | 11/40 | 27.5% | 16.75 | 37/40 |
| concave_excess | 1/40 | 2.5% | 45.04 | 30/40 |
| absolute_excess | 8/40 | 20.0% | 22.65 | 37/40 |
| no_cap_exclusion | 7/40 | 17.5% | 14.84 | 35/40 |
| smooth_cap_baseline | 11/40 | 27.5% | 18.26 | 35/40 |
| require_two_features | 13/40 | 32.5% | 24.75 | 29/40 |
| one_feature_anchor | 16/40 | 40.0% | 15.49 | 36/40 |
| window_excess_centroid | 14/40 | 35.0% | 15.13 | 38/40 |
| no_recency_prior | 12/40 | 30.0% | 21.90 | 37/40 |
| weak_recency_prior | 12/40 | 30.0% | 21.90 | 37/40 |
| width_diagnostic | 14/40 | 35.0% | 15.55 | 37/40 |
| analytic_derivatives | 11/40 | 27.5% | 19.15 | 36/40 |
| periodic_arclength | 14/40 | 35.0% | 17.17 | 36/40 |
| smoothing_half | 9/40 | 22.5% | 52.01 | 37/40 |
| smoothing_double | 10/40 | 25.0% | 18.88 | 36/40 |
| sampling_double | 13/40 | 32.5% | 16.80 | 35/40 |

The current detector already uses local-median curvature excess. Raw curvature (10/40) and normalized excess (11/40) do not improve over the 14/40 score-only control. Removing cap exclusion (7/40) is harmful; the tested Gaussian smooth cap baseline (11/40) also fails to improve. This tests one smooth-baseline implementation, not all possible cap models.

Requiring two features lowers candidate coverage to 29/40 and achieves only 13/40. A single-feature anchor reaches 16/40, the best development result, suggesting corresponding boundary peaks should not be compulsory. Its 15.49-pixel MAE is still worse than legacy, so it is not ready to adopt as a solution.

Removing or weakening the nearest-pole recency prior gives 12/40 and worse MAE. This prior is distinct from the neighbor-derived pole hint. Merely weakening positional evidence is not supported as a general fix. Width as a diagnostic gives the same results as score-only selection; these data provide no evidence to treat width as biologically reliable. Analytic derivatives, uniform arclength, changed smoothing, and doubled sample density do not improve recall over the score-only control. Contour sampling alone is not the main remedy supported by these tests.

## Algorithmic disposition and next experiment

Retain legacy as the default and keep experimental scoring and mechanism ablations available for comparison. Do not promote a change based only on the first-video improvement. The evidence supports separating candidate-coordinate refinement from candidate ranking and adding a discriminating measurement beyond maximum contour curvature.

The next most informative experiment is a blinded image-evidence audit of the GT-near candidate and the strongest distractor in each cell, especially opposite-pole mistakes E10/S0. Annotate whether each candidate actually has a visible scar-associated feature and whether a second boundary feature exists. Include negative candidate locations and independently reviewed scar coordinates. This tests whether local image intensity/texture supplies information missing from contour scores, before committing to an image-based ranker. Freeze segmentation and candidates; compare curvature-only ranking to a small prespecified image-feature ranker using held-out videos and lineage-aware grouping, reporting regressions as well as recall. A separate focused localization experiment should replace paired-peak midpoints with continuous longitudinal cross-section estimates for S1/E0. AJ first needs contour/annotation review. The present 40 annotations remain a development set; the 90% claim needs a new independent set.

## Complete metrics and exact out-of-tolerance cells

### Numerical validation results

| Run | n | Detected | Recall ≤5 px | Precision ≤5 px* | Median px | MAE px | p90 px |
|---|---:|---:|---:|---:|---:|---:|---:|
| legacy | 40 | 40 | 30.0% | 30.0% | 7.65 | 12.82 | 22.60 |
| experimental | 40 | 40 | 35.0% | 35.0% | 8.10 | 15.55 | 48.98 |

*Precision is localization precision conditional on annotated cells. Population detection precision is not estimable from this positive-only set. Localization recall and fraction within 5 pixels both divide by all annotated positives; missing predictions count as failures. Error quantiles and MAE use finite matched predictions. Detection recall is reported separately below.*

## legacy: per-video performance

| Run | n | Detected | Recall ≤5 px | Precision ≤5 px* | Median px | MAE px | p90 px |
|---|---:|---:|---:|---:|---:|---:|---:|
| experiment_03202025_wt_ymd494__xy1 | 9 | 9 | 44.4% | 44.4% | 5.76 | 16.03 | 36.16 |
| experiment_03272025_wt_ymd494__xy1 | 13 | 13 | 23.1% | 23.1% | 9.09 | 17.74 | 27.15 |
| experiment_03272025_wt_ymd494__xy2 | 5 | 5 | 60.0% | 60.0% | 3.35 | 6.81 | 12.88 |
| experiment_03272025_wt_ymd494__xy3 | 13 | 13 | 15.4% | 15.4% | 7.07 | 7.99 | 12.15 |

Detection recall: 100.0%.

Candidate oracle within 5 px: 37/40.

Failure categories: {"segmentation failure": 0, "cell-association failure": 0, "no-candidate failure": 0, "wrong-candidate selection": 25, "localization error": 3}

## experimental: per-video performance

| Run | n | Detected | Recall ≤5 px | Precision ≤5 px* | Median px | MAE px | p90 px |
|---|---:|---:|---:|---:|---:|---:|---:|
| experiment_03202025_wt_ymd494__xy1 | 9 | 9 | 66.7% | 66.7% | 3.54 | 13.68 | 25.20 |
| experiment_03272025_wt_ymd494__xy1 | 13 | 13 | 15.4% | 15.4% | 9.09 | 19.72 | 44.07 |
| experiment_03272025_wt_ymd494__xy2 | 5 | 5 | 40.0% | 40.0% | 11.88 | 13.00 | 25.15 |
| experiment_03272025_wt_ymd494__xy3 | 13 | 13 | 30.8% | 30.8% | 6.49 | 13.65 | 44.95 |

Detection recall: 100.0%.

Candidate oracle within 5 px: 37/40.

Failure categories: {"segmentation failure": 0, "cell-association failure": 0, "no-candidate failure": 0, "wrong-candidate selection": 23, "localization error": 3}

## Exact cells outside tolerance: legacy

| Video | Original frame | Cell | Error px | Oracle px | Classification |
|---|---:|---|---:|---:|---|
| experiment_03202025_wt_ymd494__xy1 | 74 | E | 7.159 | 3.953 | wrong-candidate selection |
| experiment_03202025_wt_ymd494__xy1 | 174 | AC | 22.192 | 4.138 | wrong-candidate selection |
| experiment_03202025_wt_ymd494__xy1 | 182 | B11 | 8.493 | 2.151 | wrong-candidate selection |
| experiment_03202025_wt_ymd494__xy1 | 198 | E10 | 92.014 | 0.500 | wrong-candidate selection |
| experiment_03202025_wt_ymd494__xy1 | 201 | S | 5.755 | 1.904 | wrong-candidate selection |
| experiment_03272025_wt_ymd494__xy1 | 77 | G | 26.280 | 4.596 | wrong-candidate selection |
| experiment_03272025_wt_ymd494__xy1 | 141 | AC | 9.708 | 2.264 | wrong-candidate selection |
| experiment_03272025_wt_ymd494__xy1 | 149 | C101 | 11.319 | 3.536 | wrong-candidate selection |
| experiment_03272025_wt_ymd494__xy1 | 156 | D | 8.132 | 3.021 | wrong-candidate selection |
| experiment_03272025_wt_ymd494__xy1 | 174 | E01 | 13.309 | 4.528 | wrong-candidate selection |
| experiment_03272025_wt_ymd494__xy1 | 185 | AJ | 27.370 | 7.159 | localization error |
| experiment_03272025_wt_ymd494__xy1 | 208 | K | 9.090 | 2.151 | wrong-candidate selection |
| experiment_03272025_wt_ymd494__xy1 | 216 | C101 | 8.223 | 2.828 | wrong-candidate selection |
| experiment_03272025_wt_ymd494__xy1 | 221 | S0 | 101.400 | 3.021 | wrong-candidate selection |
| experiment_03272025_wt_ymd494__xy1 | 236 | S1 | 6.801 | 5.315 | localization error |
| experiment_03272025_wt_ymd494__xy2 | 46 | C1 | 11.885 | 2.236 | wrong-candidate selection |
| experiment_03272025_wt_ymd494__xy2 | 237 | V | 13.551 | 1.118 | wrong-candidate selection |
| experiment_03272025_wt_ymd494__xy3 | 58 | B1 | 11.413 | 2.151 | wrong-candidate selection |
| experiment_03272025_wt_ymd494__xy3 | 61 | D | 13.365 | 1.803 | wrong-candidate selection |
| experiment_03272025_wt_ymd494__xy3 | 142 | B10 | 5.831 | 0.354 | wrong-candidate selection |
| experiment_03272025_wt_ymd494__xy3 | 146 | F11 | 6.255 | 1.118 | wrong-candidate selection |
| experiment_03272025_wt_ymd494__xy3 | 147 | E0 | 9.657 | 8.544 | localization error |
| experiment_03272025_wt_ymd494__xy3 | 151 | F11 | 5.590 | 2.236 | wrong-candidate selection |
| experiment_03272025_wt_ymd494__xy3 | 161 | C01 | 7.071 | 2.121 | wrong-candidate selection |
| experiment_03272025_wt_ymd494__xy3 | 171 | E1 | 9.103 | 3.918 | wrong-candidate selection |
| experiment_03272025_wt_ymd494__xy3 | 212 | AR | 12.334 | 3.953 | wrong-candidate selection |
| experiment_03272025_wt_ymd494__xy3 | 219 | U | 10.470 | 3.808 | wrong-candidate selection |
| experiment_03272025_wt_ymd494__xy3 | 225 | B111 | 6.519 | 2.550 | wrong-candidate selection |

## Exact cells outside tolerance: experimental

| Video | Original frame | Cell | Error px | Oracle px | Classification |
|---|---:|---|---:|---:|---|
| experiment_03202025_wt_ymd494__xy1 | 174 | AC | 8.078 | 4.138 | wrong-candidate selection |
| experiment_03202025_wt_ymd494__xy1 | 182 | B11 | 8.493 | 2.151 | wrong-candidate selection |
| experiment_03202025_wt_ymd494__xy1 | 198 | E10 | 92.014 | 0.500 | wrong-candidate selection |
| experiment_03272025_wt_ymd494__xy1 | 44 | F0 | 48.523 | 3.606 | wrong-candidate selection |
| experiment_03272025_wt_ymd494__xy1 | 77 | G | 26.280 | 4.596 | wrong-candidate selection |
| experiment_03272025_wt_ymd494__xy1 | 141 | AC | 9.708 | 2.264 | wrong-candidate selection |
| experiment_03272025_wt_ymd494__xy1 | 149 | C101 | 11.319 | 3.536 | wrong-candidate selection |
| experiment_03272025_wt_ymd494__xy1 | 156 | D | 8.132 | 3.021 | wrong-candidate selection |
| experiment_03272025_wt_ymd494__xy1 | 174 | E01 | 13.309 | 4.528 | wrong-candidate selection |
| experiment_03272025_wt_ymd494__xy1 | 185 | AJ | 8.254 | 7.159 | localization error |
| experiment_03272025_wt_ymd494__xy1 | 208 | K | 9.090 | 2.151 | wrong-candidate selection |
| experiment_03272025_wt_ymd494__xy1 | 216 | C101 | 8.223 | 2.828 | wrong-candidate selection |
| experiment_03272025_wt_ymd494__xy1 | 221 | S0 | 101.400 | 3.021 | wrong-candidate selection |
| experiment_03272025_wt_ymd494__xy1 | 236 | S1 | 6.801 | 5.315 | localization error |
| experiment_03272025_wt_ymd494__xy2 | 21 | C0 | 32.882 | 1.904 | wrong-candidate selection |
| experiment_03272025_wt_ymd494__xy2 | 46 | C1 | 11.885 | 2.236 | wrong-candidate selection |
| experiment_03272025_wt_ymd494__xy2 | 237 | V | 13.551 | 1.118 | wrong-candidate selection |
| experiment_03272025_wt_ymd494__xy3 | 58 | B1 | 6.490 | 2.151 | wrong-candidate selection |
| experiment_03272025_wt_ymd494__xy3 | 142 | B10 | 53.106 | 0.354 | wrong-candidate selection |
| experiment_03272025_wt_ymd494__xy3 | 146 | F11 | 6.255 | 1.118 | wrong-candidate selection |
| experiment_03272025_wt_ymd494__xy3 | 147 | E0 | 9.657 | 8.544 | localization error |
| experiment_03272025_wt_ymd494__xy3 | 161 | C01 | 54.231 | 2.121 | wrong-candidate selection |
| experiment_03272025_wt_ymd494__xy3 | 171 | E1 | 5.988 | 3.918 | wrong-candidate selection |
| experiment_03272025_wt_ymd494__xy3 | 212 | AR | 12.334 | 3.953 | wrong-candidate selection |
| experiment_03272025_wt_ymd494__xy3 | 219 | U | 10.470 | 3.808 | wrong-candidate selection |
| experiment_03272025_wt_ymd494__xy3 | 225 | B111 | 6.519 | 2.550 | wrong-candidate selection |
