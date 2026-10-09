class Config:
    """
    ══════════════════════════════════════════════════════════════════════
    POMBE BIRTH SCAR TRACKER  –  Central Configuration
    ══════════════════════════════════════════════════════════════════════
    Edit the parameters in each section to tune the analysis for your
    experiment. Each parameter has a plain-English description.
    """

    # ── DATA ──────────────────────────────────────────────────────────────
    H5_FILE_PATH   = '/content/drive/MyDrive/your_experiment.h5'
    H5_DATASET_KEY = 'frames'
    NUM_FRAMES     = None
    OUTPUT_DIR     = '/content/drive/MyDrive/pombe_results'

    # ── SEGMENTATION ──────────────────────────────────────────────────────
    # Minimum cell area in pixels. Anything smaller is treated as debris.
    MIN_CELL_AREA = 150

    # Expected cell diameter for Cellpose. None = auto-detect (usually fine).
    CELLPOSE_DIAMETER = None

    # ── SHAPE FILTER (debris / noise rejection) ───────────────────────────
    # These two filters run before tracking so noise blobs never receive a
    # lineage name or appear in the CSV.
    #
    # Aspect ratio = major_axis_length / minor_axis_length (from regionprops).
    # S. pombe rods typically have AR > 2.0; round debris will be < 1.5.
    ASPECT_RATIO_MIN = 1.5

    # Circularity = 4π × area / perimeter².  A perfect circle = 1.0; a rod
    # ≈ 0.4–0.7.  Blobs above this threshold are rejected as non-rod-shaped.
    # Default 0.85 is conservative — lower (e.g. 0.75) if round contaminants
    # are sneaking through.
    MAX_CIRCULARITY = 0.85

    # ── CONTOUR & CURVATURE ───────────────────────────────────────────────
    # B-spline smoothing factor. Higher → smoother contour, fewer spurious
    # peaks. Lower → noisier but more sensitive to subtle features.
    SMOOTH_FACTOR = 40.0

    # Number of points to resample each contour to before computing
    # curvature. Higher → smoother profile, slower.
    N_CONTOUR_POINTS = 300

    # ── BIRTH SCAR DETECTION ─────────────────────────────────────────────
    # HOW SCAR DETECTION WORKS
    # ─────────────────────────
    # The full cell contour is always searched.  All valid candidates are
    # stored per frame and used by the temporal stabilization pass to enforce
    # a cross-frame consensus without re-running the detector.
    #
    # Windowed curvature is evaluated at normalized longitudinal positions
    # rather than requiring two equally strong curvature peaks.  The first
    # and last fractions of the PCA axis are cap regions; excluding them is
    # important because ordinary pole rounding is often the strongest
    # curvature on an otherwise scar-free cell.

    # 'profile' (default): interpretable axial-profile localizer, see
    #   pombe_tracker/scar_profile.py. Scores every 1-px cross-section from
    #   each pole with the named terms below; needs the frame image for the
    #   membrane terms (falls back to mask width without it).
    # 'legacy' / 'experimental': previous windowed-curvature detector.
    SCAR_DETECTOR_MODE = 'profile'

    # Profile mode. Lengths are in units of the mid-cell width W so they
    # transfer between strains; re-fit with validation/scar_fit.py for
    # mutants with different morphology.
    # Expected scar distance from the new pole: d/W ~ Normal(mu, SD) with
    # mu = OFFSET + SLOPE * (cell length / W). Fitted on 80 WT cells.
    SCAR_PRIOR_OFFSET = -0.740
    SCAR_PRIOR_SLOPE  = 0.374
    SCAR_PRIOR_SD     = 0.194
    SCAR_PRIOR_WEIGHT = 1.0
    SCAR_MIN_DISTANCE_W = 0.18     # search window for the scar, in W
    SCAR_MAX_DISTANCE_W = 1.15
    SCAR_HINT_WEIGHT = 2.01        # bonus for the neighbor-derived new pole end
    SCAR_MEMBRANE_WIDTH_WEIGHT = 44.19   # scar ridge = widest point of the end
    SCAR_MEMBRANE_RIDGE_WEIGHT = 18.30   # concave corner/ridge on both flanks
    SCAR_PROFILE_SMOOTHING = 0.15  # spline residual budget per contour point
    SCAR_MEMBRANE_SEARCH_PX = 6.0  # membrane search range along each normal
    SCAR_MEMBRANE_SMOOTHNESS = 0.3 # penalty on jumps of the traced membrane
    SCAR_MEMBRANE_CLIP_PX = 3.0
    SCAR_WIDTH_SMOOTH_PX = 2.0
    SCAR_RIDGE_SCALE_PX = 6.0
    # Bilateral mode ('bilateral'): windowed curvature of the image-traced
    # membrane on both flanks. Lengths in W unless named _PX.
    SCAR_BILATERAL_COMBINE = 'min'      # 'min' both flanks bulge | 'sum' lopsided OK | 'max'
    SCAR_BILATERAL_WINDOW_W = 0.15      # axial averaging window per flank
    SCAR_BILATERAL_SCALE_PX = 4.0       # curvature smoothing scale
    SCAR_BILATERAL_BASELINE_PX = 10.0   # smooth cell-shape curvature removed
    SCAR_BILATERAL_SEARCH_PX = 5.0      # membrane search range along each normal
    SCAR_BILATERAL_MIN_DISTANCE_W = 0.1
    SCAR_BILATERAL_PRIOR_OFFSET = -0.155
    SCAR_BILATERAL_PRIOR_SLOPE = 0.254
    SCAR_BILATERAL_PRIOR_SD = 0.11
    SCAR_BILATERAL_PRIOR_WEIGHT = 1.0
    SCAR_BILATERAL_BLUNTNESS_WEIGHT = 0.25  # no hint: favor the blunter (newer) end
    SCAR_BILATERAL_SEPARATION_W = 0.1       # confidence margin: alternatives this far away
    SCAR_BILATERAL_CONFIDENCE_FLAG = 1.5    # below this, scar_needs_review = True

    # Confidence = probability mass within RADIUS px of the chosen point.
    # Cells below FLAG get scar_needs_review = True for manual checking.
    SCAR_CONFIDENCE_RADIUS = 5.0
    SCAR_CONFIDENCE_FLAG = 0.5
    SCAR_HINT_STRENGTH = 0.2       # experimental: at most 20% multiplicative bonus
    SCAR_N_WINDOWS = None        # None retains legacy max(12, n_points // 12)
    SCAR_BASELINE_SAMPLES = 25   # radius; scale with sampling density in ablations

    MIN_SCAR_WIDTH_RATIO = 0.80   # legacy hard gate; windowed scoring uses soft width
    MAX_ANGLE_DEVIATION  = 30.0   # soft score penalty; try 20–35
    SCAR_CURVATURE_WINDOW = 0.08  # longitudinal window, as fraction of cell length
    SCAR_CAP_EXCLUSION = 0.12     # never score windows centred in these cap regions
    SCAR_MAX_LONGITUDINAL_OFFSET = 0.08  # endpoint offset allowed within a window
    # Prefer a recent scar near (but not inside) a pole without forcing it to
    # one exact location. This soft prior is used without a pole hint.
    SCAR_RECENCY_TARGET = 0.22    # distance from nearest pole, normalized [0, 0.5]
    SCAR_RECENCY_SPREAD = 0.16    # softness of the near-pole preference
    SCAR_WIDTH_SOFT_FLOOR = 0.35  # width ratio at which the soft score is 0.5

    # ── NEIGHBOR / POLE DETECTION ─────────────────────────────────────────
    POLE_PROXIMITY_THRESHOLD      = 100.0
    NEIGHBOR_HIGH_CONFIDENCE_DIST =  75.0

    # ── TRACKING (HUNGARIAN ALGORITHM) ────────────────────────────────────
    MAX_TRACKING_DISTANCE = 80.0
    COST_WEIGHT_DISTANCE  = 1.0
    COST_WEIGHT_AREA      = 0.5
    COST_WEIGHT_CURVATURE = 0.3
    DIVISION_AREA_RATIO   = 0.35

    # ── GHOST TRACK MATCHING ──────────────────────────────────────────────
    # When Cellpose produces a bad segmentation for a frame or two, the
    # original cell track is temporarily lost.  Before minting a new base
    # name for an unmatched current cell, BS-Detector checks whether a
    # recently lost track is compatible (by centroid, area, and curvature).
    # If a match is found, the track is resumed under its original name.

    # How many frames a lost track is kept in the ghost buffer.
    GHOST_FRAMES = 3

    # Maximum curvature fingerprint L2 distance for a ghost match to be
    # accepted.  Increase if cells change shape significantly between frames.
    GHOST_FINGERPRINT_THRESHOLD = 1.0

    # ── SEGMENTATION QUALITY ──────────────────────────────────────────────
    # seg_quality values:
    #   'ok'               – passes all checks
    #   'border_clip'      – contour touches image boundary; EXCLUDED from CSV
    #   'septum_fragment'  – high-curvature spike indicating a half-cell;
    #                        INCLUDED in CSV with flag, measurements interpolated
    CURVATURE_QUALITY_THRESHOLD = 0.10

    # ── SCAR TEMPORAL STABILITY ───────────────────────────────────────────
    # After the pipeline runs, a consensus-based post-processing step:
    #   1. Collects ALL valid scar candidates across all frames for each cell.
    #   2. Finds the canonical scar position (the one with the most support).
    #   3. Retroactively corrects frames where a different candidate was
    #      selected (scar_source = 'corrected').
    #   4. Interpolates frames where no matching candidate exists
    #      (scar_source = 'interpolated').

    SCAR_STABILITY_WINDOW    = 3
    SCAR_STABILITY_THRESHOLD = 0.12
    SCAR_INTERPOLATE         = True

    # Fraction of a cell's frames that can be corrected or interpolated before
    # the entire cell is flagged scar_stable = False in the CSV.
    SCAR_INSTABILITY_FRACTION = 0.30

    # ── VISUALIZATIONS ────────────────────────────────────────────────────
    SHOW_CELL_OVERVIEW      = True
    SHOW_INDIVIDUAL_CELLS   = True
    SHOW_CURVATURE_HEATMAPS = True
    SHOW_CURVATURE_PROFILES = True
    SHOW_LINEAGE_TREE       = True
    SAVE_FIGURES            = False

    # ── CSV EXPORT ────────────────────────────────────────────────────────
    EXPORT_CSV = True

    CSV_COLUMNS = [
        'cell_name',        # Lineage-encoded name  (e.g. A, A0, A01 …)
        'frame',            # Frame index (0-based)
        'length',           # Pole-to-pole distance  [pixels]
        'width_centroid',   # Cell width ⊥ long axis at center  [pixels]
        'width_scar',       # Distance between the two scar endpoints  [pixels]
        'new_end_length',   # Scar midpoint → new pole  [pixels]
        'old_end_length',   # Scar midpoint → old pole  [pixels]
        'area',             # Cell area  [pixels²]
        'scar_detected',    # True / False
        'scar_source',      # raw | corrected | interpolated | no_detection
        'scar_stable',      # True if scar position is consistent across frames
        'seg_quality',      # ok | border_clip | septum_fragment
        'pole_method',      # How the poles were assigned
        'pole_confidence',  # Confidence level of pole assignment
        'scar_confidence',  # profile mode: probability mass near the chosen scar
        'scar_needs_review',# profile mode: low confidence or scar/pole conflict
        'scar_pole_conflict', # scar nearer the assigned old pole than the new pole
    ]
