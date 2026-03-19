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
    # Two geometric constraints suppress false positives at the poles:
    #   1. WIDTH   Scar must span >= MIN_SCAR_WIDTH_RATIO x max cell width.
    #   2. ANGLE   Scar vector must be ⊥ to the long axis within
    #              MAX_ANGLE_DEVIATION degrees.

    MIN_SCAR_WIDTH_RATIO = 0.80   # 0.0–1.0;  increase to be more strict
    MAX_ANGLE_DEVIATION  = 20.0   # degrees;  try 15–25

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
    ]
