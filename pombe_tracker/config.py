class Config:
    """
    ══════════════════════════════════════════════════════════════════════
    POMBE BIRTH SCAR TRACKER  –  Central Configuration
    ══════════════════════════════════════════════════════════════════════
    Edit the parameters in each section to tune the analysis for your
    experiment. Each parameter has a plain-English description.
    """

    # ── DATA ──────────────────────────────────────────────────────────────
    # Path to your HDF5 file (must be accessible from Colab, e.g. via Drive)
    H5_FILE_PATH   = '/content/drive/MyDrive/your_experiment.h5'
    H5_DATASET_KEY = 'frames'        # Key inside the H5 file that holds the image stack

    # How many frames to analyse. Set to an integer (e.g. 10) to limit,
    # or None to process every frame in the file.
    NUM_FRAMES = None

    # Where to save results (CSV, figures). Created automatically if absent.
    OUTPUT_DIR = '/content/drive/MyDrive/pombe_results'

    # ── SEGMENTATION ──────────────────────────────────────────────────────
    # Minimum cell area in pixels. Anything smaller is treated as debris.
    MIN_CELL_AREA = 150

    # Expected cell diameter for Cellpose. None = auto-detect (usually fine).
    CELLPOSE_DIAMETER = None

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
    # We look for two curvature peaks on OPPOSITE SIDES of the cell that
    # are connected by a vector PERPENDICULAR to the cell's long axis.
    # This geometric constraint eliminates the poles naturally:
    #   • At a pole the cell is NARROW → fails the width test
    #   • At a pole the opposite-side curvature peak would require a vector
    #     PARALLEL (not perpendicular) to the axis → fails the angle test
    # Therefore we no longer exclude any fraction of the cell from search.

    # A scar must span at least this fraction of the cell's maximum width.
    # This is the primary guard against false positives at the poles.
    MIN_SCAR_WIDTH_RATIO = 0.80   # 0.0–1.0;  increase to be more strict

    # Maximum deviation from perfectly perpendicular (90°) that is still
    # accepted as a valid scar. Smaller = stricter.
    MAX_ANGLE_DEVIATION  = 20.0   # degrees;  try 15–25

    # ── NEIGHBOR / POLE DETECTION ─────────────────────────────────────────
    # After division, the two daughters touch tip-to-tip. We use this to
    # identify the new (division-site) pole.

    # Maximum pole-to-pole distance (pixels) to even consider two cells
    # as touching neighbours.
    POLE_PROXIMITY_THRESHOLD      = 100.0

    # If the touching distance is below this, confidence is 'high'.
    NEIGHBOR_HIGH_CONFIDENCE_DIST =  75.0

    # ── TRACKING (HUNGARIAN ALGORITHM) ────────────────────────────────────
    # Maximum distance (pixels) a cell centre can move between frames
    # and still be considered the same cell.
    MAX_TRACKING_DISTANCE = 80.0

    # Relative weights for the three cost components used for matching.
    # All are normalised so 1.0 means "at the maximum tolerated value".
    COST_WEIGHT_DISTANCE  = 1.0   # centre displacement
    COST_WEIGHT_AREA      = 0.5   # change in cell area
    COST_WEIGHT_CURVATURE = 0.3   # curvature fingerprint dissimilarity

    # When a tracked cell disappears and two smaller cells appear nearby,
    # we call it a division. Each daughter must be at least this fraction
    # of the parent's area to be considered a valid daughter.
    DIVISION_AREA_RATIO = 0.35

    # ── VISUALIZATIONS ────────────────────────────────────────────────────
    # Set any of these to False to skip that visualisation.

    SHOW_CELL_OVERVIEW      = True   # Frame overview: all cells outlined
    SHOW_INDIVIDUAL_CELLS   = True   # Per-cell: scar line, poles, measurements
    SHOW_CURVATURE_HEATMAPS = True   # Curvature colour-mapped onto contour
    SHOW_CURVATURE_PROFILES = True   # Curvature vs. contour-index plots
    SHOW_LINEAGE_TREE       = True   # Gantt-style lineage diagram

    # Save figures to OUTPUT_DIR as PNG files (in addition to displaying).
    SAVE_FIGURES = False

    # ── CSV EXPORT ────────────────────────────────────────────────────────
    EXPORT_CSV = True

    # Which columns to include in the CSV. Remove any you don't need.
    CSV_COLUMNS = [
        'cell_name',        # Lineage-encoded name  (e.g. A, A0, A01 …)
        'frame',            # Frame index (0-based)
        'length',           # Pole-to-pole distance  [pixels]
        'width_centroid',   # Cell width ⊥ long axis at centre  [pixels]
        'width_scar',       # Distance between the two scar endpoints  [pixels]
        'new_end_length',   # Scar midpoint → new pole  [pixels]
        'old_end_length',   # Scar midpoint → old pole  [pixels]
        'area',             # Cell area  [pixels²]
        'scar_detected',    # True / False
        'pole_method',      # How the poles were assigned
        'pole_confidence',  # Confidence level of pole assignment
    ]
