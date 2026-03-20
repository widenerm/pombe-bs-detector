# BS-Detector
### Birth Scar Detector for *Schizosaccharomyces pombe*

Automated detection of birth scars, old/new pole identity, cell lineage tracking, and morphometric measurements in fission yeast time-lapse brightfield microscopy.

<img src="docs/figs/key_example.png" width="300" alt="Detected birth scar with new pole (green), old pole (magenta), and compartment lengths">

*Birth scar detection output. Yellow line: birth scar. Green marker: new pole. Magenta marker: old pole. Dashed lines: new-end and old-end compartment lengths.*

---

## What it does

*S. pombe* cells divide asymmetrically: each daughter inherits an **old pole** (present before the division) and one inherits a **new pole** (the freshly formed division site). The division site leaves behind a **birth scar**: a subtle ridge of elevated curvature on opposite sides of the cell wall.

BS-Detector finds these scars automatically by:

1. **Filtering** segmentation masks by area, aspect ratio, and circularity to reject debris before any tracking occurs
2. **Segmenting** cells with [Cellpose](https://github.com/MouseLand/cellpose)
3. **Computing** a smoothed curvature profile along each cell's contour
4. **Finding** curvature peak pairs that satisfy two geometric constraints:
   - The pair must be on **opposite sides** of the cell
   - The vector connecting them must be **perpendicular** to the long axis (±`MAX_ANGLE_DEVIATION`°)
   - The pair must span **≥ `MIN_SCAR_WIDTH_RATIO`** of the cell's **average mid-cell width** (sampled at seven cross-sections between 20–80% of cell length, which is more stable than the maximum width)
5. **Scoring** candidates by **local prominence** (peak curvature minus the mean curvature in a ±25-index ring window) multiplied by a **perpendicularity bonus**, rewarding sharp localised ridges over broad flat regions and selecting the most geometrically accurate scar location among candidates in the same region
6. **Identifying** new vs. old poles via, in priority order: lineage inheritance at division, birth scar geometry, unambiguous neighbor proximity, or pole morphology as a fallback. Chain cells (both poles touching credible neighbors) are detected explicitly and fall through to scar-based assignment rather than producing a false positive
7. **Tracking** cells across frames with a Hungarian algorithm using center displacement, area change, and a curvature fingerprint. Ghost-track matching resumes lost tracks through bad-segmentation frames rather than minting new lineage names
8. **Assigning lineage names**: `A → A0 / A1 → A00 / A01 / A10 / A11 → …` where `0` always denotes the new-end daughter and `1` the old-end daughter. Daughter pole assignments are corrected immediately at division using the parent's stored pole coordinates
9. **Stabilizing** scar positions across frames via a consensus pass: all valid candidates from every frame are pooled, the most-supported position is found, and frames that deviate are corrected or interpolated retroactively

### Curvature analysis

Signed curvature is computed along a smoothed B-spline contour. Birth scars appear as paired curvature peaks on opposite sides of the cell. The curvature heatmap and profile are available for every cell to aid manual inspection and parameter tuning.

<img src="docs/figs/heatmap.png" width="300" alt="Curvature heatmap overlaid on cell contour">

*Curvature heatmap. Red regions indicate high positive curvature; blue regions indicate low or negative curvature. Yellow stars mark the selected scar endpoints; white star marks the scar midpoint.*

<img src="docs/figs/curvature.png" width="300" alt="Curvature profile plot showing peaks and selected scar pair">

*Curvature profile. Red dots: detected peaks. Green triangles: the selected scar pair. Orange dotted lines: segmentation quality threshold.*

### Segmentation quality control

Cellpose occasionally produces artifact segmentations — most commonly a septum fragment (one half of a dividing cell) or a cell whose mask clips the image boundary. Both produce pathological curvature spikes well above the range of a healthy contour. BS-Detector flags these automatically with an orange overlay so they can be reviewed or excluded without disrupting the rest of the analysis.

<img src="docs/figs/bad_seg.png" width="300" alt="Example of flagged bad segmentations with orange outlines">

*Orange outlines indicate cells flagged by the segmentation quality check. Green outlines are clean detections.*

---

## Output

For each frame and each cell, BS-Detector produces:

| Column | Description |
|---|---|
| `cell_name` | Lineage-encoded name (`A`, `A0`, `A01` …) |
| `frame` | Frame index |
| `length` | Pole-to-pole distance [px] |
| `width_centroid` | Width perpendicular to long axis at cell center [px] |
| `width_scar` | Distance between the two scar endpoints [px] |
| `new_end_length` | Scar midpoint → new pole [px] |
| `old_end_length` | Scar midpoint → old pole [px] |
| `area` | Cell area [px²] |
| `scar_detected` | True / False |
| `scar_source` | `raw`, `corrected`, or `interpolated` — how the final scar position was determined |
| `scar_stable` | False if more than `SCAR_INSTABILITY_FRACTION` of the cell's frames were corrected or interpolated |
| `seg_quality` | `ok`, `border_clip`, or `septum_fragment` |
| `pole_method` | How poles were assigned (`lineage`, `scar_based`, `neighbor_proximity`, `neighbor_poles`, `morphology`) |
| `pole_confidence` | Confidence level of pole assignment |

Cells with `seg_quality = border_clip` are excluded from the CSV entirely (incompletely segmented). All other cells are included regardless of quality flag.

---

## Sample dataset

A 10-frame sample dataset is bundled at `data/sample/sample.h5` for testing and demonstration without needing your own data. The Colab notebook uses it by default — no configuration required for a first run.

---

## Installation & usage (Google Colab)

Open [`notebooks/BS_Detector_Colab.ipynb`](notebooks/BS_Detector_Colab.ipynb) in Colab.

**Step 1** — the first two cells install dependencies and clone this repo automatically.  
**Step 2** — the configuration cell defaults to the bundled sample dataset. Change `cfg.H5_FILE_PATH` to your own HDF5 file on Google Drive when you have real data. That is the only cell most users ever need to edit.  
**Step 3** — run the remaining cells top-to-bottom.

Results are saved to `cfg.OUTPUT_DIR` (a Google Drive folder for real data, or `/content/pombe_results` for the sample run).

### Local installation

```bash
git clone https://github.com/widenerm/pombe-bs-detector.git
cd pombe-bs-detector
pip install -r requirements.txt
```

```python
from pombe_tracker.config        import Config
from pombe_tracker.pipeline      import run_pipeline
from pombe_tracker.io_utils      import load_h5_data, export_csv
from pombe_tracker.tracking      import CellTracker
from pombe_tracker.postprocessing import stabilize_scars

cfg              = Config()
cfg.H5_FILE_PATH = 'my_experiment.h5'
cfg.NUM_FRAMES   = 10

frames              = load_h5_data(cfg.H5_FILE_PATH, cfg.H5_DATASET_KEY)
tracker             = CellTracker(cfg)
results             = run_pipeline(frames, cfg, tracker=tracker)
results, report     = stabilize_scars(results, cfg)
export_csv(results, 'measurements.csv')
```

---

## Repository structure

```
pombe-bs-detector/
├── data/
│   └── sample/
│       ├── sample.h5          ← bundled 10-frame test dataset
│       └── README.md          ← dataset description and format notes
├── docs/
│   └── figs/                  example figures
├── notebooks/
│   └── BS_Detector_Colab.ipynb   ← start here
├── pombe_tracker/
│   ├── config.py          central configuration (all tunable parameters)
│   ├── geometry.py        curvature, PCA, width / length measurements
│   ├── detection.py       BirthScarDetector — prominence + perpendicularity scoring
│   ├── poles.py           new/old pole assignment; chain-cell detection
│   ├── segmentation.py    Cellpose wrapper
│   ├── tracking.py        Hungarian tracker + lineage naming + ghost-track matching
│   ├── pipeline.py        CellProcessor, lineage pole correction, run_pipeline
│   ├── visualization.py   all plotting functions
│   ├── postprocessing.py  consensus-based temporal scar stabilization
│   └── io_utils.py        HDF5 loading, CSV export
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Key parameters

| Parameter | Default | Effect |
|---|---|---|
| `SMOOTH_FACTOR` | 40.0 | B-spline smoothing; increase to reduce noise |
| `MIN_SCAR_WIDTH_RATIO` | 0.80 | Minimum scar width as a fraction of average mid-cell width; increase to be stricter |
| `MAX_ANGLE_DEVIATION` | 20.0° | Max deviation from perpendicular; decrease to be stricter |
| `ASPECT_RATIO_MIN` | 1.5 | Minimum major/minor axis ratio; increase to reject shorter debris |
| `MAX_CIRCULARITY` | 0.85 | Maximum 4π·area/perimeter²; decrease to be stricter about rod shape |
| `POLE_PROXIMITY_THRESHOLD` | 100 px | Max tip-to-tip distance to call two cells neighbors |
| `NEIGHBOR_HIGH_CONFIDENCE_DIST` | 75 px | Distance below which a neighbor match is considered high-confidence |
| `MAX_TRACKING_DISTANCE` | 80 px | Max center displacement between frames |
| `GHOST_FRAMES` | 3 | Frames a lost track is kept alive before being discarded |
| `CURVATURE_QUALITY_THRESHOLD` | 0.10 | Max curvature before a cell is flagged as a segmentation artifact |
| `SCAR_STABILITY_THRESHOLD` | 0.12 | Max scar position shift (normalized) before a frame is corrected |
| `SCAR_INSTABILITY_FRACTION` | 0.30 | Fraction of corrected/interpolated frames before `scar_stable = False` |

All parameters are documented in [`pombe_tracker/config.py`](pombe_tracker/config.py).

---

## Citation

If you use BS-Detector in your research, please cite:

> [Your Name et al., *Journal*, Year. BS-Detector: Automated birth scar detection and lineage tracking in *S. pombe*.]

---

## License

MIT
