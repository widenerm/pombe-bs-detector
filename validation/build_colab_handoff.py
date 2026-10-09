"""Build a small, verifiable Colab transfer bundle; never includes raw images."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'validation/runs/scar_v2/colab'
MODEL_SHA = 'e1440429eb384f95afe32bcba6510f90d518eaedc917ede549bed6804004abe2'


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [*ROOT.glob('pombe_tracker/*.py'), *ROOT.glob('validation/scar_*.py'),
             *ROOT.glob('tests/test_scar_*.py'),
             ROOT/'validation/package_for_carmen_2/ground_truth_template_cr_v2.csv',
             ROOT/'validation/reports/scar_v2/cell_anchors.csv',
             ROOT/'validation/reports/scar_v2/AUDIT.md']
    members = {str(p.relative_to(ROOT)): p for p in paths}
    for p in sorted((ROOT/'validation/runs/scar_v2/cache').glob('*/*.npz')):
        members[str(Path('seed_cache') / p.parent.name / p.name)] = p
    manifest = {name: hashlib.sha256(p.read_bytes()).hexdigest() for name, p in members.items()}
    bundle = OUT/'scar_v2_colab_bundle.zip'
    with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, p in sorted(members.items()):
            z.write(p, name)
        z.writestr('MANIFEST.json', json.dumps(manifest, indent=2))
    sha = hashlib.sha256(bundle.read_bytes()).hexdigest()
    cells = []
    def md(text): cells.append(dict(cell_type='markdown', metadata={}, source=text.splitlines(True)))
    def code(text): cells.append(dict(cell_type='code', metadata={}, source=text.splitlines(True), execution_count=None, outputs=[]))
    md('''# S. pombe scar validation — all four v2 videos

Upload this notebook to [Google Colab](https://colab.research.google.com/), choose **Runtime → Change runtime type → GPU**, and run the cells in order. Upload `scar_v2_colab_bundle.zip` when prompted. The bundle contains the exact working code, Carmen annotations, independent cell anchors, and saved masks; it contains no raw H5 files or model weights.

Mount Google Drive and set `H5_DIR` below to the folder containing the four sparse v2 H5 files. You can upload the files if needed. Results and each completed segmentation mask persist in Drive. After interruption, rerun the notebook: verified masks are reused.

This is an unfinished scientific evaluation, not a validated 90% detector. The legacy baseline remains available. Tracking is unchanged and is not run. Positive-only annotations support localization recall and conditional precision; population detection precision cannot be estimated. The experimental comparison was informed by the first video and is a development result, not an independent test.''')
    code('''# Install in the runtime. Evaluation runs in fresh Python processes so preloaded
# Colab NumPy/Torch modules cannot silently use older versions after installation.
import subprocess, sys
PINS = ['cellpose==4.0.9', 'torch==2.10.0', 'torchvision==0.25.0',
        'numpy==2.4.3', 'scipy==1.17.1', 'scikit-image==0.26.0',
        'h5py==3.16.0', 'matplotlib==3.10.8', 'tifffile==2026.3.3',
        'opencv-python-headless==4.13.0.92']
subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', *PINS], check=True)
''')
    code('''from google.colab import drive, files
from pathlib import Path
import os, json, hashlib, shutil, zipfile, subprocess, sys

drive.mount('/content/drive')
WORK = Path('/content/drive/MyDrive/pombe_validation/scar_v2_colab')
# EDIT this path if your existing H5 files are in another Drive folder.
H5_DIR = Path('/content/drive/MyDrive/pombe_validation/carmen_package_2')
WORK.mkdir(parents=True, exist_ok=True)
H5_DIR.mkdir(parents=True, exist_ok=True)
REPO = Path('/content/pombe_scar_audit')
''')
    code(f'''# Upload the companion bundle once; subsequent runs reuse the Drive copy.
BUNDLE_SHA256 = {sha!r}
BUNDLE = WORK/'scar_v2_colab_bundle.zip'
if not BUNDLE.exists() or hashlib.sha256(BUNDLE.read_bytes()).hexdigest() != BUNDLE_SHA256:
    print('Upload scar_v2_colab_bundle.zip')
    uploaded = files.upload()
    payload = uploaded.get('scar_v2_colab_bundle.zip')
    if payload is None or hashlib.sha256(payload).hexdigest() != BUNDLE_SHA256:
        raise ValueError('Wrong bundle. Use the companion ZIP supplied with this notebook.')
    BUNDLE.write_bytes(payload)
    del uploaded, payload
REPO.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(BUNDLE) as archive:
    manifest = json.loads(archive.read('MANIFEST.json'))
    if set(archive.namelist()) != set(manifest) | {{'MANIFEST.json'}}:
        raise ValueError('Unexpected bundle members')
    for name, digest in manifest.items():
        target = (REPO/name).resolve()
        if not target.is_relative_to(REPO.resolve()):
            raise ValueError('Unsafe bundle path')
        payload = archive.read(name)
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError('Corrupt bundle member: ' + name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
os.chdir(REPO)
os.environ['MPLBACKEND'] = 'Agg'
print('Verified source files and', len(list((REPO/'seed_cache').glob('*/*.npz'))), 'saved masks')
''')
    code('''# Confirm all four original packages are present. Do not rename their files.
EXPECTED = {
    'experiment_03202025_wt_ymd494__xy1.h5': 9,
    'experiment_03272025_wt_ymd494__xy1.h5': 13,
    'experiment_03272025_wt_ymd494__xy2.h5': 4,
    'experiment_03272025_wt_ymd494__xy3.h5': 13,
}
missing = [name for name in EXPECTED if not (H5_DIR/name).exists()]
if missing:
    print('Upload these H5 files (Drive upload is preferable for large files):', missing)
    uploaded = files.upload()
    for name, payload in uploaded.items():
        if name in EXPECTED:
            (H5_DIR/name).write_bytes(payload)
    del uploaded
assert all((H5_DIR/name).exists() for name in EXPECTED), 'Four H5 files are required'
# A dedicated folder prevents accidentally including another data package.
assert {p.name for p in H5_DIR.glob('*.h5')} == set(EXPECTED), 'Use a folder containing only the four v2 H5 files'
check = """
import h5py, json, sys
from pathlib import Path
folder=Path(sys.argv[1]); expected=json.loads(sys.argv[2])
for name,n in expected.items():
    with h5py.File(folder/name) as f:
        assert len(f['frames']) == n and len(f['frame_numbers']) == n, name
        assert f['frames'].shape[1:] == (2304,2304), name
print('All four packages verified: 39 full frames')
"""
subprocess.run([sys.executable, '-c', check, str(H5_DIR), json.dumps(EXPECTED)], check=True)
''')
    md('''## GPU and reproducible segmentation

The existing masks used Cellpose 4.0.9, the pinned `cpsam` model, batch size 1, and bfloat16 precision on a Mac GPU. A GPU with native bfloat16 support resumes those masks while recording the device change. Otherwise this notebook builds a separate float32 cache for all 39 frames. Cross-device arithmetic need not be bit-identical; both detector comparisons always share exactly the same saved masks. Model hash, image hash, precision, software release, and segmentation settings are checked before reuse.

The initial model download can take time. If its hash differs, the run stops rather than silently substituting a different model.''')
    code(f'''probe = """
import torch, json
if not torch.cuda.is_available():
    raise RuntimeError('Choose a GPU runtime in Colab before proceeding')
print(json.dumps(dict(name=torch.cuda.get_device_name(0),
    bf16=torch.cuda.is_bf16_supported(including_emulation=False))))
"""
GPU = json.loads(subprocess.check_output([sys.executable, '-c', probe], text=True))
PRECISION = 'bfloat16' if GPU['bf16'] else 'float32'
CACHE = WORK/('cache_' + PRECISION)
CACHE.mkdir(exist_ok=True)
if PRECISION == 'bfloat16':
    for source in (REPO/'seed_cache').glob('*/*.npz'):
        target = CACHE/source.parent.name/source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
print(GPU, 'precision:', PRECISION, 'cached frames:', len(list(CACHE.glob('*/*.npz'))))
MODEL_SHA256 = {MODEL_SHA!r}
# Record the entire installed environment for this execution.
(WORK/'pip_freeze.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], check=True)
BASE = WORK/('baseline_' + PRECISION)
EXPERIMENTAL = WORK/('experimental_' + PRECISION)
ABLATIONS = WORK/('ablations_' + PRECISION)
REPORT = WORK/('report_' + PRECISION)
COMMON = ['--h5-dir', str(H5_DIR), '--gt', str(REPO/'validation/package_for_carmen_2/ground_truth_template_cr_v2.csv'),
          '--cache', str(CACHE), '--anchors', str(REPO/'validation/reports/scar_v2/cell_anchors.csv'),
          '--precision', PRECISION, '--allow-runtime-variation', '--expected-model-sha256', MODEL_SHA256,
          '--gallery']

def run_logged(args, filename):
    with (WORK/filename).open('a') as log:
        process = subprocess.Popen([sys.executable, '-u', *args], cwd=REPO,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            for line in process.stdout:
                print(line, end=''); log.write(line); log.flush()
            if process.wait():
                raise RuntimeError('Run failed; see ' + str(WORK/filename))
        except BaseException:
            process.terminate()
            process.wait()
            raise
''')
    code('''# Baseline first. Only missing masks invoke Cellpose. Completed masks persist.
run_logged(['-m', 'validation.scar_batch', *COMMON, '--out', str(BASE),
            '--overrides', json.dumps({'SCAR_DETECTOR_MODE':'legacy'}),
            '--segment-missing'], 'baseline.log')
print(json.loads((BASE/'metrics.json').read_text())['combined'])
''')
    code('''# Targeted mechanism ablations on frozen baseline cases, not a parameter sweep.
run_logged(['-m', 'validation.scar_ablations', '--baseline', str(BASE), '--out', str(ABLATIONS)], 'ablations.log')
''')
    code('''# Experimental comparison: weak neighbor-pole hint and width as a diagnostic.
# Uses the same masks; never changes or invokes tracking.
run_logged(['-m', 'validation.scar_batch', *COMMON, '--out', str(EXPERIMENTAL),
            '--overrides', json.dumps({'SCAR_DETECTOR_MODE':'experimental'})], 'experimental.log')
run_logged(['-m', 'validation.scar_report', str(BASE), str(EXPERIMENTAL), str(REPORT)], 'report.log')
from IPython.display import display, Markdown
display(Markdown((REPORT/'METRICS.md').read_text()))
''')
    code('''# Show the worst cases inline; complete candidate components are in gallery JSON.
from IPython.display import Image, display
import csv
for folder in [BASE, EXPERIMENTAL]:
    with (folder/'outside_5px.csv').open() as f:
        failures = list(csv.DictReader(f))
    failures.sort(key=lambda r: float(r['error_px']) if r['error_px'] else float('inf'), reverse=True)
    print(folder.name)
    for row in failures[:3]:
        name = f"{row['video_id']}_{row['frame']}_{row['cell_name']}.png"
        picture = folder/'gallery'/name
        if picture.exists(): display(Image(filename=str(picture)))
print('All outputs saved in', WORK)
''')
    code('''# Small review bundle: metrics, per-cell predictions, scores, provenance and logs.
# Full PNG galleries and local case pickles remain in Drive (not in this ZIP).
REVIEW = WORK/('scar_v2_review_' + PRECISION + '.zip')
with zipfile.ZipFile(REVIEW, 'w', zipfile.ZIP_DEFLATED) as archive:
    for folder in [BASE, EXPERIMENTAL, ABLATIONS, REPORT]:
        for path in folder.rglob('*'):
            if path.is_file() and path.suffix in {'.csv', '.json', '.md'}:
                archive.write(path, path.relative_to(WORK))
    archive.write(WORK/'pip_freeze.txt', 'pip_freeze.txt')
    for path in WORK.glob('*.log'): archive.write(path, path.name)
print('Return this review ZIP for the next scientific review:', REVIEW)
files.download(str(REVIEW))
''')
    md('''The numerical report includes localization recall, annotated-cell precision, median error, MAE, p90, fraction within 5 pixels, per-video results, and every cell outside tolerance. `per_cell.csv` preserves missing predictions in the denominator; `predictions_all_cells.csv` records independent detections. The gallery includes contour, PCA axis, poles, curvature/baseline/excess, opposite-side windows, cap exclusion, candidates, and score components.

The next step is to inspect the complete failure gallery, compare ablations, and determine whether any proposed change generalizes beyond the first video. Do not interpret the development-set comparison as independent validation of the 90% target.''')
    notebook = dict(cells=cells, metadata=dict(colab=dict(name='BS_Detector_Validation_Colab.ipynb'), accelerator='GPU',
        kernelspec=dict(display_name='Python 3', language='python', name='python3'), language_info=dict(name='python')),
        nbformat=4, nbformat_minor=5)
    for i, c in enumerate(cells): c['id'] = f'scar-colab-{i:02d}'
    target = ROOT/'notebooks/BS_Detector_Validation_Colab.ipynb'
    target.write_text(json.dumps(notebook, indent=1)+'\n')
    shutil_target = OUT/target.name
    shutil_target.write_bytes(target.read_bytes())
    (OUT/'bundle_manifest.json').write_text(json.dumps(dict(sha256=sha, size_bytes=bundle.stat().st_size,
        saved_masks=len(list((ROOT/'validation/runs/scar_v2/cache').glob('*/*.npz'))), members=manifest), indent=2))
    print(target)
    print(bundle, bundle.stat().st_size, sha)


if __name__ == '__main__':
    build()
