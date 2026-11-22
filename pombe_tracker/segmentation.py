"""
segmentation.py  –  Cellpose-SAM segmentation, saved masks, and their provenance.

Every accuracy number in validation/ was measured on masks from Cellpose 4.0.9
with the `cpsam` weights below, in float32, with Cellpose choosing the diameter.
CellposeSegmenter uses exactly that setup and refuses other weights unless told
otherwise, so a run on new data segments the same way as the validation.

Whole movies are segmented once and saved, one file per frame:

    <masks_dir>/<frame:04d>.npz   labels (uint16), image_sha256, provenance (JSON)

Segmentation is the slow step (seconds per frame on a CUDA GPU, about 90 s on an
Apple M2, much longer on a CPU), so `segment_movie` skips frames already saved
for the same image and can be run in blocks (`start`/`stop`) on several GPUs or
resumed after an interruption. `SavedMasks` hands the saved masks to
`run_pipeline` in place of a live segmenter and checks each one belongs to its
frame.
"""
import hashlib
import json
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np

# sha256 of the Cellpose-SAM weights used for every validated result (Cellpose 4.0.9 'cpsam').
PINNED_MODEL_SHA256 = 'e1440429eb384f95afe32bcba6510f90d518eaedc917ede549bed6804004abe2'


def image_sha256(image):
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def _device():
    import torch
    if torch.cuda.is_available():
        return torch.device('cuda'), torch.cuda.get_device_name(0)
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return torch.device('mps'), 'Apple GPU (MPS)'
    return torch.device('cpu'), None


class CellposeSegmenter:
    """Cellpose-SAM with the validated settings: pinned weights, float32, automatic diameter.

    allow_other_weights : run even if the weights differ from PINNED_MODEL_SHA256
                          (results are then not covered by the validation)
    batch_size          : tiles per GPU batch; changes speed and memory, not the masks
    """

    def __init__(self, config=None, batch_size=8, allow_other_weights=False):
        import torch
        from cellpose import models
        self.cfg = config
        self.batch_size = batch_size
        device, gpu_name = _device()
        self.model = models.CellposeModel(gpu=device.type != 'cpu', device=device, use_bfloat16=False)
        weights = Path(self.model.pretrained_model)
        sha = hashlib.sha256(weights.read_bytes()).hexdigest()
        if sha != PINNED_MODEL_SHA256 and not allow_other_weights:
            raise RuntimeError(
                f'Cellpose weights {weights} (sha256 {sha[:12]}...) are not the validated cpsam weights '
                f'({PINNED_MODEL_SHA256[:12]}...). Install cellpose==4.0.9, or pass allow_other_weights=True '
                f'(bsd segment --allow-other-weights) to segment anyway.')
        self.diameter = getattr(config, 'CELLPOSE_DIAMETER', None) if config is not None else None
        self.provenance = dict(
            cellpose=version('cellpose'), torch=version('torch'), model=weights.name, model_sha256=sha,
            diameter=self.diameter, channels=[0, 0], batch_size=batch_size, device=str(self.model.device),
            dtype='float32', gpu_name=gpu_name)
        print(f'Cellpose {self.provenance["cellpose"]} on {self.provenance["device"]}'
              + (f' ({gpu_name})' if gpu_name else ' (no GPU: expect several minutes per frame)'))

    def segment(self, image):
        """Return a label mask for *image* (2-D numpy array)."""
        if image.ndim > 2:
            image = image.squeeze()
        masks = self.model.eval(image, diameter=self.diameter, channels=[0, 0], batch_size=self.batch_size)[0]
        return masks


def mask_path(masks_dir, i):
    return Path(masks_dir) / f'{i:04d}.npz'


def saved_mask_matches(path, digest):
    if not path.exists():
        return False
    try:
        with np.load(path) as z:
            return str(z['image_sha256']) == digest
    except Exception:          # a truncated file from an interrupted write
        return False


def segment_movie(frames, masks_dir, start=0, stop=None, segmenter=None, log=print, **segmenter_kw):
    """Segment frames[start:stop] into masks_dir, skipping frames already saved for the same image.

    frames : array-like (T, Y, X), e.g. an open h5py dataset
    Returns the number of frames segmented in this call.
    """
    masks_dir = Path(masks_dir)
    masks_dir.mkdir(parents=True, exist_ok=True)
    stop = len(frames) if stop is None else min(stop, len(frames))
    todo = []
    for i in range(start, stop):
        image = np.asarray(frames[i])
        if not saved_mask_matches(mask_path(masks_dir, i), image_sha256(image)):
            todo.append(i)
    log(f'frames {start}-{stop - 1}: {stop - start - len(todo)} already segmented, {len(todo)} to do')
    if not todo:
        return 0
    segmenter = segmenter or CellposeSegmenter(**segmenter_kw)
    prov = json.dumps(segmenter.provenance, sort_keys=True)
    t_all = time.time()
    for n, i in enumerate(todo, 1):
        image = np.asarray(frames[i])
        t0 = time.time()
        labels = segmenter.segment(image)
        if labels.max() >= 2 ** 16:
            raise ValueError(f'frame {i}: {labels.max()} objects do not fit in uint16')
        out = mask_path(masks_dir, i)
        tmp = out.with_suffix('.partial.npz')
        np.savez_compressed(tmp, labels=labels.astype(np.uint16), image_sha256=image_sha256(image), provenance=prov)
        tmp.replace(out)
        left = (time.time() - t_all) / n * (len(todo) - n)
        log(f'frame {i}: {int(labels.max())} objects, {time.time() - t0:.1f} s  [{n}/{len(todo)}, ~{left / 60:.0f} min left]')
    return len(todo)


def missing_masks(frames, masks_dir):
    """Frame indices with no saved mask (existence only; SavedMasks checks the images)."""
    return [i for i in range(len(frames)) if not mask_path(masks_dir, i).exists()]


class SavedMasks:
    """Stands in for CellposeSegmenter in run_pipeline: returns the saved mask for each frame, in order."""

    def __init__(self, masks_dir, n_frames):
        self.paths = [mask_path(masks_dir, i) for i in range(n_frames)]
        missing = [p.name for p in self.paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f'{len(missing)} of {n_frames} frames have no saved mask in {masks_dir} '
                                    f'(first: {missing[0]}); run `bsd segment` first')
        self.i = 0
        self.provenance = set()

    def segment(self, image):
        with np.load(self.paths[self.i]) as z:
            if str(z['image_sha256']) != image_sha256(image):
                raise ValueError(f'mask {self.paths[self.i]} was made from a different image')
            labels = z['labels']
            self.provenance.add(str(z['provenance']))
        self.i += 1
        return labels
