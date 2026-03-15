"""
segmentation.py  –  Thin wrapper around Cellpose.
"""
from cellpose import models, core


class CellposeSegmenter:

    def __init__(self, config):
        self.cfg     = config
        use_gpu      = core.use_gpu()
        print(f"Cellpose GPU: {use_gpu}")
        self.model   = models.CellposeModel(gpu=use_gpu)

    def segment(self, image):
        """Return a label mask for *image* (2-D numpy array)."""
        if image.ndim > 2:
            image = image.squeeze()
        masks, _, _ = self.model.eval(
            image,
            diameter=self.cfg.CELLPOSE_DIAMETER,
            channels=[0, 0],
        )
        return masks
