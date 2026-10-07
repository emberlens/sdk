"""Nhan SDK: open a Nhan export (COCO, YOLO or Pascal VOC) and use it for training.

    import nhan
    ds = nhan.load("fruit-stand-coco.zip")
    print(ds.summary())
    unripe = ds.filter(cls="Mango", condition="Unripe")

See sdk/README.md for the full guide.
"""

from .model import Box, Dataset, Image
from .load import load

__version__ = "0.1.0"
__all__ = ["Box", "Dataset", "Image", "load", "__version__"]
