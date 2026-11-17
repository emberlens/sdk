"""PyTorch datasets built from a emberlens.Dataset.

    det = ds.torch_detection("train")      # for torchvision detection models
    image, target = det[0]                 # target: boxes (x0,y0,x1,y1 px), labels (1..N)
    det = DetectionDataset(ds, "train", size=320)  # photo stretched to 320x320, as the app does

    cls = ds.torch_classification("train") # box crops for image classifiers
    image, label_index = cls[0]

`.classes` on both lists the label names (index 0 = "background" for detection).
"""

from __future__ import annotations

from .crops import crop_for_model
from .errors import EmberlensError
from .extras import require_torch
from .model import OTHER

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Detector input side (SPEC.md section 9.2): the app stretches the whole
# photo to DETECTOR_SIZE x DETECTOR_SIZE. Training and scoring do the same.
DETECTOR_SIZE = 320  # fasterrcnn_mobilenet_v3_large_320_fpn


def stretch_box(box, width: int, height: int, size: int):
    """(x0, y0, x1, y1) pixels in a width x height photo -> pixels in the photo stretched to size x size."""
    sx, sy = size / width, size / height
    x0, y0, x1, y1 = box
    return (x0 * sx, y0 * sy, x1 * sx, y1 * sy)


def detection_collate(batch):
    """Use as DataLoader(collate_fn=...) for detection datasets."""
    return tuple(zip(*batch))


class DetectionDataset:
    """size=None keeps each photo as it is. size=N stretches the whole photo
    to N x N (no crop, aspect ratio not kept) and scales the boxes to match."""

    def __init__(self, ds, split: str = "train", transforms=None, classes=None, size=None):
        torch, _ = require_torch()
        self._torch = torch
        self.images = [img for img in ds.images if split is None or img.split == split]
        names = list(classes) if classes else ds.labels
        self.classes = ["background"] + [n for n in names if n != "background"]
        self.index = {n: i for i, n in enumerate(self.classes)}
        self.transforms = transforms
        self.size = size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        from torchvision.transforms.functional import to_tensor

        torch = self._torch
        img = self.images[i]
        photo = img.open().convert("RGB")
        W, H = photo.size
        boxes, labels = [], []
        for b in img.boxes:
            if b.label not in self.index:
                continue
            x0, y0, x1, y1 = b.pixels(W, H)
            if self.size:
                x0, y0, x1, y1 = stretch_box((x0, y0, x1, y1), W, H, self.size)
            if x1 - x0 >= 1 and y1 - y0 >= 1:
                boxes.append([x0, y0, x1, y1])
                labels.append(self.index[b.label])
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": i,
        }
        if self.size:
            photo = photo.resize((self.size, self.size))
        tensor = to_tensor(photo)
        if self.transforms:
            tensor, target = self.transforms(tensor, target)
        return tensor, target


class CropDataset:
    def __init__(self, ds, split: str = "train", size: int = 224, padding: float = 0.1, classes=None, augment=False, with_other=False):
        require_torch()
        self.items = [
            (img, b) for img in ds.images if split is None or img.split == split for b in img.boxes
        ]
        self.classes = list(classes) if classes else ds.labels
        if with_other:
            # YOLOX boxes that match no label: class "other", always last.
            self.classes = [c for c in self.classes if c != OTHER] + [OTHER]
            self.items += ds.other_boxes(split)
        self.index = {n: i for i, n in enumerate(self.classes)}
        self.items = [(img, b) for img, b in self.items if b.label in self.index]
        self.size = size
        self.padding = padding
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        from torchvision import transforms as T

        img, box = self.items[i]
        crop = crop_for_model(img.open().convert("RGB"), box, self.padding, self.size)
        steps = [T.RandomHorizontalFlip()] if self.augment else []
        steps += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
        return T.Compose(steps)(crop), self.index[box.label]


def need_two_labels(classes):
    if len(classes) < 2:
        raise EmberlensError(
            "Training needs at least 2 different labels. Found: "
            + (", ".join(classes) or "none")
            + ". Try another --target."
        )
