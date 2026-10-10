"""PyTorch datasets built from a emberlens.Dataset.

    det = ds.torch_detection("train")      # for torchvision detection models
    image, target = det[0]                 # target: boxes (x0,y0,x1,y1 px), labels (1..N)

    cls = ds.torch_classification("train") # box crops for image classifiers
    image, label_index = cls[0]

`.classes` on both lists the label names (index 0 = "background" for detection).
"""

from __future__ import annotations

from .crops import crop_box
from .errors import EmberlensError
from .extras import require_torch

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def detection_collate(batch):
    """Use as DataLoader(collate_fn=...) for detection datasets."""
    return tuple(zip(*batch))


class DetectionDataset:
    def __init__(self, ds, split: str = "train", transforms=None, classes=None):
        torch, _ = require_torch()
        self._torch = torch
        self.images = [img for img in ds.images if split is None or img.split == split]
        names = list(classes) if classes else ds.labels
        self.classes = ["background"] + [n for n in names if n != "background"]
        self.index = {n: i for i, n in enumerate(self.classes)}
        self.transforms = transforms

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
            if x1 - x0 >= 1 and y1 - y0 >= 1:
                boxes.append([x0, y0, x1, y1])
                labels.append(self.index[b.label])
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": i,
        }
        tensor = to_tensor(photo)
        if self.transforms:
            tensor, target = self.transforms(tensor, target)
        return tensor, target


class CropDataset:
    def __init__(self, ds, split: str = "train", size: int = 224, padding: float = 0.1, classes=None, augment=False):
        require_torch()
        self.items = [
            (img, b) for img in ds.images if split is None or img.split == split for b in img.boxes
        ]
        self.classes = list(classes) if classes else ds.labels
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
        crop = crop_box(img.open().convert("RGB"), box, self.padding).resize((self.size, self.size))
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
