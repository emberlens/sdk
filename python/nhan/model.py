"""The data model: a Dataset of Images, each with Boxes.

Coordinates are always normalised to 0..1 with the origin at the top-left,
whatever format the export used. Use Box.pixels() for pixel values.
"""

from __future__ import annotations

import copy
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

SPLITS = ("train", "val", "test")


def _n(count: int, word: str) -> str:
    plural = word + ("es" if word.endswith("x") else "s")
    return f"{count} {word if count == 1 else plural}"


def _key(text: str) -> str:
    return text.strip().lower()


def box_from_polygon(label: str, cls: str, points, attributes=None) -> "Box":
    """A polygon Box; x, y, w, h is the box around the points."""
    pts = [(float(px), float(py)) for px, py in points]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return Box(
        label=label,
        cls=cls,
        x=min(xs),
        y=min(ys),
        w=max(xs) - min(xs),
        h=max(ys) - min(ys),
        attributes=dict(attributes or {}),
        polygon=pts,
    )


@dataclass
class Box:
    """One labelled box.

    label       the name written in the export, e.g. "Mango_Ripe_Alphonso"
    cls         the main class, e.g. "Mango"
    attributes  sub-category values, e.g. {"Condition": "Ripe"}; a missing
                key means "no value"
    x, y, w, h  normalised 0..1, top-left origin
    polygon     corners [(x, y), ...] normalised, or None for a plain box;
                x, y, w, h is then the box around the polygon
    """

    label: str
    cls: str
    x: float
    y: float
    w: float
    h: float
    attributes: Dict[str, str] = field(default_factory=dict)
    polygon: Optional[List[Tuple[float, float]]] = None

    @property
    def is_polygon(self) -> bool:
        return self.polygon is not None

    def polygon_pixels(self, width: int, height: int) -> List[Tuple[float, float]]:
        """Polygon corners in pixels (a box gives its 4 corners)."""
        return [(px * width, py * height) for px, py in self.corners()]

    def corners(self) -> List[Tuple[float, float]]:
        """The polygon, or the 4 corners of the box (clockwise from top-left)."""
        if self.polygon is not None:
            return list(self.polygon)
        return [(self.x, self.y), (self.x + self.w, self.y), (self.x + self.w, self.y + self.h), (self.x, self.y + self.h)]

    def get(self, name: str) -> Optional[str]:
        """The value of sub-category `name` (any letter case), or None."""
        for k, v in self.attributes.items():
            if _key(k) == _key(name):
                return v
        return None

    def pixels(self, width: int, height: int) -> tuple:
        """(x_min, y_min, x_max, y_max) in pixels."""
        return (
            self.x * width,
            self.y * height,
            (self.x + self.w) * width,
            (self.y + self.h) * height,
        )

    def matches(self, cls: Optional[str] = None, **attributes) -> bool:
        """True when the box has this class and these values.

        Letter case is ignored. A value of None means "has no value".
        """
        if cls is not None and _key(self.cls) != _key(cls):
            return False
        for name, want in attributes.items():
            # Python names cannot hold spaces: skin_colour finds "Skin colour".
            have = self.get(name)
            if have is None:
                have = self.get(name.replace("_", " "))
            if want is None:
                if have is not None:
                    return False
            elif have is None or _key(have) != _key(str(want)):
                return False
        return True


@dataclass
class Image:
    """One photo and its boxes. `path` is the image file on disk."""

    id: str
    path: Path
    width: int
    height: int
    split: str
    boxes: List[Box] = field(default_factory=list)

    def open(self):
        """The photo as a Pillow image, upright (needs Pillow)."""
        from .extras import require_pillow

        PIL_Image = require_pillow()
        from PIL import ImageOps

        img = PIL_Image.open(self.path)
        img.load()
        # Nhan exports are stored upright. Photos from other tools may carry
        # an EXIF Orientation tag; turn them upright like a viewer would.
        return ImageOps.exif_transpose(img)


@dataclass
class Dataset:
    """A whole export. Iterate it to get Images.

    name            project name (from nhan.json) or folder name
    format          "coco", "yolo" or "voc"
    root            folder with the export files
    classes         main classes, in project order
    sub_categories  {"Condition": ["Ripe", "Unripe"], ...}
    card            the parsed nhan.json, or None for exports without one
    """

    name: str
    format: str
    root: Path
    images: List[Image]
    classes: List[str] = field(default_factory=list)
    sub_categories: Dict[str, List[str]] = field(default_factory=dict)
    card: Optional[dict] = None
    label_order: List[str] = field(default_factory=list)

    # ---- basics -------------------------------------------------------

    def __len__(self) -> int:
        return len(self.images)

    def __iter__(self) -> Iterator[Image]:
        return iter(self.images)

    def __getitem__(self, index: int) -> Image:
        return self.images[index]

    def __repr__(self) -> str:
        return (
            f"<nhan.Dataset {self.name!r}: {len(self.images)} images, "
            f"{self.box_count} boxes, format={self.format}>"
        )

    @property
    def boxes(self) -> List[Box]:
        return [b for img in self.images for b in img.boxes]

    @property
    def box_count(self) -> int:
        return sum(len(img.boxes) for img in self.images)

    @property
    def labels(self) -> List[str]:
        """Label names in use: the export's order first, then any new ones."""
        used = {b.label for b in self.boxes}
        ordered = [n for n in self.label_order if n in used]
        return ordered + sorted(used - set(ordered))

    def _copy(self, images: List[Image]) -> "Dataset":
        new = copy.copy(self)
        new.images = images
        return new

    # ---- choosing data ------------------------------------------------

    def split(self, name: str) -> "Dataset":
        """Only the images of one split: "train", "val" or "test"."""
        if name not in SPLITS:
            from .errors import NhanError

            raise NhanError(f'Unknown split "{name}". Use one of: train, val, test.')
        return self._copy([img for img in self.images if img.split == name])

    def filter(
        self,
        cls: Optional[str] = None,
        split: Optional[str] = None,
        keep_other_boxes: bool = False,
        **attributes,
    ) -> "Dataset":
        """Images with at least one matching box.

            ds.filter(cls="Mango")
            ds.filter(cls="Mango", condition="Unripe")
            ds.filter(condition=None)      # boxes with no Condition

        By default only the matching boxes are kept. Pass
        keep_other_boxes=True to keep every box of a matching image.
        """
        out: List[Image] = []
        for img in self.images:
            if split is not None and img.split != split:
                continue
            hits = [b for b in img.boxes if b.matches(cls, **attributes)]
            if not hits:
                continue
            new = copy.copy(img)
            new.boxes = list(img.boxes) if keep_other_boxes else hits
            out.append(new)
        return self._copy(out)

    def where(self, test: Callable[[Box], bool]) -> "Dataset":
        """Like filter(), with your own test: ds.where(lambda b: b.w > 0.5)."""
        out = []
        for img in self.images:
            hits = [b for b in img.boxes if test(b)]
            if hits:
                new = copy.copy(img)
                new.boxes = hits
                out.append(new)
        return self._copy(out)

    def resplit(
        self, train: float = 0.8, val: float = 0.1, test: float = 0.1, seed: int = 42
    ) -> "Dataset":
        """New random train/val/test split. Same seed, same result."""
        total = train + val + test
        if total <= 0:
            from .errors import NhanError

            raise NhanError("The split parts must add up to more than 0.")
        order = list(range(len(self.images)))
        random.Random(seed).shuffle(order)
        n = len(order)
        n_train = round(n * train / total)
        n_val = round(n * val / total)
        split_of = {}
        for rank, index in enumerate(order):
            split_of[index] = (
                "train" if rank < n_train else ("val" if rank < n_train + n_val else "test")
            )
        images = []
        for index, img in enumerate(self.images):
            new = copy.copy(img)
            new.split = split_of[index]
            images.append(new)
        return self._copy(images)

    def relabel(self, target: str = "label", separator: str = "_") -> "Dataset":
        """Change what the label of each box is. Useful before training.

            "label"      keep the export's names (default)
            "class"      only the main class: "Mango"
            "joined"     class plus every value: "Mango_Ripe_Alphonso"
            <sub-category name>, e.g. "condition"
                         only that value: "Ripe"; boxes without a value are
                         dropped
        """
        t = _key(target)
        sub_names = {_key(n): n for n in self.sub_categories}
        out = []
        for img in self.images:
            boxes = []
            for b in img.boxes:
                nb = copy.copy(b)
                if t == "label":
                    pass
                elif t == "class":
                    # Only the class: values are dropped, so no writer
                    # joins them back into the name.
                    nb.label = b.cls
                    nb.attributes = {}
                elif t == "joined":
                    parts = [b.cls] + [
                        v for v in (b.get(n) for n in self.sub_categories) if v is not None
                    ]
                    nb.label = separator.join(parts)
                else:
                    if t not in sub_names and not any(b.get(target) for b in self.boxes):
                        from .errors import NhanError

                        raise NhanError(
                            f'Unknown target "{target}". Use label, class, joined'
                            + "".join(f", {n}" for n in self.sub_categories)
                            + "."
                        )
                    value = b.get(sub_names.get(t, target))
                    if value is None:
                        continue
                    nb.label = value
                boxes.append(nb)
            new = copy.copy(img)
            new.boxes = boxes
            out.append(new)
        result = self._copy(out)
        if t != "label":
            result.label_order = []
        return result

    # ---- numbers ------------------------------------------------------

    def counts(self, by: str = "label") -> Dict[str, int]:
        """Boxes per label, per "class", per "split", or per sub-category.

        For a sub-category, boxes without a value count as "(none)".
        """
        t = _key(by)
        c: Counter = Counter()
        for img in self.images:
            for b in img.boxes:
                if t == "label":
                    c[b.label] += 1
                elif t == "class":
                    c[b.cls] += 1
                elif t == "split":
                    c[img.split] += 1
                else:
                    c[b.get(by) or "(none)"] += 1
        return dict(c.most_common())

    def summary(self) -> str:
        """A short, readable overview."""
        lines = [f"{self.name}  ({self.format.upper()}, {_n(len(self.images), 'image')}, {_n(self.box_count, 'box')})"]
        per_split = Counter(img.split for img in self.images)
        lines.append(
            "  Splits:  " + ", ".join(f"{s} {per_split.get(s, 0)}" for s in SPLITS)
        )
        empty = sum(1 for img in self.images if not img.boxes)
        if empty:
            lines.append(f"  Images without boxes: {empty}")
        lines.append("  Classes:")
        for name, n in self.counts("class").items():
            lines.append(f"    {name:<24} {n}")
        for sub in self.sub_categories:
            lines.append(f"  {sub}:")
            for name, n in self.counts(sub).items():
                lines.append(f"    {name:<24} {n}")
        return "\n".join(lines)

    # ---- output -------------------------------------------------------

    def to_coco(self, out_dir) -> Path:
        """Write as COCO (Nhan layout). Returns the folder."""
        from .writers import write

        return write(self, out_dir, "coco")

    def to_yolo(self, out_dir) -> Path:
        """Write as YOLO (images/, labels/, data.yaml)."""
        from .writers import write

        return write(self, out_dir, "yolo")

    def to_voc(self, out_dir) -> Path:
        """Write as Pascal VOC."""
        from .writers import write

        return write(self, out_dir, "voc")

    def to_json(self, path) -> Path:
        """Write one simple JSON file with everything (sdk/schema/nhan-dataset.schema.json)."""
        from .writers import write_json

        return write_json(self, path)

    def preview(self, path="preview.html", limit: int = 500) -> Path:
        """An HTML page with every photo and its boxes. Open it in a browser."""
        from .preview import write_preview

        return write_preview(self, path, limit=limit)

    def crops(self, out_dir, padding: float = 0.0, size: Optional[int] = None) -> Path:
        """Save each box as its own image: out_dir/<split>/<label>/... (needs Pillow)."""
        from .crops import write_crops

        return write_crops(self, out_dir, padding=padding, size=size)

    def torch_detection(self, split: str = "train", transforms=None):
        """A PyTorch Dataset for detection (torchvision style). Needs PyTorch."""
        from .torch_data import DetectionDataset

        return DetectionDataset(self, split, transforms=transforms)

    def torch_classification(self, split: str = "train", size: int = 224, padding: float = 0.1):
        """A PyTorch Dataset of box crops with label indices. Needs PyTorch."""
        from .torch_data import CropDataset

        return CropDataset(self, split, size=size, padding=padding)
