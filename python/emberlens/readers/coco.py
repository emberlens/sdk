"""COCO: annotations/instances_<split>.json + images/<split>/<file>."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

from ..errors import EmberlensError
from ..model import SPLITS, Box, Image, box_from_polygon


def _split_of(json_path: Path) -> str:
    m = re.search(r"(train|val|test)", json_path.stem)
    return m.group(1) if m else "train"


def _find_image(root: Path, split: str, file_name: str) -> Path:
    for candidate in (
        root / "images" / split / file_name,
        root / "images" / file_name,
        root / split / file_name,
        root / file_name,
    ):
        if candidate.is_file():
            return candidate
    return root / "images" / split / file_name


def read(root: Path) -> Tuple[List[Image], List[str]]:
    folder = root / "annotations"
    files = sorted(folder.glob("*.json")) if folder.is_dir() else []
    if not files:
        files = sorted(p for p in root.glob("*.json") if p.name != "emberlens.json")
    if not files:
        raise EmberlensError(f"No COCO annotation files found in {root}.")
    images: List[Image] = []
    label_order: List[str] = []
    files.sort(key=lambda p: SPLITS.index(_split_of(p)) if _split_of(p) in SPLITS else 9)
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as e:
            raise EmberlensError(f"{path.name} is not valid JSON: {e}") from None
        split = _split_of(path)
        cats: Dict[int, str] = {}
        for c in data.get("categories", []):
            cats[c["id"]] = c["name"]
            if c["name"] not in label_order:
                label_order.append(c["name"])
        by_id: Dict[int, Image] = {}
        for im in data.get("images", []):
            file_name = im["file_name"]
            img = Image(
                id=Path(file_name).stem,
                path=_find_image(root, split, file_name),
                width=int(im["width"]),
                height=int(im["height"]),
                split=split,
            )
            by_id[im["id"]] = img
            images.append(img)
        for a in data.get("annotations", []):
            img = by_id.get(a["image_id"])
            if img is None or not img.width or not img.height:
                continue
            x, y, w, h = a["bbox"]
            name = cats.get(a["category_id"], str(a["category_id"]))
            attrs = {str(k): str(v) for k, v in (a.get("attributes") or {}).items()}
            seg = a.get("segmentation")
            # Polygon form: [[x1, y1, x2, y2, ...], ...]. The first ring is
            # the outline (Emberlens writes one). Pixel masks (RLE) stay boxes.
            if isinstance(seg, list) and seg and isinstance(seg[0], list) and len(seg[0]) >= 6:
                ring = seg[0]
                pts = [(ring[i] / img.width, ring[i + 1] / img.height) for i in range(0, len(ring) - 1, 2)]
                img.boxes.append(box_from_polygon(name, name, pts, attrs))
                continue
            img.boxes.append(
                Box(
                    label=name,
                    cls=name,
                    x=x / img.width,
                    y=y / img.height,
                    w=w / img.width,
                    h=h / img.height,
                    attributes=attrs,
                )
            )
    return images, label_order
