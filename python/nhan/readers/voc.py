"""Pascal VOC: Annotations/<id>.xml, JPEGImages/, ImageSets/Main/<split>.txt.

VOC pixel boxes are 1-based and inclusive: xmin = x * W + 1, xmax = (x + w) * W.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

from ..errors import NhanError
from ..minixml import parse_string
from ..model import SPLITS, Box, Image


def _text(node, tag: str, default: str = "") -> str:
    child = node.find(tag)
    return child.text.strip() if child is not None and child.text else default


def _parse(path: Path) -> ET.Element:
    text = path.read_text(encoding="utf-8")
    try:
        return ET.fromstring(text)
    except ImportError:
        pass  # expat is missing: use the small built-in reader
    except ET.ParseError as e:
        raise NhanError(f"{path.name} is not valid XML: {e}") from None
    try:
        return parse_string(text)
    except ValueError as e:
        raise NhanError(f"{path.name} is not valid XML: {e}") from None


def read(root: Path) -> Tuple[List[Image], List[str]]:
    ann_dir = root / "Annotations"
    if not ann_dir.is_dir():
        raise NhanError(f"No Annotations folder found in {root}.")
    split_of: Dict[str, str] = {}
    for split in SPLITS:
        listing = root / "ImageSets" / "Main" / f"{split}.txt"
        if listing.is_file():
            for line in listing.read_text(encoding="utf-8").split():
                split_of.setdefault(line.strip(), split)
    images: List[Image] = []
    label_order: List[str] = []
    for xml_path in sorted(ann_dir.glob("*.xml")):
        node = _parse(xml_path)
        size = node.find("size")
        width = int(float(_text(size, "width", "0"))) if size is not None else 0
        height = int(float(_text(size, "height", "0"))) if size is not None else 0
        file_name = _text(node, "filename", xml_path.stem + ".jpg")
        img = Image(
            id=xml_path.stem,
            path=root / "JPEGImages" / file_name,
            width=width,
            height=height,
            split=split_of.get(xml_path.stem, "train"),
        )
        for obj in node.findall("object"):
            name = _text(obj, "name")
            if name not in label_order:
                label_order.append(name)
            bb = obj.find("bndbox")
            if bb is None or not width or not height:
                continue
            xmin, ymin = float(_text(bb, "xmin")), float(_text(bb, "ymin"))
            xmax, ymax = float(_text(bb, "xmax")), float(_text(bb, "ymax"))
            attributes = {}
            for attr in obj.findall("attributes/attribute"):
                attributes[_text(attr, "name")] = _text(attr, "value")
            img.boxes.append(
                Box(
                    label=name,
                    cls=name,
                    x=(xmin - 1) / width,
                    y=(ymin - 1) / height,
                    w=(xmax - xmin + 1) / width,
                    h=(ymax - ymin + 1) / height,
                    attributes=attributes,
                )
            )
        images.append(img)
    return images, label_order
