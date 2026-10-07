"""Write a Dataset as COCO, YOLO, Pascal VOC, or one dataset.json file.

The folder layouts are the same as the Nhan app's exports (sdk/SPEC.md), and
a fresh nhan.json card is written next to them, so the result can be loaded
again with nhan.load().
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from .card import CARD_FILE
from .errors import NhanError
from .model import SPLITS, Dataset, Image


def _prepare(out_dir) -> Path:
    out = Path(out_dir).expanduser()
    if out.exists() and any(out.iterdir()):
        raise NhanError(f"{out} is not empty. Choose a new folder.")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _fields_style(box) -> bool:
    """Values are written as separate fields when the label is the class."""
    return box.label == box.cls


def _label_entries(ds: Dataset) -> List[dict]:
    """One entry per label: what the name means.

    A joined name ("Mango_Ripe") carries its class and values. A label whose
    boxes do not all agree (e.g. after relabel("condition")) only names
    itself, so a reader never invents values.
    """
    seen: Dict[str, tuple] = {}
    mixed = set()
    for b in ds.boxes:
        attrs = {} if _fields_style(b) else dict(b.attributes)
        key = (b.cls, tuple(sorted(attrs.items())))
        if b.label in seen and seen[b.label][0] != key:
            mixed.add(b.label)
        seen.setdefault(b.label, (key, attrs))
    out = []
    for name in ds.labels:
        (cls, _), attrs = seen[name]
        if name in mixed:
            out.append({"name": name, "class": name, "attributes": {}})
        else:
            out.append({"name": name, "class": cls, "attributes": attrs})
    return out


def _style(ds: Dataset) -> str:
    if any(b.attributes and _fields_style(b) for b in ds.boxes):
        return "separate_fields"
    if any(b.attributes for b in ds.boxes):
        return "joined"
    return "single"


def card_for(ds: Dataset, fmt: str) -> dict:
    """A nhan.json card that describes `ds` as written in `fmt`."""
    old = ds.card or {}
    return {
        "nhan_card_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "project": old.get("project") or {"name": ds.name, "category": ""},
        "format": fmt,
        "label_style": _style(ds),
        "separator": old.get("separator", "_"),
        "lowercase": bool(old.get("lowercase", False)),
        "root_category": old.get("root_category", "class"),
        "classes": ds.classes or sorted({b.cls for b in ds.boxes}),
        "sub_categories": [{"name": k, "values": v} for k, v in ds.sub_categories.items()],
        "labels": _label_entries(ds),
        "split": old.get("split") or {},
        "seed": old.get("seed", 0),
        "images": old.get("images") or {},
        "written_by": "nhan-sdk",
    }


def _copy_image(img: Image, dest: Path) -> str:
    dest.mkdir(parents=True, exist_ok=True)
    name = img.id + (img.path.suffix or ".jpg")
    shutil.copy2(img.path, dest / name)
    return name


def _write_coco(ds: Dataset, out: Path, labels: List[str]) -> None:
    index = {name: i + 1 for i, name in enumerate(labels)}
    (out / "annotations").mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        part = [img for img in ds.images if img.split == split]
        data = {"images": [], "annotations": [], "categories": []}
        ann_id = 1
        for img_id, img in enumerate(part, 1):
            file_name = _copy_image(img, out / "images" / split)
            data["images"].append(
                {"id": img_id, "file_name": file_name, "width": img.width, "height": img.height}
            )
            for b in img.boxes:
                w, h = b.w * img.width, b.h * img.height
                ann = {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": index[b.label],
                    "bbox": [round(b.x * img.width, 2), round(b.y * img.height, 2), round(w, 2), round(h, 2)],
                    "area": round(w * h, 2),
                    "iscrowd": 0,
                }
                if b.attributes and _fields_style(b):
                    ann["attributes"] = dict(b.attributes)
                data["annotations"].append(ann)
                ann_id += 1
        data["categories"] = [{"id": i, "name": n} for n, i in index.items()]
        (out / "annotations" / f"instances_{split}.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def _yaml_quote(name: str) -> str:
    return "'" + name.replace("'", "''") + "'"


def _write_yolo(ds: Dataset, out: Path, labels: List[str]) -> None:
    index = {name: i for i, name in enumerate(labels)}
    for split in SPLITS:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    for img in ds.images:
        _copy_image(img, out / "images" / img.split)
        lines = [
            f"{index[b.label]} {b.x + b.w / 2:.6f} {b.y + b.h / 2:.6f} {b.w:.6f} {b.h:.6f}"
            for b in img.boxes
        ]
        (out / "labels" / img.split / f"{img.id}.txt").write_text(
            "".join(line + "\n" for line in lines), encoding="utf-8"
        )
    yaml = (
        "path: .\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"nc: {len(labels)}\nnames: [{', '.join(_yaml_quote(n) for n in labels)}]\n"
    )
    (out / "data.yaml").write_text(yaml, encoding="utf-8")


def _xml(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def _write_voc(ds: Dataset, out: Path) -> None:
    (out / "Annotations").mkdir(parents=True, exist_ok=True)
    (out / "ImageSets" / "Main").mkdir(parents=True, exist_ok=True)
    for img in ds.images:
        file_name = _copy_image(img, out / "JPEGImages")
        parts = [
            "<annotation>",
            "  <folder>JPEGImages</folder>",
            f"  <filename>{_xml(file_name)}</filename>",
            "  <size>",
            f"    <width>{img.width}</width>",
            f"    <height>{img.height}</height>",
            "    <depth>3</depth>",
            "  </size>",
        ]
        for b in img.boxes:
            x0, y0, x1, y1 = b.pixels(img.width, img.height)
            parts += [
                "  <object>",
                f"    <name>{_xml(b.label)}</name>",
                "    <pose>Unspecified</pose>",
                "    <truncated>0</truncated>",
                "    <difficult>0</difficult>",
                "    <bndbox>",
                f"      <xmin>{round(x0) + 1}</xmin>",
                f"      <ymin>{round(y0) + 1}</ymin>",
                f"      <xmax>{round(x1)}</xmax>",
                f"      <ymax>{round(y1)}</ymax>",
                "    </bndbox>",
            ]
            if b.attributes and _fields_style(b):
                parts.append("    <attributes>")
                for k, v in b.attributes.items():
                    parts += [
                        "      <attribute>",
                        f"        <name>{_xml(k)}</name>",
                        f"        <value>{_xml(v)}</value>",
                        "      </attribute>",
                    ]
                parts.append("    </attributes>")
            parts.append("  </object>")
        parts.append("</annotation>")
        (out / "Annotations" / f"{img.id}.xml").write_text("\n".join(parts) + "\n", encoding="utf-8")
    for split in SPLITS:
        ids = [img.id for img in ds.images if img.split == split]
        (out / "ImageSets" / "Main" / f"{split}.txt").write_text(
            "".join(i + "\n" for i in ids), encoding="utf-8"
        )


def write(ds: Dataset, out_dir, fmt: str) -> Path:
    fmt = fmt.lower()
    if fmt == "json":
        return write_json(ds, Path(out_dir) / "dataset.json")
    if fmt not in ("coco", "yolo", "voc"):
        raise NhanError(f'Unknown format "{fmt}". Use coco, yolo, voc or json.')
    missing = [img.path for img in ds.images if not img.path.is_file()]
    if missing:
        raise NhanError(f"Image file missing: {missing[0]}")
    if fmt == "yolo" and any(b.attributes and _fields_style(b) for b in ds.boxes):
        # YOLO has only one name per box: join the values into it, like the
        # app does (D-019), so no value is lost.
        sep = (ds.card or {}).get("separator", "_")
        ds = ds.relabel("joined", separator=sep)
    out = _prepare(out_dir)
    labels = ds.labels
    if fmt == "coco":
        _write_coco(ds, out, labels)
    elif fmt == "yolo":
        _write_yolo(ds, out, labels)
    else:
        _write_voc(ds, out)
    card = card_for(ds, fmt)
    (out / CARD_FILE).write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def dataset_dict(ds: Dataset, base: Path) -> dict:
    """The dataset as plain data (sdk/schema/nhan-dataset.schema.json).

    Image paths are relative to `base` when possible.
    """

    def rel(p: Path) -> str:
        try:
            return Path(p).resolve().relative_to(base.resolve()).as_posix()
        except ValueError:
            return Path(p).resolve().as_posix()

    return {
        "nhan_dataset_version": 1,
        "name": ds.name,
        "source_format": ds.format,
        "classes": ds.classes,
        "sub_categories": [{"name": k, "values": v} for k, v in ds.sub_categories.items()],
        "labels": _label_entries(ds),
        "images": [
            {
                "id": img.id,
                "file": rel(img.path),
                "width": img.width,
                "height": img.height,
                "split": img.split,
                "boxes": [
                    {
                        "label": b.label,
                        "class": b.cls,
                        "attributes": dict(b.attributes),
                        "x": round(b.x, 6),
                        "y": round(b.y, 6),
                        "w": round(b.w, 6),
                        "h": round(b.h, 6),
                    }
                    for b in img.boxes
                ],
            }
            for img in ds.images
        ],
    }


def write_json(ds: Dataset, path) -> Path:
    p = Path(path).expanduser()
    if p.suffix.lower() != ".json":
        p = p / "dataset.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dataset_dict(ds, p.parent), indent=2, ensure_ascii=False), encoding="utf-8")
    return p
