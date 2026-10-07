"""nhan.load(): open an export zip or folder."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Optional, Union

from .card import label_parts, read_card
from .errors import NhanError
from .model import Dataset
from .readers import coco, voc, yolo

READERS = {"coco": coco.read, "yolo": yolo.read, "voc": voc.read}
_MARKER = ".nhan_extracted"


def _extract(zip_path: Path) -> Path:
    """Unzip next to the zip, into a folder with the same name.

    Reused on the next run if the zip did not change.
    """
    stamp = f"{zip_path.stat().st_size}:{int(zip_path.stat().st_mtime)}"
    target = zip_path.with_suffix("")
    n = 2
    while target.exists():
        marker = target / _MARKER
        if marker.is_file():
            if marker.read_text() == stamp:
                return target
            break  # ours, but old: extract again into it
        target = zip_path.with_name(f"{zip_path.stem} ({n})")
        n += 1
    target.mkdir(parents=True, exist_ok=True)
    root = target.resolve()
    try:
        with zipfile.ZipFile(zip_path) as z:
            for member in z.infolist():
                dest = (target / member.filename).resolve()
                if root != dest and root not in dest.parents:
                    raise NhanError(f"The zip has an unsafe path: {member.filename}")
            z.extractall(target)
    except zipfile.BadZipFile:
        raise NhanError(f"{zip_path.name} is not a valid zip file.") from None
    (target / _MARKER).write_text(stamp)
    return target


def _find_root(folder: Path) -> Path:
    """The folder that holds the export (the zip may add one level)."""
    signs = ("nhan.json", "data.yaml", "annotations", "Annotations")
    if any((folder / s).exists() for s in signs):
        return folder
    children = [p for p in folder.iterdir() if p.is_dir() and not p.name.startswith((".", "__"))]
    if len(children) == 1:
        return _find_root(children[0])
    return folder


def _has(root: Path, name: str, pattern: str) -> bool:
    """True when `root/name` holds files like `pattern`.

    Checks the real file names: macOS and Windows ignore letter case, so
    "annotations" (COCO) and "Annotations" (VOC) are the same folder there.
    """
    folder = root / name
    return folder.is_dir() and any(folder.glob(pattern))


def detect_format(root: Path, card: Optional[dict]) -> str:
    if card and card.get("format") in READERS:
        return card["format"]
    if _has(root, "Annotations", "*.xml"):
        return "voc"
    if (root / "data.yaml").is_file() or _has(root, "labels", "**/*.txt"):
        return "yolo"
    if _has(root, "annotations", "*.json") or [p for p in root.glob("*.json") if p.name != "nhan.json"]:
        return "coco"
    raise NhanError(
        f"Can't tell the format of {root}. Expected a Nhan export "
        "(COCO, YOLO or Pascal VOC)."
    )


def load(path: Union[str, Path], format: Optional[str] = None) -> Dataset:
    """Open a Nhan export: the .zip from the app, or its unzipped folder.

        ds = nhan.load("fruit-stand-coco-2026-10-08.zip")

    Also opens plain COCO, YOLO and Pascal VOC datasets from other tools.
    """
    p = Path(path).expanduser()
    if not p.exists():
        raise NhanError(f"Can't find {p}. Check the path.")
    if p.is_file():
        if p.suffix.lower() != ".zip":
            raise NhanError(f"{p.name} is not a .zip or a folder.")
        p = _extract(p)
    root = _find_root(p.resolve())
    card = read_card(root)
    fmt = (format or detect_format(root, card)).lower()
    if fmt not in READERS:
        raise NhanError(f'Unknown format "{fmt}". Use coco, yolo or voc.')
    images, label_order = READERS[fmt](root)

    # Fill class and values from the card (joined names) and keep values
    # the files already carry (separate fields).
    parts = label_parts(card)
    for img in images:
        for box in img.boxes:
            known = parts.get(box.label)
            if known:
                box.cls = known[0]
                merged = dict(known[1])
                merged.update(box.attributes)
                box.attributes = merged

    if card:
        classes = list(card.get("classes") or [])
        subs = {
            s["name"]: list(s.get("values") or [])
            for s in card.get("sub_categories") or []
            if isinstance(s, dict) and "name" in s
        }
        name = (card.get("project") or {}).get("name") or root.name
        label_order = [e["name"] for e in card.get("labels") or []] or label_order
    else:
        classes = []
        for b in (b for img in images for b in img.boxes):
            if b.cls not in classes:
                classes.append(b.cls)
        subs = {}
        for b in (b for img in images for b in img.boxes):
            for k, v in b.attributes.items():
                subs.setdefault(k, [])
                if v not in subs[k]:
                    subs[k].append(v)
        name = root.name

    missing = [img for img in images if not img.path.is_file()]
    if missing and len(missing) == len(images):
        raise NhanError(f"The annotation files point to images that are not in {root}.")

    return Dataset(
        name=name,
        format=fmt,
        root=root,
        images=images,
        classes=classes,
        sub_categories=subs,
        card=card,
        label_order=label_order,
    )
