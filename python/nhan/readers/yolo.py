"""YOLO: data.yaml + images/<split>/ + labels/<split>/<id>.txt.

Each line: "<class index> <cx> <cy> <w> <h>", normalised, centre-based.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from ..errors import NhanError
from ..imagesize import image_size
from ..model import SPLITS, Box, Image

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _scalar(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        inner = text[1:-1]
        return inner.replace("''", "'") if text[0] == "'" else inner.replace('\\"', '"')
    return text


def _split_flow(text: str) -> List[str]:
    """Split "[a, 'b, c', d]" into items, respecting quotes."""
    body = text.strip()[1:-1]
    items, cur, quote = [], "", None
    i = 0
    while i < len(body):
        ch = body[i]
        if quote:
            cur += ch
            if ch == quote:
                if quote == "'" and i + 1 < len(body) and body[i + 1] == "'":
                    cur += "'"
                    i += 1
                else:
                    quote = None
        elif ch in "'\"":
            quote = ch
            cur += ch
        elif ch == ",":
            items.append(cur)
            cur = ""
        else:
            cur += ch
        i += 1
    if cur.strip():
        items.append(cur)
    return [_scalar(x) for x in items]


def parse_data_yaml(text: str) -> Dict[str, object]:
    """The small YAML subset used by data.yaml files (no PyYAML needed)."""
    out: Dict[str, object] = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].split(" #")[0].rstrip()
        i += 1
        if not line.strip() or line.lstrip().startswith("#") or line.startswith(" "):
            continue
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if value.startswith("["):
            while not value.rstrip().endswith("]") and i < len(lines):
                value += " " + lines[i].split(" #")[0].strip()
                i += 1
            out[key] = _split_flow(value)
        elif value == "":
            block_list: List[str] = []
            block_map: Dict[int, str] = {}
            while i < len(lines) and (lines[i].startswith(" ") or lines[i].startswith("-")):
                item = lines[i].strip()
                i += 1
                if item.startswith("- "):
                    block_list.append(_scalar(item[2:]))
                elif ":" in item:
                    k, _, v = item.partition(":")
                    if k.strip().isdigit():
                        block_map[int(k.strip())] = _scalar(v)
            out[key] = block_list or [block_map[k] for k in sorted(block_map)]
        else:
            out[key] = _scalar(value)
    return out


def read(root: Path) -> Tuple[List[Image], List[str]]:
    yaml_path = root / "data.yaml"
    if not yaml_path.is_file():
        found = sorted(root.glob("*.yaml"))
        if not found:
            raise NhanError(f"No data.yaml found in {root}.")
        yaml_path = found[0]
    data = parse_data_yaml(yaml_path.read_text(encoding="utf-8"))
    names = data.get("names") or []
    if not isinstance(names, list):
        raise NhanError("data.yaml has no names list.")
    base = root / str(data.get("path") or ".")
    images: List[Image] = []
    for split in SPLITS:
        rel = data.get(split)
        if not isinstance(rel, str) or not rel:
            continue
        img_dir = (base / rel).resolve()
        if not img_dir.is_dir():
            continue
        parts = list(img_dir.parts)
        if "images" in parts:
            k = len(parts) - 1 - parts[::-1].index("images")
            parts[k] = "labels"
        label_dir = Path(*parts)
        for path in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXT):
            size = image_size(path) or (0, 0)
            img = Image(id=path.stem, path=path, width=size[0], height=size[1], split=split)
            txt = label_dir / (path.stem + ".txt")
            if txt.is_file():
                for line in txt.read_text(encoding="utf-8").splitlines():
                    fields = line.split()
                    if len(fields) < 5:
                        continue
                    index = int(float(fields[0]))
                    cx, cy, w, h = (float(v) for v in fields[1:5])
                    name = names[index] if 0 <= index < len(names) else str(index)
                    img.boxes.append(
                        Box(label=name, cls=name, x=cx - w / 2, y=cy - h / 2, w=w, h=h)
                    )
            images.append(img)
    return images, list(names)
