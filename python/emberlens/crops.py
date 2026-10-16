"""Save each box as its own image, sorted into one folder per label.

The layout out/<split>/<label>/<photo>_<n>.jpg is what most image
classification tools expect ("ImageFolder").
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from .errors import EmberlensError
from .extras import require_pillow


def safe_name(text: str) -> str:
    name = re.sub(r"[^\w.\- ]+", "_", text).strip(" .")
    return name or "_"


def crop_box(pil_image, box, padding: float = 0.0):
    """Crop one box out of a Pillow image, with `padding` (share of box size) around it."""
    W, H = pil_image.size
    x0, y0, x1, y1 = box.pixels(W, H)
    px, py = (x1 - x0) * padding, (y1 - y0) * padding
    left, top = max(0, int(x0 - px)), max(0, int(y0 - py))
    right, bottom = min(W, int(round(x1 + px))), min(H, int(round(y1 + py)))
    right, bottom = max(right, left + 1), max(bottom, top + 1)
    return pil_image.crop((left, top, right, bottom))


def crop_for_model(pil_image, box, padding: float = 0.1, size: int = 224):
    """The one crop rule for models: 10% padding, then stretch to size x size.

    Training, `export-model` and the app all use this rule, so a model sees
    the same pixels in training and on the phone. Do not change it without
    changing the ONNX metadata keys (crop_pad, resize_mode) and the app.
    """
    return crop_box(pil_image, box, padding).resize((size, size))


def write_crops(ds, out_dir, padding: float = 0.0, size: Optional[int] = None) -> Path:
    require_pillow()
    out = Path(out_dir).expanduser()
    if out.exists() and any(out.iterdir()):
        raise EmberlensError(f"{out} is not empty. Choose a new folder.")
    count = 0
    for img in ds.images:
        if not img.boxes:
            continue
        photo = img.open().convert("RGB")
        for n, box in enumerate(img.boxes, 1):
            crop = crop_for_model(photo, box, padding, size) if size else crop_box(photo, box, padding)
            folder = out / img.split / safe_name(box.label)
            folder.mkdir(parents=True, exist_ok=True)
            crop.save(folder / f"{safe_name(img.id)}_{n}.jpg", quality=95)
            count += 1
    out.mkdir(parents=True, exist_ok=True)
    print(f"Saved {count} crops in {out}")
    return out
