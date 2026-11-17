"""Helpers for the training notebook (sdk/notebooks/train.ipynb).

The notebook has one cell to edit. All logic lives here, so it has tests.
It runs in the user's own Colab or Kaggle account and sends nothing to Emberlens.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Callable, Optional

from .errors import EmberlensError
from .paths import export_stem, free_path

KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")
KINDS = ("classify", "detect")
EPOCHS = {"classify": 5, "detect": 10}  # the same defaults as the command line


def find_export(text: str, exists: Callable[[Path], bool] = Path.exists) -> Path:
    """Clean the pasted path and check that it names an export zip or folder."""
    clean = text.strip().strip("\"'").strip()
    if not clean:
        raise EmberlensError("Paste the path of your export zip in the first code cell, between the quotes.")
    if clean.lower().startswith("http"):
        raise EmberlensError(
            "Paste the file path, not a link. In the Files panel on the left, "
            "open drive > MyDrive, find the zip, click ⋮ > Copy path."
        )
    path = Path(clean)
    if not exists(path) and clean.startswith("/content/drive/My Drive/"):
        alt = Path("/content/drive/MyDrive/" + clean[len("/content/drive/My Drive/"):])
        if exists(alt):
            path = alt
    if not exists(path):
        if clean.startswith("/kaggle/"):
            raise EmberlensError(
                f"Can't find {path}. Check that the dataset is added to the notebook, "
                "and that the path starts with /kaggle/input/."
            )
        raise EmberlensError(
            f"Can't find {path}. Check the name, and check that Google Drive is connected "
            "(run the cell with drive.mount, or use Run all)."
        )
    if not path.is_dir() and path.suffix.lower() != ".zip":
        raise EmberlensError(f"{path.name} is not an export. Use the .zip file from the Emberlens app.")
    return path


def model_out_path(export: Path, kind: str) -> Path:
    """Where the .onnx goes: next to the zip, or in /kaggle/working on Kaggle."""
    export = Path(export)
    folder = KAGGLE_WORKING if KAGGLE_INPUT in export.parents else export.parent
    return free_path(folder / f"{export_stem(str(export))}-{kind}.onnx")


def run(
    export_text: str,
    kind: str,
    labels: str = "label",
    epochs: Optional[int] = None,
    pretrained: bool = True,
    log: Callable[[str], None] = print,
) -> Path:
    """Train on the export and write the .onnx file. Returns the file."""
    kind = kind.strip().lower()
    labels = labels.strip()
    if kind not in KINDS:
        raise EmberlensError(f"KIND is {kind!r}. Use classify or detect.")
    export = find_export(export_text)
    from .export_model import export_model
    from .load import load
    from .train import train_classifier, train_detector

    train = train_classifier if kind == "classify" else train_detector
    out = model_out_path(export, kind)
    with tempfile.TemporaryDirectory(prefix="emberlens-") as tmp:
        source = export
        if export.is_file():
            # load() unzips next to the zip. Unzip a copy, so Drive stays clean
            # and a read-only /kaggle/input does not fail.
            source = Path(tmp) / "export.zip"
            shutil.copyfile(export, source)
        ds = load(source)
        folder = train(
            ds,
            Path(tmp) / "model",
            target=labels or "label",
            epochs=epochs or EPOCHS[kind],
            pretrained=pretrained,
            log=log,
        )
        result = export_model(folder, out=out, log=log)
    log(f"Saved {result}. Send it to your iPhone, then add it in Project settings > Your model.")
    return result
