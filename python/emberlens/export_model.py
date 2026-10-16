"""Turn a trained classifier into one standard .onnx file for the app.

    ./emberlens export-model MODEL-FOLDER [--out FILE.onnx]

The class names and the crop rule go into the ONNX metadata (sdk/SPEC.md
section 9), so the app needs no hard-coded numbers. Needs PyTorch and the
optional `onnx` library (./emberlens install onnx).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Optional

from .errors import EmberlensError
from .extras import require_onnx, require_torch
from .torch_data import IMAGENET_MEAN, IMAGENET_STD

OPSET = 17
RESIZE_MODE = "stretch"  # the crop rule in crops.crop_for_model


def build_metadata(ckpt: dict, report: Optional[dict] = None) -> Dict[str, str]:
    """The metadata_props of the file, all as text. See SPEC.md section 9."""
    size = int(ckpt.get("size", 224))
    meta = {
        "emberlens_model": "1",
        "kind": "classifier",
        "classes": json.dumps(list(ckpt["classes"]), ensure_ascii=False),
        "size": str(size),
        "input_size": str(size),
        "crop_pad": str(ckpt.get("padding", 0.1)),
        "resize_mode": RESIZE_MODE,
        "mean": json.dumps(IMAGENET_MEAN),
        "std": json.dumps(IMAGENET_STD),
        "labels_mode": str(ckpt.get("target", "label")),
    }
    accuracy = (report or {}).get("test_accuracy")
    if accuracy is not None:
        meta["test_accuracy"] = str(accuracy)
    return meta


def export_model(model_folder, out=None, log: Callable[[str], None] = print) -> Path:
    """Write <out>.onnx from a classifier folder. Returns the file."""
    onnx = require_onnx()
    torch, _ = require_torch()
    from .train import load_model

    folder = Path(model_folder).expanduser()
    model, ckpt = load_model(folder)
    if ckpt.get("kind") != "classifier":
        raise EmberlensError("export-model works on a classifier (./emberlens train classify). Detectors are not supported yet.")
    report = None
    report_file = (folder if folder.is_dir() else folder.parent) / "report.json"
    if report_file.is_file():
        try:
            report = json.loads(report_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            report = None
    path = Path(out).expanduser() if out else Path(f"{folder.stem if folder.is_file() else folder.name}.onnx")
    size = int(ckpt.get("size", 224))
    # Input 1x3xSxS (RGB, mean/std applied by the caller); output: one score
    # per class (logits, in the order of `classes`). The batch can be any size.
    # The old exporter writes plain layers and needs no extra library.
    torch.onnx.export(
        model,
        (torch.zeros(1, 3, size, size),),
        str(path),
        input_names=["input"],
        output_names=["scores"],
        dynamic_axes={"input": {0: "batch"}, "scores": {0: "batch"}},
        opset_version=OPSET,
        dynamo=False,
    )
    proto = onnx.load(str(path))
    del proto.metadata_props[:]
    for key, value in build_metadata(ckpt, report).items():
        entry = proto.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.save(proto, str(path))
    log(f"Saved {path} ({path.stat().st_size / 1e6:.1f} MB, classes: {', '.join(ckpt['classes'])})")
    return path
