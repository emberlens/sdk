"""Turn a trained model (classifier or detector) into one .onnx file for the app.

    emberlens export-model MODEL-FOLDER [--out FILE.onnx]

The class names and the crop rule go into the ONNX metadata (sdk/SPEC.md
section 9; detectors section 9.2), so the app needs no hard-coded numbers.
Needs PyTorch and the optional `onnx` library (emberlens install onnx).
"""

from __future__ import annotations

import contextlib
import json
import warnings
from pathlib import Path
from typing import Callable, Dict, Optional

from .command import command
from .errors import EmberlensError
from .extras import require_onnx, require_torch
from .torch_data import IMAGENET_MEAN, IMAGENET_STD

OPSET = 17
RESIZE_MODE = "stretch"  # the crop rule in crops.crop_for_model
DETECTOR_SIZE = 320  # fasterrcnn_mobilenet_v3_large_320_fpn
DETECTOR_MIN_SCORE = 0.5  # the threshold predict uses by default


def build_metadata(ckpt: dict, report: Optional[dict] = None) -> Dict[str, str]:
    """The metadata_props of the file, all as text. See SPEC.md section 9."""
    kind = ckpt.get("kind", "classifier")
    if kind == "detector":
        return _detector_metadata(ckpt, report)
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


def _detector_metadata(ckpt: dict, report: Optional[dict]) -> Dict[str, str]:
    """SPEC.md section 9.2. The network normalises inside: mean 0, std 1."""
    meta = {
        "emberlens_model": "1",
        "kind": "detector",
        "classes": json.dumps(list(ckpt["classes"])[1:], ensure_ascii=False),  # without "background"
        "size": str(DETECTOR_SIZE),
        "input_size": str(DETECTOR_SIZE),
        "crop_pad": "0",
        "resize_mode": RESIZE_MODE,
        "mean": json.dumps([0.0, 0.0, 0.0]),
        "std": json.dumps([1.0, 1.0, 1.0]),
        "labels_mode": str(ckpt.get("target", "label")),
        "min_score": str(DETECTOR_MIN_SCORE),
    }
    recall = ((report or {}).get("test") or {}).get("recall")
    if recall is not None:
        meta["test_recall"] = str(recall)
    return meta


def _detector_for_onnx(model, size: int):
    """Wrap a torchvision detector: 1x3xSxS in, (boxes, labels, scores) out.

    Defined here so torch stays an optional import of this module.
    """
    import torch

    class DetectorForOnnx(torch.nn.Module):
        def __init__(self, model, size: int):
            super().__init__()
            # The input is already SxS: the network must not resize it again.
            model.transform.min_size = (size,)
            model.transform.max_size = size
            self.model = model

        def forward(self, x):
            out = self.model([x[0]])[0]
            # labels: 0 is "background" in the network, the file is 0-based on `classes`.
            return out["boxes"], out["labels"] - 1, out["scores"]

    return DetectorForOnnx(model, size).eval()


def export_model(model_folder, out=None, log: Callable[[str], None] = print) -> Path:
    """Write <out>.onnx from a classifier or detector folder. Returns the file."""
    onnx = require_onnx()
    torch, _ = require_torch()
    from .train import load_model

    folder = Path(model_folder).expanduser()
    model, ckpt = load_model(folder)
    kind = ckpt.get("kind")
    if kind not in ("classifier", "detector"):
        raise EmberlensError(f"export-model works on a model from {command('train', 'classify')} or {command('train', 'detect')}.")
    report = None
    report_file = (folder if folder.is_dir() else folder.parent) / "report.json"
    if report_file.is_file():
        try:
            report = json.loads(report_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            report = None
    path = Path(out).expanduser() if out else Path(f"{folder.stem if folder.is_file() else folder.name}.onnx")
    if kind == "detector":
        _export_detector(torch, model, path)
    else:
        _export_classifier(torch, model, int(ckpt.get("size", 224)), path)
    proto = onnx.load(str(path))
    del proto.metadata_props[:]
    for key, value in build_metadata(ckpt, report).items():
        entry = proto.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.save(proto, str(path))
    classes = ckpt["classes"][1:] if kind == "detector" else ckpt["classes"]
    log(f"Saved {path} ({path.stat().st_size / 1e6:.1f} MB, classes: {', '.join(classes)})")
    return path


@contextlib.contextmanager
def _quiet_export(torch):
    """Hide the warnings the exporter always gives, and only those.

    - The old (TorchScript) exporter is deprecated: we pick it on purpose,
      it writes plain layers and needs no extra library.
    - Tracing torchvision's detector turns Python values (image size, box
      counts) into constants; they are fixed here (1x3x320x320).
    - aten::index warns about negative indices; the detector's indices are
      never negative (onnxruntime gives the same boxes, see the tests).
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", message=r"You are using the legacy TorchScript-based ONNX export", category=DeprecationWarning)
        warnings.filterwarnings("ignore", message=r"The feature will be removed", category=DeprecationWarning)
        warnings.filterwarnings("ignore", message=r"`torch\.jit\.script` is deprecated", category=FutureWarning)
        warnings.filterwarnings("ignore", message=r"To copy construct from a tensor", category=UserWarning)
        warnings.filterwarnings("ignore", message=r"Exporting aten::index operator", category=UserWarning)
        yield


def _export_classifier(torch, model, size: int, path: Path) -> None:
    # Input 1x3xSxS (RGB, mean/std applied by the caller); output: one score
    # per class (logits, in the order of `classes`). The batch can be any size.
    # The old exporter writes plain layers and needs no extra library.
    with _quiet_export(torch):
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


def _export_detector(torch, model, path: Path) -> None:
    # SPEC.md section 9.2. Input 1x3x320x320, RGB 0..1, the whole photo
    # stretched. Outputs boxes (Nx4, input pixels), labels (N, 0-based),
    # scores (N). N can be 0.
    size = DETECTOR_SIZE
    wrapper = _detector_for_onnx(model, size)
    with _quiet_export(torch):
        torch.onnx.export(
            wrapper,
            (torch.zeros(1, 3, size, size),),
            str(path),
            input_names=["input"],
            output_names=["boxes", "labels", "scores"],
            dynamic_axes={"boxes": {0: "n"}, "labels": {0: "n"}, "scores": {0: "n"}},
            opset_version=OPSET,
            dynamo=False,
        )
