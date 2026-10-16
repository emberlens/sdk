"""Train a model on a Emberlens dataset with PyTorch, on a laptop.

Two kinds:
  classifier  "what is this object?"  ResNet-18 on box crops
  detector    "where are the objects?" Faster R-CNN (MobileNet v3, 320 px)

Both write <out>/model.pt and <out>/report.json. Use predict() to run them.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable, List, Optional

from .errors import EmberlensError
from .model import OTHER, _n
from .extras import require_torch
from .crops import crop_for_model  # noqa: F401  (the shared crop rule; used through CropDataset)
from .torch_data import CropDataset, DetectionDataset, detection_collate, need_two_labels

MODEL_FILE = "model.pt"
DETECTOR_ARCH = "fasterrcnn_mobilenet_v3_large_320_fpn"


def _device(torch, name: Optional[str], allow_mps: bool):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if allow_mps and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _seed(torch, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _out(out_dir) -> Path:
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    return out


def _cpu_state(model) -> dict:
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


# ---------------------------------------------------------------- classifier


def _classifier(tv, n: int, pretrained: bool):
    import torch

    weights = tv.models.ResNet18_Weights.DEFAULT if pretrained else None
    model = tv.models.resnet18(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, n)
    return model


def _accuracy(torch, model, data, device, batch_size) -> Optional[float]:
    if len(data) == 0:
        return None
    model.eval()
    right = 0
    with torch.no_grad():
        for x, y in torch.utils.data.DataLoader(data, batch_size=batch_size):
            right += (model(x.to(device)).argmax(1).cpu() == y).sum().item()
    return right / len(data)


def train_classifier(
    ds,
    out_dir="model",
    target: str = "label",
    epochs: int = 5,
    pretrained: bool = True,
    batch_size: int = 16,
    lr: float = 1e-3,
    size: int = 224,
    padding: float = 0.1,
    seed: int = 42,
    device: Optional[str] = None,
    log: Callable[[str], None] = print,
) -> Path:
    """Teach a model to name the object in each box. Returns the model folder."""
    torch, tv = require_torch()
    data = ds.relabel(target)
    classes = data.labels
    with_other = bool(data.other_boxes())
    if with_other:
        if any(c.strip().lower() == OTHER for c in classes):
            raise EmberlensError(
                f'One of your labels is called "{OTHER}". That name is used for boxes that are none of '
                "your objects. Rename the label, or export without \"Add data for your model\"."
            )
        classes = classes + [OTHER]
    need_two_labels(classes)
    train = CropDataset(data, "train", size, padding, classes=classes, augment=True, with_other=with_other)
    val = CropDataset(data, "val", size, padding, classes=classes, with_other=with_other)
    test = CropDataset(data, "test", size, padding, classes=classes, with_other=with_other)
    if len(train) == 0:
        raise EmberlensError("There are no boxes in the train split. Try --split 80/10/10.")
    _seed(torch, seed)
    dev = _device(torch, device, allow_mps=True)
    log(f"Training a classifier for {_n(len(classes), 'label')} on {_n(len(train), 'box')} ({dev.type}).")
    if pretrained:
        log("Using pretrained ImageNet weights (downloaded once, about 45 MB).")
    model = _classifier(tv, len(classes), pretrained).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    out = _out(out_dir)
    history, best = [], None
    for epoch in range(1, epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for x, y in loader:
            opt.zero_grad()
            loss = loss_fn(model(x.to(dev)), y.to(dev))
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            seen += len(y)
        mean = total / max(seen, 1)
        val_acc = _accuracy(torch, model, val, dev, batch_size)
        history.append({"epoch": epoch, "loss": round(mean, 4), "val_accuracy": val_acc})
        log(f"Epoch {epoch}/{epochs}: loss {mean:.3f}" + (f", val accuracy {val_acc:.0%}" if val_acc is not None else ""))
        score = val_acc if val_acc is not None else -mean
        if best is None or score >= best:
            best = score
            torch.save(
                {
                    "emberlens_model": 1,
                    "kind": "classifier",
                    "arch": "resnet18",
                    "classes": classes,
                    "target": target,
                    "size": size,
                    "padding": padding,
                    "state_dict": _cpu_state(model),
                },
                out / MODEL_FILE,
            )
    model.load_state_dict(torch.load(out / MODEL_FILE, map_location=dev, weights_only=True)["state_dict"])
    test_acc = _accuracy(torch, model, test, dev, batch_size)
    report = {
        "kind": "classifier",
        "classes": classes,
        "boxes": {"train": len(train), "val": len(val), "test": len(test)},
        "history": history,
        "test_accuracy": test_acc,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"Saved {out / MODEL_FILE}" + (f"  (test accuracy {test_acc:.0%})" if test_acc is not None else ""))
    return out


# ------------------------------------------------------------------ detector


def _detector(n_classes: int, pretrained: bool):
    from torchvision.models.detection import (
        FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
        fasterrcnn_mobilenet_v3_large_320_fpn,
    )
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    if pretrained:
        model = fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        )
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_classes)
        return model
    return fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=None, weights_backbone=None, num_classes=n_classes
    )


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _detection_score(torch, model, data, device, threshold=0.5) -> Optional[dict]:
    """Found boxes (same label, overlap ≥ 50 %) and wrong extra boxes."""
    if len(data) == 0:
        return None
    model.eval()
    found = wanted = extra = 0
    with torch.no_grad():
        for i in range(len(data)):
            x, t = data[i]
            pred = model([x.to(device)])[0]
            keep = pred["scores"].cpu() >= threshold
            pb = pred["boxes"].cpu()[keep].tolist()
            pl = pred["labels"].cpu()[keep].tolist()
            used = set()
            for box, label in zip(t["boxes"].tolist(), t["labels"].tolist()):
                wanted += 1
                for j, (b2, l2) in enumerate(zip(pb, pl)):
                    if j not in used and l2 == label and _iou(box, b2) >= 0.5:
                        used.add(j)
                        found += 1
                        break
            extra += len(pb) - len(used)
    return {"found": found, "boxes": wanted, "extra": extra, "recall": found / wanted if wanted else None}


def train_detector(
    ds,
    out_dir="model",
    target: str = "label",
    epochs: int = 10,
    pretrained: bool = True,
    batch_size: int = 2,
    lr: float = 0.01,
    seed: int = 42,
    device: Optional[str] = None,
    log: Callable[[str], None] = print,
) -> Path:
    """Teach a model to find and name objects. Returns the model folder."""
    torch, _ = require_torch()
    data = ds.relabel(target)
    if not data.labels:
        raise EmberlensError("There are no boxes to learn from.")
    train = DetectionDataset(data, "train", classes=data.labels)
    val = DetectionDataset(data, "val", classes=data.labels)
    test = DetectionDataset(data, "test", classes=data.labels)
    if len(train) == 0:
        raise EmberlensError("There are no photos in the train split. Try --split 80/10/10.")
    _seed(torch, seed)
    # Apple MPS lacks some detection operations: use CPU there.
    dev = _device(torch, device, allow_mps=False)
    classes = train.classes
    log(f"Training a detector for {_n(len(classes) - 1, 'label')} on {_n(len(train), 'photo')} ({dev.type}).")
    if pretrained:
        log("Using pretrained COCO weights (downloaded once, about 75 MB).")
    model = _detector(len(classes), pretrained).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)
    loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=detection_collate)
    out = _out(out_dir)
    history, best = [], None
    for epoch in range(1, epochs + 1):
        model.train()
        total, steps = 0.0, 0
        for images, targets in loader:
            images = [i.to(dev) for i in images]
            targets = [{k: (v.to(dev) if hasattr(v, "to") else v) for k, v in t.items()} for t in targets]
            losses = model(images, targets)
            loss = sum(losses.values())
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            steps += 1
        mean = total / max(steps, 1)
        score = _detection_score(torch, model, val, dev)
        history.append({"epoch": epoch, "loss": round(mean, 4), "val": score})
        msg = f"Epoch {epoch}/{epochs}: loss {mean:.3f}"
        if score and score["boxes"]:
            msg += f", val found {score['found']}/{score['boxes']} boxes, {score['extra']} extra"
        log(msg)
        rank = (score["recall"] or 0) - 0.01 * score["extra"] if score else -mean
        if best is None or rank >= best:
            best = rank
            torch.save(
                {
                    "emberlens_model": 1,
                    "kind": "detector",
                    "arch": DETECTOR_ARCH,
                    "classes": classes,
                    "target": target,
                    "state_dict": _cpu_state(model),
                },
                out / MODEL_FILE,
            )
    model.load_state_dict(torch.load(out / MODEL_FILE, map_location=dev, weights_only=True)["state_dict"])
    test_score = _detection_score(torch, model, test, dev)
    report = {
        "kind": "detector",
        "classes": classes[1:],
        "photos": {"train": len(train), "val": len(val), "test": len(test)},
        "history": history,
        "test": test_score,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"Saved {out / MODEL_FILE}")
    return out


# ------------------------------------------------------------------- predict


def load_model(path):
    """(model, checkpoint) from a model folder or model.pt file. No download."""
    torch, tv = require_torch()
    p = Path(path).expanduser()
    if p.is_dir():
        p = p / MODEL_FILE
    if not p.is_file():
        raise EmberlensError(f"Can't find a model at {p}. Train one first with: ./emberlens train")
    ckpt = torch.load(p, map_location="cpu", weights_only=True)
    if ckpt.get("kind") == "classifier":
        model = _classifier(tv, len(ckpt["classes"]), pretrained=False)
    elif ckpt.get("kind") == "detector":
        model = _detector(len(ckpt["classes"]), pretrained=False)
    else:
        raise EmberlensError(f"{p} is not a Emberlens model.")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def predict(model_path, images: List, threshold: float = 0.5) -> List[dict]:
    """Run a trained model on photo files.

    classifier: [{"file", "label", "score"}]
    detector:   [{"file", "width", "height", "boxes": [{"label", "score", "x", "y", "w", "h"}]}]
                (x, y, w, h normalised 0..1, like emberlens.Box)
    """
    torch, _ = require_torch()
    from torchvision import transforms as T

    from .extras import require_pillow
    from .torch_data import IMAGENET_MEAN, IMAGENET_STD

    PIL_Image = require_pillow()
    model, ckpt = load_model(model_path)
    classes = ckpt["classes"]
    results = []
    with torch.no_grad():
        for f in images:
            photo = PIL_Image.open(f).convert("RGB")
            if ckpt["kind"] == "classifier":
                size = ckpt.get("size", 224)
                x = T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])(photo)
                probs = torch.softmax(model(x.unsqueeze(0))[0], 0)
                k = int(probs.argmax())
                results.append({"file": str(f), "label": classes[k], "score": round(float(probs[k]), 4)})
            else:
                W, H = photo.size
                out = model([T.ToTensor()(photo)])[0]
                boxes = []
                for b, label, score in zip(out["boxes"].tolist(), out["labels"].tolist(), out["scores"].tolist()):
                    if score >= threshold:
                        boxes.append(
                            {
                                "label": classes[label],
                                "score": round(score, 4),
                                "x": b[0] / W,
                                "y": b[1] / H,
                                "w": (b[2] - b[0]) / W,
                                "h": (b[3] - b[1]) / H,
                            }
                        )
                results.append({"file": str(f), "width": W, "height": H, "boxes": boxes})
    return results
