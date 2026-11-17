"""The `emberlens` command. Run `emberlens help` for the list."""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path
from typing import List, Optional

from . import __version__
from .command import command, program_name
from .errors import MissingExtra, EmberlensError
from .load import load
from .paths import export_stem, free_path

_EXAMPLES = """
examples:
  {prog} info   fruit-stand-coco.zip
  {prog} recipe "Fruit survey.emberlens-project.json"
  {prog} preview fruit-stand-coco.zip
  {prog} convert fruit-stand-coco.zip --to yolo
  {prog} convert fruit-stand-coco.zip --to yolo --where class=Mango --labels condition
  {prog} crops  fruit-stand-coco.zip
  {prog} train classify fruit-stand-coco.zip --labels condition
  {prog} train detect   fruit-stand-coco.zip
  {prog} export-model fruit-stand-coco-classify-model
  {prog} predict fruit-stand-coco-classify-model photo.jpg
"""


def examples() -> str:
    return _EXAMPLES.format(prog=program_name())


def _parse_split(text: str):
    try:
        parts = [float(x) for x in text.replace(",", "/").split("/")]
    except ValueError:
        parts = []
    if len(parts) != 3 or sum(parts) <= 0 or min(parts) < 0:
        raise EmberlensError('Write the split as three numbers, like --split 80/10/10.')
    return parts


def _dataset(args):
    ds = load(args.export)
    for cond in args.where or []:
        if "=" not in cond:
            raise EmberlensError(f'Write --where as name=value, like --where class=Mango (got "{cond}").')
        key, value = (s.strip() for s in cond.split("=", 1))
        val = None if value.lower() in ("none", "null", "") else value
        if key.lower() == "split":
            ds = ds.split(value)
        elif key.lower() in ("class", "cls"):
            ds = ds.filter(cls=value, keep_other_boxes=False)
        else:
            ds = ds.filter(**{key: val})
        if len(ds) == 0:
            raise EmberlensError(f"No boxes match {cond}.")
    if getattr(args, "split", None):
        ds = ds.resplit(*_parse_split(args.split))
    if getattr(args, "labels", None) and args.labels != "label":
        ds = ds.relabel(args.labels)
    return ds


# ------------------------------------------------------------------ commands


def cmd_info(args):
    ds = _dataset(args)
    print(ds.summary())
    print("\nLabels in the files:")
    for name, n in ds.counts("label").items():
        print(f"    {name:<32} {n}")


def cmd_recipe(args):
    from .recipe import describe, load_recipe

    print(describe(load_recipe(args.file)))


def cmd_preview(args):
    ds = _dataset(args)
    out = Path(args.out) if args.out else free_path(Path(f"{export_stem(args.export)}-preview.html"))
    path = ds.preview(out)
    print(f"Wrote {path}")
    if not args.no_open:
        webbrowser.open(path.as_uri())


def cmd_convert(args):
    ds = _dataset(args)
    fmt = args.to.lower()
    if fmt == "json":
        out = Path(args.out) if args.out else free_path(Path(f"{export_stem(args.export)}.json"))
        print(f"Wrote {ds.to_json(out)}")
        return
    out = Path(args.out) if args.out else free_path(Path(f"{export_stem(args.export)}-{fmt}"))
    path = getattr(ds, f"to_{fmt}")(out)
    print(f"Wrote {len(ds)} photos, {ds.box_count} boxes to {path}")


def cmd_crops(args):
    ds = _dataset(args)
    out = Path(args.out) if args.out else free_path(Path(f"{export_stem(args.export)}-crops"))
    ds.crops(out, padding=args.pad, size=args.size)


def cmd_train(args):
    from .train import train_classifier, train_detector

    ds = _dataset(argparse.Namespace(**{**vars(args), "labels": None}))
    target = args.labels or "label"
    out = Path(args.out) if args.out else free_path(Path(f"{export_stem(args.export)}-{args.kind}-model"))
    common = dict(target=target, pretrained=not args.no_pretrained, device=args.device)
    if args.kind == "classify":
        train_classifier(ds, out, epochs=args.epochs or 5, **common)
    else:
        train_detector(ds, out, epochs=args.epochs or 10, **common)
    print(f"Next: {command('predict', str(out), '<photo.jpg>')}")
    print(f"Or make the file for the app: {command('export-model', str(out))}")


def cmd_export_model(args):
    from .export_model import export_model

    model = Path(args.model).expanduser()
    out = Path(args.out) if args.out else free_path(Path(f"{model.name or 'model'}.onnx"))
    path = export_model(model, out)
    print(f"Send {path.name} to your iPhone (AirDrop or Files), then add it in Project settings > Your model.")


def cmd_predict(args):
    from .model import Box, Dataset, Image
    from .train import predict

    results = predict(args.model, args.images, threshold=args.threshold)
    out = Path(args.out) if args.out else free_path(Path("predictions.json"))
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for r in results:
        if "label" in r:
            print(f"{r['file']}: {r['label']} ({r['score']:.0%})")
        else:
            shown = r["boxes"][:10]
            found = ", ".join(f"{b['label']} {b['score']:.0%}" for b in shown) or "nothing found"
            if len(r["boxes"]) > 10:
                found += f" (+{len(r['boxes']) - 10} more)"
            print(f"{r['file']}: {found}")
    print(f"Wrote {out}")
    if results and "boxes" in results[0]:
        images = [
            Image(
                id=Path(r["file"]).stem,
                path=Path(r["file"]),
                width=r["width"],
                height=r["height"],
                split="test",
                boxes=[Box(label=b["label"], cls=b["label"], x=b["x"], y=b["y"], w=b["w"], h=b["h"]) for b in r["boxes"]],
            )
            for r in results
        ]
        page = Dataset(name="Predictions", format="model", root=Path("."), images=images).preview(out.with_suffix(".html"))
        print(f"Wrote {page}")
        if not args.no_open:
            webbrowser.open(page.as_uri())


def cmd_run(args):
    """Run your own Python script, with `import emberlens` working."""
    import runpy

    script = Path(args.script)
    if not script.is_file():
        raise EmberlensError(f"Can't find {script}.")
    sys.argv = [str(script), *args.args]
    sys.path.insert(0, str(script.resolve().parent))
    runpy.run_path(str(script), run_name="__main__")


def _tests_folder() -> Path:
    return Path(__file__).resolve().parent.parent / "tests"


def cmd_selftest(args):
    """Check that the SDK works on this computer."""
    import unittest

    tests = _tests_folder()
    if not tests.is_dir():
        print(f"The Emberlens SDK {__version__} is installed. The full checks come with the source code.")
        return
    suite = unittest.defaultTestLoader.discover(str(tests), top_level_dir=str(tests.parent))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful():
        raise EmberlensError("Some checks failed. See above.")
    print("The Emberlens SDK works on this computer.")


def cmd_install(args):
    from .extras import install

    install(args.extra)


# ---------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=program_name(),
        description="Use your Emberlens exports for training. EXPORT is the .zip from the app (or its folder).",
        epilog=examples(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"emberlens-sdk {__version__}")
    sub = p.add_subparsers(dest="command", metavar="COMMAND")

    def with_export(sp, labels=True, split=True):
        sp.add_argument("export", help="the .zip from the app, or its folder")
        sp.add_argument("--where", action="append", metavar="NAME=VALUE", help="keep only matching boxes, e.g. class=Mango or condition=Unripe (repeatable)")
        if labels:
            sp.add_argument("--labels", metavar="WHAT", help="label, class, joined, or a sub-category name (default: label)")
        if split:
            sp.add_argument("--split", metavar="T/V/T", help="new train/val/test split, e.g. 80/10/10")
        return sp

    s = with_export(sub.add_parser("info", help="show what is in an export"), labels=False, split=False)
    s.set_defaults(func=cmd_info)

    s = sub.add_parser("recipe", help="show the labels in a project file")
    s.add_argument("file", help="an .emberlens-project.json or .emberlens-invite file")
    s.set_defaults(func=cmd_recipe)

    s = with_export(sub.add_parser("preview", help="make an HTML page with every photo and box"), labels=False, split=False)
    s.add_argument("--out", help="the .html file to write")
    s.add_argument("--no-open", action="store_true", help="do not open the browser")
    s.set_defaults(func=cmd_preview)

    s = with_export(sub.add_parser("convert", help="write the dataset as coco, yolo, voc or json"))
    s.add_argument("--to", required=True, choices=["coco", "yolo", "voc", "json"], help="the new format")
    s.add_argument("--out", help="the folder (or .json file) to write")
    s.set_defaults(func=cmd_convert)

    s = with_export(sub.add_parser("crops", help="save each box as its own image, one folder per label (needs Pillow)"), split=False)
    s.add_argument("--out", help="the folder to write")
    s.add_argument("--pad", type=float, default=0.0, help="extra space around each box, e.g. 0.1 = 10%%")
    s.add_argument("--size", type=int, help="resize every crop to SIZE x SIZE pixels")
    s.set_defaults(func=cmd_crops)

    s = sub.add_parser("train", help="train a model with PyTorch (needs PyTorch)")
    s.add_argument("kind", choices=["classify", "detect"], help="classify = name the object in a box; detect = find objects in a photo")
    with_export(s)
    s.add_argument("--epochs", type=int, help="training rounds (default: 5 classify, 10 detect)")
    s.add_argument("--out", help="the model folder to write")
    s.add_argument("--no-pretrained", action="store_true", help="start from scratch (no download; needs much more data)")
    s.add_argument("--device", help="cpu, cuda or mps (default: the fastest found)")
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("predict", help="run a trained model on photos")
    s.add_argument("model", help=f"the model folder from {command('train')}")
    s.add_argument("images", nargs="+", help="one or more photos")
    s.add_argument("--threshold", type=float, default=0.5, help="detector: lowest score to keep (0..1)")
    s.add_argument("--out", help="the .json file to write")
    s.add_argument("--no-open", action="store_true", help="do not open the browser")
    s.set_defaults(func=cmd_predict)

    s = sub.add_parser("export-model", help="turn a trained classifier into one .onnx file for the app (needs onnx)")
    s.add_argument("model", help=f"the model folder from {command('train', 'classify')}")
    s.add_argument("--out", help="the .onnx file to write")
    s.set_defaults(func=cmd_export_model)

    s = sub.add_parser("install", help="install optional parts: images (Pillow), train (PyTorch) or onnx")
    s.add_argument("extra", choices=["images", "train", "onnx"])
    s.set_defaults(func=cmd_install)

    s = sub.add_parser("run", help="run your own Python script that uses `import emberlens`")
    s.add_argument("script", help="your .py file")
    s.add_argument("args", nargs=argparse.REMAINDER, help="arguments for your script")
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("selftest", help="check that the SDK works on this computer")
    s.set_defaults(func=cmd_selftest)

    sub.add_parser("help", help="show this help")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    try:
        args.func(args)
        return 0
    except MissingExtra as e:
        if sys.stdin.isatty():
            answer = input(f"{e.packages} is needed for this. Install it now? [Y/n] ").strip().lower()
            if answer in ("", "y", "yes"):
                from .extras import install

                try:
                    install(e.extra)
                except EmberlensError as err:
                    print(f"Error: {err}", file=sys.stderr)
                    return 1
                print("Run your command again.")
                return 0
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except EmberlensError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Stopped.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
