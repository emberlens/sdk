"""Task 5: the Colab and Kaggle training notebook and its helper module."""

import json
import re
import shutil
import unittest
from pathlib import Path

import emberlens
from emberlens.errors import EmberlensError
from emberlens.notebook import find_export, model_out_path, run

from .helpers import FIXTURES, TempDirCase
from .test_chain import ONNX, TORCH

NOTEBOOK = Path(__file__).resolve().parents[2] / "notebooks" / "train.ipynb"


class FindExportTest(unittest.TestCase):
    def test_empty_and_quotes(self):
        for text in ("", '  ""  ', "  ''  "):
            with self.assertRaisesRegex(EmberlensError, "Paste the path"):
                find_export(text, exists=lambda p: True)

    def test_share_link(self):
        with self.assertRaisesRegex(EmberlensError, "not a link"):
            find_export("https://drive.google.com/file/d/x", exists=lambda p: True)

    def test_my_drive_with_space(self):
        want = Path("/content/drive/MyDrive/Fruit stand.zip")
        got = find_export("/content/drive/My Drive/Fruit stand.zip", exists=lambda p: p == want)
        self.assertEqual(got, want)

    def test_not_found(self):
        with self.assertRaisesRegex(EmberlensError, "Can't find /content/drive/MyDrive/a.zip"):
            find_export("/content/drive/MyDrive/a.zip", exists=lambda p: False)

    def test_not_a_zip(self):
        with self.assertRaisesRegex(EmberlensError, "photo.jpg is not an export"):
            find_export("/content/drive/MyDrive/photo.jpg", exists=lambda p: True)

    def test_quoted_path_with_spaces(self):
        got = find_export('"/content/drive/MyDrive/Fruit stand.zip"', exists=lambda p: True)
        self.assertEqual(got, Path("/content/drive/MyDrive/Fruit stand.zip"))

    def test_a_folder_is_accepted(self):
        folder = Path(__file__).resolve().parent
        self.assertEqual(find_export(str(folder)), folder)


class ModelOutPathTest(TempDirCase):
    def test_next_to_the_zip(self):
        self.assertEqual(model_out_path(self.tmp / "a.zip", "classify"), self.tmp / "a-classify.onnx")

    def test_second_run_does_not_overwrite(self):
        (self.tmp / "a-classify.onnx").write_bytes(b"x")
        self.assertEqual(model_out_path(self.tmp / "a.zip", "classify"), self.tmp / "a-classify-2.onnx")

    def test_kaggle_goes_to_working(self):
        got = model_out_path(Path("/kaggle/input/x/a.zip"), "detect")
        self.assertEqual(got.parent, Path("/kaggle/working"))
        self.assertEqual(got.name, "a-detect.onnx")


class NotebookFileTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.code = [c for c in cls.nb["cells"] if c["cell_type"] == "code"]

    def test_is_clean_nbformat_4(self):
        self.assertEqual(self.nb["nbformat"], 4)
        self.assertEqual(self.nb["nbformat_minor"], 5)
        self.assertEqual(self.nb["metadata"]["kernelspec"]["name"], "python3")
        self.assertEqual(self.nb["metadata"]["accelerator"], "GPU")
        ids = [c["id"] for c in self.nb["cells"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids), 5)
        for c in self.code:
            self.assertEqual(c["outputs"], [])
            self.assertIsNone(c["execution_count"])

    def test_pins_the_package_version(self):
        src = "".join(self.nb["cells"][1]["source"])
        self.assertIn(f"emberlens-sdk[train,onnx]=={emberlens.__version__}", src)

    def test_one_edit_cell(self):
        editing = [c for c in self.code if re.search(r"^EXPORT_PATH\s*=", "".join(c["source"]), re.M)]
        self.assertEqual(len(editing), 1)
        src = "".join(editing[0]["source"])
        for name in ("KIND", "LABELS"):
            self.assertRegex(src, rf"(?m)^{name}\s*=")

    def test_code_cells_compile(self):
        for i, c in enumerate(self.code):
            lines = [l for l in "".join(c["source"]).splitlines() if not l.lstrip().startswith(("%", "!"))]
            compile("\n".join(lines), f"cell-{i}", "exec")

    def test_code_cells_have_no_urls(self):
        for c in self.code:
            self.assertNotIn("http", "".join(c["source"]))


@TORCH
@ONNX
class RunTest(TempDirCase):
    def test_classify_end_to_end(self):
        src = self.tmp / "pack"
        shutil.copytree(FIXTURES / "coco_joined", src)
        zipped = Path(shutil.make_archive(str(self.tmp / "stand"), "zip", src))
        shutil.rmtree(src)
        lines = []
        out = run(str(zipped), "classify", "class", epochs=1, pretrained=False, log=lines.append)
        self.assertEqual(out, self.tmp / "stand-classify.onnx")
        self.assertTrue(out.is_file())
        self.assertIn(f"Saved {out}. Send it to your iPhone, then add it in Project settings > Your model.", lines)

    def test_wrong_kind(self):
        with self.assertRaisesRegex(EmberlensError, "classify or detect"):
            run("/nowhere/a.zip", "segment")
