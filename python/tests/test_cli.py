import contextlib
import io
import os
import unittest

from emberlens.cli import main

from .helpers import FIXTURES, TempDirCase


class CliTest(TempDirCase):
    def run_cli(self, *args):
        out, err = io.StringIO(), io.StringIO()
        cwd = os.getcwd()
        os.chdir(self.tmp)
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = main(list(args))
        finally:
            os.chdir(cwd)
        return code, out.getvalue(), err.getvalue()

    def test_info(self):
        code, out, _ = self.run_cli("info", str(FIXTURES / "coco_joined"), "--where", "class=Mango")
        self.assertEqual(code, 0)
        self.assertIn("Mango_Ripe_Alphonso", out)
        self.assertNotIn("Strawberry_Ripe", out)

    def test_convert_with_where_labels_and_split(self):
        code, out, _ = self.run_cli(
            "convert", str(FIXTURES / "coco_joined"), "--to", "yolo",
            "--where", "condition=ripe", "--labels", "class", "--split", "50/50/0",
        )
        self.assertEqual(code, 0, out)
        yaml = (self.tmp / "coco_joined-yolo" / "data.yaml").read_text()
        self.assertIn("names: ['Mango', 'Strawberry']", yaml)

    def test_preview_without_browser(self):
        code, _, _ = self.run_cli("preview", str(FIXTURES / "voc_joined"), "--no-open")
        self.assertEqual(code, 0)
        self.assertTrue((self.tmp / "voc_joined-preview.html").is_file())

    def test_friendly_errors(self):
        code, _, err = self.run_cli("info", "nope.zip")
        self.assertEqual(code, 1)
        self.assertIn("Can't find", err)
        self.assertNotIn("Traceback", err)
        code, _, err = self.run_cli("info", str(FIXTURES / "coco_joined"), "--where", "class")
        self.assertIn("name=value", err)
        code, _, err = self.run_cli("convert", str(FIXTURES / "coco_joined"), "--to", "coco", "--split", "8010")
        self.assertIn("80/10/10", err)

    def test_run_a_script(self):
        script = self.tmp / "mine.py"
        script.write_text(
            "import sys, emberlens\nds = emberlens.load(sys.argv[1])\nopen('out.txt', 'w').write(str(len(ds)))\n"
        )
        code, _, _ = self.run_cli("run", str(script), str(FIXTURES / "coco_joined"))
        self.assertEqual(code, 0)
        self.assertEqual((self.tmp / "out.txt").read_text(), "4")

    def test_next_hint_quotes_a_folder_with_a_space(self):
        from unittest import mock

        with mock.patch("emberlens.train.train_classifier"):
            code, out, _ = self.run_cli(
                "train", "classify", str(FIXTURES / "coco_joined"), "--out", "my model"
            )
        self.assertEqual(code, 0)
        self.assertIn("predict 'my model' <photo.jpg>", out)
        self.assertIn("export-model 'my model'", out)


if __name__ == "__main__":
    unittest.main()
