import re
import sys
import tempfile
import unittest
from pathlib import Path

import emberlens
from emberlens import cli
from emberlens.cli import build_parser

ROOT = Path(__file__).resolve().parent.parent

if sys.version_info >= (3, 11):
    import tomllib

    PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())


@unittest.skipIf(sys.version_info < (3, 11), "tomllib needs Python 3.11 or newer")
class PackageTests(unittest.TestCase):
    def test_versions_match(self):
        self.assertEqual(PYPROJECT["project"]["version"], emberlens.__version__)
        self.assertEqual(emberlens.__version__, "0.2.0")

    def test_name_script_licence(self):
        p = PYPROJECT["project"]
        self.assertEqual(p["name"], "emberlens-sdk")
        self.assertEqual(p["scripts"], {"emberlens": "emberlens.cli:main"})
        self.assertEqual(p["license"], "BSD-3-Clause")
        self.assertEqual(p["dependencies"], [])
        self.assertEqual(set(p["optional-dependencies"]), {"images", "train", "onnx"})
        self.assertTrue((ROOT / "LICENSE").is_file() and (ROOT / "README.md").is_file())

    def test_connector_commands_parse(self):
        for argv in (["info", "x.zip"], ["install", "train"], ["train", "classify", "x.zip"],
                     ["train", "detect", "x.zip"], ["install", "onnx"], ["export-model", "m"]):
            build_parser().parse_args(argv)  # no SystemExit

    def test_pypi_readme_has_only_absolute_links(self):
        for target in re.findall(r"\]\(([^)]+)\)", (ROOT / "README.md").read_text()):
            self.assertTrue(target.startswith("https://"), target)

    def test_readme_is_short(self):
        self.assertLessEqual(len((ROOT / "README.md").read_text().splitlines()), 40)


class TestsFolderMarkerTests(unittest.TestCase):
    def test_foreign_tests_folder_is_not_the_sdk_tests(self):
        with tempfile.TemporaryDirectory() as d:
            tests = Path(d) / "tests"
            tests.mkdir()
            self.assertFalse(cli._is_sdk_tests(tests))
            (tests / "test_command.py").write_text("")
            (tests / "helpers.py").write_text("")
            self.assertTrue(cli._is_sdk_tests(tests))

    def test_selftest_ignores_foreign_tests_folder(self):
        from contextlib import redirect_stdout
        from io import StringIO
        from unittest import mock

        with tempfile.TemporaryDirectory() as d:
            tests = Path(d) / "tests"
            tests.mkdir()
            out = StringIO()
            with mock.patch.object(cli, "_tests_folder", return_value=tests), redirect_stdout(out):
                cli.cmd_selftest(None)
            self.assertIn("The full checks come with the source code.", out.getvalue())


if __name__ == "__main__":
    unittest.main()
