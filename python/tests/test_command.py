import io
import os
import tokenize
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import emberlens
from emberlens import cli
from emberlens.command import command, program_name
from emberlens.errors import MissingExtra


def strip_comments_and_docstrings(source):
    """Yield the source lines with comments and whole-statement strings removed."""
    lines = source.splitlines()
    drop = set()
    toks = list(tokenize.generate_tokens(io.StringIO(source).readline))
    prev = None
    for tok in toks:
        if tok.type == tokenize.COMMENT:
            lines[tok.start[0] - 1] = lines[tok.start[0] - 1][: tok.start[1]]
        elif tok.type == tokenize.STRING and (
            prev is None or prev.type in (tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.NL)
        ):
            nxt = toks[toks.index(tok) + 1]
            if nxt.type in (tokenize.NEWLINE, tokenize.ENDMARKER):
                drop.update(range(tok.start[0], tok.end[0] + 1))
        if tok.type not in (tokenize.NL, tokenize.COMMENT):
            prev = tok
    return [l for i, l in enumerate(lines, 1) if i not in drop]


class CommandTest(unittest.TestCase):
    def test_launcher_name_wins(self):
        self.assertEqual(program_name("/x/bin/emberlens", {"EMBERLENS_PROG": "./emberlens"}), "./emberlens")

    def test_venv_script_not_on_path_is_shown_with_tilde(self):
        script = str(Path.home() / ".emberlens/bin/emberlens")
        with mock.patch("shutil.which", return_value=None):
            self.assertEqual(program_name(script, {}), "~/.emberlens/bin/emberlens")

    def test_script_on_path_is_shown_by_name(self):
        with mock.patch("shutil.which", return_value="/usr/local/bin/emberlens"):
            self.assertEqual(program_name("/usr/local/bin/emberlens", {}), "emberlens")

    def test_python_dash_m(self):
        self.assertEqual(program_name("/a/emberlens/__main__.py", {}), "python -m emberlens")
        self.assertEqual(program_name("", {}), "python -m emberlens")

    def test_notebook_kernel_is_not_the_command(self):
        self.assertEqual(program_name("/x/ipykernel_launcher.py", {}), "python -m emberlens")
        self.assertEqual(program_name("-c", {}), "python -m emberlens")

    def test_command_joins_parts(self):
        with mock.patch.dict(os.environ, {"EMBERLENS_PROG": "./emberlens"}):
            self.assertEqual(command("install", "onnx"), "./emberlens install onnx")

    def test_missing_extra_names_the_runnable_command(self):
        with mock.patch.dict(os.environ, {"EMBERLENS_PROG": "~/.emberlens/bin/emberlens"}):
            self.assertIn("~/.emberlens/bin/emberlens install onnx", str(MissingExtra("onnx", "onnx")))

    def test_no_message_hard_codes_the_launcher(self):
        # Every "./emberlens" left in emberlens/*.py is in a comment or docstring.
        for f in Path(emberlens.__file__).parent.glob("*.py"):
            for line in strip_comments_and_docstrings(f.read_text()):
                self.assertNotIn("./emberlens", line, f.name)

    def test_help_epilog_uses_the_running_command(self):
        with mock.patch.dict(os.environ, {"EMBERLENS_PROG": "~/x/emberlens"}):
            self.assertIn("~/x/emberlens info", cli.build_parser().format_help())

    def test_selftest_without_tests_folder_says_installed(self):
        out = io.StringIO()
        with mock.patch.object(cli, "_tests_folder", return_value=Path("/nonexistent/tests")), redirect_stdout(out):
            cli.cmd_selftest(None)
        self.assertIn(f"The Emberlens SDK {emberlens.__version__} is installed.", out.getvalue())

    def test_train_detect_prints_export_hint(self):
        import argparse
        args = argparse.Namespace(kind="detect", labels=None, out="m", no_pretrained=True, device=None, epochs=1, export="x.zip")
        out = io.StringIO()
        with mock.patch.object(cli, "_dataset"), mock.patch("emberlens.train.train_detector"), \
                mock.patch.object(cli, "free_path", side_effect=lambda p: p), redirect_stdout(out), \
                mock.patch.dict(os.environ, {"EMBERLENS_PROG": "./emberlens"}):
            cli.cmd_train(args)
        self.assertIn("./emberlens export-model", out.getvalue())


if __name__ == "__main__":
    unittest.main()
