"""The sdk/ folder becomes a public repo: no private paths, no broken links."""
import re
import unittest
from pathlib import Path

SDK = Path(__file__).resolve().parents[2]
SKIP_DIRS = {"fixtures", ".venv", "__pycache__", ".git"}
BANNED = ["docs/DECISIONS", "AGENTS.md", "emberlens_core/", "relay/", "mcp/src"]


def text_files():
    for p in SDK.rglob("*"):
        if not p.is_file() or p.resolve() == Path(__file__).resolve():
            continue
        if SKIP_DIRS & set(p.relative_to(SDK).parts):
            continue
        try:
            yield p, p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue


class PublicRepoTest(unittest.TestCase):
    def test_relative_links_stay_inside_sdk(self):
        bad = []
        for p, text in text_files():
            if p.suffix != ".md":
                continue
            for link in re.findall(r"\]\(([^)\s]+)\)", text):
                if re.match(r"^[a-z][a-z0-9+.-]*:", link) or link.startswith("#"):
                    continue
                target = (p.parent / link.split("#")[0]).resolve()
                try:
                    target.relative_to(SDK)
                except ValueError:
                    bad.append(f"{p.relative_to(SDK)} -> {link} (outside sdk/)")
                    continue
                if not target.exists():
                    bad.append(f"{p.relative_to(SDK)} -> {link} (missing)")
        self.assertEqual(bad, [])

    def test_no_private_paths(self):
        found = []
        for p, text in text_files():
            for word in BANNED:
                if word in text:
                    found.append(f"{p.relative_to(SDK)}: {word}")
        self.assertEqual(found, [])

    def test_colab_link_matches_the_notebook_path(self):
        urls = []
        for p, text in text_files():
            for u in re.findall(r"https://colab\.research\.google\.com/github/[^\s)\"']+", text):
                urls.append((p, u))
        self.assertTrue(urls, "no Colab link found")
        for p, u in urls:
            self.assertTrue(u.endswith("/blob/main/notebooks/train.ipynb"), f"{p}: {u}")
        self.assertTrue((SDK / "notebooks" / "train.ipynb").is_file())

    def test_readme_has_the_colab_badge(self):
        text = (SDK / "README.md").read_text(encoding="utf-8")
        self.assertIn("colab.research.google.com/assets/colab-badge.svg", text)

    def test_publishing_steps_are_in_order(self):
        text = (SDK / "PUBLISHING.md").read_text(encoding="utf-8")
        heads = re.findall(r"^## (\d)\.", text, re.M)
        self.assertEqual(heads, ["1", "2", "3", "4", "5"])
        self.assertIn("emberlens-sdk", text.split("## 2.")[1].split("## 3.")[0])

    def test_licence_at_the_public_root(self):
        lic = SDK / "LICENSE"
        self.assertTrue(lic.is_file())
        self.assertTrue(lic.read_text(encoding="utf-8").startswith("BSD 3-Clause License"))


if __name__ == "__main__":
    unittest.main()
