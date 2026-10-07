import json
import unittest
import zipfile

import nhan
from nhan.errors import NhanError
from nhan.writers import dataset_dict

from .helpers import FIXTURES, NAMES, TempDirCase, snapshot


class GoldenTest(unittest.TestCase):
    """Each sample export gives exactly sdk/fixtures/expected/<name>.json."""

    def test_every_fixture_matches_its_golden_file(self):
        for name in NAMES:
            with self.subTest(name):
                ds = nhan.load(FIXTURES / name)
                got = dataset_dict(ds, ds.root)
                got["images"].sort(key=lambda i: i["id"])
                want = json.loads((FIXTURES / "expected" / f"{name}.json").read_text(encoding="utf-8"))
                self.assertEqual(got, want)

    def test_all_formats_agree_on_class_values_and_boxes(self):
        base = snapshot(nhan.load(FIXTURES / "coco_fields"), with_label=False)
        for name in ["coco_joined", "voc_fields", "voc_joined"]:
            with self.subTest(name):
                self.assertEqual(snapshot(nhan.load(FIXTURES / name), with_label=False), base)

    def test_joined_names_are_split_with_the_card(self):
        ds = nhan.load(FIXTURES / "yolo_joined")
        box = [b for b in ds.boxes if b.label == "mango.ripe.alphonso"][0]
        self.assertEqual(box.cls, "mango")
        self.assertEqual(box.attributes, {"condition": "ripe", "type": "alphonso"})

    def test_no_value_means_no_key(self):
        ds = nhan.load(FIXTURES / "coco_joined")
        box = [b for b in ds.boxes if b.label == "Mango_Unripe"][0]
        self.assertIsNone(box.get("Type"))
        self.assertNotIn("None", json.dumps(dataset_dict(ds, ds.root)))

    def test_images_without_boxes_are_kept(self):
        ds = nhan.load(FIXTURES / "coco_joined")
        self.assertEqual([i.id for i in ds if not i.boxes], ["photo-4"])


class ZipTest(TempDirCase):
    def make_zip(self, name, top_folder=""):
        z = self.tmp / f"{name}.zip"
        with zipfile.ZipFile(z, "w") as out:
            for p in (FIXTURES / name).rglob("*"):
                if p.is_file():
                    out.write(p, top_folder + p.relative_to(FIXTURES / name).as_posix())
        return z

    def test_loads_a_zip_and_reuses_the_folder(self):
        z = self.make_zip("voc_joined")
        ds = nhan.load(z)
        self.assertEqual(len(ds), 4)
        self.assertEqual(ds.root, (self.tmp / "voc_joined").resolve())
        again = nhan.load(z)
        self.assertEqual(snapshot(again), snapshot(ds))

    def test_zip_with_a_top_folder(self):
        z = self.make_zip("yolo_joined", top_folder="dataset/")
        self.assertEqual(nhan.load(z).box_count, 4)

    def test_existing_folder_with_same_name_is_not_overwritten(self):
        z = self.make_zip("coco_joined")
        (self.tmp / "coco_joined").mkdir()
        (self.tmp / "coco_joined" / "mine.txt").write_text("keep")
        ds = nhan.load(z)
        self.assertEqual(ds.root.name, "coco_joined (2)")
        self.assertEqual((self.tmp / "coco_joined" / "mine.txt").read_text(), "keep")

    def test_unsafe_zip_path_is_refused(self):
        z = self.tmp / "evil.zip"
        with zipfile.ZipFile(z, "w") as out:
            out.writestr("../evil.txt", "x")
        with self.assertRaises(NhanError):
            nhan.load(z)
        self.assertFalse((self.tmp.parent / "evil.txt").exists())

    def test_clear_errors(self):
        with self.assertRaisesRegex(NhanError, "Can't find"):
            nhan.load(self.tmp / "missing.zip")
        bad = self.tmp / "bad.zip"
        bad.write_text("not a zip")
        with self.assertRaisesRegex(NhanError, "not a valid zip"):
            nhan.load(bad)
        (self.tmp / "empty").mkdir()
        with self.assertRaisesRegex(NhanError, "format"):
            nhan.load(self.tmp / "empty")


class CardVersionTest(TempDirCase):
    def test_a_newer_card_is_refused_clearly(self):
        root = self.copy_fixture("coco_joined")
        card = json.loads((root / "nhan.json").read_text())
        card["nhan_card_version"] = 2
        (root / "nhan.json").write_text(json.dumps(card))
        with self.assertRaisesRegex(NhanError, "newer Nhan app"):
            nhan.load(root)


class OtherToolsTest(TempDirCase):
    """Plain datasets without nhan.json still open."""

    def test_plain_coco_without_a_card(self):
        root = self.copy_fixture("coco_fields")
        (root / "nhan.json").unlink()
        ds = nhan.load(root)
        self.assertEqual(ds.name, "coco_fields")
        self.assertEqual(ds.classes, ["Mango", "Strawberry"])
        self.assertIn("Condition", ds.sub_categories)

    def test_plain_yolo_with_a_block_list(self):
        root = self.copy_fixture("yolo_joined")
        (root / "nhan.json").unlink()
        (root / "data.yaml").write_text(
            "path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnc: 4\nnames:\n"
            "  - a\n  - 'b, c'\n  - \"d\"\n  - e\n"
        )
        ds = nhan.load(root)
        self.assertEqual(sorted({b.label for b in ds.boxes}), ["a", "b, c", "d", "e"])
        self.assertEqual(ds[0].width, 80)


if __name__ == "__main__":
    unittest.main()
