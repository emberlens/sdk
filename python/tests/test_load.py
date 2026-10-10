import json
import unittest
import zipfile

import emberlens
from emberlens.errors import EmberlensError
from emberlens.writers import dataset_dict

from .helpers import FIXTURES, NAMES, TempDirCase, snapshot


class GoldenTest(unittest.TestCase):
    """Each sample export gives exactly sdk/fixtures/expected/<name>.json."""

    def test_every_fixture_matches_its_golden_file(self):
        for name in NAMES:
            with self.subTest(name):
                ds = emberlens.load(FIXTURES / name)
                got = dataset_dict(ds, ds.root)
                got["images"].sort(key=lambda i: i["id"])
                want = json.loads((FIXTURES / "expected" / f"{name}.json").read_text(encoding="utf-8"))
                self.assertEqual(got, want)

    def test_all_formats_agree_on_class_values_and_boxes(self):
        base = snapshot(emberlens.load(FIXTURES / "coco_fields"), with_label=False)
        for name in ["coco_joined", "voc_fields", "voc_joined"]:
            with self.subTest(name):
                self.assertEqual(snapshot(emberlens.load(FIXTURES / name), with_label=False), base)

    def test_joined_names_are_split_with_the_card(self):
        ds = emberlens.load(FIXTURES / "yolo_joined")
        box = [b for b in ds.boxes if b.label == "mango.ripe.alphonso"][0]
        self.assertEqual(box.cls, "mango")
        self.assertEqual(box.attributes, {"condition": "ripe", "type": "alphonso"})

    def test_no_value_means_no_key(self):
        ds = emberlens.load(FIXTURES / "coco_joined")
        box = [b for b in ds.boxes if b.label == "Mango_Unripe"][0]
        self.assertIsNone(box.get("Type"))
        self.assertNotIn("None", json.dumps(dataset_dict(ds, ds.root)))

    def test_images_without_boxes_are_kept(self):
        ds = emberlens.load(FIXTURES / "coco_joined")
        self.assertEqual([i.id for i in ds if not i.boxes], ["photo-4"])


class ResizedExportTest(unittest.TestCase):
    """The app shrank the photos (longest side 40 px): every label shrinks
    with its photo, so the SDK reads the same positions as at full size."""

    def test_coco_labels_keep_their_place_when_photos_shrink(self):
        full = emberlens.load(FIXTURES / "coco_joined")
        small = emberlens.load(FIXTURES / "coco_resized")
        self.assertEqual(snapshot(small), snapshot(full))
        polys = lambda d: [[(round(x, 4), round(y, 4)) for x, y in b.polygon] for b in d.boxes if b.polygon]
        self.assertEqual(polys(small), polys(full))

    def test_photos_really_are_smaller_and_sizes_match_the_files(self):
        from emberlens.imagesize import image_size

        full = {i.id: i for i in emberlens.load(FIXTURES / "coco_joined")}
        for img in emberlens.load(FIXTURES / "coco_resized"):
            big = full[img.id]
            scale = min(1.0, 40 / max(big.width, big.height))
            with self.subTest(img.id):
                self.assertEqual((img.width, img.height), (round(big.width * scale), round(big.height * scale)))
                self.assertEqual(image_size(img.path), (img.width, img.height))

    def test_pixel_boxes_shrink_by_the_photo_scale(self):
        full = {i.id: i for i in emberlens.load(FIXTURES / "coco_joined")}
        for img in emberlens.load(FIXTURES / "coco_resized"):
            big = full[img.id]
            for a, b in zip(img.boxes, big.boxes):
                small_px = a.pixels(img.width, img.height)
                big_px = b.pixels(big.width, big.height)
                for s, l in zip(small_px, big_px):
                    self.assertAlmostEqual(s, l * img.width / big.width, delta=0.01)

    def test_voc_is_within_one_pixel(self):
        full = emberlens.load(FIXTURES / "voc_joined")
        small = emberlens.load(FIXTURES / "voc_resized")
        for a, b in zip(sorted(small, key=lambda i: i.id), sorted(full, key=lambda i: i.id)):
            for ba, bb in zip(a.boxes, b.boxes):
                for va, vb in ((ba.x, bb.x), (ba.y, bb.y), (ba.w, bb.w), (ba.h, bb.h)):
                    self.assertAlmostEqual(va, vb, delta=1.0 / min(a.width, a.height) + 1e-9)


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
        ds = emberlens.load(z)
        self.assertEqual(len(ds), 4)
        self.assertEqual(ds.root, (self.tmp / "voc_joined").resolve())
        again = emberlens.load(z)
        self.assertEqual(snapshot(again), snapshot(ds))

    def test_zip_with_a_top_folder(self):
        z = self.make_zip("yolo_joined", top_folder="dataset/")
        self.assertEqual(emberlens.load(z).box_count, 4)

    def test_existing_folder_with_same_name_is_not_overwritten(self):
        z = self.make_zip("coco_joined")
        (self.tmp / "coco_joined").mkdir()
        (self.tmp / "coco_joined" / "mine.txt").write_text("keep")
        ds = emberlens.load(z)
        self.assertEqual(ds.root.name, "coco_joined (2)")
        self.assertEqual((self.tmp / "coco_joined" / "mine.txt").read_text(), "keep")

    def test_unsafe_zip_path_is_refused(self):
        z = self.tmp / "evil.zip"
        with zipfile.ZipFile(z, "w") as out:
            out.writestr("../evil.txt", "x")
        with self.assertRaises(EmberlensError):
            emberlens.load(z)
        self.assertFalse((self.tmp.parent / "evil.txt").exists())

    def test_clear_errors(self):
        with self.assertRaisesRegex(EmberlensError, "Can't find"):
            emberlens.load(self.tmp / "missing.zip")
        bad = self.tmp / "bad.zip"
        bad.write_text("not a zip")
        with self.assertRaisesRegex(EmberlensError, "not a valid zip"):
            emberlens.load(bad)
        (self.tmp / "empty").mkdir()
        with self.assertRaisesRegex(EmberlensError, "format"):
            emberlens.load(self.tmp / "empty")


class CardVersionTest(TempDirCase):
    def test_a_newer_card_is_refused_clearly(self):
        root = self.copy_fixture("coco_joined")
        card = json.loads((root / "emberlens.json").read_text())
        card["emberlens_card_version"] = 2
        (root / "emberlens.json").write_text(json.dumps(card))
        with self.assertRaisesRegex(EmberlensError, "newer Emberlens app"):
            emberlens.load(root)


class OtherToolsTest(TempDirCase):
    """Plain datasets without emberlens.json still open."""

    def test_plain_coco_without_a_card(self):
        root = self.copy_fixture("coco_fields")
        (root / "emberlens.json").unlink()
        ds = emberlens.load(root)
        self.assertEqual(ds.name, "coco_fields")
        self.assertEqual(ds.classes, ["Mango", "Strawberry"])
        self.assertIn("Condition", ds.sub_categories)

    def test_plain_yolo_with_a_block_list(self):
        root = self.copy_fixture("yolo_joined")
        (root / "emberlens.json").unlink()
        (root / "data.yaml").write_text(
            "path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnc: 4\nnames:\n"
            "  - a\n  - 'b, c'\n  - \"d\"\n  - e\n"
        )
        ds = emberlens.load(root)
        self.assertEqual(sorted({b.label for b in ds.boxes}), ["a", "b, c", "d", "e"])
        self.assertEqual(ds[0].width, 80)


if __name__ == "__main__":
    unittest.main()
