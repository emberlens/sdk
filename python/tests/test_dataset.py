import unittest

import nhan
from nhan.errors import NhanError

from .helpers import FIXTURES, NAMES, TempDirCase, snapshot


class ChoosingTest(unittest.TestCase):
    def setUp(self):
        self.ds = nhan.load(FIXTURES / "coco_joined")

    def test_filter_by_class_and_value_ignores_case(self):
        out = self.ds.filter(cls="mango", condition="unripe")
        self.assertEqual([b.label for b in out.boxes], ["Mango_Unripe"])

    def test_filter_none_means_no_value(self):
        out = self.ds.filter(cls="Strawberry", condition=None)
        self.assertEqual([b.label for b in out.boxes], ["Strawberry"])

    def test_filter_keeps_other_boxes_when_asked(self):
        out = self.ds.filter(cls="Mango", type="Alphonso", keep_other_boxes=True)
        self.assertEqual(len(out[0].boxes), 2)

    def test_filter_does_not_change_the_original(self):
        self.ds.filter(cls="Mango")
        self.assertEqual(self.ds.box_count, 4)

    def test_split_and_counts(self):
        self.assertEqual(len(self.ds.split("train")), 2)
        self.assertEqual(self.ds.counts("class"), {"Mango": 2, "Strawberry": 2})
        self.assertEqual(self.ds.counts("condition")["(none)"], 1)
        with self.assertRaises(NhanError):
            self.ds.split("training")

    def test_resplit_is_repeatable(self):
        a = self.ds.resplit(0.5, 0.25, 0.25, seed=7)
        b = self.ds.resplit(0.5, 0.25, 0.25, seed=7)
        self.assertEqual([i.split for i in a], [i.split for i in b])
        self.assertEqual(sorted(i.split for i in a), ["test", "train", "train", "val"])
        self.assertEqual([i.id for i in a], [i.id for i in self.ds])

    def test_relabel(self):
        self.assertEqual(sorted(self.ds.relabel("class").labels), ["Mango", "Strawberry"])
        self.assertEqual(sorted(self.ds.relabel("condition").labels), ["Ripe", "Unripe"])
        self.assertEqual(self.ds.relabel("condition").box_count, 3)
        self.assertIn("Mango.Ripe.Alphonso", self.ds.relabel("joined", separator=".").labels)
        with self.assertRaises(NhanError):
            self.ds.relabel("colour")

    def test_summary_reads_well(self):
        text = self.ds.summary()
        self.assertIn("4 images, 4 boxes", text)
        self.assertIn("Condition:", text)
        self.assertIn("1 image, 1 box", nhan.load(FIXTURES / "coco_joined").filter(cls="Mango", type="Alphonso").summary())

    def test_pixels(self):
        img = [i for i in self.ds if i.id == "photo-1"][0]
        self.assertEqual(img.boxes[0].pixels(img.width, img.height), (8.0, 6.0, 40.0, 36.0))


class WriteTest(TempDirCase):
    def test_round_trip_in_every_format(self):
        for name in NAMES:
            ds = nhan.load(FIXTURES / name)
            for fmt in ("coco", "yolo", "voc"):
                with self.subTest(f"{name} -> {fmt}"):
                    back = nhan.load(getattr(ds, f"to_{fmt}")(self.tmp / f"{name}-{fmt}"))
                    # YOLO joins separate fields into the name; nothing is lost.
                    self.assertEqual(snapshot(back, with_label=False), snapshot(ds, with_label=False))

    def test_fields_to_yolo_gives_the_app_names(self):
        back = nhan.load(nhan.load(FIXTURES / "coco_fields").to_yolo(self.tmp / "y"))
        self.assertEqual(back.labels, ["Mango_Ripe_Alphonso", "Mango_Unripe", "Strawberry", "Strawberry_Ripe"])

    def test_relabelled_subset_writes_only_true_facts(self):
        ds = nhan.load(FIXTURES / "coco_fields").relabel("condition")
        back = nhan.load(ds.to_coco(self.tmp / "c"))
        ripe = [b for b in back.boxes if b.label == "Ripe"]
        self.assertTrue(all(b.cls == "Ripe" and b.attributes == {} for b in ripe))

    def test_polygons_survive_coco_and_yolo_and_become_boxes_in_voc(self):
        ds = nhan.load(FIXTURES / "coco_joined")
        diamond = [(0.5, 0.25), (0.75, 0.5), (0.5, 0.75), (0.25, 0.5)]

        def poly_of(d):
            return [b.polygon for b in d.boxes if b.polygon is not None]

        for fmt in ("coco", "yolo"):
            with self.subTest(fmt):
                back = nhan.load(getattr(ds, f"to_{fmt}")(self.tmp / fmt))
                self.assertEqual([[(round(x, 6), round(y, 6)) for x, y in p] for p in poly_of(back)], [diamond])
                self.assertEqual(sum(1 for b in back.boxes if b.polygon is None), 3)
        voc = nhan.load(ds.to_voc(self.tmp / "voc"))
        self.assertEqual(poly_of(voc), [])
        self.assertEqual(snapshot(voc, with_label=False), snapshot(ds, with_label=False))

    def test_polygon_area_in_coco(self):
        import json

        out = nhan.load(FIXTURES / "coco_joined").to_coco(self.tmp / "c")
        data = json.loads((out / "annotations" / "instances_train.json").read_text())
        poly = [a for a in data["annotations"] if "segmentation" in a][0]
        # diamond in a 32 x 24 px box: half of it
        self.assertAlmostEqual(poly["area"], 384.0, places=2)

    def test_box_helpers(self):
        b = [x for x in nhan.load(FIXTURES / "coco_joined").boxes if x.is_polygon][0]
        self.assertEqual(len(b.corners()), 4)
        self.assertEqual(b.polygon_pixels(64, 48)[0], (32.0, 12.0))
        plain = [x for x in nhan.load(FIXTURES / "coco_joined").boxes if not x.is_polygon][0]
        self.assertEqual(len(plain.corners()), 4)

    def test_refuses_a_folder_that_is_not_empty(self):
        (self.tmp / "busy").mkdir()
        (self.tmp / "busy" / "x").write_text("x")
        with self.assertRaises(NhanError):
            nhan.load(FIXTURES / "coco_joined").to_yolo(self.tmp / "busy")

    def test_json_file(self):
        import json

        path = nhan.load(FIXTURES / "voc_joined").to_json(self.tmp / "out" / "d.json")
        data = json.loads(path.read_text())
        self.assertEqual(data["nhan_dataset_version"], 1)
        self.assertTrue((path.parent / data["images"][0]["file"]).is_file())

    def test_preview_page(self):
        page = nhan.load(FIXTURES / "coco_joined").preview(self.tmp / "p.html")
        text = page.read_text()
        self.assertEqual(text.count("<rect "), 3)
        self.assertEqual(text.count("<polygon "), 1)
        self.assertIn("../", text)  # relative image paths
        self.assertIn("prefers-color-scheme", text)


if __name__ == "__main__":
    unittest.main()
