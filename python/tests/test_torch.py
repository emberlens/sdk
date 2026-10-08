"""Pillow / PyTorch parts. Skipped when those libraries are not installed."""

import json
import unittest

import nhan

from .helpers import FIXTURES, TempDirCase, has


@unittest.skipUnless(has("PIL"), "needs Pillow: ./nhan install images")
class CropsTest(TempDirCase):
    def test_crops_one_folder_per_label(self):
        out = nhan.load(FIXTURES / "voc_fields").relabel("condition").crops(self.tmp / "c", padding=0.1)
        files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*.jpg"))
        self.assertEqual(files, ["train/Ripe/photo-1_1.jpg", "train/Ripe/photo-1_2.jpg", "train/Unripe/photo-2_1.jpg"])

    def test_open_turns_an_exif_rotated_photo_upright(self):
        import shutil

        from nhan.model import Image

        src = FIXTURES.parent.parent / "packages" / "nhan_data" / "test" / "fixtures" / "exif6_30x60.jpg"
        dest = self.tmp / "rotated.jpg"
        shutil.copy(src, dest)
        photo = Image(id="r", path=dest, width=30, height=60, split="train").open()
        self.assertEqual(photo.size, (30, 60))
        r, g, b = photo.convert("RGB").getpixel((7, 15))
        self.assertTrue(r > 200 and g < 60 and b < 60, (r, g, b))

    def test_open_gives_a_pillow_image(self):
        self.assertEqual(nhan.load(FIXTURES / "coco_joined")[0].open().size[0] > 0, True)


@unittest.skipUnless(has("torch") and has("torchvision") and has("PIL"), "needs PyTorch: ./nhan install train")
class TorchTest(TempDirCase):
    def test_detection_dataset(self):
        det = nhan.load(FIXTURES / "coco_joined").torch_detection("train")
        image, target = det[0]
        self.assertEqual(det.classes[0], "background")
        self.assertEqual(tuple(image.shape[:1]), (3,))
        self.assertEqual(target["boxes"].shape[1], 4)
        self.assertTrue((target["labels"] >= 1).all())

    def test_classification_dataset(self):
        cls = nhan.load(FIXTURES / "coco_fields").torch_classification("train", size=32)
        image, index = cls[0]
        self.assertEqual(tuple(image.shape), (3, 32, 32))
        self.assertIn(index, range(len(cls.classes)))

    def test_train_and_predict_a_classifier(self):
        from nhan.train import predict, train_classifier

        ds = nhan.load(FIXTURES / "coco_fields").resplit(1, 0, 0)
        out = train_classifier(ds, self.tmp / "m", target="class", epochs=1, pretrained=False, size=32, device="cpu", log=lambda _: None)
        report = json.loads((out / "report.json").read_text())
        self.assertEqual(report["classes"], ["Mango", "Strawberry"])
        result = predict(out, [FIXTURES / "coco_fields" / "images" / "train" / "photo-1.jpg"])
        self.assertIn(result[0]["label"], ["Mango", "Strawberry"])

    def test_train_and_predict_a_detector(self):
        from nhan.train import predict, train_detector

        ds = nhan.load(FIXTURES / "coco_joined")
        out = train_detector(ds, self.tmp / "d", target="class", epochs=1, pretrained=False, device="cpu", log=lambda _: None)
        result = predict(out, [FIXTURES / "coco_joined" / "images" / "train" / "photo-1.jpg"], threshold=0.0)
        self.assertIn("boxes", result[0])
        for b in result[0]["boxes"]:
            self.assertIn(b["label"], ["Mango", "Strawberry"])
            self.assertTrue(0 <= b["x"] <= 1)


if __name__ == "__main__":
    unittest.main()
