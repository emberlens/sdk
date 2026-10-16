"""A1a: the shared crop, class "other", export-model, card field other_boxes."""

import json
import unittest

import emberlens

from .helpers import FIXTURES, TempDirCase, has
from .test_schema import SCHEMA, check

PILLOW = unittest.skipUnless(has("PIL"), "needs Pillow: ./emberlens install images")
TORCH = unittest.skipUnless(has("torch") and has("torchvision") and has("PIL"), "needs PyTorch: ./emberlens install train")
ONNX = unittest.skipUnless(
    has("onnx"), "needs the optional onnx library: ask the owner to run ./emberlens install onnx"
)
CV2 = unittest.skipUnless(has("cv2"), "needs OpenCV (cv2) to run the exported model")


class OtherBoxesCardTest(TempDirCase):
    def test_fixture_card_matches_the_schema(self):
        card = json.loads((FIXTURES / "coco_other" / "emberlens.json").read_text())
        schema = json.loads((SCHEMA / "emberlens-card.schema.json").read_text())
        self.assertEqual(check(card, schema, schema), [])
        self.assertEqual(len(card["other_boxes"]), 4)

    def test_schema_refuses_a_malformed_other_box(self):
        card = json.loads((FIXTURES / "coco_other" / "emberlens.json").read_text())
        schema = json.loads((SCHEMA / "emberlens-card.schema.json").read_text())
        card["other_boxes"][0]["w"] = "wide"
        del card["other_boxes"][1]["image"]
        self.assertEqual(len(check(card, schema, schema)), 2)

    def test_dataset_lists_other_boxes_per_split(self):
        ds = emberlens.load(FIXTURES / "coco_other")
        pairs = ds.other_boxes()
        self.assertEqual([(i.path.name, i.split) for i, _ in pairs],
                         [("photo-1.jpg", "train"), ("photo-2.jpg", "train"), ("photo-3.jpg", "test"), ("photo-4.jpg", "val")])
        box = pairs[0][1]
        self.assertEqual((box.label, box.cls, box.w), ("other", "other", 0.3))
        self.assertEqual(len(ds.other_boxes("train")), 2)
        self.assertEqual(len(ds.split("val").other_boxes()), 1)

    def test_no_field_means_no_other_boxes(self):
        self.assertEqual(emberlens.load(FIXTURES / "coco_joined").other_boxes(), [])

    def test_other_boxes_do_not_change_the_normal_boxes(self):
        a, b = emberlens.load(FIXTURES / "coco_joined"), emberlens.load(FIXTURES / "coco_other")
        self.assertEqual(a.box_count, b.box_count)
        self.assertEqual(a.labels, b.labels)

    def test_boxes_of_photos_not_in_the_export_are_ignored(self):
        root = self.copy_fixture("coco_other")
        card = json.loads((root / "emberlens.json").read_text())
        card["other_boxes"].append({"image": "gone.jpg", "x": 0, "y": 0, "w": 0.5, "h": 0.5, "score": 0.9})
        (root / "emberlens.json").write_text(json.dumps(card))
        self.assertEqual(len(emberlens.load(root).other_boxes()), 4)


@PILLOW
class SharedCropTest(TempDirCase):
    def test_crop_for_model_is_the_old_rule(self):
        from emberlens.crops import crop_box, crop_for_model

        img = emberlens.load(FIXTURES / "coco_joined")[0]
        photo, box = img.open().convert("RGB"), img.boxes[0]
        old = crop_box(photo, box, 0.1).resize((224, 224))
        new = crop_for_model(photo, box)
        self.assertEqual(new.size, (224, 224))
        self.assertEqual(new.tobytes(), old.tobytes())
        self.assertEqual(crop_for_model(photo, box, padding=0.0, size=32).size, (32, 32))

    def test_crop_for_model_reaches_outside_the_photo_safely(self):
        from emberlens.crops import crop_for_model

        img = emberlens.load(FIXTURES / "coco_joined")[0]
        edge = emberlens.Box(label="o", cls="o", x=0.9, y=0.9, w=0.3, h=0.3)
        self.assertEqual(crop_for_model(img.open().convert("RGB"), edge, size=16).size, (16, 16))

    def test_train_and_crops_use_the_shared_function(self):
        from emberlens import crops, torch_data, train

        self.assertIs(torch_data.crop_for_model, crops.crop_for_model)
        self.assertIs(train.crop_for_model, crops.crop_for_model)

    def test_write_crops_with_size_gives_square_crops(self):
        from PIL import Image as PILImage

        out = emberlens.load(FIXTURES / "coco_joined").crops(self.tmp / "c", padding=0.1, size=224)
        self.assertTrue(all(PILImage.open(f).size == (224, 224) for f in out.rglob("*.jpg")))


@TORCH
class OtherClassTrainingTest(TempDirCase):
    def train(self, fixture, **kw):
        from emberlens.train import train_classifier

        ds = emberlens.load(FIXTURES / fixture).resplit(1, 0, 0)
        return train_classifier(ds, self.tmp / "m", target="class", epochs=1, pretrained=False, size=32, device="cpu", log=lambda _: None, **kw)

    def test_other_is_the_last_class(self):
        out = self.train("coco_other")
        report = json.loads((out / "report.json").read_text())
        self.assertEqual(report["classes"], ["Mango", "Strawberry", "other"])
        self.assertEqual(report["boxes"]["train"], emberlens.load(FIXTURES / "coco_other").box_count + 4)

    def test_no_field_means_no_other_class(self):
        out = self.train("coco_joined")
        self.assertEqual(json.loads((out / "report.json").read_text())["classes"], ["Mango", "Strawberry"])

    def test_a_project_class_called_other_is_refused(self):
        from emberlens.errors import EmberlensError

        ds = emberlens.load(FIXTURES / "coco_other").relabel("class")
        for img in ds:
            for b in img.boxes:
                b.label = b.cls = "Other"
        with self.assertRaises(EmberlensError):
            from emberlens.train import train_classifier

            train_classifier(ds, self.tmp / "x", epochs=1, pretrained=False, size=32, device="cpu", log=lambda _: None)

    def test_crop_dataset_gives_other_the_last_index(self):
        ds = emberlens.load(FIXTURES / "coco_other").relabel("class")
        from emberlens.torch_data import CropDataset

        data = CropDataset(ds, "train", 32, 0.1, classes=ds.labels + ["other"], with_other=True)
        self.assertEqual(data.classes[-1], "other")
        self.assertEqual(sorted(set(data[i][1] for i in range(len(data))))[-1], len(data.classes) - 1)


class ExportMetadataTest(TempDirCase):
    """The metadata is built without the onnx library, so this always runs."""

    def test_metadata_keys(self):
        from emberlens.export_model import build_metadata

        ckpt = {"classes": ["Mango", "other"], "size": 224, "padding": 0.1, "target": "class"}
        meta = build_metadata(ckpt, {"test_accuracy": 0.875})
        self.assertEqual(meta["emberlens_model"], "1")
        self.assertEqual(meta["kind"], "classifier")
        self.assertEqual(json.loads(meta["classes"]), ["Mango", "other"])
        self.assertEqual((meta["input_size"], meta["size"]), ("224", "224"))
        self.assertEqual(meta["crop_pad"], "0.1")
        self.assertEqual(meta["resize_mode"], "stretch")
        self.assertEqual(json.loads(meta["mean"]), [0.485, 0.456, 0.406])
        self.assertEqual(json.loads(meta["std"]), [0.229, 0.224, 0.225])
        self.assertEqual(meta["test_accuracy"], "0.875")
        self.assertEqual(meta["labels_mode"], "class")

    def test_no_test_accuracy_leaves_the_key_out(self):
        from emberlens.export_model import build_metadata

        meta = build_metadata({"classes": ["a", "b"], "size": 32, "padding": 0.1, "target": "label"}, {"test_accuracy": None})
        self.assertNotIn("test_accuracy", meta)

    def test_cli_knows_export_model(self):
        from emberlens.cli import build_parser

        args = build_parser().parse_args(["export-model", "folder", "--out", "m.onnx"])
        self.assertEqual((args.model, args.out), ("folder", "m.onnx"))

    @unittest.skipIf(has("onnx"), "onnx is installed")
    def test_missing_onnx_says_how_to_install(self):
        from emberlens.errors import MissingExtra
        from emberlens.export_model import export_model

        with self.assertRaises(MissingExtra) as ctx:
            export_model(self.tmp)
        self.assertIn("./emberlens install onnx", str(ctx.exception))


@TORCH
@ONNX
class ExportModelTest(TempDirCase):
    def trained(self):
        from emberlens.train import train_classifier

        ds = emberlens.load(FIXTURES / "coco_other").resplit(1, 0, 0)
        return train_classifier(ds, self.tmp / "m", target="class", epochs=1, pretrained=False, size=224, device="cpu", log=lambda _: None)

    def test_the_file_has_the_metadata(self):
        import onnx

        from emberlens.export_model import export_model

        out = export_model(self.trained(), self.tmp / "m.onnx", log=lambda _: None)
        meta = {p.key: p.value for p in onnx.load(str(out)).metadata_props}
        self.assertEqual(json.loads(meta["classes"]), ["Mango", "Strawberry", "other"])
        self.assertEqual(meta["resize_mode"], "stretch")
        self.assertEqual(meta["crop_pad"], "0.1")

    @CV2
    def test_opencv_gives_the_same_top_label_as_pytorch(self):
        import cv2
        import numpy as np
        import torch

        from emberlens.crops import crop_for_model
        from emberlens.export_model import export_model
        from emberlens.torch_data import IMAGENET_MEAN, IMAGENET_STD
        from emberlens.train import load_model

        folder = self.trained()
        net = cv2.dnn.readNetFromONNX(str(export_model(folder, self.tmp / "m.onnx", log=lambda _: None)))
        model, _ = load_model(folder)
        mean, std = np.array(IMAGENET_MEAN, "float32"), np.array(IMAGENET_STD, "float32")
        for img in emberlens.load(FIXTURES / "coco_joined"):
            for box in img.boxes:
                crop = np.asarray(crop_for_model(img.open().convert("RGB"), box), "float32") / 255
                x = ((crop - mean) / std).transpose(2, 0, 1)[None]
                net.setInput(x)
                with torch.no_grad():
                    want = int(model(torch.from_numpy(x)).argmax())
                self.assertEqual(int(net.forward().argmax()), want)


if __name__ == "__main__":
    unittest.main()
