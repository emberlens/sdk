"""export-model writes a detector to one .onnx file (SPEC.md section 9.2)."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import emberlens
from emberlens.export_model import build_metadata

from .helpers import FIXTURES, has

TORCH = unittest.skipUnless(has("torch") and has("torchvision") and has("PIL"), "needs PyTorch: ./emberlens install train")
ONNX = unittest.skipUnless(
    has("onnx"), "needs the optional onnx library: ask the owner to run ./emberlens install onnx"
)
ORT = unittest.skipUnless(has("onnxruntime"), "needs onnxruntime to run the exported detector (test only)")

PHOTO = FIXTURES / "coco_joined" / "images" / "train" / "photo-1.jpg"


class DetectorMetadataTest(unittest.TestCase):
    def test_detector_metadata(self):
        ckpt = {"kind": "detector", "classes": ["background", "Mango", "Strawberry"], "target": "class"}
        meta = build_metadata(ckpt, {"test": {"recall": 0.75}})
        self.assertEqual(meta["emberlens_model"], "1")
        self.assertEqual(meta["kind"], "detector")
        self.assertEqual(json.loads(meta["classes"]), ["Mango", "Strawberry"])
        self.assertEqual((meta["size"], meta["input_size"]), ("320", "320"))
        self.assertEqual((meta["crop_pad"], meta["resize_mode"]), ("0", "stretch"))
        self.assertEqual(json.loads(meta["mean"]), [0.0, 0.0, 0.0])
        self.assertEqual(json.loads(meta["std"]), [1.0, 1.0, 1.0])
        self.assertEqual((meta["min_score"], meta["test_recall"]), ("0.5", "0.75"))
        self.assertNotIn("test_accuracy", meta)

    def test_detector_metadata_keeps_unicode_names(self):
        names = ["background", 'Mango "Kent"', "आम", "🍓"]
        meta = build_metadata({"kind": "detector", "classes": names, "target": "label"})
        self.assertEqual(json.loads(meta["classes"]), names[1:])
        self.assertIn("आम", meta["classes"])  # not \u escapes

    def test_no_recall_leaves_the_key_out(self):
        meta = build_metadata({"kind": "detector", "classes": ["background", "a"]}, {"test": None})
        self.assertNotIn("test_recall", meta)

    def test_classifier_metadata_is_unchanged(self):
        meta = build_metadata({"kind": "classifier", "classes": ["a", "b"], "size": 224}, {"test_accuracy": 0.5})
        self.assertEqual(meta["kind"], "classifier")
        self.assertEqual(json.loads(meta["classes"]), ["a", "b"])
        self.assertEqual((meta["crop_pad"], meta["test_accuracy"]), ("0.1", "0.5"))
        self.assertNotIn("min_score", meta)


@TORCH
@ONNX
class ExportDetectorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from emberlens.export_model import export_model
        from emberlens.train import train_detector

        cls.tmp = Path(tempfile.mkdtemp(prefix="emberlens-test-"))
        ds = emberlens.load(FIXTURES / "coco_joined")
        cls.folder = train_detector(ds, cls.tmp / "d", target="class", epochs=1, pretrained=False, device="cpu", log=lambda _: None)
        cls.onnx_file = export_model(cls.folder, cls.tmp / "d.onnx", log=lambda _: None)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_detector_file_has_three_outputs_and_metadata(self):
        import onnx

        proto = onnx.load(str(self.onnx_file))
        inputs = [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in proto.graph.input]
        self.assertEqual(inputs, [("input", [1, 3, 320, 320])])
        self.assertEqual([o.name for o in proto.graph.output], ["boxes", "labels", "scores"])
        meta = {p.key: p.value for p in proto.metadata_props}
        self.assertEqual(meta["kind"], "detector")
        self.assertEqual(json.loads(meta["classes"]), ["Mango", "Strawberry"])
        self.assertEqual(meta["min_score"], "0.5")

    def test_unknown_kind_is_refused(self):
        import torch

        from emberlens.errors import EmberlensError
        from emberlens.export_model import export_model

        bad = self.tmp / "bad"
        bad.mkdir(exist_ok=True)
        ckpt = torch.load(self.folder / "model.pt", map_location="cpu", weights_only=True)
        ckpt["kind"] = "segmenter"
        torch.save(ckpt, bad / "model.pt")
        with self.assertRaises(EmberlensError):
            export_model(bad, self.tmp / "bad.onnx", log=lambda _: None)

    def photo_tensor(self):
        import numpy as np
        from PIL import Image

        img = Image.open(PHOTO).convert("RGB").resize((320, 320), Image.BILINEAR)
        return (np.asarray(img, "float32") / 255).transpose(2, 0, 1)[None].copy()

    @ORT
    def test_onnxruntime_matches_pytorch(self):
        import numpy as np
        import onnxruntime
        import torch

        from emberlens.export_model import DETECTOR_SIZE, _detector_for_onnx
        from emberlens.train import load_model

        x = self.photo_tensor()
        model, _ = load_model(self.folder)
        with torch.no_grad():
            want = [t.numpy() for t in _detector_for_onnx(model, DETECTOR_SIZE)(torch.from_numpy(x))]
        session = onnxruntime.InferenceSession(str(self.onnx_file), providers=["CPUExecutionProvider"])
        got = session.run(None, {"input": x})
        self.assertEqual(got[0].shape, want[0].shape)
        self.assertTrue(np.all(np.diff(got[2]) <= 0))  # raw scores: highest first
        # Boxes with equal scores may come in either order: compare both in
        # one order (score, then label, then corner).
        def ordered(out):
            b, l, s = out
            i = np.lexsort((b[:, 1], b[:, 0], l, -np.round(s, 4)))
            return b[i], l[i], s[i]

        boxes, labels, scores = ordered(got)
        want = ordered(want)
        np.testing.assert_allclose(boxes, want[0], atol=1e-3 * 320)
        np.testing.assert_array_equal(labels, want[1])
        np.testing.assert_allclose(scores, want[2], atol=1e-4)
        self.assertEqual((boxes.dtype, labels.dtype, scores.dtype), (np.float32, np.int64, np.float32))
        if len(labels):
            self.assertGreaterEqual(int(labels.min()), 0)  # 0-based, no background

    @ORT
    def test_empty_photo_gives_empty_outputs(self):
        import numpy as np
        import onnxruntime

        session = onnxruntime.InferenceSession(str(self.onnx_file), providers=["CPUExecutionProvider"])
        boxes, labels, scores = session.run(None, {"input": np.zeros((1, 3, 320, 320), "float32")})
        self.assertEqual(boxes.shape[1], 4)
        self.assertEqual(len(boxes), len(labels))
        self.assertEqual(len(boxes), len(scores))


if __name__ == "__main__":
    unittest.main()
