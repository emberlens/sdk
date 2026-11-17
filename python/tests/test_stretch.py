"""Detectors see the whole photo stretched to 320x320, like the app (SPEC.md section 9.2)."""

import unittest

import emberlens
from emberlens.model import Box

from .helpers import FIXTURES, has


class StretchBoxTest(unittest.TestCase):
    def test_a_box_keeps_its_0_to_1_position_after_the_stretch(self):
        from emberlens.torch_data import DETECTOR_SIZE, stretch_box

        box = Box(label="Mango", cls="Mango", x=0.1, y=0.25, w=0.5, h=0.3)
        for width, height in [(640, 480), (300, 900), (320, 320)]:
            x0, y0, x1, y1 = stretch_box(box.pixels(width, height), width, height, DETECTOR_SIZE)
            s = DETECTOR_SIZE
            self.assertAlmostEqual(x0 / s, box.x)
            self.assertAlmostEqual(y0 / s, box.y)
            self.assertAlmostEqual((x1 - x0) / s, box.w)
            self.assertAlmostEqual((y1 - y0) / s, box.h)

    def test_detector_size_is_the_spec_value(self):
        from emberlens.torch_data import DETECTOR_SIZE

        self.assertEqual(DETECTOR_SIZE, 320)


@unittest.skipUnless(has("torch") and has("torchvision") and has("PIL"), "needs PyTorch: ./emberlens install train")
class StretchTorchTest(unittest.TestCase):
    def test_detection_dataset_stretches_the_photo_and_its_boxes(self):
        from emberlens.torch_data import DetectionDataset

        ds = emberlens.load(FIXTURES / "coco_joined")
        det = DetectionDataset(ds, "train", size=320)
        image, target = det[0]
        self.assertEqual(tuple(image.shape), (3, 320, 320))
        want = [b for b in det.images[0].boxes if b.label in det.index]
        self.assertEqual(len(want), target["boxes"].shape[0])
        for b, got in zip(want, target["boxes"].tolist()):
            self.assertAlmostEqual(got[0] / 320, b.x, places=5)
            self.assertAlmostEqual(got[1] / 320, b.y, places=5)
            self.assertAlmostEqual(got[2] / 320, b.x + b.w, places=5)
            self.assertAlmostEqual(got[3] / 320, b.y + b.h, places=5)

    def test_detection_dataset_without_size_keeps_the_photo_size(self):
        from emberlens.torch_data import DetectionDataset

        ds = emberlens.load(FIXTURES / "coco_joined")
        det = DetectionDataset(ds, "train")
        image, _ = det[0]
        photo = det.images[0]
        self.assertEqual(tuple(image.shape[1:]), (photo.height, photo.width))

    def test_the_network_does_not_resize_a_320_input_again(self):
        from emberlens.train import _detector

        model = _detector(3, pretrained=False)
        self.assertEqual(tuple(model.transform.min_size), (320,))
        self.assertEqual(model.transform.max_size, 320)


if __name__ == "__main__":
    unittest.main()
