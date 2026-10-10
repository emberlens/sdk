import shutil
import tempfile
import unittest
from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
NAMES = ["coco_single", "coco_joined", "coco_fields", "yolo_joined", "voc_fields", "voc_joined", "coco_resized", "voc_resized"]


def has(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


class TempDirCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="emberlens-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def copy_fixture(self, name: str) -> Path:
        dest = self.tmp / name
        shutil.copytree(FIXTURES / name, dest)
        return dest


def snapshot(ds, with_label=True):
    return [
        (
            img.id,
            img.split,
            [
                ((b.label if with_label else ""), b.cls, b.attributes, round(b.x, 3), round(b.y, 3), round(b.w, 3), round(b.h, 3))
                for b in img.boxes
            ],
        )
        for img in sorted(ds, key=lambda i: i.id)
    ]
