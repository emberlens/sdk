import unittest
import xml.etree.ElementTree as ET

from nhan.imagesize import image_size
from nhan.minixml import parse_string
from nhan.readers.yolo import parse_data_yaml

from .helpers import FIXTURES


class YamlTest(unittest.TestCase):
    def test_flow_list_with_quotes(self):
        d = parse_data_yaml("nc: 3\nnames: [a, 'it''s', \"b, c\"]\n")
        self.assertEqual(d["names"], ["a", "it's", "b, c"])
        self.assertEqual(d["nc"], "3")

    def test_block_list_and_index_map(self):
        self.assertEqual(parse_data_yaml("names:\n  - x\n  - 'y z'\n")["names"], ["x", "y z"])
        self.assertEqual(parse_data_yaml("names:\n  1: b\n  0: a\n")["names"], ["a", "b"])

    def test_multi_line_flow_list_and_comments(self):
        d = parse_data_yaml("# hi\nnames: [a,\n  b]  # two\npath: .\n")
        self.assertEqual(d["names"], ["a", "b"])
        self.assertEqual(d["path"], ".")


class ImageSizeTest(unittest.TestCase):
    def test_jpeg_and_png(self):
        self.assertEqual(image_size(FIXTURES / "coco_joined" / "images" / "test" / "photo-3.jpg"), (60, 80))
        self.assertEqual(image_size(FIXTURES / "source" / "img1.png"), (80, 60))
        self.assertIsNone(image_size(FIXTURES / "coco_joined" / "nhan.json"))


class MiniXmlTest(unittest.TestCase):
    def test_reads_a_voc_file(self):
        root = parse_string((FIXTURES / "voc_fields" / "Annotations" / "photo-1.xml").read_text())
        self.assertEqual(root.find("size/width").text, "80")
        self.assertEqual([o.find("name").text for o in root.findall("object")], ["Mango", "Strawberry"])
        self.assertEqual(root.find("object/attributes/attribute/value").text, "Ripe")

    def test_entities_and_errors(self):
        root = parse_string("<?xml version='1.0'?><a><b>x &amp; y &#65;</b><c/></a>")
        self.assertEqual(root.find("b").text, "x & y A")
        self.assertIsInstance(root.find("c"), ET.Element)
        with self.assertRaises(ValueError):
            parse_string("<a><b></a>")


if __name__ == "__main__":
    unittest.main()
