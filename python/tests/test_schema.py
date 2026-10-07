"""The JSON schemas in sdk/schema/ match the real files.

A tiny checker for the parts of JSON Schema the two schemas use, so no
extra library is needed.
"""

import json
import unittest
from pathlib import Path

import nhan
from nhan.writers import dataset_dict

from .helpers import FIXTURES, NAMES, TempDirCase

SCHEMA = Path(__file__).resolve().parents[2] / "schema"
TYPES = {"object": dict, "array": list, "string": str, "integer": int, "number": (int, float), "boolean": bool, "null": type(None)}


def check(value, schema, root, where="$"):
    if "$ref" in schema:
        name = schema["$ref"].split("/")[-1]
        return check(value, root["$defs"][name], root, where)
    errors = []
    kinds = schema.get("type")
    if kinds:
        kinds = kinds if isinstance(kinds, list) else [kinds]
        ok = any(isinstance(value, TYPES[k]) and not (k in ("integer", "number") and isinstance(value, bool)) for k in kinds)
        if not ok:
            return [f"{where}: {value!r} is not {kinds}"]
    if "const" in schema and value != schema["const"]:
        errors.append(f"{where}: must be {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{where}: {value!r} not in {schema['enum']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{where}: below {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{where}: above {schema['maximum']}")
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{where}: missing {key}")
        props = schema.get("properties", {})
        for key, sub in value.items():
            if key in props:
                errors += check(sub, props[key], root, f"{where}.{key}")
            elif isinstance(schema.get("additionalProperties"), dict):
                errors += check(sub, schema["additionalProperties"], root, f"{where}.{key}")
            elif props and "additionalProperties" not in schema:
                errors.append(f"{where}: unknown key {key}")
    if isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            errors += check(item, schema["items"], root, f"{where}[{i}]")
    return errors


def load_schema(name):
    return json.loads((SCHEMA / name).read_text(encoding="utf-8"))


class SchemaTest(TempDirCase):
    def test_app_cards_match_the_card_schema(self):
        schema = load_schema("nhan-card.schema.json")
        for name in NAMES:
            with self.subTest(name):
                card = json.loads((FIXTURES / name / "nhan.json").read_text())
                self.assertEqual(check(card, schema, schema), [])

    def test_sdk_cards_match_the_card_schema(self):
        schema = load_schema("nhan-card.schema.json")
        out = nhan.load(FIXTURES / "coco_fields").relabel("condition").to_voc(self.tmp / "v")
        card = json.loads((out / "nhan.json").read_text())
        self.assertEqual(check(card, schema, schema), [])

    def test_expected_files_match_the_dataset_schema(self):
        schema = load_schema("nhan-dataset.schema.json")
        for name in NAMES:
            with self.subTest(name):
                data = json.loads((FIXTURES / "expected" / f"{name}.json").read_text())
                self.assertEqual(check(data, schema, schema), [])

    def test_the_checker_finds_mistakes(self):
        schema = load_schema("nhan-card.schema.json")
        bad = {"nhan_card_version": 2, "format": "png", "label_style": "joined", "classes": [], "sub_categories": [], "labels": [{"name": "a"}], "oops": 1}
        errors = check(bad, schema, schema)
        self.assertTrue(any("must be 1" in e for e in errors))
        self.assertTrue(any("png" in e for e in errors))
        self.assertTrue(any("missing class" in e for e in errors))
        self.assertTrue(any("unknown key oops" in e for e in errors))


if __name__ == "__main__":
    unittest.main()
