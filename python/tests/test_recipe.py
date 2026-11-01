import json

from emberlens.errors import EmberlensError
from emberlens.recipe import describe, load_recipe

from .helpers import FIXTURES, TempDirCase

RECIPE = FIXTURES / "recipe" / "fruit.emberlens-project.json"


class RecipeTest(TempDirCase):
    def write(self, text):
        path = self.tmp / "x.json"
        path.write_text(text, encoding="utf-8")
        return path

    def test_reads_the_fixture(self):
        r = load_recipe(RECIPE)
        self.assertEqual(r["revision"], 3)
        self.assertEqual([c["name"] for c in r["classes"]], ["Mango", "Kiwi"])

    def test_describe_lists_labels_and_limits(self):
        text = describe(load_recipe(RECIPE))
        self.assertIn("Fruit survey (revision 3)", text)
        self.assertIn("Kiwi  (retired)", text)
        self.assertIn("Unripe  (only for Mango)", text)

    def test_sync_block_is_removed(self):
        data = json.loads(RECIPE.read_text(encoding="utf-8"))
        data["sync"] = {"room_secret": "abc"}
        self.assertNotIn("sync", load_recipe(self.write(json.dumps(data))))

    def test_refuses_other_files(self):
        for text in ("not json", "[]", "{}"):
            with self.assertRaises(EmberlensError):
                load_recipe(self.write(text))

    def test_refuses_a_newer_version(self):
        with self.assertRaisesRegex(EmberlensError, "newer"):
            load_recipe(self.write('{"emberlens_project_version": 2}'))
