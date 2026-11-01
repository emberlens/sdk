"""The project recipe (sdk/schema/emberlens-project.schema.json, SPEC.md §10).

It lists the labels and export settings of a shared project. It holds no
photos.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Union

from .errors import EmberlensError

RECIPE_VERSION = 1
MAX_FILE_BYTES = 1024 * 1024


def load_recipe(path: Union[str, Path]) -> dict:
    """Read a recipe or an invite. A `sync` block is removed: it holds a secret."""
    p = Path(path)
    try:
        if p.stat().st_size > MAX_FILE_BYTES:
            raise EmberlensError(f"{p.name} is too large to be a project file.")
        data = json.loads(p.read_text(encoding="utf-8"))
    except OSError:
        raise EmberlensError(f"Could not read {p}. Check the path.") from None
    except ValueError:
        raise EmberlensError(f"{p.name} is not an Emberlens project file.") from None
    if not isinstance(data, dict) or "emberlens_project_version" not in data:
        raise EmberlensError(f"{p.name} is not an Emberlens project file.")
    if data["emberlens_project_version"] != RECIPE_VERSION:
        raise EmberlensError(
            "This project file was made by a newer Emberlens app. Get the newest "
            "SDK from the Emberlens repository and try again."
        )
    if not isinstance(data.get("classes"), list) or not isinstance(data.get("sub_categories"), list):
        raise EmberlensError(f"{p.name} is damaged.")
    data.pop("sync", None)
    return data


def describe(recipe: dict) -> str:
    """The recipe as plain text."""
    names = {c["id"]: c["name"] for c in recipe["classes"]}
    lines: List[str] = [
        f'Project: {recipe.get("name", "")} (revision {recipe.get("revision", "?")})',
    ]
    if recipe.get("description"):
        lines.append(recipe["description"])
    lines.append(f'Classes ({recipe.get("root_category", "class")}):')
    for c in sorted(recipe["classes"], key=lambda c: c["order"]):
        lines.append(f'  {c["name"]}' + ("  (retired)" if c.get("retired") else ""))
    for s in sorted(recipe["sub_categories"], key=lambda s: s["order"]):
        lines.append(f'Sub-category: {s["name"]}')
        for v in sorted(s["values"], key=lambda v: v["order"]):
            only = v.get("only_for_classes")
            note = "" if only is None else "  (only for " + ", ".join(names.get(i, "?") for i in only) + ")"
            lines.append(f'  {v["name"]}{note}')
    return "\n".join(lines)
