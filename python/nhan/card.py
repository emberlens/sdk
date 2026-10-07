"""The nhan.json dataset card (sdk/schema/nhan-card.schema.json).

It maps every label name in the export to its parts, so a joined name like
"mango_alphonso_ripe" is split without guessing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

CARD_FILE = "nhan.json"
CARD_VERSION = 1


def read_card(root: Path) -> Optional[dict]:
    path = root / CARD_FILE
    if not path.is_file():
        return None
    try:
        card = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(card, dict):
        return None
    version = card.get("nhan_card_version")
    if isinstance(version, int) and version > CARD_VERSION:
        from .errors import NhanError

        raise NhanError(
            "This export was made by a newer Nhan app. Get the newest SDK "
            "from the Nhan repository and try again."
        )
    return card


def label_parts(card: Optional[dict]) -> Dict[str, Tuple[str, Dict[str, str]]]:
    """{label name: (class, {sub-category: value})} from the card."""
    out: Dict[str, Tuple[str, Dict[str, str]]] = {}
    if not card:
        return out
    for entry in card.get("labels") or []:
        name = entry.get("name")
        if isinstance(name, str):
            out[name] = (str(entry.get("class") or name), dict(entry.get("attributes") or {}))
    return out
