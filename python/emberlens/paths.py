"""Small helpers for naming output files."""

from __future__ import annotations

from pathlib import Path


def free_path(path: Path) -> Path:
    """`path`, or path-2, path-3 … if it already exists."""
    if not path.exists():
        return path
    n = 2
    while True:
        cand = path.with_name(f"{path.stem}-{n}{path.suffix}")
        if not cand.exists():
            return cand
        n += 1


def export_stem(export: str) -> str:
    p = Path(export.rstrip("/\\"))
    return p.stem if p.suffix.lower() == ".zip" else p.name
