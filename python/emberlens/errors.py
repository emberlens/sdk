"""Errors with messages that a non-programmer can act on."""

from .command import command


class EmberlensError(Exception):
    """Something the user can fix. The message says how."""


class MissingExtra(EmberlensError):
    """An optional library (Pillow, PyTorch) is not installed."""

    def __init__(self, extra: str, packages: str):
        self.extra = extra
        self.packages = packages
        super().__init__(
            f"This needs {packages}. Install it with:\n"
            f"    {command('install', extra)}\n"
            f"(or: pip install {packages})"
        )
