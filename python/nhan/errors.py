"""Errors with messages that a non-programmer can act on."""


class NhanError(Exception):
    """Something the user can fix. The message says how."""


class MissingExtra(NhanError):
    """An optional library (Pillow, PyTorch) is not installed."""

    def __init__(self, extra: str, packages: str):
        self.extra = extra
        self.packages = packages
        super().__init__(
            f"This needs {packages}. Install it with:\n"
            f"    ./nhan install {extra}\n"
            f"(or: pip install {packages})"
        )
