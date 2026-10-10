"""Optional libraries. The core SDK needs none; images and training do."""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys

from .errors import MissingExtra, EmberlensError

# Package names on PyPI per extra. All permissive: Pillow (MIT-CMU),
# PyTorch and torchvision (BSD-3).
EXTRAS = {
    "images": ["pillow"],
    "train": ["pillow", "torch", "torchvision"],
}


def require_pillow():
    try:
        from PIL import Image  # noqa: F401

        return importlib.import_module("PIL.Image")
    except ImportError:
        raise MissingExtra("images", "Pillow") from None


def require_torch():
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except ImportError:
        raise MissingExtra("train", "PyTorch and torchvision") from None
    require_pillow()
    return importlib.import_module("torch"), importlib.import_module("torchvision")


def install(extra: str) -> None:
    """pip install the packages of `extra` into the running Python."""
    if extra not in EXTRAS:
        raise EmberlensError(f'Unknown extra "{extra}". Use: {", ".join(EXTRAS)}.')
    packages = EXTRAS[extra]
    print(f"Installing {', '.join(packages)} … (this can take a few minutes)")
    has_pip = subprocess.run([sys.executable, "-m", "pip", "--version"], capture_output=True).returncode == 0
    if has_pip:
        cmd = [sys.executable, "-m", "pip", "install", "--quiet", *packages]
    elif shutil.which("uv"):
        cmd = ["uv", "pip", "install", "--python", sys.executable, "--quiet", *packages]
    else:
        subprocess.run([sys.executable, "-m", "ensurepip", "--upgrade"], check=False)
        cmd = [sys.executable, "-m", "pip", "install", "--quiet", *packages]
    if subprocess.run(cmd).returncode != 0:
        raise EmberlensError("The install failed. Check your internet connection and try again.")
    print("Done.")
