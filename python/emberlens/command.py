"""The command the user can run, for use in messages.

The repo launcher says `./emberlens`; a pip install may be a script at
`~/.emberlens/bin/emberlens` or on the PATH; `python -m emberlens` works anywhere.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Mapping, Optional


def program_name(argv0: Optional[str] = None, env: Optional[Mapping[str, str]] = None) -> str:
    env = os.environ if env is None else env
    if env.get("EMBERLENS_PROG"):
        return env["EMBERLENS_PROG"]
    argv0 = sys.argv[0] if argv0 is None else argv0
    if not argv0 or Path(argv0).name == "__main__.py":
        return "python -m emberlens"
    found = shutil.which(os.path.basename(argv0))
    if found and os.path.realpath(found) == os.path.realpath(argv0):
        return os.path.basename(argv0)
    home = str(Path.home())
    if argv0 == home or argv0.startswith(home + os.sep):
        return "~" + argv0[len(home):].replace(os.sep, "/")
    return argv0


def command(*parts: str) -> str:
    return " ".join([program_name(), *parts])
