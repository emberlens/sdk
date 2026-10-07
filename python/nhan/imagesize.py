"""Read width and height from JPEG / PNG headers, without Pillow."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Optional, Tuple


def image_size(path: Path) -> Optional[Tuple[int, int]]:
    """(width, height), or None if the file is not a JPEG or PNG we can read."""
    try:
        with open(path, "rb") as f:
            head = f.read(26)
            if head[:8] == b"\x89PNG\r\n\x1a\n":
                w, h = struct.unpack(">II", head[16:24])
                return int(w), int(h)
            if head[:2] != b"\xff\xd8":
                return None
            f.seek(2)
            while True:
                marker = f.read(2)
                if len(marker) < 2 or marker[0] != 0xFF:
                    return None
                while marker[1] == 0xFF:  # fill bytes
                    marker = marker[:1] + f.read(1)
                code = marker[1]
                if code in (0xD8, 0x01) or 0xD0 <= code <= 0xD7:
                    continue
                length = struct.unpack(">H", f.read(2))[0]
                # SOF0..SOF15 except DHT (C4), JPG (C8), DAC (CC)
                if 0xC0 <= code <= 0xCF and code not in (0xC4, 0xC8, 0xCC):
                    _precision, h, w = struct.unpack(">BHH", f.read(5))
                    return int(w), int(h)
                f.seek(length - 2, 1)
    except (OSError, struct.error):
        return None
