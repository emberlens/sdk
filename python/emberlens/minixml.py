"""A tiny XML reader for Pascal VOC files.

Used only when Python's own XML parser (expat) is missing or broken, which
happens on some Homebrew Python installs. It builds the same
xml.etree.ElementTree elements, so the VOC reader works the same way.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

_TOKEN = re.compile(r"<!--.*?-->|<\?.*?\?>|<!\[CDATA\[(.*?)\]\]>|<!.*?>|<(/?)([A-Za-z_][\w.\-:]*)([^>]*?)(/?)>|([^<]+)", re.S)
_ENTITIES = {"amp": "&", "lt": "<", "gt": ">", "quot": '"', "apos": "'"}


def _unescape(text: str) -> str:
    def repl(m):
        name = m.group(1)
        if name.startswith("#x"):
            return chr(int(name[2:], 16))
        if name.startswith("#"):
            return chr(int(name[1:]))
        return _ENTITIES.get(name, m.group(0))

    return re.sub(r"&(#x[0-9A-Fa-f]+|#\d+|\w+);", repl, text)


def parse_string(text: str) -> ET.Element:
    root = None
    stack = []
    for m in _TOKEN.finditer(text):
        cdata, closing, tag, _attrs, selfclose, chars = m.groups()
        if tag:
            if closing:
                if not stack or stack[-1].tag != tag:
                    raise ValueError(f"unexpected </{tag}>")
                stack.pop()
                continue
            el = ET.Element(tag)
            if stack:
                stack[-1].append(el)
            elif root is None:
                root = el
            if not selfclose:
                stack.append(el)
        elif (chars is not None or cdata is not None) and stack:
            piece = cdata if cdata is not None else _unescape(chars)
            el = stack[-1]
            if len(el):
                el[-1].tail = (el[-1].tail or "") + piece
            else:
                el.text = (el.text or "") + piece
    if root is None or stack:
        raise ValueError("not a complete XML document")
    return root
