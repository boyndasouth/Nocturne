"""Read a top-level JSON array one element at a time, standard library only.

LongMemEval-S is a ~280 MB array; json.load holds the text and the whole parsed tree at
once (~2.4 GB peak), which fails on a laptop with other apps open. Streaming keeps one
element in memory at a time, plus whatever the caller keeps.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

_WS = " \t\n\r"


def iter_json_array(path: str | Path, *, chunk_size: int = 1 << 20) -> Iterator[Any]:
    decoder = json.JSONDecoder()
    with open(path, encoding="utf-8") as f:
        buf, pos, eof = "", 0, False

        def fill() -> None:
            nonlocal buf, pos, eof
            more = f.read(chunk_size)
            if not more:
                eof = True
            buf, pos = buf[pos:] + more, 0

        def peek() -> str:
            """Next non-whitespace character ('' at end of file)."""
            nonlocal pos
            while True:
                while pos < len(buf) and buf[pos] in _WS:
                    pos += 1
                if pos < len(buf):
                    return buf[pos]
                if eof:
                    return ""
                fill()

        fill()
        if buf.startswith("﻿"):
            pos = 1
        if peek() != "[":
            raise ValueError(f"{path}: expected a JSON array")
        pos += 1
        if peek() == "]":
            return
        while True:
            try:
                item, end = decoder.raw_decode(buf, pos)
            except json.JSONDecodeError:
                if eof:
                    raise
                fill()
                continue
            # A number cut by a chunk edge ("-0" of "-0.5e10") decodes fine, so only accept a
            # value once the separator after it has been read.
            if not eof and (end >= len(buf) or buf[end] not in _WS + ",]"):
                fill()
                continue
            yield item
            pos = end
            sep = peek()
            if sep == "]":
                return
            if sep != ",":
                raise ValueError(f"{path}: expected ',' or ']' after element, got {sep!r}")
            pos += 1
            peek()
