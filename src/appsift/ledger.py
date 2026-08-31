# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT
"""JSON Lines stores, which is how every stage keeps what it measured.

One record per line. A line that will not parse is skipped rather than fatal,
so a run killed mid-write costs only the record it was writing.

Every attempt at a model is kept side by side, on purpose: append() adds one
without touching what is already there, and which of them is the one a card
or a ranking is built from is decided when the file is read, not when it is
written.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


def read(path: str | Path) -> list[dict]:
    """Every record on file, in order."""
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def write(path: str | Path, records) -> None:
    """Replace the file with these records, in one step.

    Written beside the store and renamed over it. A rename is atomic, so a reader
    sees either every record or none of the new ones, and a process killed during
    the write leaves the store as it was.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    os.replace(tmp, path)


def append(path: str | Path, rec: dict) -> None:
    """Add one record without touching what is already on file.

    A line written once, never a rewrite of what came before it, so it carries
    none of the hazard write() guards against and needs none of its protection.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
