# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT
"""Mutual exclusion for anything that loads a model.

Two jobs sharing one GPU thrash and silently corrupt each other's timings, so
every stage takes this lock first. fcntl is used on POSIX and msvcrt on Windows;
both provide an advisory exclusive lock released when the process exits.

The lock is per endpoint, so runs against different servers do not block each
other. It is local to one machine: several machines driving the same remote
server cannot observe each other's lock.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time

_handle = None      # module level, so the lock lives as long as the process

if sys.platform == "win32":
    import msvcrt

    def _try_lock(fh) -> bool:
        try:
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False
else:
    import fcntl

    def _try_lock(fh) -> bool:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except (BlockingIOError, OSError):
            return False


def lock_path(endpoint: str = "") -> str:
    slug = "".join(c if c.isalnum() else "-" for c in (endpoint or "default"))[-40:]
    # The name is shared with any other tool of this family on the machine, on
    # purpose: they drive the same GPU, and two jobs loading models at once thrash
    # it and corrupt each other's timings. A private lock would let that happen.
    return os.path.join(tempfile.gettempdir(), f"codesift-gpu-{slug}.lock")


def acquire(label: str = "", *, endpoint: str = "", wait: bool = True,
            poll: float = 5.0):
    global _handle
    path = lock_path(endpoint)
    # Already held by this process. Reopening would rebind the module handle and let
    # the previous file object be collected, and on POSIX closing any descriptor to a
    # file drops that process's lock on it -- so a second acquire would silently
    # release the first. A cascade runs several stages in one process, so this has to
    # be reentrant rather than merely tolerated.
    if _handle is not None and not _handle.closed and _handle.name == path:
        return _handle
    # msvcrt locks a byte range, so the file must be non-empty and seekable.
    _handle = open(path, "a+", encoding="utf-8")
    _handle.seek(0)
    try:
        empty = not _handle.read(1)
    except PermissionError:
        # Another process already holds the byte range through msvcrt, which on
        # Windows denies reads of it too, not only competing locks. That it could
        # be locked at all means it was made non-empty before this process ever
        # opened it.
        empty = False
    if empty:
        _handle.write(" ")
        _handle.flush()
    _handle.seek(0)

    started, announced = time.time(), False
    while True:
        if _try_lock(_handle):
            break
        if not wait:
            print("another job holds the GPU; refusing to run in parallel",
                  file=sys.stderr)
            raise SystemExit(3)
        if not announced:
            print("waiting for the GPU lock held by another job...", flush=True)
            announced = True
        time.sleep(poll)
    if announced:
        print(f"lock acquired after {time.time() - started:.0f}s", flush=True)
    try:
        with open(path + ".owner", "w", encoding="utf-8") as fh:
            fh.write(f"{os.getpid()} {label}\n")
    except OSError:
        pass
    return _handle
