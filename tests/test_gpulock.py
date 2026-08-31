# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT
"""The lock that stops two jobs loading models onto one card at once.

The Windows half of it cannot be reached from the platform it is developed on,
so it is exercised here with the locking primitive stood in for. What that
catches is the half that is ours: which file is opened, that a cascade may take
the lock it already holds, and that the path is one Windows can name.
"""
import importlib
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

from appsift import gpulock


class FakeMsvcrt(types.ModuleType):
    """Enough of msvcrt to answer the one call the lock makes of it."""

    LK_NBLCK = 2

    def __init__(self):
        super().__init__("msvcrt")
        self.held = set()

    def locking(self, fd, mode, nbytes):
        if fd in self.held:
            raise OSError("another process holds this byte")
        self.held.add(fd)


class TestTheLockPath(unittest.TestCase):
    def test_an_endpoint_becomes_a_name_any_filesystem_accepts(self):
        name = os.path.basename(gpulock.lock_path("http://localhost:11434"))
        self.assertEqual(name, "codesift-gpu-http---localhost-11434.lock")
        self.assertFalse(set(name) & set(':*?"<>|'), "Windows rejects these")

    def test_two_servers_do_not_share_a_lock(self):
        self.assertNotEqual(gpulock.lock_path("http://a:11434"),
                            gpulock.lock_path("http://b:11434"))


class TestOnWindows(unittest.TestCase):
    """The branch chosen at import time when the platform is Windows."""

    def setUp(self):
        # Registered before the patches, so it runs after they are undone:
        # cleanups run last in first out, and a reload while the stubs are still
        # in place would leave every later test holding them.
        self.addCleanup(self.restore)
        self.tmp = self.enterContext(tempfile.TemporaryDirectory())
        # Registered right after the directory, so it runs right before that
        # directory is removed: on Windows a file still held open cannot be
        # deleted, and acquire() never closes the module-level handle itself
        # (by design -- it is meant to outlive the call and be released only
        # when the process exits).
        self.addCleanup(self._close_handle)
        self.enterContext(mock.patch.dict(os.environ, {"TMPDIR": self.tmp}))
        self.enterContext(mock.patch.object(tempfile, "gettempdir", lambda: self.tmp))
        self.enterContext(mock.patch.dict(sys.modules, {"msvcrt": FakeMsvcrt()}))
        self.enterContext(mock.patch.object(sys, "platform", "win32"))
        importlib.reload(gpulock)

    def _close_handle(self):
        if gpulock._handle is not None:
            gpulock._handle.close()

    def restore(self):
        gpulock._handle = None
        importlib.reload(gpulock)
        self.assertNotEqual(gpulock._try_lock.__module__, "tests.test_gpulock")

    def test_the_lock_is_taken_through_msvcrt(self):
        handle = gpulock.acquire("screen", endpoint="http://localhost:11434")
        self.assertFalse(handle.closed)
        self.assertTrue(sys.modules["msvcrt"].held, "no byte was locked")

    def test_a_cascade_may_take_the_lock_it_already_holds(self):
        # Several stages run in one process. Reopening the file would drop the
        # lock the first stage took, so the second must get the same handle.
        first = gpulock.acquire("triage", endpoint="http://localhost:11434")
        second = gpulock.acquire("screen", endpoint="http://localhost:11434")
        self.assertIs(first, second)

    def test_the_file_is_never_empty_before_a_byte_range_is_locked(self):
        gpulock.acquire("probe", endpoint="http://localhost:11434")
        path = gpulock.lock_path("http://localhost:11434")
        self.assertGreater(os.path.getsize(path), 0)

    def test_a_contended_lock_denies_the_seed_read_too(self):
        # Observed for real: msvcrt's byte-range lock also blocks a plain read
        # of that range from another handle, not only a competing lock
        # attempt. The check for whether the file needs seeding read it first
        # and let that PermissionError escape uncaught, so every job but the
        # very first to touch a lock file crashed instead of waiting for it.
        real_open = open

        def flaky_open(path, mode, encoding=None):
            fh = real_open(path, mode, encoding=encoding)
            denied = []

            def read(n=-1):
                if not denied:
                    denied.append(True)
                    raise PermissionError("denied by another process's lock")
                return type(fh).read(fh, n)

            fh.read = read
            return fh

        with mock.patch("appsift.gpulock.open", flaky_open):
            handle = gpulock.acquire("probe", endpoint="http://localhost:11434")
        self.assertFalse(handle.closed)
        self.assertTrue(sys.modules["msvcrt"].held, "no byte was locked")


if __name__ == "__main__":
    unittest.main()
