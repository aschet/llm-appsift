# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT
"""Everything the stages print, as TAP version 14.

https://testanything.org/tap-version-14-specification.html

Every stage reports the same thing -- a model, the units of work done to it, and
how each one came out -- which is what TAP describes, so every stage prints it as
TAP and nothing prints any other way.

    TAP version 14
    # Subtest: triage
        # Subtest: gemma4:e4b
            ok 1 - triage speed
              ---
              duration_s: 11.8
              detail: "generates 115 tok/s"
              ...
            not ok 2 - triage context
              ---
              duration_s: 88.1
              detail: "dropped part of the prompt at its 32k window"
              ...
            1..2
        not ok 1 - gemma4:e4b
          ---
          result: "rejected at context, 100s"
          ...
        1..1
    not ok 1 - triage
    1..1

Points where the specification decides the shape:

Plans trail their test points. The spec allows either position, and a gate
cascade stops at the first failure, so how many test points a model will produce
is not known when its subtest opens.

A subtest opens only when its first test point arrives. A model already on record
produces no test points -- there is nothing to measure -- so it is written as a
plain test point carrying its stored verdict rather than as an empty subtest.

A commented subtest "must be terminated by a Test Point with a matching
Description", so the parent point repeats the subtest name exactly and carries
nothing else in its description.

The parent point fails when any point inside it failed. Consumers assume that,
and a passing parent over a failing child is the one thing a TAP reader cannot
make sense of.

Diagnostics go in the YAML block, never in the directive slot. The first
unescaped `#` after a description opens a directive, and only TODO and SKIP are
directives, so a duration parked there is not valid TAP. Descriptions escape any
`#` they contain for the same reason.
"""
from __future__ import annotations

import contextlib
import json
import sys

OK = "ok"
FAIL = "FAIL"

VERSION_LINE = "TAP version 14"
STEP = 4


class _Frame:
    """One nesting level: the document, a stage, or a model."""

    def __init__(self, depth: int, name: str = "", kind: str = "root"):
        self.depth = depth
        self.name = name
        self.kind = kind
        self.count = 0
        self.failed = 0


class Reporter:
    """A TAP document, written as the run proceeds."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.stack = [_Frame(0)]
        self.pending = None      # a subject announced but not yet written out
        self.inline = False      # the subject is the subtest already open

    # -- plumbing ---------------------------------------------------------
    def _pad(self, extra: int = 0) -> str:
        return " " * (self.stack[-1].depth * STEP + extra)

    def _say(self, text: str, stream=None) -> None:
        print(text, file=stream or sys.stdout, flush=True)

    def _point(self, ok: bool, description: str, diagnostics: dict | None = None,
               stream=None) -> None:
        frame = self.stack[-1]
        frame.count += 1
        if not ok:
            frame.failed += 1
        self._say(f"{self._pad()}{'ok' if ok else 'not ok'} {frame.count} - "
                  f"{_escape(description)}", stream)
        if diagnostics:
            pad = self._pad(2)
            self._say(f"{pad}---", stream)
            for key, value in diagnostics.items():
                self._say(f"{pad}{key}: "
                          f"{value if isinstance(value, (int, float)) else json.dumps(value)}",
                          stream)
            self._say(f"{pad}...", stream)

    def _push(self, name: str, kind: str, stream=None) -> None:
        self._say(f"{self._pad()}# Subtest: {name}", stream)
        self.stack.append(_Frame(self.stack[-1].depth + 1, name, kind))

    def _close_subject(self, text: str = "", stream=None) -> None:
        frame = self._pop(stream)
        self._point(not frame.failed, frame.name,
                    {"result": text} if text else None, stream)

    def _plan(self, frame: _Frame, stream=None) -> None:
        reason = "" if frame.count else " # no test points"
        self._say(f"{' ' * (frame.depth * STEP)}1..{frame.count}{reason}", stream)

    def _pop(self, stream=None) -> _Frame:
        frame = self.stack.pop()
        self._plan(frame, stream)
        return frame

    # -- what the stages call ---------------------------------------------
    def note(self, text: str, stream=None) -> None:
        """A remark that is not a result. A comment carries no verdict in TAP."""
        self._say(f"{self._pad()}# {text}", stream)

    def subject(self, name: str, note: str = "", stream=None) -> None:
        # A stage that forgets to close one would otherwise nest the next model
        # inside it, and the document would still parse -- wrongly.
        if self.stack[-1].kind == "subject":
            self._close_subject(stream=stream)
        if self.stack[-1].kind == "group" and self.stack[-1].name == name:
            # The pipeline drives the stages one model at a time and has already
            # opened that model's subtest. A stage running inside it adds its
            # points there rather than opening a second subtest of the same name.
            self.inline, self.pending = True, None
            if note:
                self.note(note, stream)
            return
        self.inline, self.pending = False, (name, note)

    def _open_pending(self, stream=None) -> None:
        if self.pending is None:
            return
        name, note = self.pending
        self.pending = None
        self._push(name, "subject", stream)
        if note:
            self.note(note, stream)

    def unit(self, stage: str, name: str, status: str = "",
             seconds: float | None = None, detail: str = "", stream=None,
             **extra) -> None:
        self._open_pending(stream)
        diagnostics = {}
        if seconds is not None:
            diagnostics["duration_s"] = round(seconds, 1)
        diagnostics.update({k: v for k, v in extra.items() if v is not None})
        if detail:
            diagnostics["detail"] = detail
        self._point(status != FAIL, f"{stage} {name}", diagnostics, stream)

    def result(self, text: str, stream=None) -> None:
        """Close the subject: its subtest if one opened, else one plain point.

        A subject that is the enclosing subtest is not closed here -- the stage
        that reported it is one of several running inside it, so its result is a
        remark rather than a verdict on the model.
        """
        if self.inline:
            self.inline = False
            if text:
                self.note(f"result: {text}", stream)
            return
        if self.pending is not None:
            name, note = self.pending
            self.pending = None
            self._point(True, name, {"result": text} if text else None, stream)
            if note:
                self.note(note, stream)
            return
        self._close_subject(text, stream)

    def summary(self, text: str, stream=None) -> None:
        self.note(text, stream)

    def bail(self, reason: str, stream=None) -> None:
        """Nothing further can be measured, so nothing further is claimed."""
        self._say(f"Bail out! {reason}", stream)

    # -- the document -----------------------------------------------------
    @contextlib.contextmanager
    def document(self, stream=None):
        self.reset()
        self._say(VERSION_LINE, stream)
        try:
            yield self
        except (SystemExit, KeyboardInterrupt) as exc:
            # Whatever was running cannot be finished, and the specification has
            # one line for that. No plan follows it: a plan would claim a count for
            # tests that will never run. The message is not re-raised, or the
            # interpreter prints it a second time on stderr.
            self.bail(str(exc) or type(exc).__name__, stream)
            self.reset()
            raise SystemExit(2) from None
        else:
            while len(self.stack) > 1:      # an abort mid-subtest still closes it
                frame = self._pop(stream)
                self._point(not frame.failed, frame.name, stream=stream)
            self._plan(self.stack[0], stream)
        finally:
            # The root frame is replaced rather than popped: a reporter that has
            # closed one document must still be able to write the next.
            self.reset()

    @contextlib.contextmanager
    def group(self, name: str, stream=None):
        """A stage of the pipeline, as a subtest holding one per model."""
        self._push(name, "group", stream)
        try:
            yield self
        finally:
            if self.stack[-1].kind == "subject":
                self._close_subject(stream=stream)
            frame = self._pop(stream)
            self._point(not frame.failed, frame.name, stream=stream)


def _escape(text: str) -> str:
    """`#` opens the directive slot, so a description carrying one escapes it."""
    return text.replace("#", r"\#").replace("\n", " ")


_reporter = Reporter()

reset = lambda *a, **k: _reporter.reset(*a, **k)
note = lambda *a, **k: _reporter.note(*a, **k)
subject = lambda *a, **k: _reporter.subject(*a, **k)
unit = lambda *a, **k: _reporter.unit(*a, **k)
result = lambda *a, **k: _reporter.result(*a, **k)
summary = lambda *a, **k: _reporter.summary(*a, **k)
bail = lambda *a, **k: _reporter.bail(*a, **k)
document = lambda *a, **k: _reporter.document(*a, **k)
group = lambda *a, **k: _reporter.group(*a, **k)
