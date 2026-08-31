# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT
"""The conversation opencode had with the model, read back from its own store.

The harness records what a session did -- turns, tools, tokens -- but not what was
said in it, and a session that ended having written nothing is exactly the one
worth reading. opencode keeps every session in a SQLite database of its own, so
the transcript is recoverable rather than something this tool has to duplicate.

Two ways in. A result recorded since the harness started keeping `session` is
looked up by id. An older result has no id, so it is matched on the three things
that identify a run anyway: the model, the directory the harness gave opencode,
and the moment the run started, which is the recorded finish time less its
duration. The database is opened read-only; nothing here writes to it.
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

# How far the recorded start of a run may sit from the session's own creation
# time and still be the same run. Generous enough for the model load that
# precedes the first event, tight enough that two runs of one model do not
# collide: the closest pair observed sat a minute apart.
MATCH_WINDOW_S = 240


def db_path() -> Path:
    """Where opencode keeps its store, honouring the usual overrides."""
    explicit = os.environ.get("OPENCODE_DB")
    if explicit:
        return Path(explicit)
    base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return Path(base) / "opencode" / "opencode.db"


def _connect(path: Path | None = None):
    path = Path(path or db_path())
    if not path.exists():
        return None
    try:
        # Read-only, and immutable=0 so an open opencode's WAL is still visible.
        return sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error:
        return None


def _model_of(raw) -> str:
    """`session.model` is a JSON object in newer versions and a bare id in older."""
    if not raw:
        return ""
    try:
        return json.loads(raw).get("id") or ""
    except Exception:
        return str(raw)


def resolve(rec: dict, conn=None) -> str:
    """The session id for this result, by id when it has one and by match when not."""
    if rec.get("session"):
        return rec["session"]
    own = conn or _connect()
    if own is None:
        return ""
    try:
        # The repository may have been moved since the run -- these ledgers outlive
        # the directories they name -- so the leaf, which is model and task, is what
        # is matched rather than the whole path.
        leaf = Path(rec.get("repo") or "").name
        if not leaf:
            return ""
        started = (rec.get("ts") or 0) - (rec.get("wall_s") or 0)
        best, best_gap = "", None
        for sid, directory, model, created in own.execute(
                "select id, directory, model, time_created from session"):
            if Path(directory or "").name != leaf:
                continue
            if _model_of(model) != rec.get("model"):
                continue
            gap = abs((created or 0) / 1000 - started)
            if gap <= MATCH_WINDOW_S and (best_gap is None or gap < best_gap):
                best, best_gap = sid, gap
        return best
    except sqlite3.Error:
        return ""
    finally:
        if conn is None:
            own.close()


def generation(session_id: str, conn=None) -> dict:
    """How fast the model actually generated, as opposed to how long the run took.

    Elapsed time is not a speed measurement: it counts the model's own generation,
    the prefill before each turn, and every second the tools spent running tests and
    starting servers. A model that quits early looks fast by that measure. opencode
    timestamps each assistant message, and tools run between messages, so summing
    those intervals against the output tokens gives a rate that quitting does not
    flatter. Prefill is included, which is fair: every model pays it, and it grows
    with the context each one chooses to accumulate.
    """
    empty = dict(output_tokens=0, gen_s=0.0, gen_tok_s=None)
    if not session_id:
        return empty
    own = conn or _connect()
    if own is None:
        return empty
    try:
        out = 0
        secs = 0.0
        for (data,) in own.execute(
                "select data from message where session_id=?", (session_id,)):
            try:
                msg = json.loads(data)
            except Exception:
                continue
            if msg.get("role") != "assistant":
                continue
            when = msg.get("time") or {}
            start, done = when.get("created"), when.get("completed")
            if not start or not done or done < start:
                continue
            secs += (done - start) / 1000
            out += (msg.get("tokens") or {}).get("output") or 0
        if not secs or not out:
            return empty
        return dict(output_tokens=out, gen_s=round(secs, 1),
                    gen_tok_s=round(out / secs, 1))
    except sqlite3.Error:
        return empty
    finally:
        if conn is None:
            own.close()


def spoke(session_id: str, conn=None) -> dict:
    """What the model said, and whether its last turn produced anything at all.

    opencode's event stream carries text and tool calls but never reasoning, so a
    harness reading only the stream loses the reasoning entirely -- and cannot tell
    a model that thought and then stopped from one whose output it could not parse.
    The store has both, so the question is asked of the store.

    `acted` is whether the final turn called a tool. `produced` is whether it
    yielded anything at all, reasoning included. A turn that produced something but
    did not act is a model that stopped mid-thought; one that produced nothing is a
    model whose output arrived as nothing the harness could use.
    """
    empty = dict(said="", acted=False, produced=False, final_output_tokens=0)
    if not session_id:
        return empty
    own = conn or _connect()
    if own is None:
        return empty
    try:
        rows = list(own.execute(
            "select id, data from message where session_id=? order by time_created",
            (session_id,)))
        said, last = [], None
        for mid, data in rows:
            try:
                meta = json.loads(data)
            except Exception:
                continue
            if meta.get("role") != "assistant":
                continue
            kinds, texts = [], []
            for (pd,) in own.execute(
                    "select data from part where message_id=? order by time_created",
                    (mid,)):
                try:
                    part = json.loads(pd)
                except Exception:
                    continue
                kind = part.get("type")
                kinds.append(kind)
                if kind in ("text", "reasoning") and part.get("text"):
                    texts.append(part["text"])
            said.extend(texts)
            last = dict(kinds=kinds, texts=texts,
                        out=(meta.get("tokens") or {}).get("output") or 0)
        if last is None:
            return empty
        return dict(said="\n\n".join(said),
                    acted="tool" in last["kinds"],
                    produced=bool(last["texts"]) or "tool" in last["kinds"],
                    final_output_tokens=last["out"])
    except sqlite3.Error:
        return empty
    finally:
        if conn is None:
            own.close()


def transcript(session_id: str, conn=None) -> list[dict]:
    """The session as an ordered list of turns.

    Each turn is {role, model, parts}, and each part is one of text, reasoning or
    tool. Tool parts keep their input and output: a session that went wrong is
    usually explained by what a tool returned, not by what the model said about it.
    """
    if not session_id:
        return []
    own = conn or _connect()
    if own is None:
        return []
    try:
        by_message = {}
        for mid, data in own.execute(
                "select message_id, data from part where session_id=? order by time_created",
                (session_id,)):
            try:
                part = json.loads(data)
            except Exception:
                continue
            kind = part.get("type")
            if kind == "text" and part.get("text"):
                by_message.setdefault(mid, []).append(
                    dict(kind="text", text=part["text"]))
            elif kind == "reasoning" and part.get("text"):
                by_message.setdefault(mid, []).append(
                    dict(kind="reasoning", text=part["text"]))
            elif kind == "tool":
                state = part.get("state") or {}
                by_message.setdefault(mid, []).append(dict(
                    kind="tool", tool=part.get("tool") or "",
                    status=state.get("status") or "",
                    title=state.get("title") or "",
                    input=state.get("input"),
                    output=state.get("output"),
                    error=state.get("error")))
        turns = []
        for mid, data in own.execute(
                "select id, data from message where session_id=? order by time_created",
                (session_id,)):
            try:
                meta = json.loads(data)
            except Exception:
                meta = {}
            parts = by_message.get(mid) or []
            if not parts:
                continue
            turns.append(dict(role=meta.get("role") or "assistant",
                              model=meta.get("modelID") or "", parts=parts))
        return turns
    except sqlite3.Error:
        return []
    finally:
        if conn is None:
            own.close()
