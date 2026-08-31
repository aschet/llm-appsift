# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT
"""Re-apply the current checks to applications already on disk.

A maintenance operation on this tool's own output, not something a user runs day
to day: reached as `python -m appsift.recheck`, off the primary `appsift` command,
the same way codesift keeps its own regrade stage off `codesift run`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import harness, ledger, progress
from .cli import EXECUTION_WARNING, LEDGER
from .config import DEFAULT_HOST, DEFAULT_OUTPUT, Config, data_dir, read_model_file
from .task import TASK


def _from_session(rec: dict) -> dict:
    """Fill in what only opencode's store can answer, for results recorded before
    the harness read it: what the model said, and how its final turn ended."""
    from . import session as sessions
    sid = sessions.resolve(rec)
    if not sid:
        return rec
    voice = sessions.spoke(sid)
    speed = sessions.generation(sid)
    out = dict(rec, session=sid)
    if voice["said"] or voice["produced"]:
        out.update(said=voice["said"] or rec.get("said", ""),
                   final_produced=voice["produced"], final_acted=voice["acted"])
    if speed["gen_tok_s"]:
        out.update(gen_tok_s=speed["gen_tok_s"], gen_s=speed["gen_s"],
                   output_tokens=speed["output_tokens"])
    out.pop("silent_finish", None)
    return out


def run(cfg: Config, models: list[str], apply: bool, stream=None) -> int:
    """Grade every kept application again and report what the new checks decide.

    Reads nothing but the ledger and the applications it names, so a correction to
    the checking code is applied for the cost of running it, not the cost of the
    sweep that produced them. Nothing is written without --apply, and the ledger
    it replaces is kept beside it.
    """
    out = stream or sys.stdout
    path = cfg.path(LEDGER)
    records = ledger.read(path)
    if not records:
        progress.note("nothing has been run yet", stream=out)
        return 0

    updated, changed, gone = [], 0, 0
    for rec in records:
        if models and rec.get("model") not in models:
            updated.append(rec)
            continue
        progress.subject(rec.get("model", "?"), stream=out)
        fresh = harness.recheck(TASK, rec)
        if fresh is not None:
            fresh = _from_session(fresh)
        if fresh is None:
            gone += 1
            progress.result("application no longer on disk, left as recorded", stream=out)
            updated.append(rec)
            continue
        was = sum(1 for c in rec.get("checks") or [] if c["passed"])
        now = sum(1 for c in fresh["checks"] if c["passed"])
        total = len(fresh["checks"])
        # Compared check by check, not by the total. Two corrections that happen to
        # offset each other leave the count identical, and reporting that as
        # unchanged hides both.
        before = {c["name"]: c["passed"] for c in rec.get("checks") or []}
        moved = [("+" if c["passed"] else "-") + c["name"] for c in fresh["checks"]
                 if c["name"] in before and c["passed"] != before[c["name"]]]
        for c in fresh["checks"]:
            progress.unit("recheck", c["name"],
                          progress.OK if c["passed"] else progress.FAIL,
                          detail=c["detail"][:60], stream=out)
        if moved or was != now:
            changed += 1
            progress.result(f"{was} of {total} becomes {now} of {total}"
                            + (f", {', '.join(moved[:6])}" if moved else ""),
                            stream=out)
        else:
            progress.result(f"unchanged at {now} of {total}", stream=out)
        updated.append(fresh)

    progress.summary(f"{changed} result(s) would change"
                     + (f", {gone} application(s) missing" if gone else ""), stream=out)
    if not apply:
        progress.note("nothing written; pass --apply to rewrite the ledger",
                      stream=out)
        return 0
    backup = path.with_suffix(path.suffix + ".bak")
    backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    ledger.write(path, updated)
    progress.note(f"rewrote {path}; previous ledger kept as {backup.name}",
                  stream=out)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="appsift.recheck",
        description="Re-apply the current checks to applications already on disk, "
                    "without running any model."
                    f"\n\n{EXECUTION_WARNING}",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", metavar="DIR",
                   help=f"where the ledger and applications are kept (default: "
                        f"{data_dir(DEFAULT_OUTPUT)})")
    p.add_argument("--models", nargs="+", metavar="MODEL",
                   help="only these models (default: every model in the ledger)")
    p.add_argument("--models-file", metavar="PATH",
                   help="file listing one model per line; # comments allowed")
    p.add_argument("--apply", action="store_true",
                   help="rewrite the ledger; the previous one is kept as .bak")
    args = p.parse_args(argv)

    models = args.models or (read_model_file(args.models_file)
                             if args.models_file else [])
    results_dir = Path(args.results_dir) if args.results_dir else data_dir(DEFAULT_OUTPUT)
    cfg = Config(host=DEFAULT_HOST, results_dir=results_dir, models=models)
    with progress.document():
        return run(cfg, models, args.apply)


if __name__ == "__main__":
    raise SystemExit(main())
