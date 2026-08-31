# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT
"""The command: run across models, then say what each of them managed."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import harness, ledger, progress, report
from .config import DEFAULT_HOST, DEFAULT_OUTPUT, Config, data_dir, read_model_file
from .task import TASK

LEDGER = "applications.jsonl"


def summarise(cfg: Config, models: list[str], stream=None) -> None:
    """One line per model, ordered by how much of the specification was met."""
    out = stream or sys.stdout
    path = Path(cfg.results_dir) / LEDGER
    rows = [rec for rec in ledger.read(path) if not models or rec["model"] in models]
    if not rows:
        progress.note("nothing has been run yet", stream=out)
        return

    # The same attempt harness.primary() would put on the report, so this
    # line and the page never disagree about the same model.
    by_model = {}
    for rec in rows:
        by_model.setdefault(rec["model"], []).append(rec)
    best = {model: harness.primary(attempts) for model, attempts in by_model.items()}

    ranked = sorted(best.items(), key=lambda kv: -harness.score(kv[1]))
    progress.summary(f"{len(ranked)} model(s) run", stream=out)
    for model, rec in ranked:
        checks = rec.get("checks") or []
        won, total = harness.met(rec)
        met = f"{won:g} of {total}" if total else "nothing checked"
        missed = [c["name"] for c in checks if not c["passed"]]
        progress.note(f"{model}: {met} in {rec.get('wall_s') or 0:.0f}s"
                      + (f", missing {', '.join(missed[:6])}" if missed else "")
                      + (f" and {len(missed) - 6} more" if len(missed) > 6 else "")
                      + (f"; kept at {rec['repo']}" if rec.get("repo") else ""),
                      stream=out)


# Printed by --help and said again at the point of risk, not only in the README.
# Wrapped here rather than by argparse: the parser that prints it uses the raw
# formatter, which leaves a paragraph on one line. The wording is codesift's own,
# unchanged: both tools run what a model writes with no sandbox, and the same
# sentence is true of either.
EXECUTION_WARNING = (
    "WARNING: model-written code is executed without a sandbox. It runs with the\n"
    "privileges of the invoking user and with unrestricted access to the filesystem\n"
    "and the network, and can damage the host system. Running the harness inside a\n"
    "virtual machine is strongly recommended.")


def _archive(cfg: Config, path: Path, out) -> None:
    """Move a previous sweep aside before starting a new one.

    Results are chosen per model by best score, which is right for repeated
    attempts within one sweep and wrong across two: a result from before a change
    to the task or the grader would outrank the run that replaced it, and nothing
    on the page would say so. Moved rather than deleted, because a sweep costs
    hours and the judgement to throw it away is not this tool's to make.
    """
    applications = cfg.results_dir / "applications"
    has_applications = applications.is_dir() and any(applications.iterdir())
    if not path.exists() and not has_applications:
        return
    archive = cfg.results_dir / f"sweep-{time.strftime('%Y%m%d-%H%M%S')}"
    archive.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.rename(archive / path.name)
    if has_applications:
        applications.rename(archive / applications.name)
    progress.note(f"previous sweep moved to {archive}", stream=out)


def run(cfg: Config, timeout: int, redo: bool, stream=None, variance: int = 0,
        html: Path | None = None) -> int:
    """Run the task for each model, appending to this tool's own ledger.

    Every attempt runs at the model's own default sampling; none is forced to
    temperature 0, since that mode is not one most models were tuned against
    and does not buy real reproducibility in a multi-turn agentic session
    anyway. `variance` runs further attempts so a trajectory that only went
    the way it did once can be told apart from one the model reaches
    reliably -- the report ranks on whichever attempt scored highest.
    Resuming counts existing attempts per model rather than treating any
    record as done, so raising `variance` after an earlier run tops up what
    is missing instead of starting over.

    `html` is rewritten after every attempt, not only once the whole sweep
    finishes: a run costs hours, and a reader checking in on it deserves
    something newer than what the sweep looked like before it started.
    """
    out = stream or sys.stdout
    models = cfg.resolve_models()
    problem = harness.preflight()
    if problem:
        print(problem, file=sys.stderr)
        return 2

    target = 1 + variance
    path = cfg.path(LEDGER)
    have, best_so_far = {}, {}
    if redo:
        _archive(cfg, path, out)
    else:
        for rec in ledger.read(path):
            if "model" in rec:
                m = rec["model"]
                have[m] = have.get(m, 0) + 1
                best_so_far[m] = max(best_so_far.get(m, -1.0), harness.score(rec) / 100)

    from . import gpulock
    from .ollama import Ollama
    gpulock.acquire("appsift", endpoint=cfg.host)
    client = Ollama(cfg.host, cfg.timeout)
    workdir = cfg.results_dir / "app_work"
    retain = cfg.results_dir / "applications"

    pending = [m for m in models if have.get(m, 0) < target]
    if pending:
        needed = sum(target - have.get(m, 0) for m in pending)
        budget = needed * max(timeout, TASK.get("min_timeout", 0))
        for line in EXECUTION_WARNING.splitlines():
            progress.note(line, stream=out)
        progress.note(f"{len(pending)} model(s) to run, {needed} attempt(s) "
                      f"total; at worst {budget / 3600:.1f}h if every one runs "
                      f"to its limit", stream=out)

    for model in models:
        already = have.get(model, 0)
        if already >= target:
            # Nothing to run; the verdict is what the reader came for and it
            # reads the same whether it was reached now or last week.
            progress.subject(model, stream=out)
            progress.result("already run", stream=out)
            continue
        needed = target - already
        progress.subject(model,
                         "running" if needed == 1 else f"running, {needed} attempts needed",
                         stream=out)
        for attempt in range(already + 1, target + 1):
            # Only one screenshot is ever shown per model, so once an earlier
            # attempt has set a bar, a later one that will not clear it should
            # not spend a browser launch on a picture nobody will see.
            rec = harness.run_task(model, TASK, workdir, timeout, retain_dir=retain,
                                   host=cfg.host, attempt=attempt,
                                   screenshot_floor=best_so_far.get(model))
            # Read while the model is still loaded, before it is asked to unload.
            rec["context_length"] = client.loaded_context(model)
            best_so_far[model] = max(best_so_far.get(model, -1.0), harness.score(rec) / 100)
            ledger.append(path, rec)
            if html:
                report.write(cfg, html, models)
            for c in rec.get("checks") or []:
                progress.unit("check", c["name"],
                              progress.OK if c["passed"] else progress.FAIL,
                              detail=c["detail"][:60], stream=out)
            if rec.get("screenshot"):
                progress.note(f"screenshot {rec['screenshot']}", stream=out)
            won, total_checks = harness.met(rec)
            prefix = f"attempt {attempt}: " if target > 1 else ""
            line = (f"{prefix}{won:g} of {total_checks} requirements met in "
                   f"{rec.get('wall_s') or 0:.0f}s"
                   + (", timed out" if rec.get("timed_out") else ""))
            # One subject was opened per model, not per attempt, so only the
            # last attempt may close it; an earlier one is a remark, not a
            # result, or the subject would be closed twice over.
            if attempt == target:
                progress.result(line, stream=out)
            else:
                progress.note(line, stream=out)
        client.unload(model)
    return 0


def main(argv: list[str] | None = None) -> int:
    """The one thing an end user does with this tool: run it against some
    models, or every model on the server, and get a report.

    Regrading a correction into results already on disk, or rendering the
    report again without running anything, are maintenance operations on
    this tool's own output rather than something a user runs day to day --
    `python -m appsift.recheck` and `python -m appsift.report` reach them,
    same as codesift keeps its own stages off the primary command.
    """
    p = argparse.ArgumentParser(
        prog="appsift",
        description="Create an application from a specification, across models."
                    f"\n\n{EXECUTION_WARNING}",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default=DEFAULT_HOST,
                   help="Ollama server URL (default: %(default)s, or $OLLAMA_HOST)")
    p.add_argument("--models", nargs="+", metavar="MODEL",
                   help="models to run (default: every model on the server)")
    p.add_argument("--models-file", metavar="PATH",
                   help="file listing one model per line; # comments allowed")
    p.add_argument("--results-dir", metavar="DIR",
                   help=f"where applications and the ledger are kept (default: "
                        f"{data_dir(DEFAULT_OUTPUT)}, beside the report, or "
                        f"beside -o if given)")
    p.add_argument("--timeout", type=int, default=2400, metavar="SECONDS")
    p.add_argument("--task-timeout", type=int, default=1200, metavar="SECONDS",
                   help="per model; the task raises this to its own floor")
    p.add_argument("--redo", action="store_true", help="run again for every model")
    p.add_argument("--variance", type=int, default=0, metavar="N",
                   help="N further attempts per model, at the model's own "
                        "default sampling like every attempt, beyond the "
                        "one every model always gets (default: %(default)s)")
    p.add_argument("-o", "--output", default=DEFAULT_OUTPUT, metavar="PATH",
                   help="where to write the report; the records are stored "
                        "beside it (default: %(default)s)")
    p.add_argument("--spec", action="store_true",
                   help="print the specification models are given and stop")
    args = p.parse_args(argv)

    if args.spec:
        print(TASK["files"]["SPEC.md"])
        return 0

    models = args.models or (read_model_file(args.models_file)
                             if args.models_file else [])
    output = Path(args.output)
    results_dir = Path(args.results_dir) if args.results_dir else data_dir(output)
    cfg = Config(host=args.host, results_dir=results_dir,
                 models=models, timeout=args.timeout)
    # One TAP document per invocation: the reader gets the version line, the
    # points, and the plan, and nothing else prints.
    with progress.document():
        code = run(cfg, args.task_timeout, args.redo, variance=args.variance,
                  html=output)
        # A preflight failure means nothing was ever attempted, so there is
        # nothing for the report to show yet -- writing one anyway would only
        # be the empty placeholder page, for a reader who already has the
        # real problem on stderr.
        if code == 0:
            summarise(cfg, models)
            progress.note(f"wrote {report.write(cfg, output, models)}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
