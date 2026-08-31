# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT
"""Driving a model through a task with opencode, and grading what it left behind.

The repository is seeded, the harness is pointed at it, and the result is judged
on the files that exist afterwards and on a check script run against them. What
the model said about its own work is disregarded entirely.

The task is graded per requirement: the check script reports one line each, and
the score is the fraction met.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .config import DEFAULT_HOST


OPENCODE_CONFIG = Path.home() / ".config" / "opencode" / "opencode.jsonc"

# How much of the model's own output is kept with each result. A session that ends
# having written nothing is only diagnosable from what the model said instead.
SAID_KEPT = 8000

# The harness's raw stream, written beside a retained repository.
RAW_SESSION = 2_000_000


def _agent_config(host: str, model: str) -> str:
    """The one-shot opencode configuration this run needs, passed inline rather
    than written to the user's own file.

    opencode does not discover Ollama models by itself: a provider block supplies
    the endpoint but not the model list, so `ollama/<model>` would not resolve
    unless it were declared somewhere. OPENCODE_CONFIG_CONTENT is opencode's own
    mechanism for exactly this -- configuration is merged rather than replaced,
    so this declares only the provider and the one model this run needs, and
    whatever the user already has stays untouched. Nothing is written to disk,
    so there is nothing to back up and nothing to leave behind.

    Every attempt runs at the model's own default sampling. A forced
    temperature of 0 was tried and dropped: it is not a mode most chat models
    were tuned against, it pushed at least one into abandoning the task after
    only planning it out loud, and a multi-turn agentic session has enough
    other sources of drift -- GPU floating-point non-determinism, timestamps
    and other non-deterministic tool output feeding back into later turns --
    that it would not have bought real reproducibility anyway.
    """
    return json.dumps({
        "provider": {
            "ollama": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Ollama Local",
                "options": {"baseURL": f"{host}/v1"},
                "models": {model: {"name": model}},
            }
        }
    })


def preflight() -> str | None:
    """Return an error message if the harness cannot reach opencode at all."""
    if not shutil.which("opencode"):
        return ("opencode is not on PATH. The agent stage needs it to drive the model; "
                "the other stages have no such dependency.")
    return denied_tools()


def _strip_comments(text: str) -> str:
    """jsonc to json, leaving '//' inside string literals untouched."""
    return re.sub(r'(^|\s)//(?!/).*$', r'\1', text, flags=re.M)


def denied_tools(config_path: Path | None = None) -> str | None:
    """Report tools the configuration refuses outright.

    The stage runs opencode with --auto, which approves anything not explicitly
    denied, so a task fails only if the user's own configuration forbids the tool
    it needs. That failure looks exactly like an incapable model, so it is caught
    here instead.
    """
    path = Path(config_path) if config_path else OPENCODE_CONFIG
    try:
        data = json.loads(_strip_comments(path.read_text(encoding="utf-8")))
    except Exception:
        return None
    perms = data.get("permission")
    if not isinstance(perms, dict):
        return None
    denied = sorted(k for k, v in perms.items()
                    if (v if isinstance(v, str) else (v or {}).get("*")) == "deny")
    if not denied:
        return None
    return (f"{path} denies: {', '.join(denied)}. The agent stage cannot edit or run "
            "anything it is refused, and every task would fail for that reason "
            "rather than for anything about the model. Remove those entries or set "
            "them to 'allow'.")


def _digest(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError:
        return None


def seed(task: dict, workdir: Path, at: Path | None = None) -> Path:
    """Create the task's starting repository and return its absolute path.

    Absolute because everything downstream runs with the repository as its working
    directory: a relative path handed to one of those resolves against the
    repository rather than against here, which produced a doubled path and a
    grader that failed to find its own check script.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    if at is not None:
        at = Path(at).resolve()
        shutil.rmtree(at, ignore_errors=True)
        at.mkdir(parents=True)
        repo = at
    else:
        repo = Path(tempfile.mkdtemp(prefix=f"{task['id']}_", dir=workdir))
    repo = repo.resolve()
    for rel, content in task["files"].items():
        target = repo / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return repo


# A requirement met in full, not attempted, or a real but incomplete attempt --
# worth something between the two rather than nothing. WARN is not a third kind
# of pass; `passed` stays False for it, so it never lets a model reach complete.
SCORE = {"PASS": 1.0, "WARN": 0.5, "FAIL": 0.0}


def parse_checks(output: str) -> list[dict]:
    """Read `CHECK <name> PASS|WARN|FAIL <detail>` lines from a graded check script."""
    checks = []
    for line in output.splitlines():
        parts = line.strip().split(None, 3)
        if len(parts) >= 3 and parts[0] == "CHECK" and parts[2] in SCORE:
            checks.append(dict(name=parts[1], passed=parts[2] == "PASS",
                               score=SCORE[parts[2]],
                               detail=parts[3] if len(parts) > 3 else ""))
    return checks


def verify(task: dict, repo: Path, screenshot_floor: float | None = None) -> tuple[bool, str, list[dict]]:
    """Run the task's check without a shell.

    Shell strings are not portable: heredocs are unavailable on Windows and the
    interpreter is named differently, so checks are argv lists using a {py}
    placeholder, or a script written into the repository.

    `screenshot_floor` is passed through as an environment variable a checker
    may consult on its own -- this module has no idea what a screenshot is.
    Only appsift's own todo-app checker currently reads it, to skip a browser
    launch nobody will ever look at when an earlier attempt at the same model
    already scored higher.
    """
    source = task.get("verify_src")
    if not source and task.get("verify_src_path"):
        # A tool outside this package supplies an absolute path to its own checker.
        source = Path(task["verify_src_path"]).read_text(encoding="utf-8")
    if source:
        script = (repo / "_verify_check.py").resolve()
        script.write_text(source, encoding="utf-8")
        cmd = [sys.executable, str(script)]
    else:
        cmd = [sys.executable if arg == "{py}" else arg for arg in task["verify"]]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    if screenshot_floor is not None:
        env["APPSIFT_SCREENSHOT_FLOOR"] = str(screenshot_floor)
    try:
        proc = subprocess.run(cmd, cwd=repo, capture_output=True,
                              encoding="utf-8", errors="replace", env=env,
                              timeout=task.get("verify_timeout", 60))
    except subprocess.TimeoutExpired as exc:
        raw = (exc.stdout or b"") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        partial = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
        return False, "verify timed out", parse_checks(partial)
    output = (proc.stdout or "") + (proc.stderr or "")

    if task.get("graded"):
        # A graded task has no single expected line: the score is how many of its
        # requirements were met, and "passed" keeps its plain meaning of all of them.
        checks = parse_checks(output)
        if not checks:
            lines = output.strip().splitlines()
            return False, (lines[-1][:160] if lines else "the check script reported nothing"), []
        won = sum(1 for c in checks if c["passed"])
        missed = [c["name"] for c in checks if not c["passed"]]
        detail = f"{won}/{len(checks)} checks"
        if missed:
            detail += ": missing " + ", ".join(missed[:5])
            if len(missed) > 5:
                detail += f" and {len(missed) - 5} more"
        return won == len(checks), detail, checks

    if task["expect_stdout"] not in output:
        lines = output.strip().splitlines()
        return False, (lines[-1][:160] if lines else "no output"), []
    return True, "ok", []


def parse_events(stdout: str) -> tuple[list[str], int, list[str], dict, int, str, str]:
    """One JSON event per line: {type, sessionID, part:{type, ...}}.

    The session id is returned with the rest. opencode keeps every session in its
    own store, so recording the id is what makes a result openable afterwards: the
    counts say a session wrote nothing, and only the session itself says why.

    The model's own words are returned alongside the counts. A session that ends
    after two turns having written nothing looks identical whether the model gave
    up, answered in prose instead of calling a tool, or hit something in the
    harness -- and without what it said, the three cannot be told apart. The same
    omission in the screen stage made a grading bug undiagnosable for a day.
    """
    tools, steps, errors, said = [], 0, [], []
    session, finish = "", ""
    tokens = {"input": 0, "output": 0, "total": 0, "reasoning": 0}
    peak_input = 0
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except Exception:
            continue
        kind = event.get("type") or ""
        part = event.get("part") or {}
        if not session:
            session = (event.get("sessionID") or part.get("sessionID") or "")
        if kind == "error" or part.get("type") == "error":
            err = event.get("error") or {}
            errors.append(err.get("name") or str(err)[:80] or "error")
        if kind == "step_start":
            steps += 1
        if part.get("type") in ("tool", "tool-invocation") or kind == "tool":
            name = part.get("tool") or part.get("name") or event.get("tool")
            if name:
                tools.append(name)
        # Part types observed in the stream: text, reasoning, tool, step-start,
        # step-finish. Both text and reasoning are kept, because the failure worth
        # explaining is a turn that produced neither a tool call nor an answer: one
        # model spent 6457 tokens and two minutes reasoning about what it was about
        # to do, then stopped, and only the reasoning says so.
        if part.get("type") in ("text", "reasoning"):
            body = part.get("text") or part.get("reasoning") or ""
            if body:
                said.append(body)
        if kind in ("step_finish", "step-finish") or part.get("type") == "step-finish":
            # Why the model stopped, kept from the last step. Without it a model
            # that ended its turn without calling a tool, one that exhausted the
            # context, and one that finished the work all record the same clean
            # exit -- and the first two are the failures worth telling apart.
            finish = part.get("reason") or event.get("reason") or finish
            counts = part.get("tokens") or {}
            for key in tokens:
                tokens[key] += counts.get(key, 0) or 0
            peak_input = max(peak_input, counts.get("input", 0) or 0)
    return (tools, steps, errors, tokens, peak_input, "\n\n".join(said),
            session, finish)


# The model is given a shell and told to run and test its own application, so it
# cleans up after itself. One model did that with `pkill -9 -f python`, which
# matches this harness's own command line -- and killed the sweep driving it,
# mid-run, leaving opencode orphaned and no explanation in any log.
#
# A PID namespace ends the whole class: inside one, the model's processes are the
# only ones it can see, so the broadest kill it can write reaches no further than
# its own run. The network is deliberately not unshared, since the model must
# still reach Ollama and its own server, and the filesystem is unchanged, so this
# alters what the model can destroy and nothing about what it can do.
_ISOLATION = None


def isolation() -> list[str]:
    """The prefix that confines a child to its own process namespace, if available.

    Unprivileged user namespaces are not universal -- absent on Windows, and
    disabled by policy on some distributions -- so this degrades to running
    directly rather than refusing to run at all.
    """
    global _ISOLATION
    if _ISOLATION is None:
        prefix = ["unshare", "--user", "--map-root-user", "--pid", "--fork",
                  "--mount-proc"]
        try:
            ok = subprocess.run(prefix + ["true"], capture_output=True,
                                timeout=30).returncode == 0
        except Exception:
            ok = False
        _ISOLATION = prefix if (shutil.which("unshare") and ok) else []
    return list(_ISOLATION)


def _kill_tree(proc) -> None:
    """Take down the harness and everything it started."""
    if proc is None or proc.poll() is not None:
        return
    try:
        if hasattr(os, "killpg"):
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.terminate()
        proc.wait(timeout=15)
    except Exception:
        try:
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                proc.kill()
        except Exception:
            pass


def _slug(model: str) -> str:
    return "".join(c if c.isalnum() or c in "-." else "_" for c in model)


# What is kept is meant to be read. The check script, its scratch database and the
# caches left by running the tests are not part of what the model created, and a 20KB
# grader dropped beside a small application is noise. Re-checking a kept application drops
# the same litter, so both paths clear it.
LITTER = ("_verify_check.py", "_grading_store", ".pytest_cache",
          "src/todoapp/__pycache__", "tests/__pycache__")


def tidy(repo: Path) -> None:
    """Remove what checking left behind, leaving only the model's own work."""
    for name in LITTER:
        path = repo / name
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink()


def recheck(task: dict, rec: dict) -> dict | None:
    """Grade a kept application again, without the model.

    A sweep costs hours; a mistake in the checking code costs seconds to correct
    and would otherwise cost the sweep. The application is kept, so the checks can be run
    against the real application rather than against a description of it: imported,
    launched, driven over HTTP, restarted and tested exactly as during the run.

    Only what checking produced is replaced. How the model behaved -- its turns,
    tokens, generation rate, session and the reason it stopped -- was observed once
    and is not re-derived from files. Returns None when the application is no longer on
    disk, which is the one case that still needs the model.
    """
    repo = Path(rec.get("repo") or "")
    if not repo.is_dir():
        return None
    _, detail, checks = verify(task, repo)
    tidy(repo)
    out = dict(rec, checks=checks, rechecked_ts=time.time())
    # Empty checks says a requirement was never graded; nothing says why unless
    # the reason verify() gave up is kept beside it.
    out["check_error"] = None if checks else detail
    return out


def check_score(check: dict) -> float:
    """A check's own credit, from its score if graded that finely, else the
    plain 1 or 0 a record predating partial credit implies."""
    got = check.get("score")
    return got if got is not None else (1.0 if check["passed"] else 0.0)


def met(rec: dict) -> tuple:
    """(credit earned, requirements checked), from the checks themselves.

    Credit rather than a plain count: a check graded "warn" is real, incomplete
    work, and counting it as a full zero would say a thin-but-correct attempt
    and no attempt at all are the same finding.
    """
    checks = rec.get("checks") or []
    return sum(check_score(c) for c in checks), len(checks)


def score(rec: dict) -> float:
    """The share of requirements met, or 0 where nothing was checked."""
    won, total = met(rec)
    return round(100 * won / total, 1) if total else 0.0


def primary(attempts: list[dict]) -> dict:
    """Which of one model's attempts is the one a card or a ranking is built
    from -- the highest-scoring one, the earliest breaking a tie.

    No attempt is more canonical than another: every one runs at the same
    sampling, so the only thing that distinguishes them is how each happened
    to go. Reporting anything other than the best of them as the headline
    would just be reporting a worse result on purpose.
    """
    return max(attempts, key=score)


def passed(rec: dict) -> bool:
    """Whether every requirement was met."""
    won, total = met(rec)
    return bool(total) and won == total




def run_task(model: str, task: dict, workdir: Path, timeout: int,
             retain_dir: Path | None = None, host: str = DEFAULT_HOST,
             attempt: int = 1, screenshot_floor: float | None = None) -> dict:
    # A task that produces a whole application is worth keeping and reading, so it
    # is created at a path named after the model rather than in a temporary one.
    # Only a second or later attempt carries a suffix, so the first attempt's
    # path -- and everything already keyed on it, on disk and in the ledger --
    # is unchanged from before repeat attempts existed.
    at = None
    if task.get("retain") and retain_dir is not None:
        name = f"{_slug(model)}__{task['id']}"
        if attempt > 1:
            name += f"__{attempt}"
        at = retain_dir / name
    repo = seed(task, workdir, at=at)
    timeout = max(timeout, task.get("min_timeout", 0))
    protected = {f: _digest(repo / f) for f in task.get("immutable", [])}
    prompt = task["prompt"].replace("{py}", Path(sys.executable).stem)
    cmd = isolation() + ["opencode", "run", "--format", "json", "--auto",
                         "-m", f"ollama/{model}", "--dir", str(repo),
                         "--title", f"appsift-{task['id']}", prompt]

    # A model that cannot install its package into the machine cannot contaminate
    # the next model graded on it. PEP 668 already refuses this on most Linux
    # distributions, but that refusal names --break-system-packages in its own error
    # text, and a model driven with --auto simply takes the suggestion; one did.
    # Requiring a virtualenv is not overridable by that flag, and a virtualenv the
    # model makes for itself lives inside the repository, where it belongs.
    # PYTHONIOENCODING pins opencode's own stdout/stderr to UTF-8 regardless of
    # platform: without it, capturing text on Windows decodes through the system
    # codepage, and a model's output containing a character that codepage cannot
    # spell crashes the harness outright instead of failing the run gracefully.
    env = dict(os.environ, PIP_REQUIRE_VIRTUALENV="1", PYTHONIOENCODING="utf-8",
              OPENCODE_CONFIG_CONTENT=_agent_config(host, model))

    # The model is given a shell, and it uses it: it starts servers, runs test
    # suites, leaves things listening. Killing opencode alone would leave those
    # behind to hold ports and memory for the rest of the run, so the harness takes
    # its own process group and takes the group down with it.
    group = {"start_new_session": True} if hasattr(os, "setsid") else {}
    started, timed_out = time.time(), False
    proc = None
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                encoding="utf-8", errors="replace",
                                cwd=repo, env=env, **group)
        stdout, stderr = proc.communicate(timeout=timeout)
        code = proc.returncode
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        stdout, stderr = proc.communicate()
        stderr, code, timed_out = (stderr or "") + "\ntimed out", -1, True
    except FileNotFoundError:
        stdout, stderr, code = "", "opencode not found on PATH", 127
    finally:
        _kill_tree(proc)

    (tools, turns, errors, tokens, peak_input, said, session,
     finish) = parse_events(stdout)
    _, verify_detail, checks = verify(task, repo, screenshot_floor=screenshot_floor)
    # A modified protected file voids the grading rather than lowering it: the
    # checks no longer describe the task that was set. The files are recorded, so
    # the page can say which, rather than only that nothing was checked.
    tampered = [f for f, h in protected.items() if _digest(repo / f) != h]
    if tampered:
        checks = []

    if at is not None:
        tidy(repo)

    # The harness's own output, kept verbatim beside the work. Parsing it into
    # counts throws away the one thing that explains a session which ended having
    # written nothing, and a parser written against a guessed event shape captures
    # nothing at all without saying so -- which is exactly what happened.
    if at is not None and stdout:
        (repo / "opencode.jsonl").write_text(stdout[-RAW_SESSION:], encoding="utf-8")

    # Read back from opencode's own store, which timestamps each turn. Doing it
    # here keeps the ledger self-contained: the store can be pruned later, the
    # recorded rate cannot.
    from .session import generation, spoke
    speed = generation(session)
    # The stream carries no reasoning, so what a model said and how its last turn
    # ended are both read from opencode's store rather than inferred from events.
    voice = spoke(session)

    shot = repo / "ui.png"
    # What was observed, and nothing that follows from it. Whether the run passed,
    # what share it met and how fast it generated are all read back off the checks
    # and the token counts, so a change to how they are derived does not need a
    # sweep to take effect.
    # Empty checks says a requirement was never graded; nothing says why unless
    # the reason verify() gave up -- a crash, a timeout, an import that shadowed
    # a stdlib module -- is kept beside it rather than discarded with the rest
    # of verify()'s return.
    check_error = None if checks or tampered else verify_detail
    return dict(
        model=model, task=task["id"], checks=checks, tampered=tampered,
        check_error=check_error, attempt=attempt,
        screenshot=str(shot) if shot.exists() else None,
        wall_s=round(time.time() - started, 1), timed_out=timed_out, returncode=code,
        tool_calls=len(tools), tools=tools[:40], turns=turns, errors=errors[:10],
        tokens=tokens, peak_input_tokens=peak_input, session=session,
        finish=finish,
        final_acted=voice["acted"], final_produced=voice["produced"],
        gen_s=speed["gen_s"],
        repo=str(repo), stderr=(stderr or "")[-600:], ts=time.time(),
        said=(voice["said"] or said)[-SAID_KEPT:],
    )


