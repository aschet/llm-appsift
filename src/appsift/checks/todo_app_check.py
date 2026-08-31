# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT
"""Grade the todo-application task.

The task is far too large to score pass or fail: a model that creates a working
package but forgets one filter has done most of the job, and a binary verdict
would put it level with one that produced nothing. So each requirement is a
named check worth one point, and the result is a fraction.

Every check is observed from outside the implementation. The package is imported,
launched as a module, driven over HTTP, killed, and launched again against the
same store; nothing the model wrote about its own work is consulted. Output is
one `CHECK <name> PASS|FAIL <detail>` line per requirement, which the harness
parses.

Runs with the repository as the working directory and no third-party imports of
its own, so it stays usable on a machine that has only the interpreter.
"""
import os
import sys

# This script is written into the model's repository and run from there, so
# Python puts that directory first on sys.path -- letting a stray file the
# model wrote (an `http.py`, say) shadow the stdlib module of the same name
# before this checker's own imports below ever run. Nothing here is imported
# from the repository directly -- the application under test is only ever
# driven as a subprocess -- so the entry is safe to drop.
_here = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if p not in ("", _here)]

import json
import re
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path.cwd()
SRC = REPO / "src"
HEX = re.compile(r"^#[0-9a-fA-F]{6}$")
BOOT_S = 25.0
RESULTS = []
# Set before anything is launched. A server already on Flask's default port is
# not this run's, so the fallback below must not mistake it for the model's work.
STRAY_5000 = False
# Packages found importable from outside the repository. See check_importable.
FOREIGN = []


def pkg_root():
    """Where PYTHONPATH must point, and where the package itself lives.

    The specification asks for src/todoapp. A flat todoapp/ at the repository
    root is the other layout real Python projects use, and a model that chose
    it wrote a package exactly as valid as the one asked for -- penalising it
    for that choice would grade a directory name rather than the application.
    Checked in the order the specification states them, so a repository that
    somehow has both is graded on the one that was actually requested. Neither
    existing falls back to the requested path, which is the layout every
    missing-file message should then name.
    """
    if (SRC / "todoapp" / "__init__.py").exists():
        return SRC, SRC / "todoapp"
    if (REPO / "todoapp" / "__init__.py").exists():
        return REPO, REPO / "todoapp"
    return SRC, SRC / "todoapp"


def check(name, status, detail=""):
    """Record one requirement. status is True, False, or "warn" for a real but
    incomplete attempt -- worth more than nothing and less than meeting it."""
    RESULTS.append((name, status, str(detail)[:200]))
    return status is True


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def port_open(port, host="127.0.0.1"):
    with socket.socket() as s:
        s.settimeout(0.4)
        return s.connect_ex((host, port)) == 0


def child_env(port, store):
    env = dict(os.environ)
    env["PORT"] = str(port)
    env["TODO_DB"] = str(store)
    path_root, _ = pkg_root()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(path_root)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env["PYTHONUNBUFFERED"] = "1"
    # Without this, capturing the model's stdout on Windows decodes through the
    # system codepage, and output it cannot spell crashes the check rather than
    # failing it.
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("FLASK_RUN_PORT", None)
    return env


def spawn(port, store):
    """Start the package as a module and wait for it to answer."""
    kw = {}
    if hasattr(os, "setsid"):
        kw["start_new_session"] = True
    proc = subprocess.Popen([sys.executable, "-m", "todoapp"], cwd=str(REPO),
                            env=child_env(port, store), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, encoding="utf-8",
                            errors="replace", **kw)
    deadline = time.time() + BOOT_S
    while time.time() < deadline:
        if port_open(port):
            return proc, port, ""
        if proc.poll() is not None:
            out = (proc.stdout.read() or "").strip().splitlines()
            return None, port, (out[-1][:160] if out else f"exited {proc.returncode}")
        time.sleep(0.25)
    # A model that hardcodes Flask's default rather than reading PORT still has a
    # working application; that is a smaller fault than not starting at all, and
    # grading it as a total failure would hide everything downstream of it.
    if not STRAY_5000 and port_open(5000):
        return proc, 5000, "ignored PORT, serving on 5000"
    kill(proc)
    return None, port, f"nothing listening on {port} after {BOOT_S:.0f}s"


def kill(proc):
    if proc is None or proc.poll() is not None:
        return
    try:
        if hasattr(os, "killpg"):
            os.killpg(os.getpgid(proc.pid), 15)
        else:
            proc.terminate()
        proc.wait(timeout=8)
    except Exception:
        try:
            # The group, not just the process started directly: a browser forks
            # a zygote, a GPU process and a renderer per tab, and killing only the
            # one PID this module holds leaves the rest of that tree running.
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(proc.pid), 9)
            else:
                proc.kill()
        except Exception:
            pass


def run_browser(args, timeout):
    """Run a browser to completion, or kill its whole process tree at the deadline.

    plain subprocess.run(timeout=...) only reaches the process it started directly;
    a browser is never that -- headless Chromium forks a zygote, a GPU process and
    a renderer per tab, and a hung render leaves all of them behind. Grading this
    task all afternoon on the same machine as a screen or a sweep of it, that leak
    compounds into exactly the resource pressure that causes the next check to hang.
    """
    kw = {}
    if hasattr(os, "setsid"):
        kw["start_new_session"] = True
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            encoding="utf-8", errors="replace", **kw)
    try:
        stdout, _ = proc.communicate(timeout=timeout)
        return stdout
    except subprocess.TimeoutExpired:
        kill(proc)
        # Draining the pipes after a kill is a courtesy, not a requirement -- it
        # only avoids leaving a zombie entry behind. kill() is already a
        # best-effort SIGTERM-then-SIGKILL; if the process still will not die,
        # waiting on it without end would trade one leaked process for a check
        # that never returns, which is the worse failure by far.
        try:
            proc.communicate(timeout=5)
        except Exception:
            pass
        return None
    except Exception:
        kill(proc)
        return None


def http(method, url, body=None, timeout=15):
    """Return (status, parsed-or-text). Never raises for an HTTP status."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", "replace")
            status, ctype = resp.status, resp.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        status, ctype = exc.code, exc.headers.get("Content-Type", "")
    except Exception as exc:
        return 0, f"{type(exc).__name__}: {exc}"
    if "json" in ctype or raw.startswith(("{", "[")):
        try:
            return status, json.loads(raw)
        except Exception:
            pass
    return status, raw


# ---------------------------------------------------------------- static shape

def check_layout():
    _, pkg = pkg_root()
    missing = [str(p.relative_to(REPO)) for p in
               (pkg / "__init__.py", pkg / "__main__.py", REPO / "pyproject.toml")
               if not p.exists()]
    tests = list((REPO / "tests").glob("test_*.py")) if (REPO / "tests").is_dir() else []
    if not tests:
        missing.append("tests/test_*.py")
    check("package_layout", not missing, "missing: " + ", ".join(missing) if missing else "ok")


def _toml(text):
    try:
        import tomllib
        return tomllib.loads(text)
    except Exception:
        return None


def check_pyproject():
    path = REPO / "pyproject.toml"
    if not path.exists():
        return check("pyproject", False, "no pyproject.toml")
    text = path.read_text(encoding="utf-8", errors="replace")
    data = _toml(text)
    if data is None:                     # unparseable, or no tomllib available
        wanted = ("[project]", "name", "version", "flask", "[build-system]")
        low = text.lower()
        return check("pyproject", all(w in low for w in wanted),
                     "could not parse; judged on its text")
    project = data.get("project") or {}
    faults = []
    if not project.get("name"):
        faults.append("no project.name")
    if not (project.get("version") or "version" in (project.get("dynamic") or [])):
        faults.append("no version")
    deps = " ".join(project.get("dependencies") or []).lower()
    if "flask" not in deps:
        faults.append("flask not declared as a dependency")
    if not (project.get("scripts") or project.get("gui-scripts")
            or project.get("entry-points")):
        faults.append("no console entry point")
    if not (data.get("build-system") or {}).get("build-backend"):
        faults.append("no build backend")
    check("pyproject", not faults, "; ".join(faults) or "ok")


def check_importable(store):
    """The package has to expose an application factory, not a module-level app."""
    # The path is printed and checked below: an installed copy elsewhere on the
    # system must not stand in for the package this repository was meant to contain.
    code = ("import todoapp, os\n"
            "app = todoapp.create_app()\n"
            "assert app is not None\n"
            "print('FROM', os.path.abspath(todoapp.__file__))\n"
            "print('OK', hasattr(app, 'test_client'))\n")
    try:
        proc = subprocess.run([sys.executable, "-c", code], cwd=str(REPO),
                              env=child_env(free_port(), store), capture_output=True,
                              encoding="utf-8", errors="replace", timeout=90)
    except subprocess.TimeoutExpired:
        return check("importable", False, "import hung")
    origin = next((line[5:].strip() for line in proc.stdout.splitlines()
                   if line.startswith("FROM ")), "")
    if origin and not origin.startswith(str(REPO) + os.sep):
        # A model that ran `pip install -e .` leaves its package importable by every
        # later interpreter on the machine. Blocking the user site directory would
        # also block Flask, which lives there, so the import is allowed and its
        # origin checked instead: grading this repository against a package that is
        # not in it would score one model's work under another's name.
        FOREIGN.append(origin)
        return check("importable", False,
                     f"imported from outside the repository: {origin}")
    if proc.returncode == 0 and "OK True" in proc.stdout:
        return check("importable", True, "create_app() returns a Flask app")
    err = (proc.stderr or proc.stdout or "").strip().splitlines()
    check("importable", False, err[-1][:160] if err else "create_app() unavailable")


# ------------------------------------------------------------------ behaviour

def check_seed(base):
    status, body = http("GET", f"{base}/api/tasks")
    if status != 200 or not isinstance(body, list):
        return check("seed_three", False, f"GET /api/tasks -> {status} {str(body)[:80]}")
    check("seed_three", len(body) == 3, f"{len(body)} entries in an empty store")
    return body


def check_shape(entries, source="the seeded entries"):
    """The shape of a task object, judged on whatever entries exist.

    A model that seeds nothing still has a task object; failing this for want of
    seeded entries reports one fault twice and says nothing about the shape. When
    seeding produced none, the caller supplies an entry it created instead.
    """
    if not entries:
        return check("task_shape", False, f"no entries to inspect in {source}")
    fields = {"id": int, "title": str, "description": str, "color": str,
              "done": bool, "position": int}
    faults = []
    for entry in entries:
        if not isinstance(entry, dict):
            faults.append("entry is not an object")
            break
        for key, typ in fields.items():
            if key not in entry:
                faults.append(f"no {key!r}")
            elif not isinstance(entry[key], typ) or (typ is int and isinstance(entry[key], bool)):
                faults.append(f"{key} is {type(entry[key]).__name__}")
        if isinstance(entry.get("color"), str) and not HEX.match(entry["color"]):
            faults.append(f"color {entry['color']!r} is not #rrggbb")
    check("task_shape", not faults, "; ".join(sorted(set(faults))[:4]) or "ok")


def check_create(base):
    status, body = http("POST", f"{base}/api/tasks",
                        {"title": "Write the release notes",
                         "description": "Summarise the storage rewrite",
                         "color": "#3366cc"})
    if status not in (200, 201) or not isinstance(body, dict) or "id" not in body:
        check("create", False, f"POST -> {status} {str(body)[:80]}")
        return None
    faults = []
    if status != 201:
        faults.append(f"returned {status}, not 201")
    if body.get("title") != "Write the release notes":
        faults.append("title not echoed")
    if body.get("done") is not False:
        faults.append("new entry is not open")
    listed = http("GET", f"{base}/api/tasks")[1]
    if not (isinstance(listed, list) and any(t.get("id") == body["id"] for t in listed)):
        faults.append("not present in the listing")
    check("create", not faults, "; ".join(faults) or "ok")
    return body["id"]


def check_read_one(base, made):
    if made is None:
        return check("read_one", False, "nothing was created to read")
    status, body = http("GET", f"{base}/api/tasks/{made}")
    faults = []
    if status != 200 or not isinstance(body, dict) or body.get("id") != made:
        faults.append(f"GET of the new entry -> {status}")
    missing = http("GET", f"{base}/api/tasks/98765432")[0]
    if missing != 404:
        faults.append(f"unknown id -> {missing}, not 404")
    check("read_one", not faults, "; ".join(faults) or "ok")


def check_update(base, made):
    if made is None:
        return check("update", False, "nothing was created to edit")
    status, _ = http("PATCH", f"{base}/api/tasks/{made}",
                     {"title": "Write the changelog", "color": "#aa2200"})
    after = http("GET", f"{base}/api/tasks/{made}")[1]
    faults = []
    if status != 200:
        faults.append(f"PATCH -> {status}")
    if not isinstance(after, dict):
        faults.append("entry unreadable after the edit")
    else:
        if after.get("title") != "Write the changelog":
            faults.append(f"title is {str(after.get('title'))[:30]!r}")
        if (after.get("color") or "").lower() != "#aa2200":
            faults.append(f"colour is {after.get('color')!r}")
        if after.get("description") != "Summarise the storage rewrite":
            faults.append("a field absent from the patch was overwritten")
    check("update", not faults, "; ".join(faults) or "ok")


def check_toggle(base, made):
    if made is None:
        return check("toggle_done", False, "nothing was created to check off")
    http("PATCH", f"{base}/api/tasks/{made}", {"done": True})
    after = http("GET", f"{base}/api/tasks/{made}")[1]
    ok = isinstance(after, dict) and after.get("done") is True
    detail = "ok" if ok else f"done is {after.get('done') if isinstance(after, dict) else after!r}"
    if ok and after.get("title") != "Write the changelog":
        ok, detail = False, "checking it off discarded the title"
    check("toggle_done", ok, detail)


def check_delete(base):
    made = http("POST", f"{base}/api/tasks",
                {"title": "Temporary", "description": "", "color": "#101010"})[1]
    if not isinstance(made, dict) or "id" not in made:
        return check("delete", False, "could not create an entry to delete")
    status, _ = http("DELETE", f"{base}/api/tasks/{made['id']}")
    faults = []
    if status not in (200, 204):
        faults.append(f"DELETE -> {status}")
    if http("GET", f"{base}/api/tasks/{made['id']}")[0] != 404:
        faults.append("still readable afterwards")
    check("delete", not faults, "; ".join(faults) or "ok")


def check_filters(base):
    listed = http("GET", f"{base}/api/tasks")[1]
    if not isinstance(listed, list):
        check("filter_done", False, "listing unavailable")
        return check("filter_text", False, "listing unavailable")

    done_ids = {t["id"] for t in listed if t.get("done")}
    open_ids = {t["id"] for t in listed if not t.get("done")}
    faults = []
    status, only_done = http("GET", f"{base}/api/tasks?done=true")
    if status != 200 or not isinstance(only_done, list):
        faults.append(f"?done=true -> {status}")
    elif {t["id"] for t in only_done} != done_ids:
        faults.append(f"?done=true returned {len(only_done)} of {len(done_ids)} checked")
    status, only_open = http("GET", f"{base}/api/tasks?done=false")
    if status != 200 or not isinstance(only_open, list):
        faults.append(f"?done=false -> {status}")
    elif {t["id"] for t in only_open} != open_ids:
        faults.append(f"?done=false returned {len(only_open)} of {len(open_ids)} open")
    check("filter_done", not faults, "; ".join(faults) or "ok")

    faults = []
    http("POST", f"{base}/api/tasks",
         {"title": "Xylophone lesson", "description": "", "color": "#00aa88"})
    http("POST", f"{base}/api/tasks",
         {"title": "Unrelated", "description": "buy a xylophone stand", "color": "#00aa88"})
    status, hits = http("GET", f"{base}/api/tasks?q=xylophone")
    if status != 200 or not isinstance(hits, list):
        faults.append(f"?q= -> {status}")
    else:
        titles = {t.get("title") for t in hits}
        if "Xylophone lesson" not in titles:
            faults.append("no case-insensitive match on the title")
        if "Unrelated" not in titles:
            faults.append("no match on the description")
        if len(hits) != 2:
            faults.append(f"{len(hits)} hits, expected 2")
    check("filter_text", not faults, "; ".join(faults) or "ok")


def check_reorder(base):
    listed = http("GET", f"{base}/api/tasks")[1]
    if not isinstance(listed, list) or len(listed) < 3:
        return check("reorder", False, "too few entries to reorder"), []
    wanted = [t["id"] for t in listed][::-1]
    status, _ = http("POST", f"{base}/api/tasks/reorder", {"order": wanted})
    after = http("GET", f"{base}/api/tasks")[1]
    got = [t["id"] for t in after] if isinstance(after, list) else []
    faults = []
    if status not in (200, 204):
        faults.append(f"reorder -> {status}")
    if got != wanted:
        faults.append("the listing did not follow the requested order")
    elif [t["position"] for t in after] != sorted(t["position"] for t in after):
        faults.append("position no longer ascends with the listing")
    check("reorder", not faults, "; ".join(faults) or "ok")
    # The order the store actually holds, not the one that was asked for. A model
    # whose reorder does nothing has already been marked down for it; comparing the
    # restart against the requested order would fail persistence for the same fault
    # a second time, and say nothing about whether anything survived.
    return None, (got or wanted)


def check_persistence(port, store, expected_order):
    """Restart against the same store; state written through the API must survive."""
    proc, port, note = spawn(port, store)
    if proc is None:
        return check("persist", False, f"restart failed: {note}"), None, None
    base = f"http://127.0.0.1:{port}"
    status, body = http("GET", f"{base}/api/tasks")
    faults = []
    if status != 200 or not isinstance(body, list):
        faults.append(f"listing after restart -> {status}")
    else:
        if [t["id"] for t in body] != expected_order:
            faults.append("the order before the restart did not survive it")
        if not any(t.get("title") == "Write the changelog" for t in body):
            faults.append("an edited entry did not survive")
        if not any(t.get("done") for t in body):
            faults.append("a checked entry came back open")
        if len(body) != len(expected_order):
            faults.append(f"{len(body)} entries, expected {len(expected_order)}")
    check("persist", not faults, "; ".join(faults) or "ok")
    return None, proc, base


def _uncommented(text):
    """The markup with its comments removed, so a disclaimer cannot satisfy a check."""
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"(?m)^\s*//.*$|(?<=[\s;{}])//[^\n]*", " ", text)


def rendered_dom(base):
    """The page after its scripts have run, or None if no browser is available.

    The specification asks that the page render every task; it does not say the
    server must do it. Fetching the list and building the DOM is the ordinary way
    to write this, and judging the source alone fails it -- four models in one
    sweep lost the point for a page their screenshot shows working. The browser
    that takes those screenshots can answer the question properly.
    """
    browser = next((b for b in ("chromium", "chromium-browser", "google-chrome",
                                "chrome") if shutil.which(b)), None)
    if not browser:
        return None
    stdout = run_browser([browser, "--headless", "--disable-gpu", "--no-sandbox",
                          "--virtual-time-budget=4000", "--dump-dom", base + "/"],
                         timeout=90)
    return stdout if stdout and "<" in stdout else None


def _visible(html):
    """Text a reader would actually see, with inert script and style payloads
    stripped out first.

    A title JSON-encoded into a <script type="application/json"> tag for a
    page's own JavaScript to read is real markup and a real DOM dump would
    contain it verbatim -- but nobody sees it until something renders it into
    the page, and one model's page never did. Searching the rendered DOM
    without stripping this first would call that title shown.
    """
    html = re.sub(r"<script\b[^>]*>.*?</script>", " ", html, flags=re.S | re.I)
    return re.sub(r"<style\b[^>]*>.*?</style>", " ", html, flags=re.S | re.I)


def check_ui(base, seeds):
    status, page = http("GET", base + "/")
    if status != 200 or not isinstance(page, str):
        check("ui_page", False, f"GET / -> {status}")
        check("ui_drag", False, "no page to inspect")
        return check("ui_edit", False, "no page to inspect")
    # The source carries the structure and the scripts; the rendered DOM carries
    # what a reader actually sees. Each question is asked of the one that can
    # answer it, and both are searched for drag handling, which may be attached
    # in script rather than written into the markup.
    dom = rendered_dom(base)
    seen = _visible(dom if dom is not None else page)
    low = page.lower()
    faults = []
    if "<html" not in low:
        faults.append("the root is not an HTML document")
    shown = [t["title"] for t in seeds if t.get("title") and t["title"] in seen]
    if len(shown) < len(seeds):
        faults.append(f"{len(shown)} of {len(seeds)} seeded titles rendered"
                      + ("" if dom is not None else " (source only; no browser)"))
    if not re.search(r"<(input|textarea|form)\b", (seen + page).lower()):
        faults.append("no field for entering a task")
    check("ui_page", not faults, "; ".join(faults) or "ok")

    # Comments are stripped first: one model passed this check on the strength of
    # `// Simple drag-and-drop placeholder (not fully implemented)`.
    both = _uncommented((page + (dom or "")).lower())
    drag = [w for w in ("draggable", "dragstart", "dragover", "dragend",
                        "dragenter", "sortable", "dragula") if w in both]
    # A bare "drop" is not evidence -- it appears in DROP TABLE, drop-shadow and
    # dropdown. A handler bound to the drop event is.
    if re.search(r"""ondrop|(?:addeventlistener|\.on)\s*\(\s*['"]drop['"]""", both):
        drag.append("a bound drop handler")
    check("ui_drag", bool(drag), "found " + ", ".join(drag) if drag
          else "no drag handling in the markup")

    # No single idiom for "click to edit" exists the way draggable/ondrop cover
    # drag-and-drop, so several common ones are tried rather than one exact
    # pattern: inline editing (contenteditable), open-to-edit (dblclick), or
    # an explicit labelled control.
    edit = [w for w in ("contenteditable", "dblclick") if w in both]
    if re.search(r'''class=["'][^"']*\bedit\b|>\s*(edit|rename|✎|✏)\s*<''', both):
        edit.append("an edit control")
    check("ui_edit", bool(edit), "found " + ", ".join(edit) if edit
          else "no way to edit an existing entry in the markup")


def check_own_tests(store):
    """The model's own tests, run in isolation, as a signal separate from ours."""
    env = child_env(free_port(), store)
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"],
            cwd=str(REPO), env=env, capture_output=True, encoding="utf-8",
            errors="replace", timeout=300)
    except subprocess.TimeoutExpired:
        return check("own_tests", False, "the test run did not finish")
    except FileNotFoundError:
        return check("own_tests", False, "pytest unavailable")
    text = (proc.stdout or "") + (proc.stderr or "")
    passed = int((re.search(r"(\d+) passed", text) or [0, 0])[1])
    broken = sum(int(m) for m in re.findall(r"(\d+) (?:failed|error)", text))
    if proc.returncode == 5 or (passed == 0 and not broken):
        return check("own_tests", False, "no tests were collected")
    # The specification asks for tests covering each endpoint, so a bare
    # `assert True` must not score what real coverage scores. Four is the floor
    # rather than a target: it is well under the number of endpoints, so it
    # marks only a suite that was never seriously attempted as having failed
    # outright. Below it, or with some of what was written actually wrong, is
    # real work rather than nothing -- worth a warning rather than a zero, since
    # a thin-but-correct suite and an empty one are not the same finding.
    if broken:
        return check("own_tests", "warn", f"{passed} passed, {broken} failed")
    if passed < 4:
        return check("own_tests", "warn", f"{passed} passed, too few to cover the API")
    check("own_tests", True, f"{passed} passed")


def _worth_a_screenshot():
    """Whether this attempt is doing well enough, against its siblings, to be
    the one screenshot a model is shown with.

    Only ever one screenshot is shown per model, so a --variance sweep taking
    one per attempt launches a browser for pictures nobody will see. The floor
    is the best final score an earlier attempt at this model already reached;
    own_tests has not run yet at this point, so the comparison is against
    everything checked so far rather than the eventual final tally -- close
    enough, since own_tests is one check out of many. Strictly greater, not
    equal or greater: a tie is decided in the earlier attempt's favour, so a
    later one that only ties would never be shown regardless.
    """
    floor = os.environ.get("APPSIFT_SCREENSHOT_FLOOR")
    if not floor or not RESULTS:
        return True
    got = sum(1.0 if s is True else 0.5 if s == "warn" else 0.0 for _, s, _ in RESULTS)
    return got / len(RESULTS) > float(floor)


def screenshot(base):
    """A picture of the finished interface, for the record rather than the score."""
    browser = next((b for b in ("chromium", "chromium-browser", "google-chrome",
                                "chrome") if shutil.which(b)), None)
    if not browser:
        return
    out = REPO / "ui.png"
    run_browser([browser, "--headless", "--disable-gpu", "--no-sandbox",
                "--hide-scrollbars", "--window-size=1280,900",
                "--virtual-time-budget=4000",
                f"--screenshot={out}", base + "/"], timeout=90)
    if out.exists() and out.stat().st_size:
        print(f"SHOT {out}")


def main():
    global STRAY_5000
    STRAY_5000 = port_open(5000)
    store = REPO / "_grading_store"
    shutil.rmtree(store, ignore_errors=True)
    store.mkdir()
    db = store / "tasks.db"

    check_layout()
    check_pyproject()
    check_importable(db)

    port = free_port()
    proc, port, note = spawn(port, db) if not FOREIGN else (None, 0, "")
    if FOREIGN:
        note = f"an installed copy shadows this repository: {FOREIGN[0]}"
    if proc is not None:
        # A listening socket is not a working server: spawn() only confirms the
        # port accepted a connection, and a model whose handler hangs on the
        # first real request would otherwise pay every remaining check's own
        # timeout in turn -- ten minutes of waiting to learn what this one quick
        # request already knows.
        status, _ = http("GET", f"http://127.0.0.1:{port}/", timeout=5)
        if not status:
            kill(proc)
            proc, note = None, "listening on the port but never answered a request"
    if proc is None:
        check("serves", False, note)
        for name in ("seed_three", "task_shape", "create", "read_one", "update",
                     "toggle_done", "delete", "filter_done", "filter_text",
                     "reorder", "persist", "ui_page", "ui_drag", "ui_edit"):
            check(name, False, "the application did not start")
    else:
        check("serves", True, note or "ok")
        base = f"http://127.0.0.1:{port}"
        try:
            seeds = check_seed(base)
            made = check_create(base)
            # Seeding and the shape of a task are separate requirements. If nothing
            # was seeded, the entry just created answers the shape question.
            shape_from = seeds
            source = "the seeded entries"
            if not shape_from and made is not None:
                _, one = http("GET", f"{base}/api/tasks/{made}")
                if isinstance(one, dict):
                    shape_from, source = [one], "a created entry"
            check_shape(shape_from, source)
            check_read_one(base, made)
            check_update(base, made)
            check_toggle(base, made)
            check_delete(base)
            check_filters(base)
            _, order = check_reorder(base)
        finally:
            kill(proc)
        _, proc, base = check_persistence(port, db, order)
        try:
            if proc is not None:
                check_ui(base, seeds or [])
                if _worth_a_screenshot():
                    screenshot(base)
            else:
                check_ui("http://127.0.0.1:1", [])
        finally:
            kill(proc)

    check_own_tests(store / "own.db")

    for name, status, detail in RESULTS:
        word = "PASS" if status is True else "WARN" if status == "warn" else "FAIL"
        print(f"CHECK {name} {word} {detail}")
    won = sum(1 for _, status, _ in RESULTS if status is True)
    print(f"CHECKS {won}/{len(RESULTS)}")


if __name__ == "__main__":
    main()
