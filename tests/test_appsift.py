# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT
"""appsift: the application task, and the grader that judges it.

A separate tool from the screen, so a separate suite. What matters here is that
a correct implementation scores every check, that specific damage costs specific
points, and that the tool reports across models without touching the screen's
records.
"""
import io
import json
import subprocess
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from appsift import TASK
from appsift import cli as appcli
from appsift import harness
from appsift.config import Config

OFFLINE = "http://127.0.0.1:1"


class TestItStandsAlone(unittest.TestCase):
    """No import may reach back into the project this was split out of."""

    PACKAGE = Path(__file__).parent.parent / "src" / "appsift"

    def test_nothing_imports_the_screen(self):
        for path in sorted(self.PACKAGE.rglob("*.py")):
            with self.subTest(module=path.name):
                text = path.read_text(encoding="utf-8")
                for line in text.splitlines():
                    if line.startswith(("import ", "from ")):
                        self.assertNotIn("codesift", line, f"{path.name}: {line}")

    def test_it_declares_no_dependencies(self):
        pyproject = (self.PACKAGE.parent.parent / "pyproject.toml").read_text()
        self.assertIn("dependencies = []", pyproject)

    def test_it_brings_its_own_checker(self):
        self.assertTrue(Path(TASK["verify_src_path"]).exists())
        self.assertIn("appsift", TASK["verify_src_path"])

    def test_it_shares_the_gpu_lock_deliberately(self):
        # A private lock would let this and any sibling tool load models at the
        # same time, which thrashes one GPU and corrupts both sets of timings.
        from appsift import gpulock
        self.assertIn("codesift-gpu", gpulock.lock_path("http://x"))


class TestGrading(unittest.TestCase):
    """The grader, against a solution known to be correct and then damaged."""

    REFERENCE = Path(__file__).parent / "reference"

    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def grade(self, damage=None):
        repo = harness.seed(TASK, self.tmp)
        shutil.copytree(self.REFERENCE, repo, dirs_exist_ok=True)
        if damage:
            damage(repo)
        passed, detail, checks = harness.verify(TASK, repo)
        return passed, detail, {c["name"]: c for c in checks}

    def test_the_reference_scores_every_check(self):
        passed, detail, checks = self.grade()
        failed = {n: c["detail"] for n, c in checks.items() if not c["passed"]}
        self.assertEqual(failed, {}, "a correct implementation must lose no points")
        self.assertTrue(passed)
        self.assertEqual(len(checks), 19)

    def test_an_unbuilt_repository_scores_nothing(self):
        repo = harness.seed(TASK, self.tmp)
        passed, detail, checks = harness.verify(TASK, repo)
        self.assertFalse(passed)
        self.assertEqual(len(checks), 19)
        self.assertFalse(any(c["passed"] for c in checks))

    def test_a_broken_text_filter_costs_only_that_check(self):
        def damage(repo):
            path = repo / "src" / "todoapp" / "storage.py"
            path.write_text(path.read_text().replace("if query:", "if False:"),
                            encoding="utf-8")
        _, _, checks = self.grade(damage)
        self.assertFalse(checks["filter_text"]["passed"])
        self.assertTrue(checks["filter_done"]["passed"])
        self.assertTrue(checks["persist"]["passed"])

    def test_a_store_that_forgets_costs_the_persistence_check(self):
        def damage(repo):
            path = repo / "src" / "todoapp" / "storage.py"
            path.write_text(path.read_text().replace(
                'return os.environ.get("TODO_DB") or "tasks.db"',
                'return ":memory:"'), encoding="utf-8")
        _, _, checks = self.grade(damage)
        self.assertFalse(checks["persist"]["passed"])
        self.assertTrue(checks["serves"]["passed"])

    def test_missing_tests_and_packaging_are_seen(self):
        def damage(repo):
            shutil.rmtree(repo / "tests")
            (repo / "pyproject.toml").unlink()
        _, _, checks = self.grade(damage)
        self.assertFalse(checks["package_layout"]["passed"])
        self.assertFalse(checks["pyproject"]["passed"])
        self.assertFalse(checks["own_tests"]["passed"])
        self.assertTrue(checks["serves"]["passed"], "the application still runs")

    def test_a_token_test_suite_does_not_earn_the_point(self):
        def damage(repo):
            for path in (repo / "tests").glob("test_*.py"):
                path.unlink()
            (repo / "tests" / "test_smoke.py").write_text(
                "def test_it_works():\n    assert True\n", encoding="utf-8")
        _, _, checks = self.grade(damage)
        self.assertFalse(checks["own_tests"]["passed"])
        self.assertIn("too few", checks["own_tests"]["detail"])


class TestReporting(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.cfg = Config(host=OFFLINE, results_dir=self.tmp)

    def write(self, records):
        with (self.tmp / appcli.LEDGER).open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec) + "\n")

    def record(self, model, won, total=18, wall=100.0):
        checks = [dict(name=f"c{i}", passed=i < won, detail="") for i in range(total)]
        return dict(model=model, task="ag_todoapp", checks=checks, wall_s=wall,
                    repo=str(self.tmp / f"{model}_app"))

    def summarise(self, models=()):
        out = io.StringIO()
        appcli.summarise(self.cfg, list(models), stream=out)
        return out.getvalue()

    def test_it_keeps_its_own_ledger(self):
        self.assertEqual(appcli.LEDGER, "applications.jsonl")
        self.assertNotEqual(appcli.LEDGER, "agentic.jsonl")

    def test_models_are_ordered_by_how_much_they_met(self):
        self.write([self.record("weak", 1), self.record("strong", 18),
                    self.record("middling", 9)])
        rows = [l for l in self.summarise().splitlines() if " of 18 " in l]
        self.assertEqual([r.split()[1].rstrip(":") for r in rows],
                         ["strong", "middling", "weak"])

    def test_the_best_attempt_is_the_one_shown(self):
        self.write([self.record("m", 2), self.record("m", 15), self.record("m", 4)])
        text = self.summarise()
        self.assertIn("15 of 18", text)
        self.assertNotIn("4 of 18", text)

    def test_it_names_what_was_missing_and_where_it_was_built(self):
        self.write([self.record("m", 16)])
        text = self.summarise()
        self.assertIn("missing c16, c17", text)
        self.assertIn("m_app", text)

    def test_it_can_be_narrowed_to_named_models(self):
        self.write([self.record("wanted", 5), self.record("other", 9)])
        text = self.summarise(["wanted"])
        self.assertIn("wanted", text)
        self.assertNotIn("other", text)

    def test_an_empty_ledger_says_so(self):
        self.assertIn("nothing has been run", self.summarise())


class TestCommandLine(unittest.TestCase):
    def test_the_spec_can_be_read_without_running_anything(self):
        with mock.patch("sys.stdout", new=io.StringIO()) as out:
            code = appcli.main(["--spec"])
        self.assertEqual(code, 0)
        self.assertIn("# Todo Application", out.getvalue())

    def test_running_summarises_afterwards(self):
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(appcli, "run", return_value=0), \
                 mock.patch.object(appcli, "summarise") as summarise:
                appcli.main(["--models", "a:1", "--results-dir", d])
            summarise.assert_called_once()

    def test_running_writes_the_report_by_default(self):
        # -o defaults to report.html; a user should not have to ask for the
        # report a second time after already asking the tool to run.
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(appcli, "run", return_value=0), \
                 mock.patch.object(appcli, "summarise"), \
                 mock.patch.object(appcli.report, "write") as write:
                appcli.main(["--models", "a:1", "--results-dir", d])
            write.assert_called_once()

    def test_a_preflight_failure_writes_no_empty_report(self):
        # run() returns non-zero before anything is recorded; a report at
        # that point would only be the empty placeholder page, which is
        # worse than none -- the real problem is already on stderr.
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(appcli, "run", return_value=2), \
                 mock.patch.object(appcli, "summarise") as summarise, \
                 mock.patch.object(appcli.report, "write") as write:
                code = appcli.main(["--models", "a:1", "--results-dir", d])
            self.assertEqual(code, 2)
            summarise.assert_not_called()
            write.assert_not_called()


class TestTheHelpOffersOnlyWhatAUserRuns(unittest.TestCase):
    """Regrading and rendering the report from records already on disk are
    maintenance operations on this tool's own output, not something a user
    runs day to day -- codesift keeps the same kind of operation off its own
    primary command, reachable only as `python -m codesift.<name>`.
    """

    def help_text(self):
        import contextlib
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit):
            appcli.main(["--help"])
        return out.getvalue()

    def test_maintenance_flags_are_not_offered(self):
        text = self.help_text()
        for gone in ("--recheck", "--apply", "--report", "--history"):
            with self.subTest(flag=gone):
                self.assertNotIn(gone, text)

    def test_recheck_and_report_are_their_own_runnable_modules(self):
        import importlib
        for name in ("recheck", "report"):
            with self.subTest(module=name):
                mod = importlib.import_module(f"appsift.{name}")
                self.assertTrue(callable(getattr(mod, "main", None)),
                                f"{name} has no main()")


class TestForeignInstall(unittest.TestCase):
    """A model that installs its package must not be graded on a later model's run.

    One model ran `pip install -e .`, which put its package on the import path of
    every later interpreter on the machine. An empty repository then scored 15 of
    18 checks on that model's work. Blocking the user site directory is not the
    answer, because Flask lives there too.
    """

    def test_the_check_script_refuses_a_package_from_outside_the_repository(self):
        src = Path(TASK["verify_src_path"]).read_text(encoding="utf-8")
        self.assertIn("imported from outside the repository", src)
        self.assertIn("FOREIGN", src)
        self.assertNotIn("PYTHONNOUSERSITE", src,
                         "blocking the user site would block Flask with it")


class TestInstallationIsDenied(unittest.TestCase):
    """A model must not be able to install into the machine that grades it.

    One did: PEP 668 refused the install, its error text suggested
    --break-system-packages, and the model took the suggestion. Its package then
    shadowed every later repository, and an empty one scored 15 of 18 checks.
    """

    def test_the_harness_requires_a_virtualenv_for_pip(self):
        src = (Path(__file__).parent.parent / "src" / "appsift"
               / "harness.py").read_text(encoding="utf-8")
        self.assertIn('PIP_REQUIRE_VIRTUALENV="1"', src)
        self.assertIn("env=env", src, "the environment must reach the subprocess")

    def test_the_specification_states_how_to_run_without_installing(self):
        spec = TASK["files"]["SPEC.md"]
        self.assertIn("PYTHONPATH=src python -m todoapp", spec)
        self.assertIn("that includes this package", spec,
                      "the earlier wording let a model read the ban as covering "
                      "dependencies only, which is a fair reading of it")


if __name__ == "__main__":
    unittest.main()


class TestSessionCapture(unittest.TestCase):
    """A result has to be openable in opencode afterwards, or the counts are all
    there is when a session ends having written nothing."""

    def test_the_session_id_is_taken_from_the_stream(self):
        from appsift.harness import parse_events
        stream = "\n".join(json.dumps(e) for e in [
            {"type": "step_start", "sessionID": "ses_abc123"},
            {"type": "message", "sessionID": "ses_abc123",
             "part": {"type": "text", "text": "working"}},
        ])
        session = parse_events(stream)[6]
        self.assertEqual(session, "ses_abc123")

    def test_the_finish_reason_is_kept(self):
        # A model that ends its turn without calling a tool, one that exhausts the
        # context, and one that finishes all exit cleanly; only this tells them apart.
        from appsift.harness import parse_events
        for reason in ("stop", "length", "tool-calls"):
            stream = json.dumps({"type": "step_finish",
                                 "part": {"type": "step-finish", "reason": reason,
                                          "tokens": {"input": 10, "output": 2}}})
            self.assertEqual(parse_events(stream)[7], reason)

    def test_a_stream_without_one_is_not_an_error(self):
        from appsift.harness import parse_events
        self.assertEqual(parse_events(json.dumps({"type": "step_start"}))[6], "")


class TestRequirementText(unittest.TestCase):
    def test_every_requirement_is_explained(self):
        # A column with no explanation renders an empty cell, which reads as an
        # oversight rather than a requirement with nothing to say about it.
        from appsift.checks import todo_app_check
        from appsift.report import WHAT
        import re
        named = set(re.findall(r'check\("([a-z_]+)"', todo_app_check.__file__ and
                               Path(todo_app_check.__file__).read_text()))
        self.assertTrue(named, "no checks found to compare against")
        self.assertEqual(named - set(WHAT), set())


class TestEitherPackageLayoutIsAccepted(unittest.TestCase):
    """The specification asks for src/todoapp; a flat todoapp/ at the repository
    root is the other layout real Python projects use, and a model that chose it
    wrote a package exactly as valid as the one asked for.
    """

    def in_repo(self, layout):
        """A minimal but real repository using the given layout, as cwd."""
        tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        pkg = (tmp / "src" / "todoapp") if layout == "src" else (tmp / "todoapp")
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text("def create_app():\n    pass\n")
        (pkg / "__main__.py").write_text("")
        (tmp / "pyproject.toml").write_text("")
        old = Path.cwd()
        os.chdir(tmp)
        self.addCleanup(os.chdir, old)
        return tmp, pkg

    def reload(self):
        import importlib
        from appsift.checks import todo_app_check
        importlib.reload(todo_app_check)
        self.addCleanup(lambda: importlib.reload(todo_app_check))
        return todo_app_check

    def test_the_flat_layout_is_found_at_the_repository_root(self):
        tmp, pkg = self.in_repo("flat")
        mod = self.reload()
        path_root, found = mod.pkg_root()
        self.assertEqual(path_root, tmp)
        self.assertEqual(found, pkg)

    def test_the_src_layout_is_found_when_it_exists(self):
        tmp, pkg = self.in_repo("src")
        mod = self.reload()
        path_root, found = mod.pkg_root()
        self.assertEqual(path_root, tmp / "src")
        self.assertEqual(found, pkg)

    def test_a_flat_package_is_not_reported_as_missing(self):
        self.in_repo("flat")
        mod = self.reload()
        mod.RESULTS.clear()
        mod.check_layout()
        self.assertEqual(mod.RESULTS, [("package_layout", False, "missing: tests/test_*.py")])


class TestSessionLookup(unittest.TestCase):
    """Results predate the session id, so the conversation has to be findable
    without one. Matching is on model, task directory and start time."""

    def db(self, rows):
        import sqlite3
        path = Path(self.enterContext(tempfile.TemporaryDirectory())) / "oc.db"
        c = sqlite3.connect(path)
        c.execute("create table session (id text, project_id text, directory text,"
                  " model text, time_created integer)")
        c.execute("create table message (id text, session_id text,"
                  " time_created integer, data text)")
        c.execute("create table part (id text, message_id text, session_id text,"
                  " time_created integer, data text)")
        for row in rows:
            c.execute("insert into session values (?,?,?,?,?)", row)
        c.commit()
        c.close()
        return path

    def test_a_result_with_an_id_is_taken_at_its_word(self):
        from appsift import session
        self.assertEqual(session.resolve(dict(session="ses_x")), "ses_x")

    def test_an_older_result_is_matched_on_model_task_and_start(self):
        from appsift import session
        path = self.db([
            ("ses_right", "p", "/moved/away/m1__ag_todoapp", '{"id":"m1"}', 1000_000),
            ("ses_model", "p", "/moved/away/m1__ag_todoapp", '{"id":"m2"}', 1000_000),
            ("ses_task", "p", "/moved/away/m1__ag_other", '{"id":"m1"}', 1000_000),
        ])
        conn = session._connect(path)
        rec = dict(model="m1", repo="/somewhere/else/m1__ag_todoapp",
                   ts=1030.0, wall_s=30.0)
        self.assertEqual(session.resolve(rec, conn), "ses_right")

    def test_a_run_outside_the_window_is_not_claimed(self):
        from appsift import session
        path = self.db([("ses_old", "p", "/x/m1__ag_todoapp", '{"id":"m1"}', 1000_000)])
        conn = session._connect(path)
        rec = dict(model="m1", repo="/x/m1__ag_todoapp",
                   ts=1000 + session.MATCH_WINDOW_S + 60, wall_s=0)
        self.assertEqual(session.resolve(rec, conn), "")

    def test_a_missing_store_is_not_an_error(self):
        from appsift import session
        self.assertIsNone(session._connect(Path("/nonexistent/oc.db")))
        self.assertEqual(session.transcript("ses_x", None) if False else
                         session.transcript(""), [])


class TestRedoArchives(unittest.TestCase):
    """A sweep must not inherit the previous one. Results are chosen per model by
    best score, so a stale record would outrank the run that replaced it."""

    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.cfg = Config(results_dir=self.tmp)

    def test_redo_moves_the_previous_sweep_aside(self):
        ledger = self.cfg.path(appcli.LEDGER)
        ledger.write_text(json.dumps(dict(model="m", score=100.0)) + "\n")
        builds = self.tmp / "applications" / "m__ag_todoapp"
        builds.mkdir(parents=True)
        (builds / "kept.txt").write_text("x")

        appcli._archive(self.cfg, ledger, io.StringIO())

        self.assertFalse(ledger.exists(), "the ledger was left in place")
        self.assertFalse((self.tmp / "applications").exists())
        kept = list(self.tmp.glob("sweep-*"))
        self.assertEqual(len(kept), 1, "the previous sweep was not archived")
        self.assertTrue((kept[0] / appcli.LEDGER).exists(), "the ledger was not kept")
        self.assertTrue((kept[0] / "applications" / "m__ag_todoapp" / "kept.txt").exists(),
                        "moved aside, not deleted")

    def test_archiving_nothing_is_not_an_error(self):
        appcli._archive(self.cfg, self.cfg.path(appcli.LEDGER), io.StringIO())


class TestModelCannotReachTheHarness(unittest.TestCase):
    """A model cleaning up its own servers wrote `pkill -9 -f python`, which
    matches this harness's command line and killed the sweep driving it."""

    def test_the_harness_command_line_is_within_reach_of_such_a_kill(self):
        # The premise, stated so the mitigation is not mistaken for caution.
        self.assertIn("python", "python3 -u -m appsift --models-file models.txt")

    def test_opencode_is_launched_inside_a_process_namespace(self):
        from appsift import harness
        seen = {}

        class Proc:
            returncode = 0
            pid = os.getpid()
            def communicate(self, timeout=None): return "", ""
            def poll(self): return 0
            def wait(self, timeout=None): return 0
            def kill(self): pass

        def fake_popen(cmd, *a, **kw):
            seen["cmd"] = cmd
            return Proc()

        with tempfile.TemporaryDirectory() as d, \
             mock.patch.object(harness, "isolation", return_value=["unshare", "--pid"]), \
             mock.patch.object(harness.subprocess, "Popen", fake_popen), \
             mock.patch.object(harness, "verify", return_value=(False, "x", [])), \
             mock.patch.object(harness, "generation", create=True, return_value={}):
            harness.run_task("m:1", TASK, Path(d), timeout=5)
        self.assertEqual(seen["cmd"][:2], ["unshare", "--pid"],
                         "opencode was launched outside the namespace")
        self.assertIn("opencode", seen["cmd"])

    def test_it_still_runs_where_namespaces_are_unavailable(self):
        # Windows has none, and some distributions disable them by policy.
        from appsift import harness
        harness._ISOLATION = None
        with mock.patch.object(harness.shutil, "which", return_value=None):
            self.assertEqual(harness.isolation(), [])
        harness._ISOLATION = None


class TestTheModelIsDeclaredWithoutTouchingTheUsersConfig(unittest.TestCase):
    """opencode does not discover Ollama models by itself, and used to require
    the user to hand-edit their own opencode.jsonc before a run would resolve
    ollama/<model> at all. OPENCODE_CONFIG_CONTENT merges an inline declaration
    over whatever the user already has, so nothing needs to be pre-declared and
    nothing on disk is ever touched.
    """

    def run_with_fake_opencode(self, host):
        from appsift import harness
        seen = {}

        class Proc:
            returncode = 0
            pid = os.getpid()
            def communicate(self, timeout=None): return "", ""
            def poll(self): return 0
            def wait(self, timeout=None): return 0
            def kill(self): pass

        def fake_popen(cmd, *a, **kw):
            seen["env"] = kw.get("env") or {}
            return Proc()

        with tempfile.TemporaryDirectory() as d, \
             mock.patch.object(harness, "isolation", return_value=[]), \
             mock.patch.object(harness.subprocess, "Popen", fake_popen), \
             mock.patch.object(harness, "verify", return_value=(False, "x", [])), \
             mock.patch.object(harness, "generation", create=True, return_value={}):
            harness.run_task("qwen3.6:35b", TASK, Path(d), timeout=5, host=host)
        return seen["env"]

    def test_the_one_model_this_run_needs_is_declared_inline(self):
        import json
        env = self.run_with_fake_opencode("http://192.168.0.156:11434")
        config = json.loads(env["OPENCODE_CONFIG_CONTENT"])
        models = config["provider"]["ollama"]["models"]
        self.assertEqual(set(models), {"qwen3.6:35b"})

    def test_the_declared_endpoint_matches_the_host_this_run_was_given(self):
        import json
        env = self.run_with_fake_opencode("http://192.168.0.156:11434")
        config = json.loads(env["OPENCODE_CONFIG_CONTENT"])
        self.assertEqual(config["provider"]["ollama"]["options"]["baseURL"],
                         "http://192.168.0.156:11434/v1")

    def test_nothing_is_written_to_the_users_own_configuration(self):
        # Seeding the task's own files is a real write and must stay one; the
        # one path that must never be touched is the user's real opencode
        # config, wherever this machine happens to keep it.
        from appsift import harness
        real_write_text = Path.write_text
        touched = []

        def guarded(self, *a, **kw):
            if self == harness.OPENCODE_CONFIG:
                touched.append(self)
            return real_write_text(self, *a, **kw)

        with mock.patch.object(Path, "write_text", guarded):
            self.run_with_fake_opencode("http://localhost:11434")
        self.assertEqual(touched, [])


class TestWhatTheModelSaid(unittest.TestCase):
    """opencode's event stream carries text and tool calls but never reasoning, so
    the harness reads both from the store. A model that reasoned and then stopped
    must not be recorded as one whose output could not be understood."""

    def store(self, turns):
        import sqlite3
        path = Path(self.enterContext(tempfile.TemporaryDirectory())) / "oc.db"
        c = sqlite3.connect(path)
        c.execute("create table session (id text, project_id text, directory text,"
                  " model text, time_created integer)")
        c.execute("create table message (id text, session_id text,"
                  " time_created integer, data text)")
        c.execute("create table part (id text, message_id text, session_id text,"
                  " time_created integer, data text)")
        for i, parts in enumerate(turns):
            mid = f"m{i}"
            c.execute("insert into message values (?,?,?,?)",
                      (mid, "s1", i, json.dumps(dict(role="assistant",
                                                     tokens=dict(output=100)))))
            for j, part in enumerate(parts):
                c.execute("insert into part values (?,?,?,?,?)",
                          (f"p{i}{j}", mid, "s1", j, json.dumps(part)))
        c.commit(); c.close()
        return path

    def spoke(self, turns):
        from appsift import session
        return session.spoke("s1", session._connect(self.store(turns)))

    def test_reasoning_is_captured_even_though_the_stream_omits_it(self):
        got = self.spoke([[dict(type="reasoning", text="thinking hard")]])
        self.assertIn("thinking hard", got["said"])

    def test_a_turn_that_reasoned_and_stopped_counts_as_produced(self):
        # ornith:35b: 180 tokens of plan, no tool call, seven turns in.
        got = self.spoke([[dict(type="reasoning", text="I will write the files")]])
        self.assertTrue(got["produced"], "reasoning is something, not nothing")
        self.assertFalse(got["acted"], "it called no tool")

    def test_a_turn_that_yielded_nothing_is_distinguished(self):
        got = self.spoke([[dict(type="step-start"), dict(type="step-finish")]])
        self.assertFalse(got["produced"])
        self.assertFalse(got["acted"])

    def test_a_turn_that_called_a_tool_counts_as_acting(self):
        got = self.spoke([[dict(type="tool", tool="write", state={})]])
        self.assertTrue(got["acted"])
        self.assertTrue(got["produced"])

    def test_only_the_last_turn_decides_how_it_ended(self):
        got = self.spoke([[dict(type="tool", tool="write", state={})],
                          [dict(type="reasoning", text="and now I stop")]])
        self.assertFalse(got["acted"], "an earlier tool call is not the ending")
        self.assertTrue(got["produced"])


class TestDragMustBeWired(unittest.TestCase):
    """A check for drag and drop passed on the substring `drop`, which appears in
    DROP TABLE, drop-shadow and dropdown -- and in a comment saying the feature was
    not implemented."""

    def helper(self, name):
        path = Path(__file__).parent.parent / "src/appsift/checks/todo_app_check.py"
        src = path.read_text()
        ns = {"re": __import__("re")}
        exec(src[src.index("def _uncommented"):src.index("def rendered_dom")], ns)
        return ns[name]

    def test_a_line_comment_cannot_satisfy_a_check(self):
        u = self.helper("_uncommented")
        self.assertNotIn("drop",
                         u("// Simple drag-and-drop placeholder (not fully implemented)"))

    def test_an_html_comment_cannot_either(self):
        u = self.helper("_uncommented")
        self.assertNotIn("drag", u("<!-- drag and drop, later -->"))

    def test_real_code_survives_the_stripping(self):
        u = self.helper("_uncommented")
        self.assertIn("drop", u("la([c]).on('drop', async (el, target) => {})"))

    def test_a_url_is_not_mistaken_for_a_comment(self):
        u = self.helper("_uncommented")
        self.assertIn("dragula", u('src="https://cdn.example.com/dragula.min.js"'))


class TestLevels(unittest.TestCase):
    """The level is a milestone, not a share of checks met. The checker fails every
    request-driven requirement the moment nothing serves, so a percentage would rank
    an implementation that answers no request beside one that answers most."""

    ALL = ["package_layout", "pyproject", "importable", "serves", "create",
           "persist", "own_tests"]
    BLOCKED_BY_SERVES = {"create", "persist"}

    def rec(self, passed):
        got = set(passed)
        return dict(model="m", checks=[
            dict(name=n, passed=n in got,
                 detail="ok" if n in got else
                 ("the application did not start"
                  if n in self.BLOCKED_BY_SERVES and "serves" not in got
                  else "not met"))
            for n in self.ALL])

    def tier(self, passed):
        from appsift.report import _tier
        return _tier(self.rec(passed))

    def test_everything_met_is_complete(self):
        self.assertEqual(self.tier(self.ALL), "complete")

    def test_serving_and_answering_is_working(self):
        self.assertEqual(
            self.tier(["package_layout", "pyproject", "importable", "serves",
                       "create"]), "working")

    def test_serving_and_answering_nothing_is_running(self):
        self.assertEqual(
            self.tier(["package_layout", "pyproject", "importable", "serves"]),
            "running")

    def test_importing_without_serving_is_code(self):
        self.assertEqual(
            self.tier(["package_layout", "pyproject", "importable"]), "code")

    def test_files_that_do_not_import_are_files(self):
        self.assertEqual(self.tier(["package_layout", "pyproject"]), "files")

    def test_meeting_nothing_is_nothing(self):
        self.assertEqual(self.tier([]), "nothing")

    def test_a_running_model_outranks_one_that_met_more_but_never_served(self):
        # The point of the ladder: counting checks would invert these two.
        from appsift.report import TIERS, _tier
        order = [k for k, *_ in TIERS]
        served = self.rec(["importable", "serves"])
        never = self.rec(["package_layout", "pyproject", "importable", "own_tests"])
        self.assertLess(sum(1 for c in served["checks"] if c["passed"]),
                        sum(1 for c in never["checks"] if c["passed"]))
        self.assertLess(order.index(_tier(served)), order.index(_tier(never)))

    def test_the_levels_that_served_are_the_ones_whose_application_ran(self):
        # SERVED is what "usable for work of this kind" means, so it must hold
        # exactly the levels a running application can reach.
        from appsift.report import SERVED, _tier
        served = self.rec(["package_layout", "pyproject", "importable", "serves"])
        never = self.rec(["package_layout", "pyproject", "importable"])
        self.assertIn(_tier(served), SERVED)
        self.assertNotIn(_tier(never), SERVED)
        self.assertEqual(set(SERVED), {"complete", "working", "running"})

    def test_every_level_is_named_and_described(self):
        from appsift.report import TIERS, TIER_LABEL
        self.assertEqual(len(TIER_LABEL), len(TIERS))
        self.assertTrue(all(label and desc for _, label, desc in TIERS))

    def test_the_recommendation_orders_by_met_then_wall_time(self):
        # Wall time is the second criterion, only ever breaking a tie between
        # implementations that met the same requirements. It must never promote
        # one over a model that met more, however much faster it finished.
        from appsift.report import _field as _recommend
        def rec(name, passed, wall_s):
            r = self.rec(passed)
            r["model"], r["wall_s"] = name, wall_s
            return r
        serving = ["package_layout", "pyproject", "importable", "serves", "create"]
        page = _recommend([
            rec("working_slow", serving, 200),
            rec("running_fast", serving[:4], 5),
            rec("working_fast", serving, 60),
            rec("complete", self.ALL, 300),
        ])
        names = ["complete", "working_fast", "working_slow", "running_fast"]
        self.assertEqual(sorted(names, key=page.index), names,
                         "running_fast finished fastest and still ranks last")

    def test_each_level_has_its_own_colour(self):
        from appsift.report import TIERS, STYLE
        seen = set()
        for key, *_ in TIERS:
            token = f"--t-{key}:"
            self.assertIn(token, STYLE, key)
            value = STYLE.split(token, 1)[1].split(";", 1)[0].strip()
            self.assertNotIn(value, seen, f"{key} reuses {value}")
            seen.add(value)


class TestHtmlReport(unittest.TestCase):
    """The page has to say which requirement each model dropped.

    The score is close to a single bit, so the interesting part is where the bit
    was lost. A page showing only totals would repeat what the terminal already
    said and answer nothing further.
    """

    NAMES = ["package_layout", "pyproject", "importable", "serves", "seed_three",
             "task_shape", "create", "read_one", "update", "toggle_done", "delete",
             "filter_done", "filter_text", "reorder", "persist", "ui_page",
             "ui_drag", "ui_edit", "own_tests"]

    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.cfg = Config(results_dir=self.tmp)

    def write(self, rows):
        with (self.tmp / appcli.LEDGER).open("w", encoding="utf-8") as fh:
            for model, won, wall in rows:
                checks = [dict(name=n, passed=i < won,
                               detail="ok" if i < won else "not met")
                          for i, n in enumerate(self.NAMES)]
                fh.write(json.dumps(dict(
                    model=model, task="ag_todoapp", passed=won == len(self.NAMES),
                    checks=checks, score=round(100 * won / len(self.NAMES), 1),
                    wall_s=wall, turns=20,
                    repo=str(self.tmp / f"{model}_app"))) + "\n")

    def html(self, models=()):
        from appsift import report
        return report.write(self.cfg, self.tmp / "r.html", list(models)).read_text()

    def test_every_check_becomes_a_column(self):
        self.write([("m", 18, 100)])
        page = self.html()
        for name in self.NAMES:
            self.assertIn(name, page)

    def test_a_model_that_dropped_one_shows_which(self):
        # Which requirement was missed is the grid's job, one column per
        # requirement; the card only needs to say how many.
        self.write([("m", 17, 100)])
        page = self.html()
        self.assertIn("17 of 19 met", page)
        grid = page.split("<h2>Requirements</h2>")[1]
        self.assertIn('class="cell n"', grid)

    def test_the_field_is_ordered_by_how_much_was_met(self):
        self.write([("weak", 1, 10), ("strong", 18, 10), ("middling", 9, 10)])
        # Scoped to the grid: the recommendation above it names models in its own
        # order, and a bare page.index would measure that instead.
        grid = self.html().split("<h2>Requirements</h2>")[1]
        self.assertLess(grid.index("strong"), grid.index("middling"))
        self.assertLess(grid.index("middling"), grid.index("weak"))

    def test_a_column_carries_what_its_requirement_checks(self):
        # The grid shows which requirements the field found hard, column by
        # column; what each one means belongs on the header rather than in a
        # second table saying the same thing.
        self.write([("a", 3, 10), ("b", 3, 10), ("c", 18, 10)])
        page = self.html()
        self.assertIn("The application starts and answers on its port", page)
        header = page[page.index('<th class="rot"'):]
        self.assertIn("title=", header.split(">")[0] + ">")

    def test_a_screenshot_is_embedded_rather_than_linked(self):
        # The page has to survive being sent somewhere else.
        shot = self.tmp / "ui.png"
        shot.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
        with (self.tmp / appcli.LEDGER).open("w", encoding="utf-8") as fh:
            fh.write(json.dumps(dict(
                model="m", task="ag_todoapp", passed=True, wall_s=1.0, turns=1,
                score=100.0, screenshot=str(shot), repo=str(self.tmp),
                checks=[dict(name="serves", passed=True, detail="ok")])) + "\n")
        page = self.html()
        self.assertIn("data:image/png;base64,", page)
        self.assertNotIn(str(shot), page)

    def test_it_renders_from_an_empty_ledger(self):
        page = self.html()
        self.assertIn("<title>Local Agentic Coding Evaluation</title>", page)

    def test_both_themes_are_defined(self):
        self.write([("m", 18, 100)])
        page = self.html()
        self.assertIn("prefers-color-scheme: dark", page)
        self.assertIn('[data-theme="dark"]', page)
        self.assertIn("background:var(--paper)", page)


if __name__ == "__main__":
    unittest.main()


class _OnlyOpencodeFaked:
    """Replace Popen for opencode alone.

    The verification shells out to run the model's package and its tests, so a
    blanket patch would break the very thing being exercised. Scoped to the
    harness module and to the one command, everything else runs for real.
    """

    def __init__(self, stdout):
        self.stdout, self.real = stdout, subprocess.Popen
        self.commands, self.envs = [], []

    def __call__(self, cmd, *a, **kw):
        # The command is prefixed with the namespace wrapper, so opencode is no
        # longer argv[0]; matching on position would silently stop faking it.
        if cmd and "opencode" in cmd:
            self.commands.append(cmd)
            self.envs.append(kw.get("env") or {})
            return _FakeProc(self.stdout)
        return self.real(cmd, *a, **kw)


class _FakeProc:
    returncode = 0

    def __init__(self, stdout):
        self._stdout, self.pid = stdout, os.getpid()

    def communicate(self, timeout=None):
        return self._stdout, ""

    def poll(self):
        return 0

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass


class TestRunLoop(unittest.TestCase):
    """The loop itself, with only opencode and the Ollama client faked.

    Every other test of the command line patches run() out, which proves it is
    called and nothing about what happens inside it. A sweep runs for hours, so a
    fault in the loop is expensive to discover by running one.
    """

    STREAM = "\n".join(json.dumps(e) for e in [
        {"type": "step_start", "sessionID": "ses_loop"},
        {"type": "message", "sessionID": "ses_loop",
         "part": {"type": "text", "text": "I will not build anything."}},
        {"type": "step_finish",
         "part": {"type": "step-finish", "reason": "stop",
                  "tokens": {"input": 100, "output": 20}}},
    ])

    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.cfg = Config(results_dir=self.tmp, models=["a:1", "b:1"])
        self.popen = _OnlyOpencodeFaked(self.STREAM)
        self.enterContext(mock.patch.object(harness.subprocess, "Popen", self.popen))
        self.enterContext(mock.patch.object(harness, "preflight", return_value=None))
        self.unloaded = []
        # run() imports the client inside the function, so the module it comes
        # from is what has to be patched.
        self.enterContext(mock.patch("appsift.ollama.Ollama",
                                     lambda *a, **k: _FakeClient(self.unloaded)))

    def run_build(self, redo=False, variance=0, html=None):
        out = io.StringIO()
        code = appcli.run(self.cfg, timeout=30, redo=redo, stream=out,
                          variance=variance, html=html)
        return code, out.getvalue()

    def ledger(self):
        path = self.cfg.path(appcli.LEDGER)
        if not path.exists():
            return []
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

    def test_every_model_is_run_and_recorded_once(self):
        code, _ = self.run_build()
        self.assertEqual(code, 0)
        self.assertEqual([r["model"] for r in self.ledger()], ["a:1", "b:1"])

    def test_each_model_is_unloaded_after_its_build(self):
        # Two models sharing the card thrash; the sweep depends on this.
        self.run_build()
        self.assertEqual(self.unloaded, ["a:1", "b:1"])

    def test_a_second_run_skips_what_is_already_built(self):
        self.run_build()
        before = len(self.popen.commands)
        _, text = self.run_build()
        self.assertEqual(len(self.popen.commands), before, "it rebuilt a finished model")
        self.assertIn("already run", text)

    def test_redo_archives_the_previous_sweep_and_runs_again(self):
        self.run_build()
        _, text = self.run_build(redo=True)
        self.assertIn("previous sweep moved to", text)
        self.assertEqual([r["model"] for r in self.ledger()], ["a:1", "b:1"],
                         "the new sweep did not start from an empty ledger")
        self.assertEqual(len(list(self.tmp.glob("sweep-*"))), 1)

    def test_the_finish_reason_reaches_the_ledger(self):
        self.run_build()
        self.assertEqual({r["finish"] for r in self.ledger()}, {"stop"})
        self.assertEqual({r["session"] for r in self.ledger()}, {"ses_loop"})

    def test_a_model_that_built_nothing_is_recorded_as_failing(self):
        from appsift import harness
        self.run_build()
        for rec in self.ledger():
            self.assertFalse(harness.passed(rec))
            self.assertTrue(rec["checks"], "no checks were run at all")

    def test_the_ledger_holds_no_figure_it_could_recompute(self):
        # Whether it passed, the share met and the generation rate all follow
        # from the checks and the token counts; storing them means a change to
        # how they are derived needs a sweep to take effect.
        self.run_build()
        for rec in self.ledger():
            for derived in ("passed", "score", "detail", "gen_tok_s",
                            "output_tokens", "retained"):
                self.assertNotIn(derived, rec, f"{derived} is recomputable")

    def test_variance_adds_further_attempts_at_each_model(self):
        self.run_build(variance=2)
        recs = self.ledger()
        self.assertEqual(len(recs), 6, "two models, three attempts each")
        for model in ("a:1", "b:1"):
            attempts = sorted(r["attempt"] for r in recs if r["model"] == model)
            self.assertEqual(attempts, [1, 2, 3])

    def test_every_attempt_runs_at_the_same_default_sampling(self):
        # No attempt is forced to temperature 0: a model that fails
        # deterministically there but succeeds at its own sampling would
        # otherwise be graded on a mode nobody tuned it for.
        self.run_build(variance=2)
        for cmd, env in zip(self.popen.commands, self.popen.envs):
            self.assertNotIn("--agent", cmd)
            config = json.loads(env["OPENCODE_CONFIG_CONTENT"])
            self.assertNotIn("agent", config)

    def test_raising_variance_after_the_fact_tops_up_rather_than_restarts(self):
        self.run_build()                        # one greedy attempt each
        before = len(self.popen.commands)
        self.run_build(variance=1)               # now ask for two total
        self.assertEqual(len(self.popen.commands) - before, 2,
                         "only the missing attempt should have run, per model")
        for model in ("a:1", "b:1"):
            attempts = sorted(r["attempt"] for r in self.ledger() if r["model"] == model)
            self.assertEqual(attempts, [1, 2])

    def test_a_second_attempt_does_not_overwrite_the_first_ones_repository(self):
        self.run_build(variance=1)
        recs = {r["attempt"]: r for r in self.ledger() if r["model"] == "a:1"}
        self.assertNotEqual(recs[1]["repo"], recs[2]["repo"])
        self.assertTrue(Path(recs[1]["repo"]).is_dir())
        self.assertTrue(Path(recs[2]["repo"]).is_dir())

    def test_the_report_is_flushed_after_every_attempt_not_only_at_the_end(self):
        # A sweep costs hours; a reader checking in on report.html partway
        # through deserves something newer than a blank page.
        from appsift import report
        calls = []
        real_write = report.write

        def counting_write(cfg, output, models=None):
            calls.append(1)
            return real_write(cfg, output, models)

        html = self.tmp / "report.html"
        with mock.patch.object(report, "write", counting_write):
            self.run_build(variance=1, html=html)
        self.assertEqual(len(calls), 4, "two models, two attempts each")
        self.assertTrue(html.exists())


class _FakeClient:
    def __init__(self, seen):
        self.seen = seen

    def unload(self, model):
        self.seen.append(model)

    def loaded_context(self, model):
        return None


class TestTamperingIsRecorded(unittest.TestCase):
    """A modified protected file voids the grading, and the page says which file.

    The reason used to live in a prose field the ledger no longer keeps, so it is
    recorded as the files themselves and phrased where it is shown.
    """

    def test_the_card_names_the_file_that_was_modified(self):
        from appsift.report import _ending
        text = _ending(dict(model="m", checks=[], tampered=["tests/test_app.py"],
                            finish="stop"), "nothing")
        self.assertIn("tests/test_app.py", text)
        self.assertIn("Not graded", text)

    def test_an_untouched_build_says_nothing_about_tampering(self):
        from appsift.report import _ending
        self.assertNotIn("Not graded",
                         _ending(dict(model="m", checks=[], tampered=[],
                                      finish="stop"), "nothing"))


class TestPrimaryAttemptSelection(unittest.TestCase):
    """The one function both the terminal summary and the HTML report call to
    decide which attempt represents a model, so the two cannot disagree."""

    def rec(self, won, total=18):
        checks = [dict(name=f"c{i}", passed=i < won, detail="") for i in range(total)]
        return dict(model="m", checks=checks)

    def test_the_highest_scoring_attempt_wins(self):
        from appsift import harness
        best = self.rec(18)
        chosen = harness.primary([self.rec(4), best, self.rec(10)])
        self.assertEqual(chosen, best)

    def test_the_earliest_attempt_breaks_a_tie(self):
        # No attempt is more canonical than another -- every one runs at the
        # same sampling -- so a tie is broken by list order rather than by
        # any property of the attempt itself.
        from appsift import harness
        first = self.rec(10)
        chosen = harness.primary([first, self.rec(10)])
        self.assertIs(chosen, first)

    def test_a_single_attempt_is_its_own_primary(self):
        # A ledger from before repeat attempts existed holds exactly one
        # record per model; it must be read exactly as it always was.
        from appsift import harness
        only = self.rec(15)
        self.assertEqual(harness.primary([only]), only)


class TestRepeatAttemptsAreShownNotCollapsed(unittest.TestCase):
    """A model that scores well greedily but fails most sampled attempts must
    not read as simply successful -- the whole point of running more than
    once is to tell a fluke apart from a typical result, and folding every
    attempt into one number throws that away."""

    NAMES = ["a", "b", "c", "d"]

    def rec(self, model, won, sampling="greedy", attempt=1, finish="stop",
           final_produced=True, tool_calls=1, peak=None):
        checks = [dict(name=n, passed=i < won, detail="ok" if i < won else "not met")
                  for i, n in enumerate(self.NAMES)]
        return dict(model=model, checks=checks, sampling=sampling, attempt=attempt,
                   finish=finish, final_produced=final_produced,
                   tool_calls=tool_calls, peak_input_tokens=peak, wall_s=10)

    def test_the_higher_scoring_sampled_attempt_is_primary(self):
        from appsift.report import _records
        self.write_ledger([
            self.rec("m", 2, sampling="greedy", attempt=1),
            self.rec("m", 4, sampling="sampled", attempt=2),
        ])
        primary = _records(self.cfg, appcli.LEDGER, [])[0]
        self.assertEqual(sum(1 for c in primary["checks"] if c["passed"]), 4,
                         "the weaker greedy attempt was reported instead")

    def test_a_matching_attempt_counts_toward_the_figure(self):
        from appsift.report import _consistency_cell
        primary = self.rec("m", 4, attempt=1)
        primary["_attempts"] = [self.rec("m", 4, sampling="sampled", attempt=2)]
        cell = _consistency_cell(primary)
        self.assertIn(">2/2<", cell)
        self.assertIn("matched", cell)

    def test_a_failing_attempts_reason_is_on_the_tooltip(self):
        from appsift.report import _consistency_cell
        primary = self.rec("m", 4, attempt=1)
        primary["_attempts"] = [self.rec("m", 0, sampling="sampled", attempt=2,
                                        finish="length", peak=61204)]
        cell = _consistency_cell(primary)
        self.assertIn(">1/2<", cell)
        self.assertIn("attempt 2: 0%", cell)
        self.assertIn("61,204", cell)

    def test_no_further_attempts_means_a_dash(self):
        from appsift.report import _consistency_cell
        primary = self.rec("m", 4, attempt=1)
        self.assertIn("&mdash;", _consistency_cell(primary))

    def test_the_card_shows_only_the_best_attempt_undecorated(self):
        # The whole point of the redesign: a card is the best attempt alone,
        # with nothing about how the other attempts went written onto it.
        from appsift.report import _field
        primary = self.rec("m", 4, attempt=1, finish="length", peak=9001)
        primary["_attempts"] = [self.rec("m", 0, sampling="sampled", attempt=2)]
        card = _field([primary])
        self.assertNotIn("Other attempt", card)
        self.assertNotIn("attempt 2", card)

    def test_a_model_with_only_one_attempt_gets_no_section(self):
        from appsift.report import _model_sections, _all_names
        records = [self.rec("a", 4, attempt=1), self.rec("b", 2, attempt=1)]
        self.assertEqual(_model_sections(records, _all_names(records)), "")

    def test_every_run_being_a_lone_attempt_means_no_sections_at_all(self):
        # Confirms the all-models-ran-once case is a total skip, not one
        # section per model that each happen to hold a single row.
        from appsift.report import _model_sections, _all_names
        records = [self.rec("a", 4, attempt=1), self.rec("b", 2, attempt=1)]
        self.assertNotIn("<h2>a</h2>", _model_sections(records, _all_names(records)))
        self.assertNotIn("<h2>b</h2>", _model_sections(records, _all_names(records)))

    def test_a_model_with_more_than_one_attempt_gets_its_own_section(self):
        from appsift.report import _model_sections, _all_names
        primary = self.rec("m", 4, attempt=1)
        primary["_attempts"] = [self.rec("m", 1, sampling="sampled", attempt=2)]
        records = [primary]
        html = _model_sections(records, _all_names(records))
        self.assertIn("<h2>m</h2>", html)

    def test_a_models_section_documents_every_attempt_in_order(self):
        # Not scattered across separate "Attempt 1" / "Attempt 2" sections --
        # one model's own attempts, together, in the order it made them.
        from appsift.report import _model_sections, _all_names
        primary = self.rec("m", 4, attempt=1)
        primary["_attempts"] = [self.rec("m", 1, sampling="sampled", attempt=2)]
        records = [primary]
        html = _model_sections(records, _all_names(records))
        section = html[html.index("<h2>m</h2>"):]
        first_col = '<td class="figure">{}</td>'
        self.assertLess(section.index(first_col.format(1)),
                        section.index(first_col.format(2)),
                        "attempt 1 must come before attempt 2")

    def test_a_second_models_attempts_do_not_land_in_the_first_models_section(self):
        from appsift.report import _model_sections, _all_names
        a = self.rec("a", 4, attempt=1)
        a["_attempts"] = [self.rec("a", 1, sampling="sampled", attempt=2)]
        b = self.rec("b", 4, attempt=1)
        b["_attempts"] = [self.rec("b", 1, sampling="sampled", attempt=2)]
        html = _model_sections([a, b], _all_names([a, b]))
        self.assertEqual(html.count("<h2>"), 2, "one section per model")
        section_a = html[html.index("<h2>a</h2>"):html.index("<h2>b</h2>")]
        # One header row plus one row per attempt -- a's own two, not b's.
        self.assertEqual(section_a.count("<tr"), 3, "only a's own two attempts")

    def write_ledger(self, records):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.cfg = Config(results_dir=self.tmp)
        with (self.tmp / appcli.LEDGER).open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec) + "\n")


class TestAttemptNumbersLinkToTheirOwnTranscript(unittest.TestCase):
    """The Conversation button on a card reaches the best attempt's session.
    A run number in a model's own attempt-history section must reach that
    same attempt's own session, not always the best attempt's."""

    def db(self):
        import sqlite3
        path = Path(self.enterContext(tempfile.TemporaryDirectory())) / "oc.db"
        c = sqlite3.connect(path)
        c.execute("create table message (id text, session_id text,"
                  " time_created integer, data text)")
        c.execute("create table part (id text, message_id text, session_id text,"
                  " time_created integer, data text)")
        for sid in ("ses_1", "ses_2"):
            c.execute("insert into message values (?,?,?,?)",
                      (f"m_{sid}", sid, 0, json.dumps(dict(role="assistant"))))
            c.execute("insert into part values (?,?,?,?,?)",
                      (f"p_{sid}", f"m_{sid}", sid, 0,
                       json.dumps(dict(type="text", text="hi"))))
        c.commit()
        c.close()
        return path

    def test_each_attempt_writes_and_links_to_its_own_session(self):
        from appsift import report
        db = self.db()
        best = dict(model="m", attempt=2, session="ses_2", checks=[])
        best["_attempts"] = [dict(model="m", attempt=1, session="ses_1", checks=[])]
        tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        output = tmp / "r.html"
        with mock.patch.dict(os.environ, {"OPENCODE_DB": str(db)}):
            report._write_transcripts([best], output, dict(report_name="r.html"))
        self.assertEqual(best["_transcript"], "r_sessions/m.html")
        self.assertEqual(best["_attempts"][0]["_transcript"],
                         "r_sessions/m_attempt1.html")
        self.assertTrue((tmp / "r_sessions" / "m.html").exists())
        self.assertTrue((tmp / "r_sessions" / "m_attempt1.html").exists())

    def test_the_attempt_row_links_the_number_when_a_transcript_exists(self):
        from appsift import report
        a = dict(attempt=1, checks=[], _transcript="r_sessions/m_attempt1.html")
        row = report._attempt_row(a, [])
        self.assertIn('<a class="path-link" href="r_sessions/m_attempt1.html">1</a>',
                      row)

    def test_the_attempt_row_shows_a_bare_number_without_a_transcript(self):
        from appsift import report
        a = dict(attempt=1, checks=[])
        row = report._attempt_row(a, [])
        self.assertNotIn("path-link", row)
        self.assertIn('<td class="figure">1</td>', row)


class TestTheContextWindowIsReported(unittest.TestCase):
    """Not a setting this tool chooses: opencode has no way to request a num_ctx
    from Ollama's OpenAI-compatible endpoint, so every model gets whatever the
    server defaults to. Read back per model while it was loaded, and stated on
    the page so a reader on a smaller machine can tell whether these results
    were measured at a window their own setup would never reach."""

    def rec(self, context_length=None):
        return dict(model="m", context_length=context_length)

    def test_nothing_measured_says_so_honestly(self):
        from appsift.report import _windows
        self.assertEqual(_windows([self.rec()]), "the server's own default context")

    def test_one_window_across_the_field_reads_as_one(self):
        from appsift.report import _windows
        records = [self.rec(context_length=65536), self.rec(context_length=65536)]
        self.assertEqual(_windows(records), "a 65,536-token context")

    def test_differing_windows_are_all_named_not_just_the_largest(self):
        from appsift.report import _windows
        records = [self.rec(context_length=8192), self.rec(context_length=65536)]
        phrase = _windows(records)
        self.assertIn("8,192", phrase)
        self.assertIn("65,536", phrase)


class TestThePageSaysWhatItMeasuredAndStops(TestHtmlReport):
    """The copy states the rule and the figure. It does not argue for either.

    Ported from llm-codesift, where the same two mistakes recurred across
    several edits until a test started catching them: internal vocabulary
    leaking onto the page, and a lede that argues for a design choice instead
    of stating what was measured. Caught here by eye a third time in appsift
    before this test existed to hold the line.
    """

    INTERNAL = ("ledger", "jsonl", "tier", "checker")

    def page(self):
        self.write([("m", 18, 100)])
        return self.html()

    def ledes(self, page):
        import re
        return re.findall(r'<div class="lede"><p>(.*?)</p>', page, re.S) + \
            re.findall(r'<p class="sub">(.*?)</p>', page, re.S)

    def test_no_internal_vocabulary_reaches_the_reader(self):
        page = self.page().lower()
        body = page[page.index('<div class="wrap">'):]
        for word in self.INTERNAL:
            with self.subTest(word=word):
                self.assertNotIn(word, body, f"{word!r} is how the code talks, not the page")

    def test_no_lede_appends_an_aside(self):
        import re
        for lede in self.ledes(self.page()):
            prose = re.sub(r"<code>.*?</code>", "", lede, flags=re.S)
            with self.subTest(lede=lede[:40]):
                self.assertNotIn("--", prose)
                self.assertNotIn("—", prose)

    def test_levels_and_milestones_are_not_both_used_as_the_same_word(self):
        # A page that names its own concept two ways in two sentences reads as
        # unedited, even when neither sentence is wrong on its own.
        page = self.page()
        body = page[page.index('<div class="wrap">'):]
        self.assertNotIn("milestone", body.lower())


class TestAReportAlreadyInsideItsDataDirIsNotNestedAgain(unittest.TestCase):
    """--html report_data/report.html names a report that already lives inside
    its own companion data directory. data_dir must reuse report_data as-is,
    not append _data a second time and produce report_data/report_data.
    """

    def test_a_report_already_inside_its_data_dir_does_not_double_it(self):
        from appsift.config import data_dir
        self.assertEqual(data_dir("report_data/report.html"), Path("report_data"))

    def test_a_bare_report_still_gets_a_sibling_data_dir(self):
        from appsift.config import data_dir
        self.assertEqual(data_dir("report.html"), Path("report_data"))

    def test_a_report_under_an_unrelated_directory_gets_its_own_data_dir(self):
        from appsift.config import data_dir
        self.assertEqual(data_dir("out/report.html"), Path("out/report_data"))
