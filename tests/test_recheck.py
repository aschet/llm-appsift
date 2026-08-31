# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT
"""python -m appsift.recheck: a mistake in the checking code must not cost the
sweep that produced the applications. The application is kept, so the checks
can run again against the real application, off the primary appsift command."""
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from appsift import cli as appcli
from appsift import harness
from appsift import recheck
from appsift.config import Config


class TestRecheck(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.cfg = Config(results_dir=self.tmp)
        self.repo = self.tmp / "applications" / "m__ag_todoapp"
        self.repo.mkdir(parents=True)

    def record(self, **over):
        rec = dict(model="m:1", task="ag_todoapp", passed=False, detail="1/2 checks",
                   checks=[dict(name="a", passed=True, detail="ok"),
                           dict(name="b", passed=False, detail="not met")],
                   score=50.0, wall_s=99.0, turns=7, session="ses_x",
                   gen_tok_s=42.0, output_tokens=1234, peak_input_tokens=8000,
                   finish="stop", repo=str(self.repo))
        rec.update(over)
        return rec

    def write(self, records):
        with self.cfg.path(appcli.LEDGER).open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec) + "\n")

    def fake_verify(self, passed, checks):
        return mock.patch.object(harness, "verify",
                                 return_value=(passed, f"{sum(c['passed'] for c in checks)}"
                                                       f"/{len(checks)} checks", checks))

    def ledger(self):
        return [json.loads(l) for l in
                self.cfg.path(appcli.LEDGER).read_text().splitlines() if l.strip()]

    def test_a_corrected_check_changes_the_result(self):
        self.write([self.record()])
        both = [dict(name="a", passed=True, detail="ok"),
                dict(name="b", passed=True, detail="ok")]
        with self.fake_verify(True, both):
            out = io.StringIO()
            recheck.run(self.cfg, [], apply=True, stream=out)
        self.assertIn("1 of 2 becomes 2 of 2", out.getvalue())
        rec = self.ledger()[0]
        self.assertTrue(harness.passed(rec))
        self.assertEqual(harness.score(rec), 100.0)

    def test_what_the_model_did_is_not_re_derived(self):
        # Turns, tokens, rate, session and the reason it stopped were observed once.
        self.write([self.record()])
        with self.fake_verify(True, [dict(name="a", passed=True, detail="ok")]):
            recheck.run(self.cfg, [], apply=True, stream=io.StringIO())
        rec = self.ledger()[0]
        for field, value in (("turns", 7), ("session", "ses_x"), ("gen_tok_s", 42.0),
                             ("output_tokens", 1234), ("peak_input_tokens", 8000),
                             ("finish", "stop"), ("wall_s", 99.0)):
            self.assertEqual(rec[field], value, field)
        self.assertIn("rechecked_ts", rec)

    def test_offsetting_changes_are_not_reported_as_unchanged(self):
        # One check gained and another lost leaves the total identical; reporting
        # that as unchanged would hide two corrections.
        self.write([self.record()])
        flipped = [dict(name="a", passed=False, detail="not met"),
                   dict(name="b", passed=True, detail="ok")]
        with self.fake_verify(False, flipped):
            out = io.StringIO()
            recheck.run(self.cfg, [], apply=False, stream=out)
        text = out.getvalue()
        self.assertNotIn("unchanged", text)
        self.assertIn("-a", text)
        self.assertIn("+b", text)
        self.assertIn("1 result(s) would change", text)

    def test_nothing_is_written_without_apply(self):
        self.write([self.record()])
        before = self.cfg.path(appcli.LEDGER).read_text()
        with self.fake_verify(True, [dict(name="a", passed=True, detail="ok")]):
            out = io.StringIO()
            recheck.run(self.cfg, [], apply=False, stream=out)
        self.assertEqual(self.cfg.path(appcli.LEDGER).read_text(), before)
        self.assertIn("nothing written", out.getvalue())

    def test_the_previous_ledger_is_kept(self):
        self.write([self.record()])
        with self.fake_verify(True, [dict(name="a", passed=True, detail="ok")]):
            recheck.run(self.cfg, [], apply=True, stream=io.StringIO())
        backups = list(self.tmp.glob("*.jsonl.bak"))
        self.assertEqual(len(backups), 1)
        self.assertIn('"passed": false', backups[0].read_text())

    def test_a_build_no_longer_on_disk_is_left_as_recorded(self):
        # The one case that still needs the model.
        self.write([self.record(repo=str(self.tmp / "gone"))])
        out = io.StringIO()
        recheck.run(self.cfg, [], apply=True, stream=out)
        self.assertIn("no longer on disk", out.getvalue())
        self.assertEqual(self.ledger()[0]["score"], 50.0)

    def test_it_can_be_narrowed_to_named_models(self):
        self.write([self.record(model="wanted"), self.record(model="other")])
        with self.fake_verify(True, [dict(name="a", passed=True, detail="ok")]):
            out = io.StringIO()
            recheck.run(self.cfg, ["wanted"], apply=True, stream=out)
        self.assertIn("wanted", out.getvalue())
        self.assertNotIn("other:", out.getvalue())
        by = {r["model"]: r for r in self.ledger()}
        self.assertEqual(by["other"]["score"], 50.0, "an unnamed model was touched")

    def test_checking_litter_does_not_survive_the_recheck(self):
        (self.repo / "_verify_check.py").write_text("x")
        (self.repo / "_grading_store").mkdir()
        harness.tidy(self.repo)
        self.assertEqual(list(self.repo.iterdir()), [])


class TestRecheckIsOffThePrimaryCommand(unittest.TestCase):
    """Regrading is a maintenance operation on this tool's own output, not
    something a user runs day to day -- it must not clutter `appsift --help`,
    the same way codesift keeps its own regrade stage off `codesift run`."""

    def test_recheck_is_not_offered_by_the_primary_command(self):
        with self.assertRaises(SystemExit), \
             mock.patch("sys.stderr", new=io.StringIO()):
            appcli.main(["--recheck"])

    def test_recheck_is_its_own_runnable_module(self):
        self.assertTrue(callable(getattr(recheck, "main", None)))

    def test_the_recheck_help_warns_about_execution(self):
        out = io.StringIO()
        with mock.patch("sys.stdout", out), self.assertRaises(SystemExit):
            recheck.main(["--help"])
        self.assertIn("executed without a sandbox", out.getvalue())
