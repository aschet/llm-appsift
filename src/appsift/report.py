# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT
"""A page showing the field side by side.

The terminal summary answers "how did each model do". This answers the question
that follows: which requirement did each one drop, and what does the thing it
made actually look like. Both matter, because the score is close to a single bit
and the interesting part is where the bit was lost.

The checks are laid out as a grid, one column per requirement, so a row can be
read across at a glance and a column read down to see which requirement the field
found hardest. Screenshots are embedded, since an application nobody looks at is
just a number.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import html
import json
from pathlib import Path

from . import harness
from . import ledger
from . import session as sessions
from .config import DEFAULT_OUTPUT, Config, data_dir, read_model_file

E = html.escape


def _records(cfg: Config, ledger_name: str, models: list[str]) -> list[dict]:
    """One record per model: the one the card and the grid are built from,
    chosen by harness.primary(). Any further attempts ride along on
    `_attempts`, for a reader deciding how much to trust the headline figure
    rather than for the figure itself.
    """
    path = Path(cfg.results_dir) / ledger_name
    by_model = {}
    for rec in ledger.read(path):
        if models and rec["model"] not in models:
            continue
        by_model.setdefault(rec["model"], []).append(rec)

    out = []
    for attempts in by_model.values():
        best = harness.primary(attempts)
        others = sorted((r for r in attempts if r is not best),
                        key=lambda r: r.get("attempt") or 0)
        out.append(dict(best, _attempts=others))
    return sorted(out, key=lambda r: -harness.score(r))


def _shot(rec: dict) -> str:
    """The interface, inlined, so the page is one file that can be sent anywhere.

    Returns the data: URI itself rather than a finished <img> tag: the card
    links to the full image as well as displaying a thumbnail of it, and both
    need the same source.
    """
    path = rec.get("screenshot")
    if not path or not Path(path).exists():
        return ""
    data = base64.b64encode(Path(path).read_bytes()).decode()
    if len(data) > 4_000_000:
        return ""
    return f"data:image/png;base64,{data}"


def _env(cfg: Config) -> dict:
    """What produced the page: this tool, the server, and the harness it drove.

    Not the card. Reading it meant shelling out to a vendor tool that is absent
    on most machines and says nothing the page uses, and a model's result does
    not depend on which GPU served it.
    """
    return dict(host=cfg.host, remote=cfg.is_remote,
                line=" \u00b7 ".join(["appsift", "Ollama", "opencode"]))


# The checker marks every requirement it could not attempt with this exact detail,
# so the count of blocked checks is read from the record rather than assumed.
# How far a model got, in the order the checker itself imposes. Each step is a
# milestone the implementation either reached or did not, so the ladder is the
# structure of the task rather than a set of chosen cut-offs: files exist, the
# package imports, the application serves, requests are answered, everything is
# met. A percentage would put an implementation that answers no request next to
# one that answers most of them.
# Only the levels that served are usable for work of this kind, which is what
# SERVED names.
TIERS = [
    ("complete", "complete", "Met every requirement."),
    ("working", "working", "Serves, and answers some requests but not all."),
    ("running", "running", "Serves, but answers none of the requests."),
    ("code", "code only", "The package imports, but never served."),
    ("files", "files only", "Files were written, but the package does not import."),
    ("nothing", "nothing", "No requirement was met."),
]
TIER_LABEL = {k: label for k, label, _ in TIERS}
SERVED = ("complete", "working", "running")


def _tier(rec: dict) -> str:
    """Which milestone this implementation reached, read from the checks alone."""
    checks = rec.get("checks") or []
    if not checks:
        return "nothing"
    got = {c["name"] for c in checks if c["passed"]}
    if len(got) == len(checks):
        return "complete"
    if not got:
        return "nothing"
    if "serves" in got:
        # Everything outside the structural checks and the model's own test suite
        # is answered by handling a request, so anything left here means the
        # application did more than start.
        answered = got - {"package_layout", "pyproject", "importable", "serves",
                          "own_tests"}
        return "working" if answered else "running"
    if "importable" in got:
        return "code"
    return "files"


def _bar(met: float, total: int) -> str:
    # Neutral fill: the tier pill beside the model already carries the verdict,
    # so the bar itself only needs to show the proportion. The count behind the
    # percentage rides on hover, where a reader lands when the figure alone
    # is not enough.
    pct = 100 * met / total if total else 0
    return (f'<div class="barcell" title="{met:g} of {total} met"><div class="bar">'
            f'<span style="width:{pct:.0f}%" class="s-plain"></span></div>'
            f'<span class="figure strong">{pct:.0f}<span class="pc">%</span></span></div>')


STYLE = '''
:root {
  --paper:#f6f7f9; --raised:#ffffff; --ink:#0f1319; --muted:#5b6472; --line:#dfe3ea;
  --accent:#3a6ea5; --accent-soft:#e8eef6;
  --good:#2f7d4f; --warn:#b07d1a; --bad:#b4453a;
  --good-bg:#e9f3ec; --warn-bg:#f8f1e0; --bad-bg:#f8ebe9;
  --t-complete:#2f7d4f; --t-working:#2b7d75; --t-running:#b07d1a;
  --t-code:#c2661c; --t-files:#b4453a; --t-nothing:#7e2a24;
  --t-complete-bg:#e9f3ec; --t-working-bg:#e2f1ef; --t-running-bg:#f8f1e0;
  --t-code-bg:#fbeadd; --t-files-bg:#f8ebe9; --t-nothing-bg:#f2dedb;
  --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
  --sans:"IBM Plex Sans",system-ui,-apple-system,sans-serif;
  --display:"Archivo","IBM Plex Sans",system-ui,sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --paper:#0f1319; --raised:#161b23; --ink:#e6eaf0; --muted:#8d97a6; --line:#262d38;
    --accent:#7aa9dc; --accent-soft:#1a2531;
    --good:#63b585; --warn:#d6a548; --bad:#e0776b;
    --good-bg:#15251c; --warn-bg:#2a2113; --bad-bg:#2b1917;
    --t-complete:#63b585; --t-working:#5cbfb4; --t-running:#d6a548;
    --t-code:#e39055; --t-files:#e0776b; --t-nothing:#c2564c;
    --t-complete-bg:#15251c; --t-working-bg:#122624; --t-running-bg:#2a2113;
    --t-code-bg:#2e1d12; --t-files-bg:#2b1917; --t-nothing-bg:#351a17;
  }
}
:root[data-theme="dark"] {
  --paper:#0f1319; --raised:#161b23; --ink:#e6eaf0; --muted:#8d97a6; --line:#262d38;
  --accent:#7aa9dc; --accent-soft:#1a2531;
  --good:#63b585; --warn:#d6a548; --bad:#e0776b;
  --good-bg:#15251c; --warn-bg:#2a2113; --bad-bg:#2b1917;
  --t-complete:#63b585; --t-working:#5cbfb4; --t-running:#d6a548;
  --t-code:#e39055; --t-files:#e0776b; --t-nothing:#c2564c;
  --t-complete-bg:#15251c; --t-working-bg:#122624; --t-running-bg:#2a2113;
  --t-code-bg:#2e1d12; --t-files-bg:#2b1917; --t-nothing-bg:#351a17;
}
* { box-sizing:border-box; }
body { background:var(--paper); color:var(--ink); font-family:var(--sans);
  line-height:1.6; margin:0; padding:clamp(1.5rem,4vw,4rem) clamp(1rem,4vw,2rem); }
.wrap { max-width:74rem; margin:0 auto; display:flex; flex-direction:column; gap:3.5rem; }
h1,h2,h3 { font-family:var(--display); text-wrap:balance; margin:0; letter-spacing:-.015em; }
h1 { font-size:clamp(2rem,5vw,3rem); font-weight:700; line-height:1.05; }
h2 { font-size:1.4rem; font-weight:600; padding-bottom:.6rem; border-bottom:2px solid var(--accent);
  margin-bottom:.25rem; }
h3 { font-size:1rem; font-weight:600; font-family:var(--mono); letter-spacing:-.02em; }
.eyebrow { font-family:var(--mono); font-size:.7rem; text-transform:uppercase;
  letter-spacing:.16em; color:var(--accent); font-weight:600; margin:0 0 .9rem; }
.lede { color:var(--muted); margin:.35rem 0 1.25rem; }
.lede p { margin:0 0 .7rem; }
.lede p:last-child { margin-bottom:0; }
header.top p.sub { color:var(--muted); font-size:1.05rem; margin:1rem 0 0; }
.meta { display:flex; flex-wrap:wrap; gap:.5rem 1.75rem; font-family:var(--mono);
  font-size:.76rem; color:var(--muted); margin-top:1.5rem; padding-top:1.25rem;
  border-top:1px solid var(--line); }
.meta b { color:var(--ink); font-weight:600; }
section { display:flex; flex-direction:column; }
.scroll { overflow-x:auto; border:1px solid var(--line); border-radius:3px;
  background:var(--raised); }
table { border-collapse:collapse; width:100%; font-size:.85rem; }
th,td { text-align:left; padding:.62rem .85rem; border-bottom:1px solid var(--line);
  white-space:nowrap; }
thead th { font-family:var(--sans); font-size:.74rem; text-transform:uppercase;
  letter-spacing:.07em; color:var(--ink); font-weight:600; background:var(--accent-soft);
  padding:.85rem .85rem .7rem; border-bottom:2px solid var(--accent); vertical-align:bottom; }
tbody th { font-family:var(--mono); font-weight:500; font-size:.82rem; }
tbody tr:last-child td, tbody tr:last-child th { border-bottom:none; }
tr.r-complete th { box-shadow:inset 3px 0 0 var(--t-complete); }
.card.c-complete { border-left-color:var(--t-complete); }
.p-complete { background:var(--t-complete-bg); color:var(--t-complete); }
tr.r-working th { box-shadow:inset 3px 0 0 var(--t-working); }
.card.c-working { border-left-color:var(--t-working); }
.p-working { background:var(--t-working-bg); color:var(--t-working); }
tr.r-running th { box-shadow:inset 3px 0 0 var(--t-running); }
.card.c-running { border-left-color:var(--t-running); }
.p-running { background:var(--t-running-bg); color:var(--t-running); }
tr.r-code th { box-shadow:inset 3px 0 0 var(--t-code); }
.card.c-code { border-left-color:var(--t-code); }
.p-code { background:var(--t-code-bg); color:var(--t-code); }
tr.r-files th { box-shadow:inset 3px 0 0 var(--t-files); }
.card.c-files { border-left-color:var(--t-files); }
.p-files { background:var(--t-files-bg); color:var(--t-files); }
tr.r-nothing th { box-shadow:inset 3px 0 0 var(--t-nothing); }
.card.c-nothing { border-left-color:var(--t-nothing); }
.p-nothing { background:var(--t-nothing-bg); color:var(--t-nothing); }
.figure { font-family:var(--mono); font-variant-numeric:tabular-nums; text-align:right; }
.strong { font-weight:600; }
.pc { font-size:.78em; color:var(--muted); }
.barcell { display:flex; align-items:center; gap:.6rem; min-width:8.5rem; }
.s-plain { background:var(--accent); }
.bar { flex:1; height:6px; background:var(--accent-soft); border-radius:1px; overflow:hidden;
  min-width:4rem; }
.bar span { display:block; height:100%; }
th.rot { height:8.5rem; vertical-align:bottom; padding:0 0 .55rem; width:1.9rem;
  background:var(--accent-soft); border-bottom:2px solid var(--accent); }
th.rot span { writing-mode:vertical-rl; transform:rotate(180deg); font-family:var(--mono);
  font-weight:500; font-size:.72rem; letter-spacing:0; text-transform:none; color:var(--muted); }
.cell { text-align:center; width:1.9rem; font-size:.9rem; padding:.62rem .2rem; }
.cell.y { color:var(--good); background:var(--good-bg); }
.cell.n { color:var(--bad); background:var(--bad-bg); }
.cell.w { color:var(--warn); background:var(--warn-bg); }
.cell.none { color:var(--muted); }
.cards { display:grid; gap:1rem; grid-template-columns:repeat(auto-fill,minmax(20rem,1fr)); }
.card { background:var(--raised); border:1px solid var(--line); border-radius:3px;
  border-left:4px solid var(--muted); padding:1.1rem 1.2rem; display:flex;
  flex-direction:column; gap:.85rem; }
.card header { display:flex; align-items:center; justify-content:space-between; gap:.5rem; }
.card dl { margin:0; display:grid; grid-template-columns:repeat(2,1fr); gap:.6rem .75rem; }
.card dl div { display:flex; flex-direction:column; gap:.1rem; }
dt { font-size:.68rem; text-transform:uppercase; letter-spacing:.09em; color:var(--muted); }
dd { margin:0; font-family:var(--mono); font-size:1.15rem; font-weight:500;
  font-variant-numeric:tabular-nums; }
.shot { aspect-ratio:1280/900; border:1px solid var(--line); border-radius:2px;
  overflow:hidden; display:block; cursor:zoom-in; }
.shot img { width:100%; height:100%; object-fit:cover; display:block;
  transition:opacity .15s; }
.shot:hover img { opacity:.85; }
.card .ending { margin:0; font-size:.78rem; color:var(--warn); font-family:var(--mono); }
.acts { margin:0; display:flex; gap:.4rem; }
.act { flex:1; font-family:var(--mono); font-size:.72rem; text-decoration:none;
  color:var(--accent); background:var(--accent-soft); border-radius:2px;
  padding:.4rem .55rem; text-align:center; }
.act:hover { text-decoration:underline; }
.act:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
.path-link { text-decoration:none; border-bottom:1px dotted var(--line);
  color:var(--accent); }
.path-link:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
.shot.none { display:flex; align-items:center; justify-content:center;
  border-style:dashed; }
.shot.none p { color:var(--muted); font-size:.85rem; margin:0; font-family:var(--mono);
  text-align:center; }
.pill { font-family:var(--mono); font-size:.66rem; text-transform:uppercase; letter-spacing:.08em;
  padding:.2rem .5rem; border-radius:2px; font-weight:600; white-space:nowrap; }
code { font-family:var(--mono); font-size:.88em; background:var(--accent-soft);
  padding:.1rem .3rem; border-radius:2px; }
.empty { background:var(--raised); border:1px solid var(--line); border-radius:3px;
  padding:1.4rem 1.6rem; color:var(--muted); }
@media (max-width:34rem) { .card dl { grid-template-columns:1fr; } }
'''


def _figures(rec: dict) -> str:
    """The two figures that decide something at a glance: did it work, and
    what did that cost.

    Turns, peak context, written and generation speed are all left off: none
    of them ranks the field on its own, and each stays in the requirements
    grid, where a reader is already looking at one model closely.
    """
    checks = rec.get("checks") or []
    met = sum(harness.check_score(c) for c in checks)
    total = len(checks) or 1
    return f'''<dl>
        <div><dt>Completeness</dt><dd title="{met:g} of {total}">{100 * met / total:.0f}<span
          class="pc">%</span></dd></div>
        <div><dt>Time</dt><dd>{rec.get("wall_s") or 0:.0f}<span class="pc">s</span></dd></div>
      </dl>'''


# Each requirement's column heading and what it verifies, in the checker's own
# terms. The heading is what a reader sees; the id itself (`seed_three`) means
# nothing to them and never appears on the page. The description rides beside
# it as a tooltip, for the column a field dropped saying what was actually
# asked of it.
WHAT = {
    "package_layout": ("Package Layout",
                       "The package, its entry module, pyproject.toml and a test file exist"),
    "pyproject": ("Project Metadata",
                 "Declares a name, a version, Flask, a console entry point and a build backend"),
    "importable": ("Importable",
                  "create_app() returns an application, imported from this repository"),
    "serves": ("Serves", "The application starts and answers on its port"),
    "seed_three": ("Seed Data", "An empty store is seeded with three entries"),
    "task_shape": ("Entry Fields",
                   "Entries carry id, title, description, color, done and position, "
                   "correctly typed"),
    "create": ("Create", "POST adds an entry and returns it"),
    "read_one": ("Read", "The entry just created can be read back by id"),
    "update": ("Update", "A patch changes the named fields and leaves the others alone"),
    "toggle_done": ("Toggle Done",
                    "Marking an entry done sets the flag without discarding the entry"),
    "delete": ("Delete", "A deleted entry is gone"),
    "filter_done": ("Filter By Status", "The listing can be filtered by done state"),
    "filter_text": ("Filter By Text", "The listing can be filtered by text"),
    "reorder": ("Reorder", "Entries can be moved into a given order"),
    "persist": ("Persistence",
               "State written through the API survives a restart against the same store"),
    "ui_page": ("Web Page",
               "The root returns an HTML page showing the entries with a field to add one"),
    "ui_drag": ("Drag To Reorder", "The page carries drag handling for reordering"),
    "ui_edit": ("Edit", "An existing entry's title, description or colour can be edited "
               "from the page"),
    "own_tests": ("Own Tests", "The model's own test suite passes, run in isolation"),
}


# A tool result can be an entire file. Enough to see what came back, not so much
# that one read buries the conversation around it.
OUTPUT_KEPT = 4000


def _slug(model: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-." else "_" for ch in model)


def _turn(turn: dict) -> str:
    role = turn.get("role") or "assistant"
    body = ""
    for part in turn.get("parts") or []:
        kind = part.get("kind")
        if kind == "text":
            body += f'<p class="say">{E(part["text"])}</p>'
        elif kind == "reasoning":
            body += (f'<details class="think" open><summary>reasoning</summary>'
                     f'<pre>{E(part["text"])}</pre></details>')
        elif kind == "tool":
            out = part.get("output")
            out = "" if out is None else str(out)
            clipped = len(out) > OUTPUT_KEPT
            shown = out[:OUTPUT_KEPT] + ("\n... truncated" if clipped else "")
            args = part.get("input")
            args = "" if args is None else json.dumps(args, indent=2)[:1200]
            status = part.get("status") or ""
            body += (f'<details class="tool"><summary><code>{E(part.get("tool") or "")}</code>'
                     f'<span class="pill p-{"complete" if status == "completed" else "files"}">'
                     f'{E(status or "?")}</span>'
                     f'<span class="ttitle">{E(part.get("title") or "")}</span></summary>'
                     + (f'<pre class="args">{E(args)}</pre>' if args else "")
                     + (f'<pre>{E(shown)}</pre>' if shown else "")
                     + (f'<pre class="err">{E(str(part["error"]))[:800]}</pre>'
                        if part.get("error") else "")
                     + '</details>')
    if not body:
        return ""
    return f'<article class="turn t-{E(role)}"><h3>{E(role)}</h3>{body}</article>'


def _transcript_page(rec: dict, turns: list, env: dict) -> str:
    """The conversation on its own page, in the same hand as the report."""
    tools = sum(1 for t in turns for p in t["parts"] if p["kind"] == "tool")
    return f'''<title>{E(rec["model"])} &mdash; session</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500&display=swap">
<style>{STYLE}{TRANSCRIPT_STYLE}</style>
<div class="wrap">
  <header class="top">
    <p class="eyebrow"><a class="path-link" href="../{E(env["report_name"])}">back to the report</a></p>
    <h1>{E(rec["model"])}</h1>
    <p class="sub">The exchange between the harness and this model. Each tool
    call shows the arguments it was given and the result it returned.</p>
    <div class="meta">
      <span><b>{len(turns)}</b> turns</span>
      <span><b>{tools}</b> tool calls</span>
      <span><b>{rec.get("wall_s") or 0:.0f}</b>s</span>
    </div>
  </header>
  <section class="convo">{"".join(_turn(t) for t in turns)}</section>
</div>'''


TRANSCRIPT_STYLE = '''
.convo { display:flex; flex-direction:column; gap:1rem; }
.turn { background:var(--raised); border:1px solid var(--line); border-radius:3px;
  border-left:4px solid var(--line); padding:1rem 1.2rem; display:flex;
  flex-direction:column; gap:.7rem; }
.turn.t-user { border-left-color:var(--accent); }
.turn.t-assistant { border-left-color:var(--t-complete); }
.turn h3 { font-size:.7rem; text-transform:uppercase; letter-spacing:.12em;
  color:var(--muted); font-family:var(--sans); font-weight:600; }
.say { margin:0; white-space:pre-wrap; }
details { border:1px solid var(--line); border-radius:2px; padding:.4rem .6rem; }
summary { cursor:pointer; font-size:.8rem; display:flex; align-items:center;
  gap:.5rem; flex-wrap:wrap; }
.ttitle { color:var(--muted); font-size:.75rem; font-family:var(--mono); }
details pre { margin:.6rem 0 .2rem; padding:.6rem; background:var(--paper);
  border-radius:2px; overflow-x:auto; font-family:var(--mono); font-size:.74rem;
  white-space:pre-wrap; word-break:break-word; max-height:26rem; }
details pre.args { color:var(--accent); }
details pre.err { color:var(--bad); }
.think summary { color:var(--muted); }
'''


# How a run ended, in the harness's own vocabulary. "stop" is missing on purpose:
# it is how every session ends, including a fully successful one -- tool calls to
# do the work, then a final turn of text to wrap up -- so it says nothing about
# whether the result fell short. Only the endings below are actually evidence of
# one: cut off by the context window, or by the step limit while still mid-call.
ENDINGS = {
    "length": "Exhausted the context window mid-run.",
    "tool-calls": "Ended while still calling tools.",
}

# Reasoned about the task -- real text is in opencode's own store -- but never
# once called a tool in the whole session, so nothing was ever written. Checked
# by the session total rather than the final turn alone: a model that acted many
# times and simply narrated on its last turn is a different, unremarkable case,
# not this one.
NEVER_ACTED = ("Reasoned about the task but never called a tool, so nothing "
               "was written.")


def _ending(rec: dict, tier: str) -> str:
    """Why the card's own best attempt fell short, or "" if it has nothing to
    add. An empty final turn is not explained here: it reads as noise on the
    headline card, not as something a reader deciding between models needs
    -- the per-attempt reason is still there for a model with more than one
    attempt, in its own section below.
    """
    # A modified protected file is the one ending that outranks the rest: nothing
    # was graded, so how the run stopped is beside the point.
    tampered = rec.get("tampered")
    if tampered:
        return (f'<p class="ending">Not graded: modified '
                f'{E(", ".join(tampered))}, which the task fixes.</p>')
    reason = (rec.get("finish") or "").strip()
    if tier == "complete":
        return ""
    peak = rec.get("peak_input_tokens")
    if rec.get("final_produced") and not rec.get("tool_calls"):
        detail = NEVER_ACTED
    elif reason == "length" and peak:
        detail = f"Exhausted the context window at {peak:,} input tokens."
    elif reason in ENDINGS:
        detail = ENDINGS[reason]
    else:
        return ""
    return f'<p class="ending">{E(detail)}</p>'


def _attempt_reason(rec: dict, tier: str) -> str:
    """Why one further attempt fell short, as a short clause -- or "" if it
    reached complete and there is nothing to explain."""
    if tier == "complete":
        return ""
    reason = (rec.get("finish") or "").strip()
    peak = rec.get("peak_input_tokens")
    if rec.get("final_produced") is False:
        return "produced nothing at all"
    if rec.get("final_produced") and not rec.get("tool_calls"):
        return "never called a tool"
    if reason == "length" and peak:
        return f"exhausted the context window at {peak:,} input tokens"
    if reason == "tool-calls":
        return "ended while still calling tools"
    return ""


def _consistency_cell(rec: dict) -> str:
    """How many of a model's attempts matched the best one, with every
    attempt's own number, figure and reason on hover.

    The card shows only the best attempt, undecorated by how the others went
    -- that belongs here and in the per-attempt tables, not on the headline
    result. A bare match-count on its own would not say which attempt number
    actually won; the tooltip does, and the per-attempt tables beside this one
    show each attempt in full for a reader who wants more than a count.
    """
    attempts = rec.get("_attempts") or []
    if not attempts:
        return '<td class="figure">&mdash;</td>'
    best_pct = harness.score(rec)
    matched = 1
    lines = []
    for a in sorted([rec] + attempts, key=lambda r: r.get("attempt") or 0):
        pct = harness.score(a)
        tag = " (best)" if a is rec else ", matched" if pct == best_pct else ""
        if a is not rec and pct == best_pct:
            matched += 1
        why = "" if a is rec or tag else _attempt_reason(a, _tier(a))
        lines.append(f"attempt {a.get('attempt') or 1}: {pct:.0f}%{tag}"
                    + (f", {why}" if why else ""))
    figure = f"{matched}/{len(attempts) + 1}"
    return f'<td class="figure" title="{E(chr(10).join(lines))}">{figure}</td>'


def _links(rec: dict) -> str:
    """The two things worth opening from an entry: the work, and the session.

    Named "Files" rather than "Build" or "Project": "build" reads as a compiled
    artifact, which nothing here produces, and "project" overstates what a model
    stopped at the files tier left behind -- a directory of source that never
    became one. "Files" holds at every tier, including that one.

    Offered as a file: URL, which is all a page can do. Browsers open it in
    their own directory view rather than handing it to the desktop file
    manager, and that is still the shortest route to what the model left
    behind. The full path stays in the title attribute rather than the link
    text, which otherwise fills the card.
    """
    out = []
    repo = rec.get("repo") or ""
    if repo:
        out.append(f'<a class="act" href="file://{E(repo)}" title="{E(repo)}">'
                   f'Files</a>')
    if rec.get("_transcript"):
        out.append(f'<a class="act" href="{E(rec["_transcript"])}" '
                   f'title="{rec["_turns"]} turns">Conversation</a>')
    return f'<p class="acts">{"".join(out)}</p>' if out else ""


def _ranked(records: list[dict]) -> list[dict]:
    """By completeness, then by time.

    The tiebreak only ever compares implementations that met the same
    requirements, so a faster one among equals is unambiguously the better
    information for the reader -- there is no way to meet a requirement by
    quitting before attempting it. Shared by the cards and the grid, so a
    model's rank means the same thing in both.
    """
    return sorted(records, key=lambda r: (-harness.met(r)[0],
                                          r.get("wall_s") or 9e9))


def _field(records: list[dict]) -> str:
    """One card per model: what it reached, what it created, and what is missing."""
    if not records:
        return ""
    cards = ""
    for rec in _ranked(records):
        tier = _tier(rec)
        src = _shot(rec)
        alt = f"the interface {E(rec['model'])} rendered"
        picture = (f'<a class="shot" href="{src}" target="_blank" rel="noopener">'
                  f'<img alt="{alt}" src="{src}"></a>' if src else
                  '<div class="shot none"><p>no screenshot available</p></div>')
        cards += f'''<article class="card c-{tier}">
      <header><h3>{E(rec["model"])}</h3>
        <span class="pill p-{tier}">{TIER_LABEL[tier]}</span></header>
      {_figures(rec)}
      {picture}
      {_links(rec)}
      {_ending(rec, tier)}
    </article>'''
    return f'''<section>
    <h2>Field</h2>
    <div class="lede"><p>Ordered by completeness, then by time. Each card shows
    only the best of that model's attempts. How the others went is in the
    Consistency column and the per-attempt tables below, not on the card
    itself.</p></div>
    <div class="cards">{cards}</div>
  </section>'''


def _legend(records: list[dict]) -> str:
    """What the levels mean, and how many models reached each one."""
    counts = {}
    for rec in records:
        counts[_tier(rec)] = counts.get(_tier(rec), 0) + 1
    rows = "".join(
        f'<tr class="r-{k}"><th><span class="pill p-{k}">{label}</span></th>'
        f'<td class="detail">{desc}</td>'
        f'<td class="figure">{counts.get(k, 0)}</td></tr>'
        for k, label, desc in TIERS)
    return f'''<section>
    <h2>Levels</h2>
    <div class="lede"><p>A level marks how far an implementation got, in the order
    the verification imposes.</p></div>
    <div class="scroll"><table><thead><tr>
      <th>Level</th><th>Reached When</th><th class="figure">Models</th>
      </tr></thead><tbody>{rows}</tbody></table></div>
  </section>'''


def _windows(records: list[dict]) -> str:
    """The context window models actually ran at, as a phrase.

    Not something this tool configures: opencode talks to Ollama over the
    OpenAI-compatible endpoint, which has no num_ctx field at all, so every
    model gets whatever the server defaults to, uniformly. Read back per model
    while it was loaded rather than assumed, since a reader on a smaller
    machine needs to know whether these results were measured at a window
    their own setup would never reach.
    """
    windows = sorted({r["context_length"] for r in records if r.get("context_length")})
    if not windows:
        return "the server's own default context"
    if len(windows) == 1:
        return f"a {windows[0]:,}-token context"
    return "{} and {:,}-token contexts".format(
        ", ".join(f"{c:,}" for c in windows[:-1]), windows[-1])


def _all_names(records: list[dict]) -> list[str]:
    """Every check name seen anywhere -- a model's best attempt or any of its
    others -- so every table on the page shares the same columns in the same
    order and a reader can compare across them directly."""
    names = []
    for rec in records:
        for attempt in [rec] + (rec.get("_attempts") or []):
            for check in attempt.get("checks") or []:
                if check["name"] not in names:
                    names.append(check["name"])
    return names


def _grid(records: list[dict], names: list[str], with_consistency: bool = False) -> str:
    """The requirements grid for one set of records -- the best of each model,
    or every model's Nth attempt -- sharing one row-building path so both
    read the same way."""
    if not records:
        return ('<p class="empty">No results recorded yet. Run a model against '
                'the specification and this page will show what it made.</p>')
    # The description rides on the column header, where the reader is when they
    # ask what a requirement means.
    head = "".join(
        f'<th class="rot" title="{E(WHAT.get(n, (n, ""))[1])}">'
        f'<span>{E(WHAT.get(n, (n, ""))[0])}</span></th>' for n in names)
    rows = ""
    for i, rec in enumerate(_ranked(records), 1):
        checks = {c["name"]: c for c in (rec.get("checks") or [])}
        met = sum(harness.check_score(c) for c in checks.values())
        total = len(checks) or 1
        grade = _tier(rec)
        cells = ""
        for name in names:
            check = checks.get(name)
            if check is None:
                cells += '<td class="cell none">&middot;</td>'
            else:
                score = harness.check_score(check)
                state = "y" if score == 1.0 else "n" if score == 0.0 else "w"
                mark = "&check;" if state == "y" else "&times;" if state == "n" else "~"
                cells += (f'<td class="cell {state}" '
                          f'title="{E(check["name"])}: {E(check["detail"] or "")}">'
                          f'{mark}</td>')
        consistency = _consistency_cell(rec) if with_consistency else ""
        rows += (f'<tr class="r-{grade}"><td class="figure">{i}</td>'
                 f'<th>{E(rec["model"])}</th>'
                 f'<td>{_bar(met, total)}</td>'
                 f'{consistency}'
                 f'<td class="figure">{rec.get("wall_s") or 0:.0f}s</td>'
                 f'<td class="figure">{rec.get("turns") or 0}</td>'
                 f'<td class="figure">{rec.get("tool_calls") or 0}</td>'
                 f'<td class="figure">{(rec.get("peak_input_tokens") or 0):,}</td>'
                 f'{cells}</tr>')
    consistency_th = '<th class="figure">Consistency</th>' if with_consistency else ""
    return f'''<div class="scroll"><table><thead><tr>
        <th>#</th><th>Model</th><th>Completeness</th>
        {consistency_th}
        <th class="figure">Time</th>
        <th class="figure">Turns</th><th class="figure">Tool Calls</th>
        <th class="figure">Peak Context</th>
        {head}</tr></thead>
        <tbody>{rows}</tbody></table></div>'''


def _attempt_row(a: dict, names: list[str]) -> str:
    """One row in a model's own attempt table -- the same cell language
    _grid uses for a check, so a mark means the same thing everywhere on
    the page."""
    checks = {c["name"]: c for c in (a.get("checks") or [])}
    met = sum(harness.check_score(c) for c in checks.values())
    total = len(checks) or 1
    grade = _tier(a)
    cells = ""
    for name in names:
        check = checks.get(name)
        if check is None:
            cells += '<td class="cell none">&middot;</td>'
        else:
            score = harness.check_score(check)
            state = "y" if score == 1.0 else "n" if score == 0.0 else "w"
            mark = "&check;" if state == "y" else "&times;" if state == "n" else "~"
            cells += (f'<td class="cell {state}" '
                      f'title="{E(check["name"])}: {E(check["detail"] or "")}">'
                      f'{mark}</td>')
    reason = _attempt_reason(a, grade)
    number = a.get("attempt") or 1
    cell = (f'<a class="path-link" href="{E(a["_transcript"])}">{number}</a>'
           if a.get("_transcript") else str(number))
    return (f'<tr class="r-{grade}"><td class="figure">{cell}</td>'
           f'<td>{_bar(met, total)}</td>'
           f'<td class="figure">{a.get("wall_s") or 0:.0f}s</td>'
           f'<td class="figure">{a.get("turns") or 0}</td>'
           f'<td class="figure">{a.get("tool_calls") or 0}</td>'
           f'<td class="figure">{(a.get("peak_input_tokens") or 0):,}</td>'
           f'<td class="detail">{E(reason)}</td>'
           f'{cells}</tr>')


def _model_sections(records: list[dict], names: list[str]) -> str:
    """One section per model that made more than one attempt, documenting
    that model's own runs in the order it made them.

    Grouped by model rather than by attempt number: a reader following one
    model's trajectory from one attempt to the next is a different question
    from comparing everyone's first try, and the earlier attempt-number
    tables answered neither -- they scattered one model's history across as
    many sections as it had attempts. A model with only one attempt gets no
    section here; there is no history to trace.
    """
    head = "".join(
        f'<th class="rot" title="{E(WHAT.get(n, (n, ""))[1])}">'
        f'<span>{E(WHAT.get(n, (n, ""))[0])}</span></th>' for n in names)
    sections = []
    for rec in _ranked(records):
        attempts = sorted([rec] + (rec.get("_attempts") or []),
                          key=lambda a: a.get("attempt") or 1)
        if len(attempts) <= 1:
            continue
        rows = "".join(_attempt_row(a, names) for a in attempts)
        sections.append(f'''<section>
    <h2>{E(rec["model"])}</h2>
    <div class="lede"><p>Every attempt this model made, in the order it made
    them.</p></div>
    <div class="scroll"><table><thead><tr>
        <th class="figure">Attempt</th><th>Completeness</th>
        <th class="figure">Time</th>
        <th class="figure">Turns</th><th class="figure">Tool Calls</th>
        <th class="figure">Peak Context</th><th>Ending</th>
        {head}</tr></thead>
        <tbody>{rows}</tbody></table></div>
  </section>''')
    return "".join(sections)


def render(records: list[dict], env: dict | None = None) -> str:
    env = env or dict(line="appsift \u00b7 Ollama \u00b7 opencode", host="",
                      remote=False)
    names = _all_names(records)
    gen = dt.datetime.now().strftime("%d %B %Y, %H:%M")
    ctxphrase = _windows(records)
    grid = _grid(records, names, with_consistency=True)

    return f'''<title>Local Agentic Coding Evaluation</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500&display=swap">
<style>{STYLE}</style>
<div class="wrap">
  <header class="top">
    <p class="eyebrow">{E(env["line"])}</p>
    <h1>Local Agentic Coding Evaluation</h1>
    <p class="sub">Each model receives the same specification for a todo
    application: a Python package that keeps a list of tasks, serves it over
    HTTP as JSON, and renders it as a web page, with its own test suite, made
    with an agent harness. Every requirement is verified from outside the
    implementation: the package is imported, the application launched, driven
    over HTTP, terminated, and launched again against the same store. Every
    model ran at {ctxphrase}, read from Ollama directly rather than set by
    this tool.</p>
    <div class="meta">
      <span>generated <b>{gen}</b></span>
    </div>
  </header>

  {_legend(records)}

  {_field(records)}

  <section>
    <h2>Requirements</h2>
    <div class="lede"><p>One column per requirement, named in the column header,
    which carries what the requirement checks. Consistency is how many of a
    model's attempts matched its best one. Hover it for each attempt's own
    figure, or see every attempt in full below.</p></div>
    {grid}
  </section>

  {_model_sections(records, names)}

</div>'''


def write(cfg: Config, output: Path, models: list[str] | None = None) -> Path:
    from .cli import LEDGER
    records = _records(cfg, LEDGER, list(models or []))
    env = _env(cfg)
    env["report_name"] = output.name
    _write_transcripts(records, output, env)
    output.write_text(render(records, env), encoding="utf-8")
    return output


def _write_transcripts(records: list[dict], output: Path, env: dict) -> None:
    """One page per conversation, beside the report -- a model's best attempt
    and every one of its other attempts alike, so a run's own number in its
    attempt-history section can link to the same page the card's Conversation
    button reaches, not only the best attempt's.

    Kept out of the report itself because a session runs to tens of thousands of
    words and the page it would be embedded in is meant to be skimmed. A result
    whose session cannot be found is left without a link rather than given a dead
    one -- opencode may have been pruned, or the store may be on another machine.
    """
    conn = sessions._connect()
    if conn is None:
        return
    folder = output.parent / (output.stem + "_sessions")
    try:
        for rec in records:
            for attempt in [rec] + (rec.get("_attempts") or []):
                _write_one_transcript(attempt, attempt is rec, folder, env, conn)
    finally:
        conn.close()


def _write_one_transcript(rec: dict, primary: bool, folder: Path, env: dict, conn) -> None:
    sid = sessions.resolve(rec, conn)
    if sid and rec.get("gen_s") is None:
        rec.update({k: v for k, v in sessions.generation(sid, conn).items() if v})
    if sid and rec.get("final_produced") is None:
        voice = sessions.spoke(sid, conn)
        rec["final_produced"] = voice["produced"]
        rec["final_acted"] = voice["acted"]
        rec.setdefault("said", voice["said"])
    turns = sessions.transcript(sid, conn) if sid else []
    if not turns:
        return
    folder.mkdir(parents=True, exist_ok=True)
    # The best attempt keeps the plain, model-only filename: nothing that already
    # links to it needs to change. Every other attempt gets its own number, so it
    # does not collide with the best attempt's page or with each other.
    stem = _slug(rec["model"])
    if not primary:
        stem += f"_attempt{rec.get('attempt') or 1}"
    name = stem + ".html"
    rec["_session"], rec["_turns"] = sid, len(turns)
    (folder / name).write_text(_transcript_page(rec, turns, env), encoding="utf-8")
    rec["_transcript"] = f"{folder.name}/{name}"


def main(argv: list[str] | None = None) -> int:
    """Render the report from whatever the ledger already holds, without
    running anything -- a maintenance operation on this tool's own output,
    reached as `python -m appsift.report` rather than through the primary
    `appsift` command."""
    p = argparse.ArgumentParser(
        prog="appsift.report",
        description="Render the HTML report from results already on disk.")
    p.add_argument("--results-dir", metavar="DIR",
                   help=f"where the ledger and applications are kept (default: "
                        f"{data_dir(DEFAULT_OUTPUT)})")
    p.add_argument("--models", nargs="+", metavar="MODEL",
                   help="only these models (default: every model in the ledger)")
    p.add_argument("--models-file", metavar="PATH",
                   help="file listing one model per line; # comments allowed")
    p.add_argument("-o", "--output", default=DEFAULT_OUTPUT, metavar="PATH",
                   help="where to write the report (default: %(default)s)")
    args = p.parse_args(argv)

    models = args.models or (read_model_file(args.models_file)
                             if args.models_file else [])
    output = Path(args.output)
    results_dir = Path(args.results_dir) if args.results_dir else data_dir(output)
    cfg = Config(results_dir=results_dir, models=models)
    print(f"wrote {write(cfg, output, models)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
