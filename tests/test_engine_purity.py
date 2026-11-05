"""Task 8: the engine purity ratchet.

`separate.py`, `backends.py` and every `lib_v5/**/*.py`, `demucs/**/*.py` file
must be importable with the full `GUI_PACKAGES` block set unavailable -- but read that
sentence against the split below, because it spans both guards: guard 1 imports the
**16** modules named in `ENGINE_IMPORTS` and nothing else, and the
**38-file** "every file" claim belongs to guard 2, which *lints* all of them and
imports none. Note the scope:
that is now a *module-level GUI*-freedom claim, not merely a tkinter one: guard 1
blocks the tkinter four *plus* `PySide6`/`fluentqt`/`app`, so it does prove the 16
named imports reach no GUI at import time. What it still cannot see is a GUI import
deferred into a function body -- a case `app` in the block set makes live, because
`separate_cli.run()` pulls `app.session` in from inside its body -- and a later
task's test is what closes it, not this guard. Two guards, split deliberately, and
each one is
blind to exactly what the other sees. The names after `plan` -- `plan` (15th, Task
11b) and `separate_cli` (16th, Task 13) -- are composition layers rather than part of
`engine`, and they joined on purpose: Task 13's GUI-free CLI is the thing that must
import clean with the GUI blocked, and guard 1 is the only place that proves it for
the whole call path and not only for the modules under `LINT_GLOBS`.

  1. `test_engine_imports_with_tkinter_unavailable` proves the *import graph*     behaves, which is the only way to see a coupling that arrives transitively
     through a wildcard import (`separate.py:18` does
     `from gui_data.constants import *`, and no regex over that file can see
     what it drags in). What it cannot see is a GUI import that never executes
     at import time: a deferred `import tkinter` inside a function body, or a
     `try: import tkinter; except ImportError: pass` that swallows the blocker
     and keeps running.
  2. `test_no_global_tkinter_root_reference` walks the engine's ASTs for
     attribute access on the GUI's global `root` widget -- the one thing guard 1
     is structurally blind to, because such code need never run at import. What
     it cannot see is anything else about the GUI: its violation class is
     exactly `root.<attr>`, so a hidden `import tkinter`, or a root reached
     through another name (`self.master`, a module-level alias), is out of its
     reach. It matches on the *name* `root`, not on a scope-resolution proof, so
     a legitimate parameter called `root` used as `root.something` fires it --
     the answer is to rename it to the upstream `self.root` style, not to add an
     exemption.

Both "what it cannot see" clauses are load-bearing, not hedging: a ratchet whose
docstring implies more than it delivers invites the edit that defeats it. The
red runs behind every claim here are committed as `tools/task8_red_controls.py`
(run it explicitly; it is not collected by bare `pytest`). A third test,
`test_the_two_lists_have_not_shrunk`, is the same defense one level up: both
guards and both controls take `GUI_PACKAGES` and `ENGINE_IMPORTS` as *input*, so
neither can see one of them shrink, and the pin checks the two lists against
frozen literals instead.
"""
import ast, pathlib, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Engine modules this fork must be able to import with the tkinter GUI packages
# blocked (see the module docstring: blocked != absent, and != GUI-free).
# Verified by running the blocker against each one at HEAD 4463a80. The RED set is
# separate, lib_v5.apollo_inference, demucs.{apply,utils,repo,pretrained,htdemucs}
# -- all seven trace to the SAME two dead imports -- and the already-green set is
# backends, lib_v5.{mdxnet,bs_roformer,mel_band_roformer,attend,results} and
# demucs.states. Do not trim this list to the modules that currently fail: the
# point is that deleting the two lines turns ALL of them green at once.
# (An earlier draft of this list named `lib_v5.roformer`, which does not exist --
# there is no such module. The real ones are bs_roformer and mel_band_roformer.)
# The per-module form of this table -- one child process per module, so a red run
# names every failing module instead of only the first -- is the committed control
# `tools/task8_red_controls.py::test_red_table_names_every_engine_module`; today's
# column is 16 GREEN since Task 13 (15 since Task 11b, 14 before it), and the 7-RED
# column lives in
# `git show 3e612ea` (the two
# deletions), which the controls re-create the failure signature of by planting.
# Membership is pinned by `PINNED_ENGINE_IMPORTS` below: "do not trim" in prose is
# not enforceable, and 15 of these 16 names can be deleted with both guards
# passing.
# Task 11b added `plan`, widening guard 1 from 14 to 15, and Task 13 added
# `separate_cli`, widening it to 16: `plan.py` and `separate_cli.py` are the
# composition layers between `app` and `engine`, Task 13's GUI-free CLI is the thing a
# user runs without a GUI, and the ratchet is the only thing that proves they import
# clean with tkinter, `gui_data.tkinterdnd2`, `gui_data.sv_ttk` and `PIL.ImageTk`
# blocked. They import `settings`, `catalog` and `backends`, which is legal for them
# and illegal for everything under `LINT_GLOBS`; that glob stays a statement about the
# engine, so neither file is added to it.
ENGINE_IMPORTS = ["separate", "backends", "plan", "separate_cli", "lib_v5.mdxnet",
                  "lib_v5.bs_roformer",
                  "lib_v5.mel_band_roformer", "lib_v5.attend", "lib_v5.results",
                  "lib_v5.apollo_inference", "demucs.apply", "demucs.utils",
                  "demucs.repo", "demucs.pretrained", "demucs.states",
                  "demucs.htdemucs"]


# The block set: what the child's finder refuses. Pinned by `PINNED_GUI_PACKAGES`
# for the same reason -- the controls loop over this tuple, so a narrower tuple
# audits less and still passes. Task 1 added `PySide6`/`fluentqt`/`app` to the four
# tkinter names, so guard 1 blocks the whole GUI stack, not tkinter alone (see the
# module docstring).
GUI_PACKAGES = ("tkinter", "gui_data.tkinterdnd2", "gui_data.sv_ttk", "PIL.ImageTk",
                "PySide6", "fluentqt", "app")

# Fix round 2, Item 1: frozen second copies of both name lists, asserted against
# them by `test_the_two_lists_have_not_shrunk`. They are *separate literals* on
# purpose: every other check on these two lists derives its expectation from the
# list itself (`guard_script` builds BLOCKED from `GUI_PACKAGES`; the controls
# loop over both), so a trim shrinks the audit with the audit's input. Both
# halves reproduced on this machine at `71683e9`: `GUI_PACKAGES` narrowed to two
# entries plus `ENGINE_IMPORTS` narrowed to `["separate"]` leaves this file's two
# guards PASSED (2 passed, RC=0) and every control green -- the controller's
# wider subsets measured 5 and 4 passed for the same two edits. Emptying either
# list *is* caught (the child's `blocker ineffective` self-assert), which is why
# this is exact set equality and not non-emptiness. Reordering must not trip it;
# a name arriving or leaving must, in either direction, so the pin is `==` and
# not `<=`.
# This is not the plan's forbidden "pin the count to 38": that ruling covers
# guard 2's *glob*, whose count stays printed-only because a vendored file
# appearing must not break the suite for the wrong reason. These are hand-named
# lists, and a name leaving one is exactly the change that must be deliberate.
PINNED_GUI_PACKAGES = frozenset({"tkinter", "gui_data.tkinterdnd2", "gui_data.sv_ttk",
                                 "PIL.ImageTk", "PySide6", "fluentqt", "app"})

PINNED_ENGINE_IMPORTS = frozenset({
    "separate", "backends", "plan", "separate_cli", "lib_v5.mdxnet",
    "lib_v5.bs_roformer",
    "lib_v5.mel_band_roformer", "lib_v5.attend", "lib_v5.results",
    "lib_v5.apollo_inference", "demucs.apply", "demucs.utils", "demucs.repo",
    "demucs.pretrained", "demucs.states", "demucs.htdemucs"})


# Guard 2's whole sweep, as a glob, so it follows the engine as it grows instead
# of drifting behind a hand-named list. This is the set the plan's constraint
# speaks of. A wrong pattern here does NOT shrink the printed total to zero, so a
# single `assert files` cannot see it: measured on this repo, `lib_v6/**` in place
# of `lib_v5/**` still swept 19 files and stayed green, and `LINT_GLOBS =
# ("separate.py",)` swept 1 file and stayed green -- the sweep silently narrowed to
# a fraction of the engine. What keeps a wrong glob from reading as a pass is the
# per-pattern assertion in `test_no_global_tkinter_root_reference`, which fails on
# any single pattern that stops matching. (The *total* is printed, never asserted:
# pinning it to 38 today would break the suite the day a vendored file is added,
# for the wrong reason.)
LINT_GLOBS = ("separate.py", "backends.py", "lib_v5/**/*.py", "demucs/**/*.py")

#: Hand-written copy of `LINT_GLOBS`, for the hole the per-pattern loop above cannot
#: see: *deleting* a pattern leaves every surviving pattern matching, so the sweep
#: silently narrows (measured -- `LINT_GLOBS = ("separate.py",)` swept one file and
#: the test stayed green, leaving `backends.py`, `lib_v5/**` and `demucs/**` unlinted).
#: Same reason `PINNED_ENGINE_IMPORTS` exists: the guard's own inputs must not be
#: editable in the same commit as the guard's satisfaction. Patterns are pinned, not
#: the file total -- a vendored file must be addable without breaking this.
PINNED_LINT_GLOBS = ("separate.py", "backends.py", "lib_v5/**/*.py", "demucs/**/*.py")

# The child imports torch, so the run has a real cost: measured ~5.2s for the 14
# modules on this machine. Hundreds of seconds is headroom for a cold start that
# still bounds a hang -- without a bound, a future blocking import (`demucs
# .pretrained` touching the network, MIOpen init) hangs the suite forever instead
# of failing.
CHILD_TIMEOUT_SECONDS = 600


def engine_lint_files(root=ROOT):
    """The files guard 2 sweeps, resolved from LINT_GLOBS."""
    found = {p for pattern in LINT_GLOBS for p in root.glob(pattern) if p.is_file()}
    return sorted(found)


def global_root_hits(files):
    """Walk each file's AST and return (hits, unparseable) for `root.<attr>`.

    This reads the AST instead of matching a regex, and it is the whole point of
    this round: `Attribute(value=Name(id="root"))` is the violation, so
      * docstrings and comments are not candidates at all -- a string that quotes
        `root.` is a Constant node and a comment is not in the tree, which is why
        the old `line.strip().startswith("#")` branch and the `(?<![.\\w])`
        negative lookbehind are both gone;
      * `self.root.iterdir()` stays excluded as a *nested* Attribute, whose
        `.value` is an Attribute, not a Name;
      * `root .foo()` (whitespace before the attribute dot is legal Python) and a
        call split across a line break are caught, which neither the old nor the
        narrowed regex could see.
    """
    hits, unparseable = [], []
    for f in files:
        try:
            src = f.read_text(encoding="utf-8")
            tree = ast.parse(src)
        except (SyntaxError, ValueError, OSError) as exc:
            unparseable.append(f"{f}: {type(exc).__name__}: {exc}")
            continue
        lines = src.splitlines()
        for node in ast.walk(tree):
            if (isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "root"):
                where = f"{f}:{node.lineno}:{node.col_offset + 1}"
                text = lines[node.lineno - 1].strip() if node.lineno <= len(lines) else ""
                hits.append(f"{where}: root.{node.attr} -- {text}")
    return hits, unparseable


def guard_script(modules):
    """The child interpreter's source: install the finder, prove it works, import.

    The self-assert is the first thing after the finder goes in, because
    `assert r.returncode == 0` is loud about engine couplings and silent about
    this test's own mechanism: delete the `sys.meta_path.insert` line, or empty
    `BLOCKED`, and guard 1 becomes the plain import smoke test that already lives
    at `tests/test_engine_imports.py:14-16` -- invisible in the pass count. The
    purged `sys.modules` loop is above it so the probe really re-triggers the
    finder instead of hitting a cached `tkinter`. `tools/task8_red_controls.py`
    builds its red runs out of this exact string.
    """
    return (
        "import sys\n"
        "BLOCKED = " + repr(list(GUI_PACKAGES)) + "\n"
        "class Block:\n"
        "    def find_module(self, name, path=None): return None\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        # Match the full dotted name, never name.split('.')[0]:\n"
        "        # `gui_data` is a namespace package with no __init__.py, and\n"
        "        # blocking it wholesale would wrongly forbid the pure-data\n"
        "        # `gui_data.constants` that separate.py:18 legitimately needs.\n"
        "        for pkg in BLOCKED:\n"
        "            if name == pkg or name.startswith(pkg + '.'):\n"
        "                raise ImportError('GUI package ' + name + ' blocked by test_engine_purity')\n"
        "        return None\n"
        "sys.meta_path.insert(0, Block())\n"
        "for m in list(sys.modules):\n"
        "    if any(m == p or m.startswith(p + '.') for p in BLOCKED):\n"
        "        del sys.modules[m]\n"
        "try:\n"
        "    import tkinter\n"
        "except ImportError:\n"
        "    pass\n"
        "else:\n"
        "    raise SystemExit('blocker ineffective: tkinter imported while the "
        "finder was installed, so this guard is no longer blocking anything -- "
        "it has silently become an import smoke test')\n"
        + "".join(f"import {m}\n" for m in modules)
    )

def test_engine_imports_with_tkinter_unavailable():
    """Run in a fresh subprocess: a module already in sys.modules would not
    re-trigger the finder, so an in-process test can pass by accident."""
    script = guard_script(ENGINE_IMPORTS)
    try:
        r = subprocess.run([sys.executable, "-c", script], cwd=ROOT,
                           capture_output=True, text=True,
                           timeout=CHILD_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as exc:
        raise AssertionError(
            f"TIMEOUT: the engine-import child did not exit within its "
            f"CHILD_TIMEOUT_SECONDS={CHILD_TIMEOUT_SECONDS} bound, so something "
            f"in the import graph is blocking (network, MIOpen/GPU init, a lock) "
            f"-- this is a hang, not a purity failure. Partial output: "
            f"stdout={exc.stdout!r} stderr={exc.stderr!r}"
        ) from exc
    assert r.returncode == 0, (
        "the engine cannot be imported without a GUI (unless the child's stderr "
        "says 'blocker ineffective', in which case the blocker itself stopped "
        "working and this test stopped testing anything):\n" + r.stderr)

def test_no_global_tkinter_root_reference():
    files = engine_lint_files()
    assert files, "lint swept nothing"
    assert frozenset(LINT_GLOBS) == frozenset(PINNED_LINT_GLOBS), (
        "guard 2's own pattern list has drifted from its pinned copy: "
        f"live={sorted(LINT_GLOBS)} pinned={sorted(PINNED_LINT_GLOBS)}. Dropping a "
        "pattern here drops a whole engine tree from the lint without changing any "
        "other signal in this file. Add the pattern back, or change both lists in "
        "the same commit and say why.")
    # Per pattern, not per sweep: one dead pattern hides inside a total that still
    # looks healthy (see the LINT_GLOBS comment for the two measured examples).
    for pattern in LINT_GLOBS:
        assert list(ROOT.glob(pattern)), (
            f"guard 2's pattern {pattern!r} matched no file, so the engine half it "
            "was meant to lint is no longer linted at all -- the sweep total can "
            "stay large while this goes unexamined. Point the pattern at the tree "
            "it is meant to cover, or delete it in the same commit and say why.")
    hits, unparseable = global_root_hits(files)
    print(f"[purity] guard 2 swept {len(files)} engine files: "
          f"{len(hits)} global-`root.` attribute hits, "
          f"{len(unparseable)} files the parser rejected", flush=True)
    assert not unparseable, (
        "guard 2 cannot lint a file it cannot parse, so a parse failure is a "
        "coverage hole and must not read as a pass:\n" + "\n".join(unparseable))
    assert not hits, "engine references the GUI root:\n" + "\n".join(hits)


def pin_drift(list_name, live, pinned, pin_name):
    """A pin failure text a human can act on: what left, what arrived, what to do.

    A self-referential `frozenset(GUI_PACKAGES) == frozenset(GUI_PACKAGES)` is
    what this replaces -- it can never fail, so it proves nothing. Here the two
    sides come from different places in the file, so the message is allowed to
    catch them disagreeing, and to say what the reader should do next.
    """
    now = frozenset(live)
    missing = sorted(pinned - now)
    extra = sorted(now - pinned)
    return (
        f"{list_name} has drifted from its pinned copy `{pin_name}`.\n"
        f"  MISSING from {list_name} (dropped -- and every red control that loops "
        f"over {list_name} lost its coverage of this name with it): {missing}\n"
        f"  NOT IN the pin (newly added, so nothing here proves it belongs): "
        f"{extra}\n"
        f"  pinned = {sorted(pinned)}\n"
        f"  live   = {sorted(now)}\n"
        f"  Do one of two things: put the name back, or -- if the change is "
        f"deliberate -- edit {pin_name} in this same file in the SAME commit and "
        f"say why in the commit message. Do not delete the pin, do not weaken it "
        f"to a non-emptiness or subset check, and do not rebuild it from "
        f"{list_name}: that is the circular self-audit this test exists to "
        f"remove.")


def test_the_two_lists_have_not_shrunk():
    """The pin (Item 1): both name lists still hold exactly their pinned names.

    Guard 1 and the controls take these two lists as *input*, so neither can
    report a shrink: narrow `GUI_PACKAGES` and `guard_script` narrows the blocker
    in step; trim `ENGINE_IMPORTS` and the per-module table prints all-GREEN rows
    for the shorter list. Both were reproduced at `71683e9` (4 -> 2 and 14 -> 1
    entries, guards green), which is why the pin lives here: this file is collected
    by the suite, `tools/task8_red_controls.py` deliberately is not.
    """
    assert frozenset(GUI_PACKAGES) == PINNED_GUI_PACKAGES, pin_drift(
        "GUI_PACKAGES", GUI_PACKAGES, PINNED_GUI_PACKAGES, "PINNED_GUI_PACKAGES")
    assert len(GUI_PACKAGES) == len(PINNED_GUI_PACKAGES), (
        "a duplicate name crept into GUI_PACKAGES: the set still agrees with "
        "PINNED_GUI_PACKAGES but the tuple grew -- duplicates hide from set "
        "equality and would be emitted twice into the child's BLOCKED list. "
        "Delete the repeat, or update the pin in the same commit and say why.")
    assert frozenset(ENGINE_IMPORTS) == PINNED_ENGINE_IMPORTS, pin_drift(
        "ENGINE_IMPORTS", ENGINE_IMPORTS, PINNED_ENGINE_IMPORTS,
        "PINNED_ENGINE_IMPORTS")
    assert len(ENGINE_IMPORTS) == len(PINNED_ENGINE_IMPORTS), (
        "a duplicate name crept into ENGINE_IMPORTS: the set still agrees with "
        "PINNED_ENGINE_IMPORTS but the list grew, so guard 1 imports the repeat "
        "and the per-module table spawns a child per repeat. Delete the repeat, "
        "or update the pin in the same commit and say why.")
