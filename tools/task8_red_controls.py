"""Task 8 RED controls: proof that the guards in ``tests/test_engine_purity.py`` fire.

WHAT THIS FILE PROVES
---------------------
A ratchet is only worth its red run, and Task 8's review found both halves of
that missing: the blocker's own teeth were never asserted (delete
``sys.meta_path.insert(0, Block())`` or empty ``BLOCKED`` and guard 1 quietly
becomes the import smoke test that already lives at
``tests/test_engine_imports.py:14-16`` -- so the regression is invisible in the
pass count), and the 7-RED -> 14-GREEN per-module table behind the two deleted
``import tkinter as tk`` lines came from a scratch probe that no longer exists
and can no longer be re-run, because those two lines are gone.

Each control sabotages one mechanism and asserts the committed guard fires:

  * a control that PASSES means the guard still catches its failure mode;
  * a control that goes RED means a guard has become decorative -- which is the
    finding.

Several controls also run the *undeleted* case next to the sabotaged one, so a
pass cannot come from "every child fails for an unrelated reason". Nothing here
mutates the repository: the planted modules are written into ``tmp_path`` and the
committed test is reached by monkeypatching the file list (``t8.engine_lint_files``)
and the timeout bound (``t8.CHILD_TIMEOUT_SECONDS``), the same ``monkeypatch`` /
``pytest.raises`` style as ``tools/task7_red_controls.py`` -- which this file
mirrors in shape and deliberately does not import. The one control that has to
attack a *constant* rather than a file
(``test_red_the_pin_fires_when_a_name_list_shrinks``) reads
``tests/test_engine_purity.py`` and execs a patched copy of it in memory; the
tracked file is never rewritten, and the unparseable plant
(``test_red_an_unparseable_file_goes_red_not_a_pass``) lives only in ``tmp_path``,
never in the engine tree.

The per-child bound these controls run under is the committed one, read live from
``t8.CHILD_TIMEOUT_SECONDS`` (see the ``CHILD_TIMEOUT`` note below) -- there is no
second copy of the number here to drift from it.

WHAT THE "BEFORE" COLUMN IS NOW
-------------------------------
Control ``test_red_table_names_every_engine_module`` prints today's column: one
child process per ``ENGINE_IMPORTS`` entry, each running a single strict import
under the real blocker, so a red run names *every* failing module rather than
only the first. All 14 are GREEN now. The 7 RED of the "before" state cannot be
regenerated from this working tree -- the two lines were deleted in ``3e612ea``
(``git show 3e612ea -- demucs/apply.py demucs/utils.py`` is that record) -- so the
controls re-create the same failure signature by *planting* it instead
(``test_red_table_names_every_failing_module_not_just_the_first``), including the
transitive form that made the original red run worth having.

NON-COLLECTED BY DESIGN
-----------------------
The file is named ``task8_red_controls.py``, not ``test_*.py``: a bare ``pytest``
run must never execute it (there is no ``pytest.ini``/``pyproject.toml`` here, so
the filename is the whole guard). Run it explicitly:

    .venv/Scripts/python.exe -m pytest tools/task8_red_controls.py -v -s
"""
from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import tests.test_engine_purity as t8

REPO = Path(__file__).resolve().parents[1]
BLOCK_MARK = "blocked by test_engine_purity"
SELF_ASSERT_MARK = "blocker ineffective"
GUI_ROOT_MARK = "engine references the GUI root"
PARSE_FAIL_MARK = "cannot lint a file it cannot parse"
DUP_MARK = "duplicate name crept into"
PURITY_FILE = REPO / "tests" / "test_engine_purity.py"

# The pre-fix lint, rebuilt here only so the controls can show what the `ast`
# migration bought. It is NOT in the committed test any more -- that is the point.
OLD_REGEX = re.compile(r"(?<![.\w])root\.")

# Item 3 (fix round 2): the controls' children run under the *committed* bound,
# not under a second copy of it. `CHILD_TIMEOUT = 600` was exactly such a copy --
# it never forked in value, but it could not follow `t8.CHILD_TIMEOUT_SECONDS` if
# that ever changed, so the worst case here (27 long-lived children today: one per
# `ENGINE_IMPORTS` entry, one per `GUI_PACKAGES` entry, plus the plant/armed/premise
# probes) was set by a number nothing audited. The alias keeps the tie visible and
# assertable; the live read below is what `run_child` actually uses, because a value
# captured *at import time* would un-plug the monkeypatch in
# `test_red_a_child_that_outlasts_the_bound_reads_as_a_timeout` -- that control
# proves the bound by *narrowing* `t8.CHILD_TIMEOUT_SECONDS`, and a snapshot taken
# before the patch would leave these children on the old, wider bound. That would
# be a new hole created by the very fix meant to close one, so it is asserted.
CHILD_TIMEOUT = t8.CHILD_TIMEOUT_SECONDS


def child_timeout() -> float:
    """The committed per-child bound, read at call time so a patch lands.

    The children import torch, so a run costs real seconds and the bound is
    headroom for a cold start that still catches a hang -- with exactly one
    number behind it, the one in `tests/test_engine_purity.py`.
    """
    return t8.CHILD_TIMEOUT_SECONDS


def run_child(script: str, cwd: Path = REPO) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", script], cwd=cwd,
                          capture_output=True, text=True,
                          timeout=child_timeout())


def child_with_load_path(script_tail: str, tmp: Path) -> str:
    """Prepend a sys.path entry to a guard script, without touching the guard."""
    return (f"import sys\nsys.path.insert(0, {str(tmp)!r})\n" + script_tail)


def last_line(text: str) -> str:
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else "<no output>"


# --------------------------------------------------------------------------- #
# Premise: the self-assert can only mean something on a box that has Tk
# --------------------------------------------------------------------------- #

def test_red_premise_tkinter_really_is_importable_on_this_box():
    """Without this, controls about "the blocker has teeth" are unfalsifiable.

    If ``tkinter`` could not be imported here, the child's self-assert would pass
    for the wrong reason (an absent Tk looks exactly like a blocked Tk) and every
    control below that expects an ImportError would be measuring the platform,
    not the finder.
    """
    r = run_child("import tkinter\nprint('tkinter importable')")
    assert r.returncode == 0, (
        "this box cannot import tkinter at all, so `blocker ineffective` can "
        "never trip and the self-assert cannot be proven here -- run these "
        "controls on a machine with Tk. stderr: " + r.stderr)
    print(f"[red] premise holds: {r.stdout.strip()}", flush=True)


# --------------------------------------------------------------------------- #
# Control (a): the blocker's reach -- GUI yes, pure data no
# --------------------------------------------------------------------------- #

def test_red_blocker_fires_on_each_gui_package():
    """Each name in GUI_PACKAGES is refused, by name, by the child's finder."""
    wrong = []
    for pkg in t8.GUI_PACKAGES:
        r = run_child(t8.guard_script([pkg]))
        ok = (r.returncode != 0 and f"GUI package {pkg} blocked" in r.stderr
              and BLOCK_MARK in r.stderr)
        print(f"[red] import {pkg:<22} -> rc={r.returncode} "
              f"{'BLOCKED' if ok else 'IMPORTED (!)'} | {last_line(r.stderr)}",
              flush=True)
        if not ok:
            wrong.append(f"{pkg}: rc={r.returncode} stderr={r.stderr.strip()}")
    assert not wrong, "the blocker let a GUI package through:\n" + "\n".join(wrong)


def test_red_blocker_leaves_the_pure_data_gui_modules_alone():
    """`gui_data.constants` is not a GUI coupling, so guard 1 must not forbid it.

    The finder matches full dotted names, never ``name.split('.')[0]``. Blocking
    the ``gui_data`` namespace wholesale would red-light `separate.py:18-19` for
    the wrong reason -- and this control goes red the day someone adds it.
    """
    assert "gui_data" not in t8.GUI_PACKAGES, (
        "gui_data itself is now in GUI_PACKAGES; separate.py:18 could not import "
        "and this control's premise (dotted-name matching) is gone")
    r = run_child(t8.guard_script(["gui_data.constants", "gui_data.error_handling"]))
    print(f"[red] import gui_data.constants + error_handling -> rc={r.returncode} "
          f"| {last_line(r.stderr)}", flush=True)
    assert r.returncode == 0, (
        "the pure-data gui_data modules are being blocked, which makes guard 1 "
        "wrong rather than strict:\n" + r.stderr)


# --------------------------------------------------------------------------- #
# Control (b): the Item 3 self-assert, tripped two ways
# --------------------------------------------------------------------------- #

def test_red_self_assert_trips_when_the_finder_is_never_installed():
    """Delete `sys.meta_path.insert(0, Block())` and guard 1 says so out loud."""
    script = t8.guard_script(["separate"])
    disarmed = script.replace("sys.meta_path.insert(0, Block())\n", "")
    assert disarmed != script, (
        "guard_script() no longer installs the finder with this exact line, so "
        "the sabotage this control performs is no longer the disarm it claims to "
        "prove -- read tests/test_engine_purity.py::guard_script and fix the "
        "control instead of letting it pass vacuously")
    r = run_child(disarmed)
    print(f"[red] finder removed -> rc={r.returncode} | {last_line(r.stderr)}",
          flush=True)
    assert r.returncode != 0, (
        "a child with no finder still exited 0, which means the self-assert was "
        "never reached -- tkinter must have failed to import for platform "
        "reasons (see the premise control)")
    assert SELF_ASSERT_MARK in r.stderr, (
        f"the child did not name its own broken mechanism; expected "
        f"{SELF_ASSERT_MARK!r} in stderr, got:\n{r.stderr}")


def test_red_self_assert_trips_when_blocked_is_emptied():
    """The second disarm the review named: finder installed, list renamed/emptied.

    ``Block()`` stays on ``sys.meta_path`` with nothing to block, so it is
    structurally present and functionally inert -- precisely the state a rename
    of ``BLOCKED`` produces.
    """
    script = t8.guard_script(["separate"])
    armed = "BLOCKED = " + repr(list(t8.GUI_PACKAGES)) + "\n"
    disarmed = script.replace(armed, "BLOCKED = []\n")
    assert disarmed != script, (
        f"guard_script() no longer emits {armed!r}; this control's sabotage does "
        "not match the real mechanism any more")
    r = run_child(disarmed)
    print(f"[red] BLOCKED emptied -> rc={r.returncode} | {last_line(r.stderr)}",
          flush=True)
    assert SELF_ASSERT_MARK in r.stderr, (
        f"an inert blocker was not caught; expected {SELF_ASSERT_MARK!r}, "
        f"got rc={r.returncode}:\n{r.stderr}")


def test_red_the_armed_child_does_not_trip_the_self_assert():
    """The positive half: with the real script the self-assert stays quiet.

    Without this, the two sabotage controls above would pass just as happily if
    every child failed for an unrelated reason.
    """
    r = run_child(t8.guard_script(["backends"]))
    print(f"[red] armed child, import backends -> rc={r.returncode}", flush=True)
    assert r.returncode == 0, (
        "the armed blocker broke a GUI-free module, so nothing below this line "
        "can be trusted:\n" + r.stderr)
    assert SELF_ASSERT_MARK not in r.stderr, r.stderr


# --------------------------------------------------------------------------- #
# The permanent per-module table (Item 4's reason for existing)
# --------------------------------------------------------------------------- #

def module_verdict(module: str, tmp: Path | None = None) -> tuple[str, str]:
    """One child, one strict import, under the committed blocker."""
    script = (t8.guard_script([module]))
    if tmp is not None:
        script = child_with_load_path(script, tmp)
    r = run_child(script)
    return ("GREEN" if r.returncode == 0 else "RED"), (
        r.stdout.strip() or r.stderr.strip())


def test_red_table_names_every_engine_module():
    """ENGINE_IMPORTS, one child process per module: the committed diagnostics.

    The committed test imports all 14 in one child, so a red run names only the
    first blocked edge; this is the per-module form, so it names them all. Today
    every row is GREEN -- the "before" column is the two deletions in `3e612ea`
    and its signature is re-created by the next control.
    """
    rows, red = [], []
    for module in t8.ENGINE_IMPORTS:
        verdict, detail = module_verdict(module)
        rows.append(f"  {verdict:<5} {module:<28} "
                    f"{'imports clean' if verdict == 'GREEN' else last_line(detail)}")
        if verdict == "RED":
            red.append(module)
        print(f"[red] table row: {module} -> {verdict}", flush=True)
    print("\n[red] ENGINE_IMPORTS per-module verdicts under the blocker:\n"
          + "\n".join(rows), flush=True)
    assert not red, (
        f"{len(red)} of {len(t8.ENGINE_IMPORTS)} engine modules cannot be "
        f"imported with the GUI blocked: {red}\n" + "\n".join(rows))


def test_red_table_names_every_failing_module_not_just_the_first(tmp_path):
    """The table's non-vacuity: two planted reds and one planted green, all named.

    This is the failure mode of the one-child form: `separate.py:3` ->
    `demucs/apply.py:19` aborted the child at the first blocked import, so the
    original 7-RED picture needed a scratch probe. Here the same machinery has to
    report BOTH planted violations, and must not report the clean one.
    """
    (tmp_path / "plant_direct.py").write_text("import tkinter as tk\n",
                                              encoding="utf-8")
    (tmp_path / "plant_mid.py").write_text("import plant_leaf\n", encoding="utf-8")
    (tmp_path / "plant_leaf.py").write_text(
        "import gui_data.tkinterdnd2\n", encoding="utf-8")
    (tmp_path / "plant_clean.py").write_text("import json\nJSON = json\n",
                                             encoding="utf-8")

    verdicts = {}
    details = {}
    for name in ("plant_direct", "plant_mid", "plant_clean"):
        verdicts[name], details[name] = module_verdict(name, tmp=tmp_path)
        print(f"[red] planted {name:<12} -> {verdicts[name]} | "
              f"{last_line(details[name])}", flush=True)

    assert verdicts["plant_direct"] == "RED", (
        "a re-added `import tkinter as tk` slipped past the blocker, so guard 1 "
        "is decorative: " + str(verdicts))
    assert verdicts["plant_mid"] == "RED", (
        "a transitive GUI coupling (plant_mid -> plant_leaf -> "
        "gui_data.tkinterdnd2) was not caught, which is the whole reason guard 1 "
        "runs in a subprocess: " + str(verdicts))
    # Both reds must be red *because of the blocker*: a ModuleNotFoundError from
    # the scratch path would satisfy `RED` for the wrong reason.
    for name in ("plant_direct", "plant_mid"):
        assert BLOCK_MARK in details[name], (
            f"{name} failed, but not on the blocker: {details[name]}")
    assert verdicts["plant_clean"] == "GREEN", (
        "the blocker is refusing a module that touches no GUI package; the "
        "table's GREEN rows would be meaningless: " + str(verdicts))


# --------------------------------------------------------------------------- #
# Control (c): the two guards are not substitutable, and the AST lint is exact
# --------------------------------------------------------------------------- #

NEVER_CALLED = '''"""Engine-shaped module with the violation parked in a never-called body.

The docstring quotes root. prose on purpose, and a comment says root.winfo_toplevel()
too, to show that neither reaches the AST predicate.
"""


def show_dialog():
    return root .winfo_toplevel().geometry("300x300")
'''


def test_red_guards_are_not_substitutable_on_never_called_code(tmp_path):
    """Guard 1 is blind to this file; guard 2 is not. That is the split.

    The planted module imports nothing GUI and defines the violation inside a
    function that never runs, so the import graph passes it and only the AST sweep
    can see it -- the empirical form of the review's demo C.
    """
    plant = tmp_path / "plant_deferred.py"
    plant.write_text(NEVER_CALLED, encoding="utf-8")

    verdict, detail = module_verdict("plant_deferred", tmp=tmp_path)
    print(f"[red] guard 1 (import graph) on the planted file -> {verdict} | "
          f"{last_line(detail)}", flush=True)
    assert verdict == "GREEN", (
        "guard 1 started failing on a module that imports no GUI package at all, "
        "so the two guards have stopped being complementary")

    hits, unparseable = t8.global_root_hits([plant])
    print(f"[red] guard 2 (AST sweep) on the same file -> {len(hits)} hit(s): "
          f"{hits}; unparseable={unparseable}", flush=True)
    assert hits, (
        "the AST sweep missed a never-called `root .winfo_toplevel()` -- the "
        "violation class guard 2 exists for is now invisible again")
    assert not unparseable, str(unparseable)


def test_red_ast_lint_fires_on_a_planted_engine_file(tmp_path, monkeypatch):
    """The committed test goes red on a planted violation, through its own assert.

    `engine_lint_files` is pointed at the planted file instead of the repo, so the
    control is permanent without sabotaging a tracked file.
    """
    plant = tmp_path / "demucs" / "plant_network.py"
    plant.parent.mkdir(parents=True)
    plant.write_text("import json\n\n\ndef probe():\n"
                     "    return root.winfo_toplevel()\n", encoding="utf-8")
    monkeypatch.setattr(t8, "engine_lint_files", lambda *a, **k: [plant])
    with pytest.raises(AssertionError, match=GUI_ROOT_MARK) as fired:
        t8.test_no_global_tkinter_root_reference()
    print(f"[red] committed guard 2 said: {str(fired.value).strip()}", flush=True)


# Wave 2 item 2: the third guard -- a GUI import deferred into a function body, the one
# violation class guard 1 (import-time graph) and guard 2 (`root.<attr>`) are both blind
# to. The plant imports a real GUI package from the block set inside a never-called
# body, and imports nothing at module level.
DEFERRED_GUI = '''"""Engine-shaped module with a GUI import parked in a function body.

Nothing at module level touches a GUI package, so guard 1's import graph passes it, and
there is no `root.<attr>` attribute access, so guard 2's AST sweep passes it too -- this
is the class the first two guards are structurally blind to.
"""


def open_settings():
    import fluentqt
    from PySide6.QtWidgets import QDialog
    return fluentqt, QDialog
'''

GUI_IMPORT_MARK = "imported somewhere in the engine surface"


def test_red_a_deferred_gui_import_in_a_function_body_goes_red(tmp_path, monkeypatch):
    """Guard 1 and guard 2 both wave this file through; only guard 3 refuses it.

    The three-way split is the proof: the same planted module is GREEN under the import
    graph (nothing GUI executes at import), reports ZERO hits under the `root.<attr>`
    sweep, and then makes the committed guard-3 test go red. `purity_surface_files` is
    pointed at the plant instead of the repo, so the control is permanent without
    touching a tracked engine file.
    """
    plant = tmp_path / "plant_deferred_gui.py"
    plant.write_text(DEFERRED_GUI, encoding="utf-8")

    # Guard 1's half: imports nothing at module level, so the child exits 0.
    verdict, detail = module_verdict("plant_deferred_gui", tmp=tmp_path)
    print(f"[red] guard 1 (import graph) on the deferred-GUI plant -> {verdict} | "
          f"{last_line(detail)}", flush=True)
    assert verdict == "GREEN", (
        "guard 1 failed on a module that imports no GUI package at import time, so the "
        "guards have stopped being complementary: " + str(detail))

    # Guard 2's half: no `root.<attr>` anywhere, so the AST sweep reports nothing.
    hits2, unparseable2 = t8.global_root_hits([plant])
    print(f"[red] guard 2 (`root.` sweep) on the same plant -> {len(hits2)} hit(s) "
          f"| unparseable={unparseable2}", flush=True)
    assert not hits2 and not unparseable2, (
        "the `root.` sweep was supposed to be blind to a deferred import; if it now "
        f"fires here the three guards have collapsed into two: {hits2}")

    # Guard 3's half: the predicate catches both GUI names, at function-body depth.
    blocked = frozenset(t8.GUI_PACKAGES) - {"app"}
    hits3 = t8.gui_import_hits([plant], blocked)
    print(f"[red] guard 3 predicate on the plant -> {hits3}", flush=True)
    assert len(hits3) == 2, (
        "guard 3 missed one of the two deferred GUI imports (fluentqt + "
        f"PySide6.QtWidgets), so the any-depth walk is not really any-depth: {hits3}")

    monkeypatch.setattr(t8, "purity_surface_files", lambda *a, **k: [plant])
    with pytest.raises(AssertionError, match=GUI_IMPORT_MARK) as fired:
        t8.test_no_gui_import_anywhere_in_the_engine_surface()
    print(f"[red] committed guard 3 said: {str(fired.value).splitlines()[0]}",
          flush=True)


def test_red_guard_three_leaves_the_app_edge_and_clean_files_alone(tmp_path):
    """Guard 3's non-vacuity the other way: `app` is allowed, pure data is allowed.

    Without this, the control above could pass because guard 3 refuses *everything*,
    which would be wrong -- `separate_cli.py` legitimately does `from app import
    session`, and `separate.py:18` does `from gui_data.constants import *`. Both plants
    here must report zero hits.
    """
    app_edge = tmp_path / "plant_app_edge.py"
    app_edge.write_text("def run():\n    from app import session\n    return session\n",
                        encoding="utf-8")
    pure_data = tmp_path / "plant_pure_data.py"
    pure_data.write_text("from gui_data.constants import *\n\n\ndef go():\n"
                         "    import gui_data.error_handling\n"
                         "    return gui_data\n", encoding="utf-8")

    blocked = frozenset(t8.GUI_PACKAGES) - {"app"}
    assert "app" not in blocked, "the app edge leaked into guard 3's block set"
    hits = t8.gui_import_hits([app_edge, pure_data], blocked)
    print(f"[red] guard 3 on app-edge + pure-data plants -> {hits} (want none)",
          flush=True)
    assert not hits, (
        "guard 3 refused an import the engine surface is allowed to make (the `app` "
        "edge is tests/test_session.py's job, and gui_data.* pure data is not a GUI "
        f"coupling): {hits}")


def test_red_sweep_that_sweeps_nothing_goes_red(monkeypatch):
    """`assert files, "lint swept nothing"` -- the glob's own non-vacuity."""
    monkeypatch.setattr(t8, "engine_lint_files", lambda *a, **k: [])
    with pytest.raises(AssertionError, match="lint swept nothing"):
        t8.test_no_global_tkinter_root_reference()
    print('[red] empty sweep -> AssertionError("lint swept nothing")', flush=True)


def test_red_an_unparseable_file_goes_red_not_a_pass(tmp_path, monkeypatch):
    """Item 2: guard 2's `assert not unparseable` has a red run of its own.

    Round 1 added that line on its own initiative, and every other use of
    `unparseable` in this file asserts the *happy* polarity (`not unparseable`),
    so deleting the line kept all 15 controls green -- an un-asked-for guard whose
    proof is missing is how a ratchet quietly loses a tooth. The plant is a file
    that carries BOTH a real violation and a syntax error: a sweep that skipped
    unparseable files would either report the hit or report nothing at all, and
    either would look like coverage. It is written into `tmp_path` and reached by
    monkeypatching `engine_lint_files`, never into the engine tree.
    """
    broken = tmp_path / "demucs" / "plant_unparseable.py"
    broken.parent.mkdir(parents=True)
    broken.write_text("def oops(:\n    return root.destroy()\n", encoding="utf-8")

    hits, unparseable = t8.global_root_hits([broken])
    print(f"[red] planted syntax error -> hits={hits} unparseable={unparseable}",
          flush=True)
    assert len(unparseable) == 1, (
        f"the sweep did not record the broken file as unparseable, so a parse "
        f"failure is being skipped somewhere: {unparseable}")
    assert hits == [], (
        f"a file the parser rejected cannot have yielded a real hit, so this is "
        f"a predicate change, not the guard: {hits}")
    assert str(broken) in unparseable[0] and "SyntaxError" in unparseable[0], (
        "the report does not name the file or the failure, so the red message "
        f"would not help a human: {unparseable[0]}")

    monkeypatch.setattr(t8, "engine_lint_files", lambda *a, **k: [broken])
    with pytest.raises(AssertionError, match=PARSE_FAIL_MARK) as fired:
        t8.test_no_global_tkinter_root_reference()
    message = str(fired.value)
    print(f"[red] committed guard 2 said: {message.strip()}", flush=True)
    assert GUI_ROOT_MARK not in message, (
        "the parse failure reached the hits assert, which means unparseable files "
        "are being skipped rather than reported -- the coverage hole is back:\n"
        + message)


def test_red_ast_sweep_actually_sweeps_the_engine_files():
    """The sweep's reach: every engine file is a real path and is parsed.

    This is where the coverage number lives as evidence. It is deliberately not
    asserted to be 38 -- a vendored file added later must not break the suite for
    the wrong reason -- but it does assert the sweep is the whole engine, because
    a glob that silently narrowed to two files is exactly the defect fixed here.
    """
    files = t8.engine_lint_files()
    hits, unparseable = t8.global_root_hits(files)
    print(f"[red] sweep: {len(files)} files | unparseable={len(unparseable)} | "
          f"global-root hits={len(hits)}", flush=True)
    assert files, "the engine sweep found no files at all"
    assert not unparseable, "\n".join(unparseable)
    names = {p.name for p in files}
    assert "separate.py" in names and "backends.py" in names, names
    for d in ("lib_v5", "demucs"):
        covered = [p for p in files if p.relative_to(REPO).parts[0] == d]
        on_disk = sorted(REPO.glob(f"{d}/**/*.py"))
        assert len(covered) == len(on_disk), (
            f"{d}: the sweep covers {len(covered)} of {len(on_disk)} files on "
            f"disk -- the glob narrowed itself")
    assert not hits, "\n".join(hits)


def old_lint_hits(src: str) -> list[str]:
    """Replica of the pre-fix lint: one regex pass per line, comment lines skipped.

    Rebuilt here, not imported -- the committed test no longer contains it, and
    that deletion is the point of the migration.
    """
    found = []
    for i, line in enumerate(src.splitlines(), 1):
        if line.strip().startswith("#"):
            continue
        if OLD_REGEX.search(line):
            found.append(f"{i}: {line.strip()}")
    return found


def test_red_ast_predicate_is_exact_where_the_regex_was_not(tmp_path):
    """What `ast` buys, row by row: prose and nesting out, whitespace breaks in.

    Each case is a whole file body, because the cases the review named are
    properties of code, not of a single line. The `old` column is the regex the
    pre-fix lint used, run exactly the way it ran (line-based, comments skipped),
    so the trade is visible rather than claimed: the docstring row is where the
    regex was fooled and `ast` cannot be, and the whitespace/line-break rows are
    where the regex was blind and `ast` is not.
    """
    cases = {
        "nested attribute (demucs/repo.py:76)": (
            "class R:\n    def f(self):\n        return self.root.iterdir()\n",
            False),
        "a different Name": (
            "def f():\n    return tk_root.foo()\n", False),
        "attribute ON a .root, not of a Name root": (
            "def f():\n    return model.root.resolve()\n", False),
        "module docstring quoting root.winfo_toplevel()": (
            '"""Prose: the GUI may call root.winfo_toplevel() elsewhere."""\n'
            "VALUE = 1\n", False),
        "comment mentioning root.destroy()": (
            "# root.destroy() is what the guard is for\nVALUE = 1\n", False),
        "the violation itself": (
            "def f():\n    return root.destroy()\n", True),
        "whitespace before the attribute dot": (
            "def f():\n    return root .foo()\n", True),
        "Unicode attribute name": (
            "def f():\n    return root.窗口\n", True),
        "subscript between the Name and the dot": (
            "def f():\n    return root[0].bar\n", False),
    }

    for i, (label, (src, expect_hit)) in enumerate(sorted(cases.items())):
        f = tmp_path / f"case_{i}.py"
        f.write_text(src, encoding="utf-8")
        hits, unparseable = t8.global_root_hits([f])
        assert not unparseable, f"{label}: {unparseable}"
        old = old_lint_hits(src)
        print(f"[red] {'OK ' if bool(hits) == expect_hit else 'WRONG'} "
              f"ast={'HIT ' if hits else '----'} regex={'HIT ' if old else '----'} "
              f"| {label}", flush=True)
        assert bool(hits) == expect_hit, (
            f"the AST predicate disagreed on {label!r}: expected a "
            f"hit={expect_hit}, got {hits}")
        if not expect_hit and old:
            print(f"[red]   ^ the pre-fix regex still matched this prose: {old}",
                  flush=True)

    prose = cases["module docstring quoting root.winfo_toplevel()"][0]
    assert old_lint_hits(prose), (
        "this control stopped demonstrating the regex false positive it exists "
        "to demonstrate: the pre-fix lint no longer matches the docstring prose, "
        "so `ast` is no longer buying anything here and the premise moved")
    whitespace = cases["whitespace before the attribute dot"][0]
    assert not old_lint_hits(whitespace), (
        "the pre-fix regex now sees `root .foo()`, so this control no longer "
        "demonstrates the blind spot the AST migration closed")

    # Both legal spellings of an attribute access split across a line break: a
    # line-based regex cannot see either, because the dot is not on the `root`
    # line. (A bare newline before the dot is NOT valid Python, so these are the
    # only two forms the blind spot actually has.)
    broken = {
        "line_broken_paren": "def f():\n    return (root\n"
                            "            .winfo_toplevel())\n",
        "line_broken_backslash": "def f():\n    return root \\\n"
                                "        .winfo_toplevel()\n",
    }
    for label, src in broken.items():
        f = tmp_path / f"{label}.py"
        f.write_text(src, encoding="utf-8")
        hits, unparseable = t8.global_root_hits([f])
        assert not unparseable, f"{label}: {unparseable}"
        print(f"[red] line-broken attribute access ({label}) -> ast {len(hits)} "
              f"hit(s) | regex {len(old_lint_hits(src))} hit(s): {hits}", flush=True)
        assert hits, (
            f"{label}: the call split across a line break is back in the blind "
            "spot that motivated the AST migration")
        assert not old_lint_hits(src), (
            f"{label}: the pre-fix regex now sees this too, so the control no "
            "longer demonstrates the blind spot")


def test_red_ast_predicate_fires_on_a_locally_bound_root_too(tmp_path):
    """The known over-capture, committed so nobody meets it as a surprise.

    The predicate matches the *name* `root`, not a proof that it is the tkinter
    global. A parameter called `root` used as `root.something` therefore fires it
    -- `demucs/repo.py` and `demucs/pretrained.py` bind that name today but never
    take an attribute off it. The pre-decided answer (plan Task 8 Step 5) is to
    match upstream's `self.root` style, not to add an exemption list.
    """
    f = tmp_path / "shadowing.py"
    f.write_text("def __init__(self, root):\n    self.root = root\n"
                 "    return root.exists()\n", encoding="utf-8")
    hits, unparseable = t8.global_root_hits([f])
    print(f"[red] parameter-shadowed root -> {len(hits)} hit(s): {hits}", flush=True)
    assert not unparseable, str(unparseable)
    assert len(hits) == 1, (
        f"expected the documented over-capture to still fire (it is the "
        f"vendored-sync case the plan pre-decides); got {hits}")


# --------------------------------------------------------------------------- #
# Fix round 2, Item 1: the pin -- the only audit that does not read its own input
# --------------------------------------------------------------------------- #

def rebind_literal(source: str, name: str, value) -> str:
    """`source` with the top-level literal bound to `name` replaced by `value`.

    The sabotage is rewritten from the *live* list (never a copy of the names
    typed into this file, which is the forking habit Item 3 exists to break) and
    located with `ast`, so a re-wrapped or re-ordered list in the test file cannot
    make the patch silently miss.
    """
    target = next((node for node in ast.parse(source).body
                   if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == name
                           for t in node.targets)), None)
    assert target is not None, (
        f"{name} is no longer a plain top-level assignment in "
        f"tests/test_engine_purity.py, so this control's sabotage no longer "
        f"reaches the list it claims to narrow -- read that file and fix the "
        f"control instead of letting it pass vacuously")
    current = ast.literal_eval(target.value)
    assert type(current) is type(value), (
        f"{name} is a {type(current).__name__} in the test file but this control "
        f"patched it with a {type(value).__name__}; the rewrite would not be the "
        f"same edit a human would make")
    lines = source.splitlines(keepends=True)
    first = lines[target.lineno - 1]
    indent = first[: len(first) - len(first.lstrip())]
    replacement = f"{indent}{name} = {value!r}\n"
    return "".join(lines[: target.lineno - 1] + [replacement]
                   + lines[target.end_lineno:])


def patched_purity(name: str, value):
    """The committed test source, one list rebound, loaded in memory.

    The tracked file is only *read*: the patched copy is compiled and exec'd into
    a fresh namespace whose `__file__` still points at the real test, so `ROOT`
    resolves normally and nothing is written to disk.
    """
    source = PURITY_FILE.read_text(encoding="utf-8")
    patched = rebind_literal(source, name, value)
    assert patched != source, (
        f"rebinding {name} to {value!r} changed nothing, so the sabotage is a "
        f"no-op and the pin could not have been tested by it")
    ns = {"__name__": f"patched_{name.lower()}_purity",
          "__file__": str(PURITY_FILE)}
    exec(compile(patched, "patched_test_engine_purity.py", "exec"), ns)
    return ns


def test_red_the_pin_fires_when_a_name_list_shrinks():
    """Trim one name from each list and the pin goes red *naming that name*.

    The positive case runs first: if the committed pin did not pass against the
    committed lists, a red from a patched copy would prove nothing. Then each
    sabotage checks the property that makes the pin worth having -- the dropped
    name is still in the pin, so the pin is a second source and not
    `frozenset(GUI_PACKAGES) == frozenset(GUI_PACKAGES)` in disguise.
    """
    t8.test_the_pinned_lists_have_not_shrunk()
    print("[red] pin on the committed lists -> PASSES (as it must)", flush=True)

    for name, pin_name in (("GUI_PACKAGES", "PINNED_GUI_PACKAGES"),
                           ("ENGINE_IMPORTS", "PINNED_ENGINE_IMPORTS")):
        live = getattr(t8, name)
        dropped = live[-1]
        trimmed = type(live)(x for x in live if x != dropped)
        assert len(trimmed) == len(live) - 1, (
            f"{dropped} is not a single entry of {name}; the sabotage would trim "
            f"more than it claims")
        ns = patched_purity(name, trimmed)
        assert dropped not in ns[name], "the patch did not reach the live module"
        assert dropped in ns[pin_name], (
            f"{dropped} is missing from {pin_name} as well, so the pin is derived "
            f"from {name} rather than copied next to it -- this control would be "
            f"watching a tautology go red")
        with pytest.raises(AssertionError) as fired:
            ns["test_the_pinned_lists_have_not_shrunk"]()
        message = str(fired.value)
        missing_line = next((ln for ln in message.splitlines()
                             if ln.strip().startswith("MISSING from " + name)), "")
        assert dropped in missing_line and pin_name in message, (
            f"the pin failed, but not in the way that helps a human: it must name "
            f"{dropped!r} as missing and {pin_name} as the thing to edit.\n"
            f"{message}")
        print(f"[red] {name} trimmed by one ({dropped}) -> pin AssertionError:\n"
              f"      {missing_line.strip()}", flush=True)

    # The length half of the pin: a repeated name agrees with the pin as a *set*
    # and grows the *list*, so only the len() assert can see it.
    dup_name = "ENGINE_IMPORTS"
    live = list(t8.ENGINE_IMPORTS)
    ns = patched_purity(dup_name, live + [live[0]])
    assert frozenset(ns[dup_name]) == ns["PINNED_ENGINE_IMPORTS"], (
        "premise broken: the duplicate was supposed to be invisible to set "
        "equality; if it is not, the len() assert this control proves is dead code")
    with pytest.raises(AssertionError, match=DUP_MARK) as fired:
        ns["test_the_pinned_lists_have_not_shrunk"]()
    print(f"[red] {dup_name} with a repeated name -> "
          f"{str(fired.value).splitlines()[0]}", flush=True)


# --------------------------------------------------------------------------- #
# Item 5: the subprocess bound is real, not a decoration
# --------------------------------------------------------------------------- #

def test_red_a_child_that_outlasts_the_bound_reads_as_a_timeout(monkeypatch):
    """Bound too small -> the failure message names the bound and says TIMEOUT.

    Without `timeout=`, a blocking engine import (network in
    `demucs.pretrained`, MIOpen init) hangs the whole suite forever; this control
    proves the bound fires and, crucially, that it fires as a *timeout* rather
    than as a mysterious purity failure.

    Item 3 rides on the same monkeypatch: it is the only place that both narrows
    `t8.CHILD_TIMEOUT_SECONDS` and needs the controls' *own* children to obey the
    narrowing, so it is where a fork between the two bounds shows up.
    """
    assert CHILD_TIMEOUT == t8.CHILD_TIMEOUT_SECONDS, (
        f"the controls file has forked from the bound it exists to demonstrate "
        f"(CHILD_TIMEOUT={CHILD_TIMEOUT} vs t8.CHILD_TIMEOUT_SECONDS="
        f"{t8.CHILD_TIMEOUT_SECONDS}). Read the Item 3 comment above "
        f"`CHILD_TIMEOUT`: the controls must not carry a second copy of this "
        f"number.")
    monkeypatch.setattr(t8, "CHILD_TIMEOUT_SECONDS", 0.05)
    assert child_timeout() == 0.05, (
        "the controls captured the bound at import time, so this monkeypatch "
        "changes only the test's child and not these -- the tie Item 3 asks for "
        "would be decorative")
    # The probe must outlast the bound, not outlive the run. It used to be
    # `while True: time.sleep(0.05)`, which is unsound on this venv and cost this plan
    # hours: `.venv/Scripts/python.exe` is a *redirector* -- it launches the base
    # interpreter as a child (measured: the process holding the pipe is
    # `%LOCALAPPDATA%\...\Python312\python.exe` running the same `-c` script) -- so when
    # `subprocess.run`'s timeout path calls `process.kill()` it kills the stub, the real
    # interpreter survives and keeps the stdout pipe open, and `communicate()` blocks
    # until that grandchild exits of its own accord. For `while True` that is never: the
    # control hung, the whole controls file hung (~1200 s observed, no summary line), and
    # an interrupted run left a permanent sleeper behind (six were alive on this machine,
    # each one a plausible explanation for the ledger's "unreproducible suite time").
    # 5 s is 100x the bound this control narrows to (0.05 s), so the assertion is
    # unchanged -- the child still outlasts the bound and the failure still reads as
    # TIMEOUT -- while the worst case for the pipe is now 5 s and an orphan self-exits.
    with pytest.raises(subprocess.TimeoutExpired) as sleeping:
        run_child("import time\n"
                  "deadline = time.time() + 5\n"
                  "while time.time() < deadline: time.sleep(0.05)\n")
    print(f"[red] run_child under the patched bound -> {str(sleeping.value)}",
          flush=True)
    with pytest.raises(AssertionError, match="TIMEOUT") as fired:
        t8.test_engine_imports_with_tkinter_unavailable()
    message = str(fired.value)
    print(f"[red] with CHILD_TIMEOUT_SECONDS=0.05 the committed test said: "
          f"{message.splitlines()[0]}", flush=True)
    assert "CHILD_TIMEOUT_SECONDS" in message, message


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
