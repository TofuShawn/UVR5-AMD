"""Task 8: the engine purity ratchet.

`separate.py`, `backends.py` and every `lib_v5/**/*.py`, `demucs/**/*.py` file
must be importable with the tkinter GUI packages unavailable. Note the scope:
that is a *tkinter*-freedom claim, not a GUI-freedom one -- PySide6/Qt are
deliberately not blocked here (Task 13 owns that), so nothing in this file
proves the engine is GUI-free. Two guards, split deliberately, and each one is
blind to exactly what the other sees:

  1. `test_engine_imports_with_tkinter_unavailable` proves the *import graph*
     behaves, which is the only way to see a coupling that arrives transitively
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
(run it explicitly; it is not collected by bare `pytest`).
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
# column is 14 GREEN, and the 7-RED column lives in `git show 3e612ea` (the two
# deletions), which the controls re-create the failure signature of by planting.
ENGINE_IMPORTS = ["separate", "backends", "lib_v5.mdxnet", "lib_v5.bs_roformer",
                  "lib_v5.mel_band_roformer", "lib_v5.attend", "lib_v5.results",
                  "lib_v5.apollo_inference", "demucs.apply", "demucs.utils",
                  "demucs.repo", "demucs.pretrained", "demucs.states",
                  "demucs.htdemucs"]


GUI_PACKAGES = ("tkinter", "gui_data.tkinterdnd2", "gui_data.sv_ttk", "PIL.ImageTk")

# Guard 2's whole sweep, as a glob, so it follows the engine as it grows instead
# of drifting behind a hand-named list. This is the set the plan's constraint
# speaks of; `assert files` below is what keeps a wrong glob from reading as a
# pass. (The count is printed, never asserted: pinning it to 38 today would break
# the suite the day a vendored file is added, for the wrong reason.)
LINT_GLOBS = ("separate.py", "backends.py", "lib_v5/**/*.py", "demucs/**/*.py")

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
    hits, unparseable = global_root_hits(files)
    print(f"[purity] guard 2 swept {len(files)} engine files: "
          f"{len(hits)} global-`root.` attribute hits, "
          f"{len(unparseable)} files the parser rejected", flush=True)
    assert not unparseable, (
        "guard 2 cannot lint a file it cannot parse, so a parse failure is a "
        "coverage hole and must not read as a pass:\n" + "\n".join(unparseable))
    assert not hits, "engine references the GUI root:\n" + "\n".join(hits)
