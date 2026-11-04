"""Task 8: the engine purity ratchet.

`separate.py`, `lib_v5/` and `demucs/` must be importable with no GUI present,
because Task 13 ships a GUI-free CLI runner over them and the tkinter app in
`UVR.py` is being replaced by PySide6. Two guards, split deliberately:

  1. `test_engine_imports_with_tkinter_unavailable` proves the *import graph*
     behaves, which is the only way to see a coupling that arrives transitively
     through a wildcard import (`separate.py:18` does
     `from gui_data.constants import *`, and no regex over that file can see
     what it drags in).
  2. `test_no_global_tkinter_root_reference` lints for the one thing only text
     can catch: code reaching for the GUI's global `root` widget.
"""
import pathlib, re, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Engine modules this fork must be able to import with no GUI present.
# Verified by running the blocker against each one at HEAD 4463a80. The RED set is
# separate, lib_v5.apollo_inference, demucs.{apply,utils,repo,pretrained,htdemucs}
# -- all seven trace to the SAME two dead imports -- and the already-green set is
# backends, lib_v5.{mdxnet,bs_roformer,mel_band_roformer,attend,results} and
# demucs.states. Do not trim this list to the modules that currently fail: the
# point is that deleting the two lines turns ALL of them green at once.
# (An earlier draft of this list named `lib_v5.roformer`, which does not exist --
# there is no such module. The real ones are bs_roformer and mel_band_roformer.)
ENGINE_IMPORTS = ["separate", "backends", "lib_v5.mdxnet", "lib_v5.bs_roformer",
                  "lib_v5.mel_band_roformer", "lib_v5.attend", "lib_v5.results",
                  "lib_v5.apollo_inference", "demucs.apply", "demucs.utils",
                  "demucs.repo", "demucs.pretrained", "demucs.states",
                  "demucs.htdemucs"]


GUI_PACKAGES = ("tkinter", "gui_data.tkinterdnd2", "gui_data.sv_ttk", "PIL.ImageTk")

# `root.` needs the negative lookbehind: bare \broot. also matches the
# legitimate `self.root.iterdir()` in demucs/repo.py:76,114. The global tkinter
# root is, by construction, a name nothing is attached to.
GLOBAL_ROOT = re.compile(r"(?<![.\w])root\.")
LINT_FILES = ["separate.py", "backends.py"]

def test_engine_imports_with_tkinter_unavailable():
    """Run in a fresh subprocess: a module already in sys.modules would not
    re-trigger the finder, so an in-process test can pass by accident."""
    script = (
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
        + "".join(f"import {m}\n" for m in ENGINE_IMPORTS)
    )
    r = subprocess.run([sys.executable, "-c", script], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode == 0, (
        "the engine cannot be imported without a GUI:\n" + r.stderr)

def test_no_global_tkinter_root_reference():
    hits = []
    for f in LINT_FILES:
        for i, line in enumerate(
                (ROOT / f).read_text(encoding="utf-8").splitlines(), 1):
            if line.strip().startswith("#"):
                continue
            if GLOBAL_ROOT.search(line):
                hits.append(f"{f}:{i}: {line.strip()}")
    assert not hits, "engine references the GUI root:\n" + "\n".join(hits)
