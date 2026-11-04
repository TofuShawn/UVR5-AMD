"""Task 10 RED controls: proof that finding #4's arch pin really was the bug.

WHAT THIS FILE PROVES
---------------------
Task 10's review finding #4 was that ``Catalog.demucs_pre_proc_for`` passed
``arch="demucs"`` whenever a ``primary_stem`` arrived, which sent a VR or MDX name
in the pre-proc slot to ``_locate_weight`` against ``Demucs_Models/`` only, raised
``MissingWeightError``, and let the ``except CatalogError: return None`` swallow it
-- the fork quietly loses the pre-process model. The fix deleted the parameter and
the branch (``catalog.py``, `UVR.py:7094-7096` passes no arch at all).

Fix round 1 documented that before/after pair *inside* the live test, as an
``inspect.signature`` gate whose ``if`` branch could never be taken again -- a dead
branch in a shipped test that only exists to describe a bug that is gone. That shape
is what this file moves: the committed test now asserts the post-fix behaviour only,
and the "before" picture is re-created here by **replaying the sabotage** -- a plant
module written into ``tmp_path`` that subclasses ``Catalog`` and puts the stem branch
back, run against the same fixture as the shipped class.

  * the planted method returns ``None`` for a VR pre-proc name (the bug, on demand);
  * the shipped method returns the VR spec for the same call shape (the fix);
  * the planted method still returns the VR spec when no stem is passed, and still
    resolves a genuine Demucs name when one is -- so the red cannot come from a plant
    that is simply broken for every input (the undeleted case beside the sabotaged
    one, as ``tools/task8_red_controls.py`` does it).

No subprocess is needed for any of that: the sabotage is a Python-level override, not
an import-graph or GUI-blocking property, so ``task8_red_controls.py``'s
``run_child`` would only add cost and a second interpreter to keep honest. Nothing
here writes outside ``tmp_path``, and the repository -- including ``catalog.py`` -- is
only ever read, so the file is idempotent: run it twice, same three verdicts.

NON-COLLECTED BY DESIGN
-----------------------
The file is named ``task10_red_controls.py``, not ``test_*.py``: a bare ``pytest``
run must never execute it (there is no ``pytest.ini``/``pyproject.toml`` here, so the
filename is the whole guard). Run it explicitly:

    .venv/Scripts/python.exe tools/task10_red_controls.py
"""
from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import catalog
from gui_data.constants import DEMUCS_ARCH_TYPE, VR_ARCH_TYPE

REPO = Path(__file__).resolve().parents[1]
CATALOG_SOURCE = (REPO / "catalog.py").read_text(encoding="utf-8")

#: The pre-fix method, verbatim from `git show a57069a~1:catalog.py` (`:689`).
PRE_FIX_PIN = '''"""Planted pre-fix `demucs_pre_proc_for`: the arch pin, back on the stem branch."""
import catalog
from gui_data.constants import NO_MODEL


class PinnedCatalog(catalog.Catalog):
    def demucs_pre_proc_for(self, requested, *, is_activated=True, primary_stem=None):
        if not requested or requested == NO_MODEL or not is_activated:
            return None
        try:
            return self.resolve(requested, arch="demucs") if primary_stem else self.resolve(requested)
        except catalog.CatalogError:
            return None
'''


def arch_kwarg_calls(source: str) -> list[str]:
    """Calls of the shape ``x.resolve(..., arch=<anything>)`` in `source`, as text.

    An AST predicate rather than a substring scan, because both the shipped method
    and the plant *talk about* ``arch="demucs"`` -- the docstring explains why it is
    gone -- and a string match cannot tell prose from code. This is the same reason
    ``tests/test_engine_purity.py`` moved its lint to ``ast``.
    """
    tree = ast.parse(textwrap.dedent(source))
    return [ast.unparse(node) for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute) and node.func.attr == "resolve"
            and any(kw.arg == "arch" for kw in node.keywords)]


def load_plant(tmp: Path):
    """Write the plant into ``tmp`` and import it -- never into the repo tree."""
    module_file = tmp / "task10_plant_arch_pin.py"
    module_file.write_text(PRE_FIX_PIN, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(module_file.stem, module_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert arch_kwarg_calls(inspect.getsource(module.PinnedCatalog.demucs_pre_proc_for)), (
        "the plant no longer passes an arch to resolve(), so it is not the pre-fix "
        "shape this file exists to replay -- read git show a57069a~1:catalog.py and "
        "fix PRE_FIX_PIN instead of letting the controls pass vacuously")
    return module


def build_root(tmp: Path) -> tuple[catalog.Catalog, str, str]:
    """A synthetic root holding one VR weight with its baked params and one Demucs weight.

    Fully synthetic on purpose: ``tools/task10_red_controls.py`` is verified without
    ``UVR_MODEL_ROOT``, so a fixture that read the installed tree would measure the
    environment instead of the pin.
    """
    root = tmp / "weights"
    vr_name, demucs_name = "PreProc-VR", "955717e8-8726e21a.th"
    weight = root / "VR_Models" / f"{vr_name}.pth"
    weight.parent.mkdir(parents=True, exist_ok=True)
    weight.write_bytes(b"task10-preproc-weight")
    table = weight.parent / "model_data" / "model_data.json"
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text(json.dumps({catalog.hash_of(weight): {
        "vr_model_param": "4band_44100", "primary_stem": "Vocals"}}), encoding="utf-8")
    (root / "Demucs_Models").mkdir(parents=True, exist_ok=True)
    (root / "Demucs_Models" / demucs_name).write_bytes(b"task10-demucs-weight")
    names = root / "Demucs_Models" / "model_data" / "model_name_mapper.json"
    names.parent.mkdir(parents=True, exist_ok=True)
    names.write_text(json.dumps({demucs_name: "UVR-Demucs-Probe"}), encoding="utf-8")
    writable = tmp / "appdata" / "model_data"
    return catalog.Catalog(root, writable, fallback_root=None), vr_name, demucs_name


# --------------------------------------------------------------------------- #
# Premise: the shipped method is the post-fix shape this plant claims to undo
# --------------------------------------------------------------------------- #

def test_red_premise_the_shipped_method_no_longer_takes_a_stem():
    """If `primary_stem` came back, this file's sabotage would be the shipped code.

    The control is only evidence while the thing it plants is genuinely gone; the
    committed ``tests/test_catalog.py::test_demucs_pre_proc_for_has_no_primary_stem_parameter``
    is the guard, and the message below says so rather than letting the control pass
    vacuously.
    """
    params = inspect.signature(catalog.Catalog.demucs_pre_proc_for).parameters
    assert "primary_stem" not in params, (
        "`primary_stem` is back on Catalog.demucs_pre_proc_for, so the plant in "
        "tools/task10_red_controls.py is no longer a sabotage -- it is the shipped "
        "behaviour. Decide the finding-#4 ruling again, then fix this file.")
    shipped = inspect.getsource(catalog.Catalog.demucs_pre_proc_for)
    # AST, not a substring: the shipped docstring *explains* the deleted pin, and a
    # string scan cannot tell that prose from the code the fix removed.
    assert arch_kwarg_calls(shipped) == [], (
        "demucs_pre_proc_for passes an arch to resolve() again, so the plant in "
        "tools/task10_red_controls.py is no longer a sabotage -- it is the shipped "
        "behaviour. Decide the finding-#4 ruling again, then fix this file.")
    print("[red] premise: shipped method has no stem parameter and no arch pin",
          flush=True)


def test_red_the_stem_argument_is_a_typeerror_now(tmp_path):
    """The deleted signature, proven from the caller's side.

    Task 11's port cannot pass a stem believing it selects the architecture, because
    there is nothing to pass: the keyword is gone, not ignored.
    """
    c, vr_name, _ = build_root(tmp_path)
    with pytest.raises(TypeError) as fired:
        c.demucs_pre_proc_for(vr_name, primary_stem="Vocals")
    print(f"[red] shipped call with primary_stem -> {str(fired.value).strip()}",
          flush=True)
    assert "primary_stem" in str(fired.value), fired.value


# --------------------------------------------------------------------------- #
# The sabotage: put the pin back and watch the VR name disappear
# --------------------------------------------------------------------------- #

def test_red_the_planted_arch_pin_loses_the_vr_pre_proc_model(tmp_path):
    """Same catalog, same call, two classes: the pin returns None, the fix returns VR.

    Four probes run side by side so no verdict can come from "the plant resolves
    nothing at all": planted+stem is the only one that may be ``None``, and the
    genuine Demucs name still answers ``demucs`` under the pin -- the case the pin
    was originally written for.
    """
    c, vr_name, demucs_name = build_root(tmp_path)
    pinned = load_plant(tmp_path).PinnedCatalog(c.model_root, c.writable_dir,
                                                fallback_root=c.fallback_root)

    shipped = c.demucs_pre_proc_for(vr_name)
    pin_stem = pinned.demucs_pre_proc_for(vr_name, primary_stem="Vocals")
    pin_nostem = pinned.demucs_pre_proc_for(vr_name)
    pin_demucs = pinned.demucs_pre_proc_for(demucs_name, primary_stem="Vocals")

    rows = {
        "shipped, no stem (the fix)": (shipped, "vr"),
        "planted, no stem (undeleted case)": (pin_nostem, "vr"),
        "planted, stem, genuine demucs name": (pin_demucs, "demucs"),
        "planted, stem, VR name (the bug)": (pin_stem, None),
    }
    for label, (spec, _) in rows.items():
        print(f"[red] {label:<38} -> "
              f"{'None' if spec is None else spec.arch}", flush=True)

    assert shipped is not None and shipped.arch == "vr", (
        "the shipped method already loses the VR pre-proc model, so there is no "
        "before/after pair to demonstrate and this control is measuring a regression "
        "instead of the fix: " + str(shipped))
    assert pin_stem is None, (
        "the planted `arch=\"demucs\"` did NOT swallow the VR name -- the pin was "
        "never the cause finding #4 names, so the committed test's docstring is "
        "describing a bug that does not exist")
    assert pin_nostem is not None and pin_nostem.arch == "vr", (
        "the plant returns None for every input, so `pin_stem is None` above proves "
        "nothing: the sabotage must bite only on the stem branch")
    assert pin_demucs is not None and pin_demucs.arch == "demucs", (
        "the plant broke the genuine Demucs case too, which is the case the pin was "
        "written for -- the control would then be demonstrating a broken class, not "
        "an over-narrow arch")

    # and the failure the pin hides is a real one, not a silent None:
    assert shipped.process_method == VR_ARCH_TYPE
    assert pin_demucs.process_method == DEMUCS_ARCH_TYPE
    with pytest.raises(catalog.CatalogError):
        pinned.resolve(vr_name, arch="demucs")
    print("[red] verdict: the pin bites, the shipped code does not -- finding #4 "
          "is demonstrated, not asserted", flush=True)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
