"""Tests for `catalog.py` (Task 10 — ModelCatalog owns model metadata).

Every assertion about *counts* reads the JSON files at test time; nothing here
hardcodes a number that came from the brief. The tier/provenance assertions use
two real roots (the installed weights dir and the repo's version-controlled
`models/`) plus synthetic `tmp_path` roots, because the real roots disagree —
see `task-10-brief.md` "the repo tree's copies are a stale subset".

`F:/Software/Ultimate Vocal Remover/` is read-only input: nothing in this file
writes inside it. Every write goes to `tmp_path`.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
from pathlib import Path

import pytest

import catalog
import uvr_paths
from gui_data.constants import (BASS_STEM, DEMUCS_ARCH_TYPE, DRUM_STEM, INST_STEM,
                                MDX_ARCH_TYPE, NO_MODEL, OTHER_STEM, VR_ARCH_TYPE,
                                VOCAL_STEM)

REPO = Path(__file__).resolve().parents[1]
MODEL_ROOT = uvr_paths.MODEL_ROOT  # the *installed* root when UVR_MODEL_ROOT is set
REPO_MODELS = REPO / "models"

VR_TABLE = "VR_Models/model_data/model_data.json"
MDX_TABLE = "MDX_Net_Models/model_data/model_data.json"
MDX_NAMES = "MDX_Net_Models/model_data/model_name_mapper.json"
DEMUCS_NAMES = "Demucs_Models/model_data/model_name_mapper.json"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def _catalog(tmp_path: Path, model_root: Path = None, fallback_root=None) -> catalog.Catalog:
    """Catalog over `model_root` with a scratch writable dir outside every root."""
    writable = tmp_path / "appdata" / "model_data"
    writable.mkdir(parents=True, exist_ok=True)
    return catalog.Catalog(model_root if model_root is not None else MODEL_ROOT,
                           writable, fallback_root=fallback_root)


def _fake_weight(path: Path, payload: bytes = b"q") -> Path:
    """A weight big enough to exercise the tail window (11 MiB > 10000 KiB)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload * (11 * 1024 * 1024))
    return path


def _vr_root(tmp_path: Path, name: str, *, table=None, per_hash=None) -> tuple[Path, str]:
    """Build a synthetic model root holding one VR weight plus its tables.

    Returns (root, digest-of-the-weight). `table=None` means "empty baked table",
    so by default nothing but the weight exists and resolution lands on tier 3.
    """
    root = tmp_path / "root"
    weight = _fake_weight(root / "VR_Models" / f"{name}.pth")
    digest = catalog.hash_of(weight)
    _write_json(root / VR_TABLE, {} if table is None else table)
    for h, params in (per_hash or {}).items():
        _write_json(root / "VR_Models/model_data" / f"{h}.json", params)
    return root, digest


def _params(stem: str = "Vocals") -> dict:
    return {"vr_model_param": "4band_44100", "primary_stem": stem}


# The precedence fixtures below put one relative table path in all three roots.
_ROOT_ROLES = ("writable_dir", "model_root", "fallback_root")
_PRECEDENCE_WEIGHT = b"task10-precedence-weight"


def _three_roots(tmp_path: Path, tag: str) -> dict[str, Path]:
    """Three disjoint scratch roots, keyed by the `RootRole` names they stand for."""
    roots = {role: tmp_path / f"{tag}-{role}" for role in _ROOT_ROLES}
    for path in roots.values():
        path.mkdir(parents=True, exist_ok=True)
    return roots


def _same_weight(root: Path, rel: str) -> str:
    """Write one small weight and return its digest — identical bytes everywhere.

    Every root's table keys on this digest, so a wrong answer in the precedence
    tests can only come from the wrong *root* answering, never from a hash mismatch.
    """
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_PRECEDENCE_WEIGHT)
    return catalog.hash_of(path)


# --------------------------------------------------------------------------- #
# 1. hash compatibility — the one test that must not be flexible
# --------------------------------------------------------------------------- #

def test_hash_uses_tail_10mb(tmp_path):
    blob = b"x" * (11 * 1024 * 1024)
    p = tmp_path / "m.pth"
    p.write_bytes(blob)
    expected = hashlib.md5(blob[-10000 * 1024:]).hexdigest()
    assert catalog.hash_of(p) == expected


def test_hash_falls_back_to_whole_file_below_10mb(tmp_path):
    blob = b"y" * (1024 * 1024)
    p = tmp_path / "s.pth"
    p.write_bytes(blob)
    assert catalog.hash_of(p) == hashlib.md5(blob).hexdigest()


def test_hash_of_agrees_with_the_golden_harness_helper():
    """`catalog.hash_of` vs `tests/golden/settings.py:_vr_model_hash`, real fixture.

    The brief's rule is one implementation, not two: the golden helper already
    ports `UVR.py:793-799` correctly, so the two digests must be equal on the
    weight the golden suite actually runs. Migrating the harness onto `catalog`
    is Task 11's job; this test is the leash that keeps them equal until then.
    """
    from tests.golden.settings import _vr_model_hash

    weight = uvr_paths.MODELS["vr"]
    assert catalog.hash_of(weight) == _vr_model_hash(weight)


def test_tail_window_survives_the_10000_kib_vs_10_mibibyte_distinction():
    """The digest must come from 10,240,000 bytes, never from 10 MiB.

    `UVR.py:795` is `f.seek(- 10000 * 1024, 2)`. An implementer who "rounds" that
    to `10 * 1024 * 1024` produces a different digest for every weight larger
    than the window -- i.e. every weight -- and each existing
    `model_data/<hash>.json` lookup then fails silently.
    """
    assert catalog.TAIL_WINDOW_BYTES == 10000 * 1024
    assert catalog.TAIL_WINDOW_BYTES != 10 * 1024 * 1024

    weight = uvr_paths.MODELS["vr"]
    digest = catalog.hash_of(weight)
    assert digest == hashlib.md5(weight.read_bytes()[-catalog.TAIL_WINDOW_BYTES:]).hexdigest()
    assert digest != hashlib.md5(weight.read_bytes()[-(10 * 1024 * 1024):]).hexdigest()


def test_hash_memo_is_keyed_by_stat_not_bare_path(tmp_path):
    """Tier 0 (`UVR.py:315 model_hash_table`) is keyed by path alone -- not ported.

    Same name, swapped bytes: a bare-path memo serves the old hash forever, which
    is the staleness consequence 3 of the plan names. `Catalog.hash_of` keys by
    (path, size, mtime), so re-hashing happens the moment the file changes.
    """
    c = _catalog(tmp_path)
    weight = _fake_weight(tmp_path / "swap" / "VR_Models" / "X.pth", b"a")
    first = c.hash_of(weight)
    assert first == c.hash_of(weight)  # memo hit, same answer

    weight.write_bytes(b"b" * (11 * 1024 * 1024))
    stat = weight.stat()
    import os
    os.utime(weight, (stat.st_atime, stat.st_mtime + 10))  # deterministic mtime bump
    assert c.hash_of(weight) == hashlib.md5(b"b" * (10000 * 1024)).hexdigest()
    assert c.hash_of(weight) != first


def test_catalog_has_no_gui_dependency():
    """AST count, not a string scan: 0 tkinter imports, 0 non-constants gui_data.

    `catalog.py` is app-side but GUI-free; the only `gui_data` import allowed is
    `gui_data.constants` (the exception `separate.py:18` uses).
    """
    tree = ast.parse((REPO / "catalog.py").read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
        elif isinstance(node, ast.Import):
            imported.extend(a.name for a in node.names)
    tkinter_nodes = [m for m in imported if m.split(".")[0] == "tkinter"]
    gui_nodes = [m for m in imported if m.split(".")[0] == "gui_data"
                 and m != "gui_data.constants"]
    settings_nodes = [m for m in imported if m.split(".")[0] == "settings"]
    assert tkinter_nodes == []
    assert gui_nodes == []
    assert settings_nodes == [], "R-T10-writable: writable_dir is a ctor arg, not settings.config_dir()"
    assert "gui_data.constants" in imported

    # and no tkinter is pulled in transitively by importing the catalog
    assert not any(m.split(".")[0] == "tkinter" for m in __import__("sys").modules)


def test_catalog_reads_no_tkinter_app_global_root_attribute():
    """The `root.` reads these tests replace were `Attribute(Name('root'))` nodes."""
    tree = ast.parse((REPO / "catalog.py").read_text(encoding="utf-8"))
    hits = [node for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
            and node.value.id == "root"]
    assert hits == []


# --------------------------------------------------------------------------- #
# 2. provenance: which root and which tier answered
# --------------------------------------------------------------------------- #

def test_resolve_finds_the_vr_fixture(tmp_path):
    """Step 1's third test from the plan, plus what it answers with."""
    c = _catalog(tmp_path)
    spec = c.resolve("1_HP-UVR")
    assert spec.weights_path.exists()
    assert spec.process_method == VR_ARCH_TYPE
    assert spec.arch == "vr"
    assert spec.model_data == _json(MODEL_ROOT / VR_TABLE)[spec.hash]
    assert spec.primary_stem == spec.model_data["primary_stem"]
    assert spec.secondary_stem != spec.primary_stem
    assert spec.needs_user_params is False


def test_baked_table_precedence_writable_then_model_root_then_fallback(tmp_path):
    """Finding #1: `model_root` SHADOWS `fallback_root` on a baked params table.

    This replaces `test_resolve_reports_root_and_tier_for_every_source`, which
    asserted `weights_path.exists()` and `process_method` -- neither of which any
    order of `_roots()` can falsify, so deleting provenance entirely kept it green.

    Here the same relative table path sits in all three roots with three DIFFERENT
    contents, and each content carries a filler key of its own, so "first file
    wins", "merge the roots" and "fallback wins" are three different answers. The
    weight bytes are identical in every root, so one digest keys every table and
    only precedence -- never the hash -- can change what `resolve()` returns.
    """
    name = "Precedence-VR"
    weight_rel = f"VR_Models/{name}.pth"
    roots = _three_roots(tmp_path, "vr")
    digest = _same_weight(roots["model_root"], weight_rel)
    _same_weight(roots["fallback_root"], weight_rel)

    stems = {"writable_dir": "Instrumental", "model_root": "Vocals",
             "fallback_root": "Bass"}
    contents = {role: {digest: _params(stem), f"filler-only-in-{role}": _params("Other")}
                for role, stem in stems.items()}
    for role, table in contents.items():
        _write_json(roots[role] / VR_TABLE, table)

    def answer(model_root: Path, writable: Path, fallback_root: Path):
        c = catalog.Catalog(model_root, writable, fallback_root=fallback_root)
        return c, c.resolve(name)

    def expect(role: str, root: Path, c, spec):
        assert spec.provenance.tier is catalog.Tier.BAKED_TABLE
        assert spec.model_data == contents[role][digest], \
            f"{role}'s table content must answer, not a merge of the roots"
        assert spec.provenance.root_role is getattr(catalog.RootRole, role.upper()), \
            f"_roots() reordered: {role} no longer answers"
        assert spec.provenance.root == root
        assert spec.provenance.source == root / VR_TABLE
        assert set(c.baked_table("vr", "params")) == set(contents[role]), \
            "the roots are shadowed, never merged -- a merge carries the other fillers"

    # (i) all three roots hold the table: the refresh location wins. This is the
    # shape Task 12 actually produces.
    c, spec = answer(roots["model_root"], roots["writable_dir"], roots["fallback_root"])
    expect("writable_dir", roots["writable_dir"], c, spec)
    assert spec.weights_path == roots["model_root"] / weight_rel, \
        "writable_dir holds params, never weights"

    # (ii) no refresh yet, so the two weights roots compete: model_root wins.
    c, spec = answer(roots["model_root"], tmp_path / "vr-writable-fresh",
                     roots["fallback_root"])
    expect("model_root", roots["model_root"], c, spec)

    # (iii) model_root ships no table at all: the answer falls through to
    # fallback_root -- the pair with (ii) is what a reorder cannot survive.
    bare = tmp_path / "vr-model-bare"
    bare.mkdir()
    c, spec = answer(bare, tmp_path / "vr-writable-fresh2", roots["fallback_root"])
    expect("fallback_root", roots["fallback_root"], c, spec)


def test_names_table_precedence_mdx_answers_by_its_own_table(tmp_path):
    """Finding #1, second available shape: MDX's names table is shadowed too.

    `("mdx", "names")` is pinned on the mapping it hands the weight lookup -- which
    file the name becomes and where that file was found -- because for MDX the
    names table is what decides the weight, and `provenance` then reports the
    params table that supplied `model_data`. `ARCHS` gives vr no names table and
    demucs no params table, so this and the test above are the two shapes.
    """
    display = "Precedence-MDX"
    picked = {"writable_dir": "writable-pick.ckpt", "model_root": "modelroot-pick.ckpt",
              "fallback_root": "fallbackroot-pick.ckpt"}
    stems = {"writable_dir": "Instrumental", "model_root": "Vocals",
             "fallback_root": "Bass"}
    roots = _three_roots(tmp_path, "mdx")
    # writable_dir is not a weights root, so the file its table names lives in
    # model_root: a writable answer must move the *mapping*, not the lookup order.
    digest = _same_weight(roots["model_root"], f"MDX_Net_Models/{picked['writable_dir']}")
    _same_weight(roots["model_root"], f"MDX_Net_Models/{picked['model_root']}")
    _same_weight(roots["fallback_root"], f"MDX_Net_Models/{picked['fallback_root']}")

    names = {role: {picked[role]: display,
                    f"filler-only-in-{role}.ckpt": f"Filler Only In {role}"}
             for role in _ROOT_ROLES}
    params = {role: {digest: _params(stem), f"filler-only-in-{role}": _params("Other")}
              for role, stem in stems.items()}
    for role in _ROOT_ROLES:
        _write_json(roots[role] / MDX_NAMES, names[role])
        _write_json(roots[role] / MDX_TABLE, params[role])

    def answer(model_root: Path, writable: Path, fallback_root: Path):
        c = catalog.Catalog(model_root, writable, fallback_root=fallback_root)
        return c, c.resolve(display, arch="mdx")

    def expect(role: str, table_root: Path, weights_root: Path, c, spec):
        assert spec.weights_path == weights_root / "MDX_Net_Models" / picked[role], \
            f"the {role} names table did not answer: it maps a different file"
        assert set(c.baked_table("mdx", "names")) == set(names[role]), \
            "names tables are shadowed, never merged"
        assert c.known_names("mdx") == sorted(names[role])
        assert spec.provenance.tier is catalog.Tier.BAKED_TABLE
        assert spec.provenance.root_role is getattr(catalog.RootRole, role.upper())
        assert spec.provenance.source == table_root / MDX_TABLE
        assert spec.model_data == params[role][digest]

    c, spec = answer(roots["model_root"], roots["writable_dir"], roots["fallback_root"])
    expect("writable_dir", roots["writable_dir"], roots["model_root"], c, spec)

    c, spec = answer(roots["model_root"], tmp_path / "mdx-writable-fresh",
                     roots["fallback_root"])
    expect("model_root", roots["model_root"], roots["model_root"], c, spec)

    bare = tmp_path / "mdx-model-bare"
    bare.mkdir()
    c, spec = answer(bare, tmp_path / "mdx-writable-fresh2", roots["fallback_root"])
    expect("fallback_root", roots["fallback_root"], roots["fallback_root"], c, spec)


def test_names_table_precedence_demucs_reports_the_table_that_answered(tmp_path):
    """The names-table half of the same pin, on the one arch whose `provenance`
    *is* the names table: demucs takes its params from the engine (`ARCHS`), so
    `resolve()` reports root_role and source of `model_name_mapper.json` itself.
    """
    display = "Precedence-Demucs"
    picked = {"writable_dir": "dwritable.th", "model_root": "dmodel.th",
              "fallback_root": "dfallback.th"}
    roots = _three_roots(tmp_path, "demucs")
    _same_weight(roots["model_root"], f"Demucs_Models/{picked['writable_dir']}")
    _same_weight(roots["model_root"], f"Demucs_Models/{picked['model_root']}")
    _same_weight(roots["fallback_root"], f"Demucs_Models/{picked['fallback_root']}")
    names = {role: {picked[role]: display,
                    f"filler-only-in-{role}.th": f"Filler Only In {role}"}
             for role in _ROOT_ROLES}
    for role in _ROOT_ROLES:
        _write_json(roots[role] / DEMUCS_NAMES, names[role])

    def answer(model_root: Path, writable: Path, fallback_root: Path):
        c = catalog.Catalog(model_root, writable, fallback_root=fallback_root)
        return c, c.resolve(display, arch="demucs")

    def expect(role: str, root: Path, weights_root: Path, c, spec):
        assert spec.provenance.tier is catalog.Tier.BAKED_TABLE
        assert spec.provenance.root_role is getattr(catalog.RootRole, role.upper())
        assert spec.provenance.root == root
        assert spec.provenance.source == root / DEMUCS_NAMES, \
            f"{role}'s names table must be the source named in provenance"
        assert spec.weights_path == weights_root / "Demucs_Models" / picked[role]
        assert set(c.baked_table("demucs", "names")) == set(names[role])

    c, spec = answer(roots["model_root"], roots["writable_dir"], roots["fallback_root"])
    expect("writable_dir", roots["writable_dir"], roots["model_root"], c, spec)

    c, spec = answer(roots["model_root"], tmp_path / "demucs-writable-fresh",
                     roots["fallback_root"])
    expect("model_root", roots["model_root"], roots["model_root"], c, spec)

    bare = tmp_path / "demucs-model-bare"
    bare.mkdir()
    c, spec = answer(bare, tmp_path / "demucs-writable-fresh2", roots["fallback_root"])
    expect("fallback_root", roots["fallback_root"], roots["fallback_root"], c, spec)


def test_roots_are_ordered_writable_model_fallback_with_weights_skipping_writable(tmp_path):
    """R-T10-precedence stated as data: the list `_find()` walks, in order.

    The behavioural tests above prove the order *bites*; this one names it, so a
    reorder fails twice rather than quietly landing in a branch nobody exercises.
    """
    roots = _three_roots(tmp_path, "order")
    c = catalog.Catalog(roots["model_root"], roots["writable_dir"],
                        fallback_root=roots["fallback_root"])
    assert c._roots() == [(catalog.RootRole.WRITABLE_DIR, roots["writable_dir"]),
                          (catalog.RootRole.MODEL_ROOT, roots["model_root"]),
                          (catalog.RootRole.FALLBACK_ROOT, roots["fallback_root"])]
    assert c._roots(weights=True) == [(catalog.RootRole.MODEL_ROOT, roots["model_root"]),
                                      (catalog.RootRole.FALLBACK_ROOT,
                                       roots["fallback_root"])]
    no_fallback = catalog.Catalog(roots["model_root"], roots["writable_dir"],
                                  fallback_root=None)
    assert [role for role, _ in no_fallback._roots()] == [catalog.RootRole.WRITABLE_DIR,
                                                          catalog.RootRole.MODEL_ROOT]


def test_provenance_distinguishes_the_four_answering_sources(tmp_path):
    # (a) tier 2, baked table in the *installed* root
    installed = _catalog(tmp_path, model_root=MODEL_ROOT, fallback_root=None)
    spec = installed.resolve("1_HP-UVR")
    assert spec.provenance.tier is catalog.Tier.BAKED_TABLE
    assert spec.provenance.root_role is catalog.RootRole.MODEL_ROOT
    assert spec.provenance.root == MODEL_ROOT
    assert spec.provenance.source == MODEL_ROOT / VR_TABLE
    assert spec.hash == catalog.hash_of(spec.weights_path)

    # (b) tier 2, baked table in the *repo* root -- a different root answering the
    # same arch, which is what "a fresh clone resolves fewer models" is made of.
    repo = _catalog(tmp_path, model_root=REPO_MODELS, fallback_root=None)
    repo_spec = repo.resolve("UVR-DeNoise-Lite")
    assert repo_spec.provenance.tier is catalog.Tier.BAKED_TABLE
    assert repo_spec.provenance.root_role is catalog.RootRole.MODEL_ROOT
    assert repo_spec.provenance.root == REPO_MODELS
    assert repo_spec.provenance.source == REPO_MODELS / VR_TABLE

    # (c) tier 1, per-model JSON in the weights root
    name = "Synthetic-UVR"
    root, digest = _vr_root(tmp_path, name)
    _write_json(root / "VR_Models/model_data" / f"{digest}.json", _params("Other"))
    c_root = _catalog(tmp_path, model_root=root, fallback_root=None)
    spec = c_root.resolve(name)
    assert spec.provenance.tier is catalog.Tier.PER_MODEL_JSON
    assert spec.provenance.root_role is catalog.RootRole.MODEL_ROOT
    assert spec.model_data == _params("Other")

    # (d) tier 2 served by the fallback root: a root that ships no tables at all
    # (first file wins -- the plan's consequence 1 -- so an empty table in
    # `model_root` is an answer, not a miss to fall through)
    bare = tmp_path / "bare"
    bare.mkdir()
    c_fb = catalog.Catalog(bare, tmp_path / "w2", fallback_root=MODEL_ROOT)
    spec = c_fb.resolve("1_HP-UVR")
    assert spec.provenance.tier is catalog.Tier.BAKED_TABLE
    assert spec.provenance.root_role is catalog.RootRole.FALLBACK_ROOT
    assert spec.provenance.source == MODEL_ROOT / VR_TABLE


def test_tier2_entry_counts_are_read_from_the_files(tmp_path):
    """No number in this test came from prose; each is re-read from its JSON."""
    expected = {
        (MODEL_ROOT, VR_TABLE): len(_json(MODEL_ROOT / VR_TABLE)),
        (MODEL_ROOT, MDX_TABLE): len(_json(MODEL_ROOT / MDX_TABLE)),
        (MODEL_ROOT, MDX_NAMES): len(_json(MODEL_ROOT / MDX_NAMES)),
        (MODEL_ROOT, DEMUCS_NAMES): len(_json(MODEL_ROOT / DEMUCS_NAMES)),
        (REPO_MODELS, VR_TABLE): len(_json(REPO_MODELS / VR_TABLE)),
        (REPO_MODELS, MDX_TABLE): len(_json(REPO_MODELS / MDX_TABLE)),
        (REPO_MODELS, MDX_NAMES): len(_json(REPO_MODELS / MDX_NAMES)),
        (REPO_MODELS, DEMUCS_NAMES): len(_json(REPO_MODELS / DEMUCS_NAMES)),
    }
    for root, rel in expected:
        c = catalog.Catalog(root, tmp_path / f"w-{root.name}-{len(rel)}", fallback_root=None)
        kind = "names" if rel.endswith("model_name_mapper.json") else "params"
        arch = {"VR_Models": "vr", "MDX_Net_Models": "mdx", "Demucs_Models": "demucs"}[rel.split("/")[0]]
        assert len(c.baked_table(arch, kind)) == expected[(root, rel)]

    # and the disagreement the brief measured is real, in this machine's files:
    def _size_diff_message(installed_rel: str, label: str) -> str:
        return (
            f"expected {installed_rel} to hold more entries in the installed root "
            f"({MODEL_ROOT}) than in the repo tree ({REPO_MODELS / installed_rel}) — "
            f"the plan's 'the repo tree's copies are a stale subset'. If it is empty, "
            f"the installed root is not newer than the clone: check UVR_MODEL_ROOT "
            f"(currently {os.environ.get('UVR_MODEL_ROOT', '<unset>')}, which resolves "
            f"to MODEL_ROOT above) and re-run with it pointing at the installed "
            f"weights dir; {label} is otherwise measuring a machine that has nothing "
            f"to compare.")

    assert len(_json(MODEL_ROOT / MDX_TABLE)) > len(_json(REPO_MODELS / MDX_TABLE)), \
        _size_diff_message(MDX_TABLE, "the MDX params table")
    assert len(_json(MODEL_ROOT / MDX_NAMES)) > len(_json(REPO_MODELS / MDX_NAMES)), \
        _size_diff_message(MDX_NAMES, "the MDX names table")


def test_uninstalled_names_is_declared_minus_present_in_both_directions(tmp_path):
    """Finding #5: `uninstalled_names` had zero references outside `catalog.py`.

    Non-vacuous in both directions: the same catalog answers the missing name while
    the weight is absent and `[]` once it is dropped in, so the first assert cannot
    come from a listing that is simply always empty (a table that declared nothing,
    or a `names()` that saw nothing) and the second cannot come from a method that
    always reports everything.
    """
    root = tmp_path / "unin"
    declared = {"present-model.ckpt": "Present Model", "absent-model.ckpt": "Absent Model"}
    _write_json(root / MDX_NAMES, declared)
    (root / "MDX_Net_Models" / "present-model.ckpt").write_bytes(b"p")

    c = catalog.Catalog(root, tmp_path / "appdata", fallback_root=None)
    assert c.known_names("mdx") == sorted(declared), "the table declares both"
    assert c.names("mdx") == ["present-model.ckpt"], "the tree holds one weight"
    assert c.uninstalled_names("mdx") == ["absent-model.ckpt"]

    (root / "MDX_Net_Models" / "absent-model.ckpt").write_bytes(b"a")
    assert c.names("mdx") == sorted(declared), "the second weight is now on disk"
    assert c.uninstalled_names("mdx") == [], \
        "installed is not 'declared' here: a present weight must drop out"


def test_uninstalled_names_is_nonempty_on_a_fresh_clone(tmp_path):
    """The shipped tables name models a clone cannot run — no count involved."""
    repo = catalog.Catalog(REPO_MODELS, tmp_path / "appdata", fallback_root=None)
    missing = repo.uninstalled_names("mdx")
    assert missing, (
        f"every name in {REPO_MODELS / MDX_NAMES} was found as a weight under "
        f"{REPO_MODELS / 'MDX_Net_Models'}; this machine holds a full MDX set, so "
        f"the 'a fresh clone cannot run these' premise of the test is gone here — "
        f"run it where the installed weights are not all in the repo tree")
    assert set(missing) <= set(repo.known_names("mdx")), "declared, by definition"
    assert not set(missing) & set(repo.names("mdx")), "and absent from the tree"


def test_repo_tree_resolves_fewer_models_and_says_so(tmp_path):
    """Installed-only names are visible from the installed root, absent from a clone."""
    installed_names = set(_json(MODEL_ROOT / MDX_NAMES))
    repo_names = set(_json(REPO_MODELS / MDX_NAMES))
    missing = sorted(installed_names - repo_names)
    assert missing, "the stale-subset finding is the point of this test"

    installed = _catalog(tmp_path, model_root=MODEL_ROOT, fallback_root=None)
    repo = _catalog(tmp_path, model_root=REPO_MODELS, fallback_root=None)
    assert installed.known_names("mdx") == sorted(installed_names)
    assert repo.known_names("mdx") == sorted(repo_names)

    name = missing[0]
    with pytest.raises(catalog.MissingWeightError):
        repo.resolve(name, arch="mdx")
    assert name not in repo.known_names("mdx") and name in installed.known_names("mdx")

    # first file wins, exactly as consequence 1 of the plan prescribes: a root that
    # ships its own (stale) table is never overridden by the fallback, which is why
    # the writable refresh location is searched *before* both roots.
    bare = tmp_path / "bare-root"
    bare.mkdir()
    merged = catalog.Catalog(bare, tmp_path / "w-merge", fallback_root=MODEL_ROOT)
    assert merged.known_names("mdx") == sorted(installed_names)
    merged_repo = catalog.Catalog(bare, tmp_path / "w-merge2", fallback_root=REPO_MODELS)
    assert merged_repo.known_names("mdx") == sorted(repo_names)


# --------------------------------------------------------------------------- #
# 3. writable_dir wins, and writes never touch the weights root
# --------------------------------------------------------------------------- #

def test_writable_tier1_beats_the_weights_root_tier1(tmp_path):
    name = "Conflict-UVR"
    root, digest = _vr_root(tmp_path, name)
    weights_answer = _params("Vocals")
    writable_answer = _params("Other")
    _write_json(root / "VR_Models/model_data" / f"{digest}.json", weights_answer)

    writable = tmp_path / "appdata"
    _write_json(writable / "VR_Models/model_data" / f"{digest}.json", writable_answer)
    c = catalog.Catalog(root, writable, fallback_root=None)

    spec = c.resolve(name)
    assert spec.model_data == writable_answer, "R-T10-writable: writable_dir wins on conflict"
    assert spec.provenance.root_role is catalog.RootRole.WRITABLE_DIR
    assert spec.provenance.source == writable / "VR_Models/model_data" / f"{digest}.json"


def test_write_params_lands_in_writable_dir_and_creates_missing_dirs(tmp_path):
    """`mkdir -p` is mandatory: `UVR.py` has no makedirs for any model dir."""
    root = tmp_path / "weights"
    (root / "VR_Models").mkdir(parents=True)
    (root / "VR_Models/model_data/model_data.json").parent.mkdir(parents=True, exist_ok=True)
    _write_json(root / VR_TABLE, {})
    before = sorted(p.relative_to(root) for p in root.rglob("*") if p.is_file())

    writable = tmp_path / "appdata/model_data"
    c = catalog.Catalog(root, writable, fallback_root=None)
    written = c.write_params("vr", "0" * 32, _params("Bass"))
    assert written == writable / "VR_Models/model_data" / f"{'0' * 32}.json"
    assert written.exists()
    assert _json(written) == _params("Bass")

    # a directory that does not exist anywhere yet (the official build's Apollo dir)
    apollo_write = c.write_params("apollo", "1" * 32, {"config_yaml": "apollo.yaml"})
    assert apollo_write.parent.is_dir()
    assert apollo_write.exists()

    after = sorted(p.relative_to(root) for p in root.rglob("*") if p.is_file())
    assert before == after, "a tier-1 write must never touch the weights root"


def test_constructor_refuses_a_writable_dir_inside_a_version_controlled_root(tmp_path):
    """Consequence 2: tier-1 writes must land outside any git-tracked path."""
    inside = REPO_MODELS / "VR_Models" / "model_data"
    with pytest.raises(catalog.WritableLocationError):
        catalog.Catalog(MODEL_ROOT, inside)
    nested = tmp_path / "weights"
    nested.mkdir()
    with pytest.raises(catalog.WritableLocationError):
        catalog.Catalog(nested, nested / "model_data")  # inside model_root
    # and a legal location still constructs
    _catalog(tmp_path)


# --------------------------------------------------------------------------- #
# 4. tier 3 returns without blocking
# --------------------------------------------------------------------------- #

def test_tier3_returns_a_needs_user_params_spec_without_blocking(tmp_path):
    name = "Unknown-UVR"
    root, digest = _vr_root(tmp_path, name)  # weight present, no answer in tier 1/2
    writable = tmp_path / "appdata"
    c = catalog.Catalog(root, writable, fallback_root=None)

    spec = c.resolve(name)  # must NOT raise, must NOT open a dialog
    assert spec.needs_user_params is True
    assert spec.model_data is None
    assert spec.hash == digest
    assert spec.provenance.tier is catalog.Tier.NEEDS_USER_PARAMS
    message = spec.missing_message
    assert digest in message
    assert str(writable / "VR_Models/model_data" / f"{digest}.json") in message
    assert name in message


def test_tier3_required_raises_a_printable_typed_error(tmp_path):
    name = "Unknown-UVR"
    root, digest = _vr_root(tmp_path, name)
    c = catalog.Catalog(root, tmp_path / "appdata", fallback_root=None)
    with pytest.raises(catalog.UnknownModelParamsError) as excinfo:
        c.resolve(name, required=True)
    assert isinstance(excinfo.value, catalog.CatalogError)
    assert digest in str(excinfo.value)
    assert f"{digest}.json" in str(excinfo.value)


def test_missing_weight_raises_rather_than_fabricating_a_spec(tmp_path):
    root = tmp_path / "empty-root"
    _write_json(root / VR_TABLE, {"a" * 32: _params()})
    c = catalog.Catalog(root, tmp_path / "appdata", fallback_root=None)
    # no arch declared and nothing in any table declares the name -> unknown model
    with pytest.raises(catalog.UnknownModelNameError):
        c.resolve("Not-Installed-Model")
    # arch known, weight file absent -> the honest "not on disk" failure
    with pytest.raises(catalog.MissingWeightError):
        c.resolve("Totally-Unknown-Model", arch="vr")


# --------------------------------------------------------------------------- #
# 5. Apollo's shape
# --------------------------------------------------------------------------- #

def test_apollo_resolves_via_tier1_only(tmp_path):
    """Apollo has exactly one source: `<arch>/model_data/<hash>.json` (`:252`)."""
    root = tmp_path / "weights"
    weight = _fake_weight(root / "Apollo_Models" / "BS-Roformer-Viperx-1297.ckpt")
    digest = catalog.hash_of(weight)
    _write_json(root / "Apollo_Models/model_data" / f"{digest}.json",
                {"config_yaml": "apollo.yaml"})
    (root / "Apollo_Models/model_configs").mkdir(parents=True, exist_ok=True)
    _write_json(root / "Apollo_Models/model_configs/apollo.yaml", {})  # path only

    c = catalog.Catalog(root, tmp_path / "appdata", fallback_root=None)
    spec = c.resolve("BS-Roformer-Viperx-1297.ckpt", arch="apollo")
    assert spec.provenance.tier is catalog.Tier.PER_MODEL_JSON
    assert spec.config_path == root / "Apollo_Models/model_configs/apollo.yaml"
    # upstream's ApolloModelData never sets a process method -- not invented here
    assert spec.process_method is None


def test_apollo_has_no_baked_table_and_that_is_not_an_empty_dict(tmp_path):
    c = _catalog(tmp_path)
    assert c.has_baked_table("apollo", "params") is False
    with pytest.raises(catalog.NoBakedTableError):
        c.baked_table("apollo", "params")
    # the two archs that DO have a params table answer with dicts
    assert c.has_baked_table("vr", "params") is True
    assert c.has_baked_table("mdx", "params") is True
    assert c.has_baked_table("demucs", "params") is False


def test_repo_ships_only_three_apollo_tier1_files(tmp_path):
    """The committed Apollo JSON *is* the entire Apollo params dataset."""
    files = sorted(p.name for p in (REPO_MODELS / "Apollo_Models/model_data").glob("*.json"))
    assert len(files) == 3
    for name in files:
        assert set(_json(REPO_MODELS / "Apollo_Models/model_data" / name)) == {"config_yaml"}
    c = catalog.Catalog(REPO_MODELS, tmp_path / "appdata", fallback_root=None)
    assert c.has_baked_table("apollo", "params") is False


# --------------------------------------------------------------------------- #
# 6. blocked archs refuse, not simulate (fork policy)
# --------------------------------------------------------------------------- #

def test_blocked_archs_refuse_rather_than_simulate(tmp_path):
    blocked = {"mdx_c", "onnx", "roformer"}
    assert set(catalog.BLOCKED_ARCHS) == blocked
    # pins.json is the harness's policy record -- read-only here, never written
    pins = _json(REPO / "tests/golden/pins.json")
    assert set(pins["archs_blocked"]) == blocked
    assert set(pins["archs_covered"]) <= set(catalog.SUPPORTED_ARCHS)

    c = _catalog(tmp_path)
    for arch in sorted(blocked):
        assert c.is_blocked(arch)
        with pytest.raises(catalog.BlockedArchError) as excinfo:
            c.names(arch)
        assert arch in str(excinfo.value)
        with pytest.raises(catalog.BlockedArchError):
            c.resolve("anything", arch=arch)
    assert c.is_blocked("vr") is None


def test_unknown_arch_is_rejected(tmp_path):
    c = _catalog(tmp_path)
    with pytest.raises(catalog.UnknownArchError):
        c.resolve("1_HP-UVR", arch="not-an-arch")
    with pytest.raises(catalog.UnknownArchError):
        c.names("not-an-arch")


# --------------------------------------------------------------------------- #
# names() / secondary decision methods
# --------------------------------------------------------------------------- #

def test_names_lists_resolvable_models_per_arch(tmp_path):
    c = _catalog(tmp_path)
    vr_names = c.names("vr")
    assert "1_HP-UVR" in vr_names
    assert all(isinstance(n, str) for n in vr_names)
    demucs = c.names("demucs")
    assert demucs, "the installed root has demucs weights"
    assert "mdx_c" not in c.archs()


def test_secondary_slot_for_ports_the_stem_branches():
    assert catalog.secondary_slot_for(VR_ARCH_TYPE, VOCAL_STEM) == "voc_inst_secondary_model"
    assert catalog.secondary_slot_for(MDX_ARCH_TYPE, INST_STEM) == "voc_inst_secondary_model"
    assert catalog.secondary_slot_for(MDX_ARCH_TYPE, OTHER_STEM) == "other_secondary_model"
    assert catalog.secondary_slot_for(MDX_ARCH_TYPE, "No Other") == "other_secondary_model"
    assert catalog.secondary_slot_for(DEMUCS_ARCH_TYPE, DRUM_STEM) == "drums_secondary_model"
    assert catalog.secondary_slot_for(DEMUCS_ARCH_TYPE, BASS_STEM) == "bass_secondary_model"
    assert catalog.secondary_slot_for(VR_ARCH_TYPE, "nonsense") is None


def test_secondary_for_ports_all_three_decision_methods(tmp_path):
    c = _catalog(tmp_path)
    main = c.resolve("1_HP-UVR")
    assert catalog.secondary_slot_for(main.process_method, main.primary_stem) \
        == "voc_inst_secondary_model"

    # UVR.py:7075 -- "No Model Selected" and None both mean "no secondary model"
    assert c.secondary_for(main, NO_MODEL) is None
    assert c.secondary_for(main, None) is None
    assert c.secondary_for(main, "") is None

    picked = c.secondary_for(main, "1_HP-UVR")
    assert isinstance(picked, catalog.ModelSpec)
    assert picked.weights_path.exists()

    # UVR.py:7081-7082 -- an invalid secondary is dropped, not raised through
    assert c.secondary_for(main, "Model-That-Is-Not-Installed") is None

    # process_determine_demucs_pre_proc_model (UVR.py:7088)
    assert c.demucs_pre_proc_for(NO_MODEL) is None
    assert c.demucs_pre_proc_for("955717e8-8726e21a.th", is_activated=False) is None
    pre = c.demucs_pre_proc_for("955717e8-8726e21a.th")
    assert pre is not None and pre.arch == "demucs"

    # process_determine_vocal_split_model (UVR.py:7103)
    assert c.vocal_split_for(NO_MODEL) is None
    assert c.vocal_split_for("1_HP-UVR", is_activated=False) is None
    split = c.vocal_split_for("1_HP-UVR")
    assert split is not None and split.arch == "vr"


def test_demucs_spec_carries_no_fabricated_params_table(tmp_path):
    c = _catalog(tmp_path)
    spec = c.resolve("955717e8-8726e21a.th", arch="demucs")
    assert spec.process_method == DEMUCS_ARCH_TYPE
    assert spec.needs_user_params is False
    assert spec.provenance.tier is catalog.Tier.BAKED_TABLE
    assert spec.primary_stem is None  # UVR.py:737 derives stems from the user's choice


# --------------------------------------------------------------------------- #
# 7. tier-2 writes: Catalog.write_table + an honest table cache
#    (fix round 1, findings #2 + #3 — ruling R-T10-tier2write)
# --------------------------------------------------------------------------- #

def _under(path: Path, base: Path) -> bool:
    try:
        Path(path).relative_to(base)
    except ValueError:
        return False
    return True


def test_write_table_returns_the_writable_path_and_lands_there(tmp_path):
    """Finding #2: Task 12's refresh needs a tier-2 write API, not a re-derived path.

    `write_params` owns tier-1 `<hash>.json` only, so a refresh of a *baked table* had
    no owner: the caller would have to rebuild `<writable_dir>/<arch params_table>` by
    hand from `ArchSpec` fields. That path convention is the catalog's, so the catalog
    exposes the write and returns the path it wrote.
    """
    root = tmp_path / "weights"
    baked = {"a" * 32: _params("Vocals")}
    _write_json(root / VR_TABLE, baked)
    before = sorted(p.relative_to(root) for p in root.rglob("*") if p.is_file())

    writable = tmp_path / "appdata" / "model_data"   # does not exist yet: mkdir -p
    c = catalog.Catalog(root, writable, fallback_root=None)
    refreshed = {"a" * 32: _params("Other"), "b" * 32: _params("Bass")}
    target = c.write_table("vr", "params", refreshed)

    assert target == writable / VR_TABLE
    assert target.is_file()
    assert _json(target) == refreshed
    assert _under(target, writable), "tier-2 writes go to writable_dir and nowhere else"
    assert not _under(target, root)

    after = sorted(p.relative_to(root) for p in root.rglob("*") if p.is_file())
    assert before == after, "a tier-2 write must never touch the weights root"


def test_write_table_refresh_is_seen_by_the_same_catalog_instance(tmp_path):
    """Findings #2 + #3: refresh-then-resolve in ONE process — exactly Task 12's shape.

    The memo was keyed `(arch, kind, "")`, so it never recorded which root answered and
    nothing ever cleared it: once `resolve()` had loaded the installed table, a table
    written into `writable_dir` afterwards was invisible to every later `resolve()` for
    the life of the process. Task 12's refresh runs inside the GUI process that already
    resolved models, so this is the feature, not a corner of it.
    """
    c = _catalog(tmp_path)
    first = c.resolve("1_HP-UVR")
    assert first.provenance.tier is catalog.Tier.BAKED_TABLE
    assert first.model_data == _json(MODEL_ROOT / VR_TABLE)[first.hash]

    refreshed = {first.hash: _params("Drums")}
    returned = c.write_table("vr", "params", refreshed)
    assert returned == c.writable_dir / VR_TABLE

    second = c.resolve("1_HP-UVR")
    assert second.model_data == _params("Drums"), "the refresh must answer, not the memo"
    assert second.primary_stem == "Drums"


def test_table_memo_follows_an_external_edit_of_the_table_file(tmp_path):
    """Finding #3: the cache key must carry the file's stat, not just (arch, kind).

    This bypasses the write API entirely (`Path.write_text`), so no clear-on-write can
    be what makes it pass — the memo has to notice that the file underneath it changed,
    which is the same `(resolved path, st_size, st_mtime_ns)` key `hash_of` already uses.
    """
    root = tmp_path / "weights"
    original = {"a" * 32: _params("Vocals")}
    table_path = root / VR_TABLE
    _write_json(table_path, original)
    c = catalog.Catalog(root, tmp_path / "appdata", fallback_root=None)

    assert c.baked_table("vr") == original          # loads and memoizes

    changed = {"a" * 32: _params("Bass"), "b" * 32: _params("Other")}
    table_path.write_text(json.dumps(changed), encoding="utf-8")
    stat = table_path.stat()
    os.utime(table_path, (stat.st_atime, stat.st_mtime + 10))  # deterministic bump

    assert c.baked_table("vr") == changed, "a stale table is a wrong resolution"
    assert c.baked_table("vr") == changed          # stable across a re-read, too


def test_mutating_a_spec_does_not_poison_the_table_cache(tmp_path):
    """Finding #3: `baked_table` must not hand out the cached object by reference.

    `resolve()` stores the table's *own* sub-dict into `ModelSpec.model_data`
    (`catalog.py:570-572`), so with the cached object returned directly one caller
    editing its spec edits the catalog's cache for the whole process.
    """
    c = _catalog(tmp_path)
    first = c.resolve("1_HP-UVR")
    expected = _json(MODEL_ROOT / VR_TABLE)[first.hash]
    assert first.model_data == expected

    first.model_data["primary_stem"] = "Mutated"

    second = c.resolve("1_HP-UVR")
    assert second.model_data["primary_stem"] == expected["primary_stem"], \
        "the cache answered with a caller's mutation"
    assert c.baked_table("vr")[first.hash] == expected
    assert c.baked_table("vr")[first.hash]["primary_stem"] != "Mutated"
    # the strongest form of the same rule: the cached object never leaves the cache,
    # so no reference a caller holds can be the reference the next reader gets.
    assert c.baked_table("vr") is not c.baked_table("vr")
    assert second.model_data is not first.model_data


def test_write_table_refuses_an_arch_kind_that_has_no_such_table(tmp_path):
    """Finding #2: never write to a path the spec does not declare.

    Apollo and Demucs declare no params table and VR no names table (`ARCHS`), so
    `write_table` there is meaningless; it raises the same `NoBakedTableError`
    `baked_table` raises for the identical condition rather than inventing a file.
    """
    c = _catalog(tmp_path)
    for arch, kind in (("apollo", "params"), ("vr", "names"), ("demucs", "params")):
        assert c.has_baked_table(arch, kind) is False
        with pytest.raises(catalog.NoBakedTableError) as excinfo:
            c.write_table(arch, kind, {"a" * 32: _params()})
        assert arch in str(excinfo.value) and kind in str(excinfo.value)

    assert list(c.writable_dir.rglob("*")) == [], "a refused write must create nothing"


def test_writable_baked_table_overrides_the_weights_root_table(tmp_path):
    """Finding #2's other half: the writable table is the one that answers *content*.

    First-file-wins already prefers `writable_dir` (`_roots()`), and that precedence is
    finding #1's business; what a reorder could hide is a resolution that reports the
    writable path while returning the memoized installed table. Pinned on both halves.
    """
    c = _catalog(tmp_path)
    digest = c.resolve("1_HP-UVR").hash
    installed = _json(MODEL_ROOT / VR_TABLE)[digest]
    refreshed = {digest: _params("Bass")}
    assert refreshed[digest] != installed, "the pin only bites if the two answers differ"
    target = c.write_table("vr", "params", refreshed)

    spec = c.resolve("1_HP-UVR")
    assert spec.provenance.tier is catalog.Tier.BAKED_TABLE
    assert spec.provenance.root_role is catalog.RootRole.WRITABLE_DIR
    assert spec.provenance.root == c.writable_dir
    assert spec.provenance.source == target == c.writable_dir / VR_TABLE
    assert spec.model_data == _params("Bass")


# --------------------------------------------------------------------------- #
# 8. the demucs pre-proc arch pin (fix round 1, finding #4)
# --------------------------------------------------------------------------- #

def test_demucs_pre_proc_for_does_not_pin_the_arch(tmp_path):
    """Finding #4: the pre-proc slot is stem-filtered, not arch-restricted.

    `UVR.py:4540` fills it from `model_list(..., is_no_demucs=True)`, whose branch at
    `UVR.py:2663-2666` filters by *stem*, and `UVR.py:6109`'s ensemble list is
    `vr + mdx + demucs` — so VR and MDX names legitimately arrive here. With
    `arch="demucs"` the lookup only searches `Demucs_Models/…`, raises
    `MissingWeightError`, and the `except CatalogError: return None` swallows it: the
    fork loses the pre-process model whenever `primary_stem` is set.

    Only the post-fix behaviour is asserted here; the "before" picture is re-created
    by `tools/task10_red_controls.py`, which replays the pin against the shipped
    class instead of leaving a dead branch in a live test.
    """
    c = _catalog(tmp_path)
    assert c.resolve("1_HP-UVR").arch == "vr"

    spec = c.demucs_pre_proc_for("1_HP-UVR")
    assert spec is not None and spec.arch == "vr"


def test_demucs_pre_proc_for_has_no_primary_stem_parameter():
    """Finding #4: drop the dead parameter, do not leave it in place.

    A parameter that never affects the result but *looks* load-bearing is worse than the
    pin itself: Task 11's caller would pass `primary_stem` believing it selects the
    architecture. Upstream's `ModelData(...)` call at `UVR.py:7094-7096` passes no arch.
    """
    params = inspect.signature(catalog.Catalog.demucs_pre_proc_for).parameters
    assert "primary_stem" not in params
    assert set(params) == {"self", "requested", "is_activated"}


def test_a_real_demucs_pre_proc_name_still_resolves_without_the_pin(tmp_path):
    """Item 8: removing the pin must not break the case it was written for.

    Same guarantee as `test_secondary_for_ports_all_three_decision_methods` :525-527,
    pinned on its own: a genuine Demucs weight still answers `demucs` once the
    architecture is left to the name and the file location.
    """
    c = _catalog(tmp_path)
    assert c.demucs_pre_proc_for(NO_MODEL) is None
    pre = c.demucs_pre_proc_for("955717e8-8726e21a.th")
    assert pre is not None and pre.arch == "demucs"
    assert pre.process_method == DEMUCS_ARCH_TYPE
    assert c.demucs_pre_proc_for("955717e8-8726e21a.th", is_activated=False) is None

