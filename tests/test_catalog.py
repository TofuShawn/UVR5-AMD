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
import json
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


def test_resolve_reports_root_and_tier_for_every_source(tmp_path):
    c = _catalog(tmp_path)
    spec = c.resolve("1_HP-UVR")
    assert spec.weights_path.exists()
    assert spec.process_method == VR_ARCH_TYPE


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
    assert len(_json(MODEL_ROOT / MDX_TABLE)) > len(_json(REPO_MODELS / MDX_TABLE))
    assert len(_json(MODEL_ROOT / MDX_NAMES)) > len(_json(REPO_MODELS / MDX_NAMES))


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
