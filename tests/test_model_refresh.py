"""Task 12: `model_refresh.py` — the retained, user-invoked table refresh.

Three promises, each of which is a failure mode this file exists to catch:

1. The four tables land on the four paths **`Catalog`** declares, via
   `Catalog.write_table(arch, kind, table)` — the method Task 10 added precisely so this
   task would not have to re-derive `<writable_dir>/VR_Models/model_data/model_data.json`
   by hand.
2. A refresh is *seen* by the same process that performed it (`baked_table`,
   `known_names`, and therefore `resolve`) — Task 10 finding #3's contract.
3. Nothing is written under `model_root`, the fallback root, or the repo `models/` tree:
   proved by a `before == after` file listing, the pattern `tests/test_catalog.py` uses.

**No test in this file performs a real network fetch.** An autouse fixture replaces
`urllib.request.urlopen` with a raiser, so even a test that forgot to inject a fake fails
loudly instead of touching the network; every fetch here is canned JSON against `tmp_path`.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pytest

import catalog
import model_refresh

REPO = Path(__file__).resolve().parents[1]
REPO_MODELS = REPO / "models"

# The four Catalog-owned relative table paths, written out here so a change to
# catalog.ARCHS shows up as a red test rather than as a silently moved file.
EXPECTED_REL = {
    ("vr", "params"): "VR_Models/model_data/model_data.json",
    ("mdx", "params"): "MDX_Net_Models/model_data/model_data.json",
    ("mdx", "names"): "MDX_Net_Models/model_data/model_name_mapper.json",
    ("demucs", "names"): "Demucs_Models/model_data/model_name_mapper.json",
}


@pytest.fixture(autouse=True)
def no_real_network(monkeypatch):
    """Whatever a test forgets, nothing in this module may reach the network."""

    def forbidden(*args, **kwargs):
        raise AssertionError(f"a test tried to fetch for real: {args!r} {kwargs!r}")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)


def _canned() -> dict[str, dict]:
    """One distinguishable table per URL, keyed by the (arch, kind) it must reach."""
    return {
        s.url: {f"table-{s.arch}-{s.kind}": {"marker": s.moved_from}}
        for s in model_refresh.TABLES
    }


def _fake_fetch(canned: dict[str, dict], calls: list[str]):
    def fetch(url: str) -> dict:
        calls.append(url)
        return json.loads(json.dumps(canned[url]))  # a fresh object per call, like a real GET

    return fetch


def _roots(tmp_path: Path) -> tuple[Path, Path]:
    model_root = tmp_path / "weights"
    writable = tmp_path / "appdata" / "model_data"
    for rel in ("VR_Models/model_data", "MDX_Net_Models/model_data",
                "Demucs_Models/model_data"):
        (model_root / rel).mkdir(parents=True, exist_ok=True)
    writable.mkdir(parents=True, exist_ok=True)
    return model_root, writable


def _listing(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


# --------------------------------------------------------------------------- #
# 1. the four tables land on the four Catalog-owned paths
# --------------------------------------------------------------------------- #

def test_refresh_writes_all_four_tables_to_catalog_paths(tmp_path):
    model_root, writable = _roots(tmp_path)
    cat = catalog.Catalog(model_root, writable, fallback_root=None)
    canned, calls = _canned(), []
    printed: list[str] = []

    targets = model_refresh.refresh(cat, confirm=True, fetch=_fake_fetch(canned, calls),
                                    printer=printed.append)

    assert calls == [s.url for s in model_refresh.TABLES], (
        "the refresh must fetch exactly the four table URLs it moved from gui_data/constants.py")
    assert len(targets) == 4
    for source in model_refresh.TABLES:
        expected = writable / EXPECTED_REL[(source.arch, source.kind)]
        assert expected in targets, f"{source.label()} did not land on the Catalog-owned path"
        wrote = json.loads(expected.read_text(encoding="utf-8"))
        assert wrote == canned[source.url], f"{source.label()} wrote the wrong table"
        # announced both the URL it contacted and the path it wrote, per the brief
        assert any(source.url in line for line in printed), printed
        assert any(str(expected) in line for line in printed), printed


def test_write_is_the_catalogs_own_method_not_a_hand_derived_path(tmp_path):
    """If this module re-derived a path instead of calling `write_table`, the memo would
    not be cleared and `baked_table` in the same process would keep answering the stale
    table — so this is the same assertion test 2 makes, one level down."""
    model_root, writable = _roots(tmp_path)
    cat = catalog.Catalog(model_root, writable, fallback_root=None)
    seen: list[tuple[str, str]] = []
    original = cat.write_table

    def spy(arch, kind, table):
        seen.append((arch, kind))
        return original(arch, kind, table)

    cat.write_table = spy  # instance-level: the module must go through this door
    model_refresh.refresh(cat, confirm=True, fetch=_fake_fetch(_canned(), []))
    assert seen == [(s.arch, s.kind) for s in model_refresh.TABLES], (
        f"refresh did not call Catalog.write_table with the four coordinates: {seen}")


# --------------------------------------------------------------------------- #
# 2. refresh-then-resolve in the same process (Task 10 finding #3)
# --------------------------------------------------------------------------- #

def test_baked_table_and_known_names_see_the_refresh_in_the_same_process(tmp_path):
    model_root, writable = _roots(tmp_path)
    # A stale shipped table exists in the weights root; the writable refresh must win.
    stale = {"old-hash": {"vr_model_param": "4band_44100", "primary_stem": "Vocals"}}
    (model_root / EXPECTED_REL[("vr", "params")]).write_text(json.dumps(stale), encoding="utf-8")
    (model_root / EXPECTED_REL[("mdx", "names")]).write_text(
        json.dumps({"Old-MDX-Model.onnx": {}}), encoding="utf-8")

    cat = catalog.Catalog(model_root, writable, fallback_root=None)
    assert cat.baked_table("vr", "params") == stale
    assert cat.known_names("mdx") == ["Old-MDX-Model.onnx"]

    fresh_vr = {"a" * 32: {"vr_model_param": "4band_v3", "primary_stem": "Vocals"}}
    fresh_mdx_names = {"UVR-MDX-NET-Inst_HQ_4.onnx": {}, "BS-Roformer-Viperx-1297.ckpt": {}}
    canned = {s.url: (fresh_vr if (s.arch, s.kind) == ("vr", "params")
                      else fresh_mdx_names if (s.arch, s.kind) == ("mdx", "names")
                      else {}) for s in model_refresh.TABLES}
    model_refresh.refresh(cat, confirm=True, fetch=_fake_fetch(canned, []))

    assert cat.baked_table("vr", "params") == fresh_vr, (
        "the table cache was not invalidated by the refresh — the whole point of writing "
        "through Catalog.write_table instead of open()-ing a path by hand")
    assert cat.known_names("mdx") == sorted(fresh_mdx_names)

    # ...and a model that only the *refreshed* table knows about now resolves on tier 2.
    weight = model_root / "VR_Models" / "1_Refreshed-Only.pth"
    weight.write_bytes(b"refresh-then-resolve")  # under the 10 MB tail window
    digest = catalog.hash_of(weight)
    canned2 = {s.url: (dict(fresh_vr, **{digest: {"vr_model_param": "4band_44100",
                                                  "primary_stem": "Vocals"}})
                       if (s.arch, s.kind) == ("vr", "params") else {})
               for s in model_refresh.TABLES}
    model_refresh.refresh(cat, confirm=True, fetch=_fake_fetch(canned2, []))
    spec = cat.resolve("1_Refreshed-Only")
    assert spec.provenance.tier is catalog.Tier.BAKED_TABLE
    assert spec.model_data["primary_stem"] == "Vocals"


# --------------------------------------------------------------------------- #
# 3. nothing is written into the weights root or the repo models/ tree
# --------------------------------------------------------------------------- #

def test_refresh_writes_nothing_under_model_root_or_the_repo_models_tree(tmp_path):
    model_root, writable = _roots(tmp_path)
    cat = catalog.Catalog(model_root, writable, fallback_root=REPO_MODELS)
    before_root, before_repo = _listing(model_root), _listing(REPO_MODELS)

    targets = model_refresh.refresh(cat, confirm=True, fetch=_fake_fetch(_canned(), []))

    assert _listing(model_root) == before_root, (
        "the refresh wrote into model_root — R-T10-writable says never")
    assert _listing(REPO_MODELS) == before_repo, (
        "the refresh wrote into the version-controlled repo models/ tree, dirtying the worktree")
    for target in targets:
        for forbidden in (model_root, REPO_MODELS):
            assert forbidden not in target.parents and target != forbidden


def test_a_writable_dir_inside_a_root_is_refused_before_any_fetch(tmp_path):
    """The refusal that makes 'never writes into the weights root' structural rather than
    a convention this module happens to follow."""
    model_root, _ = _roots(tmp_path)
    with pytest.raises(catalog.WritableLocationError):
        model_refresh.build_catalog(model_root, model_root / "model_data")


# --------------------------------------------------------------------------- #
# 4. explicit only — no default-on network
# --------------------------------------------------------------------------- #

def test_refresh_refuses_without_confirm_and_fetches_nothing(tmp_path):
    model_root, writable = _roots(tmp_path)
    cat = catalog.Catalog(model_root, writable, fallback_root=None)
    calls: list[str] = []
    with pytest.raises(model_refresh.RefreshError, match="confirm"):
        model_refresh.refresh(cat, fetch=_fake_fetch(_canned(), calls))
    assert calls == []
    assert _listing(writable) == [], "a refused refresh must not have written anything"


def test_cli_without_confirm_is_a_dry_run(tmp_path, capsys):
    model_root, writable = _roots(tmp_path)
    code = model_refresh.main(["--model-root", str(model_root),
                               "--writable-dir", str(writable)])
    out = capsys.readouterr().out
    assert code != 0, "the CLI must not report success when it did nothing"
    assert "DRY RUN" in out
    for source in model_refresh.TABLES:
        assert source.url in out, "it has to say which URLs it would contact"
    assert _listing(writable) == []


def test_cli_with_confirm_writes_through_the_catalog(tmp_path, monkeypatch, capsys):
    model_root, writable = _roots(tmp_path)
    canned = _canned()
    calls: list[str] = []
    monkeypatch.setattr(model_refresh, "fetch_json", _fake_fetch(canned, calls))
    code = model_refresh.main(["--confirm", "--model-root", str(model_root),
                               "--writable-dir", str(writable)])
    assert code == 0, capsys.readouterr().out
    assert calls == [s.url for s in model_refresh.TABLES]
    assert {p.relative_to(writable) for p in writable.rglob("*.json")} == {
        Path(rel) for rel in EXPECTED_REL.values()}, "the CLI wrote the wrong files"
    assert _listing(model_root) == [], "the CLI wrote into the weights root"


# --------------------------------------------------------------------------- #
# 5. the URLs moved here and did not stay behind
# --------------------------------------------------------------------------- #

def test_the_four_table_urls_are_gone_from_gui_data_constants():
    """Moved, not copied — `gui_data/constants.py` ships in the published fork."""
    constants_src = (REPO / "gui_data" / "constants.py").read_text(encoding="utf-8")
    here_src = (REPO / "model_refresh.py").read_text(encoding="utf-8")
    for source in model_refresh.TABLES:
        assert source.url in here_src, f"{source.moved_from} is not held by model_refresh.py"
        assert source.url not in constants_src, (
            f"{source.moved_from} is still defined in gui_data/constants.py")
        assert source.url.startswith("https://")


def test_every_table_url_here_maps_to_a_real_catalog_table(tmp_path):
    """A typo in (arch, kind) would write a file no reader ever opens; `Catalog` is the
    only thing that can say whether the coordinate exists."""
    model_root, writable = _roots(tmp_path)
    cat = catalog.Catalog(model_root, writable, fallback_root=None)
    for source in model_refresh.TABLES:
        rel = cat._table_rel(cat._arch(source.arch), source.kind)
        assert rel == EXPECTED_REL[(source.arch, source.kind)], source
