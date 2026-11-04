"""Task 9: `Settings` — the 50 engine-read fields as a typed frozen dataclass.

The field list, the type buckets and the default values asserted here are not
recalled from the plan's prose:
`test_field_list_matches_the_measured_model_data_reads` re-derives the list from
`UVR.py`'s `ModelData` body at test time, and
`test_defaults_are_the_resolved_default_data_values` reads them out of
`gui_data.constants.DEFAULT_DATA`. Those two guards are why no invented default
(`window_size=128`, `crop_size=320`, `gpu_conversion=True`, `batch_size=1`,
`device_index="0"` — all five were in an earlier sketch of this step, all five
wrong) can come back, and why a 51st field cannot appear without `ModelData`
growing a read to justify it.

Rulings this file encodes, from the plan's Task 9:
  R-T9-params — no `vr_model_params`/`mdx_model_params`/`*_MAPPER` here (Task 10).
  R-T9-paths  — no `input_paths`/`export_path`/`lastDir`/`fileOneEntry*` here.
  R-T9-types  — type by stored domain, not by the consumer's `int()`/`float()`.
  R-T9-device — `device_set` is one raw `str`; no `:` split, no `device_index`,
                and no validation against the GPU list (that needs `backends.py`).
"""
from __future__ import annotations

import ast
import dataclasses
import json
import pathlib
import pickle
import re
import subprocess
import sys
import typing

import pytest

import settings
from gui_data.constants import (
    ALL_STEMS,
    CHOOSE_STEM_PAIR,
    DEFAULT,
    DEFAULT_DATA,
    DEF_OPT,
)

REPO = pathlib.Path(__file__).resolve().parents[1]
MODEL_DATA_START, MODEL_DATA_END = 364, 804          # UVR.py:365-804, first class in the file

# Real DEFAULT_DATA keys that are deliberately NOT settings (R-T9-paths), plus
# the name the plan's own Step 4 sketch invented (R-T9-device).
RUN_INPUT_KEYS = (
    "input_paths", "export_path", "last_export_path", "lastDir",
    "fileOneEntry", "fileOneEntry_Full", "fileTwoEntry", "fileTwoEntry_Full",
    "DualBatch_inputPaths", "vr_model", "demucs_model", "mdx_net_model",
    "model_hash_table", "help_hints_var", "model_sample_mode",
)


# --------------------------------------------------------------------------- #
# The measured field list (re-derived from UVR.py, never copied from the plan)
# --------------------------------------------------------------------------- #

def measured_model_data_reads() -> tuple[list[str], list[str], list[str]]:
    """Return (`*_var` names, plain-data names, `*_MAPPER` names) inside ModelData.

    Same slice and same regex as the plan's Step 3 script, including its
    assertions: if a count here moves, `ModelData` moved (or the class boundary
    did), and the derivation must be fixed — not the assertion, and not by
    adding a field to `Settings`.
    """
    src = (REPO / "UVR.py").read_text(encoding="utf-8").splitlines()
    body = "\n".join(src[MODEL_DATA_START:MODEL_DATA_END])
    names = sorted(set(re.findall(r"(?<![.\w])root\.([A-Za-z_][A-Za-z0-9_]*)", body)))
    var = [n for n in names if n.endswith("_var")]
    mapper = [n for n in names if n.endswith("_MAPPER")]
    other = [n for n in names if n not in var and n not in mapper]
    called = {n for n in re.findall(r"(?<![.\w])root\.([A-Za-z_][A-Za-z0-9_]*)\(", body)}
    plain = [n for n in other if n not in called]
    assert len(names) == len(var) + len(mapper) + len(other), "classification lost a name"
    assert len(var) == 49, f"expected the 49 measured _var reads, got {len(var)}"
    assert len(plain) == 3, f"expected 3 plain attributes, got {plain}"
    return var, plain, mapper


def settings_field_names() -> set[str]:
    return {f.name for f in dataclasses.fields(settings.Settings)}


def test_field_list_matches_the_measured_model_data_reads():
    """`Settings` == the 49 `ModelData` `*_var` reads + `wav_type_set`."""
    var, plain, mapper = measured_model_data_reads()
    fields = settings_field_names()

    expected = {v[:-len("_var")] for v in var} | set(plain)
    assert "wav_type_set" in plain
    # the two remaining plain reads are the modal dialog's out-parameters
    # (R-T9-params), and the mappers are model knowledge, not settings.
    expected -= {"vr_model_params", "mdx_model_params"}
    assert len(expected) == 50, sorted(expected)

    # one deliberate rename: `is_gpu_conversion_var` -> `gpu_conversion`
    assert "is_gpu_conversion" in expected
    assert "gpu_conversion" in fields
    assert fields == (expected - {"is_gpu_conversion"}) | {"gpu_conversion"}
    assert len(fields) == 50

    assert not fields & {m[:-len("_MAPPER")] for m in mapper}
    assert not fields & {"vr_model_params", "mdx_model_params"}


@pytest.mark.parametrize("banned", RUN_INPUT_KEYS + ("device_index",))
def test_run_inputs_are_not_settings_fields(banned):
    """R-T9-paths / R-T9-device: the field list is closed at 50.

    Every name here except `device_index` is a real `DEFAULT_DATA` key — which is
    exactly how an earlier draft of this step ended up with `input_paths` and
    `export_path` as fields — and `device_index` is the field the plan's Step 4
    sketch asked for. None is read inside `ModelData`, so none is a field.
    """
    assert banned == "device_index" or banned in DEFAULT_DATA, (
        f"{banned!r} is not a DEFAULT_DATA key; this guard lost its point"
    )
    assert banned not in settings_field_names()


@pytest.mark.parametrize("banned", ["vr_hash_MAPPER", "mdx_hash_MAPPER",
                                    "mdx_name_select_MAPPER", "demucs_name_select_MAPPER"])
def test_catalog_mappers_are_not_settings_fields(banned):
    """R-T9-params: `ModelCatalog` (Task 10) owns the mappers and the param channel."""
    assert banned not in settings_field_names()


# --------------------------------------------------------------------------- #
# Types: by stored domain, not by the consumer's coercion (R-T9-types)
# --------------------------------------------------------------------------- #

def field_types() -> dict[str, type]:
    return typing.get_type_hints(settings.Settings)


BOOL_FIELDS = {
    "is_demucs_combine_stems", "is_demucs_pre_proc_model_activate",
    "is_demucs_pre_proc_model_inst_mix", "is_deverb_vocals", "is_invert_spec",
    "is_high_end_process", "is_chunk_demucs", "is_mdx_c_seg_def",
    "is_match_frequency_pitch", "is_mdx23_combine_stems", "is_normalization",
    "is_post_process", "is_primary_stem_only", "is_primary_stem_only_Demucs",
    "is_save_inst_set_vocal_splitter", "is_secondary_stem_only",
    "is_secondary_stem_only_Demucs", "is_split_mode", "is_tta",
    "is_use_directml", "gpu_conversion", "demucs_is_secondary_model_activate",
    "mdx_is_secondary_model_activate", "vr_is_secondary_model_activate",
}
NUMERIC_FIELDS = {
    "aggression_setting": int, "crop_size": int, "margin": int,
    "margin_demucs": int, "mdx_segment_size": int, "shifts": int,
    "window_size": int, "post_process_threshold": float,
}
NUMERIC_NAMES = set(NUMERIC_FIELDS)


def test_type_buckets_are_24_bool_8_numeric_18_str():
    types = field_types()
    assert len(types) == 50
    assert set(BOOL_FIELDS) == {n for n, t in types.items() if t is bool}
    assert set(NUMERIC_NAMES) == {n for n, t in types.items() if t in (int, float)}
    assert len(BOOL_FIELDS) == 24
    assert len(NUMERIC_NAMES) == 8
    strings = {n for n, t in types.items() if t is str}
    assert len(strings) == 18, sorted(strings ^ (set(types) - BOOL_FIELDS - NUMERIC_NAMES))
    assert strings == set(types) - BOOL_FIELDS - NUMERIC_NAMES
    # every field is exactly one of the three buckets -- no Any, no Optional
    assert set(types.values()) <= {bool, int, float, str}


@pytest.mark.parametrize("field_name", ["overlap_mdx23", "semitone_shift"])
def test_numeric_looking_string_fields_stay_strings(field_name):
    """R-T9-types: `overlap_mdx23='8'` / `semitone_shift='0'` are *stored* as text.

    `ModelData` writes `int(float(...))` / `float(...)` (UVR.py:395-396), but a
    float default would match no entry in the combobox list that produced the
    value, and the mismatch would surface as a silently unselected dropdown.
    """
    assert field_types()[field_name] is str
    assert getattr(settings.Settings(), field_name) == DEFAULT_DATA[field_name]
    assert isinstance(DEFAULT_DATA[field_name], str)


@pytest.mark.parametrize("field_name", [
    "batch_size", "mdx_batch_size", "compensate", "overlap", "overlap_mdx",
])
def test_sentinel_or_number_fields_are_strings(field_name):
    """The five dual-read fields keep their sentinel, because it is their domain.

    Each raw read is a sentinel comparison first — `== DEF_OPT` (UVR.py:391,
    :496), `== AUTO_SELECT` (:584), `== DEFAULT` (:393-394) — so a field typed
    `int`/`float` could not hold the value the GUI persists. The coercion stays
    at the use site (Task 11's RunPlan).
    """
    assert field_types()[field_name] is str
    raw = DEFAULT_DATA[field_name]
    default = getattr(settings.Settings(), field_name)
    assert isinstance(default, str)
    assert default == (raw if isinstance(raw, str) else str(raw))


def test_every_default_is_an_instance_of_its_declared_type():
    types = field_types()
    s = settings.Settings()
    for name, typ in types.items():
        value = getattr(s, name)
        assert isinstance(value, typ), (name, value, typ)
        if typ is int:
            assert not isinstance(value, bool), f"{name}: bool is not an int here"


# --------------------------------------------------------------------------- #
# Defaults: resolved from DEFAULT_DATA, never hand-written
# --------------------------------------------------------------------------- #

def test_defaults_match_constants():
    """The plan's Step 1 guard, on the two keys that happen to be literals."""
    s = settings.Settings()
    assert s.window_size == int(DEFAULT_DATA["window_size"])
    assert s.crop_size == int(DEFAULT_DATA["crop_size"])


def test_defaults_are_the_resolved_default_data_values():
    """Every DEFAULT_DATA-backed default equals the table value for that key.

    Independent of `settings.py`'s own coercion: the expected value is rebuilt
    here from the table plus the field's declared type, so a literal typed into
    the dataclass body goes red. `overlap` is the one field whose table value is
    a float (`DEMUCS_OVERLAP[0]` == 0.25) on a field the ruling types `str`,
    because the `tk.StringVar` at UVR.py:7378 hands `ModelData` the text '0.25'.
    """
    types = field_types()
    s = settings.Settings()
    checked = 0
    for f in dataclasses.fields(settings.Settings):
        key = settings.data_key(f.name)
        if key not in DEFAULT_DATA:
            continue
        checked += 1
        raw = DEFAULT_DATA[key]
        typ = types[f.name]
        if typ is bool:
            assert isinstance(raw, bool), f"{key}: non-bool table value in a bool field"
            expected = raw
        elif typ is int:
            assert not isinstance(raw, bool), f"{key}: bool literal in an int field"
            expected = int(raw)
        elif typ is float:
            expected = float(raw)
        else:
            expected = raw if isinstance(raw, str) else str(raw)
        assert getattr(s, f.name) == expected, f.name
        assert isinstance(getattr(s, f.name), typ), f.name
    assert checked == 48, f"expected 48 of the 50 fields to come from DEFAULT_DATA, got {checked}"


def test_the_two_fields_without_a_default_data_key_use_the_app_constructor():
    """`ensemble_main_stem` (UVR.py:1669) and `mdxnet_stems` (:1673)."""
    assert "ensemble_main_stem" not in DEFAULT_DATA
    assert "mdxnet_stems" not in DEFAULT_DATA
    s = settings.Settings()
    assert s.ensemble_main_stem == CHOOSE_STEM_PAIR
    assert s.mdxnet_stems == ALL_STEMS


def test_the_five_invented_defaults_are_not_present():
    """The exact literals from the plan's Step 4 sketch, each wrong on its own.

    Real values: window_size 512, crop_size 256, is_gpu_conversion False,
    batch_size DEF_OPT (a sentinel, so `int` is the wrong type), and there is no
    `device_index` key at all — the key is `device_set`, default `DEFAULT`.
    """
    s = settings.Settings()
    assert s.window_size == 512 != 128
    assert s.crop_size == 256 != 320
    assert s.gpu_conversion is False                      # not True
    assert s.batch_size == DEF_OPT                        # not 1
    assert s.device_set == DEFAULT                        # and no device_index field
    assert not hasattr(s, "device_index")


def test_a_mutable_default_data_value_is_refused_not_adopted():
    """`DEFAULT_DATA` does hold `[]` and `{}` values; mapping a field onto one must
    fail at import rather than give the frozen dataclass a shared mutable default.

    That is the shape R-T9-params forbids ("a `Settings` that grows a `dict` field
    per model"), and `@dataclass(frozen=True)` accepts a mutable default exactly as
    a plain class does, so nothing else would catch it. The guard compares with
    `isinstance`, not `==`, because `DEFAULT_DATA['is_gpu_conversion']` is `False`
    and `False == 0` would read as a match.
    """
    assert isinstance(DEFAULT_DATA["input_paths"], list)
    assert isinstance(DEFAULT_DATA["model_hash_table"], dict)
    with pytest.raises(TypeError, match="not a scalar"):
        settings._data_default("input_paths", "str", "crop_size")
    with pytest.raises(TypeError, match="not a scalar"):
        settings._data_default("model_hash_table", "str", "crop_size")
    # a key that genuinely is not in the table is a field-list bug, named as one
    with pytest.raises(KeyError, match="no key"):
        settings._data_default("ensemble_main_stem", "str", "ensemble_main_stem")


def test_gpu_conversion_keeps_the_is_gpu_conversion_key_visible():
    """Field name != table key for exactly one field, and the code says so."""
    assert settings.data_key("gpu_conversion") == "is_gpu_conversion"
    assert settings.data_key("crop_size") == "crop_size"
    assert "is_gpu_conversion" in DEFAULT_DATA and "gpu_conversion" not in DEFAULT_DATA
    assert settings.field_for_data_key("is_gpu_conversion") == "gpu_conversion"
    assert settings.field_for_data_key("crop_size") == "crop_size"
    for f in dataclasses.fields(settings.Settings):
        key = settings.data_key(f.name)
        assert key in DEFAULT_DATA or f.name in ("ensemble_main_stem", "mdxnet_stems"), key


# --------------------------------------------------------------------------- #
# Frozen
# --------------------------------------------------------------------------- #

def test_settings_is_frozen():
    s = settings.Settings()
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.crop_size = 352                                 # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.device_set = "0"                                # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.window_size += 1                                # type: ignore[misc]
    assert s.crop_size == 256           # the rejected writes changed nothing
    assert s.device_set == DEFAULT
    assert s.window_size == 512


def test_a_replacement_is_a_new_snapshot_not_a_mutation():
    """`ModelData` reads each var exactly once per run (UVR.py:378-470), which is
    what makes a frozen dataclass the right shape here."""
    base = settings.Settings()
    changed = dataclasses.replace(base, crop_size=352)
    assert changed.crop_size == 352
    assert base.crop_size == 256
    assert base != changed
    assert base == settings.Settings()
    assert len({base, settings.Settings()}) == 1


def test_every_field_is_a_scalar_so_frozenness_is_total():
    """No nested container: a frozen dataclass of scalars really is immutable.

    This is also the mechanical form of R-T9-params — a `dict` field added per
    model "to make it work" would fail here as well as on the field-list guard.
    """
    s = settings.Settings()
    for f in dataclasses.fields(settings.Settings):
        value = getattr(s, f.name)
        assert isinstance(value, (bool, int, float, str)), (f.name, type(value))
    # and two independently built instances never share mutable state
    assert dataclasses.asdict(s) == dataclasses.asdict(settings.Settings())
    assert s.ensemble_main_stem == settings.Settings().ensemble_main_stem == CHOOSE_STEM_PAIR


# --------------------------------------------------------------------------- #
# Persistence: %APPDATA%, never CWD-relative
# --------------------------------------------------------------------------- #

@pytest.fixture
def appdata(tmp_path, monkeypatch):
    base = tmp_path / "Roaming"
    base.mkdir()
    monkeypatch.setenv("APPDATA", str(base))
    return base


def test_roundtrip(appdata):
    s = settings.Settings(gpu_conversion=True, crop_size=352)
    s.save()
    assert settings.load().crop_size == 352
    assert settings.load().gpu_conversion is True


def test_every_field_survives_save_then_load(appdata):
    """Round-trip loop over `dataclasses.fields`, so a 51st field is covered too.

    Each field gets a value distinct from its default *and* of its declared type,
    so a field that serialises to `null`, or comes back as a string, fails here.
    """
    types = field_types()
    overrides = {}
    for f in dataclasses.fields(settings.Settings):
        typ = types[f.name]
        default = getattr(settings.Settings(), f.name)
        if typ is bool:
            overrides[f.name] = not default
        elif typ is int:
            overrides[f.name] = default + 7
        elif typ is float:
            overrides[f.name] = default + 0.5
        else:
            overrides[f.name] = f"{default}-UVRTEST"
    s = settings.Settings(**overrides)
    s.save()
    loaded = settings.load()
    for name, value in overrides.items():
        got = getattr(loaded, name)
        assert got == value, name
        assert type(got) is type(value), (name, type(got), type(value))
    assert loaded == s


def test_defaults_round_trip_when_nothing_is_overridden(appdata):
    settings.Settings().save()
    assert settings.load() == settings.Settings()
    assert (appdata / "UVR-fork" / "settings.json").is_file()


def test_settings_file_ignores_cwd(appdata, tmp_path, monkeypatch):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    settings.Settings().save()
    assert (appdata / "UVR-fork" / "settings.json").exists()
    assert not (elsewhere / "settings.json").exists()
    assert not (REPO / "settings.json").exists()


def test_config_dir_falls_back_to_user_config_without_appdata(tmp_path, monkeypatch):
    """No `APPDATA` (POSIX, or a stripped Windows env) -> `~/.config/UVR-fork`."""
    monkeypatch.delenv("APPDATA", raising=False)
    monkeypatch.setattr(pathlib.Path, "home", lambda: tmp_path)
    assert settings.config_dir() == tmp_path / ".config" / "UVR-fork"
    settings.Settings(crop_size=384).save()
    assert (tmp_path / ".config" / "UVR-fork" / "settings.json").is_file()
    assert settings.load().crop_size == 384


def test_blank_appdata_is_treated_as_absent(tmp_path, monkeypatch):
    """Unset, empty and whitespace-only all fall back, like `uvr_paths._env_path`.

    `os.environ.get("APPDATA")` alone raises `KeyError` on unset; `.get(..., real)`
    alone takes `""` literally and builds `Path('')/'.config'`-ish relative junk
    that `os.chdir(BASE_PATH)` (`UVR.py:83`) then resolves against the install dir.
    """
    monkeypatch.setattr(pathlib.Path, "home", lambda: tmp_path)
    expected = tmp_path / ".config" / "UVR-fork"
    for value in (None, "", "   ", "\t"):
        if value is None:
            monkeypatch.delenv("APPDATA", raising=False)
        else:
            monkeypatch.setenv("APPDATA", value)
        assert settings.config_dir() == expected, repr(value)


def test_explicit_path_wins_over_appdata(appdata, tmp_path):
    direct = tmp_path / "custom" / "s.json"
    settings.Settings(crop_size=448).save(direct)
    assert direct.is_file()
    assert not (appdata / "UVR-fork" / "settings.json").exists()
    assert settings.load(direct).crop_size == 448
    assert settings.load() == settings.Settings()


def test_load_returns_defaults_when_the_file_is_absent(appdata, tmp_path, monkeypatch):
    """A missing file is defaults -- and no relative path is consulted."""
    assert settings.load() == settings.Settings()
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps({"window_size": 1024}), encoding="utf-8")
    assert settings.load(raw).window_size == 1024          # explicit path still wins
    monkeypatch.chdir(REPO)
    assert settings.load() == settings.Settings()
    assert settings.load().window_size == 512              # not a repo-root file's value


def test_load_ignores_unknown_keys_and_fills_missing_ones(appdata):
    p = appdata / "UVR-fork" / "settings.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "crop_size": 352,
        "input_paths": ["C:/should/be/dropped"],
        "device_index": "0",
        "some_future_key": 1,
    }), encoding="utf-8")
    s = settings.load()
    assert s.crop_size == 352
    assert s.device_set == DEFAULT                    # absent -> default, not ""
    assert not hasattr(s, "input_paths")
    assert not hasattr(s, "device_index")


def test_load_reparses_its_own_json_types(appdata):
    """Numbers written as text by a hand-edit still land in the typed field."""
    p = appdata / "UVR-fork" / "settings.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"window_size": "512", "post_process_threshold": "0.2",
                             "gpu_conversion": 1, "mp3_bit_set": 192}),
                 encoding="utf-8")
    s = settings.load()
    assert s.window_size == 512 and type(s.window_size) is int
    assert s.post_process_threshold == 0.2 and type(s.post_process_threshold) is float
    assert s.gpu_conversion is True
    assert s.mp3_bit_set == "192" and type(s.mp3_bit_set) is str


def test_load_rejects_an_uncoercible_value_loudly(appdata):
    p = appdata / "UVR-fork" / "settings.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"window_size": "not-a-number"}), encoding="utf-8")
    with pytest.raises(ValueError, match="window_size"):
        settings.load()


def test_saved_json_is_readable_text_with_all_fifty_keys(appdata):
    settings.Settings(device_set="AMD Radeon RX 7900 XTX:0").save()
    raw = json.loads((appdata / "UVR-fork" / "settings.json").read_text(encoding="utf-8"))
    assert set(raw) == settings_field_names()
    assert len(raw) == 50
    assert raw["device_set"] == "AMD Radeon RX 7900 XTX:0"
    assert raw["batch_size"] == DEF_OPT


def test_non_ascii_values_survive_the_json_round_trip(appdata):
    settings.Settings(mp3_bit_set="320k — ünïcode").save()
    assert settings.load().mp3_bit_set == "320k — ünïcode"


# --------------------------------------------------------------------------- #
# device_set: one raw string, four shapes, no split, no validation (R-T9-device)
# --------------------------------------------------------------------------- #

DEVICE_SHAPES = (
    DEFAULT,                                  # "Default" sentinel (constants.py:31)
    "0",                                      # bare index, GPU_DEVICE_NUM_OPTS
    "AMD Radeon RX 7900 XTX:0",               # name:index built at UVR.py:3546
    "AMD Radeon RX 7900 XTX: 1 ",             # the whitespace :410's .strip() survives
    "NVIDIA GeForce RTX 3060:0",              # stale device from another machine
)


@pytest.mark.parametrize("value", DEVICE_SHAPES)
def test_device_set_survives_all_four_stored_shapes(tmp_path, value):
    """The same widget is filled from two different value sets, so all shapes persist."""
    pkl = tmp_path / "data.pkl"
    pkl.write_bytes(pickle.dumps({"device_set": value}))
    assert settings.import_legacy_data_pickle(pkl).device_set == value, value


@pytest.mark.parametrize("value", DEVICE_SHAPES)
def test_device_set_round_trips_untouched(value, appdata):
    s = settings.Settings(device_set=value)
    s.save()
    assert settings.load().device_set == value
    on_disk = json.loads((appdata / "UVR-fork" / "settings.json").read_text(encoding="utf-8"))
    assert on_disk["device_set"] == value


def test_settings_does_not_split_or_validate_device_set(appdata):
    """No `:` split (that is Task 11's RunPlan), no device-count check (backends.py)."""
    s = settings.Settings(device_set="NVIDIA GeForce RTX 3060:0")
    assert s.device_set == "NVIDIA GeForce RTX 3060:0"
    s.save()
    loaded = settings.load()
    assert loaded.device_set == "NVIDIA GeForce RTX 3060:0"      # not reset to DEFAULT
    assert ":" in loaded.device_set


def test_device_set_stays_the_same_with_directml_off_and_on(appdata):
    """Which list is authoritative depends on `is_use_directml` (UVR.py:3559), so
    `Settings` stores both raw and reconciles nothing."""
    dml = "AMD Radeon RX 7900 XTX:1"
    for use_directml in (False, True):
        s = settings.Settings(device_set=dml, is_use_directml=use_directml)
        s.save()
        loaded = settings.load()
        assert loaded.device_set == dml
        assert loaded.is_use_directml is use_directml


def test_settings_never_imports_torch():
    """R-T9-device: a persisted object must not depend on the GPU present at load.

    Validating `device_set` against `torch.cuda.device_count()` would silently
    rewrite a saved multi-GPU choice on a one-card machine — the reset the App
    performs interactively at UVR.py:3560-3561, which is right for a combobox
    and wrong for a file.
    """
    tree = ast.parse((REPO / "settings.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "torch" not in imported, "Settings must not resolve devices; that is backends.py"


# --------------------------------------------------------------------------- #
# Legacy data.pkl import (the official build's shapes)
# --------------------------------------------------------------------------- #

def legacy_pickle(tmp_path: pathlib.Path, payload: dict, name: str = "data.pkl") -> pathlib.Path:
    pkl = tmp_path / name
    pkl.write_bytes(pickle.dumps(payload))
    return pkl


def test_legacy_pickle_import_preserves_device_set_raw(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    legacy = {"is_gpu_conversion": 0, "device_set": "NVIDIA GeForce RTX 3060:0",
              "crop_size": "256", "window_size": "512"}
    pkl = tmp_path / "data.pkl"
    pkl.write_bytes(pickle.dumps(legacy))
    s = settings.import_legacy_data_pickle(pkl)
    assert s.gpu_conversion is True          # 0 is "gpu" here, not False — see :254 in the harness
    # R-T9-device: Settings stores the combobox value verbatim. There is no
    # `device_index` field; the ':' split is RunPlan's (Task 11).
    assert s.device_set == "NVIDIA GeForce RTX 3060:0"
    assert s.crop_size == 256 and type(s.crop_size) is int
    assert s.window_size == 512 and type(s.window_size) is int


def test_field_for_data_key_inverts_data_key_over_all_fifty_fields():
    """The legacy import maps by table key, so the round trip must be exact.

    `data_key` is what `save`/`load` never need (the JSON is keyed by field name)
    but what `data.pkl` always uses, so one wrong entry silently drops a setting
    during the one-time migration.
    """
    for f in dataclasses.fields(settings.Settings):
        key = settings.data_key(f.name)
        assert settings.field_for_data_key(key) == f.name, (f.name, key)
    assert settings.field_for_data_key("gpu_conversion") is None
    assert settings.field_for_data_key("vr_model") is None
    assert settings.data_key("crop_size") == "crop_size"


def test_legacy_pickle_import_reads_the_engine_encoding_of_gpu_conversion(tmp_path):
    """`is_gpu_conversion` can arrive as the derived value, not the widget value.

    UVR.py:384 computes `0 if var.get() else -1` and the golden harness mirrors it
    (`tests/golden/settings.py:254`), so `0` means GPU-on; a naive
    `bool(0) == False` is the wrong import. Plain booleans still pass through.
    """
    for raw, expected in ((0, True), (-1, False), (1, True),
                          (True, True), (False, False), ("True", True), ("False", False)):
        s = settings.import_legacy_data_pickle(
            legacy_pickle(tmp_path, {"is_gpu_conversion": raw}, f"gpu_{raw}.pkl"))
        assert s.gpu_conversion is expected, raw


@pytest.mark.parametrize("field_name,value,expected", [
    ("crop_size", "256", 256),                       # tk.StringVar stored numbers as text
    ("window_size", "1024", 1024),
    ("margin", "44100", 44100),
    ("aggression_setting", "5", 5),
    ("post_process_threshold", "0.2", 0.2),
    ("is_tta", 1, True),                             # booleans arrived as 0/1
    ("is_split_mode", 0, False),
    ("is_normalization", "False", False),
    ("is_high_end_process", "True", True),
    ("device_set", "7", "7"),
    ("overlap", 0.25, "0.25"),
    ("overlap_mdx23", 8, "8"),
    ("save_format", "FLAC", "FLAC"),
    ("segment", "Default", "Default"),
])
def test_legacy_pickle_import_coerces_the_official_shapes(tmp_path, field_name, value, expected):
    pkl = legacy_pickle(tmp_path, {settings.data_key(field_name): value})
    got = getattr(settings.import_legacy_data_pickle(pkl), field_name)
    assert got == expected, field_name
    assert type(got) is type(expected), field_name


def test_legacy_pickle_import_keeps_unread_keys_at_their_defaults(tmp_path):
    """Anything the legacy file does not say is the default, never a guess."""
    s = settings.import_legacy_data_pickle(legacy_pickle(tmp_path, {"crop_size": "352"}))
    assert s.crop_size == 352
    assert s.device_set == DEFAULT
    assert s.batch_size == DEF_OPT
    assert s.overlap_mdx23 == "8"
    assert s.gpu_conversion is False
    assert s == dataclasses.replace(settings.Settings(), crop_size=352)


def test_legacy_pickle_import_drops_keys_settings_does_not_have(tmp_path):
    pkl = legacy_pickle(tmp_path, {"input_paths": ["C:/song.flac"],
                                   "export_path": "C:/out",
                                   "vr_model": "1_HP-UVR.pth",
                                   "crop_size": "256"})
    s = settings.import_legacy_data_pickle(pkl)
    assert s.crop_size == 256
    assert not hasattr(s, "input_paths")
    assert not hasattr(s, "export_path")
    assert not hasattr(s, "vr_model")


def test_legacy_pickle_import_is_read_only(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    pkl = legacy_pickle(tmp_path, {"crop_size": "352"})
    before = pkl.read_bytes()
    s = settings.import_legacy_data_pickle(pkl)
    assert pkl.read_bytes() == before
    assert not (tmp_path / "UVR-fork" / "settings.json").exists()
    s.save()
    assert (tmp_path / "UVR-fork" / "settings.json").is_file()


def test_legacy_pickle_import_rejects_a_non_mapping(tmp_path):
    with pytest.raises(TypeError):
        settings.import_legacy_data_pickle(legacy_pickle(tmp_path, ["not", "a", "mapping"]))


def test_legacy_import_then_save_then_load_is_a_full_bridge(tmp_path, monkeypatch):
    """The one-time migration path ends in the fork's own JSON, types and all."""
    monkeypatch.setenv("APPDATA", str(tmp_path))
    pkl = legacy_pickle(tmp_path, {"crop_size": "352", "window_size": "1024",
                                   "is_gpu_conversion": 0, "is_tta": 1,
                                   "device_set": "AMD Radeon RX 7900 XTX:0",
                                   "batch_size": DEF_OPT})
    imported = settings.import_legacy_data_pickle(pkl)
    imported.save()
    assert settings.load() == imported
    assert settings.load().crop_size == 352
    assert settings.load().gpu_conversion is True
    assert settings.load().device_set == "AMD Radeon RX 7900 XTX:0"


# --------------------------------------------------------------------------- #
# No GUI in the seam
# --------------------------------------------------------------------------- #

def test_settings_imports_no_gui_package():
    """`tkinter` stays out of this module's import graph (Plan §3 dependency rule).

    A subprocess, because `sys.modules` in the suite process is already populated
    by whatever other tests imported first; the check is about what *importing
    settings* drags in.
    """
    probe = ("import sys, settings;"
             "print([m for m in ('tkinter', 'PySide6', 'PySide2') if m in sys.modules])")
    result = subprocess.run([sys.executable, "-c", probe], cwd=str(REPO),
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]", result.stdout


def test_settings_does_not_read_the_global_root_object():
    """The one thing Task 8's ratchet guards in the engine, guarded here by hand.

    `Settings` replaces the tkinter reads, so a name bound to the tk root is a
    read that escaped the dataclass. The check is on the AST, not on the prose:
    `getattr(root, ...)` has no `root.<attr>` token to find, and a source-text
    regex would fire on the docstring's English instead of on real code.
    """
    tree = ast.parse((REPO / "settings.py").read_text(encoding="utf-8"))
    bound = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "root" not in bound, "the tk global root reached this module"
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            assert node.value.id != "root", f"global tk root touched at line {node.lineno}"
    # and no GUI package is imported, at any depth (AST again, never source text:
    # the docstring discusses tkinter in English and must not trip a guard)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {"tkinter", "PySide6", "PySide2", "gui_data.tk_dpi_manager"}, imported


def test_engine_never_imports_settings():
    """The dependency rule points one way: the engine may not know about Settings.

    Task 8's purity ratchet guards the engine from the GUI, and `settings.py` is
    deliberately not in its glob (adding it there would invert the rule). This is
    the matching guard for the other direction. It sweeps the *same* file set the
    ratchet lints by importing its `LINT_GLOBS`, so it cannot silently shrink;
    the count is printed, not pinned, because a vendored engine file appearing
    must not break the suite for the wrong reason.
    """
    from tests.test_engine_purity import LINT_GLOBS

    files = {p for pattern in LINT_GLOBS for p in REPO.glob(pattern) if p.is_file()}
    assert len(files) >= 38, sorted(files)
    print(f"[settings] swept {len(files)} engine files for an `import settings`")
    assert (REPO / "settings.py") not in files
    for path in sorted(files):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                # a relative import cannot reach a top-level `settings` module
                names = [] if node.level else [node.module or ""]
            else:
                continue
            for name in names:
                assert name.split(".")[0] != "settings", f"{path.name} imports settings"
