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

Fix round 1 changed three of this file's own guards, not the dataclass:
  * the GUI-import check is an allow-list of full dotted names
    (`ALLOWED_SETTINGS_IMPORTS`), because the old blacklist truncated imports to
    their top-level package and banned a module that does not exist;
  * the constructor-coercion property (`constructed == load(save(constructed))`) is
    asserted inside the existing `dataclasses.fields` loop, so a 51st field is
    covered by it automatically;
  * the `ModelData` line range the field list is measured over comes from the AST
    instead of a typed-in `364, 804`.
"""
from __future__ import annotations

import ast
import dataclasses
import importlib.util
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


def model_data_class_node() -> ast.ClassDef:
    """The `class ModelData` node of `UVR.py`, found by parsing, not by line number."""
    tree = ast.parse((REPO / "UVR.py").read_text(encoding="utf-8"))
    nodes = [node for node in tree.body
             if isinstance(node, ast.ClassDef) and node.name == "ModelData"]
    assert len(nodes) == 1, (
        f"expected exactly one top-level `class ModelData` in UVR.py, found {len(nodes)}"
        " -- the field list is measured over that class, so this guard has lost its"
        " subject (Task 11 removes the class; delete this derivation with it)"
    )
    return nodes[0]


def model_data_body_bounds() -> tuple[int, int]:
    """0-based, end-exclusive `(start, end)` line indices of the `ModelData` body.

    Fix round 1 (folded Minor): these were the hand-typed `364, 804` -- the plan
    prose's `UVR.py:365-804` -- and nothing ever checked that the slice bracketed the
    class. Parsing is the same move `tests/test_engine_purity.py` makes for its own
    sweep, and it removes the brittleness as well as the blindness: `lineno` and
    `end_lineno` follow the class when anything above it moves, so Task 11 editing
    `UVR.py` cannot turn this guard red for a reason unrelated to `Settings`.
    """
    node = model_data_class_node()
    return node.lineno - 1, node.end_lineno


#: Measured at collection time; `node.lineno - 1` lands on the `class ModelData():`
#: header line, exactly where the old hand-typed `364` did. For the record, the
#: parser puts the body's last statement at 801, so the old `804` reached three
#: lines past the class -- two blank lines and a commented-out `#print` -- which is
#: why narrowing it here loses no `root.` read (see
#: `test_the_derivation_slice_brackets_model_data_and_nothing_else`).
MODEL_DATA_START, MODEL_DATA_END = model_data_body_bounds()

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

    Same regex and same assertions as the plan's Step 3 script; the *range* is now
    the AST-derived `ModelData` body rather than a typed-in pair of line numbers (see
    `model_data_body_bounds`). If a count here moves, `ModelData` moved, and the
    derivation must be fixed -- not the assertion, and not by adding a field to
    `Settings`.
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


def test_the_derivation_slice_brackets_model_data_and_nothing_else():
    """Fix round 1 (folded Minor): the measured range is checked, not assumed.

    The old `MODEL_DATA_START, MODEL_DATA_END = 364, 804` was never validated, so a
    slice that shifted but still happened to contain 49 `_var` reads could have passed
    while measuring the wrong region. Now the bounds come from the class node, and
    this test asserts the region is exactly `ModelData`: it starts on the header,
    ends on the body's last statement, and the next top-level statement is a
    *different* class with nothing but blank lines and a comment in between.
    """
    source = (REPO / "UVR.py").read_text(encoding="utf-8")
    lines = source.splitlines()
    tree = ast.parse(source)
    node = model_data_class_node()

    assert (MODEL_DATA_START, MODEL_DATA_END) == (node.lineno - 1, node.end_lineno)
    assert lines[MODEL_DATA_START].strip() == "class ModelData():", lines[MODEL_DATA_START]
    assert lines[MODEL_DATA_END - 1].strip(), "the body's last line must not be blank"

    following = [n for n in tree.body
                 if isinstance(n, (ast.ClassDef, ast.FunctionDef))
                 and n.lineno > MODEL_DATA_END]
    assert following, "nothing follows ModelData, so the end bound is unproven"
    assert following[0].name == "Ensembler", f"next class is {following[0].name}"
    gap = lines[MODEL_DATA_END:following[0].lineno - 1]
    assert all((not line.strip()) or line.lstrip().startswith("#") for line in gap), gap
    assert "root." not in "\n".join(gap), "code that reads the tk root is outside the slice"

    body = "\n".join(lines[MODEL_DATA_START:MODEL_DATA_END])
    assert body.count("root.") >= 49, body.count("root.")


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


def widget_form_of(field_name: str, value, typ: type):
    """The same value as a UI hands it over: every non-`str` scalar as *text*.

    Plan 2 builds `Settings` from `StringVar` contents, so an `int` field receives
    `'352'` and a `bool` field `'True'`. The declared type is what has to come out
    the other side. `str` fields are returned untouched -- this helper rewrites no
    string, because R-T9-types says the string *is* the stored value.
    """
    if typ is str:
        return value
    if typ is bool:
        return "True" if value else "False"
    if typ is float:
        return repr(value)
    return str(value)


def test_every_field_survives_save_then_load(appdata):
    """Round-trip loop over `dataclasses.fields`, so a 51st field is covered too.

    Each field gets a value distinct from its default *and* of its declared type,
    so a field that serialises to `null`, or comes back as a string, fails here.

    Fix round 1 (Important 2) extends this same loop rather than adding 50 asserts:
    for every field, the value re-entered as widget text must land on the declared
    type, and `Settings(**{name: text})` must equal
    `load(save(Settings(**{name: text})))` -- the equality that was False for
    `crop_size='352'` before `__post_init__` coerced.
    """
    types = field_types()
    overrides = {}
    widget_forms = {}
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
        widget_forms[f.name] = widget_form_of(f.name, overrides[f.name], typ)
    s = settings.Settings(**overrides)
    s.save()
    loaded = settings.load()
    for name, value in overrides.items():
        got = getattr(loaded, name)
        assert got == value, name
        assert type(got) is type(value), (name, type(got), type(value))
    assert loaded == s

    failures = []
    for name, text in widget_forms.items():
        typ = types[name]
        constructed = settings.Settings(**{name: text})
        one = appdata / "UVR-fork" / f"{name}.json"
        constructed.save(one)
        reloaded = settings.load(one)
        held = getattr(constructed, name)
        if type(held) is not typ:
            failures.append(f"{name}: the constructor kept {held!r} "
                            f"({type(held).__name__}) in a {typ.__name__} field")
        if held != overrides[name]:
            failures.append(f"{name}: {text!r} parsed to {held!r}, "
                            f"not {overrides[name]!r}")
        if reloaded != constructed:
            failures.append(f"{name}: constructed != loaded -- {held!r} vs "
                            f"{getattr(reloaded, name)!r}")
    assert not failures, "the constructor does not mean its annotations:\n" + "\n".join(failures)

    # ...and all fifty entered as text at once is the same snapshot as all fifty
    # entered typed, which is what Task 11's `Settings == load()` comparisons rely on.
    mixed = settings.Settings(**widget_forms)
    mixed.save(appdata / "UVR-fork" / "all-fields-as-text.json")
    assert settings.load(appdata / "UVR-fork" / "all-fields-as-text.json") == mixed
    assert mixed == s


def test_constructed_equals_loaded_for_the_same_widget_string(appdata):
    """Important 2, reproduced as a test: this whole assertion was False before.

    The controller's read-only repro was
    `Settings(crop_size='352').crop_size -> '352'` while `save()`/`load()` gave the
    `int` `352`, so `constructed == loaded` was False for one logical value.
    """
    constructed = settings.Settings(crop_size='352')
    constructed.save()
    loaded = settings.load()
    assert constructed.crop_size == 352 and type(constructed.crop_size) is int
    assert loaded == constructed


@pytest.mark.parametrize("field_name,text_value,expected", [
    ("crop_size", "352", 352),
    ("window_size", " 1024 ", 1024),       # int() tolerates the padding a widget holds
    ("margin", "44100", 44100),
    ("shifts", "2", 2),
    ("post_process_threshold", "0.2", 0.2),
    ("is_tta", "True", True),
    ("gpu_conversion", "false", False),
])
def test_widget_text_entered_in_a_typed_field_is_reparsed(field_name, text_value, expected):
    """Case 1 of Important 2: text in an `int`/`float`/`bool` field parses."""
    s = settings.Settings(**{field_name: text_value})
    got = getattr(s, field_name)
    assert got == expected, field_name
    assert type(got) is type(expected), (field_name, type(got))
    # the same instance `load()` produces from the equivalent file, not a lookalike
    assert s == dataclasses.replace(settings.Settings(), **{field_name: expected})


@pytest.mark.parametrize("value", [
    "Default", "0.25", "0", "", "   8  ", "AMD Radeon RX 7900 XTX: 1 ",
    "320k — ünïcode", "None", "Main Vocals Only", "PCM_16",
])
def test_a_string_field_is_handed_through_verbatim_never_reformatted(value):
    """R-T9-types / R-T9-device: coercion parses *types*; it does not tidy strings.

    `batch_size='Default'`, `overlap='0.25'` and `semitone_shift='0'` are strings on
    purpose, and `device_set` keeps its trailing space because the `.strip()` at
    `UVR.py:410` is RunPlan's business. `is` is the load-bearing assertion: the very
    object that came in comes out, so nothing stripped, cased or `str()`-round-tripped
    it. Looping over the `str`-typed fields rather than naming them covers a 51st.
    """
    str_fields = [name for name, typ in field_types().items() if typ is str]
    assert len(str_fields) == 18, sorted(str_fields)
    for field_name in str_fields:
        s = settings.Settings(**{field_name: value})
        held = getattr(s, field_name)
        assert held is value, (field_name, repr(held))


def test_a_number_entered_in_a_string_field_becomes_the_widgets_text():
    """The one rewrite a `str` field accepts is a value that was never a string.

    `load()` already does this (`_coerce`'s `str` branch, tested at
    `test_load_reparses_its_own_json_types`), so the constructor has to agree with
    it -- otherwise `constructed != loaded` again, just in the other direction.
    """
    assert settings.Settings(overlap=0.25).overlap == "0.25"
    assert settings.Settings(mp3_bit_set=192).mp3_bit_set == "192"
    assert settings.Settings(overlap_mdx23=8).overlap_mdx23 == "8"
    assert settings.Settings(device_set=0).device_set == "0"


@pytest.mark.parametrize("field_name,value", [
    ("crop_size", "abc"),
    ("crop_size", "3.5"),
    ("crop_size", 1.5),                    # a fractional float is not a whole number
    ("window_size", ""),
    ("margin", None),
    ("shifts", [2]),
    ("aggression_setting", {}),
    ("post_process_threshold", "wide"),
    ("device_set", None),
    ("is_tta", -1),                        # the legacy CPU encoding is not a flag here
    ("is_tta", 2),
    ("is_split_mode", "yes"),
    ("gpu_conversion", "maybe"),
    ("is_use_directml", None),
])
def test_a_value_that_cannot_be_parsed_raises_instead_of_becoming_zero(field_name, value):
    """Important 2's other half: silence is how a wrong value reaches the engine.

    `int('abc')` failing must not be papered over with `0`, and a bare
    `Settings(is_tta=-1)` is a caller error: the `-1`-means-CPU decoding belongs to
    the import path (`_legacy_gpu_conversion`), so `-1` in a plain `bool` field has
    no defensible reading and raises.
    """
    with pytest.raises((ValueError, TypeError)):
        settings.Settings(**{field_name: value})


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
    names = imported_modules(SETTINGS_SOURCE)
    # full dotted names, so `import torch.cuda` is the same violation as `import torch`
    assert not [n for n in names if n == "torch" or n.startswith("torch.")], names


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
# No GUI in the seam: an ALLOW-LIST of what `settings.py` may import
# --------------------------------------------------------------------------- #

#: Every module `settings.py` may import, as full dotted names.
#:
#: An allow-list, not a longer blacklist. The guard this replaces truncated each
#: import to its top-level package and intersected a set of four banned names, which
#: broke twice over: `from gui_data.tkinterdnd2 import …` normalised to
#: `{"gui_data"}` and passed (so *every* dotted entry was dead), and one of the four
#: terms — `gui_data.tk_dpi_manager` — names a module that does not exist in this
#: repo, which is decoration rather than a guard. The coupling that walked through is
#: not hypothetical: `gui_data/app_size_values.py:1-5` imports `screeninfo` and
#: `PIL.ImageTk`, so `from gui_data.app_size_values import …` in `settings.py` kept
#: all ninety tests green under a docstring claiming no GUI package is imported at
#: any depth.
#:
#: Deliberately *narrower* than "stdlib plus `gui_data.constants`": `tkinter` is in
#: the standard library, so a stdlib-wide allow-list would re-open the exact hole it
#: is meant to close. The list is the modules this file actually uses, held in place
#: by `test_settings_imports_only_the_allow_listed_modules` (set equality, so a dead
#: entry fails too) and audited for being real by
#: `test_the_allow_list_holds_only_modules_that_exist_and_are_not_gui`.
#:
#: Matching full dotted names is the purity ratchet's own rule —
#: `tests/test_engine_purity.py:176-179` carries the comment *"Match the full dotted
#: name, never `name.split('.')[0]`"*, because blocking `gui_data` wholesale would
#: forbid the pure-data `gui_data.constants` that `separate.py:18` legitimately needs.
ALLOWED_SETTINGS_IMPORTS = frozenset({
    "__future__", "dataclasses", "json", "os", "pathlib", "pickle", "typing",
    "gui_data.constants",
})

#: Names that must never be reachable through the allow-list. Kept as a list to fail
#: loudly and to be read, not because the check needs them: the allow-list already
#: rejects all of them, including `tkinter` (stdlib) and a bare `gui_data`.
GUI_IMPORT_NAMES = (
    "tkinter", "tkinter.filedialog", "PySide2", "PySide6", "PySide6.QtWidgets",
    "PIL", "PIL.ImageTk", "screeninfo", "sv_ttk", "torch",
    "gui_data", "gui_data.app_size_values", "gui_data.tkinterdnd2",
    "gui_data.sv_ttk", "gui_data.error_handling", "gui_data.old_data_check",
)

#: The file under guard, read once for every check below.
SETTINGS_SOURCE = (REPO / "settings.py").read_text(encoding="utf-8")

#: The one GUI-adjacent line `settings.py` really has, used as the swap target for
#: the red controls. `settings.py` resolves every default out of this table (see its
#: module docstring), so the line is not removable — but it is replaceable in memory.
_CONSTANTS_IMPORT_LINE = "from gui_data.constants import"


def import_lines(source: str) -> list[tuple[str, int]]:
    """`(module name, line number)` for every import in `source`, in source order.

    Full dotted names, at any nesting depth, including inside function bodies. A
    relative import (`from . import x`) has no module name to judge, so it is
    reported under a marker no allow-list contains.
    """
    found: list[tuple[str, int]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found.extend((alias.name, node.lineno) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                found.append(("<relative import>", node.lineno))
            elif node.module:
                found.append((node.module, node.lineno))
    return found


def imported_modules(source: str) -> set[str]:
    """The module names `source` imports, never truncated to their roots."""
    return {name for name, _lineno in import_lines(source)}


def is_allow_listed(name: str) -> bool:
    """`dataclasses.fields` rides on `dataclasses`; `gui_data.app_size_values` does
    NOT ride on `gui_data.constants`.

    The `+ "."` in the prefix is what makes this a dotted-name match rather than a
    substring match: `gui_data.constantsside` and `tkinterfoo` stay outside.
    """
    return any(name == entry or name.startswith(entry + ".")
               for entry in ALLOWED_SETTINGS_IMPORTS)


def import_violations(source: str) -> list[str]:
    """Off-list imports as `"<module> (line N)"`, empty for a module that obeys."""
    return sorted(f"{name} (line {lineno})"
                  for name, lineno in import_lines(source) if not is_allow_listed(name))


def planted_settings_source(statement: str) -> str:
    """`settings.py`'s text with its `gui_data` import line swapped for `statement`.

    In memory only: the file on disk is never written, so no red control can leave a
    stray GUI import in `settings.py`, and the guard under test still reads the real
    thing.
    """
    lines = SETTINGS_SOURCE.splitlines(keepends=True)
    hits = [i for i, line in enumerate(lines) if line.startswith(_CONSTANTS_IMPORT_LINE)]
    assert len(hits) == 1, f"expected exactly one {_CONSTANTS_IMPORT_LINE!r} line, got {hits}"
    lines[hits[0]] = statement + "\n"
    return "".join(lines)


def test_settings_imports_only_the_allow_listed_modules():
    """The positive property: `settings.py`'s import set *is* the allow-list.

    Exact set equality, so the guard fails in both directions — for an import that is
    not on the list, and for a list entry the file no longer uses. A stale entry is
    how `gui_data.tk_dpi_manager` came to look like a guard while pointing at nothing.
    """
    assert import_violations(SETTINGS_SOURCE) == []
    assert imported_modules(SETTINGS_SOURCE) == set(ALLOWED_SETTINGS_IMPORTS)


@pytest.mark.parametrize("statement,offending", [
    ("from gui_data.app_size_values import SCREEN_WIDTH", "gui_data.app_size_values"),
    ("import gui_data.app_size_values", "gui_data.app_size_values"),
    ("from gui_data import constants", "gui_data"),
    ("from gui_data.tkinterdnd2 import TkinterDnD", "gui_data.tkinterdnd2"),
    ("from gui_data.sv_ttk import ttk", "gui_data.sv_ttk"),
    ("from gui_data.old_data_check import read_old_data", "gui_data.old_data_check"),
    ("import tkinter", "tkinter"),
    ("from tkinter import ttk", "tkinter"),
    ("import json, tkinter", "tkinter"),
    ("from PySide6.QtWidgets import QSlider", "PySide6.QtWidgets"),
    ("import PySide2.QtCore", "PySide2.QtCore"),
    ("from PIL import ImageTk", "PIL"),
    ("from screeninfo import get_monitors", "screeninfo"),
    ("import torch", "torch"),
    ("from . import constants", "<relative import>"),
    ("from gui_data.constants import DEFAULT_DATA", None),        # the allowed spelling
])
def test_the_import_guard_fires_on_a_real_gui_coupling(statement, offending):
    """Red control for Item 1, run against a scratch copy of the file's text.

    Each statement is planted in place of `settings.py`'s one real `gui_data` import
    and judged by the guard's own function, so this exercises the shipped check and
    not a copy of it. `import json, tkinter` is the case the old blacklist *did*
    catch, only because `tkinter` was spelled in it; `gui_data.app_size_values` is
    the one it could never catch, because truncation erased the part that made it a
    GUI module. The final row is the control proving the allow-list still admits the
    pure-data import `settings.py` needs.
    """
    planted = planted_settings_source(statement)
    if offending is None:
        assert import_violations(planted) == [], statement
        return
    violations = import_violations(planted)
    assert len(violations) == 1, (statement, violations)
    assert offending in violations[0], (statement, violations)
    assert "line" in violations[0], violations


def test_the_guard_fires_on_a_scratch_copy_written_to_disk(tmp_path):
    """The same control through a real file, with the real file provably untouched.

    The planted copy lands in `tmp_path`, never in the repo, so a red control cannot
    leave a stray GUI import behind in `settings.py` -- and the last line re-reads the
    real file to show this test did not modify it either.
    """
    scratch = tmp_path / "settings_with_a_gui_import.py"
    scratch.write_text(
        planted_settings_source("from gui_data.app_size_values import SCREEN_WIDTH"),
        encoding="utf-8",
    )
    violations = import_violations(scratch.read_text(encoding="utf-8"))
    assert violations and "gui_data.app_size_values" in violations[0], violations
    assert (REPO / "settings.py").read_text(encoding="utf-8") == SETTINGS_SOURCE, (
        "the real settings.py was written by a control that should only ever read it")
    assert import_violations(SETTINGS_SOURCE) == []


def test_the_old_truncating_blacklist_could_not_fire_at_all():
    """The before/after contrast on one scratch text, kept as a regression test.

    Measured before the fix on a scratch copy: the shipped `settings.py` and a
    version planted with `from gui_data.app_size_values import SCREEN_WIDTH`
    normalise to the *same* truncated set, and that set intersects the old four-name
    blacklist to nothing — so the guard passed a real GUI coupling, which is exactly
    what `tests/test_engine_purity.py`'s *"never `name.split('.')[0]`"* comment warns
    about. Both halves are re-derived here from the live file, so this cannot rot into
    a story about a guard that no longer exists.
    """
    banned_before_the_fix = {"tkinter", "PySide6", "PySide2", "gui_data.tk_dpi_manager"}

    def truncated(source: str) -> set[str]:
        return {name.split(".")[0] for name in imported_modules(source)}

    planted = planted_settings_source("from gui_data.app_size_values import SCREEN_WIDTH")
    assert truncated(SETTINGS_SOURCE) == truncated(planted), "truncation hid the import"
    assert truncated(planted) & banned_before_the_fix == set()
    assert import_violations(planted), "the allow-list guard must fire where old passed"
    # and the dead term really was dead: no such module, while the allowed one exists
    assert not (REPO / "gui_data" / "tk_dpi_manager.py").exists()
    assert (REPO / "gui_data" / "constants.py").is_file()


def test_the_allow_list_holds_only_modules_that_exist_and_are_not_gui():
    """No decoration entries, and no third-party module smuggled in as "stdlib".

    Every allow-list name must resolve here and now — the check the banned
    `gui_data.tk_dpi_manager` would have failed — and must be standard library, apart
    from the one pure-data exception the module is built on. The other direction is
    the reason the list is a list: `tkinter` is itself a stdlib module, so admitting
    "the stdlib" wholesale would let the GUI back in.
    """
    for name in sorted(ALLOWED_SETTINGS_IMPORTS):
        assert importlib.util.find_spec(name) is not None, f"{name} is not importable"
        if name != "gui_data.constants":
            assert name in sys.stdlib_module_names, f"{name} is not stdlib"
    for name in GUI_IMPORT_NAMES:
        assert not is_allow_listed(name), f"{name} became reachable through the list"
    assert "tkinter" in sys.stdlib_module_names, (
        "the 'a stdlib allow-list is not enough' argument depends on tkinter being "
        "standard library; if that stops being true, widen the list deliberately")


def test_settings_imports_no_gui_package():
    """`tkinter` stays out of this module's import graph (Plan §3 dependency rule).

    A subprocess, because `sys.modules` in the suite process is already populated
    by whatever other tests imported first; the check is about what *importing
    settings* drags in — the transitive fact the AST scan cannot see (a coupling
    arriving through a wildcard import, or an `importlib.import_module` call inside a
    function body, the limitation the Task 8 ratchet already defers).

    It used to enumerate three module names, which is the same enumerative weakness the
    blacklist had. Now it reports every GUI-shaped module the child has loaded and the
    only acceptable answer is the pure-data `gui_data` pair. Note the one place a
    `.split(".")[0]` is right: this is a *deny* sweep, so truncating makes it broader
    — it catches `tkinter.filedialog` and `PySide6.QtCore` without naming them —
    whereas the allow-list must match full dotted names or it catches nothing.
    """
    probe = (
        "import sys, settings;"
        "roots = {'tkinter', 'PySide2', 'PySide6', 'PIL', 'screeninfo', 'sv_ttk',"
        " 'gui_data'};"
        "print(sorted(m for m in sys.modules if m.split('.')[0] in roots))"
    )
    result = subprocess.run([sys.executable, "-c", probe], cwd=str(REPO),
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "['gui_data', 'gui_data.constants']", result.stdout


def test_settings_does_not_read_the_global_root_object():
    """The one thing Task 8's ratchet guards in the engine, guarded here by hand.

    `Settings` replaces the tkinter reads, so a name bound to the tk root is a
    read that escaped the dataclass. The check is on the AST, not on the prose:
    `getattr(root, ...)` has no `root.<attr>` token to find, and a source-text
    regex would fire on the docstring's English instead of on real code.
    """
    tree = ast.parse(SETTINGS_SOURCE)
    bound = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "root" not in bound, "the tk global root reached this module"
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            assert node.value.id != "root", f"global tk root touched at line {node.lineno}"
    # and no GUI package is imported, at any depth -- the allow-list, on full dotted
    # names (AST again, never source text: the docstring discusses tkinter in English
    # and must not trip a guard).
    assert import_violations(SETTINGS_SOURCE) == [], import_violations(SETTINGS_SOURCE)


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
