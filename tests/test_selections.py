"""The 26 selections Plan 1 left unreachable, plus the 7 that address them.

Ruling R-T11-selections kept these out of `Settings` because the criterion is "the
engine reads it inside the model-data class" -- they arrive as `plan.build()`
arguments. So they get their own dataclass, and `Settings`' 50-field invariant is
untouched (spec §10, ruling 2).
"""
from __future__ import annotations

import json

import plan
import selections
import settings as settings_module
import pytest

EXPECTED_FIELDS = 33


def test_field_count_and_names_are_pinned():
    names = selections.SELECTION_FIELD_NAMES
    assert len(names) == EXPECTED_FIELDS, sorted(names)
    assert names == {
        "vr_model", "mdx_net_model", "demucs_model", "apollo_model",
        "demucs_pre_proc_model",
        "set_vocal_splitter", "is_set_vocal_splitter",
        "vocal_splitter_primary_label", "vocal_splitter_secondary_label",
    } | {f"{a}_{g}_secondary_model" for a in ("vr", "mdx", "demucs")
         for g in ("voc_inst", "other", "drums", "bass")} | {
        f"{a}_{g}_secondary_model_scale" for a in ("vr", "mdx", "demucs")
        for g in ("voc_inst", "other", "drums", "bass")}


def test_defaults_round_trip_through_the_same_json_file(tmp_path):
    target = tmp_path / "settings.json"
    sel = selections.Selections(vr_model="1_HP-UVR", vr_voc_inst_secondary_model="UVR-BVE-458n")
    selections.save(sel, target)
    assert selections.load(target) == sel


def test_saving_settings_does_not_erase_selections_and_vice_versa(tmp_path):
    """One file, two writers. `Settings.save()` used to write its own dict as the whole
    file, which would silently delete the selections block -- and the reverse would
    delete 50 user settings. This is the test that makes that impossible."""
    target = tmp_path / "settings.json"
    sel = selections.Selections(vr_model="1_HP-UVR")
    st = settings_module.Settings(window_size=1024, save_format="FLAC")
    selections.save(sel, target)
    st.save(target)
    assert selections.load(target) == sel
    sel2 = selections.Selections(demucs_model="htdemucs")
    selections.save(sel2, target)
    assert settings_module.load(target).window_size == 1024
    assert settings_module.load(target).save_format == "FLAC"


def test_to_plan_dict_matches_the_shape_plan_build_documents():
    sel = selections.Selections(
        vr_voc_inst_secondary_model="UVR-BVE-458n", vr_voc_inst_secondary_model_scale=0.7,
        demucs_pre_proc_model="UVR-DeNoise-Lite",
        set_vocal_splitter="vocal_splitter_model_bs_roformer",
        is_set_vocal_splitter=True,
        vocal_splitter_primary_label="Vocals", vocal_splitter_secondary_label="Instrumental")
    d = sel.to_plan_dict()
    assert d["vr"]["voc_inst_secondary_model"] == "UVR-BVE-458n"
    assert d["vr"]["voc_inst_secondary_model_scale"] == 0.7
    assert d["demucs_pre_proc_model"] == "UVR-DeNoise-Lite"
    assert d["vocal_splitter"] == {"name": "vocal_splitter_model_bs_roformer",
                                   "is_activated": True,
                                   "primary_stem_only_label": "Vocals",
                                   "secondary_stem_only_label": "Instrumental"}
    # The keys plan.py reads, verbatim from its docstring.
    assert set(d) == {"vr", "mdx", "demucs", "demucs_pre_proc_model", "vocal_splitter"}


def test_empty_selections_is_the_cli_behaviour(tmp_path):
    """`selections={}` is what separate_cli passes; an all-default Selections must
    produce a plan identical to that, or the GUI changes runs it does not touch."""
    assert selections.Selections().to_plan_dict() == {}


def test_unknown_keys_are_dropped_on_load(tmp_path):
    target = tmp_path / "settings.json"
    target.write_text(json.dumps({"selections": {"vr_model": "1_HP-UVR", "nonsense": 1}}),
                      encoding="utf-8")
    assert selections.load(target).vr_model == "1_HP-UVR"


def test_a_wrongly_typed_selection_raises_instead_of_silently_passing(tmp_path):
    """Mirrors `settings.load`'s rule. `isinstance(v, int)` alone would accept `True`
    for a scale field, and a bool reaching `plan.build` as a scale is a 0.0 blend."""
    target = tmp_path / "settings.json"
    target.write_text(json.dumps({"selections": {"vr_voc_inst_secondary_model_scale": True}}),
                      encoding="utf-8")
    with pytest.raises(TypeError, match="vr_voc_inst_secondary_model_scale"):
        selections.load(target)


def test_a_numeric_string_scale_coerces_to_the_float():
    """R-T3-coerce. The Task 5/6 UI builds `Selections` from widget text, so `'0.7'`
    is a real constructor argument -- and before `__post_init__` it *stayed* text in a
    field annotated `float`. `save()` then wrote `"0.7"` and the next `load()` raised
    `TypeError` at its float gate: the app poisoning its own config. The value is in
    its declared type at construction now, so what the writer stores is what the reader
    accepts, and `to_plan_dict()` hands `plan.build` a number. (The engine itself
    survived the string -- `plan.py:280` does `float(scale)` -- which is exactly why
    this was a persistence hazard rather than a crash, and why it needed a test.)
    """
    sel = selections.Selections(vr_voc_inst_secondary_model="UVR-BVE-458n",
                                vr_voc_inst_secondary_model_scale="0.7")
    held = sel.vr_voc_inst_secondary_model_scale
    assert held == 0.7
    assert type(held) is float, f"the constructor kept {held!r}"
    # `to_plan_dict()` ships a scale only with its model name, so the slot is present
    # here -- and it ships as the number, not the text the widget handed over.
    assert type(sel.to_plan_dict()["vr"]["voc_inst_secondary_model_scale"]) is float


def test_a_junk_string_scale_raises_instead_of_becoming_zero():
    """The other half of the coercion: `'0.7'` parses, and a value that does not parse
    **raises** rather than landing on `0.0` -- a silent `0.0` would mute the secondary
    model in the plan while the UI still showed it selected. Same rule as
    `_coerce`'s float branch in `Settings` (settings.py:311-323), including the refusal
    of a PEP 515 separator: `float('1_0')` parses, to a number no widget typed."""
    with pytest.raises(ValueError, match="mdx_bass_secondary_model_scale"):
        selections.Selections(mdx_bass_secondary_model_scale="loud")
    with pytest.raises(ValueError, match="mdx_bass_secondary_model_scale"):
        selections.Selections(mdx_bass_secondary_model_scale="1_0")
    with pytest.raises(ValueError, match="demucs_other_secondary_model_scale"):
        selections.Selections(demucs_other_secondary_model_scale=float("nan"))


def test_the_splitter_switch_takes_only_a_real_bool():
    """The hazard this fix exists for. `is_set_vocal_splitter='False'` used to
    construct happily, and `to_plan_dict()` re-coerces the field with `bool(...)`
    (its `vocal_splitter` entry), so the *text* `'False'` -- truthy -- silently
    activated a second model in the plan. `0` and `''` are the same category of input:
    a non-bool this module's loader refuses. Construction now takes only a `bool`, so
    nothing untyped ever reaches that `bool(...)`; the real values still pass
    untouched."""
    for bad in ("False", "True", "0", "1", 1, 0, 2.0, "", [], {}, object()):
        with pytest.raises(TypeError, match="is_set_vocal_splitter"):
            selections.Selections(is_set_vocal_splitter=bad)
    assert selections.Selections(is_set_vocal_splitter=True).is_set_vocal_splitter is True
    assert selections.Selections().is_set_vocal_splitter is False


def test_construction_refuses_what_the_loader_refuses(tmp_path):
    """The two gates must agree on the bool field's domain. `load()` has always
    required `isinstance(value, bool)` there, so a constructor accepting `1` or
    `'False'` would let Task 6 write, via the bare constructor, a value the same file
    cannot be read back as -- and the round trip would only ever have agreed by
    accident. Checked from both sides here."""
    target = tmp_path / "settings.json"
    for bad in (1, "False"):
        target.write_text(json.dumps({"selections": {"is_set_vocal_splitter": bad}}),
                          encoding="utf-8")
        with pytest.raises(TypeError, match="is_set_vocal_splitter"):
            selections.load(target)
        with pytest.raises(TypeError, match="is_set_vocal_splitter"):
            selections.Selections(is_set_vocal_splitter=bad)


def test_a_str_field_keeps_its_text_and_does_not_render_objects():
    """`_coerce`'s `str` branch as `Settings`' does it (settings.py:324-331): a `str`
    returns as the very object given, so the two checkbox labels and the model names
    stay verbatim (whitespace included -- `test_defaults_round_trip` would not notice
    `' Vocals '` becoming `'Vocals'`), while a container raises instead of becoming the
    text `'[]'`, which no model directory is named."""
    label = " Vocals "
    sel = selections.Selections(vocal_splitter_primary_label=label)
    assert sel.vocal_splitter_primary_label is label
    assert sel.vocal_splitter_primary_label == " Vocals "
    for bad in ([], {}, {"a": 1}, (1,)):
        with pytest.raises(TypeError, match="vocal_splitter_primary_label"):
            selections.Selections(vocal_splitter_primary_label=bad)


def test_constructing_from_what_the_loader_accepts_is_idempotent(tmp_path):
    """The property R-T3-coerce had to preserve: coercion runs on every construction,
    so running it twice must change nothing. `load()` hands the constructor values it
    has already type-checked -- including the `int` JSON writes for a `float` field --
    and `Selections(**those)` must equal the instance `load()` returned, so
    `constructed == loaded` holds for every file this module can read."""
    target = tmp_path / "settings.json"
    stored = {"vr_model": "1_HP-UVR",
              "vr_voc_inst_secondary_model_scale": 1,
              "mdx_drums_secondary_model_scale": 0.7,
              "is_set_vocal_splitter": True,
              "vocal_splitter_primary_label": " Vocals "}
    target.write_text(json.dumps({"selections": stored}), encoding="utf-8")
    loaded = selections.load(target)
    assert loaded.vr_voc_inst_secondary_model_scale == 1.0
    assert loaded == selections.Selections(**stored)
    assert loaded == selections.Selections(**json.loads(
        (target.read_text(encoding="utf-8")))["selections"])
    selections.save(loaded, target)
    assert selections.load(target) == loaded


def test_a_selection_reaches_the_plan_only_when_its_switch_is_on(tmp_path):
    """Not a mock: build real plans with one secondary model selected, once with the
    gate off and once with it on, and assert the plan carries the value only in the
    second case. `plan.py:484` reads `settings.vr_is_secondary_model_activate and not
    is_secondary_model`, so an all-default Settings silently drops every secondary
    selection -- a GUI that persists a model name the user can see and then ignores it
    is the failure this test exists to make impossible.
    """
    import separate_cli
    from app import session

    cat = separate_cli.make_catalog(None, tmp_path / "w")   # installed root, read-only
    sel = selections.Selections(
        vr_model="1_HP-UVR",
        vr_voc_inst_secondary_model="UVR-DeNoise-Lite",
        vr_voc_inst_secondary_model_scale=0.6)

    off = session.assemble(st=settings_module.Settings(), cat=cat, model="1_HP-UVR",
                           force_device="cpu", selections=sel.to_plan_dict(),
                           printer=lambda line: None)
    assert off.rp.secondary_model is None
    assert off.rp.is_secondary_model_activated is False

    on = session.assemble(st=settings_module.Settings(vr_is_secondary_model_activate=True),
                          cat=cat, model="1_HP-UVR", force_device="cpu",
                          selections=sel.to_plan_dict(), printer=lambda line: None)
    name = getattr(on.rp.secondary_model, "model_basename", on.rp.secondary_model)
    assert name == "UVR-DeNoise-Lite", f"the plan carried {name!r}"
    assert on.rp.secondary_model_scale == 0.6
    assert on.rp.is_secondary_model_activated is True
    assert session.model_count(on.rp) == 2      # base 2 when a secondary is activated


# -- I9: the two halves of one file, one parse -------------------------------- #

#: The float field of each half, for the pairs below: `post_process_threshold` is the
#: only float in `Settings`, the twelve scales are the floats in `Selections`.
_FLOAT_PAIRS = (("post_process_threshold", "vr_voc_inst_secondary_model_scale"),)


@pytest.mark.parametrize("written", ["0.7", 0.7, 7, "7", "0.0"])
def test_a_number_a_widget_or_file_holds_loads_the_same_way_in_both_halves(written,
                                                                          tmp_path):
    """I9. `Selections.load`'s own docstring promises that coercion makes a value parse
    the same way wherever it came from -- file, widget text, or a bare constructor
    argument -- and the loader then refused the text `0.7` for a float field with its own
    `isinstance` gate, while `_coerce` parses it and the *sibling half of this same file*
    accepts it (`settings.load` routes every key through `_coerce`, settings.py:546).

    The consequence was not cosmetic: a hand-edited scale in the selections block made
    `Controller.__init__` raise before any window existed -- an app that will not launch,
    with no surface to say so on -- while the identical edit in the settings half was
    fine. Both halves now parse through the one table."""
    target = tmp_path / "settings.json"
    for s_key, sel_key in _FLOAT_PAIRS:
        target.write_text(json.dumps({s_key: written,
                                      "selections": {sel_key: written}}), encoding="utf-8")
        st = settings_module.load(target)
        sel = selections.load(target)
        assert getattr(st, s_key) == float(written), (s_key, getattr(st, s_key))
        assert getattr(sel, sel_key) == float(written), (sel_key, getattr(sel, sel_key))
        assert type(getattr(sel, sel_key)) is float, sel_key


@pytest.mark.parametrize("written", ["loud", "1_0", [], None])
def test_a_value_neither_half_can_parse_refuses_from_both_and_names_the_file(written,
                                                                            tmp_path):
    """The agreement has to run the other way too: what the shared table refuses, each
    half refuses -- and the selections loader keeps raising *its* `TypeError`, the one
    that names the file, because a corrupt config is a file-level diagnosis. That is the
    reason the gate stays in the loop instead of leaning on `__post_init__`; only its
    parser moves."""
    target = tmp_path / "settings.json"
    for s_key, sel_key in _FLOAT_PAIRS:
        target.write_text(json.dumps({s_key: written,
                                      "selections": {sel_key: written}}), encoding="utf-8")
        with pytest.raises((TypeError, ValueError)):
            settings_module.load(target)
        with pytest.raises(TypeError) as sel_exc:
            selections.load(target)
        message = str(sel_exc.value)
        assert str(target) in message, message
        assert sel_key in message, message


def test_the_selections_gate_still_refuses_a_bool_for_a_scale(tmp_path):
    """`_coerce`'s float branch refuses a bool (`a bool is not a scale`), so routing the
    loader through it cannot let `True` reach a scale as `1.0` -- the very confusion
    `test_a_wrongly_typed_selection_raises_instead_of_silently_passing` pins from the
    other side."""
    target = tmp_path / "settings.json"
    target.write_text(json.dumps({"selections": {"vr_voc_inst_secondary_model_scale":
                                   True}}), encoding="utf-8")
    with pytest.raises(TypeError, match="vr_voc_inst_secondary_model_scale"):
        selections.load(target)
