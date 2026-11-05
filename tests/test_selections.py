"""The 26 selections Plan 1 left unreachable, plus the 7 that address them.

Ruling R-T11-selections kept these out of `Settings` because the criterion is "the
engine reads it inside the model-data class" -- they arrive as `plan.build()`
arguments. So they get their own dataclass, and `Settings`' 50-field invariant is
untouched (spec §10, ruling 2).
"""
from __future__ import annotations

import json
from pathlib import Path

import catalog
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
