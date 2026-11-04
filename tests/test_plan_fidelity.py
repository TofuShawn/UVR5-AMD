"""Branch-review fidelity findings against `plan.py`, one test each.

`ModelData` was deleted from `UVR.py` on this branch, so `plan.build()` is the sole
authority for the 92 attributes `separate.py` reads.  The whole-branch review
compared the derived fields against the only ground truth still readable,
`git show 9ffd8f2:UVR.py`, and this file pins the five findings it raised, each
against the upstream lines the port has to match:

  * **mixer_path** -- upstream sets `BASE_PATH/lib_v5/mixer.ckpt` (`UVR.py:419`,
    built at `:243`) and then `get_mdx_model_path` (`UVR.py:530` -> `:709`)
    overwrites it with `MDX_MODELS_DIR/mixer_val.ckpt` (`:240`) for MDX models.
    A different file, in the *model* tree instead of the repo tree.
  * **the self-naming secondary slot** -- `UVR.py:677-678` stores the nested model
    and its scale, and `:681` clears only `is_secondary_model_activated`; the
    4-stem loop (`:633-637`) keeps every slot and `:640` recomputes the flag from
    the list.  `separate.py:1126-1129` then requires *both* the flag and a
    non-None list entry, so a nulled slot is a skipped blend.
  * **`vocal_splitter["is_activated"]` absent** -- upstream's gate is
    `UVR.py:7107`, which reads `is_set_vocal_splitter_var.get()`, a BooleanVar
    whose default is False (`gui_data/constants.py:703`).
  * **`compensate`** -- `UVR.py:584` indexes `self.model_data["compensate"]`; a
    missing key raises, it does not default.
  * **`aggression_setting`** -- `UVR.py:492` is `float(int(<var>) / 100)`.
  * **`is_roformer` / the karaoke trio** -- `UVR.py:541-542` sits in the MDX
    branch only, and `check_if_karaokee_model` (`:685-691`) is called only at
    `:520` (VR) and `:590` (MDX without `config_yaml`).

`tests/test_plan.py` stays the coverage gate; this file only adds the cases the
review found uncovered, so the helpers below are deliberately self-contained --
importing a sibling test module's fixtures would let another agent's edit to
`test_plan.py` break this file.
"""
import dataclasses
import pathlib

import pytest

import backends
import catalog
import plan
import settings as settings_module
import uvr_paths

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_ROOT = uvr_paths.MODEL_ROOT
VR_WEIGHT = "1_HP-UVR"
DEMUCS_WEIGHT = "955717e8-8726e21a.th"
MDX_WEIGHT = "UVR-MDX-NET-Inst_HQ_3"
CPU = backends.Backend(name="cpu", torch_device="cpu", ort_providers=["CPUExecutionProvider"])

#: `gui_data.constants.AUTO_SELECT`, spelled as `Settings` stores it; imported
#: through `plan` so the test cannot pass by comparing two local literals.
AUTO_SELECT = plan.AUTO_SELECT


def _cat(tmp_path):
    return catalog.Catalog(MODEL_ROOT, writable_dir=tmp_path / "appdata")


def _build(tmp_path, name, *, cat=None, s=None, selections=None, spec=None):
    cat = cat or _cat(tmp_path)
    return plan.build(settings=s or settings_module.Settings(), catalog=cat,
                      backend=CPU, spec=spec or cat.resolve(name), selections=selections)


def _all_slots(secondary: str, scale: str = "0.5") -> dict:
    """The four stem slots in each per-arch group (`UVR.py:1634/1644/1654`)."""
    slots = {}
    for stem in ("voc_inst", "other", "drums", "bass"):
        slots[f"{stem}_secondary_model"] = secondary
        slots[f"{stem}_secondary_model_scale"] = scale
    return {"vr": dict(slots), "mdx": dict(slots), "demucs": dict(slots)}


def _spec_with(tmp_path, name, **extra_params) -> catalog.ModelSpec:
    """The real spec with its params row *replaced*, to exercise the key lookups."""
    spec = _cat(tmp_path).resolve(name)
    return dataclasses.replace(spec, model_data={**(spec.model_data or {}), **extra_params})


# --------------------------------------------------------------------------- #
# Important 1 -- mixer_path is re-derived in the MDX branch
# --------------------------------------------------------------------------- #

def test_mixer_path_is_the_repo_ckpt_for_vr_and_demucs(tmp_path):
    """`UVR.py:419` (via `:243`): `BASE_PATH/lib_v5/mixer.ckpt`.

    That default is what survives for the two archs whose branch never calls
    `get_mdx_model_path`: `:530` is inside `if self.process_method == MDX_ARCH_TYPE`,
    so VR and Demucs keep the repo-tree file.
    """
    for name in (VR_WEIGHT, DEMUCS_WEIGHT):
        p = _build(tmp_path, name)
        assert p.mixer_path == ROOT / "lib_v5" / "mixer.ckpt", (
            f"{name}: mixer_path must stay the UVR.py:419/:243 default -- only the MDX "
            "branch reaches UVR.py:709's overwrite.")


def test_mixer_path_is_mixer_val_under_the_model_root_for_mdx(tmp_path):
    """`UVR.py:530` -> `:709`: `MDX_MODELS_DIR/mixer_val.ckpt`, i.e. `:240`/`:238`.

    Two different files in two different trees: the default is
    `BASE_PATH/lib_v5/mixer.ckpt` (`:243`), the MDX value is
    `MODELS_DIR/MDX_Net_Models/mixer_val.ckpt` (`:238`+`:240`).  Fork-equivalent of
    `MODELS_DIR` is the catalog's active model root, not the repo tree.  Unreachable
    at run time today because `is_mixer_mode` is hard-False (`UVR.py:414`,
    `plan.py:339`) and `lib_v5/mdxnet.py:123`'s `Mixer` is never constructed -- which
    is why no golden test could have caught the missing branch, and why this pins the
    value `separate.py:164` would hand on.
    """
    cat = _cat(tmp_path)
    p = _build(tmp_path, MDX_WEIGHT, cat=cat)
    assert p.mixer_path == cat.model_root / "MDX_Net_Models" / "mixer_val.ckpt", (
        f"the MDX plan still carries the UVR.py:419 repo default ({p.mixer_path}); "
        "UVR.py:709 overwrites it inside get_mdx_model_path, called at :530.")
    assert p.mixer_path != ROOT / "lib_v5" / "mixer.ckpt"
    assert p.mixer_path.parent.name == "MDX_Net_Models", (
        "UVR.py:709 builds the path in MDX_MODELS_DIR (:240), the model tree, not the "
        "repo tree -- the directory is part of the finding, not a detail.")
    assert p.mixer_path.name == "mixer_val.ckpt", "not mixer.ckpt; different file"
    # And the gate that keeps it unread stays shut: fixing the path must not, and
    # does not, switch mixer mode on.
    assert p.is_mixer_mode is False                                  # UVR.py:414
    assert _build(tmp_path, VR_WEIGHT).is_mixer_mode is False


# --------------------------------------------------------------------------- #
# Important 2 -- a self-naming stem slot keeps its nested plan
# --------------------------------------------------------------------------- #

def test_all_stem_ensemble_keeps_a_slot_that_names_the_primarys_own_weight(tmp_path):
    """`UVR.py:633-641` blend it, `:677-681` clears only the flag.

    `demucs_stems` is ALL_STEMS by default, so each of the four stems resolves its
    own slot.  When a slot names the primary's own weight, upstream appends the
    nested plan anyway (`:635`, fed by `:677`) and only `:681` inside
    `secondary_model_data` clears `is_secondary_model_activated` -- which `:640` then
    recomputes from the list, so it is True again.  `separate.py:1126-1129` needs
    both the flag *and* the list entry, so nulling the slot is what made this fork
    skip a blend upstream performs.

    `tests/test_plan.py`'s existing case uses VR as the secondary for a Demucs
    primary, i.e. a different basename, so it cannot reach this branch.
    """
    cat = _cat(tmp_path)
    s = settings_module.Settings(demucs_stems=plan.ALL_STEMS,
                                 demucs_is_secondary_model_activate=True)
    assert s.demucs_stems == plan.ALL_STEMS, "the case depends on the All Stems value"
    p = _build(tmp_path, DEMUCS_WEIGHT, cat=cat, s=s, selections=_all_slots(DEMUCS_WEIGHT))

    assert len(p.secondary_model_4_stem) == 4 and len(p.secondary_model_4_stem_scale) == 4
    for index, (stem_plan, stem) in enumerate(zip(p.secondary_model_4_stem,
                                                  plan.DEMUCS_4_SOURCE_LIST)):
        assert isinstance(stem_plan, plan.RunPlan), (
            f"slot {index} ({stem}) was zeroed because the selection names the primary's "
            "own weight; UVR.py:635 appends whatever :677 stored, and only :681 clears the "
            "flag -- separate.py:1127 would then skip the blend for that stem.")
        assert stem_plan.model_basename == p.model_basename
        assert stem_plan.is_secondary_model is True                    # UVR.py:7077
        assert p.secondary_model_4_stem_scale[index] == 0.5            # UVR.py:678/:7073
    assert p.is_secondary_model_activated is True, (
        "UVR.py:640 recomputes the flag as any(i is not None for i in the list); a list "
        "of self-naming plans is still a list of plans.")
    # `secondary_model`/`_scale` hold the LAST slot after the loop, both here (`:637-638`
    # equivalents) and upstream.
    assert p.secondary_model is p.secondary_model_4_stem[-1]
    assert p.secondary_model_scale == p.secondary_model_4_stem_scale[-1]


def test_a_different_basename_in_the_same_slot_still_blends(tmp_path):
    """The control: the case `tests/test_plan.py` already covers must not move.

    VR-as-secondary-for-Demucs has a different basename, so the flag is True for the
    reason it always was -- this is the half that proves the previous test's
    expectation is about the *basename*, not about the loop ignoring it wholesale.
    """
    s = settings_module.Settings(demucs_stems=plan.ALL_STEMS,
                                 demucs_is_secondary_model_activate=True)
    p = _build(tmp_path, DEMUCS_WEIGHT, s=s, selections=_all_slots(VR_WEIGHT))
    assert all(isinstance(i, plan.RunPlan) for i in p.secondary_model_4_stem)
    assert p.secondary_model_4_stem[0].model_basename != p.model_basename
    assert p.is_secondary_model_activated is True                      # UVR.py:640


def test_single_nested_self_naming_keeps_the_plan_and_clears_only_the_flag(tmp_path):
    """`UVR.py:677-681` for the one-slot path (`:647-648`).

    The basename gate lives here, at the caller that upstream's
    `secondary_model_data` compares inside -- and it decides
    `is_secondary_model_activated` only.  The plan and the scale stay on the object,
    because `:677-678` write them before `:681` reads them; `separate.py:662`/`:848`
    gate the model behind `is_secondary_model_activated and self.secondary_model`, so
    a flag that is False is already the whole stop.
    """
    p = _build(tmp_path, VR_WEIGHT,
               s=settings_module.Settings(vr_is_secondary_model_activate=True),
               selections=_all_slots(VR_WEIGHT))
    assert isinstance(p.secondary_model, plan.RunPlan), (
        "UVR.py:677-678 keep the nested model; only :681 clears the activation flag.")
    assert p.secondary_model.model_basename == p.model_basename
    assert p.secondary_model_scale == 0.5
    assert p.is_secondary_model_activated is False                     # UVR.py:681
    assert p.secondary_model_4_stem == [], "the 4-stem lists are Demucs-only (:632)"


# --------------------------------------------------------------------------- #
# Minor 3 -- an absent `is_activated` means switched off
# --------------------------------------------------------------------------- #

def test_named_splitter_without_an_activation_flag_is_not_activated(tmp_path):
    """`UVR.py:7107` gates on `is_set_vocal_splitter_var.get()`.

    The widget is a BooleanVar whose table default is False
    (`gui_data/constants.py:703`), so "the selection carries a name but no flag" is
    the switched-off case, not the switched-on one.  A default of True hands a
    second model to a caller that never said it wanted one.
    """
    off = _build(tmp_path, VR_WEIGHT, selections={"vocal_splitter": {"name": DEMUCS_WEIGHT}})
    assert off.vocal_split_model is None, (
        "plan.py read `is_activated` with a True default; UVR.py:7107 requires the "
        "activation var, whose default is False (gui_data/constants.py:703).")
    assert off.is_sec_bv_rebalance is False

    on = _build(tmp_path, VR_WEIGHT,
                selections={"vocal_splitter": {"name": DEMUCS_WEIGHT, "is_activated": True}})
    assert isinstance(on.vocal_split_model, plan.RunPlan)
    assert on.vocal_split_model.is_vocal_split_model is True           # UVR.py:7108
    assert on.vocal_split_model.vocal_split_model is None              # UVR.py:667


def test_absent_name_still_short_circuits_before_the_flag(tmp_path):
    """Why the default is not a behaviour change for the inputs `build()` sees.

    `catalog.vocal_split_for` returns None on `not requested` (`catalog.py:804`)
    before it looks at `is_activated`, and no production caller of `build()` puts a
    bare `{"name": ...}` in the selection (`separate_cli.py:486` passes
    `selections={}`), so the flag default only decides the named-but-unflagged case.
    """
    for sel in ({}, {"vocal_splitter": {}}, {"vocal_splitter": {"name": catalog.NO_MODEL}},
                {"vocal_splitter": {"name": ""}}):
        assert _build(tmp_path, VR_WEIGHT, selections=sel).vocal_split_model is None, sel
    # ... and a name that cannot be resolved is dropped rather than raised (:7081-7084).
    assert _build(tmp_path, VR_WEIGHT,
                  selections={"vocal_splitter": {"name": "gone", "is_activated": True}}
                  ).vocal_split_model is None


# --------------------------------------------------------------------------- #
# Minor 4 -- a params row with no `compensate` key raises
# --------------------------------------------------------------------------- #

def test_missing_compensate_key_raises_instead_of_leaking_the_sentinel(tmp_path):
    """`UVR.py:584` indexes `self.model_data["compensate"]`.

    The fallback the review removed returned `AUTO_SELECT` -- a *string* -- into
    `RunPlan.compensate`, and `separate.py:717` multiplies the decoded source by it
    (`source * self.compensate`), after the weights are loaded.  Upstream raises at
    build time; so does the plan, and the message names the arch, the hash and the
    key so the operator knows which table row to fix.
    """
    spec = _spec_with(tmp_path, MDX_WEIGHT)
    params = dict(spec.model_data)
    params.pop("compensate")
    broken = dataclasses.replace(spec, model_data=params)
    s = settings_module.Settings()
    assert s.compensate == AUTO_SELECT, "the AUTO_SELECT branch is the one under test"
    with pytest.raises(plan.PlanError) as caught:
        _build(tmp_path, None, spec=broken, s=s)
    message = str(caught.value)
    assert MDX_WEIGHT in message and "compensate" in message, message
    assert broken.arch in message and str(broken.hash) in message, (
        f"the error must name arch/hash/key: {message}")
    assert AUTO_SELECT in message


def test_present_compensate_key_and_manual_value_both_survive(tmp_path):
    """The control: the two paths `UVR.py:584` actually takes still work."""
    assert _build(tmp_path, MDX_WEIGHT).compensate == 1.022            # params row
    manual = _build(tmp_path, MDX_WEIGHT, s=settings_module.Settings(compensate="1.5"))
    assert manual.compensate == 1.5 and isinstance(manual.compensate, float)


# --------------------------------------------------------------------------- #
# Minor 5 -- aggression_setting keeps upstream's int() truncation
# --------------------------------------------------------------------------- #

def test_aggression_setting_is_truncated_then_divided(tmp_path):
    """`UVR.py:492` is `float(int(root.aggression_setting_var.get()) / 100)`.

    `Settings` types the field `int` and `_coerce` refuses `'5.7'`/`5.7`, so through
    a typed instance the `int()` is a no-op -- which is why the dropped `int()` was
    invisible.  Forcing the attribute (the route `settings.py:290` names as the one
    `_coerce` cannot police) shows the two readings part: the GUI can only ever mean
    the integer step, so the plan must truncate rather than divide through.
    """
    for value, expected in ((5, 0.05), (7, 0.07), (10, 0.1)):
        assert _build(tmp_path, VR_WEIGHT,
                      s=settings_module.Settings(aggression_setting=value)
                      ).aggression_setting == expected

    fractional = settings_module.Settings()
    object.__setattr__(fractional, "aggression_setting", "5.7")
    p = _build(tmp_path, VR_WEIGHT, s=fractional)
    assert p.aggression_setting == 0.05, (
        "the int() of UVR.py:492 was missing: 5.7/100 = 0.057 reached "
        "separate.py:371, a value the combobox cannot produce.")


# --------------------------------------------------------------------------- #
# Minor 7 -- the karaoke/roformer params reads are branch-gated
# --------------------------------------------------------------------------- #

def test_is_roformer_is_only_read_in_the_mdx_branch(tmp_path):
    """`UVR.py:471` initialises False; `:541-542` is the only write, inside MDX.

    A VR params row that carries `is_roformer` says nothing to the VR engine --
    `separate.py:857`/`:960`/`:969` read the flag on the MDX path -- so upstream
    leaves it False, and now so does the plan.
    """
    vr = _build(tmp_path, VR_WEIGHT, spec=_spec_with(tmp_path, VR_WEIGHT, is_roformer=True))
    assert vr.is_roformer is False, (
        "a VR plan adopted is_roformer from a row UVR.py:541-542 never reads; the flag "
        "switches separate.py:960's denoise path off for a model that is not roformer.")
    de = _build(tmp_path, DEMUCS_WEIGHT, spec=_spec_with(tmp_path, DEMUCS_WEIGHT,
                                                        is_roformer=True))
    assert de.is_roformer is False
    mdx = _build(tmp_path, None, spec=_spec_with(tmp_path, MDX_WEIGHT, is_roformer=True))
    assert mdx.is_roformer is True                                     # UVR.py:541-542


def test_karaoke_flags_are_read_for_vr_and_plain_mdx_and_not_for_demucs(tmp_path):
    """`check_if_karaokee_model` (`UVR.py:685-691`) is called at `:520` and `:590`.

    `:590` sits in the MDX `else` of `if "config_yaml" in self.model_data` (`:544`),
    so an mdx_c row is skipped too -- and Demucs never calls it at all.  What the
    skip protects is not cosmetic: `separate.py:451-452` refuses a karaoke/bv model
    on one side of the mix, `:530-532` rebalances stems off `is_bv_model_rebalenced`,
    and `UVR.py:658` picks the splitter's own primary stem from `is_bv_model`.  A
    Demucs row that happens to carry the keys would silently change all three.
    """
    keys = {plan.IS_KARAOKEE: True, plan.IS_BV_MODEL: True, plan.IS_BV_MODEL_REBAL: 0.5}
    vr = _build(tmp_path, None, spec=_spec_with(tmp_path, VR_WEIGHT, **keys))
    assert (vr.is_karaoke, vr.is_bv_model, vr.bv_model_rebalance) == (True, True, 0.5)
    mdx = _build(tmp_path, None, spec=_spec_with(tmp_path, MDX_WEIGHT, **keys))
    assert (mdx.is_karaoke, mdx.is_bv_model, mdx.bv_model_rebalance) == (True, True, 0.5)

    demucs = _build(tmp_path, None, spec=_spec_with(tmp_path, DEMUCS_WEIGHT, **keys))
    assert (demucs.is_karaoke, demucs.is_bv_model, demucs.bv_model_rebalance) == (False,
                                                                                 False, 0), (
        "UVR.py:600-613 never calls check_if_karaokee_model, so a Demucs model keeps the "
        ":457-459 initialisers however its row is tagged.")
    # The widening that mattered: a Demucs splitter under a VR primary.  Upstream's
    # :672-673 reads `vocal_split_model.bv_model_rebalance`, which for Demucs is 0.
    split = _build(tmp_path, VR_WEIGHT,
                   selections={"vocal_splitter": {"name": DEMUCS_WEIGHT, "is_activated": True}},
                   spec=_cat(tmp_path).resolve(VR_WEIGHT))
    assert split.vocal_split_model is not None
    assert split.vocal_split_model.bv_model_rebalance == 0
    assert split.is_sec_bv_rebalance is False                          # UVR.py:672-673
