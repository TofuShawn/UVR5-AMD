"""Task 11b: `plan.RunPlan` carries the 92 attributes `separate.py` reads.

The one test that matters most here is `test_runplan_covers_every_attribute_separate_reads`.
It is the objective gate for Task 11c: it re-derives the read set from
`separate.py`'s AST *at test time* and asserts it against `RunPlan`'s fields *and* --
since 11d -- against a hand-written literal of the 92 names.  Without it, a rename of
one read in the MDX path would go unnoticed on this machine: the SNR gate covers `vr`
and `demucs` only (`pins.json["archs_blocked"]` = `mdx_c`, `roformer`, `onnx`;
`test_golden_rocm.py:137` records that `mdx` has no CPU baseline).

The rest of the file guards the two things that a name-by-name port can still get
wrong.  *Values*: 34 fields are "the same name in `Settings`", but `ModelData`
coerced or re-selected most of them (`UVR.py:391-396`, `:492-499`, `:610-611`), so a
copy-through is not always a copy-through.  *Selection*: the 26 secondary-model
choices are `build()` arguments (R-T11-selections), and the per-arch group is picked
by `process_method`, not read once globally (`UVR.py:7052-7057`).

`crop_size` is asserted *absent*.  It is a `Settings` field and it appears in the
plan document's Step-1 sketch, but `separate.py` never reads it off the plan object,
so R-T11-scope keeps it out of the 92 -- see the brief's ruling section.

## 11d changed the posture on pinning the 92

11b's brief said "do not hardcode the 92 names", and that was right *for deriving
the count*: a pasted list would only have proved that the file was written down
correctly once, and the live sweep is what found the 4 two-read lines.  11d renames
the base those reads hang off (`model_data` -> `plan`), so the sweep now has a second
job: catching an edit that re-points a read to the wrong object, or drops one.  A
sweep compared only to `RunPlan`'s fields cannot do that, because the fields and the
reads are both 92 and both move together.  So `PINNED_PLAN_READS` is a hand-written
literal of the 92 names, asserted by *exact equality in both directions* -- the
`pin_drift` pattern from `tests/test_engine_purity.py`, whose whole reason to exist
is that a subset-only check lets a dropped read pass silently.  The live sweep and
the literal are independent sources; disagreement between them is the finding.
"""
import ast
import dataclasses
import pathlib

import pytest

import backends
import catalog
import plan
import settings as settings_module
import uvr_paths

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_ROOT = uvr_paths.MODEL_ROOT          # the *installed* root when UVR_MODEL_ROOT is set
VR_WEIGHT = "1_HP-UVR"
DEMUCS_WEIGHT = "955717e8-8726e21a.th"
MDX_WEIGHT = "UVR-MDX-NET-Inst_HQ_3"
#: The base names whose attribute reads count as plan reads.  Since 11d the single
#: constructor parameter is `plan` and `separate.py` reads the plan *only* through
#: it -- the three old bases (`model_data`, `mdx_net_model_data`,
#: `demucs_model_data`) are gone from the file, so this is a one-element set and
#: `grep -c model_data separate.py` is 0.  A set rather than a bare string on
#: purpose: the sweep resolves nested attributes off any base listed here, so a
#: second plan-bearing parameter is added to this set, not to the walker.
PLAN_BASE = {"plan"}

CPU = backends.Backend(name="cpu", torch_device="cpu",
                       ort_providers=["CPUExecutionProvider"])


def _cat(tmp_path):
    return catalog.Catalog(MODEL_ROOT, writable_dir=tmp_path / "appdata")


def _build(tmp_path, name, *, cat=None, s=None, selections=None,
           backend=CPU, spec=None):
    """`plan.build` with the composition inputs defaulting to the real fixtures."""
    cat = cat or _cat(tmp_path)
    return plan.build(settings=s or settings_module.Settings(), catalog=cat,
                      backend=backend, spec=spec or cat.resolve(name),
                      selections=selections)


# --------------------------------------------------------------------------- #
# 1. the coverage gate
# --------------------------------------------------------------------------- #

#: The 92 attribute names `separate.py` reads off its single `plan` parameter,
#: written out by hand.  This literal and the AST sweep in `model_reads` are
#: independent sources -- the sweep walks the file, this list is what 11d measured
#: it to be -- so the equality below is a finding rather than a tautology.
#:
#: If a name here disappears the file stopped reading it, which is how a plan field
#: becomes dead weight; if a name appears that is not here, someone added a read
#: without growing `RunPlan` (or re-pointed one at the wrong object).  Either way
#: the fix is in the same commit as the change, with a stated reason -- do not
#: weaken this to a subset check, which is exactly what lets a dropped read pass.
PINNED_PLAN_READS = frozenset({
    "DENOISER_MODEL", "DEVERBER_MODEL", "aggression_setting", "batch_size",
    "bv_model_rebalance", "chunks", "compensate", "demucs_source_list",
    "demucs_source_map", "demucs_stem_count", "demucs_stems", "demucs_version",
    "deverb_vocal_opt", "device_set", "ensemble_primary_stem", "ensemble_secondary_stem",
    "is_bv_model", "is_chunk_demucs", "is_demucs_combine_stems", "is_demucs_pre_proc_model_inst_mix",
    "is_denoise", "is_denoise_model", "is_deverb_vocals", "is_ensemble_mode",
    "is_gpu_conversion", "is_high_end_process", "is_inst_only_voc_splitter", "is_invert_spec",
    "is_karaoke", "is_match_frequency_pitch", "is_mdx_c", "is_mdx_c_seg_def",
    "is_mdx_ckpt", "is_mdx_combine_stems", "is_mixer_mode", "is_multi_stem_ensemble",
    "is_normalization", "is_pitch_change", "is_post_process", "is_pre_proc_model",
    "is_primary_model_primary_stem_only", "is_primary_model_secondary_stem_only", "is_primary_stem_only", "is_roformer",
    "is_save_inst_vocal_splitter", "is_save_vocal_only", "is_sec_bv_rebalance", "is_secondary_model",
    "is_secondary_model_activated", "is_secondary_stem_only", "is_split_mode", "is_target_instrument",
    "is_tta", "is_vocal_split_model", "is_vr_51_model", "margin",
    "mdx_batch_size", "mdx_c_configs", "mdx_dim_f_set", "mdx_dim_t_set",
    "mdx_n_fft_scale_set", "mdx_segment_size", "mdxnet_stem_select", "mixer_path",
    "model_basename", "model_capacity", "model_name", "model_path",
    "model_samplerate", "mp3_bit_set", "overlap", "overlap_mdx",
    "overlap_mdx23", "post_process_threshold", "pre_proc_model", "primary_model_primary_stem",
    "primary_stem", "primary_stem_native", "process_method", "save_format",
    "secondary_model", "secondary_model_4_stem", "secondary_model_4_stem_scale", "secondary_model_scale",
    "secondary_stem", "segment", "semitone_shift", "shifts",
    "vocal_split_model", "vr_model_param", "wav_type_set", "window_size",
})

#: `separate.py` may never import these three.  The design's direction is
#: `app -> engine` and `app -> settings`, with composition only in the controller;
#: `plan`/`settings`/`catalog` are the controller's inputs, and an engine that
#: reaches back for them re-couples the layer 11 exists to decouple.  The temptation
#: is concrete: `plan: RunPlan` is a nicer annotation than `plan: "RunPlan"`, and
#: writing it costs one `import plan` at the top of the file.
FORBIDDEN_SEPARATE_IMPORTS = frozenset({"plan", "settings", "catalog"})


def model_reads(source_text: str) -> dict[str, list[int]]:
    """`{attr: [linenos]}` for every attribute read off the plan object.

    The base is resolved through nested `Attribute` nodes, so `plan.foo.bar` and
    `plan.foo()[0]` both count as a read of `foo`; a name that merely *ends* like a
    plan object (`self.plan`) does not, because its outermost base is `self`.  Keyed
    by attribute, not by line: four lines in `separate.py` read two attributes each,
    and a line-keyed dict silently loses 4 of the 92.
    """
    found: dict[str, list[int]] = {}
    for node in ast.walk(ast.parse(source_text)):
        if not isinstance(node, ast.Attribute):
            continue
        base = node.value
        while isinstance(base, ast.Attribute):
            base = base.value
        if isinstance(base, ast.Name) and base.id in PLAN_BASE:
            found.setdefault(node.attr, []).append(node.lineno)
    return found


def plan_read_drift(live) -> str:
    """A pin failure a human can act on: what left, what arrived, what to do.

    Same shape as `pin_drift` in `tests/test_engine_purity.py`, for the same reason:
    the two sides have to be able to disagree, and the message has to say which one
    moved.  A one-directional (`pinned <= live`) check would let a dropped read
    through -- the exact failure this pin exists to prevent.
    """
    now = frozenset(live)
    missing = sorted(PINNED_PLAN_READS - now)
    extra = sorted(now - PINNED_PLAN_READS)
    return (
        f"separate.py's `plan.` reads have drifted from PINNED_PLAN_READS.\n"
        f"  MISSING (the file no longer reads it -- so RunPlan now carries a field "
        f"nothing consumes, and the 92 is no longer the 92): {missing}\n"
        f"  NOT IN the pin (the file reads a name the pin never saw -- a new field "
        f"is needed, or a read was re-pointed/renamed by mistake): {extra}\n"
        f"  pinned = {sorted(PINNED_PLAN_READS)}\n"
        f"  live   = {sorted(now)}\n"
        f"  Do one of two things: put the read back, or -- if the change is "
        f"deliberate -- edit PINNED_PLAN_READS *and* plan.RunPlan's fields in the "
        f"SAME commit and say why, including the count in "
        f"test_runplan_field_count_matches_the_ruling. Do not delete the pin, do "
        f"not weaken it to a subset check, and do not rebuild it from the sweep: "
        f"that is the circular self-audit this test exists to remove.")


def test_runplan_covers_every_attribute_separate_reads():
    reads = set(model_reads((ROOT / "separate.py").read_text(encoding="utf-8")))
    fields = {f.name for f in dataclasses.fields(plan.RunPlan)}
    # First, because the pin names the offending attribute; the field-side asserts
    # below say what is missing from RunPlan, which is the less useful headline.
    assert frozenset(reads) == PINNED_PLAN_READS, plan_read_drift(reads)
    assert len(PINNED_PLAN_READS) == 92, (
        f"PINNED_PLAN_READS has {len(PINNED_PLAN_READS)} names, not 92. The literal "
        "and the count in test_runplan_field_count_matches_the_ruling are the same "
        "ruling stated twice; fix both or neither.")
    missing = sorted(reads - fields)
    assert not missing, (
        f"separate.py reads {len(missing)} attribute(s) no RunPlan field carries: "
        f"{missing} -- the plan cannot be built from the catalog while the name is "
        "missing, and adding it here needs the same name, not a tidier one.")
    assert len(reads) == 92, (
        f"the plan-object read set is {len(reads)}, not 92. The count is part of "
        "ruling R-T11-shape: if a read genuinely appeared, RunPlan grows to match it "
        "and this number, the brief and the ruling change in the SAME commit with a "
        "stated reason; if one disappeared, the engine is further off than 11d left it.")
    extra = sorted(fields - reads)
    assert not extra, (
        f"RunPlan carries {len(extra)} field(s) separate.py never reads: {extra}. "
        "R-T11-scope says exactly the 92 -- anything else needs a reason in the "
        "commit that adds it.")


def test_coverage_check_is_not_decorative_on_a_renamed_read():
    """The gate bites on the *read* side too, not only the field side."""
    text = (ROOT / "separate.py").read_text(encoding="utf-8")
    before = set(model_reads(text))
    renamed = set(model_reads(text.replace("plan.is_mdx_c_seg_def",
                                           "plan.is_mdx_c_seg_def_RENAMED", 1)))
    assert "is_mdx_c_seg_def_RENAMED" in renamed - before
    assert {f.name for f in dataclasses.fields(plan.RunPlan)} >= before
    assert not {f.name for f in dataclasses.fields(plan.RunPlan)} >= renamed


def test_the_plan_read_pin_bites_in_both_directions():
    """PINNED_PLAN_READS can fail on its own, with the sabotage done in memory.

    `test_coverage_check_is_not_decorative_on_a_renamed_read` proves the *field*
    side; this proves the pin side, which is the half 11d added.  Both directions
    are exercised because a pin that only reports extras is a subset check wearing
    a frozenset.
    """
    text = (ROOT / "separate.py").read_text(encoding="utf-8")
    live = frozenset(model_reads(text))
    assert live == PINNED_PLAN_READS, plan_read_drift(live)

    # EXTRA side -- a read re-pointed at a name nothing here has ever claimed.
    renamed = frozenset(model_reads(
        text.replace("plan.window_size", "plan.window_size_RENAMED")))
    assert renamed - PINNED_PLAN_READS == frozenset({"window_size_RENAMED"}), (
        "the mutation did not produce an extra name, so this control proves nothing")
    assert "window_size_RENAMED" in plan_read_drift(renamed)
    assert "NOT IN the pin" in plan_read_drift(renamed)

    # MISSING side -- a read simply taken away (re-pointed at `self`, which the
    # sweep deliberately does not follow).
    dropped = frozenset(model_reads(
        text.replace("plan.window_size", "self.window_size")))
    assert PINNED_PLAN_READS - dropped == frozenset({"window_size"}), (
        "the mutation did not remove a pinned name, so this control proves nothing")
    assert "MISSING" in plan_read_drift(dropped)
    assert "window_size" in plan_read_drift(dropped)


def test_separate_py_does_not_import_plan_settings_or_catalog():
    """The dependency direction stays `app -> engine`, never engine -> plan.

    11d renames the parameter to `plan`, which makes `import plan` look free: the
    annotation would read `plan: RunPlan` and stop being a string.  It is not free
    -- that import is the engine reaching for the controller's input -- so this
    test exists to make the temptation cost a red build instead of a review comment.
    """
    tree = ast.parse((ROOT / "separate.py").read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    hit = sorted(roots & FORBIDDEN_SEPARATE_IMPORTS)
    assert not hit, (
        f"separate.py imports {hit}. plan/settings/catalog are composed by the "
        "controller and handed to the engine as already-built objects; importing "
        "them here inverts the direction Task 11 exists to establish. Keep the "
        'annotation the string "RunPlan".')


# --------------------------------------------------------------------------- #
# 2. no GUI, structurally
# --------------------------------------------------------------------------- #

ALLOWED_PLAN_IMPORTS = frozenset({
    "__future__", "dataclasses", "os", "pathlib", "typing",
    "settings", "catalog", "backends",
    "gui_data.constants", "yaml", "lib_v5.vr_network.model_param_init",
})

#: Must never be reachable. The allow-list already rejects all of them; they are
#: spelled out because the day someone "just adds" one of these the diff is a line.
GUI_IMPORT_NAMES = (
    "tkinter", "tkinter.filedialog", "PySide2", "PySide6", "PIL", "PIL.ImageTk",
    "screeninfo", "sv_ttk", "gui_data", "gui_data.tkinterdnd2", "gui_data.sv_ttk",
    "gui_data.error_handling", "gui_data.app_size_values", "gui_data.old_data_check",
    "UVR", "separate",
)


def plan_imports() -> set[str]:
    """Every module `plan.py` imports, by AST, dotted and submodule-accurate."""
    tree = ast.parse((ROOT / "plan.py").read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:                     # relative import: the module itself
                found.append("." * node.level + (node.module or ""))
            elif node.module:
                found.add(node.module)
    return found


def test_plan_imports_are_exactly_the_allow_list():
    found = plan_imports()
    assert found == ALLOWED_PLAN_IMPORTS or found <= ALLOWED_PLAN_IMPORTS, (
        f"plan.py imports {sorted(found - ALLOWED_PLAN_IMPORTS)} that the allow-list "
        f"does not name; the list is {sorted(ALLOWED_PLAN_IMPORTS)}. Extend it "
        "deliberately in this same file -- `plan.py` is the composition layer, so a "
        "new import here is a new dependency for every caller of `build()`.")
    assert found == ALLOWED_PLAN_IMPORTS, (
        f"the allow-list names {sorted(ALLOWED_PLAN_IMPORTS - found)} that plan.py no "
        "longer imports; a list wider than the module hides whatever returns.")


def test_plan_never_imports_the_gui():
    found = plan_imports()
    for name in GUI_IMPORT_NAMES:
        # `gui_data` is a namespace package with no `__init__.py`; the pure-data
        # `gui_data.constants` is on the allow-list, so that one name is matched
        # exactly while every other entry is matched as a package prefix.
        prefix_ok = name != "gui_data"
        hit = any(f == name or (prefix_ok and f.startswith(name + ".")) for f in found)
        assert not hit, (
            f"plan.py imports {name}, which puts the tkinter GUI back on the "
            "engine's input path -- the one thing Task 11 exists to remove.")
    assert "tkinter" not in found
    gui_data_imports = {f for f in found if f == "gui_data" or f.startswith("gui_data.")}
    assert gui_data_imports == {"gui_data.constants"}, (
        f"plan.py reaches for {sorted(gui_data_imports)}; only the pure-data "
        "`gui_data.constants` is allowed, which is the same carve-out "
        "test_engine_purity's blocker makes.")


def test_runplan_is_frozen(tmp_path):
    p = _build(tmp_path, VR_WEIGHT)
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.window_size = 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.secondary_model = None


# --------------------------------------------------------------------------- #
# 3. build() on the real fixtures
# --------------------------------------------------------------------------- #

def test_vr_plan_takes_from_the_spec_the_params_and_settings(tmp_path):
    s = settings_module.Settings(window_size=352, crop_size=352, wav_type_set="FLOAT")
    p = _build(tmp_path, VR_WEIGHT, s=s)
    spec = _cat(tmp_path).resolve(VR_WEIGHT)
    assert p.process_method == spec.process_method == "VR Arc"   # from the spec
    assert p.primary_stem == spec.primary_stem                   # from the spec
    assert p.secondary_stem == spec.secondary_stem               # from the spec
    assert p.model_name == VR_WEIGHT and p.model_path == spec.weights_path
    assert p.model_basename == "1_HP-UVR"                        # UVR.py:616
    assert p.window_size == 352 and p.wav_type_set == "FLOAT"    # from Settings
    assert "crop_size" not in {f.name for f in dataclasses.fields(plan.RunPlan)}, (
        "crop_size is a Settings field that separate.py never reads off the model "
        "object; R-T11-scope keeps RunPlan to the 92 (see test_settings for it).")
    assert p.vr_model_param is not None and p.vr_model_param.param["sr"] == p.model_samplerate
    assert p.vr_model_param is not spec.model_data          # the params object, not the table row


def test_demucs_plan_carries_the_four_demucs_keys_and_the_vr_one_does_not(tmp_path):
    demucs = _build(tmp_path, DEMUCS_WEIGHT)
    assert demucs.demucs_version == "v4"                         # UVR.py:725-729
    assert demucs.demucs_source_list and demucs.demucs_source_map
    assert demucs.demucs_stem_count == 4
    assert demucs.demucs_source_map == _cat(tmp_path).resolve(DEMUCS_WEIGHT).model_data["demucs_source_map"]
    vr = _build(tmp_path, VR_WEIGHT)
    for name in ("demucs_version", "demucs_source_list", "demucs_source_map",
                 "demucs_stem_count"):
        value = getattr(vr, name)
        assert value in (None, [], {}, 0), (
            f"the VR plan invented {name}={value!r}; upstream only assigns it in "
            "`get_demucs_model_data` (UVR.py:723-738).")
    assert vr.mdx_dim_f_set is None and vr.mdx_n_fft_scale_set is None  # UVR.py:404-408
    assert demucs.vr_model_param is None                         # UVR.py:511-514


def test_backend_reaches_the_plan_only_through_fields_that_exist(tmp_path):
    """`backend` is an argument (R-T11-scope) and is not consumed by the 92.

    `separate.py:236-276` asks `backends.resolve()` for the device and the providers
    at run time; upstream's `ModelData` never read either.  Asserting the two plans
    are equal is what keeps a future "helpful" `device` field from arriving without
    a conversation: the 92 contain `device_set` (a Settings widget value) and no
    `device`.
    """
    names = {f.name for f in dataclasses.fields(plan.RunPlan)}
    assert "device_set" in names and "device" not in names
    other = backends.Backend(name="rocm", torch_device="cuda",
                             ort_providers=["DmlExecutionProvider", "CPUExecutionProvider"])
    s = settings_module.Settings(device_set="cuda:1")
    a = _build(tmp_path, VR_WEIGHT, s=s)
    b = _build(tmp_path, VR_WEIGHT, s=s, backend=other)
    # `vr_model_param` is a `ModelParameters` instance, and that class defines no
    # `__eq__`, so two loads of the same file are "different"; every other field --
    # all 91 of them -- must agree, and a plan for a weight with no params object
    # must agree outright.
    differing = [name for name in ("vr_model_param",)
                 if getattr(a, name) is not getattr(b, name)]
    assert differing == ["vr_model_param"] and a.vr_model_param.param == b.vr_model_param.param
    for f in dataclasses.fields(plan.RunPlan):
        if f.name != "vr_model_param":
            assert getattr(a, f.name) == getattr(b, f.name), (
                f"{f.name} changed when only the Backend did: the plan is supposed to "
                "carry the GUI-free inputs, and the device is resolved later, in "
                "separate.py:236-276, not here.")
    assert _build(tmp_path, DEMUCS_WEIGHT) == _build(tmp_path, DEMUCS_WEIGHT, backend=other)
    assert a.device_set == "1", "UVR.py:410 strips the device prefix; the plan must " \
                                "carry the stripped value separate.py:276 formats."


def test_build_is_keyword_only(tmp_path):
    cat = _cat(tmp_path)
    with pytest.raises(TypeError):
        plan.build(settings_module.Settings(), cat, CPU, cat.resolve(VR_WEIGHT))


# --------------------------------------------------------------------------- #
# 4. the derivations that are actually dangerous
# --------------------------------------------------------------------------- #

def _selections_for(secondary: str, scale: str = "0.5") -> dict:
    """All four stem slots in each per-arch group (`UVR.py:1634/1644/1654`).

    A Demucs "All Stems" run reads one slot per stem (`UVR.py:632-637`), so a test
    that fills only `voc_inst` would leave three of the four lists empty and prove
    nothing about the loop.
    """
    slots: dict[str, str] = {}
    for stem in ("voc_inst", "other", "drums", "bass"):
        slots[f"{stem}_secondary_model"] = secondary
        slots[f"{stem}_secondary_model_scale"] = scale
    return {"vr": dict(slots), "mdx": dict(slots), "demucs": dict(slots)}


def test_the_group_is_chosen_by_process_method_not_read_once(tmp_path):
    """`UVR.py:491` / `:526` / `:601` are three different Settings fields."""
    s = settings_module.Settings(vr_is_secondary_model_activate=True,
                                 mdx_is_secondary_model_activate=False,
                                 demucs_is_secondary_model_activate=True)
    sel_for_vr_and_mdx = _selections_for(DEMUCS_WEIGHT)
    sel_for_demucs = _selections_for(VR_WEIGHT)
    vr = _build(tmp_path, VR_WEIGHT, s=s, selections=sel_for_vr_and_mdx)
    mdx = _build(tmp_path, MDX_WEIGHT, s=s, selections=sel_for_vr_and_mdx)
    de = _build(tmp_path, DEMUCS_WEIGHT, s=s, selections=sel_for_demucs)
    assert vr.is_secondary_model_activated is True
    assert mdx.is_secondary_model_activated is False, (
        "the MDX plan read the VR activation flag: UVR.py:526 uses "
        "mdx_is_secondary_model_activate, and a single global read would light up "
        "a secondary the user never switched on for MDX-Net.")
    assert de.is_secondary_model_activated is True
    assert vr.secondary_model is not None and vr.secondary_model_scale == 0.5  # float() at :7073
    assert mdx.secondary_model is None
    # `UVR.py:610-611`: Demucs answers to the *_Demucs pair, the others to the plain pair.
    only = settings_module.Settings(is_primary_stem_only=False, is_primary_stem_only_Demucs=True,
                                    is_secondary_stem_only=True,
                                    is_secondary_stem_only_Demucs=False,
                                    demucs_is_secondary_model_activate=True)
    assert _build(tmp_path, DEMUCS_WEIGHT, s=only).is_primary_stem_only is True
    assert _build(tmp_path, VR_WEIGHT, s=only).is_primary_stem_only is False
    assert _build(tmp_path, DEMUCS_WEIGHT, s=only).is_secondary_stem_only is False
    assert _build(tmp_path, MDX_WEIGHT, s=only).is_secondary_stem_only is True
    # ... and the Demucs choice is handed to the child as `is_primary_model_*`
    # (`UVR.py:676` -> `:7079` -> `:622`), which `separate.py:184-185` then reads.
    child = _build(tmp_path, DEMUCS_WEIGHT, s=only,
                   selections=sel_for_demucs).secondary_model_4_stem[0]
    assert child.is_primary_model_primary_stem_only is True
    assert child.is_primary_model_secondary_stem_only is False
    assert child.is_secondary_model is True                      # UVR.py:7077
    assert child.secondary_model is None, (
        "a secondary resolved a secondary of its own: UVR.py:491/:526/:601 zero the "
        "activation when is_secondary_model, so the chain must stop here.")
    assert child.pre_proc_model is None and child.vocal_split_model is None  # :620/:650/:667


def test_the_four_near_miss_renames(tmp_path):
    """Where the plan's name and the `Settings` name differ by a word, by design."""
    s = settings_module.Settings(gpu_conversion=True, mdxnet_stems="Bass",
                                 is_save_inst_set_vocal_splitter=True,
                                 is_mdx23_combine_stems=False)
    p = _build(tmp_path, MDX_WEIGHT, s=s)
    assert p.is_gpu_conversion == 0, "UVR.py:384 maps the bool to 0/-1, not True/False"
    assert p.mdxnet_stem_select == "Bass"
    assert p.is_save_inst_vocal_splitter is True
    assert p.is_mdx_combine_stems is False
    assert _build(tmp_path, MDX_WEIGHT, s=settings_module.Settings()).is_gpu_conversion == -1


def test_ensemble_mode_is_false_and_the_pre_proc_activation_is_zeroed(tmp_path):
    """`UVR.py:479-480` (not secondary, not pre-proc) and `:620`."""
    p = _build(tmp_path, DEMUCS_WEIGHT,
               s=settings_module.Settings(demucs_is_secondary_model_activate=True,
                                          is_demucs_pre_proc_model_activate=True),
               selections={**_selections_for(VR_WEIGHT), "demucs_pre_proc_model": VR_WEIGHT})
    assert p.is_ensemble_mode is False                # a concrete spec, not "Ensemble Mode"
    assert p.pre_proc_model is not None               # UVR.py:650-654
    assert p.pre_proc_model.is_pre_proc_model is True
    assert p.pre_proc_model.pre_proc_model is None, (
        "UVR.py:650 only reaches the pre-proc for the Demucs arch; a VR child naming "
        "itself again is the recursion this gate is supposed to stop.")
    assert p.pre_proc_model.is_secondary_model is False   # UVR.py:438 leaves it a pre-proc
    assert p.secondary_model_4_stem[0].pre_proc_model is None, (
        "UVR.py:620 zeroes the pre-proc activation for a secondary model.")
    assert p.secondary_model_4_stem[0].is_secondary_model is True  # UVR.py:7077


def test_a_self_naming_pre_proc_chain_is_an_error_not_a_recursion(tmp_path):
    """The one place `build()` is deliberately stricter than `ModelData`.

    A Demucs primary whose pre-proc selection names a Demucs 4-stem weight: upstream
    re-enters `:650` with the activation still set (`:603` keys off `demucs_stems`,
    not off being a pre-proc) and recurses until the stack dies.  Here the depth cap
    turns it into a typed error at the third level.
    """
    s = settings_module.Settings(is_demucs_pre_proc_model_activate=True)
    with pytest.raises(plan.PlanError, match="recurse"):
        _build(tmp_path, DEMUCS_WEIGHT, s=s,
               selections={"demucs_pre_proc_model": DEMUCS_WEIGHT})


def test_nested_slots_are_none_when_empty_or_unresolvable_and_a_plan_when_named(tmp_path):
    no = _build(tmp_path, VR_WEIGHT, s=settings_module.Settings(
        vr_is_secondary_model_activate=True),
        selections={"vr": {"voc_inst_secondary_model": catalog.NO_MODEL}})
    assert no.secondary_model is None and no.secondary_model_scale is None  # :7075/:7084
    assert no.is_secondary_model_activated is False                         # :679
    bad = _build(tmp_path, VR_WEIGHT, s=settings_module.Settings(
        vr_is_secondary_model_activate=True),
        selections=_selections_for("no-such-weight-at-all"))
    assert bad.secondary_model is None, (
        "UVR.py:7081-7082 drops a secondary whose weight is not there; raising here "
        "would turn a dropped checkbox into a failed conversion.")
    ok = _build(tmp_path, VR_WEIGHT, s=settings_module.Settings(
        vr_is_secondary_model_activate=True), selections=_selections_for(DEMUCS_WEIGHT))
    assert isinstance(ok.secondary_model, plan.RunPlan)
    assert ok.secondary_model.primary_model_primary_stem == ok.primary_stem   # :7078
    split = _build(tmp_path, VR_WEIGHT, selections={"vocal_splitter": {"name": DEMUCS_WEIGHT}})
    assert isinstance(split.vocal_split_model, plan.RunPlan)
    assert split.vocal_split_model.is_vocal_split_model is True               # :7108
    assert split.vocal_split_model.vocal_split_model is None                  # :667
    assert _build(tmp_path, VR_WEIGHT, selections={"vocal_splitter": {"name": "gone"}}
                  ).vocal_split_model is None


def test_a_secondary_named_like_the_primary_drops_itself(tmp_path):
    """`UVR.py:680-681`: the basename comparison, not an identity comparison."""
    p = _build(tmp_path, VR_WEIGHT, s=settings_module.Settings(
        vr_is_secondary_model_activate=True), selections=_selections_for(VR_WEIGHT))
    assert p.secondary_model is None and p.is_secondary_model_activated is False


# --------------------------------------------------------------------------- #
# 6. no fabricated values for what this fork blocks
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("arch", ["mdx_c", "roformer", "onnx"])
def test_blocked_archs_are_refused_not_simulated(tmp_path, arch):
    cat = _cat(tmp_path)
    assert cat.is_blocked(arch), f"{arch} is no longer in catalog.BLOCKED_ARCHS"
    spec = catalog.ModelSpec(name="some-weight", arch=arch, process_method="MDX-Net",
                             primary_stem="Vocals", secondary_stem="Instrumental",
                             weights_path=pathlib.Path("some-weight.ckpt"),
                             config_path=None, hash="h", model_data={})
    with pytest.raises(plan.PlanError, match="blocks|blocked"):
        _build(tmp_path, None, cat=cat, spec=spec)


def test_a_spec_the_catalog_cannot_answer_is_refused(tmp_path):
    cat = _cat(tmp_path)
    with pytest.raises(plan.PlanError):
        # Tier 3: no params. Upstream opens a dialog (UVR.py:759-777); a plan raises.
        _build(tmp_path, None, cat=cat, spec=catalog.ModelSpec(
            "unknown.pth", "vr", "VR Arc", None, None, pathlib.Path("unknown.pth"),
            None, "hash", None))
    with pytest.raises(plan.PlanError):
        # Apollo has no `process_method`, so no branch of separate.py would read it.
        _build(tmp_path, None, cat=cat, spec=catalog.ModelSpec(
            "apollo.bin", "apollo", None, None, None, pathlib.Path("apollo.bin"),
            None, "hash", {}))


def test_runplan_field_count_matches_the_ruling():
    assert len(dataclasses.fields(plan.RunPlan)) == 92
    assert len(set(plan.PLAN_FIELD_NAMES)) == 92
