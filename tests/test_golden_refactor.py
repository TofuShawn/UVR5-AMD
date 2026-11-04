"""Task 11c Gate 1 -- `plan.build()` and the frozen golden dict agree field by field.

WHAT THIS FILE PROVES, AND WHAT IT CANNOT
-----------------------------------------
Until this slice, ``tests/golden/runner.separate`` handed ``separate.py`` a
``types.SimpleNamespace(**settings.make(arch, device))`` -- an attribute dict written
by hand against ``UVR.py:365-804``.  It now hands the engine a ``RunPlan`` built by
``plan.build()`` from a real ``Settings``, a real ``catalog.Catalog``, a real
``backends.Backend`` and a ``ModelSpec`` resolved from that catalog.  ``RunPlan``
already satisfies ``separate.py``'s *current* constructor (11b), so the switch needs
no rename and can be judged on values alone.

The switch is the easy half; *this* file is the deliverable.  Two gates, two
questions, deliberately separated:

* **Gate 1 (here)** -- does ``plan.build()`` compute the right values?  Answered by
  comparing every key present in *both* the frozen dict and the 92 plan fields,
  per arch and per device, and by accounting for everything that is not shared:
  the residue on the dict side and the unreached bucket on the plan side.  Field
  counts and both residual sets are asserted explicitly, so a field cannot vanish
  from both sides and hide.
* **Gate 2** (``tests/test_golden_cpu.py`` + ``tests/test_golden_rocm.py``, run
  against the **unchanged** published ``baseline/*.wav``) -- does the engine still
  produce the pinned bytes?

Say the scope of each out loud rather than implying what the SNR gate does not
cover: ``pins.json`` pins ``vr`` and ``demucs`` only (its ``archs_blocked`` names
``mdx_c``/``roformer``/``onnx``, and ``test_golden_rocm.py:137`` records that ``mdx``
has no CPU baseline).  Gate 2 therefore proves two arches, on one device pair, with
no secondary / pre-proc / vocal-split model selected.  Everything outside that --
the MDX block, the nested-model fields, the export-format fields -- is covered only
by Gate 1's equality, which is why the comparison is field-by-field and not an
end-to-end run.

THREE WAYS TWO SIDE-BY-SIDE VALUES CAN FAIL TO AGREE, AND THIS FILE KEEPS THEM APART
------------------------------------------------------------------------------------
1. *Representation*: the dict stores paths as ``str`` (it copies ``os.path.join``
   output from upstream), the plan stores ``pathlib.Path``.  Same location,
   different type -- compared by ``str()`` and reported as a type-only difference,
   asserted to be *exactly* the three keys that behave this way.
2. *Identity-bearing objects*: ``vr_model_param`` is a ``ModelParameters``, which
   defines no ``__eq__``, so two parses of the *same* json compare unequal.  Its
   payload ``.param`` is compared instead; see ``VALUE_NORMALIZATION``.
3. *Genuine value divergence*: anything left after 1 and 2.  Those are NOT waved
   through -- ``EXPECTED_DIVERGENCES`` is a ledger with a stated reason per entry,
   and the test asserts the observed divergence set equals that set *exactly*.  A
   new divergence fails; so does a divergence this file fixed without updating the
   ledger, which is the direction that keeps the ledger from rotting.

The coercions are the trap this file exists to walk into deliberately: the dict
holds post-coercion attribute values while ``Settings`` holds the tkinter domain
(Task 9), so ``runner.settings_from_ui`` is built from the harness's *pre*-coercion
inputs (``overlap`` arrives as the text ``'0.25'``, ``mdx_batch_size`` as the
sentinel ``'Default'``) and ``plan.build`` re-applies ``UVR.py``'s coercions.  The
last test checks exactly that, because a plan built from the dict's own coerced
values would prove nothing about the port.
"""
from __future__ import annotations

import ast
import dataclasses
import pathlib
import re

import pytest

import plan
import settings as app_settings
import soundfile as sf

from tests.golden import runner
from tests.golden import settings as gs
from tests.test_plan import model_reads

REPO = pathlib.Path(__file__).resolve().parents[1]
SEPARATE_PY = (REPO / "separate.py").read_text(encoding="utf-8")

APPLY = gs.APPLY                                   # ["vr", "demucs"]
DEVICES = ["cpu", "cuda"]
COMBINATIONS = [(arch, dev) for arch in APPLY for dev in DEVICES]

#: Measured here, and the reason the counts are asserted: a shared-key count that
#: drifts means either a field left one side or an arch block stopped being
#: described by the dict.
EXPECTED_SHARED_COUNT = {"vr": 71, "demucs": 72}

#: Keys the frozen dict supplies that are **not** ``RunPlan`` fields, and why each
#: one belongs on exactly one side of the seam.  Verified against the source in
#: ``test_residue_keys_are_exactly_these_four_and_each_is_explained`` -- the
#: sentences below are the explanation, not the evidence.
RESIDUE_EXPLAINED = {
    "crop_size": (
        "vr only. Not read by the engine at all: `separate.py` has no occurrence of "
        "the name (the brief's `grep -c crop_size separate.py` -> 0, re-measured via "
        "AST below). It is a `Settings` field and a `DEFAULT_DATA` key, so "
        "`plan.py:216` lists it among the names R-T11-scope keeps out of the 92."),
    "is_4_stem_ensemble": (
        "both archs. Arrives through the *other* constructor argument, not through "
        "the model object: `separate.py:150` "
        "`self.is_4_stem_ensemble = process_data['is_4_stem_ensemble']`, and the "
        "harness sets it in `runner.make_process_data`. R-T11-scope keeps "
        "process_data out of the plan. (The brief cited `:155`; measured `:154` -- "
        "`:155` is `list_all_models`. Re-checked below, not trusted.)"),
    "model_status": (
        "both archs. `ModelData`'s own success flag -- `UVR.py:422` "
        "`False if model_name in (CHOOSE_MODEL, NO_MODEL)`, cleared again at "
        "`:523`/`:582`/`:596`/`:783` when a weight or config yaml cannot be found, "
        "and read back at `:615`/`:625`/`:656`/`:667` to decide whether to resolve "
        "nested models. `separate.py` never reads `model_data.model_status` (0 "
        "occurrences, AST-checked below). A plan is only ever built from a spec that "
        "resolved, so it has no failure state to carry -- `plan.build` raises "
        "`PlanError` where upstream would have set the flag False, which is the "
        "`plan.py:93-100` docstring's stated reason."),
    "margin_demucs": (
        "demucs only. `UVR.py:604` sets `self.margin_demucs` from `margin_demucs_var` "
        "inside the Demucs branch, while `UVR.py:527` sets `self.margin` from "
        "`margin_var` inside the MDX branch. `separate.py` reads exactly ONE margin "
        "off the model object -- `plan.margin` at `:312`, MDX-only -- so the "
        "Demucs-branch attribute is dead weight: no read of `model_data.margin_demucs` "
        "exists anywhere. See `test_margin_demucs_never_reaches_the_engine`."),
}

#: Value divergences that survive the two documented normalizations, per arch.
#: This is the finding list, not a suppression list: the test asserts the observed
#: set matches it exactly and prints each pair with its `UVR.py` citation.
EXPECTED_DIVERGENCES = {
    ("vr", "DEVERBER_MODEL"),
    ("demucs", "DEVERBER_MODEL"),
    ("demucs", "model_basename"),
}
DIVERGENCE_EXPLAINED = {
    "DEVERBER_MODEL": (
        "The deverb weight is **absent from this machine** -- "
        "`<MODEL_ROOT>/VR_Models/` holds `1_HP-UVR.pth`, `UVR-DeNoise-Lite.pth` and "
        "`model_data/` only. The dict writes the path anyway (`settings.make` builds "
        "the string unconditionally and only consults `.is_file()` for the flag), "
        "while `plan._model_file` returns a path *only if the file exists* "
        "(`UVR.py:296`), so the plan says `None`. Inert either way: both sides agree "
        "`is_deverb_vocals is False` (`UVR.py:381` gates the flag on the weight "
        "existing) and `separate.py:526` reaches `vr_denoiser(model_path="
        "self.DEVERBER_MODEL)` only under that flag (`:515`). `DENOISER_MODEL`, which "
        "*is* on disk, agrees as a path."),
    "model_basename": (
        "The frozen dict is the one that is wrong here, and the plan matches "
        "upstream. `UVR.py:616` derives `model_basename` from `self.model_path`, and "
        "`get_demucs_model_path` (`UVR.py:711-721`) sets `model_path` to the bag "
        "descriptor `v3_v4_repo/htdemucs.yaml` -- which is also what the *catalog* "
        "resolves, and what `settings.make` itself puts in `model_path`. But "
        "`settings.make:355` computes the basename from `require_model('demucs')`, "
        "the ``.th`` weight, giving `'955717e8-8726e21a'` where `UVR.py:616` gives "
        "'htdemucs'. Audio-inert on this harness's three uses: `separate.py:356` "
        "hands it to `cached_source_callback` (which always answers "
        "`(None, None)`), `:435` counts it inside `list_all_models` (`[]` -> 0 "
        "either way), and `:583`/`:994` compare it against `primary_model_name` "
        "(None). NOT fixed by editing `settings.make`: that dict is what the "
        "published baselines were described by, and the report says so."),
}

#: Fields compared after one stated read-back instead of raw ``==``, because the
#: type at that key is a container whose own equality is not the value's equality.
VALUE_NORMALIZATION = {
    "vr_model_param": (
        "Both sides are a `ModelParameters` parsed from the same "
        "`lib_v5/vr_network/modelparams/4band_44100.json` (`UVR.py:260/514`); the "
        "class defines no `__eq__`, so raw `==` would compare object identity. "
        "The `.param` payload is what the engine consumes "
        "(`separate.py` -> `lib_v5` reads `model_params.param[...]`), so that is "
        "what is compared."),
}

#: Keys whose two sides name the same location in different types (dict ``str``,
#: plan ``pathlib.Path``). Asserted as an exact set, so a future field cannot join
#: this bucket silently.
PATH_LIKE_KEYS = {"DENOISER_MODEL", "mixer_path", "model_path"}

#: The 21 (`vr`) / 20 (`demucs`) ``RunPlan`` fields the frozen dict does not carry,
#: by *why* this frozen scenario never reaches them.  The bucket is only
#: trustworthy if every name in it is a field the arch's own read set excludes,
#: which `test_unreached_fields_are_exactly_the_paths_this_scenario_never_takes`
#: checks against the AST-derived per-arch requirement set.
MDX_AND_DENOISE_UNREACHED = [
    # -- MDX block only: `separate.py:292-312`, inside
    #    `if model_data.process_method == MDX_ARCH_TYPE`.  Upstream assigns
    #    `margin`/`chunks`/`compensate`/`mdx_*` only there (`UVR.py:525-529`), which
    #    is why the dict carries no such key for vr or demucs at all.
    "chunks", "compensate", "is_mdx_c_seg_def", "is_mdx_ckpt", "margin",
    "mdx_batch_size", "mdx_dim_f_set", "mdx_dim_t_set", "mdx_n_fft_scale_set",
    "mdx_segment_size",
    # -- denoise/deverb flags, read only in that same MDX block (`:298-299`) and
    #    under the denoise gate; the frozen run leaves `denoise_option` at "None".
    "is_denoise", "is_denoise_model",
]
VR_BRANCH_UNREACHED = [
    # -- VR branch (`separate.py:277-291`) plus `SeperateVR`; a demucs-shaped dict
    #    has no reason to carry them, and `plan.py` defaults them for other archs.
    "aggression_setting", "batch_size", "is_high_end_process", "is_post_process",
    "is_tta", "post_process_threshold", "vr_model_param", "window_size",
]
DEMUCS_BRANCH_UNREACHED = [
    # -- Demucs block (`separate.py:316-326`), same reasoning for a vr-shaped dict.
    "demucs_source_list", "demucs_source_map", "demucs_stem_count", "demucs_stems",
    "demucs_version", "is_chunk_demucs", "is_split_mode", "segment", "shifts",
]
UNREACHED_EXPECTED = {
    "vr": sorted(MDX_AND_DENOISE_UNREACHED + DEMUCS_BRANCH_UNREACHED),
    "demucs": sorted(MDX_AND_DENOISE_UNREACHED + VR_BRANCH_UNREACHED),
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _plan_field_citations() -> dict[str, str]:
    """``{field: the comment plan.py carries for it}``, read out of plan.py itself.

    Built from the source rather than retyped, so the failure messages cite the
    `UVR.py:` line the port actually claims for that field -- and so a citation
    that changes in `plan.py` changes here with it.
    """
    text = (REPO / "plan.py").read_text(encoding="utf-8")
    start = text.index("class RunPlan:")
    body = text[start:text.index("PLAN_FIELD_NAMES: tuple")]
    out: dict[str, str] = {}
    for line in body.splitlines():
        match = re.match(r"^    ([a-zA-Z_]\w*)\s*:\s*.*?#\s*(.*)$", line)
        if match:
            out[match.group(1)] = match.group(2).strip()
    return out


CITATIONS = _plan_field_citations()
PLAN_FIELDS = {f.name for f in dataclasses.fields(plan.RunPlan)}


def _comparable(key: str, dict_value, plan_value) -> tuple[bool, str]:
    """(agree?, how it was judged) for one shared key, under the documented rules."""
    if key == "vr_model_param":
        same = dict_value.param == plan_value.param
        return same, "ModelParameters.payload .param"
    if key in PATH_LIKE_KEYS or isinstance(dict_value, pathlib.PurePath) \
            or isinstance(plan_value, pathlib.PurePath):
        if isinstance(dict_value, str) and plan_value is None:
            return False, "path-string vs None (both sides checked)"
        return str(dict_value) == str(plan_value), "path by str()"
    return dict_value == plan_value, "=="


def _diff(arch: str, device: str, tmp_path: pathlib.Path) -> dict:
    attrs = gs.make(arch, device)
    built = runner.make_plan(arch, device, tmp_path / f"plan-{arch}-{device}")
    shared = sorted(set(attrs) & PLAN_FIELDS)
    divergences, type_only = [], []
    for key in shared:
        dict_value, plan_value = attrs[key], getattr(built, key)
        agree, how = _comparable(key, dict_value, plan_value)
        if not agree:
            divergences.append({
                "key": key, "dict": dict_value, "plan": plan_value, "how": how,
                "citation": CITATIONS.get(key, "(no inline citation in plan.py)"),
            })
        elif type(dict_value) is not type(plan_value):
            type_only.append((key, type(dict_value).__name__, type(plan_value).__name__))
    return {
        "attrs": attrs, "built": built, "shared": shared,
        "divergences": divergences, "type_only": sorted(type_only),
        "residue": sorted(set(attrs) - PLAN_FIELDS),
        "unreached": sorted(PLAN_FIELDS - set(attrs)),
    }


# --------------------------------------------------------------------------- #
# Gate 1
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("arch", APPLY)
def test_shared_key_count_is_what_was_measured(arch, device, tmp_path):
    """71 shared keys for vr, 72 for demucs -- on cpu and on cuda alike.

    The count is the assertion: a field renamed on one side would leave the shared
    set and shrink it, and that is precisely the failure this slice must not be able
    to hide. (92 plan fields total; `test_plan.py` pins the 92.)
    """
    result = _diff(arch, device, tmp_path)
    assert len(result["shared"]) == EXPECTED_SHARED_COUNT[arch], (
        f"{arch}/{device}: {len(result['shared'])} shared keys, expected "
        f"{EXPECTED_SHARED_COUNT[arch]}. Dict has {len(result['attrs'])}, plan has "
        f"{len(PLAN_FIELDS)}. Only {EXPECTED_SHARED_COUNT[arch]} dict keys "
        f"(vr) / {EXPECTED_SHARED_COUNT['demucs']} (demucs) exist at time of writing."
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("arch", APPLY)
def test_every_shared_key_holds_the_same_value(arch, device, tmp_path):
    """The equality test. Field by field, with the plan's own citation in the message.

    Expected: zero divergences beyond the ledger above. If a new key appears here,
    it is a `plan.build()` finding -- do not add it to `EXPECTED_DIVERGENCES`
    without the reason it carries below.
    """
    result = _diff(arch, device, tmp_path)
    observed = {(arch, d["key"]) for d in result["divergences"]}
    report = "\n".join(
        f"  {d['key']}: dict={d['dict']!r} ({type(d['dict']).__name__}) vs "
        f"plan={d['plan']!r} ({type(d['plan']).__name__}) [compared by {d['how']}]\n"
        f"    plan.py cites -> {d['citation']}"
        for d in result["divergences"])
    unexpected = sorted(observed - EXPECTED_DIVERGENCES)
    healed = sorted(EXPECTED_DIVERGENCES - {(arch, k) for (_, k) in observed}
                    - {(a, k) for (a, k) in EXPECTED_DIVERGENCES if a != arch})
    assert not unexpected, (
        f"\n{arch}/{device}: plan.build() disagrees with the frozen harness dict on "
        f"{unexpected}.\n{report}\n\nThis is the divergence Gate 1 exists to find. "
        f"Read the citation above, then decide whether the plan or the harness is "
        f"wrong -- and note that `settings.make()` is what the published baselines "
        f"were *described* by, so changing it changes what the pins mean."
    )
    assert not healed, (
        f"\n{arch}/{device}: these divergences were recorded as known but no longer "
        f"happen: {healed}. Delete them from EXPECTED_DIVERGENCES / "
        f"DIVERGENCE_EXPLAINED so the ledger keeps describing reality."
    )
    ledger = "\n".join(f"  {d['key']}: {DIVERGENCE_EXPLAINED[d['key']]}"
                       for d in result["divergences"])
    print(f"\n{arch}/{device}: {len(result['shared'])} shared keys, "
          f"{len(result['divergences'])} ledgered divergence(s), "
          f"{len(result['type_only'])} type-only.\n{ledger}")


@pytest.mark.parametrize("arch", APPLY)
@pytest.mark.parametrize("device", DEVICES)
def test_type_only_differences_are_exactly_the_path_fields(arch, device, tmp_path):
    """str in the dict, Path in the plan -- three keys, no more.

    Same location, different container, so `separate.py`'s `os.path`/`torch.load`
    calls behave identically; recorded because "value-identical" should not quietly
    mean "type-identical".
    """
    result = _diff(arch, device, tmp_path)
    keys = {t[0] for t in result["type_only"]}
    assert keys == PATH_LIKE_KEYS, (
        f"{arch}/{device}: type-only differences are {result['type_only']}, expected "
        f"exactly {sorted(PATH_LIKE_KEYS)}. A key joining this bucket changed "
        f"str<->Path representation, which is an engine-visible change."
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("arch", APPLY)
def test_residue_keys_are_exactly_these_four_and_each_is_explained(arch, device, tmp_path):
    """The dict's non-plan keys: crop_size (vr), is_4_stem_ensemble, model_status,
    margin_demucs (demucs) -- measured, not assumed, and pinned as an exact set.

    A fifth unexplained key would mean `plan.build()` dropped something the harness
    thinks the engine needs (or that `settings.make()` invented one).
    """
    result = _diff(arch, device, tmp_path)
    expected = {"is_4_stem_ensemble", "model_status"} | (
        {"crop_size"} if arch == "vr" else {"margin_demucs"})
    assert set(result["residue"]) == expected, (
        f"{arch}/{device}: residue is {result['residue']}, expected {sorted(expected)}."
    )
    for key in result["residue"]:
        assert key in RESIDUE_EXPLAINED, f"{key} has no explanation"


def test_residue_claims_are_verified_against_the_engine_source():
    """The four explanations, checked against `separate.py` rather than asserted.

    Written because the brief itself flagged two of its own citations for re-check
    (`is_4_stem_ensemble` at ':155', the `margin_demucs` story), and because a
    residue entry that stops being true is a leak, not a footnote.
    """
    reads = model_reads(SEPARATE_PY)
    assert "crop_size" not in reads, "crop_size IS read off the model object now"
    assert "model_status" not in reads, "model_status IS read off the model object now"
    assert "margin_demucs" not in reads, "margin_demucs IS read off the model object now"
    assert "is_4_stem_ensemble" not in reads, (
        "is_4_stem_ensemble now arrives on the model object, not process_data")
    assert SEPARATE_PY.count("crop_size") == 0
    assert SEPARATE_PY.count("model_status") == 0
    # ...and exactly one margin, on the MDX-only line.
    assert list(reads["margin"]) == [312], reads["margin"]
    assert "self.is_4_stem_ensemble = process_data['is_4_stem_ensemble']" in SEPARATE_PY
    line = 1 + SEPARATE_PY.split(
        "self.is_4_stem_ensemble = process_data['is_4_stem_ensemble']")[0].count("\n")
    assert line == 150, (
        f"is_4_stem_ensemble now moves off separate.py:150 (found :{line}); the "
        f"citation in RESIDUE_EXPLAINED and the brief need updating with it.")


def test_margin_demucs_never_reaches_the_engine():
    """Upstream behaviour finding: the Demucs margin is read into ModelData and
    then dropped on the floor.

    `UVR.py:527` sets `self.margin` from `margin_var` in the **MDX** branch and
    `UVR.py:604` sets `self.margin_demucs` from `margin_demucs_var` in the
    **Demucs** branch; `separate.py` reads `plan.margin` once (`:312`, inside
    the MDX block) and never reads `margin_demucs`. So a Demucs run's margin value
    cannot affect the engine at all -- and, since the two `DEFAULT_DATA` entries are
    both `44100` (`constants.py:601`/`:624`), even the MDX path could not tell.
    Measured here rather than trusted: `Settings` keeps BOTH fields, so a later
    slice that wires `plan.build` to pick `margin_demucs` for demucs plans would
    change engine behaviour on a path the baselines do cover. This slice does not.
    """
    from gui_data.constants import DEFAULT_DATA
    s = runner.settings_from_ui("demucs", "cpu")
    assert s.margin == DEFAULT_DATA["margin"] == 44100
    assert s.margin_demucs == DEFAULT_DATA["margin_demucs"] == 44100
    assert "margin" not in gs.make("demucs", "cpu")
    assert gs.make("demucs", "cpu")["margin_demucs"] == 44100
    assert "margin_demucs" not in PLAN_FIELDS
    assert "margin" in PLAN_FIELDS


@pytest.mark.parametrize("arch", APPLY)
def test_unreached_fields_are_exactly_the_paths_this_scenario_never_takes(arch, tmp_path):
    """92 - 71 = 21 (vr) / 92 - 72 = 20 (demucs) plan fields the dict does not carry.

    Pinned as an exact list, and cross-checked against the *AST-derived* per-arch
    requirement set used by `test_golden_cpu.py`: no field the engine actually
    reaches on this arch may sit in this bucket. That is what stops a reached field
    from hiding among the unreached ones.
    """
    from tests.test_golden_cpu import _required_attrs
    result = _diff(arch, "cpu", tmp_path)
    expected = UNREACHED_EXPECTED[arch]
    assert result["unreached"] == expected, (
        f"{arch}: unreached bucket is {result['unreached']}, expected {expected}"
    )
    assert len(result["unreached"]) == len(PLAN_FIELDS) - EXPECTED_SHARED_COUNT[arch]
    reached = _required_attrs(arch)
    leaked = sorted(set(expected) & reached)
    assert not leaked, (
        f"{arch}: {leaked} are read by separate.py on this arch's path, so they must "
        f"be in the shared set -- an unreached field is one this scenario never hits."
    )


# --------------------------------------------------------------------------- #
# The coercion trap: the plan is built from pre-coercion inputs
# --------------------------------------------------------------------------- #

def test_settings_from_ui_holds_the_widget_domain_not_the_attribute_domain(tmp_path):
    """`Settings` gets the tkinter text; `settings.make()` holds the coerced value.

    If this test ever shows the two agreeing on *types* for these fields, someone
    round-tripped the dict's own post-coercion output back through `Settings` -- and
    then Gate 1 would be comparing `plan.build()` with itself, which proves nothing.
    """
    s = runner.settings_from_ui("vr", "cpu")
    attrs = gs.make("vr", "cpu")
    built = runner.make_plan("vr", "cpu", tmp_path)
    assert s.overlap == "0.25" and isinstance(s.overlap, str)
    assert attrs["overlap"] == 0.25 and isinstance(attrs["overlap"], float)
    assert s.semitone_shift == "0" and isinstance(s.semitone_shift, str)
    assert attrs["semitone_shift"] == 0.0
    assert s.batch_size == "Default" and s.mdx_batch_size == "Default"
    assert attrs["batch_size"] == 1                      # UVR.py:496
    assert built.batch_size == 1
    # `mdx_batch_size` is not a dict key at all: the frozen run never enters the
    # MDX block that reads it (`separate.py:297`), so `UVR.py:391`'s
    # sentinel-to-1 coercion is only observable on the plan side.
    assert built.mdx_batch_size == 1
    # UVR.py:492 divides the widget's integer by 100, so the stored domain (5) and
    # the attribute domain (0.05) differ by exactly that coercion.
    assert s.aggression_setting == 5 and attrs["aggression_setting"] == 0.05
    assert built.aggression_setting == 0.05
    assert s.mdxnet_stems == attrs["mdxnet_stem_select"]
    assert s.chosen_process_method == "VR Arc"
    with pytest.raises(TypeError):
        # Settings' own coercion still refuses a rendered container, so the harness
        # cannot smuggle the dict's non-scalar values in as widget text.
        app_settings.Settings(device_set=[])


@pytest.mark.parametrize("arch", APPLY)
def test_device_reaches_the_plan_as_the_legacy_gpu_conversion_encoding(arch, tmp_path):
    """`device` is not a plan field (R-T11-scope); `is_gpu_conversion` is.

    `UVR.py:384` / `:954` encode the GPU checkbox as 0/-1 and `settings.py`
    decodes that back to a bool on import, so the harness's `"cpu"`/`"cuda"`
    string travels as `gpu_conversion` and is re-encoded by `plan.build`.
    """
    for device, expected in (("cpu", -1), ("cuda", 0)):
        s = runner.settings_from_ui(arch, device)
        assert s.gpu_conversion is (device != "cpu")
        assert gs.make(arch, device)["is_gpu_conversion"] == expected
        built = runner.make_plan(arch, device, tmp_path / device)
        assert built.is_gpu_conversion == expected
        # ...and the Backend that names the same device changes nothing else.
        assert device != "cpu" or built.device_set == gs.make(arch, device)["device_set"]


# --------------------------------------------------------------------------- #
# The switch itself
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("arch", APPLY)
def test_separate_hands_the_engine_a_runplan(arch, monkeypatch, tmp_path):
    """The call site, not the helper: `separate()` must construct a `RunPlan`.

    The engine class is replaced by a recorder that writes one fake stem so
    `separate()` completes -- no weights are loaded. This is what makes the switch
    (rather than the existence of `make_plan`) the tested thing.
    """
    seen: dict[str, object] = {}

    class Recorder:
        def __init__(self, model_data, process_data):
            seen["model_data"] = model_data
            seen["process_data"] = dict(process_data)

        def seperate(self):
            export = pathlib.Path(seen["process_data"]["export_path"])
            sf.write(str(export / f"{seen['process_data']['audio_file_base']}_(X).wav"),
                     [[0.0, 0.0]], 44100, subtype="PCM_16")

    monkeypatch.setitem(runner.ARCH_CLASS, arch, Recorder)
    stems = runner.separate(arch, "cpu", tmp_path)
    model_data = seen["model_data"]
    assert isinstance(model_data, plan.RunPlan), type(model_data)
    assert stems, "the recorder's stem was not picked up"
    # process_data keeps its 11 keys and its three callables: R-T11-scope says the
    # plan does not absorb it, and Plan 2 attaches to exactly those seams.
    assert set(seen["process_data"]) == {
        "audio_file", "audio_file_base", "export_path", "set_progress_bar",
        "write_to_console", "cached_source_callback", "cached_model_source_holder",
        "is_ensemble_master", "is_4_stem_ensemble", "list_all_models",
        "process_iteration"}
    assert callable(seen["process_data"]["set_progress_bar"])
    assert callable(seen["process_data"]["cached_source_callback"])
    assert model_data.process_method in (gs.make(arch, "cpu")["process_method"],)


def test_runplan_is_not_a_simplenamespace_any_more():
    """The harness's engine input is a `RunPlan`, checked on the AST not the prose.

    (The module docstring still *names* ``SimpleNamespace`` -- that is the switch
    being described, so the test walks the syntax tree instead of the text.)
    """
    tree = ast.parse(pathlib.Path(runner.__file__).read_text(encoding="utf-8"))
    constructors = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "SimpleNamespace"
    ]
    assert not constructors, f"runner.py still builds a SimpleNamespace at {constructors}"
    imported = [node.names[0].name for node in ast.walk(tree) if isinstance(node, ast.Import)]
    assert "types" not in imported, "`types` is only imported for SimpleNamespace"
    assert "plan" in imported
    assert "plan.build(" in pathlib.Path(runner.__file__).read_text(encoding="utf-8")


def test_plan_field_citations_are_readable():
    """The message machinery the brief asks for exists and is not empty."""
    assert CITATIONS, "no citations parsed from plan.py"
    for field in sorted(PLAN_FIELDS):
        if field not in CITATIONS:
            continue
        assert "UVR.py" in CITATIONS[field] or field in ("process_method",), field
    assert len(CITATIONS) >= 80, len(CITATIONS)
