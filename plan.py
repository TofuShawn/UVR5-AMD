"""plan.py -- one frozen `RunPlan` carrying exactly the 92 attributes `separate.py` reads.

Task 11b of the ROCm-engine plan.  `ModelData` (`UVR.py:365-804`) is the GUI's
bag-of-state: it reads ~77 `root.` tkinter vars, then derives 44 more attributes
from them and from the params tables.  An AST sweep of `separate.py` (ruling
R-T11-shape) shows the engine reads **92 distinct attributes** off that object and
writes none, all of them inside `SeperateAttributes` (`separate.py:158-374`).
`RunPlan` is those 92 names and nothing else, computed once from typed inputs:
`settings.Settings` (Task 9), `catalog.ModelSpec` (Task 10), `backends.Backend`
(Task 6) and the 26 secondary-model *selections* (arguments here, not `Settings`
fields).

Decomposition of the 92, in the order the ruling states it: **34** names are
byte-identical to a `Settings` field, **3** to a `ModelSpec` field
(`primary_stem`, `process_method`, `secondary_stem`), **7** are params-table keys
read out of `spec.model_data` (`is_bv_model`, `is_karaoke`, `is_roformer`,
`mdx_dim_f_set`, `mdx_dim_t_set`, `mdx_n_fft_scale_set`, `vr_model_param`), **4**
are `DEFAULT_DATA`-backed (`chunks`, `is_denoise`, `is_gpu_conversion`,
`is_mixer_mode` -- `gui_data/constants.py:572`) and **44** are computed by
`ModelData`'s own body or by the three `root.` decision methods
(`UVR.py:7046-7114`).

The 92 names are the contract, not a first draft: Task 11c replaces `model_data.`
with `plan.` and changes nothing else.  This machine has golden baselines for `vr`
and `demucs` only (`pins.json["archs_blocked"]` = `mdx_c`, `roformer`, `onnx`, and
`test_golden_rocm.py:137` records that `mdx` has no CPU baseline), so the SNR gate
cannot catch a renamed field in the MDX path.  Names are therefore reproduced
verbatim, near-misses included (`is_gpu_conversion`<-`gpu_conversion`,
`mdxnet_stem_select`<-`mdxnet_stems`, `is_mdx_combine_stems`<-`is_mdx23_combine_stems`,
`is_save_inst_vocal_splitter`<-`is_save_inst_set_vocal_splitter`).  Tidying them is
a separate, separately provable commit.

Two rulings bound the shape and are not adjustable from here.  **R-T11-scope**:
`RunPlan` is *exactly* the 92 -- no `audio_path`, no `export_path`, no
`stems_only`, no `device` field (the 92 contain `device_set` and no `device`) --
because those live in `process_data` (`separate.py:129-157`), which also holds
three callables (`set_progress_bar`, `write_to_console`, `cached_source_callback`)
and a mutable accumulator (`cached_model_source_holder`).  Callbacks and shared
caches must not sit inside a reproducible value object, and Plan 2 attaches its
`QThread` signals precisely to those callables.  **R-T11-selections**: the 26
secondary-model selections arrive as a `build()` argument, not as `Settings`
fields; `None`/absent means "no model selected".  (`catalog.py:280-284` still says
"the group is `Settings`' concern", which that ruling declares wrong; 11a owns that
file, so the correction happens in the slice allowed to touch it.)

`backend` is accepted per R-T11-scope's signature and is *not* consumed by any of
the 92: upstream's `ModelData` never reads a device, and `separate.py:236-276`
resolves torch device and ORT providers at run time from `backends.resolve()`.
`tests/test_plan.py` pins that non-interference rather than pretending otherwise.

This module is the composition layer between `app` and `engine`, so -- unlike
`engine` proper -- it may import `settings`, `catalog` and `backends`.  It must
never import `tkinter`, `gui_data.tkinterdnd2`, `gui_data.sv_ttk`, `PIL.ImageTk`
or `UVR`; `tests/test_plan.py` checks the import set by AST, and
`tests/test_engine_purity.py` imports it clean with the GUI packages blocked.
"""
from __future__ import annotations

import dataclasses
import os
import pathlib
import typing

import backends
import catalog
import settings as settings_module
import yaml
from gui_data.constants import (
    ALL_STEMS, AUTO_SELECT, BV_VOCAL_STEM, DEFAULT, DEF_OPT,
    DEMUCS_4_SOURCE_LIST, LEAD_VOCAL_STEM,
    DEMUCS_ARCH_TYPE, DEMUCS_V3,
    DEMUCS_V4, DEVERB_MAPPER, DENOISE_M, DENOISE_NONE, INST_STEM,
    INST_STEM_ONLY, IS_BV_MODEL, IS_BV_MODEL_REBAL, IS_KARAOKEE,
    IS_SAVE_VOC_ONLY, MDX_ARCH_TYPE, PRIMARY_STEM, VOCAL_STEM, VOCAL_STEM_ONLY,
    VR_ARCH_TYPE, WOOD_INST_MODEL_HASH, WOOD_INST_PARAMS,
)

from lib_v5.vr_network.model_param_init import ModelParameters

__all__ = ["RunPlan", "PlanError", "build", "PLAN_FIELD_NAMES"]

#: Repo root, the anchor for the two repo-relative paths upstream also builds from
#: `BASE_PATH` (`UVR.py:243` mixer ckpt, `UVR.py:260` VR param dir).
_REPO = pathlib.Path(__file__).resolve().parent

# The per-run-selection keys, using `catalog.secondary_slot_for`'s vocabulary so no
# new name enters the codebase (`UVR.py:1634/1644/1654`).
_SELECTION_GROUP = {VR_ARCH_TYPE: "vr", MDX_ARCH_TYPE: "mdx", DEMUCS_ARCH_TYPE: "demucs"}
_SCALES = {"voc_inst_secondary_model", "other_secondary_model",
           "drums_secondary_model", "bass_secondary_model"}


class PlanError(Exception):
    """This spec cannot become a plan -- refused rather than simulated.

    Raised for a blocked arch, an arch with no engine branch (`apollo`'s `None`
    process method), a tier-3 spec with no params, and a config-driven MDX-C model
    whose yaml is not on disk (`UVR.py:549-582` sets `model_status = False` there,
    and a plan has no `model_status` to hide behind).
    """


def _model_file(cat: catalog.Catalog, rel: str) -> pathlib.Path | None:
    """The first existing `rel` under `model_root` then `fallback_root`.

    Same order as `Catalog._roots` (R-T10-precedence), reached through the public
    fields instead of the private helper, for the two files upstream locates by
    constant path rather than by name: the MDX-C config yaml (`UVR.py:546`) and the
    denoise/deverb weights (`UVR.py:295-296`).
    """
    for root in (cat.model_root, cat.fallback_root):
        if root is None:
            continue
        candidate = root / rel
        if candidate.is_file():
            return candidate
    return None


def _num(field: str, value: object) -> float:
    """`float()` over the tkinter string, with the GUI's digit separators removed.

    `Settings` stores the numeric knobs as the widget stored them (`'0.25'`,
    `'  8  '`, `'44100'`), so the coercions `ModelData` applied at
    `UVR.py:391-396`/`:492-499` are applied here.
    """
    text = str(value).replace(",", "").replace(" ", "")
    try:
        return float(text)
    except ValueError as exc:  # a widget could hold anything; a dataclass cannot
        raise PlanError(f"{field} is not numeric: {value!r}") from exc


def _dig(config: object, *keys: str) -> object:
    """`config.training.target_instrument` over a dict or an OmegaConf node (`UVR.py:552`)."""
    current = config
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key)
        else:
            current = getattr(current, key, None)
        if current is None:
            return None
    return current


@dataclasses.dataclass(frozen=True)
class RunPlan:
    """Exactly the 92 attributes `separate.py` reads off `model_data`.

    Field order follows the buckets, not upstream's assignment order, and every
    derived field carries the `UVR.py` line it comes from.  Frozen: nothing in the
    engine may write it (the AST sweep finds 0 writes, so a write would be a new
    dependency, not a refactor).
    """

    # -- 3 names that are byte-identical to a `catalog.ModelSpec` field -------- #
    process_method: str | None      # UVR.py:421 (spec.process_method)
    primary_stem: str | None        # UVR.py:512/588/737 (spec.primary_stem, branch-specific)
    secondary_stem: str | None      # UVR.py:513/594/738 (secondary_stem(primary_stem))

    # -- 7 keys read out of `spec.model_data` (the baked/per-model params) ----- #
    vr_model_param: typing.Any = None       # UVR.py:511-514
    mdx_dim_f_set: typing.Any = None        # UVR.py:585
    mdx_dim_t_set: typing.Any = None        # UVR.py:586
    mdx_n_fft_scale_set: typing.Any = None  # UVR.py:587
    is_roformer: bool = False               # UVR.py:541-542
    is_karaoke: bool = False                # UVR.py:686-687
    is_bv_model: bool = False               # UVR.py:688-689

    # -- 4 `DEFAULT_DATA`-backed names (`gui_data/constants.py:572) ------------ #
    is_gpu_conversion: int = -1             # UVR.py:384/414, DEFAULT_DATA['is_gpu_conversion']
    is_mixer_mode: bool = False             # UVR.py:414 (hard False; DEFAULT_DATA['is_mixer_mode'])
    chunks: typing.Any = "Auto"             # UVR.py:528 sets 0 for MDX; DEFAULT_DATA['chunks']
    is_denoise: bool = False                # UVR.py:389, DEFAULT_DATA['is_denoise']

    # -- 34 names byte-identical to a `Settings` field (value coercions cited),
    # plus the 2 near-miss renames whose value also comes straight from `Settings`
    # (`is_mdx_combine_stems`, `mdxnet_stem_select`) -- 36 fields in this block. -- #
    is_normalization: bool = False          # UVR.py:385
    is_primary_stem_only: bool = False      # UVR.py:387, overridden at :610 for Demucs
    is_secondary_stem_only: bool = False    # UVR.py:388, overridden at :611 for Demucs
    is_mdx_c_seg_def: bool = False          # UVR.py:390
    mdx_batch_size: int = 1                 # UVR.py:391 (1 for 'Default')
    overlap: float = 0.25                   # UVR.py:393
    overlap_mdx: typing.Any = DEFAULT       # UVR.py:394
    overlap_mdx23: int = 8                  # UVR.py:395
    semitone_shift: float = 0.0             # UVR.py:396
    is_match_frequency_pitch: bool = True   # UVR.py:398
    wav_type_set: str = "PCM_16"            # UVR.py:409
    device_set: str = DEFAULT               # UVR.py:410 (':' suffix stripped)
    mp3_bit_set: str = "320k"               # UVR.py:411
    save_format: str = "WAV"                # UVR.py:412
    is_invert_spec: bool = False            # UVR.py:413
    demucs_stems: str = ALL_STEMS           # UVR.py:415
    is_demucs_combine_stems: bool = True    # UVR.py:416
    is_deverb_vocals: bool = False          # UVR.py:381 (gated on the deverb weight)
    deverb_vocal_opt: str = VOCAL_STEM      # UVR.py:382 via DEVERB_MAPPER
    is_mdx_combine_stems: bool = True       # UVR.py:401 <- Settings.is_mdx23_combine_stems
    mdxnet_stem_select: str = ALL_STEMS     # UVR.py:392 <- Settings.mdxnet_stems
    aggression_setting: float = 0.05        # UVR.py:492 (/100)
    is_tta: bool = False                    # UVR.py:493
    is_post_process: bool = False           # UVR.py:494
    window_size: int = 512                  # UVR.py:495
    batch_size: int = 1                     # UVR.py:496 (1 for 'Default')
    is_high_end_process: str = "None"       # UVR.py:498 ('mirroring'/'None')
    post_process_threshold: float = 0.2     # UVR.py:499
    margin: int = 44100                     # UVR.py:527
    mdx_segment_size: int = 256             # UVR.py:529
    shifts: int = 2                         # UVR.py:606
    is_split_mode: bool = True              # UVR.py:607
    segment: typing.Any = DEFAULT           # UVR.py:608
    is_chunk_demucs: bool = False           # UVR.py:609
    is_demucs_pre_proc_model_inst_mix: bool = False  # UVR.py:654 (gated on pre_proc)
    compensate: typing.Any = None           # UVR.py:584 for MDX, Settings' string otherwise
    # NOTE: `chunk_size`/`crop_size`/`margin_demucs` are `Settings` fields that
    # `separate.py` does NOT read off the model object, so they are not here.

    # -- 42 of the 44 derived by `ModelData`'s body or the three `root.` decision
    # methods; the remaining 2 are the renamed Settings reads in the block above, so
    # this section plus that one is the ruling's 44. -------------------------------- #
    model_name: str = ""                    # UVR.py:421/475
    model_path: pathlib.Path | None = None  # UVR.py:501/704-707/716-721
    model_basename: str | None = None       # UVR.py:616
    model_samplerate: int = 44100           # UVR.py:440/515
    model_capacity: tuple[typing.Any, ...] = (32, 128)  # UVR.py:441/500/518
    is_vr_51_model: bool = False            # UVR.py:442/517-519
    primary_stem_native: str | None = None  # UVR.py:516/589
    is_pitch_change: bool = False           # UVR.py:397
    is_mdx_c: bool = False                  # UVR.py:400/545
    is_mdx_ckpt: bool = False               # UVR.py:399/695-696
    mdx_c_configs: typing.Any = None        # UVR.py:402/546-550
    is_target_instrument: bool = False      # UVR.py:472/552-553
    mixer_path: pathlib.Path | None = None  # UVR.py:419/709 (BASE_PATH/lib_v5/mixer.ckpt)
    DENOISER_MODEL: pathlib.Path | None = None   # UVR.py:379
    DEVERBER_MODEL: pathlib.Path | None = None   # UVR.py:380
    is_denoise_model: bool = False          # UVR.py:383 (gated on the denoise weight)
    is_save_inst_vocal_splitter: bool = False    # UVR.py:468 <- Settings.is_save_inst_set_vocal_splitter
    is_inst_only_voc_splitter: bool = False      # UVR.py:469 via root.check_only_selection_stem (:7116)
    is_save_vocal_only: bool = False        # UVR.py:470 via root.check_only_selection_stem (:7116)
    is_secondary_model: bool = False        # UVR.py:430
    is_pre_proc_model: bool = False         # UVR.py:438
    is_vocal_split_model: bool = False      # UVR.py:466
    primary_model_primary_stem: str | None = None  # UVR.py:429
    is_primary_model_primary_stem_only: bool = False    # UVR.py:622 (from :7079)
    is_primary_model_secondary_stem_only: bool = False  # UVR.py:623 (from :7079)
    is_ensemble_mode: bool = False          # UVR.py:426/480
    is_multi_stem_ensemble: bool = False    # UVR.py:456/484-485
    ensemble_primary_stem: str | None = None    # UVR.py:427/477
    ensemble_secondary_stem: str | None = None  # UVR.py:427/477
    bv_model_rebalance: typing.Any = 0      # UVR.py:459/690-691
    is_sec_bv_rebalance: bool = False       # UVR.py:460/672-673
    demucs_version: str | None = None       # UVR.py:725-729 (via spec.model_data, 11a); unassigned for other archs
    demucs_source_list: list = dataclasses.field(default_factory=list)     # UVR.py:732/734
    demucs_source_map: dict = dataclasses.field(default_factory=dict)      # UVR.py:732/734
    demucs_stem_count: int = 0              # UVR.py:732/734
    is_secondary_model_activated: bool = False  # UVR.py:491/526/601, refined :640/:679-681/:657
    secondary_model: typing.Optional["RunPlan"] = None       # UVR.py:677 (:7076)
    secondary_model_scale: float | None = None               # UVR.py:678 (:7073)
    secondary_model_4_stem: list = dataclasses.field(default_factory=list)       # UVR.py:445/635
    secondary_model_4_stem_scale: list = dataclasses.field(default_factory=list)  # UVR.py:446/636
    pre_proc_model: typing.Optional["RunPlan"] = None        # UVR.py:652 (:7093)
    vocal_split_model: typing.Optional["RunPlan"] = None     # UVR.py:668 (:7108)


#: The 92 names, from the dataclass itself -- never hand-copied into an assertion.
PLAN_FIELD_NAMES: tuple[str, ...] = tuple(f.name for f in dataclasses.fields(RunPlan))

#: Sentinel: `_build` must derive every one of the 92, so "absent" is visible.
_UNSET = object()


def _selection(selections: dict, group: str, slot: str) -> tuple[str | None, float | None]:
    """One of the 24 per-arch slot pairs (`UVR.py:1634/1644/1654`, read at :7060-7070)."""
    table = selections.get(group) or {}
    name = table.get(slot)
    scale = table.get(f"{slot}_scale")
    return (None if name in (None, "") else str(name)), (None if scale in (None, "") else float(scale))


def _build(*, settings: settings_module.Settings, cat: catalog.Catalog,
           backend: backends.Backend, spec: catalog.ModelSpec,
           selections: dict, is_secondary_model: bool = False,
           primary_model_primary_stem: str | None = None,
           is_primary_model_primary_stem_only: bool = False,
           is_primary_model_secondary_stem_only: bool = False,
           is_pre_proc_model: bool = False, is_vocal_split_model: bool = False,
           depth: int = 0) -> RunPlan:
    """The body of `build()`, with the recursion-only arguments visible.

    Upstream hands these to `ModelData.__init__` (`UVR.py:366-376`) when the GUI
    builds a secondary, a pre-proc or a vocal-split model (`:7076-7080`, `:7093-7095`,
    `:7108`); `build()`'s public signature keeps none of them (R-T11-scope).
    """
    if depth > 3:
        # Upstream's deepest legitimate chain is primary -> pre-proc -> vocal split
        # -> secondary (`UVR.py:652` -> `:668` -> `:648`), and it terminates because
        # a secondary built with the *ctor argument* `is_secondary_model=True` never
        # resolves another one (`:491/:526/:601`), never a pre-proc (`:620/:650`) and
        # never a vocal split (`:667`, via the effective flag at `:430`).  A fourth
        # level therefore only happens when the selections point at each other in a
        # cycle; refusing is the honest answer, not a hang at run time.
        raise PlanError(f"{spec.name}: nested plans recurse at most three levels (UVR.py:620/:650/:667)")

    blocked = cat.is_blocked(spec.arch)
    if blocked:
        raise PlanError(f'"{spec.name}" is arch {spec.arch!r}, which this fork blocks: {blocked}')
    process_method = spec.process_method
    if process_method not in (VR_ARCH_TYPE, MDX_ARCH_TYPE, DEMUCS_ARCH_TYPE):
        raise PlanError(
            f'"{spec.name}" has process method {process_method!r}: no branch of '
            f"separate.py:295-374 reads such a model, so a plan would be invented.")

    params = spec.model_data
    if params is None:
        # Tier 3: upstream's `get_model_data_from_popup` (`UVR.py:759-777`) is a
        # blocking dialog. A plan cannot open it, and filling the gaps with
        # defaults would be exactly the "plausible-looking" fabrication 11b forbids.
        raise PlanError(spec.missing_message)
    if spec.hash == WOOD_INST_MODEL_HASH and process_method == VR_ARCH_TYPE:
        params = WOOD_INST_PARAMS  # UVR.py:509's hardcoded special case

    model_path = spec.weights_path                       # UVR.py:501/704-707/716-721
    model_basename = (os.path.splitext(os.path.basename(str(model_path)))[0]
                      if model_path else None)           # UVR.py:616

    device_set = (settings.device_set.split(":")[-1].strip()
                  if ":" in settings.device_set else settings.device_set)  # UVR.py:410
    is_gpu_conversion = 0 if settings.gpu_conversion else -1               # UVR.py:384
    is_denoise = settings.denoise_option != DENOISE_NONE                   # UVR.py:389
    denoiser = _model_file(cat, f"VR_Models/{'UVR-DeNoise-Lite.pth'}")     # UVR.py:295
    deverber = _model_file(cat, f"VR_Models/{'UVR-DeEcho-DeReverb.pth'}")  # UVR.py:296
    is_denoise_model = (settings.denoise_option == DENOISE_M
                        and denoiser is not None and denoiser.is_file())   # UVR.py:383
    is_deverb_vocals = (settings.is_deverb_vocals
                        and deverber is not None and deverber.is_file())   # UVR.py:381
    is_mixer_mode = False                                                  # UVR.py:414

    plan_kwargs: dict[str, typing.Any] = {name: _UNSET for name in PLAN_FIELD_NAMES}
    plan_kwargs.update(dict(
        process_method=process_method,
        model_name=spec.name,                             # UVR.py:421
        model_path=model_path,
        model_basename=model_basename,
        mixer_path=_REPO / "lib_v5" / "mixer.ckpt",       # UVR.py:243/419/709
        DENOISER_MODEL=denoiser,                          # UVR.py:379
        DEVERBER_MODEL=deverber,                          # UVR.py:380
        is_normalization=settings.is_normalization,       # UVR.py:385
        is_primary_stem_only=settings.is_primary_stem_only,      # UVR.py:387 (:610 overrides for Demucs)
        is_secondary_stem_only=settings.is_secondary_stem_only,  # UVR.py:388 (:611 overrides for Demucs)
        is_mdx_c_seg_def=settings.is_mdx_c_seg_def,       # UVR.py:390
        mdx_batch_size=(1 if settings.mdx_batch_size == DEF_OPT
                        else int(_num("mdx_batch_size", settings.mdx_batch_size))),  # UVR.py:391
        is_mdx_combine_stems=settings.is_mdx23_combine_stems,   # UVR.py:401 (near-miss rename)
        wav_type_set=settings.wav_type_set,               # UVR.py:409
        device_set=device_set,
        mp3_bit_set=settings.mp3_bit_set,                 # UVR.py:411
        save_format=settings.save_format,                 # UVR.py:412
        is_invert_spec=settings.is_invert_spec,           # UVR.py:413
        demucs_stems=settings.demucs_stems,               # UVR.py:415
        is_demucs_combine_stems=settings.is_demucs_combine_stems,  # UVR.py:416
        is_deverb_vocals=is_deverb_vocals,
        deverb_vocal_opt=DEVERB_MAPPER[settings.deverb_vocal_opt],  # UVR.py:382
        is_match_frequency_pitch=settings.is_match_frequency_pitch,  # UVR.py:398
        semitone_shift=_num("semitone_shift", settings.semitone_shift),  # UVR.py:396
        is_pitch_change=_num("semitone_shift", settings.semitone_shift) != 0,  # UVR.py:397
        overlap=(0.25 if settings.overlap == DEFAULT else _num("overlap", settings.overlap)),  # UVR.py:393
        overlap_mdx=(settings.overlap_mdx if settings.overlap_mdx == DEFAULT
                     else _num("overlap_mdx", settings.overlap_mdx)),  # UVR.py:394
        overlap_mdx23=int(_num("overlap_mdx23", settings.overlap_mdx23)),  # UVR.py:395
        is_gpu_conversion=is_gpu_conversion,
        is_mixer_mode=is_mixer_mode,
        # `chunks` is assigned only inside upstream's MDX branch (`UVR.py:528` sets
        # 0); `separate.py:315` reads it there and nowhere else, so the other archs
        # keep the `DEFAULT_DATA['chunks']` value rather than inventing one.
        chunks=0 if process_method == MDX_ARCH_TYPE else "Auto",
        is_denoise=is_denoise,
        is_denoise_model=is_denoise_model,
        is_save_inst_vocal_splitter=settings.is_save_inst_set_vocal_splitter,  # UVR.py:468 rename
        is_secondary_model=is_vocal_split_model or is_secondary_model,  # UVR.py:430
        is_pre_proc_model=is_pre_proc_model,              # UVR.py:438
        is_vocal_split_model=is_vocal_split_model,        # UVR.py:466
        primary_model_primary_stem=primary_model_primary_stem,  # UVR.py:429
        is_primary_model_primary_stem_only=is_primary_model_primary_stem_only,  # UVR.py:622
        is_primary_model_secondary_stem_only=is_primary_model_secondary_stem_only,  # UVR.py:623
        is_ensemble_mode=False,                           # UVR.py:426/480 (no ensemble spec)
        is_multi_stem_ensemble=False,                     # UVR.py:456/484-485
        ensemble_primary_stem=None,                       # UVR.py:427/477
        ensemble_secondary_stem=None,                     # UVR.py:427/477
        model_samplerate=44100,                           # UVR.py:440
        model_capacity=(32, 128),                         # UVR.py:441/500
        is_roformer=bool(params.get("is_roformer", False)),  # UVR.py:541-542
        is_karaoke=bool(params.get(IS_KARAOKEE, False)),  # UVR.py:686-687
        is_bv_model=bool(params.get(IS_BV_MODEL, False)),  # UVR.py:688-689
        bv_model_rebalance=(params.get(IS_BV_MODEL_REBAL, 0)
                            if params.get(IS_BV_MODEL, False) else 0),  # UVR.py:690-691
        # Assigned unconditionally by `ModelData` in one of the three arch branches,
        # with the same coercion whichever branch ran, so they are set once here.
        margin=int(_num("margin", settings.margin)),                    # UVR.py:527
        mdx_segment_size=int(_num("mdx_segment_size", settings.mdx_segment_size)),  # UVR.py:529
        shifts=int(_num("shifts", settings.shifts)),                    # UVR.py:606
        is_split_mode=settings.is_split_mode,                           # UVR.py:607
        segment=settings.segment,                                       # UVR.py:608
        is_chunk_demucs=settings.is_chunk_demucs,                       # UVR.py:609
        mdxnet_stem_select=settings.mdxnet_stems,                       # UVR.py:392 (near-miss rename)
        aggression_setting=_num("aggression_setting", settings.aggression_setting) / 100,  # UVR.py:492
        is_tta=settings.is_tta,                     # UVR.py:493
        is_post_process=settings.is_post_process,   # UVR.py:494
        window_size=int(_num("window_size", settings.window_size)),  # UVR.py:495
        batch_size=1 if settings.batch_size == DEF_OPT else int(_num("batch_size", settings.batch_size)),  # :496
        is_high_end_process="mirroring" if settings.is_high_end_process else "None",  # :498
        post_process_threshold=_num("post_process_threshold", settings.post_process_threshold),  # :499
        # Upstream's own initializers for the fields that only one arch's branch
        # fills (`UVR.py:399-472`): for another arch the attribute exists and holds
        # this, and `separate.py` reads it only inside the branch that sets it.
        compensate=None,                                                # UVR.py:407 (:584 for MDX)
        vr_model_param=None,                                            # UVR.py:514 (VR only)
        mdx_dim_f_set=None,                                             # UVR.py:404 (:585)
        mdx_dim_t_set=None,                                             # UVR.py:405 (:586)
        mdx_n_fft_scale_set=None,                                       # UVR.py:408 (:587)
        mdx_c_configs=None,                                             # UVR.py:402 (:550)
        is_mdx_c=False,                                                 # UVR.py:400 (:545)
        is_mdx_ckpt=False,                                              # UVR.py:399 (:696)
        is_target_instrument=False,                                     # UVR.py:472 (:553)
        primary_stem_native=None,                                       # UVR.py:425 (:516/:589)
        is_vr_51_model=False,                                           # UVR.py:442 (:519)
        secondary_model=None,                                           # UVR.py:431
        secondary_model_scale=None,                                     # UVR.py:432
        secondary_model_4_stem=[],                                      # UVR.py:445
        secondary_model_4_stem_scale=[],                                # UVR.py:446
        pre_proc_model=None,                                            # UVR.py:436
        vocal_split_model=None,                                         # UVR.py:465
        is_sec_bv_rebalance=False,                                      # UVR.py:460 (:673)
        is_demucs_pre_proc_model_inst_mix=False,                        # UVR.py:443 (:654)
        is_secondary_model_activated=False,                             # UVR.py:464
        demucs_version=None,                                            # UVR.py:725 (Demucs only)
        demucs_source_list=[],                                          # UVR.py:417 (:732/734)
        demucs_source_map={},                                           # UVR.py:732/734
        demucs_stem_count=0,                                            # UVR.py:418 (:732/734)
    ))

    # -- stem-only selections (`UVR.py:7116-7145), keyed off the arch the GUI
    # chose, exactly as `check_only_selection_stem` does.  The two label vars
    # (`is_primary_stem_only_Text_var`) are checkbox *text* and have no `Settings`
    # field; the caller supplies them inside the vocal-split selection entry.
    vsel = selections.get("vocal_splitter") or {}
    demucs_branch = settings.chosen_process_method == DEMUCS_ARCH_TYPE
    primary_label = vsel.get("primary_stem_only_label") or ""
    secondary_label = vsel.get("secondary_stem_only_label") or ""
    primary_only = (settings.is_primary_stem_only_Demucs if demucs_branch
                    else settings.is_primary_stem_only)                  # UVR.py:7122
    secondary_only = (settings.is_secondary_stem_only_Demucs if demucs_branch
                      else settings.is_secondary_stem_only)              # UVR.py:7124
    splitter_name = vsel.get("name")
    splitter_on = bool(splitter_name) and splitter_name != catalog.NO_MODEL
    # UVR.py:7136-7140 (`IS_SAVE_VOC_ONLY`)
    plan_kwargs["is_save_vocal_only"] = bool(
        (VOCAL_STEM_ONLY == primary_label and primary_only)
        or (VOCAL_STEM_ONLY == secondary_label and secondary_only))
    # UVR.py:7131-7134 (`INST_STEM_ONLY`)
    plan_kwargs["is_inst_only_voc_splitter"] = bool(
        (INST_STEM_ONLY == primary_label and primary_only
         and settings.is_save_inst_set_vocal_splitter and splitter_on)
        or (INST_STEM_ONLY == secondary_label and secondary_only
            and settings.is_save_inst_set_vocal_splitter and splitter_on))

    # -- VR branch (`UVR.py:490-520) ------------------------------------------- #
    primary_stem = spec.primary_stem
    # `UVR.py:430`: a vocal-split model is a secondary model as far as every later
    # gate (`:620`, `:667`) is concerned, even though the ctor argument differs.
    sec_effective = is_vocal_split_model or is_secondary_model
    if process_method == VR_ARCH_TYPE:
        plan_kwargs["is_secondary_model_activated"] = (
            settings.vr_is_secondary_model_activate and not is_secondary_model)  # UVR.py:491
        param_name = params.get("vr_model_param")     # UVR.py:511
        param_file = _REPO / "lib_v5" / "vr_network" / "modelparams" / f"{param_name}.json"
        if not param_file.is_file():
            raise PlanError(
                f'"{spec.name}" names VR params {param_name!r}, which is not at '
                f"{param_file} (UVR.py:260/511-514); refusing to guess a sample rate.")
        vr_params = ModelParameters(str(param_file))  # UVR.py:514
        plan_kwargs.update(
            vr_model_param=vr_params,
            model_samplerate=vr_params.param["sr"],   # UVR.py:515
            primary_stem_native=primary_stem,         # UVR.py:516
        )
        if "nout" in params and "nout_lstm" in params:  # UVR.py:517-519
            plan_kwargs["model_capacity"] = (params["nout"], params["nout_lstm"])
            plan_kwargs["is_vr_51_model"] = True
    elif process_method == MDX_ARCH_TYPE:
        # -- MDX branch (`UVR.py:525-596) ------------------------------------- #
        plan_kwargs["is_secondary_model_activated"] = (
            settings.mdx_is_secondary_model_activate and not is_secondary_model)  # UVR.py:526
        plan_kwargs["is_mdx_ckpt"] = str(spec.name).endswith(".ckpt")     # UVR.py:695-696
        config_yaml = params.get("config_yaml")
        if config_yaml:                                                     # UVR.py:544-550
            plan_kwargs["is_mdx_c"] = True
            config_file = _model_file(cat, f"MDX_Net_Models/model_data/mdx_c_configs/{config_yaml}")
            if config_file is None:
                raise PlanError(
                    f'"{spec.name}" is a config-driven model whose yaml {config_yaml!r} '
                    f"is not under {cat.model_root}; upstream marks it not-runnable "
                    "(UVR.py:549-582 sets model_status False), and this fork blocks "
                    "mdx_c rather than simulating it.")
            with config_file.open("r", encoding="utf-8") as handle:
                plan_kwargs["mdx_c_configs"] = yaml.safe_load(handle)
            target = _dig(plan_kwargs["mdx_c_configs"], "training", "target_instrument")
            if target:                                                      # UVR.py:552-557
                plan_kwargs["is_target_instrument"] = True
                primary_stem = target
            else:
                instruments = _dig(plan_kwargs["mdx_c_configs"], "training", "instruments") or []
                if len(instruments) == 2:                                 # UVR.py:570-576
                    primary_stem = instruments[0]
                else:
                    primary_stem = settings.mdxnet_stems                  # UVR.py:576 (mdxnet_stem_select)
        else:
            plan_kwargs.update(                                           # UVR.py:584-589
                compensate=(params.get("compensate", AUTO_SELECT)
                            if settings.compensate == AUTO_SELECT
                            else _num("compensate", settings.compensate)),
                mdx_dim_f_set=params.get("mdx_dim_f_set"),
                mdx_dim_t_set=params.get("mdx_dim_t_set"),
                mdx_n_fft_scale_set=params.get("mdx_n_fft_scale_set"),
                primary_stem_native=params.get("primary_stem"),
            )
    else:
        # -- Demucs branch (`UVR.py:600-613) + `get_demucs_model_data` (:723-738) #
        plan_kwargs.update(
            is_secondary_model_activated=settings.demucs_is_secondary_model_activate  # UVR.py:601
            and not is_secondary_model,
            is_primary_stem_only=(settings.is_primary_stem_only            # UVR.py:610
                                  if plan_kwargs["is_ensemble_mode"]
                                  else settings.is_primary_stem_only_Demucs),
            is_secondary_stem_only=(settings.is_secondary_stem_only        # UVR.py:611
                                    if plan_kwargs["is_ensemble_mode"]
                                    else settings.is_secondary_stem_only_Demucs),
            demucs_version=params.get("demucs_version", DEMUCS_V4),         # UVR.py:725-729
            demucs_source_list=list(params.get("demucs_source_list", [])),  # UVR.py:732/734
            demucs_source_map=dict(params.get("demucs_source_map", {})),    # UVR.py:732/734
            demucs_stem_count=int(params.get("demucs_stem_count", 0)),      # UVR.py:732/734
        )
        primary_stem = (PRIMARY_STEM if settings.demucs_stems == ALL_STEMS  # UVR.py:737
                        else settings.demucs_stems)

    if is_vocal_split_model and model_path is not None:                    # UVR.py:656-662
        # A splitter is named for what it takes *out* of the vocal track, so its own
        # stems swap: a bv model separates Backing from the rest, a plain one Lead.
        if plan_kwargs["is_bv_model"]:
            primary_stem = (BV_VOCAL_STEM if plan_kwargs["primary_stem_native"] == VOCAL_STEM
                            else LEAD_VOCAL_STEM)                          # UVR.py:659
        else:
            primary_stem = (LEAD_VOCAL_STEM if plan_kwargs["primary_stem_native"] == VOCAL_STEM
                            else BV_VOCAL_STEM)                            # UVR.py:661

    plan_kwargs["primary_stem"] = primary_stem
    plan_kwargs["secondary_stem"] = catalog.secondary_stem(primary_stem) if primary_stem else None  # :513/594/738

    # -- nested models: the four `root.` decision methods (`UVR.py:7046-7114) --- #
    def child(child_spec: catalog.ModelSpec, **flags) -> RunPlan:
        return _build(settings=settings, cat=cat, backend=backend, spec=child_spec,
                      selections=selections, depth=depth + 1, **flags)

    def _secondary_for_stem(stem: str | None):
        """One slot of `process_determine_secondary_model` (`UVR.py:7046-7086`).

        Returns the child plan and its scale, or `(None, None)` when the slot says
        "No Model Selected" or names a weight the catalog cannot resolve
        (`UVR.py:7075-7084`, which upstream also reaches through
        `secondary_for`'s `CatalogError` swallow).
        """
        slot = catalog.secondary_slot_for(process_method, stem)             # UVR.py:7059-7070
        if slot is None:
            return None, None
        requested, scale = _selection(selections, group, slot)
        child_spec = cat.secondary_for(spec, requested)                     # UVR.py:7075-7082
        if child_spec is None:
            return None, None                                               # UVR.py:7081-7084
        nested = child(child_spec, is_secondary_model=True,
                       primary_model_primary_stem=stem,
                       is_primary_model_primary_stem_only=plan_kwargs["is_primary_stem_only"],
                       is_primary_model_secondary_stem_only=plan_kwargs["is_secondary_stem_only"])
        if nested.model_basename == model_basename:                         # UVR.py:680-681
            return None, None
        return nested, scale

    activated = bool(plan_kwargs["is_secondary_model_activated"]) and model_path is not None
    group = _SELECTION_GROUP[process_method]                                # UVR.py:7052-7057
    if activated:
        is_valid_ensemble = (not plan_kwargs["is_ensemble_mode"]
                             and settings.demucs_stems == ALL_STEMS
                             and process_method == DEMUCS_ARCH_TYPE)        # UVR.py:626-628
        if is_valid_ensemble:
            # A Demucs "All Stems" run carries one secondary per stem, in the
            # source list's order (`UVR.py:632-641`), and `separate.py:323-324`
            # reads the two parallel lists.  `secondary_model`/`_scale` keep the
            # last slot's value, as they do upstream after the loop (`:635-636`).
            for stem in DEMUCS_4_SOURCE_LIST:                             # UVR.py:633
                nested, scale = _secondary_for_stem(stem)
                plan_kwargs["secondary_model_4_stem"].append(nested)        # UVR.py:635
                plan_kwargs["secondary_model_4_stem_scale"].append(scale)   # UVR.py:636
                plan_kwargs["secondary_model"] = nested
                plan_kwargs["secondary_model_scale"] = scale
            activated = any(i is not None
                            for i in plan_kwargs["secondary_model_4_stem"])  # UVR.py:640
        else:
            stem = primary_stem                                             # UVR.py:647
            nested, scale = _secondary_for_stem(stem)
            plan_kwargs["secondary_model"] = nested
            plan_kwargs["secondary_model_scale"] = scale
            activated = nested is not None                                  # UVR.py:679

    if process_method == DEMUCS_ARCH_TYPE and not is_secondary_model:      # UVR.py:650
        # `UVR.py:620` zeroes the pre-proc activation once `is_secondary_model` is
        # the *effective* flag (`:430` makes a vocal-split model a secondary too).
        pre_activated = (settings.is_demucs_pre_proc_model_activate
                         and not sec_effective
                         and settings.demucs_stems not in (VOCAL_STEM, INST_STEM))  # UVR.py:603
        if pre_activated and plan_kwargs["demucs_stem_count"] >= 3:
            requested = (selections.get("demucs_pre_proc_model")
                         or (selections.get("demucs") or {}).get("demucs_pre_proc_model"))
            pre_spec = cat.demucs_pre_proc_for(requested, is_activated=True)  # UVR.py:7088-7101
            if pre_spec is not None:
                plan_kwargs["pre_proc_model"] = child(pre_spec, is_pre_proc_model=True,
                                                      primary_model_primary_stem=primary_stem)
                plan_kwargs["is_demucs_pre_proc_model_inst_mix"] = (
                    settings.is_demucs_pre_proc_model_inst_mix)            # UVR.py:654

    if not sec_effective:                                                  # UVR.py:667
        vspec = cat.vocal_split_for(splitter_name, is_activated=bool(vsel.get("is_activated", True)))  # :7103-7114
        if vspec is not None:
            nested = child(vspec, is_vocal_split_model=True)
            plan_kwargs["vocal_split_model"] = nested
            # UVR.py:659-662 re-derives the splitter's own stems; `is_sec_bv_rebalance`
            # below reads it back off the child, and the splitter's primary_stem comes
            # from ITS own params, which `_build` already produced.
            plan_kwargs["is_sec_bv_rebalance"] = bool(nested.bv_model_rebalance)  # UVR.py:672-673

    if is_vocal_split_model and model_path is not None:
        # UVR.py:656-657 zeroes the *claim* on a vocal-split model, after `:648` may
        # already have resolved one; `secondary_model` keeps its value upstream, and
        # so does here -- `separate.py:173` reads the flag, which is what decides
        # whether the secondary is mixed in at all.
        activated = False

    plan_kwargs["is_secondary_model_activated"] = activated                # UVR.py:679-681/:640
    unset = sorted(name for name, value in plan_kwargs.items() if value is _UNSET)
    if unset:
        # A field the dataclass declares but `_build` never derives would otherwise
        # take its default and read as a value 11c can trust. It cannot: every one of
        # the 92 is derived above, and this is what keeps that claim alive as the
        # branches change.
        raise PlanError(f"RunPlan fields never derived by build(): {unset}")
    return RunPlan(**plan_kwargs)


def build(*, settings: settings_module.Settings, catalog: catalog.Catalog,
          backend: backends.Backend, spec: catalog.ModelSpec,
          selections: dict | None = None) -> RunPlan:
    """Compose one `RunPlan` (keyword-only: 92 fields make a positional call unauditable).

    `selections` is upstream's structure with the tk vars resolved
    (`UVR.py:1634/1644/1654`): `{"vr"|"mdx"|"demucs": {"voc_inst_secondary_model":
    <name>, "voc_inst_secondary_model_scale": <float>, "other_...", "drums_...",
    "bass_..."}, "demucs_pre_proc_model": <name>, "vocal_splitter": {"name": <name>,
    "is_activated": <bool>, "primary_stem_only_label": <str>, "secondary_stem_only_label":
    <str>}}`.  Absent or `catalog.NO_MODEL` means no model selected (`UVR.py:7075/:7084`).
    """
    return _build(settings=settings, cat=catalog, backend=backend, spec=spec,
                  selections=dict(selections or {}), is_secondary_model=False,
                  primary_model_primary_stem=None,
                  is_primary_model_primary_stem_only=False,
                  is_primary_model_secondary_stem_only=False,
                  is_pre_proc_model=False, is_vocal_split_model=False, depth=0)
