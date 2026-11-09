"""Frozen per-arch settings for the golden harness.

Every value here is *derived*, never invented. Two sources of truth:

* ``gui_data.constants.DEFAULT_DATA`` -- the application's out-of-the-box UI
  state. ``ui_defaults()`` snapshots it and only the model-selection keys are
  replaced (``DEFAULT_DATA`` ships ``CHOOSE_MODEL``, i.e. "nothing picked").
* ``UVR.py`` ``ModelData.__init__`` (lines 365-624) -- the tkinter -> engine
  translation the GUI performs. The derivation rules below mirror it line for
  line (each carries a ``UVR.py:NNN`` comment) because ``separate.py`` types its
  first constructor argument as ``ModelData`` and reads ~50 attributes off it.

Since Task 11c this module is no longer the object the engine is handed: the run
goes through ``runner.settings_from_ui`` (a real ``Settings`` over the same
``DEFAULT_DATA`` snapshot) and ``plan.build`` (the ``RunPlan`` ``separate()``
receives). What ``make()`` still is: the description the published pins were
captured under, and the reference side of ``tests/test_golden_refactor.py``'s
field-by-field comparison. It stays the only place the harness decides what "the
same run" means, which is why the values land in the baseline manifest rather
than in code comments.

The gate this feeds is *cross-device* equivalence on the modernised stack --
see ``tests.golden.runner``'s module docstring for why these numbers are not,
and must never be treated as, official-UVR reference output.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from gui_data.constants import (
    ALL_STEMS,
    DEF_OPT,
    DEMUCS_4_SOURCE,
    DEMUCS_4_SOURCE_MAPPER,
    DEMUCS_2_SOURCE,
    DEMUCS_2_SOURCE_MAPPER,
    DEMUCS_ARCH_TYPE,
    DEMUCS_UVR_MODEL,
    DEMUCS_VERSION_MAPPER,
    DEFAULT,
    DEFAULT_DATA,
    DEVERB_MAPPER,
    PRIMARY_STEM,
    VR_ARCH_TYPE,
    secondary_stem,
)
from lib_v5.vr_network.model_param_init import ModelParameters

import catalog
import uvr_paths

# --------------------------------------------------------------------------- #
# Harness-wide pinned inputs
# --------------------------------------------------------------------------- #

#: Runtime bounding ruling (plan Task 4): every capture -- Task 4 CPU baselines,
#: Task 7 ROCm comparison and Task 11 refactor re-run -- uses the same 30-second
#: crop of ``uvr_paths.INPUT_FILE`` (the committed CC0 excerpt, which is exactly this
#: long -- see ``tests/fixtures/PROVENANCE.md``).
CROP_SECONDS = 30

ARTIFACTS_DIR = Path(__file__).resolve().parent / "_artifacts"
CROP_FILE = ARTIFACTS_DIR / "golden_input_30s.wav"
BASELINE_DIR = ARTIFACTS_DIR / "baseline"
MANIFEST_FILE = BASELINE_DIR / "manifest.json"

#: The authoritative fingerprint store. Unlike everything under
#: ``_artifacts/`` (git-ignored, regenerated) this file is **tracked**, and it
#: is the only thing ``test_pinned_fingerprint_did_not_move`` compares against.
#: Kept outside ``_artifacts/`` on purpose: a ``git clean`` of the generated
#: baselines must not be able to take the pins with it.
PINS_FILE = Path(__file__).resolve().parent / "pins.json"

#: The ONE way the harness is allowed to move a pin or to overwrite a baseline
#: that disagrees with its pin. ``1`` exactly; anything else means "do not".
#: See ``runner.should_repin`` / ``runner.write_pins``.
REPIN_ENV_VAR = "UVR_REPIN"

#: Seed for ``random`` before each separation. demucs/apply.py:220 draws
#: ``random.randint`` for the "shift trick" (``shifts=2`` is the shipped
#: default), so an unseeded capture is not repeatable *on the same device*, let
#: onwards to a CPU-vs-ROCm gate. Pinning the seed keeps DEFAULT_DATA's real
#: ``shifts`` value while making the only remaining variable the numerics.
RANDOM_SEED = 20261103

#: The two architectures the golden gate can actually cover on this machine:
#: ``uvr_paths.MODELS`` has ``vr``/``demucs``/``onnx``; there is no MDX-C
#: checkpoint and no Roformer weight anywhere (configs only). ONNX is covered by
#: Task 6's provider assertion, not by this harness.
APPLY = ["vr", "demucs"]


class FixtureMissing(RuntimeError):
    """A golden fixture (weight/config/input) is absent -- a blocked task, not a
    skipped test. Raised rather than returning a placeholder path."""


def require_model(key: str) -> Path:
    if key not in uvr_paths.MODELS:
        raise FixtureMissing(
            f"no weight registered for {key!r} under {uvr_paths.MODEL_ROOT}; "
            "the golden gate cannot cover that arch (see APPLY in this module)."
        )
    return uvr_paths.MODELS[key]


# --------------------------------------------------------------------------- #
# UI layer: DEFAULT_DATA, with only the model selection filled in
# --------------------------------------------------------------------------- #

#: Model names as the GUI lists them. ``1_HP-UVR`` is the file stem of
#: ``uvr_paths.MODELS['vr']``; ``"v4 | htdemucs"`` is the value
#: ``Demucs_Models/model_data/model_name_mapper.json`` gives for
#: ``htdemucs.yaml``, the bag descriptor that resolves the ``955717e8`` weight.
VR_MODEL_NAME = "1_HP-UVR"
DEMUCS_MODEL_NAME = "v4 | htdemucs"


def ui_defaults(arch: str) -> dict:
    """``DEFAULT_DATA`` plus this harness's model selection for ``arch``.

    Values are read out of ``DEFAULT_DATA`` at call time rather than retyped
    here, so an invented default can never enter the gate.
    """
    data = dict(DEFAULT_DATA)
    if arch == "vr":
        data["vr_model"] = VR_MODEL_NAME
    elif arch == "demucs":
        data["demucs_model"] = DEMUCS_MODEL_NAME
    else:
        raise ValueError(f"no frozen settings for arch {arch!r}")
    return data


# --------------------------------------------------------------------------- #
# Derivation helpers (each mirrors a UVR.py ModelData line)
# --------------------------------------------------------------------------- #

def _vr_hash_entry() -> tuple[dict, Path]:
    """Look the fixture up in the model root's ``VR_Models/model_data/model_data.json``.

    Same two-step the GUI performs (``UVR.py:505-509``): prefer a per-hash
    ``{hash}.json`` in the hash dir, else the aggregate hash mapper. Raises
    rather than inventing parameters for an unrecognised weight.

    The digest comes from ``catalog.hash_of`` -- the one implementation in the
    repo (Task 11a retires this module's private copy, which duplicated
    ``UVR.py:793-799`` with a different fallback branch).
    """
    weights = require_model("vr")
    digest = catalog.hash_of(weights)
    hash_dir = uvr_paths.MODEL_ROOT / "VR_Models" / "model_data"
    per_hash = hash_dir / f"{digest}.json"
    if per_hash.is_file():
        return json.loads(per_hash.read_text(encoding="utf-8")), weights
    aggregate = hash_dir / "model_data.json"
    if aggregate.is_file():
        entry = json.loads(aggregate.read_text(encoding="utf-8")).get(digest)
        if entry:
            return entry, weights
    raise FixtureMissing(
        f"{weights.name} hashes to {digest}, which is not in {aggregate}. "
        "Refusing to guess its VR network parameters."
    )


def _model_params(param_name: str):
    """``UVR.py:511`` -- ``VR_PARAM_DIR/{vr_model_param}.json`` from lib_v5."""
    param_dir = Path(__file__).resolve().parents[2] / "lib_v5" / "vr_network" / "modelparams"
    param_file = param_dir / f"{param_name}.json"
    if not param_file.is_file():
        raise FixtureMissing(f"VR model parameters json missing at {param_file}")
    return ModelParameters(str(param_file))


def _demucs_version_and_sources(model_name: str) -> tuple[str, list, dict, int]:
    """``UVR.py:723-738 get_demucs_model_data`` for a non-ensemble run.

    That method branches only on ``DEMUCS_UVR_MODEL`` in the display name (2
    sources) else 4; the 6-stem model is handled elsewhere in the GUI and is not
    reproduced here because the fixture is ``htdemucs`` (4 sources).
    """
    version = "v4"
    for key, tag in DEMUCS_VERSION_MAPPER.items():
        if tag in model_name:
            version = key
    if DEMUCS_UVR_MODEL in model_name:
        return version, DEMUCS_2_SOURCE, DEMUCS_2_SOURCE_MAPPER, 2
    return version, DEMUCS_4_SOURCE, DEMUCS_4_SOURCE_MAPPER, 4


def _demucs_model_path() -> Path:
    """``UVR.py:711-721 get_demucs_model_path``.

    The GUI's ``model_path`` is the bag *descriptor*
    ``v3_v4_repo/htdemucs.yaml``; ``separate.py:909`` then calls
    ``demucs.pretrained.get_model(name='htdemucs', repo=<that dir>)``, which
    resolves the ``955717e8`` signature inside the local repo (``LocalRepo``
    keys by the pre-dash signature, so the ``.th`` file name itself is *not* a
    loadable name). ``uvr_paths.MODELS['demucs']`` is the weight, so the yaml is
    taken from its directory.
    """
    weights = require_model("demucs")
    yaml_path = weights.parent / "htdemucs.yaml"
    if not yaml_path.is_file():
        raise FixtureMissing(
            f"demucs bag descriptor missing at {yaml_path}; the weight "
            f"{weights.name} alone cannot be loaded by demucs.pretrained."
        )
    return yaml_path


# --------------------------------------------------------------------------- #
# Public interface
# --------------------------------------------------------------------------- #

def make(arch: str, device: str) -> dict:
    """Return the complete ``ModelData``-shaped attribute dict for ``arch``.

    ``device`` is ``"cpu"`` or a torch device name ("cuda" for ROCm). It maps to
    ``is_gpu_conversion``/``device_set`` exactly the way the GUI's GPU checkbox
    does (``UVR.py:384``, ``separate.py:174-184``); the engine decides the real
    device from ``torch.cuda.is_available()`` sampled at import time.

    The keys are the union of what ``SeperateAttributes.__init__`` reads
    unconditionally and what it reads inside the ``VR Arc`` / ``Demucs``
    architecture blocks, plus anything ``SeperateVR``/``SeperateDemucs``
    themselves touch. ``tests/test_golden_cpu.py`` re-derives that set from the
    ``separate.py`` source with AST and asserts this dict covers it, so a new
    ``model_data.<attr>`` read cannot silently break the harness.
    """
    arch = arch.lower()
    ui = ui_defaults(arch)

    # ---- shared derivation (UVR.py:379-472) --------------------------------
    denoiser_model = uvr_paths.MODEL_ROOT / "VR_Models" / "UVR-DeNoise-Lite.pth"
    deverb_model = uvr_paths.MODEL_ROOT / "VR_Models" / "UVR-DeEcho-DeReverb.pth"

    semitone_shift = float(ui["semitone_shift"])          # UVR.py:396
    attrs = {
        # UVR.py:379-383 -- paths only consumed when denoise/deverb are on; the
        # frozen defaults keep both off, so these stay inert but must exist.
        "DENOISER_MODEL": str(denoiser_model),
        "DEVERBER_MODEL": str(deverb_model),
        "is_deverb_vocals": bool(ui["is_deverb_vocals"]) if deverb_model.is_file() else False,
        "deverb_vocal_opt": DEVERB_MAPPER[ui["deverb_vocal_opt"]],
        "is_gpu_conversion": -1 if device == "cpu" else 0,   # UVR.py:384
        "is_normalization": ui["is_normalization"],           # UVR.py:385
        "is_primary_stem_only": ui["is_primary_stem_only"],   # UVR.py:387
        "is_secondary_stem_only": ui["is_secondary_stem_only"],  # UVR.py:388
        # UVR.py:392-395: all four are read unconditionally by
        # separate.py:92-98 even though only the MDX path uses their values.
        # The GUI leaves the MDX overlaps as the literal "Default" string;
        # separate.py only casts them on the MDX-Net path.
        "mdxnet_stem_select": ui["mdx_stems"],                # UVR.py:392
        "overlap": float(ui["overlap"]) if ui["overlap"] != DEFAULT else 0.25,
        "overlap_mdx": ui["overlap_mdx"],
        "overlap_mdx23": int(float(ui["overlap_mdx23"])),
        "semitone_shift": semitone_shift,
        "is_pitch_change": semitone_shift != 0,               # UVR.py:397
        "is_match_frequency_pitch": ui["is_match_frequency_pitch"],  # UVR.py:398
        "is_mdx_c": False,                                    # UVR.py:400
        "is_mdx_combine_stems": ui["is_mdx23_combine_stems"],  # UVR.py:401
        "mdx_c_configs": None,                                # UVR.py:402
        "model_samplerate": 44100,                            # UVR.py:440
        "model_capacity": (32, 128),                          # UVR.py:441
        "is_vr_51_model": False,                              # UVR.py:442
        "is_pre_proc_model": False,                           # UVR.py:438
        "is_demucs_pre_proc_model_inst_mix": False,           # UVR.py:443
        "wav_type_set": ui["wav_type_set"],                   # UVR.py:409
        # UVR.py:410 splits "Vendor:Model:Index" down to the index; the harness
        # leaves device selection to the GPU checkbox + torch, so DEFAULT.
        "device_set": DEFAULT,
        "mp3_bit_set": ui["mp3_bit_set"],                     # UVR.py:411
        "save_format": ui["save_format"],                     # UVR.py:412
        "is_invert_spec": ui["is_invert_spec"],               # UVR.py:413
        "is_mixer_mode": ui["is_mixer_mode"],                 # UVR.py:414
        "is_demucs_combine_stems": ui["is_demucs_combine_stems"],  # UVR.py:416
        "mixer_path": str(Path(__file__).resolve().parents[2] / "lib_v5" / "mixer.ckpt"),  # UVR.py:419
        "model_status": True,
        # UVR.py:429-435: single-model, non-ensemble, non-secondary run.  The
        # two is_primary_model_* keys are only read when is_secondary_model is
        # true (separate.py:114-115); they are supplied anyway because the AST
        # guard in tests/test_golden_cpu.py works per *line*, not per branch,
        # and over-approximating coverage is the safe direction for a guard.
        "primary_model_primary_stem": None,
        "is_secondary_model": False,
        "is_primary_model_primary_stem_only": False,
        "is_primary_model_secondary_stem_only": False,
        "is_secondary_model_activated": False,
        "secondary_model": None,
        "secondary_model_scale": None,
        "secondary_model_4_stem": [],
        "secondary_model_4_stem_scale": [],
        "pre_proc_model": None,
        "is_ensemble_mode": False,
        "ensemble_primary_stem": None,
        "ensemble_secondary_stem": None,
        "is_multi_stem_ensemble": False,
        "is_4_stem_ensemble": False,
        # UVR.py:465-472
        "vocal_split_model": None,
        "is_vocal_split_model": False,
        "is_save_inst_vocal_splitter": ui["is_save_inst_set_vocal_splitter"],
        "is_inst_only_voc_splitter": False,        # root.check_only_selection_stem(...)
        "is_save_vocal_only": False,               # root.check_only_selection_stem(...)
        "is_karaoke": False,
        "is_bv_model": False,
        "bv_model_rebalance": 0,
        "is_sec_bv_rebalance": False,
        "is_target_instrument": False,
        "is_roformer": False,
        "primary_stem_native": None,
    }

    # ---- per-architecture derivation ---------------------------------------
    if arch == "vr":
        entry, weights = _vr_hash_entry()
        params = _model_params(entry["vr_model_param"])
        primary_stem = entry["primary_stem"]
        attrs.update({
            "process_method": VR_ARCH_TYPE,                  # UVR.py:421
            "model_name": VR_MODEL_NAME,
            "model_path": str(weights),                      # UVR.py:501
            "model_basename": os.path.splitext(os.path.basename(weights))[0],  # UVR.py:616
            "aggression_setting": float(int(ui["aggression_setting"]) / 100),   # UVR.py:492
            "is_tta": ui["is_tta"],                          # UVR.py:493
            "is_post_process": ui["is_post_process"],        # UVR.py:494
            "window_size": int(ui["window_size"]),           # UVR.py:495
            "batch_size": 1 if ui["batch_size"] == DEF_OPT else int(ui["batch_size"]),  # UVR.py:496
            "crop_size": int(ui["crop_size"]),               # UVR.py:497
            "is_high_end_process": "mirroring" if ui["is_high_end_process"] else "None",  # UVR.py:498
            "post_process_threshold": float(ui["post_process_threshold"]),  # UVR.py:499
            "vr_model_param": params,                        # UVR.py:514
            "model_samplerate": params.param["sr"],          # UVR.py:515
            "primary_stem": primary_stem,                    # UVR.py:512
            "secondary_stem": secondary_stem(primary_stem),  # UVR.py:513
            "primary_stem_native": primary_stem,             # UVR.py:516
        })
        # UVR.py:517-519 + check_if_karaokee_model (UVR.py:685-691)
        if "nout" in entry and "nout_lstm" in entry:
            attrs["model_capacity"] = (entry["nout"], entry["nout_lstm"])
            attrs["is_vr_51_model"] = True
        attrs["is_karaoke"] = bool(entry.get("is_karaoke", False))
        attrs["is_bv_model"] = bool(entry.get("is_bv_model", False))
        if attrs["is_bv_model"]:
            attrs["bv_model_rebalance"] = entry.get("is_bv_model_rebalanced", 0)
    elif arch == "demucs":
        version, sources, source_map, stem_count = _demucs_version_and_sources(DEMUCS_MODEL_NAME)
        # UVR.py:736-738: demucs_stems is ALL_STEMS in DEFAULT_DATA, so the GUI
        # asks for the "Primary Stem" pair and separate.py:963 then takes the
        # all-stems branch and writes every demucs source instead of a pair.
        primary_stem = PRIMARY_STEM if ui["demucs_stems"] == ALL_STEMS else ui["demucs_stems"]
        weights = require_model("demucs")
        attrs.update({
            "process_method": DEMUCS_ARCH_TYPE,
            "model_name": DEMUCS_MODEL_NAME,
            "model_path": str(_demucs_model_path()),
            "model_basename": os.path.splitext(os.path.basename(weights))[0],
            "demucs_stems": ui["demucs_stems"],              # UVR.py:415
            "demucs_version": version,
            "demucs_source_list": sources,
            "demucs_source_map": source_map,
            "demucs_stem_count": stem_count,
            "segment": ui["segment"],                        # UVR.py:608
            "shifts": int(ui["shifts"]),                     # UVR.py:606
            "is_split_mode": ui["is_split_mode"],            # UVR.py:607
            "is_chunk_demucs": ui["is_chunk_demucs"],        # UVR.py:609
            "margin_demucs": int(ui["margin_demucs"]),       # UVR.py:604 (unused by separate.py)
            "primary_stem": primary_stem,
            "secondary_stem": secondary_stem(primary_stem),
            # UVR.py:610-611: the Demucs arch has its own stem-only checkboxes.
            "is_primary_stem_only": ui["is_primary_stem_only_Demucs"],
            "is_secondary_stem_only": ui["is_secondary_stem_only_Demucs"],
        })
    else:
        raise ValueError(f"no frozen settings for arch {arch!r}")

    return attrs
