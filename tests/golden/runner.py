"""GUI-free golden separation runner -- what Tasks 7 and 11 gate against.

WHAT THIS HARNESS IS
--------------------
Two pinned facts, both measurable on this machine:

1. *Cross-device equivalence*: CPU torch vs ROCm torch, same venv, same frozen
   settings, same 30-second crop, same ``random`` seed. Scored by
   ``tests.golden.snr.snr_db >= 40``. This is the gate that can catch a real
   MIOpen/ROCm numeric defect.
2. *A pinned CPU fingerprint*: the WAVs captured here on the modernised stack,
   re-scored after the Phase 1 refactor (Task 11) to prove it behaviour-
   preserving.

WHAT IT IS NOT
--------------
These baselines are **not** official-UVR reference output and must never be
treated as such. The dependency modernisation in Task 2/3 moved librosa to
1.0.0, whose own default ``res_type`` has been ``soxr_hq`` since 0.10, and of
the nineteen ``librosa.resample``/``librosa.load`` call sites in the modules
this engine executes (``separate.py``, ``lib_v5/**``, ``demucs/**``) **thirteen
bind no ``res_type`` at all** -- so they already resample differently from the
reference build no matter what this plan does. The historical WAVs in
``C:/Users/Shawn_Artifical/Documents/UVR5`` (November, produced by the official
CPU bundle) are a *listening* reference only: their parameters are unknown and
they are not a numeric target.

Consequences to respect:

* Do not "fix" a failing gate by loosening 40 dB or by re-scoring against the
  Documents/UVR5 files. If CPU-vs-ROCm SNR drops below 40 dB, a kernel or
  MIOpen behaviour changed; that is the finding, not the threshold.
* Do not add ``res_type`` to engine call sites to chase historical output; the
  Task 3 guard pins what is already there.
* The gate is deterministic *only* because ``random.seed`` is pinned:
  ``demucs/apply.py:220`` draws ``random.randint`` for the shift trick
  (``shifts=2`` is the shipped default), so an unseeded run varies between runs
  of the same arch on the same device. ``test_separate_pins_the_random_seed``
  checks the pinning and ``test_cpu_capture_is_byte_reproducible`` shows two
  seeded runs are identical to the byte.
* Stem energy matters at the gate. The fixture input is an *instrumental* mix,
  so the vocal stems are near-silence (vr_Vocals RMS 5e-3, demucs_Vocals RMS
  2e-4). ``snr_db`` divides by the reference's power, so a quiet stem yields a
  low dB number for a perfectly correct rerun while ``max_abs_dev`` stays tiny.
  Task 7 must report both, and judge a quiet-stem row on max deviation rather
  than decibels -- never by moving the threshold.

Model weights under ``uvr_paths.MODEL_ROOT`` are read-only inputs; this module
writes exclusively inside ``tests/golden/_artifacts/``.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import random
import shutil
import sys
import time
import types
from fractions import Fraction
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import separate as sep
import uvr_paths
from tests.golden import settings

#: arch -> Seperate* class. MDX-C and Roformer are absent on this machine (no
#: checkpoint, configs only) and ONNX is gated by Task 6's provider assertion,
#: so nothing here pretends to cover them. ``tests.golden.settings.APPLY`` lists
#: the archs whose weights actually resolve.
ARCH_CLASS = {
    "vr": sep.SeperateVR,
    "demucs": sep.SeperateDemucs,
}


class FixtureUnavailable(settings.FixtureMissing):
    """A weight/config/input the harness needs is missing (blocked, not skipped)."""


# --------------------------------------------------------------------------- #
# torch >= 2.6 `weights_only` allowlist for the demucs checkpoint
# --------------------------------------------------------------------------- #

def _numpy_scalar_reconstructor():
    """The callable numpy uses to rebuild pickled scalars, version-agnostically.

    ``numpy.core`` was renamed to ``numpy._core`` in numpy 2 and touching the
    old name emits a DeprecationWarning; the *checkpoint* still records the
    legacy dotted path, so the object is looked up under the new name and
    registered under the old one.
    """
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            from numpy._core.multiarray import scalar
        except ImportError:                      # numpy 1.x
            from numpy.core.multiarray import scalar
    return scalar


def _checkpoint_safe_globals():
    """Globals ``demucs/states.py:46`` needs in order to unpickle a v3/v4 ``.th``.

    ``torch.load(path, 'cpu')`` there inherits torch's 2.6+ default
    ``weights_only=True``, and a demucs checkpoint stores the network *class*
    itself (``{'klass', 'args', 'kwargs', 'state', 'training_args', 'metrics'}``),
    which weights-only unpickling rejects. Editing ``demucs/**`` is out of
    scope for this task, so the harness scopes PyTorch's own sanctioned
    allowlist around the call instead of turning the check off process-wide.

    The list is not guesswork: unpickling the fixture's ``archive/data.pkl``
    with a tracing ``pickle.Unpickler`` (no code executed) shows it references
    exactly eight globals, of which ``collections.OrderedDict``,
    ``_codecs.encode``, ``torch.HalfStorage`` and
    ``torch._utils._rebuild_tensor_v2`` are already on torch's built-in list.
    """
    from demucs.htdemucs import HTDemucs

    return [
        HTDemucs,
        Fraction,
        np.dtype,
        # ``np.dtype('float64')`` returns a *subclass* instance, and torch's
        # weights-only BUILD step matches on type(inst), so the concrete dtype
        # classes the checkpoint builds have to be named as well.
        np.dtypes.Float64DType,
        (_numpy_scalar_reconstructor(), "numpy.core.multiarray.scalar"),
    ]


@contextlib.contextmanager
def _inference_context(arch: str):
    if arch == "demucs":
        with torch.serialization.safe_globals(_checkpoint_safe_globals()):
            yield
    else:
        # VR checkpoints are plain state dicts; weights_only is left in force.
        yield


# --------------------------------------------------------------------------- #
# The pinned 30-second crop
# --------------------------------------------------------------------------- #

def golden_input() -> Path:
    """Return the 30-second crop of ``uvr_paths.INPUT_FILE``, creating it once.

    Written as PCM_16 at the source sample rate. The fixture input is a
    16-bit FLAC, and a 16-bit sample is exactly representable in the float
    round trip soundfile performs, so the crop is bit-identical to the first 30
    seconds of the source -- see ``crop_fingerprint`` and the Task 4 report.
    Tasks 7 and 11 reuse this file; they must never re-crop independently.
    """
    if settings.CROP_FILE.is_file():
        return settings.CROP_FILE

    source = uvr_paths.INPUT_FILE
    if not source.is_file():
        raise FixtureUnavailable(
            f"golden input missing at {source}; set UVR_INPUT_FILE."
        )

    data, samplerate = sf.read(str(source), dtype="float64")
    frames = int(round(settings.CROP_SECONDS * samplerate))
    if data.shape[0] < frames:
        raise FixtureUnavailable(
            f"{source} is shorter ({data.shape[0]} frames) than the "
            f"{settings.CROP_SECONDS}s golden crop."
        )

    settings.CROP_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = settings.CROP_FILE.with_suffix(".wav.part")
    sf.write(str(tmp), data[:frames], samplerate, subtype="PCM_16", format="WAV")
    shutil.move(str(tmp), str(settings.CROP_FILE))
    return settings.CROP_FILE


def crop_fingerprint() -> dict:
    """Exact duration/rate record for the baseline manifest (plan Task 4)."""
    path = golden_input()
    info = sf.info(str(path))
    source_info = sf.info(str(uvr_paths.INPUT_FILE))
    return {
        "crop_path": str(path),
        "crop_frames": int(info.frames),
        "crop_sample_rate": int(info.samplerate),
        "crop_channels": int(info.channels),
        "crop_duration_seconds": round(info.frames / info.samplerate, 6),
        "crop_subtype": info.subtype,
        "crop_format": info.format,
        "crop_sha256": _sha256(path),
        "crop_seconds_requested": settings.CROP_SECONDS,
        "source_path": str(uvr_paths.INPUT_FILE),
        "source_duration_seconds": round(source_info.frames / source_info.samplerate, 3),
        "source_subtype": source_info.subtype,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# process_data: the engine's second constructor argument
# --------------------------------------------------------------------------- #

def make_process_data(audio_file: Path, export_path: Path) -> dict:
    """The 11 keys ``SeperateAttributes.__init__`` indexes unconditionally.

    ``set_progress_bar``/``write_to_console``/``process_iteration`` keep the
    real GUI signatures (``UVR.py:7226-7227``) but are no-ops here; the source
    cache callbacks report "nothing cached" so every run does real inference
    instead of reusing a previous arch's tensors. Plain callables, not mocks.
    """
    export_path.mkdir(parents=True, exist_ok=True)
    console_log: list[str] = []

    def set_progress_bar(step, inference_iterations=0):
        return None

    def write_to_console(progress_text, base_text=""):
        console_log.append(f"{base_text}{progress_text}".strip())
        return None

    def cached_source_callback(process_method, model_name=None):
        # (model_name, sources) -- never a hit, so no branch short-circuits.
        return None, None

    def cached_model_source_holder(process_method, sources, model_name=None):
        return None

    def process_iteration():
        return None

    return {
        "audio_file": str(audio_file),
        # UVR.py:7229 builds "<index>_<stem>"; the crop is a single fixture so
        # the index is dropped. The engine appends "_(<Stem>).wav" itself.
        "audio_file_base": os.path.splitext(os.path.basename(str(audio_file)))[0],
        "export_path": str(export_path),
        "set_progress_bar": set_progress_bar,
        "write_to_console": write_to_console,
        "cached_source_callback": cached_source_callback,
        "cached_model_source_holder": cached_model_source_holder,
        "is_ensemble_master": False,
        "is_4_stem_ensemble": False,
        "list_all_models": [],
        "process_iteration": process_iteration,
    }


# --------------------------------------------------------------------------- #
# Separation
# --------------------------------------------------------------------------- #

def separate(arch: str, device: str, work_dir: Path) -> dict[str, Path]:
    """Run one frozen-settings separation and return ``{stem_name: wav_path}``.

    Keys are the engine's own stem names as they appear in the written file
    names -- ``{"Instrumental": ..., "Vocals": ...}`` for VR, and all four
    ``Bass/Drums/Other/Vocals`` for htdemucs, because ``DEFAULT_DATA`` ships
    ``demucs_stems = 'All Stems'`` and ``separate.py:963`` then takes the
    all-stems branch. Returning real names (rather than the plan sketch's
    "Primary"/"Secondary" placeholders) keeps Task 7's
    ``baseline/{arch}_{stem}.wav`` lookup honest for both shapes.

    There is no ``job.start()`` and no ``job.output_file_base`` attribute: the
    entry point is ``job.seperate()`` (``separate.py:471,638,873,1100``) and the
    only trustworthy record of what was written is the files in ``export_path``.
    """
    arch = arch.lower()
    if arch not in ARCH_CLASS:
        raise ValueError(
            f"arch {arch!r} is not covered by the golden harness "
            f"(available: {sorted(ARCH_CLASS)})."
        )
    settings.require_model(arch)

    work_dir = Path(work_dir)
    export_path = work_dir / f"{arch}_{device}"
    audio_file = golden_input()
    process_data = make_process_data(audio_file, export_path)

    attrs = settings.make(arch, device)
    model_data = types.SimpleNamespace(**attrs)

    # Deterministic shift trick (see module docstring) -- pinned per capture.
    random.seed(settings.RANDOM_SEED)

    job = ARCH_CLASS[arch](model_data, process_data)
    with _inference_context(arch):
        job.seperate()

    stems: dict[str, Path] = {}
    for written in sorted(export_path.rglob("*.wav")):
        if "_(" not in written.stem:
            continue
        stem_name = written.stem.split("_(")[-1].rstrip(")")
        stems[stem_name] = written

    if not stems:
        raise RuntimeError(
            f"{arch}/{device} produced no <base>_(<Stem>).wav files in {export_path}"
        )
    return stems


def capture_baseline(arch: str, device: str, work_dir: Path) -> dict:
    """Run ``separate`` and stage its WAVs as ``baseline/{arch}_{stem}.wav``.

    Returns the manifest record for this capture (timing, per-stem size and
    sha256, settings snapshot). Re-running replaces the previous baseline for
    that arch -- that is the intended way to refresh it after a deliberate
    dependency change, never as a way to make a failing gate pass.
    """
    started = time.perf_counter()
    stems = separate(arch, device, work_dir)
    elapsed = time.perf_counter() - started

    settings.BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    entries = {}
    for stem_name, path in stems.items():
        target = settings.BASELINE_DIR / f"{arch}_{stem_name}.wav"
        shutil.copyfile(path, target)
        info = sf.info(str(target))
        entries[stem_name] = {
            "path": str(target),
            "bytes": target.stat().st_size,
            "sha256": _sha256(target),
            "frames": int(info.frames),
            "sample_rate": int(info.samplerate),
            "channels": int(info.channels),
            "duration_seconds": round(info.frames / info.samplerate, 6),
            "subtype": info.subtype,
        }

    return {
        "arch": arch,
        "device": device,
        "elapsed_seconds": round(elapsed, 3),
        "stems": entries,
        "model_data_attrs": _jsonable(settings.make(arch, device)),
    }


def _jsonable(value):
    """Make a settings snapshot JSON-serialisable without losing information."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    param = getattr(value, "param", None)  # ModelParameters
    if param is not None:
        return {"ModelParameters": _jsonable(param)}
    return repr(value)


def write_manifest(records: list[dict], extra: dict | None = None) -> Path:
    """Write ``baseline/manifest.json`` next to the baseline WAVs."""
    import librosa
    import scipy

    settings.BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "purpose": (
            "CPU fingerprint of the modernised stack for cross-device (Task 7) "
            "and refactor (Task 11) SNR gating. NOT official UVR reference "
            "output -- see tests/golden/runner.py module docstring."
        ),
        "gate": {"metric": "snr_db(reference=cpu_baseline, candidate=rerun)", "threshold_db": 40.0},
        "archs_covered": [r["arch"] for r in records],
        "archs_blocked": {
            "mdx_c": "no MDX-C checkpoint on this machine (model_data/mdx_c_configs yaml only)",
            "roformer": "no BS/Mel-Band/Apollo weights on this machine (configs only)",
            "onnx": "out of scope for this harness; Task 6 asserts the DirectML provider",
        },
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "torch_hip": getattr(torch.version, "hip", None),
            "numpy": np.__version__,
            "librosa": librosa.__version__,
            "scipy": scipy.__version__,
            "soundfile": sf.__version__,
            "model_root": str(uvr_paths.MODEL_ROOT),
        },
        "input": crop_fingerprint(),
        "random_seed": settings.RANDOM_SEED,
        "captures": records,
    }
    if extra:
        manifest.update(extra)

    settings.MANIFEST_FILE.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return settings.MANIFEST_FILE
