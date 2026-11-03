"""GUI-free golden separation runner -- what Tasks 7 and 11 gate against.

WHAT THIS HARNESS IS
--------------------
Two pinned facts, both measurable on this machine:

1. *Cross-device equivalence*: CPU torch vs ROCm torch, same venv, same frozen
   settings, same 30-second crop, same ``random`` seed. Scored by
   ``tests.golden.snr.snr_db >= 40``. This is the gate that can catch a real
   MIOpen/ROCm numeric defect.
2. *A pinned CPU fingerprint*: the WAVs captured on the stack this fork ships,
   re-scored after the Phase 1 refactor (Task 11) to prove it behaviour-
   preserving. The fingerprints live in **``tests/golden/pins.json``** (tracked,
   next to this module) and only that file is a pin. ``_artifacts/`` is generated
   output: ``baseline/*.wav`` are publications of the pinned bytes and
   ``baseline/manifest.json`` is a *report about the last run* -- never a pin,
   because a file every run rewrites cannot be the thing that notices a run
   changed. Moving a pin is an explicit operator act: ``UVR_REPIN=1``.

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
* A red ``test_pinned_fingerprint_did_not_move`` is a *finding*, not a step. On
  drift the harness leaves ``baseline/`` untouched -- the WAVs that carry the old
  numbers are the evidence -- and only ``UVR_REPIN=1`` re-publishes and moves the
  pins, which also records the audio delta it is absorbing.
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
from tests.golden.snr import max_abs_dev, snr_db

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


def capture_baseline(arch: str, device: str, work_dir: Path,
                     publish: bool = False) -> dict:
    """Run ``separate`` and return this capture's manifest record.

    The WAVs land in ``work_dir`` (the *staging* area). They are copied into
    ``baseline/`` only when ``publish`` is set -- the caller (the test fixture)
    decides, and it is only ever allowed to say yes when the capture reproduces
    the pinned bytes or when the operator asked for a deliberate re-pin. That is
    the property that was missing in Task 5: an automatic copy-over destroyed the
    evidence of the very drift it had just reported.

    Returns the record for this capture (timing, per-stem size and sha256,
    settings snapshot), with ``stems[stem]["path"]`` pointing at the staged file
    unless the record was published.
    """
    started = time.perf_counter()
    stems = separate(arch, device, work_dir)
    elapsed = time.perf_counter() - started

    entries = {}
    for stem_name, path in stems.items():
        info = sf.info(str(path))
        entries[stem_name] = {
            "path": str(path),
            "staged_path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "frames": int(info.frames),
            "sample_rate": int(info.samplerate),
            "channels": int(info.channels),
            "duration_seconds": round(info.frames / info.samplerate, 6),
            "subtype": info.subtype,
        }

    record = {
        "arch": arch,
        "device": device,
        "elapsed_seconds": round(elapsed, 3),
        "stems": entries,
        "model_data_attrs": _jsonable(settings.make(arch, device)),
    }
    if publish:
        publish_baseline(record)
    return record


def baseline_target(arch: str, stem_name: str) -> Path:
    """Where a published baseline WAV for this stem lives."""
    return settings.BASELINE_DIR / f"{arch}_{stem_name}.wav"


def publish_baseline(record: dict) -> None:
    """Copy a staged capture into ``baseline/`` and point the record at it.

    Safe by construction for the callers in this harness: it is only reached for
    a capture whose fingerprints already match ``pins.json`` (or during an
    explicit re-pin), so it republishes bytes the pins already vouch for.
    """
    settings.BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    for stem_name, entry in record["stems"].items():
        target = baseline_target(record["arch"], stem_name)
        shutil.copyfile(entry["staged_path"], target)
        if _sha256(target) != entry["sha256"]:
            raise RuntimeError(
                f"published baseline {target} does not match the capture it came "
                f"from ({entry['sha256']}); refusing to record a lie."
            )
        entry["path"] = str(target)


# --------------------------------------------------------------------------- #
# The pin store -- the only place a fingerprint is allowed to live
# --------------------------------------------------------------------------- #:

#: One LSB of the PCM_16 container the baselines are written in. Drift of this
#: size is what a different float32 reduction order costs; more than this means
#: the numbers moved for a reason worth reading.
PCM_16_LSB = 1.0 / 32768.0


def should_repin() -> bool:
    """True only when the operator said so out loud: ``UVR_REPIN=1``.

    Accidental triggers this rules out: the variable being absent, being set to
    ``0``/``""``/``no``/``yes``/``true``/``2`` (a bare ``export UVR_REPIN`` in a
    shell sets ``""``, which means *no*), and any value that merely contains a
    1. Re-pinning is a deliberate act, so it asks for a deliberate string.
    """
    return os.environ.get(settings.REPIN_ENV_VAR, "").strip() == "1"


def read_pins() -> dict | None:
    """The committed pin document, or ``None`` when no pin store exists yet."""
    if not settings.PINS_FILE.is_file():
        return None
    return json.loads(settings.PINS_FILE.read_text(encoding="utf-8"))


def pins_by_arch(document: dict | None) -> dict[str, dict[str, str]]:
    """``{arch: {stem: sha256}}`` flattened out of a pin document."""
    out: dict[str, dict[str, str]] = {}
    for arch, stems in ((document or {}).get("pins") or {}).items():
        out[arch] = {
            stem: (entry["sha256"] if isinstance(entry, dict) else entry)
            for stem, entry in stems.items()
        }
    return out


def capture_fingerprints(record: dict) -> dict[str, str]:
    """``{stem: sha256}`` for one capture record."""
    return {stem: entry["sha256"] for stem, entry in record["stems"].items()}


def baseline_fingerprints(arch: str | None = None) -> dict[str, dict[str, str]]:
    """``{arch: {stem: sha256}}`` read off the published WAVs that exist now.

    Files that are absent are simply not reported; "pinned but missing" is the
    caller's ``missing`` bucket, derived from the pin set.
    """
    out: dict[str, dict[str, str]] = {}
    if not settings.BASELINE_DIR.is_dir():
        return out
    for stem_file in sorted(settings.BASELINE_DIR.glob("*.wav")):
        arch_part, _, stem_part = stem_file.stem.partition("_")
        if arch is not None and arch_part != arch:
            continue
        out.setdefault(arch_part, {})[stem_part] = _sha256(stem_file)
    return out


def compare_fingerprints(pinned: dict[str, str], measured: dict[str, str]) -> dict:
    """Classify a measured fingerprint set against the pinned one.

    Three ways to disagree, all of them loud: a stem whose bytes moved
    (``changed``), a pinned stem this run no longer writes (``missing``), and a
    stem this run writes that nobody ever pinned (``unpinned`` -- the shape a
    silently-widened gate would take).
    """
    return {
        "changed": {
            stem: {"pinned": pinned[stem], "measured": measured[stem]}
            for stem in sorted(measured)
            if stem in pinned and measured[stem] != pinned[stem]
        },
        "missing": sorted(set(pinned) - set(measured)),
        "unpinned": sorted(set(measured) - set(pinned)),
    }


def measure_audio_delta(reference_paths: dict[str, str | Path],
                        candidate_paths: dict[str, str | Path]) -> dict:
    """Per-stem SNR / changed-sample / max-deviation report, old vs new.

    Used at re-pin time to record *what a re-pin is absorbing*, which is exactly
    the number Task 5 could only reconstruct afterwards from a pytest tmp dir
    that no longer exists. ``snr_db`` divides by the reference's power, so a
    near-silent stem reads low in dB -- hence ``max_abs_dev`` and the
    ``deviation_above_1_pcm16_lsb`` flag next to it.
    """
    report: dict[str, dict] = {}
    for stem in sorted(set(reference_paths) & set(candidate_paths)):
        ref_path, cand_path = Path(reference_paths[stem]), Path(candidate_paths[stem])
        if not (ref_path.is_file() and cand_path.is_file()):
            continue
        ref, _ = sf.read(str(ref_path), dtype="float64")
        cand, _ = sf.read(str(cand_path), dtype="float64")
        n = min(int(ref.size), int(cand.size))
        ref, cand = ref.ravel()[:n], cand.ravel()[:n]
        dev = max_abs_dev(ref, cand)
        snr = snr_db(ref, cand)
        changed = int(np.count_nonzero(ref != cand))
        report[stem] = {
            # ``snr_db`` is ``inf`` for byte-identical reruns; JSON has no
            # infinity, so the fact gets a flag instead of a fake number.
            "snr_db": round(snr, 3) if np.isfinite(snr) else None,
            "snr_is_infinite": bool(np.isinf(snr)),
            "samples_differing": changed,
            "samples_compared": n,
            "changed_fraction": round(changed / n, 8) if n else 0.0,
            "max_abs_dev": dev,
            "one_pcm16_lsb": PCM_16_LSB,
            "deviation_above_1_pcm16_lsb": bool(dev > PCM_16_LSB * 1.000001),
            "snr_below_40db_gate": bool(np.isfinite(snr) and snr < 40.0),
            "frames_equal": ref_path.stat().st_size == cand_path.stat().st_size,
        }
    return report


def engine_identity(device: str) -> dict:
    """Which stack a fingerprint belongs to -- recorded with the pins.

    ``device`` is the string the harness asked the engine for (``"cpu"`` under
    the ROCm wheel still means CPU kernels), and ``torch_cuda_available`` says
    whether a GPU was even in the room while the capture ran. Without these a
    sha256 cannot be attributed to a build, which is what made the Task 4 pins
    so easy to mistake for still-current ones.
    """
    import librosa
    import scipy

    crop = crop_fingerprint()
    return {
        "device": device,
        "torch": torch.__version__,
        "torch_hip": getattr(torch.version, "hip", None),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": (torch.cuda.get_device_name(0)
                     if torch.cuda.is_available() else None),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "librosa": librosa.__version__,
        "scipy": scipy.__version__,
        "soundfile": sf.__version__,
        "model_root": str(uvr_paths.MODEL_ROOT),
        "random_seed": settings.RANDOM_SEED,
        "crop_sha256": crop["crop_sha256"],
        "crop_frames": crop["crop_frames"],
        "crop_sample_rate": crop["crop_sample_rate"],
        "crop_channels": crop["crop_channels"],
    }


def write_pins(records: list[dict], *, reason: str,
               absorbed: dict | None = None,
               supersedes: dict | None = None) -> Path:
    """Re-pin: write ``tests/golden/pins.json`` from a fresh capture.

    The only writer of the pin store. Guarded twice over -- it refuses to run
    unless ``UVR_REPIN=1`` is in the environment, and it records *why* (the
    caller's reason string), which stack it was captured under, what it
    supersedes, and the measured audio delta it is absorbing, so a re-pin is a
    signed act with its evidence attached instead of a silent overwrite.
    """
    if not should_repin():
        raise RuntimeError(
            f"refusing to write {settings.PINS_FILE}: re-pinning requires the "
            f"explicit opt-in {settings.REPIN_ENV_VAR}=1."
        )
    devices = {record["device"] for record in records}
    if len(devices) != 1:
        raise ValueError(f"a pin store covers one device, got {sorted(devices)}")

    pins = {
        record["arch"]: {
            stem: {
                "sha256": entry["sha256"],
                "bytes": entry["bytes"],
                "frames": entry["frames"],
                "duration_seconds": entry["duration_seconds"],
                "subtype": entry["subtype"],
            }
            for stem, entry in sorted(record["stems"].items())
        }
        for record in sorted(records, key=lambda r: r["arch"])
    }
    document = {
        "schema": 1,
        "purpose": (
            "Committed CPU fingerprints for the golden gate. The ONLY pin store "
            "tests/golden reads; _artifacts/baseline/manifest.json is a run "
            "report, not a pin. Rewritten only by an operator passing "
            f"{settings.REPIN_ENV_VAR}=1 -- see tests/golden/runner.write_pins."
        ),
        "gate": {
            "fingerprint_rule": "sha256 of the PCM_16 WAV container, exact",
            "metric": "snr_db(reference=pinned_baseline, candidate=rerun)",
            "threshold_db": 40.0,
            "expected_repin_drift_class": "1 PCM_16 LSB = 3.0518e-05 per sample",
        },
        "identity": engine_identity(next(iter(devices))),
        "archs_covered": sorted(pins),
        "archs_blocked": _blocked_archs(),
        "pins": pins,
        "provenance": {
            "written_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "reason": reason,
            "supersedes": {
                "identity": (supersedes or {}).get("identity", {}),
                "written_at_utc": (supersedes or {}).get("written_at_utc"),
                "reason": (supersedes or {}).get("provenance", {}).get("reason"),
            },
            "superseded_pins": pins_by_arch(supersedes),
            "absorbed_audio_delta": absorbed or {},
            "elapsed_seconds": {r["arch"]: r["elapsed_seconds"] for r in records},
        },
    }
    settings.PINS_FILE.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return settings.PINS_FILE


def _blocked_archs() -> dict:
    """Archs the gate cannot cover on this machine, kept in the pin document too
    so a reader cannot mistake a 2-arch pin store for a 5-arch one."""
    return {
        "mdx_c": "no MDX-C checkpoint on this machine (model_data/mdx_c_configs yaml only)",
        "roformer": "no BS/Mel-Band/Apollo weights on this machine (configs only)",
        "onnx": "out of scope for this harness; Task 6 asserts the DirectML provider",
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
    """Write ``baseline/manifest.json`` next to the baseline WAVs.

    This file is a *report about the run that wrote it* -- regenerated every
    suite pass, and git-ignored. It is deliberately no longer the pin store:
    reading one's own output back as the reference is what let Task 5's drift
    heal itself. The pins live in ``tests/golden/pins.json``; the manifest says
    what this run decided about them (``pin_action``, drift, integrity).
    """
    import librosa
    import scipy

    settings.BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "purpose": (
            "Run report of the latest CPU capture: what this run measured, how "
            "long it took and what it decided about the pins. Fingerprints here "
            "are measurements, NOT pins -- see tests/golden/pins.json."
        ),
        "pin_store": str(settings.PINS_FILE),
        "gate": {"metric": "snr_db(reference=cpu_baseline, candidate=rerun)", "threshold_db": 40.0},
        "archs_covered": [r["arch"] for r in records],
        "archs_blocked": _blocked_archs(),
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "torch_hip": getattr(torch.version, "hip", None),
            "torch_cuda_version": getattr(torch.version, "cuda", None),
            # the device the harness *asked for*; under the ROCm wheel "cpu" is
            # a real choice, not the absence of a GPU (torch_cuda_available
            # below says whether one was in the room while this ran).
            "capture_device": sorted({r["device"] for r in records}),
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
