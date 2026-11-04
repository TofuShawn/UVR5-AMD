"""Task 7: ROCm <-> CPU golden equivalence -- the gate the fork's claim rests on.

THE PREMISE UNDER TEST
----------------------
AMD's ROCm torch wheel presents itself as CUDA: ``torch.cuda.is_available()`` is
True and the torch device type is ``"cuda"``; ``torch.version.hip`` is the only
discriminator (see ``backends.py``). If that is true, model code needs no device
changes, and the proof of it is that the *same* engine, the *same* frozen
settings and the *same* 30-second crop produce the *same* audio on the GPU as the
pinned CPU baselines. If some architecture fails to match, the premise is wrong
for that arch and this fork's rationale has to be reported as such -- which is a
result, not a step to route around.

WHAT IS PINNED AND WHAT VARIES
------------------------------
The references are ``tests/golden/_artifacts/baseline/{arch}_{stem}.wav``,
published by Task 4's CPU suite from the *same venv, same torch*
(``2.9.1+rocm7.2.1``) under ``device="cpu"`` -- Task 5 re-pinned them for exactly
that reason (``pins.json`` ``provenance.reason``). So the only variable this file
moves is the device. ``settings.make(arch, "cuda")`` sets ``is_gpu_conversion =
0``, which is the GUI's GPU checkbox, and the engine takes its device from
``backends.resolve()`` + ``torch.cuda``.

THE TWO GUARDS THAT MAKE "A GREEN ROW IS A REAL GPU RUN" A CHECKED FACT
----------------------------------------------------------------------
The switch above is a *request*, not an outcome: ``separate.py`` initialises
``self.device = cpu`` and overwrites it only inside
``if self.backend.torch_device == 'cuda' and cuda_available``. If that condition
ever fails, the engine warns and computes on CPU; the candidate is then
byte-identical to the CPU baseline, ``snr_db`` returns ``inf``, ``inf >= 40``
passes, and the stems still match by name. That is CPU scored against CPU while
the suite reads green, which is the one failure mode this file exists to prevent,
so every run here is wrapped twice:

1. **in-run warning inspection** -- ``separate()`` is called inside
   ``warnings.catch_warnings(record=True)``, and any warning carrying the
   engine's ``"computes on CPU"`` marker makes the test red. It has to be
   recorded *per run*: ``backends.warn_degradation`` de-dupes per message
   through the module-level ``_warned`` set, so inspecting that set afterwards
   says "this process degraded once", never "this run degraded" -- and a second
   CPU-fallback run in a long-lived process never warns again.
2. **allocator-window comparison** -- ``torch.cuda.memory_allocated()`` is read
   *before* ``reset_peak_memory_stats()``, and the run must push
   ``max_memory_allocated()`` strictly above that inherited value. ``peak > 0``
   is not a guard: resetting the peak seeds it with the live usage an earlier
   arch still holds (this suite leaves ~76 MiB of it, measured), so ``peak > 0``
   asks only "has this process ever touched the device". Comparing against
   ``before`` asks "did *this* run allocate", which is the question. Both
   guards are needed: the warning one is de-duped away after the first hit, the
   allocator one is not.

What neither guard can see is *placement* -- ``runner.separate()`` returns only
``{stem: Path}``, so no consumer can assert ``SeperateAttributes.device``. That
is Task 11's harness fix, not a loosening of these.

THE GATE
--------
``SNR >= pins.json["gate"]["threshold_db"]`` (40 dB), read from the pin store so
the CPU suite's gate and this one cannot diverge. ``max_abs_dev`` is reported
next to it but is *not* a gate: one LSB of the PCM_16 container
(``3.0518e-05``) is the expected drift class, and the fixture mix is instrumental,
so the quiet vocal stems read low in dB for a byte-near-identical rerun -- see
``tests.golden.runner``'s docstring. Nothing here may widen the threshold: a
number below 40 dB is the finding.

MEASUREMENT TRAPS THIS FILE AVOIDS (both observed in task-5-report.md)
----------------------------------------------------------------------
1. ``torch.cuda.max_memory_allocated()`` is a process-wide high-water mark that
   is never reset; in Task 5 that put a 3577 MiB figure -- which was not any one
   arch's peak -- on the demucs row. The counter is reset immediately before each
   arch's run here, so a printed peak covers that arch's window. The leak in that
   reasoning is now part of the assertion instead of a comment: live device
   tensors an earlier arch still holds are counted *forward* into the next
   arch's window (~76 MiB from demucs into vr in suite order), so the guard
   compares the peak against that inherited ``before`` value and the print
   shows both. Numbers quoted *as* per-arch peaks are still re-measured alone
   (`-k <arch>`, fresh process) for task-7-report.md: 1081 MiB vr / 617 MiB
   demucs, versus 1157 MiB for vr in-suite.
2. MIOpen's cold kernel search makes the *first* separation after an install
   roughly 20x slower than every later one (36.84 s -> 1.65 s). Nothing in this
   file is a speed claim; Step 3's timings are warm and counted, and they live in
   the task report rather than in the suite (the plan's position: speed is
   measured, never asserted).

SCOPE: the archs in ``pins.json["archs_covered"]`` and nothing more -- two archs,
six stems (``demucs/{Bass,Drums,Other,Vocals}`` + ``vr/{Instrumental,Vocals}``),
and the stem set is asserted per arch rather than assumed, so a stem the engine
stops writing narrows the gate loudly instead of silently. mdx_c,
roformer and onnx are recorded in ``pins.json["archs_blocked"]`` with a reason
each (no weights on this machine; ``onnx`` runs on DirectML and is covered by
Task 6's provider assertion). Reaching them would mean simulating archs, which
Task 4 ruled out. This file only ever writes into the ``tmp_path`` it is handed:
no pin store, no ``baseline/``, no ``F:/.../models``.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pytest
import soundfile as sf
import torch

import backends
from tests.golden import runner, settings
from tests.golden.runner import separate
from tests.golden.snr import max_abs_dev, snr_db

#: Absolute instead of repo-relative (the brief sketch's ``Path("tests/golden/...")``
#: only resolves when pytest is invoked from the repo root). Same two files, and
#: ``settings.PINS_FILE`` is the harness's own name for the one pin store.
ART = settings.ARTIFACTS_DIR
PINS = settings.PINS_FILE

# Derived, never hardcoded. The pin store already knows which archs it could
# capture on this machine and which it could not; a hardcoded list here is how
# Task 7 came to name "mdx", which has no CPU baseline and would fail on
# `ref.exists()` while measuring nothing about ROCm.
APPLY = sorted(json.loads(PINS.read_text(encoding="utf-8"))["archs_covered"])

#: The engine's own words for "I did not get the device I was asked for"
#: (``separate.py`` -> ``backends.warn_degradation``). Matched on a fragment, not
#: the whole sentence, because the sentence names the backend and the torch
#: device -- neither is this file's text to pin.
DEGRADATION_MARK = "computes on CPU"

#: Substrings of the runtime notices that say *which* attention kernel served the
#: math (``torch.nn.MultiheadAttention`` dispatches through the SDPA backends).
ATTENTION_MARKS = ("Attention", "attention")


def _sdpa_config() -> str:
    """The scaled-dot-product-attention backend switches, as torch reports them.

    Recorded and printed, never asserted. Task 5 established that this wheel has
    no flash/cuDNN SDPA kernels, so the flag list alone does not say what ran;
    the runtime notice printed next to it (``_attention_notices``) is the
    observation of the backend that actually served the dB values in the table.
    Moving either into ``pins.json["identity"]`` is a pin change and is not this
    task's to make.
    """
    flags = torch.backends.cuda
    return " ".join(
        f"{name}={'on' if getattr(flags, name + '_sdp_enabled')() else 'off'}"
        for name in ("flash", "mem_efficient", "cudnn", "math"))


def _attention_notices(caught: list[warnings.WarningMessage]) -> list[str]:
    """Deduplicated attention-backend notices emitted *during this run*."""
    seen: list[str] = []
    for w in caught:
        text = str(w.message).splitlines()[0].strip()
        if any(mark in text for mark in ATTENTION_MARKS) and text not in seen:
            seen.append(text)
    return seen


@pytest.fixture(autouse=True)
def _rocm_is_in_the_room():
    """Refuse to run a *device* comparison when no device differs.

    ``separate(arch, "cuda")`` asks for the GPU; if torch or the backend says
    there is none, the engine quietly lands on CPU (``separate.py``'s
    ``warn_degradation`` path). Scoring that output against the pinned CPU
    baselines would give inf dB and a green suite proving nothing. This fixture
    is the *process-level* half of the answer -- it stops the file running when no
    device is in the room; the per-run guards in ``test_rocm_matches_cpu`` are the
    half that checks each run actually got there, because a healthy torch does not
    guarantee a healthy engine.

    Per the Task 7 ruling there is ONE pin store and the golden suite requires the
    ``bootstrap.ps1 -Backend rocm`` venv, so a cuda/cpu venv is meant to be red
    here, not skipped.
    """
    if not torch.cuda.is_available():
        pytest.fail("torch reports no GPU: separate(arch, 'cuda') would compute on "
                    "CPU, so this file would compare CPU against CPU and pass "
                    "without testing anything.")
    if getattr(torch.version, "hip", None) is None:
        pytest.fail(f"{torch.__version__} is not a ROCm build (torch.version.hip is "
                    "None): these pins were captured under the ROCm wheel this fork "
                    "ships, and the golden suite runs there (-Backend rocm).")


def test_apply_list_is_the_covered_set():
    """Fail loudly if the pin store and this test ever drift apart."""
    assert APPLY, "pins.json declares no covered archs"
    assert set(APPLY) <= {"vr", "mdx", "mdx_c", "demucs", "roformer", "onnx"}
    for arch in APPLY:
        for stem in json.loads(PINS.read_text(encoding="utf-8"))["pins"][arch]:
            assert (ART / "baseline" / f"{arch}_{stem}.wav").exists()


def test_the_gpu_presents_as_cuda_and_is_the_chosen_device():
    """The premise, stated as a fact about this machine rather than assumed.

    ``backends.resolve()`` is the single device/provider authority and
    ``separate.py`` reads it unconditionally, so "rocm -> the cuda torch device"
    is what makes `separate(arch, "cuda")` a GPU run at all. Task 6 pins the
    decision table with monkeypatched probes; this is the live-hardware half, and
    it is also the check that the reference and the candidate differ in exactly
    one variable (device).
    """
    assert getattr(torch.version, "hip", None), "not a ROCm build"
    assert torch.version.cuda is None, "hip *and* cuda: not the wheel these pins describe"
    assert torch.cuda.is_available() is True, "ROCm must present itself as CUDA"

    backend = backends.resolve()
    assert (backend.name, backend.torch_device) == ("rocm", "cuda"), backend

    # And the harness really does ask for the GPU: ``device="cuda"`` maps to the
    # GUI's GPU switch (is_gpu_conversion >= 0), ``"cpu"`` to the off switch, so
    # the two sides of every row below differ in the device and nothing else.
    for arch in APPLY:
        assert settings.make(arch, "cuda")["is_gpu_conversion"] >= 0, arch
        assert settings.make(arch, "cpu")["is_gpu_conversion"] < 0, arch


@pytest.mark.parametrize("arch", APPLY)
def test_rocm_matches_cpu(arch, tmp_path):
    # Guards, in the order the failure mode needs them. One window does both
    # measurements, and it is on purpose: torch's allocator counters are global,
    # so the reset and the peak read are only attributable to the run between
    # them -- and the warning recorder has to be open over that same run to
    # answer "did THIS run degrade" rather than "did this process ever warn".
    #
    # `before` is read *before* reset_peak_memory_stats(), because that call does
    # not zero the high-water mark -- it seeds it with the memory this process is
    # holding right now -- and empty_cache() only returns cached blocks, never
    # live tensors. In suite order demucs leaves ~76 MiB of device tensors alive
    # (measured), so an unconditional `peak > 0` would be answered by the
    # previous arch and stay true through a run that never touched the GPU.
    # Comparing the peak with `before` asks the question that matters: did *this*
    # run allocate.
    with warnings.catch_warnings(record=True) as caught:
        # Warnings are recorded inside the run, not read back afterwards:
        # backends.warn_degradation de-dupes per message through a module-level
        # set, so a post-hoc look at that set is stale the second time the engine
        # degrades in the same process, and `separate.py`'s import-time
        # filterwarnings("ignore") is exactly what warn_degradation exists to
        # punch through. simplefilter("always") does not clear that pre-existing
        # ignore-all entry, it puts "always" in front of it.
        before_bytes = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        warnings.simplefilter("always")
        rocm = separate(arch, "cuda", tmp_path)
        peak_bytes = torch.cuda.max_memory_allocated()

    degraded = [str(w.message) for w in caught if DEGRADATION_MARK in str(w.message)]
    assert not degraded, (
        f"{arch}/cuda never reached the device: the engine reported {degraded[0]!r}. "
        "Its CPU output scored against the CPU baselines would be inf dB and green "
        "while proving nothing about ROCm (see THE TWO GUARDS in this file's "
        "docstring)."
    )

    # peak_bytes was read inside the window above, not here: reading it after the
    # recorder closes would let anything allocated between the two count as this
    # run's GPU work.
    assert peak_bytes > before_bytes, (
        f"{arch}/cuda allocated no device memory beyond the "
        f"{before_bytes / 2**20:.2f} MiB this process already held when the peak was "
        f"reset (peak {peak_bytes / 2**20:.2f} MiB): this run did not reach the "
        "device, so scoring it against the CPU baselines would be vacuous."
    )

    pins = json.loads(PINS.read_text(encoding="utf-8"))
    # Coverage cannot narrow: the loop below walks the run's own stems, so a stem
    # the engine stops writing would simply drop out of the gate. The harness
    # calls this bucket `missing` (runner.compare_fingerprints); assert it here.
    pinned_stems = set(pins["pins"][arch])
    assert set(rocm) == pinned_stems, (
        f"{arch}/cuda wrote {sorted(rocm)} against the pinned {sorted(pinned_stems)}: "
        f"missing={sorted(pinned_stems - set(rocm))} "
        f"unpinned={sorted(set(rocm) - pinned_stems)}. The gate counts "
        f"{len(pinned_stems)} stems for this arch, and it is not allowed to be "
        "measured on fewer."
    )

    # Both numbers print, because only one of them is the arch's own: the peak is
    # inflated by whatever the earlier arch leaked into this window (vr reads
    # 1157 MiB in-suite against 1081 MiB isolated), so a reader copying the
    # printed figure has to be able to see the inheritance in the same line.
    print(f"\n[task7] {arch}/cuda peak GPU memory allocated: {peak_bytes / 2**20:.0f} "
          f"MiB (inherited before this run: {before_bytes / 2**20:.0f} MiB -- the "
          "per-arch peak is that peak *with* the inheritance, isolated numbers are "
          "in task-7-report.md; gate: SNR >= "
          f"{pins['gate']['threshold_db']}dB, {len(pinned_stems)} stems)")
    # Recorded, never pinned: which attention path produced these numbers. The
    # flags are torch's switches; the notices are what the run actually said.
    print(f"[task7] {arch}/cuda attention: sdpa[{_sdpa_config()}] "
          f"hip={getattr(torch.version, 'hip', None)} "
          f"gpu={torch.cuda.get_device_name(0)}")
    for notice in _attention_notices(caught):
        print(f"[task7] {arch}/cuda attention notice: {notice}")

    for stem, path in rocm.items():
        ref = ART / "baseline" / f"{arch}_{stem}.wav"
        assert ref.exists(), f"no CPU baseline for {arch}/{stem}; rerun Task 4"
        a, _ = sf.read(str(ref))
        b, _ = sf.read(str(path))

        ref_frames = sf.info(str(ref)).frames
        cand_frames = sf.info(str(path)).frames
        assert ref_frames == cand_frames, (
            f"{arch}/{stem}: the GPU run wrote {cand_frames} frames against the "
            f"pinned CPU baseline's {ref_frames}; SNR truncates to the shorter "
            "signal, so a length change must be reported, not averaged away."
        )

        s = snr_db(a, b)
        dev = max_abs_dev(a, b)
        flat_a = np.asarray(a, dtype=np.float64).ravel()
        flat_b = np.asarray(b, dtype=np.float64).ravel()
        differing = int(np.count_nonzero(flat_a != flat_b))
        # Reported, not gated: one PCM_16 LSB (3.0518e-05) is the drift class the
        # pin store itself names, so "at one LSB" is the sentence a reader of the
        # headline claim needs beside the dB number.
        print(f"[task7] {arch}/{stem} SNR={s:.2f}dB maxdev={dev:.3e} "
              f"at_one_lsb={dev <= runner.PCM_16_LSB * 1.000001} "
              f"differing={differing}/{flat_a.size} "
              f"({differing / max(flat_a.size, 1):.2%}) frames={cand_frames}")
        assert s >= pins["gate"]["threshold_db"], (
            f"{arch}/{stem} SNR={s:.1f}dB maxdev={max_abs_dev(a, b):.4f}")
