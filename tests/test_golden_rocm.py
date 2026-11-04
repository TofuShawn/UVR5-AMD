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
moves is the device. ``settings.make(arch, "cuda")`` sets
``is_gpu_conversion = 0``, which is the GUI's GPU checkbox; the engine then takes
its device from ``backends.resolve()`` + ``torch.cuda``, so a green row here is a
real GPU run, not a relabelled CPU one.

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
   arch's run here, so a printed peak covers that arch's window. The one leak in
   that reasoning is documented at the call site: device tensors an earlier arch
   still holds are counted *forward* into the next arch's number, so both archs
   were re-measured alone (`-k <arch>`, fresh process) for task-7-report.md.
2. MIOpen's cold kernel search makes the *first* separation after an install
   roughly 20x slower than every later one (36.84 s -> 1.65 s). Nothing in this
   file is a speed claim; Step 3's timings are warm and counted, and they live in
   the task report rather than in the suite (the plan's position: speed is
   measured, never asserted).

SCOPE: the archs in ``pins.json["archs_covered"]`` and nothing more. mdx_c,
roformer and onnx are recorded in ``pins.json["archs_blocked"]`` with a reason
each (no weights on this machine; ``onnx`` runs on DirectML and is covered by
Task 6's provider assertion). Reaching them would mean simulating archs, which
Task 4 ruled out. This file only ever writes into the ``tmp_path`` it is handed:
no pin store, no ``baseline/``, no ``F:/.../models``.
"""
from __future__ import annotations

import json

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


@pytest.fixture(autouse=True)
def _rocm_is_in_the_room():
    """Refuse to run a *device* comparison when no device differs.

    ``separate(arch, "cuda")`` asks for the GPU; if torch or the backend says
    there is none, the engine quietly lands on CPU (``separate.py``'s
    ``warn_degradation`` path). Scoring that output against the pinned CPU
    baselines would give inf dB and a green suite proving nothing. Failing loudly
    here is what keeps a green row in this file worth reading.

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
    # Trap 1: the peak-memory counter is process-wide and never reset on its own,
    # so zero it immediately before this arch's only GPU run and the printed peak
    # below is this arch's. Caveat kept honest: a tensor an earlier run leaked and
    # still held would be counted inside the *later* arch's window, so a peak that
    # looks inherited is re-measured with `-k <arch>` in a fresh process (both
    # archs were checked that way for task-7-report.md).
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    rocm = separate(arch, "cuda", tmp_path)
    peak_mib = torch.cuda.max_memory_allocated() / 2**20
    assert peak_mib > 0, (
        f"{arch}/cuda allocated no GPU memory at all: the run did not reach the "
        "device, so scoring it against the CPU baseline would be vacuous."
    )

    pins = json.loads(PINS.read_text(encoding="utf-8"))
    print(f"\n[task7] {arch}/cuda peak GPU memory allocated: {peak_mib:.0f} MiB "
          f"(gate: SNR >= {pins['gate']['threshold_db']}dB)")

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
