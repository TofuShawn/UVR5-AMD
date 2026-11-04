"""Task 7 Step 3 measurement driver: warm CPU-vs-ROCm timings on the golden crop.

WHAT THIS FILE PROVES -- AND WHAT IT ONLY QUOTES
------------------------------------------------
Two different kinds of claim come out of this driver, and they have to be
published differently.

REPRODUCIBLE IN-TREE (run this file and you get these columns again): the GPU
warm time, this wheel's own CPU time, and therefore the whole
``speedup_vs_same_wheel_cpu`` column. Re-deriving it needs this file plus the
hardware it was measured on -- this AMD GPU, a ``bootstrap.ps1 -Backend rocm``
venv and the read-only model root -- and nothing that lives outside the
repository. It re-derives, but not to the digit -- a ~10 % session-to-session
spread is normal: the Step 3 run printed vr 4.71x / demucs 20.62x, the two
fix-round-3 re-runs printed 4.30x / 18.53x and 4.42x / 18.02x (warm GPU 1.88 s
then 1.83 s for vr, against Step 3's 1.67 s; demucs 2.74 s both times), because
both sides wander. That is why every individual run is printed.

EXTERNALLY MEASURED, NOT REPRODUCIBLE FROM THIS REPOSITORY: the headline
``speedup_vs_healthy_cpu`` column (the published vr 5.90x / demucs 4.05x). Its
two sides have different status. The GPU warm time is measured by the runs
below -- 5.90x is 9.845 s divided by Step 3's 1.67 s vr warm time, and the same
baseline against the fix-round-3 re-runs' 1.88 s and 1.83 s gives 5.24x and
5.38x. The CPU side of that ratio is ``HEALTHY_CPU_SECONDS`` below: a constant
taken elsewhere, on a different torch build, in a run no file in this repository
records or can re-measure. So 5.90x is a measured GPU time against a quoted
historical CPU time, which makes the ratio a historical figure of a particular
session rather than a constant of this box. Whoever copies it into a README
copies it with that sentence attached; read the block above the constant before
publishing anything. It asserts nothing about speed -- the plan's position is
that speed is measured, never asserted -- and it writes only to a temp dir.

    UVR_MODEL_ROOT="F:/Software/Ultimate Vocal Remover/models" \
      .venv/Scripts/python.exe tools/task7_timing.py

Deliberately NOT named ``test_*.py``: this is a driver, and the suite must not
start benchmarking every run.

Traps it is built around. Both were first observed on this box during Task 5,
whose report for them is plan scratch inside the git-ignored ``.superpowers/``
tree: not in this repository, and gone once the plan completes. They are restated
here in full so nothing below cites a file a reader cannot open.
  * ``torch.cuda.max_memory_allocated()`` is a process-wide high-water mark that
    is never reset on its own -- how 3577 MiB of VR's ended up printed on the
    demucs row. The counter is reset before EVERY segment (CPU runs included),
    and each GPU segment must beat the memory already live when its window
    opened, which is also what stops "peak > 0" being mistaken for "this run
    reached the device".
  * MIOpen's cold kernel search costs roughly 20x on the first separation after
    an install (36.84 s -> 1.65 s, measured that way on this box). Per process on
    a box whose find-db is already warm the cost is smaller but real (~2-3x), so
    the first GPU run of each arch is reported separately and excluded from
    "warm".
"""
from __future__ import annotations

import json
import shutil
import statistics
import sys
import tempfile
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

import backends  # noqa: E402
from tests.golden import runner, settings  # noqa: E402

APPLY = sorted(json.loads(settings.PINS_FILE.read_text(encoding="utf-8"))["archs_covered"])
CPU_RUNS = 2
GPU_RUNS = 4

#: THE HEADLINE CPU BASELINE: EXTERNALLY MEASURED, NOT REPRODUCIBLE FROM THIS REPO.
#: These are *not* measurements this script can take. Task 4 captured them on
#: this machine under torch 2.15.0+cpu -- a different build from the
#: 2.9.1+rocm7.2.1 this venv holds -- on the same 30 s golden crop, and no
#: committed artifact records that capture: ``pins.json``
#: ``provenance.elapsed_seconds`` (8.458 vr / 54.302 demucs) are CPU times of
#: THIS wheel with the device forced to cpu, i.e. the same-wheel column, not this
#: one. What survives in the tree is the two digits below and this paragraph; the
#: measurement event itself is external history.
#:
#: Their source, task-5-report.md section 5.5's reference table ("VR ... 9.845 s
#: -> 6.0x", "Demucs ... 11.108 s -> 4.0x"), restated in section 6.2, is a
#: document NOT in this repository: plan scratch under ``.superpowers/`` (see
#: ``.gitignore``), never committed, and deleted when this plan completes. It is
#: named below as a credit, not as something a reader can open. (That table's
#: 6.0x / 4.0x divided the same CPU baseline by Task 5's GPU times; this script's
#: 5.90x / 4.05x divide it by the warm GPU times it measured -- the CPU baseline
#: is fixed, the GPU time is what moves.) Important 5 of the Task 7 review asked
#: for a provenance line because a baseline the script cannot re-measure has to
#: say where it came from; that ruling also carried "the evidence must outlive
#: the workspace", and a citation into scratch does not -- so this paragraph, not
#: the report, is the durable record of the claim.
#:
#: Route by which a reader could measure a healthy-CPU baseline today (a NEW
#: number, not these digits): bootstrap a second checkout with
#: ``bootstrap.ps1 -Backend cpu`` -- plain PyPI torch, the 2.15.0+cpu family these
#: came from per that script's own index note -- and time one
#: ``tests.golden.runner.separate(arch, "cpu", work_dir)`` per arch on the crop.
#: Not with this script: ``main()`` reads ``torch.cuda.get_device_name(0)`` and
#: every cuda segment asserts the device was reached, so it dies on a CPU-only
#: wheel (and the golden suite fails its identity pin there by the Task 7
#: ruling). Expect the re-measured digits to differ by ~10 %: CPU separations
#: wander with desktop load, which is why this driver prints every individual run
#: ("all runs per arch/device") instead of one average.
#:
#: Why both columns exist: this ROCm wheel's own CPU Demucs path runs ~5x slower
#: than torch 2.15's does (51-62 s here, 54.745 s for the same wheel in Task 5's
#: scratch log, vs 11.108 s healthy), and every ratio here is CPU seconds divided
#: by GPU seconds, so building one on *this* wheel's CPU time inflates it to
#: ~18-20x. That number is a measurement of this wheel's CPU regression, not a
#: GPU claim; the healthy column is the publishable one, the same-wheel column is
#: the one this script fully re-derives. If this wheel's CPU path ever gets
#: fixed, the same-wheel column moves and this quoted one does not.
HEALTHY_CPU_SECONDS = {
    "vr": 9.845,
    "demucs": 11.108,
}
HEALTHY_CPU_PROVENANCE = (
    "EXTERNAL / HISTORICAL, not reproducible from this repository: torch "
    "2.15.0+cpu, same 30 s crop, measured there by Task 4. Its write-up "
    "(task-5-report.md section 5.5, restated in section 6.2) is plan scratch in "
    "the git-ignored .superpowers/ tree -- not in the repo, gone after the plan. "
    "This venv holds only 2.9.1+rocm7.2.1 and cannot re-measure this baseline; "
    "bootstrap.ps1 -Backend cpu can measure a NEW one (see this file's "
    "HEALTHY_CPU_SECONDS block). In speedup_vs_healthy_cpu only the GPU seconds "
    "come from this run; these CPU seconds are quoted.")

#: ``separate.py``'s own words for "the GPU was asked for and not used".
DEGRADATION_MARK = "computes on CPU"


def one(arch: str, device: str) -> tuple[float, float]:
    """Run one separation; return (seconds, peak device MiB for its own window)."""
    work_dir = Path(tempfile.mkdtemp(prefix=f"uvr_t7_{arch}_{device}_"))
    try:
        # Reset before EVERY segment, not just the GPU ones: that is what makes a
        # CPU row read 0 MiB by construction instead of inheriting the previous
        # arch's high-water mark (the first draft of this script printed demucs'
        # 618 MiB on vr's CPU rows -- trap 1, live, in the tool built to avoid it).
        before = torch.cuda.memory_allocated()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            stems = runner.separate(arch, device, work_dir)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        peak_mib = torch.cuda.max_memory_allocated() / 2**20
        assert len(stems) >= 2, f"{arch}/{device} wrote {sorted(stems)}"
        if device == "cuda":
            degraded = [w for w in caught if DEGRADATION_MARK in str(w.message)]
            assert not degraded, f"{arch}/cuda degraded to CPU: {degraded[0]}"
            assert peak_mib > before / 2**20, (
                f"{arch}/cuda allocated nothing beyond the {before / 2**20:.0f} MiB "
                "already live when its window opened: that run never reached the "
                "device and its timing is a CPU timing.")
        return elapsed, peak_mib
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def main() -> None:
    hip = getattr(torch.version, "hip", None)
    print(f"torch={torch.__version__} hip={hip} cuda_available={torch.cuda.is_available()}")
    print(f"device={torch.cuda.get_device_name(0)} backend={backends.resolve()}")
    print(f"cpu_threads={torch.get_num_threads()} APPLY={APPLY} "
          f"cpu_runs={CPU_RUNS} gpu_runs={GPU_RUNS}\n")

    results: dict[tuple[str, str], list[float]] = {}
    for arch in APPLY:
        for device, count in (("cpu", CPU_RUNS), ("cuda", GPU_RUNS)):
            for run in range(1, count + 1):
                elapsed, peak = one(arch, device)
                results.setdefault((arch, device), []).append(elapsed)
                print(f"[t7] arch={arch:7s} device={device:4s} run={run} "
                      f"elapsed={elapsed:6.2f}s peak_gpu={peak:7.0f} MiB", flush=True)
        print("", flush=True)

    same_wheel_cpu: dict[str, float] = {}
    warm_gpu: dict[str, float] = {}
    for arch in APPLY:
        same_wheel_cpu[arch] = min(results[(arch, "cpu")])
        warm_gpu[arch] = min(results[(arch, "cuda")][1:])   # drop first GPU run
        healthy = HEALTHY_CPU_SECONDS.get(arch)
        ratio_healthy = (f"{healthy / warm_gpu[arch]:5.2f}x"
                         if healthy else "   n/a (no quoted baseline for this arch)")
        print(f"[t7] arch={arch:7s} cpu_same_wheel={same_wheel_cpu[arch]:6.2f}s "
              f"gpu_warm={warm_gpu[arch]:6.2f}s "
              f"speedup_vs_same_wheel_cpu={same_wheel_cpu[arch] / warm_gpu[arch]:5.2f}x "
              f"speedup_vs_healthy_cpu={ratio_healthy} "
              f"healthy_cpu={healthy}s (external, quoted -- not measured here) "
              f"cold_gpu={results[(arch, 'cuda')][0]:.2f}s")

    print(f"\n[t7] healthy-CPU baseline provenance: {HEALTHY_CPU_PROVENANCE} "
          f"{HEALTHY_CPU_SECONDS}")
    print("[t7] which column is the published claim: speedup_vs_healthy_cpu -- and "
          "only one side of it was measured by this run. The GPU seconds on the "
          "bottom of that fraction came from above; the healthy-CPU seconds on top "
          "are the quoted external constant, so 5.90x/4.05x are historical figures "
          "restated here, not numbers this repository can re-derive. "
          "speedup_vs_same_wheel_cpu IS fully re-derived by every run above, but it "
          "divides by this wheel's own degraded CPU path, so its demucs figure is a "
          "CPU-regression artifact, not a GPU result.")
    print("\n[t7] all runs per arch/device (seconds):")
    for key, values in results.items():
        print(f"     {key[0]}/{key[1]}: n={len(values)} "
              f"min={min(values):.2f} median={statistics.median(values):.2f} "
              f"all={[round(v, 2) for v in values]}")


if __name__ == "__main__":
    main()
