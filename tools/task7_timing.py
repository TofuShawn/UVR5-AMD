"""Task 7 Step 3 measurement driver: warm CPU-vs-ROCm timings on the golden crop.

WHAT THIS FILE PROVES
---------------------
It is the only reproducible source for the fork's headline speed claim. The
numbers in ``task-7-report.md`` section 3 (vr 5.90x / demucs 4.05x against a
healthy CPU build, and 4.71x / 20.62x against this wheel's own CPU path) came
from running this script; nobody can re-derive them from the repository without
it. It asserts nothing about speed -- the plan's position is that speed is
measured, never asserted -- and it writes only to a temp dir.

    UVR_MODEL_ROOT="F:/Software/Ultimate Vocal Remover/models" \
      .venv/Scripts/python.exe tools/task7_timing.py

Deliberately NOT named ``test_*.py``: this is a driver, and the suite must not
start benchmarking every run.

Traps it is built around (both from task-5-report.md):
  * ``torch.cuda.max_memory_allocated()`` is a process-wide high-water mark that
    is never reset on its own -- how 3577 MiB of VR's ended up printed on the
    demucs row. The counter is reset before EVERY segment (CPU runs included),
    and each GPU segment must beat the memory already live when its window
    opened, which is also what stops "peak > 0" being mistaken for "this run
    reached the device".
  * MIOpen's cold kernel search costs roughly 20x on the first separation after
    an install (36.84 s -> 1.65 s in task-5-report.md). Per process on a box
    whose find-db is already warm the cost is smaller but real (~2-3x), so the
    first GPU run of each arch is reported separately and excluded from "warm".
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

#: PROVENANCE OF THE HEADLINE DIVISOR. These are *not* measurements this script
#: can take: they were captured on a different torch build (2.15.0+cpu) than the
#: one in this venv, so the only honest form is a quoted constant with its source.
#: task-5-report.md section 5.5's reference table ("VR ... 9.845 s -> 6.0x",
#: "Demucs ... 11.108 s -> 4.0x"), restated in section 6.2; those CPU times were
#: themselves measured in Task 4 under torch 2.15.0+cpu on this same 30-second
#: crop. Important 5 of the Task 7 review is exactly this: a divisor the script
#: cannot re-measure has to name where it came from, in the output as well as in
#: the source.
#:
#: Why both columns exist: this ROCm wheel's own CPU Demucs path runs ~5x slower
#: than torch 2.15's does (51-62 s here, 54.745 s in task-5-report.md section 6.2, vs
#: 11.108 s healthy), so dividing a GPU time by *it* inflates the ratio to ~18-20x.
#: That number is a measurement of this wheel's CPU regression, not a GPU claim;
#: the healthy column is the publishable one. If this wheel's CPU path ever gets
#: fixed, the same-wheel column moves and this quoted one does not.
HEALTHY_CPU_SECONDS = {
    "vr": 9.845,
    "demucs": 11.108,
}
HEALTHY_CPU_PROVENANCE = (
    "torch 2.15.0+cpu, same 30 s crop; task-5-report.md section 5.5 table, restated in "
    "section 6.2, measured there by Task 4 -- a quoted constant this venv cannot "
    "re-measure (it holds only 2.9.1+rocm7.2.1)")

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
              f"healthy_cpu={healthy}s cold_gpu={results[(arch, 'cuda')][0]:.2f}s")

    print(f"\n[t7] healthy-CPU divisor provenance: {HEALTHY_CPU_PROVENANCE} "
          f"{HEALTHY_CPU_SECONDS}")
    print("[t7] which column is the claim: speedup_vs_healthy_cpu. The "
          "same-wheel column divides by this wheel's own degraded CPU path, so "
          "its demucs figure is a CPU-regression artifact, not a GPU result.")
    print("\n[t7] all runs per arch/device (seconds):")
    for key, values in results.items():
        print(f"     {key[0]}/{key[1]}: n={len(values)} "
              f"min={min(values):.2f} median={statistics.median(values):.2f} "
              f"all={[round(v, 2) for v in values]}")


if __name__ == "__main__":
    main()
