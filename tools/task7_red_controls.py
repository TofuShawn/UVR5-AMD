"""Task 7 RED controls: proof that the guards in ``tests/test_golden_rocm.py`` fire.

WHAT THIS FILE PROVES
---------------------
A measurement test is worth nothing unless the same code path goes red on the
failure it claims to catch, and Task 7's review found exactly that: the peak
guard ``assert peak_mib > 0`` had already been demonstrated red **only in an
empty process** -- the one condition where it was never broken. These controls
are the committed provenance for the answer. Each one sabotages one input to one
guard and asserts the guard fires:

  * a control that PASSES means the guard still catches its failure mode;
  * a control that goes RED means a guard has become decorative -- which is the
    finding, and it is why the inversion from the scratch version ("everything
    here must fail") to this one ("everything here must catch") is the point.

Nothing here touches ``pins.json``, ``_artifacts/baseline/`` or the 40 dB
threshold; the wrong references are made of already-pinned audio and of copies
inside ``tmp_path``. Several controls *are* destructive -- they patch the
engine's own device decision and force whole CPU separations -- so the file is
named ``task7_red_controls.py``, not ``test_*.py``: a bare ``pytest`` run must
never execute it. Run it explicitly:

    UVR_MODEL_ROOT="F:/Software/Ultimate Vocal Remover/models" \
      .venv/Scripts/python.exe -m pytest tools/task7_red_controls.py -v -s

Two mechanics the controls depend on, both of which the review named:
``backends.warn_degradation`` de-dupes per message through a module-level set,
so a post-hoc look at that set is not a per-run answer (controls 6 and 7); and
``separate.py`` samples ``cuda_available`` at import time, so forcing the
engine's CPU branch means patching the place it reads (control 5), not
``torch.cuda.is_available``.
"""
from __future__ import annotations

import contextlib
import json
import re
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import soundfile as sf
import torch

import backends
import separate as engine                       # the engine module, not the runner
import tests.test_golden_rocm as t7
from tests.golden import runner, settings
from tests.golden.snr import max_abs_dev, snr_db

PINS = json.loads(settings.PINS_FILE.read_text(encoding="utf-8"))
THRESHOLD = PINS["gate"]["threshold_db"]

# Which guard the committed test is expected to name in its AssertionError. Both
# of these are the Critical's closure; matching on them is what keeps a control
# from going green for an unrelated reason.
REACHED_DEVICE = re.compile(
    r"never reached the device|allocated no device memory beyond")


def _baseline(arch: str, stem: str):
    data, _ = sf.read(str(settings.BASELINE_DIR / f"{arch}_{stem}.wav"))
    return data


def _residue_after_gpu_demucs(tmp_path: Path) -> int:
    """Run a real GPU demucs separation and return the live device bytes it left.

    This is the premise of the inheritance case: in suite order demucs runs
    before vr, and ~76 MiB of its allocations are still live when vr's window
    opens. If this ever returns 0 the controls below that need inheritance say
    so instead of quietly degenerating into the empty-process case.
    """
    before = torch.cuda.memory_allocated()
    stems = runner.separate("demucs", "cuda", tmp_path / "gpu_demucs")
    assert len(stems) == len(PINS["pins"]["demucs"]), sorted(stems)
    residue = torch.cuda.memory_allocated() - before
    print(f"[red] live device bytes after a real GPU demucs run: "
          f"{residue / 2**20:.2f} MiB", flush=True)
    assert residue > 0, (
        "the GPU demucs run left no live device allocations, so this process is "
        "the artificially clean case the reviewer rejected -- the inheritance "
        "controls below would not be testing inheritance.")
    return residue


@contextlib.contextmanager
def engine_blind_to_the_gpu():
    """Take the engine's real CPU-fallback branch, at the place it reads.

    ``separate.py`` does ``cuda_available = torch.cuda.is_available()`` at import
    time and ``SeperateAttributes.__init__`` branches on that *module global*, so
    patching ``torch.cuda.is_available`` afterwards leaves the engine alone and
    only drives the control's own asserts. Patching the module attribute is what
    reaches ``warn_degradation`` and the ``self.device = cpu`` that survives.
    """
    saved = engine.cuda_available
    engine.cuda_available = False
    try:
        yield
    finally:
        engine.cuda_available = saved


# --------------------------------------------------------------------------- #
# 1-3: the gate and the two premise guards
# --------------------------------------------------------------------------- #

def test_red_gate_fires_when_the_audio_really_is_different():
    """The gate assertion is not decorative: real audio from another network fails it.

    ``vr_Vocals`` and ``demucs_Vocals`` are both correct separations of the same
    crop by different networks. If the 40 dB gate let that through it would also
    let a GPU run pass while computing something else entirely.
    """
    a = _baseline("vr", "Vocals")
    b = _baseline("demucs", "Vocals")
    s = snr_db(a, b)
    assert s < THRESHOLD, (
        f"gate does not discriminate: vr/Vocals vs demucs_Vocals scored "
        f"SNR={s:.1f}dB maxdev={max_abs_dev(a, b):.4f} against a "
        f"{THRESHOLD}dB gate")


def test_red_device_guard_fires_when_torch_has_no_gpu(monkeypatch):
    """If the GPU disappears, the premise test must be red, not vacuously green."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(AssertionError, match="must present itself as CUDA"):
        t7.test_the_gpu_presents_as_cuda_and_is_the_chosen_device()


def test_red_device_guard_fires_when_the_authority_degrades_to_cpu(monkeypatch):
    """Task 6 wiring gone wrong: resolve() handing back a cpu torch device."""
    monkeypatch.setattr(
        backends, "resolve",
        lambda force=None: backends.Backend(name="cpu", torch_device="cpu",
                                            ort_providers=["CPUExecutionProvider"]))
    with pytest.raises(AssertionError, match="'rocm'"):
        t7.test_the_gpu_presents_as_cuda_and_is_the_chosen_device()


# --------------------------------------------------------------------------- #
# 4-5: the allocator guard -- empty process, then with real inheritance
# --------------------------------------------------------------------------- #

def test_red_peak_guard_fires_on_a_run_that_never_touched_the_device(tmp_path,
                                                                    monkeypatch):
    """Empty-process case: files appear, no device allocation at all.

    ``separate(arch, "cuda")`` stubbed to hand back the pinned WAVs themselves.
    This is the control as Task 7 first ran it -- it demonstrates the guard in a
    process that has done no GPU work, which is the condition where the old
    ``peak > 0`` happened to work. Control 5 is the one that matters.
    """
    assert torch.cuda.memory_allocated() == 0, (
        "expected a clean process for the empty-process control")
    staged = {stem: settings.BASELINE_DIR / f"vr_{stem}.wav"
              for stem in PINS["pins"]["vr"]}
    monkeypatch.setattr(t7, "separate", lambda arch, device, work_dir: staged)
    with pytest.raises(AssertionError) as empty_case:
        t7.test_rocm_matches_cpu("vr", tmp_path)
    assert REACHED_DEVICE.search(str(empty_case.value)), str(empty_case.value)
    # What this control is NOT: evidence about the second arch in a process. Here
    # `before` is 0, so the old `peak > 0` and the new `peak > before` agree and
    # the vacuous pass it closes was never reachable in this state. The case
    # where the two assertions part company is the next control.
    assert torch.cuda.memory_allocated() == 0, (
        "this control only means something in a process with no live device "
        "memory; if the premise moved, its verdict moved with it")


def test_red_cpu_fallback_is_caught_after_a_real_gpu_run_left_memory(tmp_path):
    """THE inheritance control: old guard passes, new guard fails, engine really degraded.

    Sequence, in one process, in suite order:

      1. a real GPU demucs separation leaves ~76 MiB of device tensors alive --
         the inheritance every in-suite vr window starts on, because APPLY is
         sorted and demucs comes first;
      2. ``separate.cuda_available`` is patched False (the place the engine
         actually reads) and vr is asked for ``device="cuda"``. The engine takes
         its own ``warn_degradation`` branch, chooses ``cpu``, and its output is
         byte-identical to the pinned CPU baselines;
      3. that window measured by hand: ``peak > 0`` -- the old assertion -- is
         TRUE on the inherited bytes while ``peak > before`` is False, and the
         per-run recorder holds the engine's degradation warning. The old guard
         waved the fallback through; the new one cannot;
      4. every *other* assertion in the committed file passes on that same
         output (stem set, frame counts, gate at ``inf`` dB), which is the whole
         vacuous path: CPU scored against CPU, green, zero GPU work;
      5. the committed test is then run on the same sabotage in the same process
         and goes red. Its own warning window is empty by now
         (``backends._warned`` de-duped the message), so the answer comes from
         the allocator guard -- the reason the file needs both guards and not
         just the polite one.
    """
    residue = _residue_after_gpu_demucs(tmp_path)
    print(f"[red] vr's window opens on {torch.cuda.memory_allocated() / 2**20:.2f} "
          f"MiB of inherited live device memory ({residue / 2**20:.2f} MiB of it is "
          "what the demucs run left); APPLY is sorted, so demucs is always first",
          flush=True)

    with engine_blind_to_the_gpu():
        # (3) the hand-measured window, nested exactly like the committed test:
        # before / reset / run / peak all inside one recorder, so the numbers
        # name this window and nothing else.
        chosen: dict[str, str] = {}
        vr_cls = runner.ARCH_CLASS["vr"]
        real_init = vr_cls.__init__

        def spy(self, *a, **k):
            real_init(self, *a, **k)
            chosen["device"] = str(self.device)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            manual_before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            vr_cls.__init__ = spy
            try:
                stems = runner.separate("vr", "cuda", tmp_path / "manual")
            finally:
                vr_cls.__init__ = real_init
            peak = torch.cuda.max_memory_allocated()
        degraded = [str(w.message) for w in caught
                    if t7.DEGRADATION_MARK in str(w.message)]

    assert manual_before > 0, (
        "no live device memory was inherited into this window, so it is the "
        "artificially clean case `peak > 0` could already see -- this control "
        "would no longer be demonstrating inheritance")
    old_guard = peak > 0
    new_guard = peak > manual_before
    print(f"[red] engine device for a run that asked for \"cuda\": "
          f"{chosen.get('device')!r} (the branch at separate.py:274-276 took the "
          "warn-and-keep-CPU path)", flush=True)
    assert chosen.get("device") == "cpu", (
        f"the sabotage did not reach the engine: it still chose "
        f"{chosen.get('device')!r}, so this window is not the CPU-fallback case")
    print(f"[red] manual window: before={manual_before / 2**20:.2f} MiB "
          f"peak={peak / 2**20:.2f} MiB | OLD `peak > 0` -> {old_guard} | "
          f"NEW `peak > before` -> {new_guard} | degradation warnings recorded "
          f"inside this run: {len(degraded)}", flush=True)

    assert degraded, (
        "the engine did not warn that it degraded, so the in-run recorder in "
        "tests/test_golden_rocm.py has nothing to catch -- either the message "
        "changed or the branch did")
    assert old_guard, (
        "the old assertion is no longer vacuous under inheritance, so this "
        "control has stopped proving the Critical existed")
    assert not new_guard, (
        "the new guard failed to fire on a run that allocated nothing beyond the "
        "inherited live memory -- the vacuous pass is back")

    # (4) every other assertion in the committed file passes on this output: same
    # stem set, same frame counts, gate satisfied. This is the whole vacuous path
    # -- CPU scored against CPU, inf dB, green, zero GPU work.
    assert set(stems) == set(PINS["pins"]["vr"]), sorted(stems)
    for stem, path in sorted(stems.items()):
        ref = settings.BASELINE_DIR / f"vr_{stem}.wav"
        a, _ = sf.read(str(ref))
        b, _ = sf.read(str(path))
        s = snr_db(a, b)
        frames_equal = sf.info(str(ref)).frames == sf.info(str(path)).frames
        print(f"[red] vr/{stem}: SNR={s:.2f}dB maxdev={max_abs_dev(a, b):.3e} "
              f"frames_equal={frames_equal} -> old-suite verdict: "
              f"{'PASS' if (s >= THRESHOLD and frames_equal) else 'FAIL'}", flush=True)
        assert s >= THRESHOLD and frames_equal, (
            "the vacuous path needs the gate and the frame count to pass too; if "
            "they stopped passing this control no longer demonstrates "
            "green-with-no-GPU-work")

    # (5) the committed test, on the same sabotage, in the same process. By now
    # backends._warned already holds the engine's message, so the recorder inside
    # the committed run catches nothing and only the allocator guard can answer.
    # If it does answer here, the file is red even when the polite channel is
    # exhausted -- which is why the review asked for both guards, not one.
    with engine_blind_to_the_gpu():
        with pytest.raises(AssertionError) as committed:
            t7.test_rocm_matches_cpu("vr", tmp_path / "committed")
        message = str(committed.value)
        print(f"[red] committed test said: {message}", flush=True)
        assert "allocated no device memory beyond" in message, (
            f"expected the allocator guard to answer the de-duped second fallback, "
            f"got: {message}")


# --------------------------------------------------------------------------- #
# 6: the warning guard, measured on its own
# --------------------------------------------------------------------------- #

def test_red_warning_guard_fires_on_a_run_that_really_did_allocate(tmp_path,
                                                                   monkeypatch):
    """The other guard alone: a run that touched the device *and* degraded.

    The allocator guard is satisfied here (a real device tensor is allocated
    inside the window), so the only thing that can catch this is the warning
    recorded inside the run. That is what makes the warning guard non-vacuous
    rather than a comment next to the peak check. A unique message sidesteps
    ``backends._warned`` without pretending the set is per-run.
    """
    staged = {stem: settings.BASELINE_DIR / f"vr_{stem}.wav"
              for stem in PINS["pins"]["vr"]}
    sentinel = (f"control stub: this run computes on CPU while claiming a GPU "
                f"device (unique id={id(staged)})")

    def degrading_run(arch, device, work_dir):
        alive = torch.zeros(4096, device="cuda")   # 16 KiB: the window now has a peak
        backends.warn_degradation(sentinel)
        return dict(staged)

    monkeypatch.setattr(t7, "separate", degrading_run)
    try:
        with pytest.raises(AssertionError) as caught_case:
            t7.test_rocm_matches_cpu("vr", tmp_path)
        message = str(caught_case.value)
        print(f"[red] committed test said: {message}", flush=True)
        assert "never reached the device" in message, (
            f"expected the in-run warning guard to answer, got: {message}")
    finally:
        backends._warned.discard(sentinel)


# --------------------------------------------------------------------------- #
# 7: the de-dupe mechanism the review warned about
# --------------------------------------------------------------------------- #

def test_red_warning_channel_is_per_run_not_per_process():
    """Why the committed guard records warnings inside each run.

    ``backends._warned`` is a module-level set keyed by message, so a run that
    degrades a second time in the same process emits *nothing*. A sentinel
    message shows the two facts separately: the first window records it, the
    second records nothing, and the set says "warned" through both.
    """
    sentinel = ("control sentinel: this run computes on CPU while claiming a GPU "
                f"device (unique per run, id={id(locals())})")
    try:
        with warnings.catch_warnings(record=True) as first_run:
            warnings.simplefilter("always")
            backends.warn_degradation(sentinel)
        with warnings.catch_warnings(record=True) as second_run:
            warnings.simplefilter("always")
            backends.warn_degradation(sentinel)
        assert len(first_run) == 1, (
            f"a per-run recorder does not see the engine's degradation warning "
            f"({len(first_run)} recorded), so the committed guard that counts on "
            "it would never fire")
        assert len(second_run) == 0, (
            "warn_degradation stopped de-duping per message; the comment in "
            "tests/test_golden_rocm.py needs revisiting")
        assert sentinel in backends._warned, (
            "the post-hoc set is not the per-run answer this control says it is")
        print("[red] first run recorded 1 warning, second recorded 0, "
              "backends._warned says 'warned' through both -> post-hoc set "
              "inspection cannot answer 'did THIS run degrade'", flush=True)
    finally:
        backends._warned.discard(sentinel)


# --------------------------------------------------------------------------- #
# 8-9: coverage narrowing and the frame guard
# --------------------------------------------------------------------------- #

def test_red_coverage_guard_fires_when_a_stem_stops_being_written(tmp_path,
                                                                 monkeypatch):
    """Important 1: the loop walked the run's own stems, so a dropped stem vanished.

    A GPU-shaped stub that writes one of the two pinned vr stems: with the peak
    guard satisfied by a patched counter, the only thing standing between this
    and a green-but-narrower gate is the coverage assert.
    """
    pinned = PINS["pins"]["vr"]
    kept = next(iter(sorted(pinned)))
    staged = {kept: settings.BASELINE_DIR / f"vr_{kept}.wav"}
    assert set(staged) < set(pinned), "need at least one dropped stem"
    monkeypatch.setattr(t7, "separate", lambda arch, device, work_dir: staged)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a, **k: 1 << 30)
    with pytest.raises(AssertionError, match="not allowed to be measured on fewer"):
        t7.test_rocm_matches_cpu("vr", tmp_path)


def test_red_frame_count_guard_fires_on_a_short_write(tmp_path, monkeypatch):
    """A GPU run writing fewer frames must not be averaged away by SNR's truncation.

    Both pinned stems are returned, ``Vocals`` truncated to half its frames, and
    ``max_memory_allocated`` inflated so the run gets past the device guards and
    reaches the frame assert.
    """
    pinned = sorted(PINS["pins"]["vr"])
    full = {stem: settings.BASELINE_DIR / f"vr_{stem}.wav" for stem in pinned}
    a = _baseline("vr", "Vocals")
    short = tmp_path / "vr_Vocals_short.wav"
    sf.write(str(short), a[: len(a) // 2], 44100, subtype="PCM_16", format="WAV")
    staged = {**full, "Vocals": short}
    monkeypatch.setattr(t7, "separate", lambda arch, device, work_dir: staged)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a, **k: 1 << 30)
    with pytest.raises(AssertionError, match="truncates to the shorter"):
        t7.test_rocm_matches_cpu("vr", tmp_path)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
