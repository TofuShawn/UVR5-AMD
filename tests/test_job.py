"""The worker thread -- 27 tests without torch, and one real separation with it.

Three properties matter most here and all three come from Plan 1's review lesson or
from the callbacks' real contract: the progress signal must carry a *percentage*
(the engine passes a fraction), cancellation must actually leave the engine, and the
default path -- what happens when the user changes nothing -- has to be exercised.
The last test is the one that pays for the other 27: the real VR engine on the golden
crop, on the thread, end to end.

`session.run_job` is called with a fake `engine_class`, so the wrapper that does the
arithmetic and the polling is the real code under test; only the engine is fake.

Every test that asserts a signal *arrived* requests the session `qapp` fixture and
calls `_deliver()` after `worker.wait(...)`. `Job` lives on the main thread and the
worker emits from a `QThread`, so those connections are queued -- and
`QThread.wait()` does not run the main thread's event loop. Without the pump the
lists below stay empty for a reason that has nothing to do with the code under test --
confirmed on the real engine, where a loop that never pumped reported zero progress
values for a separation that had in fact fired fourteen of them. Turning the
connections into `Qt.DirectConnection` to avoid the pump would delete the marshalling
the GUI depends on, so the tests pump the loop instead.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from PySide6.QtCore import QThread

from app import job, session


class _Rp:
    def __init__(self, n):
        self.n = n


class PrimaryOnlyEngine:
    """The default single-model run: a VR-shaped `set_progress_bar(0.1, 0.8/8*i)` ramp
    (`separate.py:1383`, once per batch) and two stems -- and no `process_iteration` call,
    because the primary pass never fires it (`separate.py:1455-1456`/`:1481-1482` call it
    only for secondary/chain models, `:1082`/`:1085` only in a Demucs pre-proc pass). Pair
    it with `model_count == 1`: this is the run a user gets when they change nothing.
    """

    def __init__(self, rp, process_data):
        self.pd = process_data

    def _ramp(self):
        for i in range(8):
            self.pd["set_progress_bar"](0.1, 0.8 / 8 * (i + 1))

    def _write_stems(self):
        out = Path(self.pd["export_path"])
        for stem in ("Vocals", "Instrumental"):
            (out / f"song_({stem}).wav").write_bytes(b"RIFF")

    def seperate(self):
        self._ramp()
        self._write_stems()


class FakeEngine(PrimaryOnlyEngine):
    """A two-model run: the primary ramp, then `process_iteration()` when the second model
    starts (`separate.py:1455-1456`/`:1481-1482`, never the primary pass), then the
    secondary ramp. Pair it with `model_count >= 2` -- a single-model plan firing this
    hook would be a story the shipped code never tells, which is exactly the pairing the
    review found the earlier draft getting backwards."""

    def seperate(self):
        self._ramp()
        self.pd["process_iteration"]()
        self._ramp()
        self._write_stems()


class SlowEngine(PrimaryOnlyEngine):
    def seperate(self):
        for _ in range(50):
            self.pd["set_progress_bar"](0.02, 0.0)
            time.sleep(0.01)
        super().seperate()


class StuckEngine(PrimaryOnlyEngine):
    """A job whose cancel provably cannot land inside a few milliseconds: it parks in a
    sleep that fires *no* progress hook, and only reaches `PrimaryOnlyEngine`'s ramp -- the
    first polling point, `separate.py:1383` -- once it wakes. That is the shape of the real
    hazard `shutdown()`'s timeout branch is about (mid model-load, or one long batch, with
    the bound already elapsed): the cooperative STOP has nowhere to be noticed.

    `entered` lets the test prove the thread is already inside `seperate()` before it calls
    `shutdown()`, so a False return can only be the timeout, never a thread that had not
    started yet.
    """

    entered = threading.Event()

    def seperate(self):
        self.entered.set()
        time.sleep(0.25)
        super().seperate()


@pytest.fixture(autouse=True)
def _model_count_from_the_fake_plan(monkeypatch):
    """The stand-in plans above carry their model count as `.n`. One test does not use
    a stand-in (`test_one_real_job_on_the_worker_thread`) and undoes this itself."""
    monkeypatch.setattr(session, "model_count", lambda rp: rp.n)


# `model_count` is patched away for every test in this file, so the one test that
# asserts its arithmetic holds the un-patched function object (captured at import,
# before the fixture above runs).
_REAL_MODEL_COUNT = session.model_count


def _deliver(qapp):
    """Hand the main thread's queue over to the events the worker already posted.

    After a successful `wait()` the events are posted but undelivered, because the
    emitting thread is not the one that owns `Job`'s connections. One
    `processEvents()` pass drains the queue (`AllEvents` is its default); the extra two
    cover a slot that posts more work, and cost nothing once the queue is empty.
    """
    for _ in range(3):
        qapp.processEvents()


def _session(engine=PrimaryOnlyEngine, n=1):
    return session.Session(spec=None, backend=None, rp=_Rp(n),
                           engine_class=engine, compute_lines=())


def _request(tmp_path, engine=PrimaryOnlyEngine, n=1):
    return job.JobRequest(session=_session(engine, n), audio=tmp_path / "in.wav",
                          export_path=tmp_path / "out")


@pytest.mark.parametrize("started,count,step,inf,expected", [
    (1, 1, 0.1, 0.0, 10.0),      # primary of one model: base*step
    (1, 1, 0.1, 0.8, 90.0),      # last batch of a single-model run
    (1, 2, 0.5, 0.0, 25.0),      # primary of two: 0 -> 50 band
    (2, 2, 0.0, 1.0, 100.0),     # secondary of two, one step in: 50 -> 100 band
    (0, 1, 0.25, 0.0, 0.0),      # below the seeded range: negative -> clamp to 0
    (3, 2, 1.0, 0.0, 100.0),     # 150 before the upper clamp
])
def test_progress_percent_is_upstreams_formula_exactly(started, count, step, inf, expected):
    assert session.progress_percent(models_started=started, model_count=count,
                                    step=step, inference_iterations=inf) == pytest.approx(expected)


def test_run_job_reports_percentages_monotonically(tmp_path):
    """The default single-model path (`_session()` -> `PrimaryOnlyEngine`, `model_count`
    1): the bar rises with the batch ramp from 20 to 90, never clamped to 0."""
    seen: list[float] = []
    written = session.run_job(_session(), audio=tmp_path / "in.wav",
                              export_path=tmp_path / "out",
                              progress=lambda pct: seen.append(pct))
    assert len(written) == 2
    assert seen == pytest.approx([20, 30, 40, 50, 60, 70, 80, 90]), seen
    assert seen == sorted(seen), seen
    assert all(0.0 <= v <= 100.0 for v in seen), seen
    assert seen[-1] == pytest.approx(90.0), seen      # 0.1 + 0.8 at the last batch
    assert seen[0] > 0.0, seen                       # seeded at 1, so the primary pass is
                                                    # never clamped to zero


def test_a_two_model_run_walks_the_primary_then_secondary_band(tmp_path):
    """Ruling R-T4-iteration's two-model claim, exercised end to end (`FakeEngine` fires
    `process_iteration` once, so it is paired with `model_count` 2, never 1): the primary
    ramp lives in the 0->50 band, and after the second model starts the bar moves into
    the 50->100 band. That is upstream's shape (`UVR.py:6632-6633` bumps `iteration` at
    each model's head), not a smoothing trick."""
    seen: list[float] = []
    session.run_job(_session(FakeEngine, n=2), audio=tmp_path / "in.wav",
                    export_path=tmp_path / "out", progress=lambda pct: seen.append(pct))
    assert seen == sorted(seen), seen
    primary, secondary = seen[:8], seen[8:]
    assert len(primary) == len(secondary) == 8, seen
    assert all(0.0 <= v < 50.0 for v in primary), primary    # base*1 - base + base*step
    assert all(50.0 < v <= 100.0 for v in secondary), secondary  # base*2 - base + base*step


def test_a_cancel_raised_at_the_first_progress_hook_stops_the_job(tmp_path):
    """Named for what it proves (wave 2 minor): the flag is set inside the *first* hook
    call, so this is one branch -- the job notices a cancellation and raises -- not the
    "every hook is polled" sweep the old name promised. The per-hook sweep is the
    progress-monotonicity test above, which counts the hooks (8 + 8) but never cancels."""
    flags = {"cancel": False}
    with pytest.raises(session.Cancelled):
        session.run_job(_session(), audio=tmp_path / "in.wav", export_path=tmp_path / "out",
                        progress=lambda pct: flags.__setitem__("cancel", True),
                        cancel=lambda: flags["cancel"])


class _PlanRp:
    """Stand-in for the four `RunPlan` attributes `model_count` reads -- no engine, no
    catalog, no Qt.

    Carried in from the branch-end review: the real `model_count` had exactly one
    assertion in the whole suite (`tests/test_selections.py:233`, the secondary term),
    so its pre-proc term of 2 (`UVR.py:6602`), its 4-stem term and its splitter term
    were all unreached. This is the task that already builds fake `rp` objects, so the
    gap closes here.
    """

    def __init__(self, *, secondary=False, four_stem=(), pre_proc=None, splitter=None):
        self.is_secondary_model_activated = secondary
        self.secondary_model_4_stem = list(four_stem)
        self.pre_proc_model = pre_proc
        self.vocal_split_model = splitter


@pytest.mark.parametrize("rp,expected", [
    (_PlanRp(), 1),
    (_PlanRp(secondary=True), 2),
    (_PlanRp(pre_proc="UVR-De-Ess-Standard"), 3),                    # 1 + the 2 of :6602
    (_PlanRp(four_stem=["drums", "bass"]), 3),                       # 1 + one per slot
    (_PlanRp(splitter="5_String_Vocal_Split"), 2),                   # 1 + 1
    (_PlanRp(secondary=True, four_stem=["guitar"],
             pre_proc="UVR-De-Reverb", splitter="5_String_Vocal_Split"), 6),  # 2+1+2+1
])
def test_model_count_sums_every_term(rp, expected):
    assert _REAL_MODEL_COUNT(rp) == expected


def test_worker_emits_cancelled_not_failed(qapp, tmp_path):
    worker = job.Job()
    got, errs = [], []
    worker.cancelled.connect(got.append)
    worker.failed.connect(errs.append)
    worker.start(_request(tmp_path, SlowEngine))
    worker.request_cancel()
    assert worker.wait(10_000), "worker thread did not finish"
    _deliver(qapp)
    assert got == ["stopped by the user"], got
    assert errs == [], errs


def test_engine_errors_become_failed_with_the_type_name(qapp, tmp_path):
    class Boom:
        def __init__(self, rp, process_data):
            pass

        def seperate(self):
            raise ValueError("MIOpen has no kernel for this conv")

    worker = job.Job()
    errs = []
    worker.failed.connect(errs.append)
    worker.start(job.JobRequest(session=session.Session(spec=None, backend=None, rp=_Rp(1),
                                                        engine_class=Boom, compute_lines=()),
                                audio=tmp_path / "in.wav", export_path=tmp_path / "out"))
    assert worker.wait(10_000)
    _deliver(qapp)
    assert errs and "ValueError" in errs[0] and "MIOpen" in errs[0], errs


def test_is_running_is_false_before_start_and_after_finish(qapp, tmp_path):
    """The un-flagged default: a Job nobody started must not claim to be running, and a
    finished one must not keep STOP enabled. Both are UI states the user sees."""
    worker = job.Job()
    assert worker.is_running() is False
    worker.start(_request(tmp_path))
    assert worker.wait(10_000)
    assert worker.is_running() is False


def test_the_work_does_not_run_on_the_main_thread(qapp, tmp_path):
    """The reason this file exists: an engine call on the GUI thread freezes the window.

    `moveToThread` refuses objects that have a parent (it returns False and leaves the
    affinity where it was), so the drafted shape -- a `_Worker` parented to the `Job`
    and moved onto a `QThread` -- ran the whole engine on the main thread while the
    fresh thread idled. That freeze is invisible to every other assertion in this file,
    because the signals still arrive; the only thing that sees it is asking which
    thread the engine woke up on, so `Job` makes the worker the thread itself.
    """
    threads = []

    class WhereEngine(PrimaryOnlyEngine):
        def seperate(self):
            threads.append(QThread.currentThread())
            super().seperate()

    worker = job.Job()
    done = []
    worker.succeeded.connect(done.append)
    worker.start(_request(tmp_path, WhereEngine))
    assert worker.wait(10_000)
    _deliver(qapp)
    assert len(done) == 1, done
    assert threads and threads[0] is not QThread.currentThread(), \
        "the engine ran on the main thread"


def test_two_overlapping_jobs_are_refused(qapp, tmp_path):
    """Two live jobs share the export dir and the VRAM (Plan 1 measured a 76 MiB leak
    per separation), so the second request is refused rather than queued."""
    worker = job.Job()
    failures = []
    worker.failed.connect(failures.append)
    worker.start(_request(tmp_path, SlowEngine))
    worker.start(_request(tmp_path))
    assert any("already running" in f for f in failures), failures
    assert worker.is_running(), "the refused call must not have replaced the live job"
    worker.request_cancel()
    assert worker.wait(10_000)
    _deliver(qapp)
    assert len(failures) == 1, failures


def test_console_lines_reach_the_job_signal(qapp, tmp_path):
    """The engine's own text must arrive on `Job.console`. `run_job` hands `printer` to
    `separate_cli.build_process_data`, which installs it as `write_to_console`, so a
    fake engine that calls it is the whole path -- and the earlier draft of this file
    passed `printer=lambda line: None`, which discarded every line while Task 5's
    console widget listened for them.

    The exact text is `build_process_data`'s, not the engine's: it prefixes `[cli] `
    (that is the CLI's own stdout formatting, which `run_job` reuses) and concatenates
    `base_text` in front of the message, the way the tkinter GUI's console callback does
    it (`UVR.py:6455`, `f"{new_line}{self.base_text} {text}"`). Both are asserted rather
    than cleaned up here, because Task 5 owns the console widget and the presentation
    question with it.
    """
    class Chatty(PrimaryOnlyEngine):
        def seperate(self):
            self.pd["write_to_console"]("loading 1_HP-UVR", base_text="vr")
            super().seperate()

    worker = job.Job()
    lines, done = [], []
    worker.console.connect(lines.append)
    worker.succeeded.connect(done.append)
    worker.start(_request(tmp_path, Chatty))
    assert worker.wait(10_000)
    _deliver(qapp)
    assert lines == ["[cli] vrloading 1_HP-UVR",
                     "[cli] progress 0.1 (inference_iterations=0.1)"], lines
    assert len(done) == 1, done


def test_a_base_exception_from_the_engine_still_ends_the_job(qapp, tmp_path):
    """M4. `_Worker.run` caught `Exception`, and anything outside that -- a
    `BaseException` from a dependency, an interpreter-level abort -- used to walk out of
    the thread having emitted nothing. The page then stayed on `set_busy(True)` with
    START disabled, STOP inert and no message anywhere, while `Job.is_running()` already
    read False, so the two halves of the same object disagreed about a job that was gone.

    The guarantee this class makes is *one terminal per accepted job*, so it is made in
    a `finally` and not negotiated with the exception's class; what it says stays honest
    about not having classified the failure.
    """
    class EngineAbort(BaseException):
        pass

    class Aborts(PrimaryOnlyEngine):
        def seperate(self):
            raise EngineAbort("the engine walked out through BaseException")

    worker = job.Job()
    failures, dones = [], []
    worker.failed.connect(failures.append)
    worker.succeeded.connect(dones.append)
    worker.start(_request(tmp_path, Aborts))
    assert worker.wait(10_000)
    _deliver(qapp)
    assert dones == [], dones
    assert len(failures) == 1, failures
    assert "without reporting an outcome" in failures[0], failures[0]
    assert "EngineAbort" in failures[0], failures[0]
    assert worker.is_running() is False

    # the next job is startable, and its terminal is its own
    worker.start(_request(tmp_path))
    assert worker.wait(10_000)
    _deliver(qapp)
    assert len(dones) == 1 and len(failures) == 1, (dones, failures)


def test_a_warning_raised_inside_the_run_travels_on_its_own_signal(qapp, tmp_path):
    """I4. The engine's degradations are `RuntimeWarning`s -- `backends.py:43-60` for the
    device layer, `separate.py:80-100` when ONNX Runtime drops a provider -- and until
    now nothing in `app/**` listened, so the GUI's fourth abnormal state had no producer
    and the warning died behind the filter `separate.py:101` sets to `ignore` at import.

    They travel on a signal of their own rather than as console text, because the page
    has to be able to tell *a warning that the machine is degraded* from a line the
    engine printed -- and doing that by string-matching the message is the invention this
    item removes. The capture is scoped to the call, so the process's filter list and its
    `showwarning` come back afterwards: the suite's own warning reporting is not quietly
    replaced.
    """
    import warnings

    original_showwarning = warnings.showwarning
    original_filters = list(warnings.filters)

    class Warns(PrimaryOnlyEngine):
        def seperate(self):
            warnings.warn("ONNX Runtime dropped an execution provider for model.onnx",
                          RuntimeWarning)
            super().seperate()

    worker = job.Job()
    warned, done = [], []
    worker.warning.connect(warned.append)
    worker.succeeded.connect(done.append)
    worker.start(_request(tmp_path, Warns))
    assert worker.wait(10_000)
    _deliver(qapp)
    assert len(warned) == 1, warned
    assert "dropped an execution provider" in warned[0], warned
    assert "RuntimeWarning" in warned[0], warned
    assert len(done) == 1, done
    assert warnings.showwarning is original_showwarning, (
        "the scoped capture left the process's own warning hook replaced")
    assert warnings.filters[:3] == original_filters[:3], (
        "the scoped capture left the engine's `ignore` filter undone")


def test_progress_signal_carries_a_percentage_not_a_fraction(qapp, tmp_path):
    """`Signal(int)` on the far side of a float percentage: the GUI bar reads 0-100 and
    the engine hands over a fraction, so the conversion happens once, in `run_job`.
    Same ramp as `PrimaryOnlyEngine` (the default, single model) -> 20, 30, ... 90 per
    batch, i.e. `int(progress_percent)` of the fraction, not the fraction itself."""
    worker = job.Job()
    pct, done = [], []
    worker.progress.connect(pct.append)
    worker.succeeded.connect(done.append)
    worker.start(_request(tmp_path))
    assert worker.wait(10_000)
    _deliver(qapp)
    assert pct == [20, 30, 40, 50, 60, 70, 80, 90], pct
    assert all(0 <= v <= 100 for v in pct), pct
    assert len(done) == 1 and len(done[0]) == 2, done


def test_the_default_single_model_path_fills_the_bar(qapp, tmp_path):
    """Ruling R-T4-iteration, inverted from what this used to be. It previously asserted
    `[0]*8` (as `test_a_primary_pass_that_never_calls_process_iteration_reports_zero`) to
    pin a *measured gap*: `run_job` seeded its model counter at 0 and waited for a
    `process_iteration` hook the primary VR pass never fires (`separate.py:1455-1456`/
    `:1481-1482` fire it only for secondary/chain models), so `base*iteration - base` was
    -100 and the whole bar clamped to 0%. That seed was wrong -- `UVR.py:6632-6633` bumps
    `self.iteration` the instant each model *starts* (single model => 1 for the whole run).
    With the counter seeded at 1 this positive assertion is the honest default-path shape:
    a strictly rising bar reaching `int(step*100)`, i.e. 90 at the last batch, not zeros."""
    worker = job.Job()
    pct, done = [], []
    worker.progress.connect(pct.append)
    worker.succeeded.connect(done.append)
    worker.start(_request(tmp_path, PrimaryOnlyEngine))          # _Rp(1): model_count == 1
    assert worker.wait(10_000)
    _deliver(qapp)
    assert pct == [20, 30, 40, 50, 60, 70, 80, 90], pct
    assert len(set(pct)) == len(pct) and pct == sorted(pct), "not strictly rising"
    assert pct[-1] == int(0.9 * 100), pct                        # base*(step+inf) last batch
    assert len(done) == 1, done


def test_a_started_job_can_be_run_again_after_it_finishes(qapp, tmp_path):
    """`is_running()` is the only gate on `start()`, so a job that has finished must
    release it -- otherwise one window could separate one file, ever."""
    worker = job.Job()
    results = []
    worker.succeeded.connect(results.append)
    worker.start(_request(tmp_path))
    assert worker.wait(10_000)
    _deliver(qapp)
    worker.start(_request(tmp_path))
    assert worker.wait(10_000)
    _deliver(qapp)
    assert len(results) == 2, results
    assert not worker.is_running()


def test_a_stale_terminal_does_not_disarm_the_overlap_guard(qapp, tmp_path):
    """Finding 1 (generation stamp). `is_running()` releases at `isFinished()`, which
    flips the instant `run()` returns -- so a job-1 terminal can still be sitting in the
    queue when `start()` accepts job 2, and a button click's order against that queued
    event within one dispatcher pass is not specified. Without a stamp job 1's `_on_done`
    would then clear `_started` mid-job-2: the guard is gone, the page's `succeeded` fires
    for the old job, and a later `start()` calls `QThread::start()` on a live thread (a
    no-op that strands it -- no progress, no terminal). Here job 2 starts *without*
    pumping the loop, then the stale terminal is delivered, and the guard survives."""
    worker = job.Job()
    succeeded = []
    worker.succeeded.connect(succeeded.append)

    worker.start(_request(tmp_path, PrimaryOnlyEngine))     # job 1: fast, loop NOT pumped
    assert worker.wait(10_000)                              # run() returned; done still queued
    assert worker.is_running() is False                     # isFinished released the gate

    worker.start(_request(tmp_path, SlowEngine))            # job 2 accepted; still running
    assert worker.is_running() is True
    _deliver(qapp)                                          # now hand job 1's stale done over
    assert worker.is_running() is True, "a stale terminal disarmed the guard"
    assert succeeded == [], succeeded                       # job 1's succeeded ignored

    worker.request_cancel()                                 # clean up job 2
    assert worker.wait(10_000)
    _deliver(qapp)
    assert succeeded == [], succeeded                       # job 2 cancelled, never succeeded


def test_shutdown_stops_a_running_job(qapp, tmp_path):
    """Finding 4 (R-T4-shutdown). The `QThread` is a child of the `Job` and its destructor
    hard-aborts the process while the thread runs, so the window-close handler must drain
    the job before the `Job` dies -- the reachable case is closing the window mid-separation.
    `shutdown()` requests cancellation, waits for the cooperative STOP, and reports not
    running so the object can be destroyed safely; the cancelled terminal still arrives.
    The True return is the drain guarantee, and it is what the timeout case below takes away."""
    worker = job.Job()
    got, errs = [], []
    worker.cancelled.connect(got.append)
    worker.failed.connect(errs.append)
    worker.start(_request(tmp_path, SlowEngine))
    assert worker.is_running() is True
    assert worker.shutdown(10_000) is True, "a landed STOP must report the thread drained"
    assert worker.is_running() is False, "shutdown must leave the job not running"
    _deliver(qapp)
    assert got == ["stopped by the user"], got
    assert errs == [], errs


def test_shutdown_reports_false_when_the_cancel_cannot_land(qapp, tmp_path):
    """Fix round 2, finding 1 (the timeout branch round 1 left unproven).

    `shutdown(ms)`'s "the object can be destroyed safely" is only paid when it returns True.
    An engine parked inside a hook-less sleep cannot see the STOP at all until it wakes
    (`separate.py:1383` is the only polling point), so with a 5 ms bound `QThread::wait(ms)`
    times out with the thread still running -- and round 1 cleared `_started` regardless,
    which made `is_running()` lie, let Task 5's `closeEvent` destroy a live `QThread` child
    (the very abort finding 4 was raised to prevent) and let a later `start()` reach
    `QThread::start()` on a running thread (a silent no-op that strands the GUI).

    Option A: a failed `shutdown()` leaves the flag set, so `is_running()` keeps answering
    True, `start()` keeps refusing, and the guard releases on its own via `isFinished()` the
    moment the engine wakes -- which is then exercised here as the caller's second option:
    retry with a longer bound."""
    StuckEngine.entered.clear()
    worker = job.Job()
    got, errs = [], []
    worker.cancelled.connect(got.append)
    worker.failed.connect(errs.append)

    worker.start(_request(tmp_path, StuckEngine))
    assert StuckEngine.entered.wait(5.0), "the engine never reached the worker thread"
    assert worker.is_running() is True

    assert worker.shutdown(5) is False, "the bound must expire inside the hook-less sleep"
    assert worker.is_running() is True, "a failed shutdown must not report the job as over"

    worker.start(_request(tmp_path))                    # the gate still holds -> refused
    assert len(errs) == 1 and "already running" in errs[0], errs
    assert worker.is_running() is True, "the refused start must not replace the live job"

    # The docstring's instruction to the caller, exercised: retry with a longer bound, and
    # only a True return means the thread is drained and the `Job` may be destroyed.
    assert worker.shutdown(10_000) is True, "the retry must drain the thread"
    assert worker.is_running() is False
    _deliver(qapp)
    assert got == ["stopped by the user"], got          # the STOP landed at the next hook
    assert len(errs) == 1, errs                         # nothing beyond the one refusal


def test_wait_on_an_idle_job_does_not_disarm_the_current_generation(qapp, tmp_path):
    """R-T4-wait (controller-authorised into Task 5, because Task 5 is the first consumer).

    `QThread::wait()` answers **True** for a thread that never started -- Qt even warns
    `QThread: Waiting on thread before it starts` -- so returning its raw result let an idle
    `Job` report a drain it had not performed. `wait()` never wrote `_started` itself; the
    damage is in the answer a caller acts on: a close handler that reaches for `wait()`
    instead of `shutdown()` is handed a True it cannot honour, and anything it releases on
    that basis is released without a thread having been drained -- the generation stamp the
    next `start()` applies then has no live worker behind it, later terminals read as stale,
    and the GUI sits busy with no job.

    Three assertions, the third being the symptom that matters: after an idle `wait()` the
    *next* job's terminal is still honoured, so the page's START/STOP pairing keeps working.
    """
    worker = job.Job()
    done = []
    worker.succeeded.connect(done.append)
    worker.start(_request(tmp_path))
    assert worker.wait(10_000)
    _deliver(qapp)
    assert len(done) == 1, done                      # the stamp matched: a real terminal

    assert worker.wait(50) is False, "an idle Job has no live thread to drain"
    assert worker.is_running() is False
    assert worker._generation == 1 and worker._started is False, (
        "an idle wait() disarmed the current generation")

    worker.start(_request(tmp_path))                 # job 2, stamped generation 2
    assert worker.wait(10_000)
    _deliver(qapp)
    assert len(done) == 2, f"job 2's terminal was ignored as stale: {done}"
    assert worker._generation == 2, worker._generation


def test_one_real_job_on_the_worker_thread(qapp, tmp_path, monkeypatch):
    """The fakes prove the plumbing, not the path. This runs the real VR engine on the
    golden crop through Job and asserts the stems landed. It is the only test in Plan 2
    that proves the thread model can carry an actual separation."""
    from tests.golden import runner
    import settings as settings_module
    import separate_cli
    from app import session as s

    # The autouse fixture above replaces `model_count` with one that reads `.n` off the
    # stand-in plans; here the `rp` is a real `RunPlan`, so the fake has to come off
    # before the engine runs or the progress hook dies on an AttributeError.
    monkeypatch.undo()

    cat = separate_cli.make_catalog(None, tmp_path / "w")
    sess = s.assemble(st=settings_module.Settings(), cat=cat, model="1_HP-UVR",
                      force_device="cpu")
    crop = runner.golden_input()
    worker = job.Job()
    done, errs, lines = [], [], []
    worker.succeeded.connect(done.append)
    worker.failed.connect(errs.append)
    worker.console.connect(lines.append)
    worker.start(job.JobRequest(session=sess, audio=crop, export_path=tmp_path / "out"))
    assert worker.wait(240_000), "real job did not finish in 240 s"
    _deliver(qapp)
    assert not errs, errs
    assert len(done) == 1 and len(done[0]) == 2, done
    assert all(p.is_file() and p.stat().st_size > 1000 for p in done[0]), done[0]
    assert lines, "the real engine produced no console text at all"
