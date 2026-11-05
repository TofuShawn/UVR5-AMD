"""The worker thread -- 23 tests without torch, and one real separation with it.

Three properties matter most here and all three come from Plan 1's review lesson or
from the callbacks' real contract: the progress signal must carry a *percentage*
(the engine passes a fraction), cancellation must actually leave the engine, and the
default path -- what happens when the user changes nothing -- has to be exercised.
The last test is the one that pays for the other 23: the real VR engine on the golden
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

import time
from pathlib import Path

import pytest
from PySide6.QtCore import QThread

from app import job, session


class _Rp:
    def __init__(self, n):
        self.n = n


class FakeEngine:
    """Two coarse iteration points and a VR-shaped ramp: `set_progress_bar(0.1, 0.8/n*i)`
    is the call `separate.py:1383` makes per batch, and `process_iteration()` is what
    `separate.py:1456`/`:1482` make."""

    def __init__(self, rp, process_data):
        self.pd = process_data

    def seperate(self):
        self.pd["process_iteration"]()
        for i in range(8):
            self.pd["set_progress_bar"](0.1, 0.8 / 8 * (i + 1))
        out = Path(self.pd["export_path"])
        for stem in ("Vocals", "Instrumental"):
            (out / f"song_({stem}).wav").write_bytes(b"RIFF")


class SlowEngine(FakeEngine):
    def seperate(self):
        self.pd["process_iteration"]()
        for _ in range(50):
            self.pd["set_progress_bar"](0.02, 0.0)
            time.sleep(0.01)
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


def _session(engine=FakeEngine):
    return session.Session(spec=None, backend=None, rp=_Rp(1),
                           engine_class=engine, compute_lines=())


def _request(tmp_path, engine=FakeEngine):
    return job.JobRequest(session=_session(engine), audio=tmp_path / "in.wav",
                          export_path=tmp_path / "out")


@pytest.mark.parametrize("completed,count,step,inf,expected", [
    (1, 1, 0.1, 0.0, 10.0),
    (1, 1, 0.1, 0.8, 90.0),
    (1, 2, 0.5, 0.0, 25.0),
    (2, 2, 0.0, 1.0, 100.0),
    (0, 1, 0.25, 0.0, 0.0),      # negative before the clamp
    (3, 2, 1.0, 0.0, 100.0),     # 150 before the clamp
])
def test_progress_percent_is_upstreams_formula_exactly(completed, count, step, inf, expected):
    assert session.progress_percent(completed_models=completed, model_count=count,
                                    step=step, inference_iterations=inf) == pytest.approx(expected)


def test_run_job_reports_percentages_monotonically(tmp_path):
    seen: list[float] = []
    written = session.run_job(_session(), audio=tmp_path / "in.wav",
                              export_path=tmp_path / "out",
                              progress=lambda pct: seen.append(pct))
    assert len(written) == 2
    assert seen == sorted(seen), seen
    assert all(0.0 <= v <= 100.0 for v in seen), seen
    assert seen[-1] == pytest.approx(90.0), seen      # 0.1 + 0.8 at the last batch
    assert seen[0] > 0.0, seen                       # nothing clamped to zero means the
                                                    # iteration hook fired before the ramp


def test_cancel_is_polled_at_every_progress_hook(tmp_path):
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

    class WhereEngine(FakeEngine):
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
    class Chatty(FakeEngine):
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


def test_progress_signal_carries_a_percentage_not_a_fraction(qapp, tmp_path):
    """`Signal(int)` on the far side of a float percentage: the GUI bar reads 0-100 and
    the engine hands over a fraction, so the conversion happens once, in `run_job`.
    Same ramp as the fake above -> 20, 30, ... 90 per batch."""
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


def test_a_primary_pass_that_never_calls_process_iteration_reports_zero(qapp, tmp_path):
    """A measured gap, pinned so it cannot be shipped by accident -- not a target.

    `run_job`'s `completed_models` only advances when the engine calls
    `process_iteration`, and `separate.py` calls it for secondary models
    (`:1456`/`:1482`) and one Demucs pre-proc branch (`:1082`/`:1085`) -- never for the
    primary pass. So on the default single-model VR run the counter stays 0 and
    upstream's `base*iteration - base` term is `-100`, which the clamp turns into a bar
    that reads 0% for the whole separation. Measured on the real engine: 14 progress
    hooks, every one of them 0, job succeeded (see task-4-report.md). Upstream's GUI
    does not show this because it advances `self.iteration` once per *file*
    (`UVR.py:6451`) before the separator runs, and `run_job` handles one file per call.
    The fix belongs to `session.run_job`, not to this worker, so this test documents
    the shape rather than repairing it; Task 5's bar needs to know either way.
    """
    class PrimaryOnlyEngine:
        """The real VR primary pass: batches and console, no `process_iteration`."""

        def __init__(self, rp, process_data):
            self.pd = process_data

        def seperate(self):
            for i in range(8):
                self.pd["set_progress_bar"](0.1, 0.8 / 8 * (i + 1))
            out = Path(self.pd["export_path"])
            for stem in ("Vocals", "Instrumental"):
                (out / f"song_({stem}).wav").write_bytes(b"RIFF")

    worker = job.Job()
    pct, done = [], []
    worker.progress.connect(pct.append)
    worker.succeeded.connect(done.append)
    worker.start(_request(tmp_path, PrimaryOnlyEngine))
    assert worker.wait(10_000)
    _deliver(qapp)
    assert pct == [0] * 8, pct
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
