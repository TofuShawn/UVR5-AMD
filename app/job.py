"""One separation job on a worker thread.

The engine is synchronous and CPU/GPU bound, so it must not run on the GUI thread.
Everything it needs is already a plain value (`Session`), and everything it produces
travels back through the two callbacks `process_data` carries -- `set_progress_bar`
and `write_to_console` -- which `session.run_job` wraps into Qt signals here.

Cancellation is cooperative and its granularity is the engine's own: `separate.py`
has no stop flag, so `session.Cancelled` is raised from the progress hook
(`separate.py:1383`, once per batch on the VR path). A STOP pressed mid-batch takes
effect at the next batch, which is honest; upstream's STOP was `thread.terminate()`
(`UVR.py:6297`), which is not.

Two deviations from the shape this task was drafted with, both measured at PySide6
6.9.3 before writing it:

* The worker *is* the `QThread` (its `run()` overridden) instead of a `QObject` moved
  onto one. `QObject::moveToThread` refuses objects that already have a parent -- it
  returns False and leaves the thread affinity where it was -- so the drafted shape, a
  `_Worker` parented to the `Job`, ran the entire engine on the GUI thread while a
  fresh `QThread` idled in an empty event loop that `Job.wait()` could only time out
  on. Both halves are pinned: `test_the_work_does_not_run_on_the_main_thread` for the
  freeze (otherwise invisible, because the signals still arrive -- from the thread
  that was supposed to be idle), and every `assert worker.wait(...)` here for the
  timeout.
* One worker object per `Job`, built on the first `start()` and started again for each
  job (a finished `QThread` is reusable; measured), never deleted. Giving every job its
  own worker and `deleteLater()`ing it from the terminal handler aborted the pytest
  process with an access violation inside `QCoreApplication::processEvents()` a third of
  the way through this file's own tests.
"""
from __future__ import annotations

import dataclasses
import traceback
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal

from app import session


@dataclasses.dataclass(frozen=True)
class JobRequest:
    session: session.Session
    audio: object
    export_path: object


class Job(QObject):
    """The GUI-facing half of one job: five signals, one `start()`, one STOP.

    Teardown: this object owns a `QThread` child, and `QThread`'s destructor hard-aborts
    the process when its thread is still running. A window-close handler that lets the
    `Job` die mid-separation therefore kills the app, so it must call `shutdown()` first
    -- that is the path `closeEvent` takes (Task 5's window owns the close handler).
    `shutdown()` returns whether the drain actually happened, though: the guarantee is paid
    by a True, not by the call (cancellation is cooperative, so a short bound can time out
    with the thread still live). Read the return; see `shutdown()`.
    """

    progress = Signal(int)
    console = Signal(str)
    succeeded = Signal(list)
    failed = Signal(str)
    cancelled = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: _Worker | None = None
        self._started = False
        self._cancel = False
        # Bumped by every accepted `start()`. A terminal event still sitting in the
        # queue when the next job starts carries the generation it ran under, so the
        # `_on_*` handlers can tell a stale job-1 report from job-2's real one -- see
        # `start()` and the handlers below.
        self._generation = 0

    # -- state --------------------------------------------------------------- #

    def is_running(self) -> bool:
        """True from the moment `start()` accepts a job until that job's worker thread
        finishes -- releasing on `isFinished()`, before the queued terminal handler runs.

        Measured at PySide6 6.9.3, because each half earns its place: a thread that was
        built but never started reports `isRunning()` False *and* `isFinished()` False,
        so the thread state alone cannot answer this question at all -- hence
        `_started`. `isFinished()` is what releases the gate: it flips the moment the
        body returns, so a flag whose queued terminal handler has not run yet cannot
        strand the GUI on the other side -- a `BaseException` that walked out of the
        body without emitting anything would otherwise refuse every later job forever.
        Releasing here, while the terminal is still in the queue, is deliberate: what
        keeps a stale job's terminal from wrongly disarming this guard is the generation
        stamp `start()` applies and the `_on_*` handlers check, not a later release.

        The same two halves make this stay honest after a `shutdown()` whose bound expired:
        that path leaves `_started` up, so while the thread is still live this answers True
        and `start()` keeps refusing; the instant the engine finally wakes and returns from
        `run()`, `isFinished()` releases it even if the queued terminal is never delivered
        (a close handler that has already left the event loop). It is therefore a report on
        the thread, not on `shutdown()`'s return value.
        """
        if not self._started or self._worker is None:
            return False
        return not self._worker.isFinished()

    def request_cancel(self) -> None:
        """Ask the running job to stop.

        Not instant, and not meant to be read as instant: the engine has no stop flag,
        so this only takes effect when the wrapped `set_progress_bar` next fires --
        once per batch on the VR path (`separate.py:1383`).
        """
        self._cancel = True

    # -- drive --------------------------------------------------------------- #

    def start(self, request: JobRequest) -> None:
        """Begin one job, refusing to overlap the one *this object* is already running.

        The guard is `is_running()` on this single `Job`: it stops two runs from one
        instance, not two `Job` instances. Two live runs would share the export directory
        and the VRAM (Plan 1 measured a 76 MiB leak per separation), and worse,
        `run_job` decides which files it wrote by diffing a pre-run snapshot of that
        directory (`session.py:187`), so a second run into the same directory would
        attribute the first's files to itself -- a reporting bug, not only a resource one.

        Deliberately not process-wide (ruling R-T4-onejob): no registry or lock is added,
        because a second `Job` bypasses this guard and nothing here stops two threads. What
        makes "one job at a time" true in the shipped app is that the composition root
        (Task 5's controller) owns exactly one `Job`; Task 5 must keep to that one instance.
        """
        if self.is_running():
            self.failed.emit("a job is already running; wait for it or press STOP")
            return
        worker = self._ensure_worker()
        # `is_running()` releases on `isFinished()`, so a previous job's terminal can still
        # be queued here. Stamping the new generation before `start()` lets the `_on_*`
        # handlers ignore that stale report instead of clearing `_started` mid-job (which
        # would disarm this guard, fire the page's handler for the old job, and let a third
        # `start()` call `QThread::start()` on a live thread -- a no-op that strands it).
        self._generation += 1
        self._cancel, self._started = False, True
        worker.generation = self._generation
        worker.request = request
        worker.start()

    def _ensure_worker(self) -> _Worker:
        """Build the one worker this `Job` will ever own and wire its signals once.

        Lazy rather than in `__init__` so a `Job` nobody starts stays free, and so
        `is_running()` has a `None` to answer False with before the first job. The
        relays to `progress`/`console` are connected here rather than through the
        worker's own signals because Task 5's page listens to the `Job`, not to it:
        the worker is the part that lives on another thread.
        """
        if self._worker is None:
            worker = _Worker(lambda: self._cancel, self)
            worker.progress.connect(self.progress.emit)
            worker.line.connect(self.console.emit)
            worker.done.connect(self._on_done)
            worker.fail.connect(self._on_fail)
            worker.cancel.connect(self._on_cancel)
            self._worker = worker
        return self._worker

    # These three run as *queued* slots on the main thread, which is what marshals the
    # worker's reports into something a widget may touch. Each carries the generation its
    # job was started under and ignores a report whose generation is stale: a terminal can
    # still be in the queue when `start()` accepts the next job, and the ordering of that
    # queued terminal against the button-click that starts the next one is *not* guaranteed
    # (a click arrives as a window-system event, whose order against `sendPostedEvents` in
    # one dispatcher pass is unspecified). Without the stamp a stale job-1 terminal would
    # clear `_started` mid-job-2 and fire the page's handler for the wrong job.
    def _on_done(self, written: list, generation: int) -> None:
        if generation != self._generation:
            return
        self._started = False
        self.succeeded.emit(written)

    def _on_fail(self, text: str, generation: int) -> None:
        if generation != self._generation:
            return
        self._started = False
        self.failed.emit(text)

    def _on_cancel(self, text: str, generation: int) -> None:
        if generation != self._generation:
            return
        self._started = False
        self.cancelled.emit(text)

    def shutdown(self, ms: int = 5000) -> bool:
        """Teardown path: request cancellation, block up to `ms` for it to land, and report
        whether the worker thread is actually drained.

        True is the guarantee the class docstring advertises: the thread has finished, so
        this object may now be destroyed -- the `QThread` is a child of the `Job`, and
        `QThread`'s destructor hard-aborts the process when its thread is still running,
        which is why the reachable sequence (the user closing the window during a long
        separation) has to come through here at all.

        False means the bound expired with the thread still running, the `Job` is **not**
        drained, and destroying it now is exactly the abort this method exists to avoid --
        so the caller's only honest moves are to retry with a longer bound or to refuse the
        close and tell the user the run is still winding down, and it must not let this
        object die until some call returns True. Which of the two Task 5's `closeEvent`
        does is Task 5's call; what it may not do is assume the drain from the call alone.
        On False `_started` is deliberately left set, so `is_running()` keeps answering True
        and `start()` keeps refusing -- see that method's docstring.

        Why a timeout is a real possibility and not a formality: cancellation is cooperative
        at the engine's own granularity, never instant -- `request_cancel()` only lands when
        the wrapped `set_progress_bar` next fires (`separate.py:1383`, once per batch on the
        VR path) -- so an engine parked mid model-load or inside one long batch is unreachable
        for the whole `ms`, and `ms=5000` is a bound, not a promise. Pinned both ways by
        `tests/test_job.py`: the happy path and `StuckEngine`, whose hook-less sleep makes the
        False branch provable.

        The terminal signal (cancelled/failed) is still queued either way -- a close handler
        that is leaving the event loop should not wait for it to be delivered.
        """
        self.request_cancel()
        ok = self._worker.wait(ms) if self._worker is not None else True
        if ok:
            self._started = False
        # On False the flag stays up on purpose: clearing it here would make `is_running()`
        # lie about a live thread, hand Task 5 a "safe to destroy" it cannot honour, and let
        # `start()` call `QThread::start()` on a running thread -- the no-op that strands it.
        return ok

    def wait(self, ms: int = 1000) -> bool:
        """Test/dev helper: block until the worker thread finishes.

        It does not run the main thread's event loop, so the terminal signal the body
        posted is still in the queue when this returns True -- a caller that wants to
        observe it has to process events afterwards. That is exactly the distinction
        this suite's `_deliver()` helper exists for, and the reason a real GUI must
        never call this from a slot.
        """
        if self._worker is None:
            return True
        return self._worker.wait(ms)


class _Worker(QThread):
    """The engine call, off the GUI thread.

    Overriding `run()` means this thread has no event loop left running once the work
    is done, which is what makes `Job.wait(ms)` a timeout on the separation itself --
    the suite has no `pytest-timeout`, so that is the only bound the tests have.

    `printer=self.line.emit` is the whole console path: `session.run_job` hands it to
    `separate_cli.build_process_data`, which installs it as
    `process_data["write_to_console"]`, so every engine line arrives as a `line`
    signal. Emitting from a thread other than the `Job`'s makes those connections
    queued, and that marshalling is why the printer is a signal on the worker instead
    of a lambda that touches a widget.
    """

    progress = Signal(int)
    line = Signal(str)
    done = Signal(list, int)
    fail = Signal(str, int)
    cancel = Signal(str, int)

    def __init__(self, cancel_flag, parent=None):
        super().__init__(parent)
        self.cancel_flag = cancel_flag
        self.request: JobRequest | None = None
        # The generation the `Job` stamped onto the current request; `run()` captures it at
        # entry so a terminal always reports the job it actually ran, never the one that
        # happens to be current when the queued event is finally delivered.
        self.generation = 0

    def run(self) -> None:
        generation = self.generation
        try:
            written = session.run_job(
                self.request.session,
                audio=self.request.audio,
                export_path=self.request.export_path,
                printer=self.line.emit,
                console_log=None,
                progress=lambda pct: self.progress.emit(int(pct)),
                cancel=self.cancel_flag,
            )
        except session.Cancelled as exc:
            self.cancel.emit(str(exc), generation)
            return
        except Exception as exc:                              # noqa: BLE001 - report it
            self.fail.emit(f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}",
                           generation)
            return
        self.done.emit([Path(p) for p in written], generation)
