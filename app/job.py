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
    """The GUI-facing half of one job: five signals, one `start()`, one STOP."""

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

    # -- state --------------------------------------------------------------- #

    def is_running(self) -> bool:
        """True from the moment `start()` accepts a job until that job has finished and
        its terminal signal has been handled.

        Measured at PySide6 6.9.3, because each half earns its place: a thread that was
        built but never started reports `isRunning()` False *and* `isFinished()` False,
        so the thread state alone cannot answer this question at all -- hence
        `_started`. `isFinished()` is what releases the gate: it flips the moment the
        body returns, so a flag whose queued terminal handler has not run yet cannot
        strand the GUI on the other side -- a `BaseException` that walked out of the
        body without emitting anything would otherwise refuse every later job forever.
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
        """Begin one job. Refuses to overlap: two concurrent runs share the export
        directory and the VRAM, and the 76 MiB leak measured in Plan 1 makes a second
        live job a reporting problem as well as a correctness one."""
        if self.is_running():
            self.failed.emit("a job is already running; wait for it or press STOP")
            return
        worker = self._ensure_worker()
        self._cancel, self._started = False, True
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
    # worker's reports into something a widget may touch. Clearing `_started` here is
    # safe against the one ordering question this raises -- a terminal event still in the
    # queue when `start()` is called again -- because Qt delivers same-priority events in
    # post order, so the older job's report is always delivered before the click that
    # starts a newer one.
    def _on_done(self, written: list) -> None:
        self._started = False
        self.succeeded.emit(written)

    def _on_fail(self, text: str) -> None:
        self._started = False
        self.failed.emit(text)

    def _on_cancel(self, text: str) -> None:
        self._started = False
        self.cancelled.emit(text)

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
    done = Signal(list)
    fail = Signal(str)
    cancel = Signal(str)

    def __init__(self, cancel_flag, parent=None):
        super().__init__(parent)
        self.cancel_flag = cancel_flag
        self.request: JobRequest | None = None

    def run(self) -> None:
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
            self.cancel.emit(str(exc))
            return
        except Exception as exc:                              # noqa: BLE001 - report it
            self.fail.emit(f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}")
            return
        self.done.emit([Path(p) for p in written])
