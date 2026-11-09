"""The composition root: the one place that sees catalog, settings, selections, Job.

Everything else in `app/` takes one of those as a given. This module owns the pairing:
it builds the `Catalog`, loads `Settings`/`Selections`, constructs the `WorkPage`, and
connects the page's `run_requested` to the single assembly path (`session.assemble`) and
the single worker (`page.job.start`). Its third job is the settings dialog: the page
edits the same two objects this root holds, so a save here is what the next run
assembles from. Two rules live here and nowhere else:

* **Exactly one `Job`** (ruling R-T4-onejob). It is the page's `job` attribute, built once
  when the page is built, and this module is the only place that starts it. No second
  `Job` in a helper, no process-wide registry -- the one-instance rule holds because this
  root constructs one page, and a reviewer checks it by grep.
* **One file per `Job`, many files per run.** `session.run_job()` separates one audio file
  into one output directory, so a batch is this root walking a queue of `JobRequest`s --
  `_Batch`, advanced by the terminals the one `Job` already emits. The engine, `Job` and
  `session` are untouched by it, which is the reason the rule above can stay true while the
  page runs forty files.
"""
from __future__ import annotations

import dataclasses
import weakref
from pathlib import Path

import catalog
import fluentqt
import plan
import separate_cli
import selections as selections_module
import settings as settings_module
import ui_prefs as ui_prefs_module
from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QVBoxLayout

from app import dialogs, job, session, settings_page, shell, work_page
from app.i18n import tr


@dataclasses.dataclass
class _Batch:
    """The queue, its position, and its tally. The root's business, all of it.

    Deliberately not in `app/job.py`: `Job` is one file (`start()`'s guard at
    `job.py:141-158` explains the three reasons -- a shared export directory, shared VRAM,
    and `run_job`'s before/after snapshot of the former attributing one run's stems to
    another), `session.run_job` is one file, and the engine is one file. The only change to
    `job.py` for batch mode is one constant and no behaviour: `BUSY_REFUSAL` is the sentence
    `start()` emits when it refuses, held there so `_start_current_file` can pre-empt the
    same refusal without a second copy of it. `tests/test_job.py` runs unmodified as the
    evidence that nothing else moved.
    """

    sess: session.Session
    files: list[Path]
    index: int = 0
    written: int = 0
    failures: list[str] = dataclasses.field(default_factory=list)
    stopping: bool = False

    @property
    def current(self) -> Path:
        return self.files[self.index]


class Controller:
    """Wires the page's buttons to the assembly path, the worker thread and the settings.

    Three signals, three owners: `run_requested` assembles and starts the one `Job`,
    `cancel_requested` asks that same `Job` to stop, and `settings_requested` opens the
    settings dialog -- which does not construct a `Job` (R-T4-onejob) and does not fetch
    anything until the user confirms inside `model_refresh`.
    """

    def __init__(self, *, cat: catalog.Catalog | None = None,
                 st=None, selections=None, ui_obj=None):
        self.catalog = cat if cat is not None else separate_cli.make_catalog()
        self.settings = st if st is not None else settings_module.load()
        self.selections = (selections if selections is not None
                           else selections_module.load())
        self.ui_prefs = ui_obj if ui_obj is not None else ui_prefs_module.load()
        self.work_page = work_page.WorkPage(cat=self.catalog, settings_obj=self.settings,
                                             selections_obj=self.selections,
                                             ui_obj=self.ui_prefs)
        self.work_page.run_requested.connect(self._on_run)
        # The root, not `job.request_cancel`, is what STOP now reaches: the mark that stops
        # the *rest* of a batch has to be set by the button, because a cancellation lands
        # when the engine's next progress hook fires and a STOP before that has no terminal
        # to learn from. See `_on_cancel_requested`.
        self.work_page.cancel_requested.connect(self._on_cancel_requested)
        # The queue walks itself from the one `Job`'s terminals. Connected after the page
        # built its own handlers, so Qt delivers the page's first and the root's second --
        # the display knows about the file before the state machine moves past it.
        self.work_page.job.succeeded.connect(self._on_file_done)
        self.work_page.job.failed.connect(self._on_file_failed)
        self.work_page.job.cancelled.connect(self._on_file_cancelled)
        # The run in flight, or None. `None` is what makes the three slots above ignore the
        # "a job is already running" report `Job.start` emits on a double click: that is not
        # a file ending, and treating it as one would walk the queue on a run that never
        # started.
        self._batch: _Batch | None = None
        # The default output folder is the one setting the *work page* owns, and it comes
        # through this root for the same reason the other two objects do: this is the place
        # that knows where the settings file is and what a refused write means.
        self.work_page.default_output_requested.connect(
            self._on_default_output_requested)
        # The two `Settings` fields that now live on the work page, on the same seam as
        # the one above: the widget asks, this root writes, and a refusal reverts the
        # widget rather than letting it keep showing what did not get stored.
        self.work_page.format_requested.connect(self._on_format_requested)
        self.work_page.gpu_requested.connect(self._on_gpu_requested)
        # The run-output list's three buttons. The page shows this session's stems and
        # asks; opening a file, revealing its folder and deleting it are desktop effects
        # with a confirmation and an error report, so they live in the root.
        self.work_page.open_output_requested.connect(self._on_open_output)
        self.work_page.reveal_output_requested.connect(self._on_reveal_output)
        self.work_page.delete_output_requested.connect(self._on_delete_output)
        # The one window's shell: a right rail that swaps the main content between the work
        # page and a full-page settings view, replacing the old second (dialog) window.
        self.shell = shell.AppShell(self, self.work_page, settings_obj=self.settings,
                                    selections_obj=self.selections)
        self.work_page.settings_requested.connect(self._on_settings_requested)

    # -- START ---------------------------------------------------------------- #

    def _on_run(self) -> None:
        """Assemble once, then walk the queue one file at a time.

        Nothing here runs the engine: `job.start()` moves it to the worker thread, and the
        page's own handlers marshal the results back. Every refusal below happens *before*
        a thread exists, so the busy state is never left on by a failed start.

        The plan is built once for the whole batch, because it does not depend on the
        file: `session.assemble` reads settings, catalog, selections and the machine, and
        `run_job` is the half that takes an audio path. Assembling per file would reload a
        model table per file for no reason and could make two files of one batch disagree
        about the plan they ran under.
        """
        page = self.work_page
        blockers = page.blockers()
        if blockers:
            # The same list the reason label is showing, joined the same way -- which is
            # why `WorkPage.blockers()` exists at all.
            self._refuse("not started: " + "; ".join(blockers))
            return
        files = page.pending_files()
        model = page.selected_model()

        try:
            sess = session.assemble(st=self.settings, cat=self.catalog, model=model,
                                    arch=page.current_arch(),
                                    force_device=page.selected_force_device(),
                                    # The line that makes secondary models reachable from a
                                    # GUI for the first time: Task 3's 33 fields, in the shape
                                    # `plan.build` documents.
                                    selections=self.selections.to_plan_dict(),
                                    # `assemble` prints the machine-derived device lines; the
                                    # console widget is where a GUI user reads them.
                                    printer=page.console.appendPlainText)
        except catalog.UnknownModelParamsError as exc:
            self._offer_params(exc.spec)
            return
        except (catalog.CatalogError, ValueError, plan.PlanError) as exc:
            self._refuse(f"not started: {exc}")
            return
        except Exception as exc:                              # noqa: BLE001 - report it
            # The last resort, and it is load-bearing rather than defensive. Two ways this
            # slot used to be able to die quietly: a `Settings` value that `plan.build`
            # subscripts raw -- `plan.py:368`'s `DEVERB_MAPPER[settings.deverb_vocal_opt]`,
            # for which the settings page used to render a free-text box, so any other
            # string raised `KeyError` -- and any engine-side surprise. Neither was in the
            # tuple above; an exception that escapes a Qt slot on PySide6 6.9.3 prints to
            # stderr and the event loop carries on, which `app/settings_page.py:281-291`
            # already documents as "looked exactly like a button that did nothing". Naming
            # the type keeps a real bug actionable instead of invisible.
            self._refuse(f"not started: {type(exc).__name__}: {exc}")
            return

        page.show_compute(sess.compute_lines)
        self._batch = _Batch(sess=sess, files=files)
        page.begin_batch(len(files))
        self._start_current_file()

    # -- the batch ------------------------------------------------------------ #

    def _start_current_file(self) -> None:
        """Hand the queue's current file to the one `Job`.

        `set_busy(True)` is here rather than in `_on_run` because it has to be true for
        every file and released for none of them: the release is `end_batch`'s, and a busy
        state that came off at file 1's terminal would let START overlap file 2 -- which
        `Job.start` refuses with "a job is already running", and a refusal on a button the
        user was told was enabled is exactly the thing this fork keeps being asked to fix.
        """
        batch = self._batch
        if batch is None:
            return
        if self.work_page.job.is_running():
            # `Job.start` reports this very refusal on `failed` (`job.py:156-158`) -- the
            # same signal a file's engine error arrives on. Letting it through would make
            # `_on_file_failed` tally a file that never ran and `_advance` start the next
            # one, which refuses again: a three-file batch ends as "3 of 3 failed" with a
            # named failure per file, for runs that never existed. So the refusal is caught
            # here, where the queue is still known, and ends the batch once.
            self._batch = None
            self.work_page.end_batch(written=batch.written, failures=batch.failures,
                                     stopped=True)
            self._refuse(f"not started: {job.BUSY_REFUSAL}")
            return
        self.work_page.set_busy(True)
        self.work_page.job.start(job.JobRequest(session=batch.sess, audio=batch.current,
                                                export_path=self.export_for(batch.current)))

    def _on_file_done(self, written: list) -> None:
        if self._batch is None:
            return                                  # the "a job is already running" report
        self.work_page.add_written(written)
        self._batch.written += len(written)
        self._advance()

    # -- run output ------------------------------------------------------------ #

    def _on_open_output(self, path: str) -> None:
        self._reveal(Path(path), Path(path))

    def _on_reveal_output(self, path: str) -> None:
        self._reveal(Path(path), Path(path).parent)

    def _reveal(self, source: Path, target: Path) -> None:
        """Hand `target` to the desktop; say so when the desktop refuses.

        `openUrl` answers False for a missing handler (a file type with no association,
        a folder the shell will not open). Swallowing that leaves a button that appears
        to do nothing, which is the failure mode this page keeps being written against.
        """
        if QDesktopServices.openUrl(QUrl.fromLocalFile(str(target))):
            return
        self.work_page.console.appendPlainText(
            tr("open_failed", path=str(source)))

    def _on_delete_output(self, path: str) -> None:
        """Delete one stem, after a confirmation.

        The path is the list's selection, and the list only ever holds what
        `Job.succeeded` reported, so this cannot reach an input file. A refused unlink
        (a file open in a player, a read-only folder) keeps both the file and its row,
        because a row removed over a file still on disk is the list lying about the disk.
        """
        target = Path(path)
        if not dialogs.confirm_delete(target, parent=self.shell):
            return
        try:
            target.unlink()
        except OSError as exc:
            self.work_page.console.appendPlainText(
                tr("delete_failed", name=target.name, err=exc))
            return
        self.work_page.forget_written(str(target))
        self.work_page.console.appendPlainText(tr("deleted", name=target.name))

    def _on_file_failed(self, text: str) -> None:
        """Continue on error, and name the file that errored.

        5.6's batch does not stop on one bad file, and neither should this: the everyday
        shape of a batch is forty files, one of which is corrupt, and a run that abandoned
        the other thirty-nine on that would make the user find out which one by guessing.
        The first line is what gets tallied -- `Job.failed`'s text carries four frames of
        engine source after it, and the console above already has the whole thing.
        """
        batch = self._batch
        if batch is None:
            return
        first = text.splitlines()[0] if text else "the engine gave no reason"
        batch.failures.append(f"{batch.current.name}: {first}")
        self._advance()

    def _on_file_cancelled(self, text: str) -> None:
        batch = self._batch
        if batch is None:
            return
        batch.stopping = True                       # STOP stops the *rest*
        self._advance()

    def _advance(self) -> None:
        """One file ended: move on, or end the run.

        Reentrancy is the thing to look at here. `Job`'s terminals are *queued* when a
        worker thread is running, so in the shipped app this walks the queue one event-loop
        turn at a time; in a test, where `start` is replaced on the main thread, Qt delivers
        directly and the whole batch runs inside `run_requested.emit()`. Both are the same
        state machine, and neither is allowed to overlap two files: `_start_current_file`
        is only ever reached from a terminal, at which point `Job.is_running()` has already
        released on `isFinished()` (`app/job.py:96-120`).
        """
        batch, page = self._batch, self.work_page
        if batch is None:
            return
        page.end_file()
        batch.index += 1
        if batch.stopping or batch.index >= len(batch.files):
            self._batch = None
            page.end_batch(written=batch.written, failures=batch.failures,
                           stopped=batch.stopping)
        else:
            self._start_current_file()

    def _on_cancel_requested(self) -> None:
        """Mark the queue stopped, then ask the one `Job`.

        The mark goes first, and it is the button that sets it rather than the terminal
        that reports it: cancellation is cooperative at the engine's own batch granularity
        (`separate.py:1383`), so a STOP pressed during a model load has no terminal to come
        from yet -- and a queue that only learned about stopping from terminals would start
        file 2 anyway.
        """
        if self._batch is not None:
            self._batch.stopping = True
        self.work_page.job.request_cancel()

    def export_for(self, audio: Path) -> Path:
        """Where this file's stems go, in the three places that can decide it.

        Typed path > `ui.export_path` (the GUI's configured default) > `UVR_out` beside
        the input. The last tier is `session.default_export_path` and is what the CLI does
        with no flag; the middle one is deliberately absent there -- a folder the user
        ticked a box over must not silently move a scripted CLI run's output, which is why
        the override is resolved in this composition root and not inside `session`, the one
        assembly path the two front ends share.
        """
        typed = self.work_page.output_edit.text().strip()
        if typed:
            return Path(typed)
        if self.ui_prefs.export_path:
            return Path(self.ui_prefs.export_path)
        return session.default_export_path(audio)

    def _on_default_output_requested(self, value: str) -> None:
        """Persist the page's toggle, then hand it back what the disk says.

        The order is the point: the page is only allowed to show a default that is
        actually stored. A `settings.json` open in an editor is the everyday Windows case
        -- `SettingsPage.apply` already documents that failure for the settings half -- so
        the checkbox is reverted to the old value and the reason is printed, rather than
        remembering a folder the next launch will not know about.

        `UiPrefs(export_path=value)` rather than a fresh `load()`: the write just succeeded,
        so this *is* the file's content, and re-reading it would make every test that
        redirects `APPDATA` depend on which config file happened to be reachable.
        """
        page = self.work_page
        try:
            ui_prefs_module.save(ui_prefs_module.UiPrefs(export_path=value),
                                 settings_module.config_file())
        except (OSError, TypeError, ValueError) as exc:
            page.set_default_output(self.ui_prefs.export_path)
            page.console.appendPlainText(
                f"default output not saved: {type(exc).__name__}: {exc} -- the default "
                f"stays {self.ui_prefs.export_path or 'UVR_out beside each input'}")
            return
        self.ui_prefs = ui_prefs_module.UiPrefs(export_path=value)
        page.set_default_output(value)
        page.console.appendPlainText(
            f"default output folder {'set to ' + value if value else 'cleared'} -- "
            f"written to {settings_module.config_file()}")

    def _apply_setting(self, republish, field: str, value) -> bool:
        """One field at a time, into the frozen `Settings`, then onto the disk.

        `dataclasses.replace` rather than a widget-side draft: the object the next
        `session.assemble` reads is this one, so the change takes effect on the next run --
        the same contract `save_settings` already advertises for the dialog
        (`controller.py:271-273`), and the reason the page's parameter summary refreshes.

        `republish` is the page's own un-emitting setter (`set_stem_format`, `set_gpu`) and
        it is called on *both* outcomes, which is the whole of the widget's honesty: after
        a success the control is re-set from the object that is now stored, and after a
        refusal from the one that always was. A control that only got told the truth when
        the write failed would be showing what the user clicked rather than what the file
        says -- and in the gap between the click and the failure the summary line would
        already have promised the run the engine will not perform.

        It arrives as a callable rather than a field name so this root never learns which
        widget holds which value; a root that knew both would be a root with two things to
        update when a control moves.

        A dialog left open across a successful write is the one way this contract could be
        undone behind the user: `SettingsPage.apply` builds all 50 values from the widgets
        it was handed, so its next Save would write back the pre-change numbers. Closing it
        is visible; losing an edit to a stale Save is not recoverable, so it closes and the
        console says why.
        """
        page = self.work_page
        old = getattr(self.settings, field)
        try:
            new = dataclasses.replace(self.settings, **{field: value})
            new.save(settings_module.config_file())
        except (OSError, TypeError, ValueError) as exc:
            republish(old)
            page.console.appendPlainText(
                f"not saved: {type(exc).__name__}: {exc} -- {field} stays {old}")
            return False
        self.settings = new
        page.settings = new
        page.refresh_parameter_summary()
        if self.shell.has_settings():
            self.shell.reset_settings()
            page.console.appendPlainText(
                "settings closed: it still held the values from before this change"
                " -- open Settings again to edit them")
        republish(getattr(new, field))
        page.console.appendPlainText(f"settings saved: {field} = {value}")
        return True

    def _on_format_requested(self, fmt: str) -> None:
        self._apply_setting(self.work_page.set_stem_format, "save_format", fmt)

    def _on_gpu_requested(self, on: bool) -> None:
        self._apply_setting(self.work_page.set_gpu, "gpu_conversion", on)

    def _refuse(self, text: str) -> None:
        """Say why nothing was started. No dialog, no traceback, no fake progress.

        The compute line goes empty with the refusal (I5(a)): it is a statement about
        what the *engine will compute on*, and every path that reaches here has decided
        the engine will not run at all. Leaving the previous run's line up meant the page
        still asserted `compute device: cuda` under "not started" -- one of the two ways
        this root used to tell a truth it had stopped being true.
        """
        self.work_page.show_compute(())
        self.work_page.console.appendPlainText(text)

    # -- TIER 3 --------------------------------------------------------------- #

    def _offer_params(self, spec) -> None:
        """Turn the catalog's tier-3 refusal into a form, then run what it answered.

        Modal, and that is the measured shape rather than a preference: a `ContentDialog`
        reports `isModal()` True on this wheel, and this branch runs **before**
        `job.start()` -- no worker thread exists yet, so a nested loop cannot interleave
        with a run and R-T4-onejob is untouched. `exec()` is also what makes reading the
        answer on the line after asking honest: `collected_params` asks for `result()`,
        and a modeless dialog would still report 0 (`ResultNone`) with the user's typing
        still to come.

        Three exits, and only one of them is worth a message:

        * no field list for the arch -> `model_params_dialog` already said so through the
          notification queue, and the catalog's own line goes to the console;
        * dismissed (the close button, Escape, or the window) -> nothing is written and
          nothing is reported: a change of mind is not a failure, and the run stays
          unresolved exactly as the CLI left it;
        * answered -> `write_params` into `writable_dir`, where `resolve()` looks first
          (R-T10-writable: preferring the older weights-dir copy would make this popup
          read as "it did not save"), then START again with what the user said.

        Pressing Save on a half-empty form is the fourth case, and it is the one the
        refusal names: `unanswered_fields` lists the boxes, because a JSON with a hole in
        it would fail later, inside `plan.build`, with a message that blames the engine.
        """
        page = self.work_page
        dlg = dialogs.model_params_dialog(spec, self.catalog, offer=False, parent=page)
        if dlg is None:
            page.show_cancelled(spec.missing_message)
            return
        try:
            dlg.exec()
            dismissed = dlg.result() != fluentqt.ContentDialog.ResultPrimary
            params = dialogs.collected_params(dlg)
            if params is not None:
                written = dialogs.write_answer(self.catalog, spec, params)
                page.console.appendPlainText(f"params written to {written}")
                self._on_run()
                return
            if not dismissed:
                page.console.appendPlainText(
                    "params not written: this form still needs a value for "
                    + ", ".join(dialogs.unanswered_fields(dlg))
                    + " -- press START again and fill them in")
        finally:
            # Detach, whatever the answer was. The form is built with the page as its
            # parent, so a dialog that stays parented is a Card, seven editors and a
            # Catalog reference the page holds for the rest of the session -- one per
            # refusal, and a user who mistypes an answer gets several. Dropping the
            # parent hands ownership back to Python, and the local goes with it; measured
            # on this wheel, `deleteLater()` would not have gone before the next real
            # event loop, so a census taken after this line would still see it.
            dlg.setParent(None)

    # -- SETTINGS -------------------------------------------------------------- #

    def _on_settings_requested(self) -> None:
        """Swap the one window's main content to the full-page settings view.

        The shell builds the 83-field page lazily on first open (a user who never opens
        Settings does not pay for it at start-up) and `Work` in the right rail swaps back;
        the old separate settings window is gone.
        """
        self.shell.show_settings()

    def save_settings(self, page) -> list[str]:
        """Write what the page validated, then hand the new objects to every later reader.

        The destination is the fork's own settings file -- `config_file()`, the same path
        `load()` read at start-up. `apply(path)` takes the destination precisely so a test
        can point the page somewhere else; nothing here writes a second format.

        A refusal means the settings view stays open with the message in the console, because the
        user has to be able to act on it. `SettingsPage.apply` reports both ways a save can
        fail -- a value the type system refused (nothing was written) and a file that
        refused to be written (a `settings.json` open in an editor is the everyday Windows
        case) -- and its message says which half reached the disk, so this line does not
        have to guess. On success the controller's two objects become the page's: the next
        START assembles from *these*, so a saved change takes effect on the next run rather
        than after a restart. The work page's reference follows, but `refresh_models()` is
        deliberately **not** called: the Model combo is a per-run pick that may legitimately
        differ from the persisted `*_model` selection, and re-populating it here would
        discard a model the user chose and has not run yet.
        """
        messages = page.apply(settings_module.config_file())
        console = self.work_page.console.appendPlainText
        if messages:
            # One line per message -- `apply` names the field it refused or the half it
            # could not write, and that is what the user has to act on.
            for line in messages:
                console(line)
            console("settings not applied: see the line above, fix it, and press Save "
                    "again -- the settings view stayed open for that")
            return messages
        self.settings, self.selections = page.st, page.sel
        self.work_page.selections = self.selections
        # The work page now *shows* settings values (M1's parameter summary), so the
        # object it reads has to be the one that was just saved. `refresh_models()` is
        # still deliberately not called -- the Model combo is a per-run pick that may
        # legitimately differ from the persisted `*_model` selection -- but the summary
        # has no such excuse for going stale.
        self.work_page.settings = self.settings
        self.work_page.refresh_parameter_summary()
        # Two of these fields are now *displayed* on the work page, and the truth strip reads
        # the widgets rather than the object -- so they follow the save. Both republish with
        # signals blocked: showing a stored value is not a user action, and a `toggled` here
        # would ask `_apply_setting` to write back what was just read.
        self.work_page.set_stem_format(self.settings.save_format)
        self.work_page.set_gpu(bool(self.settings.gpu_conversion))
        console(f"settings saved to {settings_module.config_file()}")
        if self._batch is not None:
            # The settings view is non-modal, so Save can land mid-run. The batch keeps the plan it
            # assembled once, and a bare "settings saved" over a running job reads as a
            # change to that job -- the deferred-write lie the page's own controls are
            # frozen against while busy.
            console("the run in flight keeps the plan it started with; these values are "
                    "the next run's")
        return messages

    # -- teardown ------------------------------------------------------------- #

    def request_close(self, ms: int) -> bool:
        """Ruling R-T4-shutdown: a window may only be released once the worker thread is
        actually gone, because `QThread`'s destructor hard-aborts the process while its
        thread runs and the `Job`'s thread is a child of the `Job`.

        `Job.shutdown(ms)` returns True for that, and **False while the thread is still
        live** -- cancellation is cooperative at the engine's own batch granularity
        (`separate.py:1383`), so a bound can expire inside one long batch. This root takes
        the second option the return offers: refuse the close and tell the user the run is
        still winding down, rather than destroy a live job. The user closes again once the
        engine reports back.
        """
        if self.work_page.job.shutdown(ms):
            return True
        self.work_page.console.appendPlainText(
            "close refused: this separation is still winding down, so the window stayed "
            "open -- destroying it now would abort the process. Close again once the run "
            "reports back (or press STOP and wait for it).")
        return False
