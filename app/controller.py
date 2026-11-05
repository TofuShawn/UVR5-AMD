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
* **One file per job.** A run is one audio file into one output directory, because that is
  what `session.run_job()` does; the page's percentage is honest only in that shape.
"""
from __future__ import annotations

import weakref
from pathlib import Path

import catalog
import fluentqt
import plan
import separate_cli
import selections as selections_module
import settings as settings_module
from PySide6.QtWidgets import QDialog, QVBoxLayout

from app import dialogs, job, session, settings_page, work_page


class SettingsDialog(QDialog):
    """The shell around `SettingsPage` -- still a `QDialog`, and still on purpose.

    Task 7 delivered `app/dialogs.py` (the tier-3 form, the failure surfaces, the
    pickers) and did not take this shell over: a plain `QDialog` and not a
    `fluentqt.ContentDialog` is what the plan allows for the settings page, because
    `ContentDialog` is modal and the page has to keep printing while it is open. The
    guard in `tests/test_ui_surfaces.py` treats a `QDialog` as an allowed surface for the
    same reason, and everything *inside* this one is Fluent: the page is a `Card`, its
    fields are Fluent widgets.

    Non-modal on purpose. `SettingsPage.refresh_tables` prints four fetch lines into the
    page's own status box and a refused save is reported in the work page's console, so a
    modal `exec()` loop would hide the output this dialog exists to produce -- and would
    also stop the user watching a run they started before opening it.
    """

    def __init__(self, ctrl, *, settings_obj, selections_obj, parent=None):
        super().__init__(parent)
        # A weak reference to the controller, not a strong one. `Controller` holds this
        # dialog in `self.settings_dialog`, so a strong `self.controller = ctrl` closes a
        # Python cycle (controller -> dialog -> controller) that only the *cyclic*
        # collector can free -- and pytest's `gc_collect_harder()` at unconfigure runs
        # exactly that collector. Measured: with the strong back-reference the dialog and
        # its page were collected there and the process died with heap corruption
        # (`0xc0000374`) during "Garbage-collecting", 6 runs out of 6 once four dialogs
        # had accumulated; a single dialog was clean, which is why it hides from a
        # one-test run and bites the full suite. Held weakly, the dialog's lifetime is the
        # Qt parent chain's alone, and the collector never finalises it behind Qt's back.
        self._ctrl_ref = weakref.ref(ctrl)
        self.setWindowTitle("Settings")
        self.page = settings_page.SettingsPage(settings_obj=settings_obj,
                                               selections_obj=selections_obj, parent=self)
        self.save_button = fluentqt.Button("Save settings", self)
        # Connected through a weak reference for the same reason as the back-reference
        # above, and the same reason `SettingsPage` connects its own refresh button (see
        # `settings_page._weak_slot`): a bound `self.save` held by the child button is a
        # second dialog -> button -> bound-method -> dialog cycle.
        self.save_button.clicked.connect(
            settings_page._weak_slot(weakref.ref(self), "save"))
        layout = QVBoxLayout(self)
        layout.addWidget(self.page, 1)
        layout.addWidget(self.save_button)

    def save(self) -> list[str]:
        ctrl = self._ctrl_ref()
        if ctrl is None:
            return []
        return ctrl.save_settings(self.page, self)


class Controller:
    """Wires the page's buttons to the assembly path, the worker thread and the settings.

    Three signals, three owners: `run_requested` assembles and starts the one `Job`,
    `cancel_requested` asks that same `Job` to stop, and `settings_requested` opens the
    settings dialog -- which does not construct a `Job` (R-T4-onejob) and does not fetch
    anything until the user confirms inside `model_refresh`.
    """

    def __init__(self, *, cat: catalog.Catalog | None = None,
                 st=None, selections=None):
        self.catalog = cat if cat is not None else separate_cli.make_catalog()
        self.settings = st if st is not None else settings_module.load()
        self.selections = (selections if selections is not None
                           else selections_module.load())
        self.work_page = work_page.WorkPage(cat=self.catalog, settings_obj=self.settings,
                                             selections_obj=self.selections)
        self.work_page.run_requested.connect(self._on_run)
        self.work_page.cancel_requested.connect(self.work_page.job.request_cancel)
        # The settings dialog, kept so it is not collected while it is on screen. One at a
        # time: a second click re-raises the dialog it already opened rather than stacking
        # another copy of 83 fields over the first.
        self.settings_dialog: SettingsDialog | None = None
        self.work_page.settings_requested.connect(self._on_settings_requested)

    # -- START ---------------------------------------------------------------- #

    def _on_run(self) -> None:
        """Assemble, publish the compute line, hand the job to the one `Job`.

        Nothing here runs the engine: `job.start()` moves it to the worker thread, and the
        page's own handlers marshal the results back. Every refusal below happens *before*
        a thread exists, so the busy state is never left on by a failed start.
        """
        page = self.work_page
        model = page.selected_model()
        if model == catalog.NO_MODEL:
            self._refuse("not started: pick a model from the Model list first")
            return

        raw_input = page.input_edit.text().strip()
        if not raw_input:
            self._refuse("not started: choose an input audio file first")
            return
        audio = Path(raw_input)
        if not audio.is_file():
            # Same wording as the CLI's (`separate_cli.run()`), because the file really is
            # the thing the user needs to see in order to fix it.
            self._refuse(f"not started: input audio not found: {audio}")
            return

        # Empty output means the CLI's default: a folder next to the file. Two front ends
        # choosing different places for the same run is the bug this line prevents.
        raw_output = page.output_edit.text().strip()
        export = Path(raw_output) if raw_output else audio.parent / "UVR_out"

        try:
            sess = session.assemble(st=self.settings, cat=self.catalog, model=model,
                                    arch=page.current_arch(), force_device=None,
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

        page.show_compute(sess.compute_lines)
        page.set_busy(True)
        page.job.start(job.JobRequest(session=sess, audio=audio, export_path=export))

    def _refuse(self, text: str) -> None:
        """Say why nothing was started. No dialog, no traceback, no fake progress."""
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
        """One dialog, re-raised rather than stacked.

        Built lazily: the page is 83 widgets, and a user who never opens it should not
        pay for it at start-up -- which also keeps it out of `tests/test_ui_surfaces.py`'s
        tree until something asks for it.
        """
        if self.settings_dialog is None:
            self.settings_dialog = SettingsDialog(self, settings_obj=self.settings,
                                                  selections_obj=self.selections,
                                                  parent=self.work_page)
        self.settings_dialog.show()
        self.settings_dialog.raise_()

    def save_settings(self, page, dialog) -> list[str]:
        """Write what the page validated, then hand the new objects to every later reader.

        The destination is the fork's own settings file -- `config_file()`, the same path
        `load()` read at start-up. `apply(path)` takes the destination precisely so a test
        can point the page somewhere else; nothing here writes a second format.

        A refusal means the dialog stays open with the message in the console, because the
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
                    "again -- the dialog stayed open for that")
            return messages
        self.settings, self.selections = page.st, page.sel
        self.work_page.selections = self.selections
        console(f"settings saved to {settings_module.config_file()}")
        dialog.close()
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
