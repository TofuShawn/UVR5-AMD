"""The composition root: the one place that sees catalog, settings, selections, Job.

Everything else in `app/` takes one of those as a given. This module owns the pairing:
it builds the `Catalog`, loads `Settings`/`Selections`, constructs the `WorkPage`, and
connects the page's `run_requested` to the single assembly path (`session.assemble`) and
the single worker (`page.job.start`). Two rules live here and nowhere else:

* **Exactly one `Job`** (ruling R-T4-onejob). It is the page's `job` attribute, built once
  when the page is built, and this module is the only place that starts it. No second
  `Job` in a helper, no process-wide registry -- the one-instance rule holds because this
  root constructs one page, and a reviewer checks it by grep.
* **One file per job.** A run is one audio file into one output directory, because that is
  what `session.run_job()` does; the page's percentage is honest only in that shape.
"""
from __future__ import annotations

from pathlib import Path

import catalog
import plan
import separate_cli
import selections as selections_module
import settings as settings_module
from app import job, session, work_page


class Controller:
    """Wires the page's two buttons to the assembly path and the worker thread."""

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
            # Tier 3: the model needs params nobody has supplied. Task 7 replaces this
            # dismissal with the params dialog; until then the catalog's own message is
            # shown whole, because it names the hash and the JSON it wanted.
            page.show_cancelled(str(exc))
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
