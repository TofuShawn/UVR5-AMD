"""The single page a user actually works on: pick, configure, run, watch.

Layout is a Fluent `Card` because a plain QWidget would keep the native palette while
its children follow the theme (spec §5.3). No NavigationView: the Python binding
exposes no item/selection API (spec §5.2), so the arch switch is a `SelectorBar` and
the settings page is a dialog opened from a button.

One file per job, and that is the whole scope of the percentage: `session.run_job()`
separates a single audio file and never passes `total_files`, and `Job.progress` carries
no file index, so the `ProgressBar` below is honest only for one file per job -- a
multi-file batch through this widget would reset per file and read as a saw-tooth. Batch
queueing is not added here (Task 4's carry, R-T4-iteration).
"""
from __future__ import annotations

from pathlib import Path

import backends
import catalog
import fluentqt
import selections as selections_module
import settings as settings_module
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QVBoxLayout

from app import job

#: arch -> the `Selections` field holding this arch's main model pick.
MAIN_MODEL_FIELD = {"vr": "vr_model", "mdx": "mdx_net_model", "demucs": "demucs_model",
                    "apollo": "apollo_model"}


def _classify_failure(text: str) -> str | None:
    """Map engine text to one of ``vram`` / ``missing_weight`` / ``cpu_fallback``, or
    `None` when the text says nothing certain.

    `None` is the point, not a fallback: a *cancel* is a separate signal
    (`Job.cancelled`), so guessing an unknown failure into a category would let Task 7's
    dialog label a crash as something the user asked for.

    The needles are the strings the engine actually emits: `torch.cuda.OutOfMemoryError`
    / `CUDA out of memory` from the ROCm wheel, `No such file or directory` from an
    `OSError` on a missing weight, and `backends.py`'s degradation notices (which say
    `Running on CPU` / `CPUExecutionProvider` while the machine asked for a GPU).
    """
    if "OutOfMemoryError" in text or "out of memory" in text:
        return "vram"
    if "No such file or directory" in text or "weight" in text:
        return "missing_weight"
    if "Running on CPU" in text or "CPUExecutionProvider" in text:
        return "cpu_fallback"
    return None


def device_summary(backend: backends.Backend) -> str:
    """One truthful line about the machine, built from `Backend` and nothing else.

    Deliberately not a plan-derived statement: `session.assemble()` prints the
    *compute* line separately because that is what the engine acts on
    (`separate.py:236-276` resolves device and providers at run time). Showing only
    this line would let a ROCm-capable machine that runs on CPU look like it computed
    on the GPU -- Plan 1's Critical, in GUI form.
    """
    return (f"{backend.name} | torch {backend.torch_device} | "
            f"ort {', '.join(backend.ort_providers)}")


class WorkPage(fluentqt.Card):
    run_requested = Signal()
    cancel_requested = Signal()

    def __init__(self, *, cat: catalog.Catalog, settings_obj, selections_obj, parent=None):
        super().__init__(parent)
        self.catalog = cat
        self.settings = settings_obj if settings_obj is not None else settings_module.load()
        self.selections = (selections_obj if selections_obj is not None
                           else selections_module.load())
        self.job = job.Job()
        self._build()
        self.refresh_models()
        self._sync_from_state()

    # -- construction ---------------------------------------------------------- #

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)

        self.device_label = fluentqt.Label("", self)
        self.compute_label = fluentqt.Label("", self)
        root.addWidget(self.device_label)
        root.addWidget(self.compute_label)

        self.arch_bar = fluentqt.SelectorBar(self)
        self._arch_names = self._archs()
        for arch in self._arch_names:
            self.arch_bar.addItem(fluentqt.SelectorBarItem(arch))   # text-only form:
            # SelectorBarItem('vr', parent) hits the same Shiboken overload defect as
            # Expander('title') (spec §5.2.1), so the parent is never passed here.
        self.arch_bar.currentChanged.connect(self._on_arch_changed)
        self.arch_bar.setSelectedIndex(0)
        root.addWidget(self.arch_bar)

        form = QVBoxLayout()
        root.addLayout(form)
        form.addWidget(fluentqt.Label("Model", self))
        self.model_combo = fluentqt.ComboBox(self)
        form.addWidget(self.model_combo)

        form.addWidget(fluentqt.Label("Input file", self))
        path_row = QHBoxLayout()
        self.input_edit = fluentqt.LineEdit(self)
        browse = fluentqt.Button("Browse", self)
        browse.clicked.connect(self._pick_input)
        path_row.addWidget(self.input_edit)
        path_row.addWidget(browse)
        form.addLayout(path_row)

        form.addWidget(fluentqt.Label("Output directory", self))
        out_row = QHBoxLayout()
        self.output_edit = fluentqt.LineEdit(self)
        out_btn = fluentqt.Button("Browse", self)
        out_btn.clicked.connect(self._pick_output)
        out_row.addWidget(self.output_edit)
        out_row.addWidget(out_btn)
        form.addLayout(out_row)

        self.params = fluentqt.Expander(self)                # string-first ctor raises
        self.params.setHeaderText("Model parameters")        # (spec §5.2.1) -- and there
        form.addWidget(self.params)                          # is no setTitle on Expander

        buttons = QHBoxLayout()
        self.start_button = fluentqt.Button("START", self)
        self.start_button.setFluentStyle(fluentqt.Button.Accent)
        self.stop_button = fluentqt.Button("STOP", self)
        self.stop_button.setEnabled(False)
        # START asks the controller for one run of one file; the percentage this page
        # then shows is that file's (`run_job` never sees a file index -- module
        # docstring). STOP only *asks*: cancellation is cooperative at the engine's own
        # batch granularity (`separate.py:1383`), and the busy state releases when the
        # terminal arrives, not when the button is pressed.
        self.start_button.clicked.connect(self.run_requested.emit)
        self.stop_button.clicked.connect(self.cancel_requested.emit)
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)
        root.addLayout(buttons)

        self.progress = fluentqt.ProgressBar(self)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        root.addWidget(self.progress)

        self.console = fluentqt.TextEdit(self)
        self.console.setReadOnly(True)
        self.console.setMaxVisibleLines(8)
        root.addWidget(self.console)

        self.job.progress.connect(self.progress.setValue)
        self.job.console.connect(self._on_line)
        self.job.succeeded.connect(self._on_done)
        self.job.failed.connect(self._on_failed)
        self.job.cancelled.connect(self._on_cancelled)

    def _archs(self) -> list[str]:
        return sorted(self.catalog.archs())

    def _sync_from_state(self) -> None:
        """Read the machine, change nothing.

        `backends.resolve(force=None)` -- **no argument** -- because that is the path a
        user who touches nothing takes, and it is the one Plan 1's review found untested.
        `compute_label` stays empty: that line is plan-derived and may not appear before
        a plan exists (the controller fills it via `show_compute`).
        """
        self.device_label.setText(device_summary(backends.resolve(force=None)))
        self.compute_label.setText("")

    # -- public surface -------------------------------------------------------- #

    def arch_names(self) -> list[str]:
        """Public because a test must be able to ask what the page offered; the page
        keeps its own list rather than reading item text back out of SelectorBar."""
        return list(self._arch_names)

    def current_arch(self) -> str:
        index = self.arch_bar.selectedIndex()
        if 0 <= index < len(self._arch_names):
            return self._arch_names[index]
        return self._arch_names[0]

    def selected_model(self) -> str:
        return self.model_combo.currentText()

    def refresh_models(self) -> None:
        arch = self.current_arch()
        names = self.catalog.known_names(arch)
        self.model_combo.clear()
        self.model_combo.addItems([catalog.NO_MODEL, *names])
        current = getattr(self.selections, MAIN_MODEL_FIELD[arch], catalog.NO_MODEL)
        self.model_combo.setCurrentText(current if current in names else catalog.NO_MODEL)

    def set_busy(self, busy: bool) -> None:
        self.start_button.setEnabled(not busy)
        self.stop_button.setEnabled(busy)

    def show_compute(self, lines: tuple[str, ...]) -> None:
        """Publish the plan-derived compute line. The controller calls this the moment
        `session.assemble()` returns, because `Session.compute_lines` is the only place
        those words come from -- and `compute_label` must stay empty until then.
        `describe_compute_device()` prefixes `[cli] `; the prefix is the CLI's convention
        for interleaving with stdout, so it is dropped here and nothing else changes.
        """
        self.compute_label.setText("\n".join(ln.removeprefix("[cli] ") for ln in lines))

    def show_cancelled(self, text: str) -> None:      # used by the tier-3 dismissal
        self.set_busy(False)
        self.console.appendPlainText(text)

    # -- slots ----------------------------------------------------------------- #

    def _on_arch_changed(self) -> None:
        self.refresh_models()

    def _pick_input(self) -> None:
        # Fluent has no file dialog in Python and `FluentMenu` is broken (spec §5.2),
        # so paths come from Qt's own dialog. `getOpenFileName` returns (text, filter).
        chosen, _ = QFileDialog.getOpenFileName(
            self, "Choose an audio file", self.input_edit.text() or str(Path.home()),
            "Audio (*.wav *.mp3 *.flac *.ogg *.m4a *.opus *.aac *.aiff *.wma)")
        if chosen:
            self.input_edit.setText(str(Path(chosen)))

    def _pick_output(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, "Choose an output directory",
            self.output_edit.text() or str(Path.home()))
        if chosen:
            self.output_edit.setText(str(Path(chosen)))

    def _on_line(self, text: str) -> None:
        self.console.appendPlainText(text)

    def _on_done(self, written: list) -> None:
        self.set_busy(False)
        for path in written:
            self.console.appendPlainText(f"wrote {path} ({path.stat().st_size} bytes)")
        self._notify("Done", f"{len(written)} file(s) written")

    def _on_failed(self, text: str) -> None:
        self.set_busy(False)
        # Task 5 has no failure vocabulary yet: show the engine's own words verbatim.
        # Task 7 replaces this body with dialogs.report_failure(_classify_failure(text)),
        # and `_classify_failure` stays here because it is a property of the page's text.
        self._notify("Failed", text)

    def _on_cancelled(self, text: str) -> None:
        self.set_busy(False)
        self.progress.setValue(0)
        self._notify("Stopped", text)

    def _notify(self, title: str, text: str):
        bar = fluentqt.InfoBar(parent=self)     # setTitle/setMessage/show are the real
        bar.setTitle(title)                      # names; InfoBar.show_message does not
        bar.setMessage(text)                     # exist -- pinned by Task 1.
        # Measured at 1.11.2: a parented InfoBar keeps Qt's 100x30 default size, which
        # clips both labels to an unreadable box in the corner. `adjustSize()` is the
        # binding's own size-to-content call (600x50 with these two labels), so the
        # message a user has to read is actually readable. Task 7 replaces this body.
        bar.adjustSize()
        bar.show()
        return bar
