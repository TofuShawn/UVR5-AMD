"""The single page a user actually works on: pick, configure, run, watch.

Layout is a Fluent `Card` because a plain QWidget would keep the native palette while
its children follow the theme (spec §5.3). No NavigationView: the Python binding
exposes no item/selection API (spec §5.2), so the arch switch is a `Pivot` and the
settings page is a dialog opened from a button. (It was a `SelectorBar` until ruling
R-T6-archtabs measured that `SelectorBar` ellipsizes the third label to `m…` at every
width -- see the note on the bar below.)

One file per job, and that is the whole scope of the percentage: `session.run_job()`
separates a single audio file and never passes `total_files`, and `Job.progress` carries
no file index, so the `ProgressBar` below is honest only for one file per job -- a
multi-file batch through this widget would reset per file and read as a saw-tooth. Batch
queueing is not added here (Task 4's carry, R-T4-iteration).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import backends
import catalog
import fluentqt
import selections as selections_module
import settings as settings_module
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout

from app import dialogs, job, settings_page

#: The page's own terminal vocabulary. These are the titles the terminal handlers speak
#: to `_notify` with, and the seam the suite spies on: `Stopped` *is* the cancel surface
#: (kind `cancelled`), and `Failed` is what a message the page cannot classify stays.
#: `Degraded` and `Warning` are the two halves the I4 warning path chooses between --
#: the first is spec §8's CPU-fallback surface, the second what a warning the failure
#: vocabulary has no category for gets. The widget's own title comes from `app/dialogs.py`.
DONE_TITLE = "Done"
FAILED_TITLE = "Failed"
CANCELLED_TITLE = "Stopped"
DEGRADED_TITLE = "Degraded"
WARNING_TITLE = "Warning"

#: arch -> the `Selections` field holding this arch's main model pick.
MAIN_MODEL_FIELD = {"vr": "vr_model", "mdx": "mdx_net_model", "demucs": "demucs_model",
                    "apollo": "apollo_model"}


def _classify_failure(text: str) -> str | None:
    """Map engine text to one of ``vram`` / ``missing_weight``, or `None` when the text
    says nothing certain.

    `None` is the point, not a fallback: a *cancel* is a separate signal
    (`Job.cancelled`), so guessing an unknown failure into a category would let Task 7's
    dialog label a crash as something the user asked for.

    Only the **first line** is read, because that is the only part that is the exception:
    `Job.failed`'s text is `f"{type}: {message}"` plus four frames of engine source
    (`app/job.py:313-314`), and `lib_v5/spec_utils.py:165-188` has
    `weight = np.zeros_like(...)` on a normal path -- so a bare `"weight"` substring of
    the whole text called an unrelated crash "Weight file not found" (M6).

    The third of the four abnormal states is not here any more. `cpu_fallback` used to
    be keyed on the needle `Running on CPU`, a string the shipped code emits nowhere
    (repo-wide it appeared only in the plan text and in a test fixture), while the real
    degradations arrive as `RuntimeWarning`s. They come in through `Job.warning` and are
    decided by `is_cpu_fallback()` against the machine, not against a sentence (I4).
    """
    first = text.splitlines()[0] if text else ""
    if "OutOfMemoryError" in first or "out of memory" in first.lower():
        return "vram"
    if ("No such file or directory" in first or "FileNotFoundError" in first
            or "weight" in first.lower()):
        return "missing_weight"
    return None


#: The part of `describe_compute_device()`'s refusal that is CLI advice, and what to say
#: instead. `separate_cli.py`'s bytes are the CLI-parity property spec §8 pins, so the
#: rewrite happens here, in the page's own translation layer, and nowhere else: this GUI
#: has no `--device` flag -- the affordance is the `gpu_conversion` checkbox the Settings
#: button opens (I5(b)). The clause is matched on the CLI's own wording so a change
#: there is visible here rather than silently un-translated;
#: `tests/test_work_page.py` pins both halves.
_CLI_ADVICE = ("pass --device rocm to use the GPU",
               "GPU conversion is off in Settings -- turn it on there to use the GPU")


def translate_compute_line(line: str) -> str:
    """One `compute device:` line, as a GUI user should read it."""
    line = line.removeprefix("[cli] ")
    if _CLI_ADVICE[0] in line:
        line = line.replace(_CLI_ADVICE[0], _CLI_ADVICE[1])
    return line


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
    settings_requested = Signal()

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

        self.arch_bar = fluentqt.Pivot(self)
        self._arch_names = self._archs()
        for arch in self._arch_names:
            self.arch_bar.addItem(fluentqt.PivotItem(arch))   # header-only form:
            # PivotItem('vr', parent) hits the same Shiboken overload defect as
            # Expander('title') (spec §5.2.1), so the parent is never passed here.
        # Ruling R-T6-archtabs. This bar was a `SelectorBar` and Task 5's launch
        # photograph showed the third arch label ellipsized to `m…` at the 760 px
        # default. Measured on the pinned wheel (four probes, offscreen, comparing the
        # *painted* pixels to `fontMetrics.horizontalAdvance`, not to `itemGeometry`):
        # `SelectorBar` elides `mdx` to `m…` at 520, 728, 1200 and 2000 px alike -- the
        # elision is a rule of the control itself, so sizing it to content, bounding its
        # width or widening the window cannot reach it. Its `itemGeometry` reports a 60 px
        # rect for a 23 px label and still clips, which is why the geometry numbers are
        # untrustworthy here. `Pivot` (already in the reviewed `UI_CLASSES`) paints all
        # four labels in full at every width tested -- its `itemHeaderGeometry` matches
        # the paint, so the geometry *is* honest for this control. The arch switch needs
        # only headers plus `currentChanged`/`selectedIndex`; the model list lives in the
        # combo below, so `Pivot`'s lack of a content setter (which rules it out for the
        # settings sections) costs nothing here. The two sizing lines keep the bar at its
        # own content height (44 px, not the 158 px band Task 5 photographed) and give it
        # at least its content width; the labels stay `Catalog.archs()`' own strings --
        # nothing abbreviated, filtered or renamed.
        self.arch_bar.setFixedHeight(self.arch_bar.sizeHint().height())
        self.arch_bar.setMinimumWidth(self.arch_bar.sizeHint().width())
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
        self.browse_input = fluentqt.Button("Browse", self)
        self.browse_input.clicked.connect(self._pick_input)
        path_row.addWidget(self.input_edit)
        path_row.addWidget(self.browse_input)
        form.addLayout(path_row)

        form.addWidget(fluentqt.Label("Output directory", self))
        out_row = QHBoxLayout()
        self.output_edit = fluentqt.LineEdit(self)
        self.browse_output = fluentqt.Button("Browse", self)
        self.browse_output.clicked.connect(self._pick_output)
        out_row.addWidget(self.output_edit)
        out_row.addWidget(self.browse_output)
        form.addLayout(out_row)

        # M1. This Expander was Task 5's placeholder -- "no content until Task 6" -- and
        # Task 6 put the fields on the settings page without removing it, so the primary
        # page carried a control that opened onto nothing and no test named it. Spec §10
        # assigns the arch's engine parameters to the work page, so what it opens onto now
        # is a read-only summary of the arch's own parameter group plus the one action that
        # goes where they are editable. One editable surface, no dead widget, and §10's
        # promise visibly honoured instead of duplicated. `string-first ctor raises`
        # (spec §5.2.1) and there is no `setTitle` on Expander, so the header is set after.
        self.params = fluentqt.Expander(self)
        self.params.setHeaderText("Model parameters")
        self.params.setAnimationEnabled(False)
        self.params_host = fluentqt.Card()                # a widget, never a bare QWidget
        params_form = QVBoxLayout(self.params_host)       # (spec §5.3, Task 5's guard)
        self.params_summary = fluentqt.Label("", self.params_host)
        self.params_edit = fluentqt.Button("Edit these in Settings", self.params_host)
        self.params_edit.clicked.connect(self.settings_requested.emit)
        params_form.addWidget(self.params_summary)
        params_form.addWidget(self.params_edit)
        self.params.setContentWidget(self.params_host)
        form.addWidget(self.params)

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
        # The one entry to the settings page. It asks the controller for the dialog and
        # does nothing else: the page itself (`app/settings_page.py`) owns the fields, and
        # this page keeps no copy of them -- `Controller` hands it the same two objects it
        # hands `session.assemble()`.
        self.settings_button = fluentqt.Button("Settings", self)
        self.settings_button.clicked.connect(self.settings_requested.emit)
        buttons.addWidget(self.settings_button)
        root.addLayout(buttons)

        # Ruling R-T7-infobar. The band is a *layout*, not a widget: a plain QWidget here
        # would keep the native palette (spec §5.3) and the surface census would rightly
        # refuse it. A layout sibling of the console also cannot overlap it, which is the
        # property the ruling asks for -- Task 5's bars laid out at (0,0) over the content,
        # none ever closed, and a run with several left a pile that hid the output. An
        # empty nested layout has no size, so a page with nothing to say pays nothing.
        # `app/dialogs.py`'s queue puts at most one bar in it at a time.
        self.notification_band = QVBoxLayout()
        self.notification_band.setContentsMargins(0, 0, 0, 0)
        root.addLayout(self.notification_band)      # between the buttons and the bar
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
        self.job.warning.connect(self._on_warning)
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

        The `Backend` object is kept, because `is_cpu_fallback()` is the only honest
        producer the degraded-GPU surface has: the degradation this machine can be in is
        a *state*, and it was being inferred from a sentence (I4). The `catch_warnings`
        block around the probe is for the notices `backends.resolve()` itself raises --
        `_probe_failed` when WMI refuses to say whether the card is AMD's, for instance:
        at start-up there is no worker thread to carry them, and nothing else in the app
        listens, so they would print to stderr and vanish. Scoped, so no global hook is
        left installed for a test that runs after this one.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._backend = backends.resolve(force=None)
        for notice in caught:
            self._on_warning(f"{notice.category.__name__}: {notice.message}")
        self.device_label.setText(device_summary(self._backend))
        self.compute_label.setText("")

    def is_cpu_fallback(self) -> bool:
        """True when the machine offers a GPU backend but ONNX Runtime only has the CPU
        provider, which is the reviewer's condition and the one `backends.resolve()` can
        actually answer (`backends.py`'s `resolve`: the vendor scan or the provider list
        can both fail, and when they do the run is correct and far slower).

        Deliberately not derived from any text: a warning that *says* CPU on a machine
        that is honestly CPU-only has not fallen back from anything.
        """
        backend = getattr(self, "_backend", None)
        if backend is None:
            return False
        return (backend.name != "cpu" and backend.torch_device != "cpu"
                and list(backend.ort_providers) == ["CPUExecutionProvider"])

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
        self.refresh_parameter_summary()

    def parameter_summary(self) -> str:
        """The selected arch's engine parameters and what each one currently is.

        Read-only, and derived: the *names* are `settings_page.ARCH_FIELD_GROUPS`' own
        tuple for this arch -- the same table the editable page builds its widgets from,
        so the two cannot drift apart -- and the *values* come from the `Settings` object
        the controller assembles runs with, which is why `Controller.save_settings` hands
        this page the new one. An arch the table has no group for (apollo) is said so,
        rather than shown another arch's numbers.
        """
        arch = self.current_arch()
        group = settings_page.ARCH_FIELD_GROUPS.get(arch)
        if group is None:
            return (f"{arch} reads no per-arch engine parameters; the general group's "
                    f"format and device choices are what apply to it.")
        lines = [f"{arch}: {len(group)} engine parameters (read-only here -- use the "
                 f"button below)"]
        lines += [f"{name} = {getattr(self.settings, name, '?')!r}" for name in group]
        return "\n".join(lines)

    def refresh_parameter_summary(self) -> None:
        """Re-publish `parameter_summary()` -- on an arch change and after a settings
        save, the two moments the numbers can move under the page."""
        self.params_summary.setText(self.parameter_summary())

    def set_busy(self, busy: bool) -> None:
        self.start_button.setEnabled(not busy)
        self.stop_button.setEnabled(busy)

    def show_compute(self, lines: tuple[str, ...]) -> None:
        """Publish the plan-derived compute line. The controller calls this the moment
        `session.assemble()` returns, because `Session.compute_lines` is the only place
        those words come from -- and `compute_label` must stay empty until then. An empty
        tuple is how a refusal clears the line again (I5(a)), and `translate_compute_line`
        drops the CLI's `[cli] ` prefix and its `--device` advice, which is the GUI's
        translation layer and the only place either of those is rewritten.
        """
        self.compute_label.setText("\n".join(translate_compute_line(ln) for ln in lines))

    def show_cancelled(self, text: str) -> None:      # used by the tier-3 dismissal
        self.set_busy(False)
        self.console.appendPlainText(text)

    # -- slots ----------------------------------------------------------------- #

    def _on_arch_changed(self) -> None:
        self.refresh_models()

    def _pick_input(self) -> None:
        # Fluent has no file dialog in Python and `FluentMenu` is broken (spec §5.2),
        # so paths come from Qt's own dialog -- through `dialogs.Pickers`, which is the
        # one place that says which dialog, with what caption and what filter.
        chosen = dialogs.Pickers.input_file(
            self, self.input_edit.text() or str(Path.home()))
        if chosen:
            self.input_edit.setText(str(Path(chosen)))

    def _pick_output(self) -> None:
        chosen = dialogs.Pickers.output_dir(
            self, self.output_edit.text() or str(Path.home()))
        if chosen:
            self.output_edit.setText(str(Path(chosen)))

    def _on_line(self, text: str) -> None:
        self.console.appendPlainText(text)

    def _on_warning(self, text: str) -> None:
        """A degradation the engine warned about, said twice in the only two ways a user
        can be told: the console keeps it, the band points at it.

        Whether it is worth the *degraded* surface is `is_cpu_fallback()`'s answer about
        the machine, not a match against the sentence -- which is what spec §8's fourth
        abnormal state was missing a producer for (I4). Anything else the engine warns
        about is still a warning, so it gets `notify`'s plain surface rather than being
        dropped or dressed up as one of the four.
        """
        self.console.appendPlainText(f"warning: {text}")
        self._notify(DEGRADED_TITLE if self.is_cpu_fallback() else WARNING_TITLE, text)

    def _on_done(self, written: list) -> None:
        self.set_busy(False)
        for path in written:
            self.console.appendPlainText(f"wrote {path} ({path.stat().st_size} bytes)")
        self._notify(DONE_TITLE, f"{len(written)} file(s) written")

    def _on_failed(self, text: str) -> None:
        self.set_busy(False)
        # The console is the permanent record and the surface is the one that gets your
        # attention: both get the engine's own words, never a paraphrase. Task 5 showed
        # these lines in the console only; the category comes from `_classify_failure`,
        # which answers None for text nobody recognises -- and that stays `Failed`.
        self.console.appendPlainText(f"failed: {text}")
        self._notify(FAILED_TITLE, text)

    def _on_cancelled(self, text: str) -> None:
        self.set_busy(False)
        self.progress.setValue(0)
        self._notify(CANCELLED_TITLE, text)

    def _notify(self, title: str, text: str):
        """Hand a terminal message to the queue in `app/dialogs.py`.

        Ruling R-T7-infobar. This is the page's one notification seam, and the queue is
        what makes it honest: `InfoBar` has no auto-dismiss on this wheel (no
        `show_message`, no `setDuration` -- pinned by Task 1), so without a queue a bar
        landed at (0,0) over the content and stayed for ever. One bar goes in
        `notification_band` at a time, and it goes away on its own or when the user
        presses its close button.

        The page's title chooses the surface, and only from what the page actually knows:
        a cancel is the `cancelled` kind because it arrived on `Job.cancelled` (a signal,
        never a guess); a failure is one of the two classified kinds only if its own
        first line says so, and otherwise keeps the plain `Failed` surface with the
        engine's words as the message. `cpu_fallback` is not reached from here at all --
        it is `_on_warning`'s, because that state is a warning and not a crash.
        """
        kind = None
        if title == CANCELLED_TITLE:
            kind = "cancelled"
        elif title == FAILED_TITLE:
            kind = _classify_failure(text)
        elif title == DEGRADED_TITLE:
            kind = "cpu_fallback"
        if kind is None:
            return dialogs.notify(title, text, parent=self)
        return dialogs.report_failure(kind, text, parent=self)
