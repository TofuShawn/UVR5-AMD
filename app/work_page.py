"""The single page a user actually works on: pick, configure, run, watch.

Layout is a Fluent `Card` because a plain QWidget would keep the native palette while
its children follow the theme (spec §5.3). No NavigationView: the Python binding
exposes no item/selection API (spec §5.2), so the arch switch is a `Pivot` and the
settings page is a full-page view swapped in from the window's right rail
(`app/shell.py`). (It was a `SelectorBar` until ruling
R-T6-archtabs measured that `SelectorBar` ellipsizes the third label to `m…` at every
width -- see the note on the bar below.)

A batch runs through this page as a queue the root walks one file at a time, and the
percentage it publishes is the queue's, not the file's: `session.run_job()` still separates
one audio file per call and never passes `total_files`, and `Job.progress` carries no file
index, so the bar is `(files done + this file's fraction) / total` -- see `_on_progress`.
The claim this module used to make ("a multi-file batch through this widget would reset per
file and read as a saw-tooth") was true of the raw signal and is the reason the page
aggregates instead of forwarding it. What a file is *on* is `count_label`'s job, and what
stops the run is `blockers()`'.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import backends
import catalog
import fluentqt
import selections as selections_module
import settings as settings_module
import ui_prefs
from PySide6.QtCore import Qt, Signal
# `QButtonGroup` is a QObject, not a widget: the surface census walks widgets, so it never
# sees this, and `BARE_QT_CONSTRUCTIBLES` (QWidget/QDialog/QFrame) does not cover it. It is
# here because the format radios need an exclusivity guarantee the binding's parent-based
# auto-exclusive behaviour does not provide -- see the comment at their construction.
from PySide6.QtWidgets import QButtonGroup, QHBoxLayout, QVBoxLayout

from app import dialogs, job, session, settings_page
from app.i18n import LANGUAGES, load_language, save_language, tr

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


def audio_files_in(folder: Path) -> list[Path]:
    """Every audio file under `folder`, recursively, in path order.

    Matched against `dialogs.AUDIO_EXTENSIONS`, which is the app's pinned copy of the
    engine's `gui_data.constants.FFMPEG_EXT` -- the same list the file dialog's filter is
    built from, so a folder pick cannot accept a file the multi-select dialog would have
    hidden. Nothing here filters by content, by mtime or by decodability: a file that is
    on the list is queued, and if the engine cannot read it that is a per-file failure the
    batch reports and moves on from.
    """
    return sorted(p for p in Path(folder).rglob("*")
                  if p.is_file() and p.suffix.lower() in dialogs.AUDIO_EXTENSIONS)


class WorkPage(fluentqt.Card):
    run_requested = Signal()
    cancel_requested = Signal()
    settings_requested = Signal()
    #: The default output folder, asked for as text. A signal and not a `save()` call
    #: here because the page is not the thing that owns the settings file: the composition
    #: root writes it, reloads it, and hands the page back what the disk actually says --
    #: so a `settings.json` that refused to be written cannot leave this checkbox showing
    #: a default that was never stored. See `Controller._on_default_output_requested`.
    default_output_requested = Signal(str)
    #: The two `Settings` fields that moved onto this page, asked for as the values they
    #: are. Neither is written by the widget that shows it: the root applies it to the
    #: frozen `Settings`, saves it, and hands the control back the truth on a refusal --
    #: the same seam `default_output_requested` documents above, and for the same reason.
    format_requested = Signal(str)
    gpu_requested = Signal(bool)

    def __init__(self, *, cat: catalog.Catalog, settings_obj, selections_obj,
                 ui_obj: ui_prefs.UiPrefs | None = None, parent=None):
        super().__init__(parent)
        self.catalog = cat
        self.settings = settings_obj if settings_obj is not None else settings_module.load()
        self.selections = (selections_obj if selections_obj is not None
                           else selections_module.load())
        # `ui_prefs.load()` rather than a default-constructed `UiPrefs()`: the whole point
        # of the block is that a user's configured folder is there when the window opens.
        self.ui = ui_obj if ui_obj is not None else ui_prefs.load()
        # The batch queue's state, and all four before `_build()`: `refresh_queue()` is
        # called from inside it and reads every one of them. `_extra_files` is the queue
        # minus the typed file -- see `pending_files()` for why the typed one is not in
        # the list as well.
        self._extra_files: list[Path] = []
        self._files_done = 0
        self._batch_total = 0
        self._batch_active = False
        # START's other half: `set_busy` and `refresh_queue` both write that button, so
        # both read this flag. See `set_busy`.
        self._busy = False
        self.job = job.Job()
        self._build()
        self.refresh_models()
        self._sync_from_state()
        self._apply_mono()

    # -- construction ---------------------------------------------------------- #

    def _build(self) -> None:
        """One column of expandable groups of setting rows (Wintoys/Windows Settings).

        Each row is a leading icon, a title + secondary description, and the control on the
        right; the groups (Input / Model & output / Device) collapse. Same widget attributes
        and signals as the earlier two-column form -- behaviour unchanged, arrangement only.
        """
        self.setAppearance(fluentqt.Card.Appearance.Canvas)
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(14)

        masthead = QHBoxLayout()
        masthead.setSpacing(12)
        self.app_title = fluentqt.Label("UVR5 PySide6 GUI", self)
        self.app_title.setFluentTypography(fluentqt.FontRole.Title)
        masthead.addWidget(self.app_title)
        self.fork_label = self._caption(tr("fork_badge"), fluentqt.Label.TextColorRole.Tertiary)
        masthead.addWidget(self.fork_label)
        masthead.addStretch(1)
        self.orientation_label = self._caption(tr("orientation"),
                                               fluentqt.Label.TextColorRole.Secondary)
        self.orientation_label.setWordWrap(True)
        masthead.addWidget(self.orientation_label)
        root.addLayout(masthead)
        machine_row = QHBoxLayout()
        machine_row.setSpacing(8)
        self.device_label = self._caption("", fluentqt.Label.TextColorRole.Tertiary)
        self.compute_label = self._caption("", fluentqt.Label.TextColorRole.Secondary)
        machine_row.addWidget(self.device_label)
        machine_row.addWidget(self.compute_label, 1)
        root.addLayout(machine_row)

        columns = QHBoxLayout()
        columns.setSpacing(12)
        left = QVBoxLayout()
        right = QVBoxLayout()
        columns.addLayout(left, 1)
        columns.addLayout(right, 1)

        def group(container, title):
            exp = fluentqt.Expander(self)
            exp.setHeaderText(title)
            exp.setAnimationEnabled(False)
            body = fluentqt.Card()
            col = QVBoxLayout(body)
            col.setContentsMargins(16, 6, 16, 12)
            col.setSpacing(8)
            exp.setContentWidget(body)
            exp.setExpanded(True)
            container.addWidget(exp)
            return col

        def row(col, icon, title, desc, *controls):
            bar = QHBoxLayout()
            bar.setContentsMargins(4, 8, 4, 8)
            bar.setSpacing(12)
            ic = fluentqt.Button("")
            ic.setIconGlyph(icon)
            ic.setFluentStyle(fluentqt.Button.ButtonStyle.Subtle)
            ic.setFixedSize(32, 32)
            ic.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            bar.addWidget(ic)
            text = QVBoxLayout()
            text.setSpacing(2)
            lead = fluentqt.Label(title)
            lead.setFluentTypography(fluentqt.FontRole.BodyStrong)
            text.addWidget(lead)
            if desc:
                sub = fluentqt.Label(desc)
                sub.setFluentTypography(fluentqt.FontRole.Caption)
                sub.setTextColorRole(fluentqt.Label.TextColorRole.Secondary)
                text.addWidget(sub)
            bar.addLayout(text, 1)
            for control in controls:
                if isinstance(control, (QHBoxLayout, QVBoxLayout)):
                    bar.addLayout(control)
                else:
                    bar.addWidget(control)
            col.addLayout(bar)
            col.addWidget(fluentqt.Divider())
            return bar

        # -- Input ------------------------------------------------------------- #
        g = group(left, tr("group_input"))
        self.input_edit = fluentqt.LineEdit()
        self.input_edit.setPlaceholderText(tr("input_ph"))
        self.input_edit.setClearButtonEnabled(True)
        self.browse_input = self._icon_button(tr("browse"), fluentqt.Icons.Folder, None)
        self.browse_input.clicked.connect(self._pick_input)
        path_row = QHBoxLayout()
        path_row.setSpacing(8)
        path_row.addWidget(self.input_edit, 1)
        path_row.addWidget(self.browse_input)
        row(g, fluentqt.Icons.File, tr("input_file"), tr("row_input_desc"), path_row)

        self.add_files_button = self._icon_button(tr("add_files"), fluentqt.Icons.Add, None)
        self.add_folder_button = self._icon_button(tr("add_folder"), fluentqt.Icons.Folder, None)
        self.clear_files_button = self._icon_button(tr("clear"), fluentqt.Icons.Delete, None,
                                                    style=fluentqt.Button.ButtonStyle.Subtle)
        self.add_files_button.clicked.connect(self._pick_more_inputs)
        self.add_folder_button.clicked.connect(self._pick_input_folder)
        self.clear_files_button.clicked.connect(lambda: self.add_extra_files([]))
        batch_row = QHBoxLayout()
        batch_row.setSpacing(8)
        batch_row.addWidget(self.add_files_button)
        batch_row.addWidget(self.add_folder_button)
        batch_row.addWidget(self.clear_files_button)
        row(g, fluentqt.Icons.Add, tr("row_batch"), tr("row_batch_desc"), batch_row)
        self.file_list = fluentqt.TextEdit()
        self.file_list.setReadOnly(True)
        self.file_list.setFontRole(fluentqt.FontRole.Caption)
        self.file_list.setPlaceholderText(tr("queue_ph"))
        self.file_list.setMinVisibleLines(3)
        self.file_list.setMaxVisibleLines(6)
        g.addWidget(self.file_list)

        self.output_edit = fluentqt.LineEdit()
        self.output_edit.setPlaceholderText(tr("output_ph"))
        self.output_edit.setClearButtonEnabled(True)
        self.browse_output = self._icon_button(tr("browse"), fluentqt.Icons.Folder, None)
        self.browse_output.clicked.connect(self._pick_output)
        out_row = QHBoxLayout()
        out_row.setSpacing(8)
        out_row.addWidget(self.output_edit, 1)
        out_row.addWidget(self.browse_output)
        row(g, fluentqt.Icons.Folder, tr("output_dir"), tr("row_output_desc"), out_row)
        self.default_output_check = fluentqt.CheckBox(tr("default_output_check"))
        self.default_output_check.setChecked(bool(self.ui.export_path))
        self.default_output_check.toggled.connect(self._on_default_output_toggled)
        g.addWidget(self.default_output_check)
        self.default_output_label = self._caption("", fluentqt.Label.TextColorRole.Tertiary)
        g.addWidget(self.default_output_label)

        # -- Model & output ---------------------------------------------------- #
        g = group(right, tr("group_model"))
        self.arch_bar = fluentqt.Pivot()
        self._arch_names = self._archs()
        for arch in self._arch_names:
            self.arch_bar.addItem(fluentqt.PivotItem(arch))
        self.arch_bar.setFixedHeight(self.arch_bar.sizeHint().height())
        self.arch_bar.setMinimumWidth(self.arch_bar.sizeHint().width())
        self.arch_bar.currentChanged.connect(self._on_arch_changed)
        self.arch_bar.setSelectedIndex(0)
        row(g, fluentqt.Icons.Grid, tr("architecture"), tr("row_arch_desc"), self.arch_bar)
        self.model_combo = fluentqt.ComboBox()
        row(g, fluentqt.Icons.Music, tr("model"), tr("row_model_desc"), self.model_combo)
        self.model_hint = self._caption("", fluentqt.Label.TextColorRole.Tertiary)
        self.model_hint.setWordWrap(True)
        g.addWidget(self.model_hint)
        self.format_radios = {}
        format_group = QButtonGroup(self)
        format_group.setExclusive(True)
        self.format_group = format_group
        format_row = QHBoxLayout()
        format_row.setSpacing(16)
        for fmt in session.SAVE_FORMATS:
            radio = fluentqt.RadioButton(fmt)
            radio.blockSignals(True)
            radio.setChecked(fmt == self.settings.save_format)
            radio.blockSignals(False)
            radio.toggled.connect(lambda on, f=fmt: on and self.format_requested.emit(f))
            format_group.addButton(radio)
            self.format_radios[fmt] = radio
            format_row.addWidget(radio)
        row(g, fluentqt.Icons.Save, tr("output_format"), tr("row_format_desc"), format_row)

        # -- Device ------------------------------------------------------------ #
        g = group(right, tr("group_device"))
        self.gpu_check = fluentqt.CheckBox(tr("gpu_check"))
        self.gpu_check.blockSignals(True)
        self.gpu_check.setChecked(bool(self.settings.gpu_conversion))
        self.gpu_check.blockSignals(False)
        self.gpu_check.toggled.connect(self.gpu_requested.emit)
        self.gpu_icon = fluentqt.Button("")
        self.gpu_icon.setIconGlyph(fluentqt.Icons.Desktop)
        self.gpu_icon.setFluentStyle(fluentqt.Button.ButtonStyle.Subtle)
        self.gpu_icon.setFixedSize(28, 28)
        self.gpu_icon.setToolTip(tr("gpu_check"))
        self.gpu_icon.clicked.connect(self.gpu_check.toggle)
        self.device_combo = fluentqt.ComboBox()
        self.device_combo.setToolTip(tr("compute_device"))
        self._device_codes = [None, "rocm", "dml", "cpu"]
        self.device_combo.addItems(["Auto", "ROCm", "DirectML", "CPU"])
        self.device_combo.setCurrentIndex(0)
        device_row = QHBoxLayout()
        device_row.setSpacing(8)
        device_row.addWidget(self.gpu_check)
        device_row.addWidget(self.device_combo)
        row(g, fluentqt.Icons.Desktop, tr("row_device"), tr("row_device_desc"), device_row)

        self.params = fluentqt.Expander(self)
        self.params.setHeaderText(tr("params_header"))
        self.params.setAnimationEnabled(True)
        self.params_host = fluentqt.Card()
        params_form = QVBoxLayout(self.params_host)
        params_form.setContentsMargins(12, 10, 12, 12)
        params_form.setSpacing(8)
        self.params_summary = self._caption("", fluentqt.Label.TextColorRole.Secondary,
                                            parent=self.params_host)
        self.params_summary.setWordWrap(True)
        self.params_edit = self._icon_button(tr("params_edit"), fluentqt.Icons.Edit,
                                             self.params_host)
        self.params_edit.clicked.connect(self.settings_requested.emit)
        params_form.addWidget(self.params_summary)
        params_form.addWidget(self.params_edit)
        self.params.setContentWidget(self.params_host)
        right.addWidget(self.params)
        left.addStretch(1)
        right.addStretch(1)
        root.addLayout(columns)

        # -- the run band ------------------------------------------------------ #
        self.action_row_host = fluentqt.Card(self)
        self.action_row_host.setAppearance(fluentqt.Card.Appearance.LayerAlt)
        self.action_row_host.setBorderVisible(True)
        band = QVBoxLayout(self.action_row_host)
        band.setContentsMargins(16, 14, 16, 14)
        band.setSpacing(10)
        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        self.start_button = self._icon_button(tr("start"), fluentqt.Icons.Play,
                                              self.action_row_host,
                                              style=fluentqt.Button.ButtonStyle.Accent,
                                              size=fluentqt.Button.ButtonSize.Large)
        self.stop_button = self._icon_button(tr("stop"), fluentqt.Icons.Stop,
                                             self.action_row_host)
        self.stop_button.setCriticalOnHover(True)
        self.start_button.clicked.connect(self.run_requested.emit)
        self.stop_button.clicked.connect(self.cancel_requested.emit)
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)
        self.language_combo = fluentqt.ComboBox(self.action_row_host)
        self.language_combo.setToolTip(tr("language"))
        self._lang_codes = ["system", *[code for code, _name in LANGUAGES]]
        names = [tr("lang_system"), *[name for _code, name in LANGUAGES]]
        self.language_combo.blockSignals(True)
        self.language_combo.addItems(names)
        saved = load_language()
        self.language_combo.setCurrentIndex(
            self._lang_codes.index(saved) if saved in self._lang_codes else 0)
        self.language_combo.blockSignals(False)
        self.language_combo.currentIndexChanged.connect(self._on_language_changed)
        buttons.addWidget(self.language_combo)
        buttons.addStretch(1)
        self.count_label = fluentqt.Label("", self.action_row_host)
        self.count_label.setFluentTypography(fluentqt.FontRole.BodyStrong)
        buttons.addWidget(self.count_label)
        band.addLayout(buttons)
        self.reason_label = self._caption("", fluentqt.Label.TextColorRole.Secondary,
                                          parent=self.action_row_host)
        self.reason_label.setWordWrap(True)
        band.addWidget(self.reason_label)
        self.progress = fluentqt.ProgressBar(self.action_row_host)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTrackThickness(8)
        self.progress.setRailVisible(True)
        progress_row = QHBoxLayout()
        progress_row.setSpacing(10)
        progress_row.addWidget(self.progress, 1)
        self.progress_label = self._caption("0%", fluentqt.Label.TextColorRole.Secondary,
                                            parent=self.action_row_host)
        self.progress_label.setMinimumWidth(40)
        progress_row.addWidget(self.progress_label)
        band.addLayout(progress_row)
        self.summary_label = self._caption("", fluentqt.Label.TextColorRole.Secondary,
                                           parent=self.action_row_host)
        self.summary_label.setWordWrap(True)
        band.addWidget(self.summary_label)
        root.addWidget(self.action_row_host)

        self.notification_band = QVBoxLayout()
        self.notification_band.setContentsMargins(0, 0, 0, 0)
        root.addLayout(self.notification_band)
        root.addWidget(self._field_label(tr("console"), self))
        self.console = fluentqt.TextEdit(self)
        self.console.setReadOnly(True)
        self.console.setFontRole(fluentqt.FontRole.Caption)
        self.console.setPlaceholderText(tr("console_ph"))
        self.console.setMinVisibleLines(4)
        self.console.setMaxVisibleLines(8)
        root.addWidget(self.console)
        root.addStretch(1)

        self.job.progress.connect(self._on_progress)
        self.job.console.connect(self._on_line)
        self.job.warning.connect(self._on_warning)
        self.job.succeeded.connect(self._on_done)
        self.job.failed.connect(self._on_failed)
        self.job.cancelled.connect(self._on_cancelled)

        self.input_edit.textChanged.connect(lambda _text: self.refresh_queue())
        self.model_combo.currentTextChanged.connect(lambda _text: self.refresh_queue())
        self.output_edit.textChanged.connect(lambda _text: self.refresh_queue())
        self.refresh_default_output()
        self.refresh_queue()

    # -- the page's typographic voice ----------------------------------------- #
    # Three label helpers for the three roles that recur a dozen times -- heading, field
    # label, caption -- because a page where every label is the default body ink is the
    # flat wall this pass exists to fix; a fourth puts the same glyph-before-text shape on
    # every action button. They set Fluent's *own* roles and colour roles -- no token
    # overrides (spec §5.3 ruling A) -- so the hierarchy follows the system theme instead
    # of fighting it.

    def _heading(self, text: str, parent) -> fluentqt.Label:
        lab = fluentqt.Label(text, parent)
        lab.setFluentTypography(fluentqt.FontRole.Subtitle)
        return lab

    def _field_label(self, text: str, parent) -> fluentqt.Label:
        lab = fluentqt.Label(text, parent)
        lab.setFluentTypography(fluentqt.FontRole.Caption)
        lab.setTextColorRole(fluentqt.Label.TextColorRole.Secondary)
        return lab

    def _caption(self, text: str, color, *, parent=None) -> fluentqt.Label:
        lab = fluentqt.Label(text, parent if parent is not None else self)
        lab.setFluentTypography(fluentqt.FontRole.Caption)
        lab.setTextColorRole(color)
        return lab

    def _icon_button(self, text: str, glyph, parent, *, style=None, size=None) -> fluentqt.Button:
        btn = fluentqt.Button(text, parent)
        btn.setIconGlyph(glyph)
        btn.setFluentLayout(fluentqt.Button.ButtonLayout.IconBefore)
        if style is not None:
            btn.setFluentStyle(style)
        if size is not None:
            btn.setFluentSize(size)
        return btn

    def _apply_mono(self) -> None:
        """The data/status lines read as data: switch them to the bundled monospace face
        (the machine line, the truth strip, the queue count and the console)."""
        from PySide6.QtGui import QFont
        for widget in (self.device_label, self.compute_label, self.summary_label,
                       self.count_label, self.console, self.file_list):
            font = widget.font()
            font.setFamilies(["JetBrains Mono", "Noto Sans SC"])
            widget.setFont(font)

    def _on_language_changed(self, index: int) -> None:
        """Persist the choice; the running page keeps its language until relaunch."""
        save_language(self._lang_codes[index])
        self.console.appendPlainText(tr("lang_restart"))

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
        # The first moment the summary can be honest, because it is the first moment the
        # machine is known. `_build()`'s `refresh_queue()` deliberately skipped it.
        self.refresh_summary()

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

    def selected_force_device(self) -> "str | None":
        """The compute-backend override for the next run; Auto means let resolve pick."""
        return self._device_codes[self.device_combo.currentIndex()]

    # -- the queue ------------------------------------------------------------- #

    def add_extra_files(self, paths: list[str]) -> None:
        """Set the queue behind the typed file. The pickers' answer, or `[]` to clear.

        A replacement and not an append, because "Add folder…" over a list of 400 stale
        paths has to be able to win: `_pick_more_inputs` passes the union it read back out
        of `pending_files()`, so the append lives in the caller that knows what it is
        adding to.
        """
        self._extra_files = [Path(p) for p in paths]
        self.refresh_queue()

    def pending_files(self) -> list[Path]:
        """What START would run: the typed file first, then the extras, deduped, in order.

        `input_edit` stays file #1 rather than joining the list because that is where the
        single-file case has always been, every existing test in this suite sets it the way
        a user types it, and a second holder of the same path is how one file comes to run
        twice. Deduped on `Path` identity, so re-adding a file from a folder pick that
        overlaps the typed one is a no-op rather than a duplicate stem written twice.
        """
        out: list[Path] = []
        typed = self.input_edit.text().strip()
        if typed:
            out.append(Path(typed))
        out.extend(self._extra_files)
        seen: set[Path] = set()
        return [p for p in out if not (p in seen or seen.add(p))]

    def blockers(self) -> list[str]:
        """Why START would refuse right now, in the words START uses.

        One table, two readers: this list is what `Controller._on_run` refuses with and
        what `reason_label` shows, and that is the whole reason it exists. The checks used
        to live in the root and the page could only guess at them, which is how an
        explanation and a refusal came to disagree -- and the everyday form of that
        disagreement is the button that looks like it did nothing. The strings are the
        pre-existing refusal's, verbatim, so nothing a user read last week changed wording.

        Existence is checked per file because a batch is usually a folder picked last
        month, and a missing path is the thing the user has to see in order to fix it --
        named up to three deep, then counted, because 400 of them must not print 400 paths
        into a label.
        """
        out: list[str] = []
        if self.selected_model() == catalog.NO_MODEL:
            out.append(tr("blocker_model"))
        files = self.pending_files()
        if not files:
            out.append(tr("blocker_input"))
            return out
        missing = [str(p) for p in files if not p.is_file()]
        if missing:
            head = ", ".join(missing[:3])
            out.append(tr("blocker_missing", head=head,
                       more=tr("blocker_more", n=len(missing) - 3) if len(missing) > 3 else ""))
        return out

    def refresh_queue(self) -> None:
        """Republish the queue, its count, the reason and the button's enablement.

        Called from every widget change that can move the answer, so the page cannot show
        one thing and run another. The list is numbered because a batch reports "file 2"
        and the user has to be able to find out which file that was.
        """
        files = self.pending_files()
        self.file_list.setPlainText(
            "\n".join(f"{i}. {p}" for i, p in enumerate(files, 1)))
        blockers = self.blockers()
        if self._batch_total > 1:
            # A single file is not a batch and does not get batch vocabulary: "0 / 1
            # file(s)" over a one-file run is noise, and the queue wording is what a user
            # of the single-file path has always read.
            self.count_label.setText(tr("count_progress", done=self._files_done, total=self._batch_total))
        else:
            self.count_label.setText(tr("count_queued", n=len(files)))
        self.reason_label.setText("; ".join(blockers))
        self.refresh_summary()
        self._publish_run_state(blockers)

    def refresh_models(self) -> None:
        arch = self.current_arch()
        # `names()` (weights on disk, resolves right now), not `known_names()` (everything
        # the baked tables declare). A selectable model with no weight file is a control
        # that lies: START can only refuse with MissingWeightError ("not started: no
        # demucs weight file"), and this fork has no downloader to make the ghost real.
        names = self.catalog.names(arch)
        self.model_combo.clear()
        self.model_combo.addItems([catalog.NO_MODEL, *names])
        current = getattr(self.selections, MAIN_MODEL_FIELD[arch], catalog.NO_MODEL)
        self.model_combo.setCurrentText(current if current in names else catalog.NO_MODEL)
        self.model_hint.setText(tr("models_installed", n=len(names)) if names
                                else tr("models_none"))
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
            return tr("params_none", arch=arch)
        lines = [tr("params_line", arch=arch, n=len(group))]
        lines += [f"{name} = {getattr(self.settings, name, '?')!r}" for name in group]
        return "\n".join(lines)

    def refresh_parameter_summary(self) -> None:
        """Re-publish `parameter_summary()` -- on an arch change and after a settings
        save, the two moments the numbers can move under the page."""
        self.params_summary.setText(self.parameter_summary())

    def _publish_run_state(self, blockers: list[str]) -> None:
        """The one expression that decides whether START is pressable.

        Two features write this button -- the busy state and the blockers -- and two
        spellings of the same rule is the pattern this fork keeps finding: a mid-batch
        `refresh_queue()` that re-enabled START would hand the click to `Job.start`, which
        answers "a job is already running", and a `set_busy(False)` that ignored the
        blockers would enable a START that can only refuse. So the rule lives here, once,
        and both callers call it.

        The list arrives rather than being recomputed because `blockers()` stats every
        queued path: `refresh_queue` already has it for the reason label, and asking again
        would stat a 400-file queue twice per keystroke in the Input box.
        """
        self.start_button.setEnabled(not self._busy and not blockers)
        self.stop_button.setEnabled(self._busy)
        # Everything that decides the run in flight is read-only for its duration. The plan
        # is assembled once (`Controller._on_run`) and `export_for` is asked per file, so a
        # mid-batch Output edit would move the rest of one batch into a second folder -- the
        # folder `run_job` diffs to decide what it wrote -- and a format or GPU click would
        # print "settings saved" while every remaining file still ran under the plan
        # assembled before it. Freezing the controls means there is no deferred write to
        # explain. The queue is deliberately absent: lining up the next run while this one
        # finishes is legitimate, and `_Batch.files` is already a copy of the paths.
        for widget in (self.output_edit, self.browse_output, self.default_output_check,
                       self.model_combo, self.arch_bar, self.gpu_check,
                       self.device_combo, self.params_edit,
                       *self.format_radios.values()):
            widget.setEnabled(not self._busy)

    def set_busy(self, busy: bool) -> None:
        """Whether a run is in flight -- one of the two halves of START's enablement."""
        self._busy = busy
        self._publish_run_state(self.blockers())

    def show_compute(self, lines: tuple[str, ...]) -> None:
        """Publish the plan-derived compute line. The controller calls this the moment
        `session.assemble()` returns, because `Session.compute_lines` is the only place
        those words come from -- and `compute_label` must stay empty until then. An empty
        tuple is how a refusal clears the line again (I5(a)), and `translate_compute_line`
        drops the CLI's `[cli] ` prefix and its `--device` advice, which is the GUI's
        translation layer and the only place either of those is rewritten.
        """
        self.compute_label.setText("\n".join(translate_compute_line(ln) for ln in lines))

    def set_default_output(self, value: str) -> None:
        """Publish what the file actually says.

        Called by the root after a successful write *and* after a refused one, which is
        what makes the checkbox honest: it shows the stored default, never the last click.
        `blockSignals` around the state change because setting the check from the file is
        not a user action -- re-emitting here would ask the root to write the value it just
        read back, and a signal loop that terminates only because the value happens to be
        equal is not a design.
        """
        self.ui = ui_prefs.UiPrefs(export_path=value)
        self.default_output_check.blockSignals(True)
        self.default_output_check.setChecked(bool(value))
        self.default_output_check.blockSignals(False)
        self.refresh_default_output()

    def _checked_format(self) -> str:
        """The format the radios actually hold, or `"?"`.

        `"?"` is a real answer and not a fallback: `settings.save_format` is a free `str`
        field in a file a user can hand-edit, so `"OGG"` loads, matches none of the three
        radios, and a page that defaulted its display to WAV would be asserting a format
        the engine-side gate (`session.check_save_format`) is about to refuse. Showing the
        unknown value is what makes the refusal explicable.
        """
        return next((f for f, r in self.format_radios.items() if r.isChecked()), "?")

    def truth_summary(self) -> str:
        """One line that repeats, out of the live widgets, what this run will do.

        Derived and never cached: every term reads a widget or the `Backend` this page
        resolved, so no change can leave five terms still asserting the previous run. It is
        the line a user skims before pressing START, and it is the line that would have
        told whoever sent the first screenshot that the page was on `apollo` with nothing
        installed for that arch -- the state that made the app read as broken rather than
        as unconfigured.

        It claims nothing the widgets do not already say, which is the property that keeps
        a summary from becoming a second source of truth: no computed verdict, no
        "expected time", nothing the engine has not been asked.
        """
        files = self.pending_files()
        return " · ".join([
            device_summary(self._backend), self.current_arch(), self.selected_model(),
            tr("summary_files", n=len(files)) if files else tr("summary_no_input"),
            tr("summary_output", fmt=self._checked_format()),
            "GPU" if self.gpu_check.isChecked() else "CPU",
        ])

    def refresh_summary(self) -> None:
        """Republish the strip -- which needs the machine, and so runs after `_sync_from_state`.

        The `getattr` is the constructor's own ordering, not a state that cannot happen:
        `_build()` refreshes the queue before `_sync_from_state()` has resolved a `Backend`,
        and `is_cpu_fallback()` above reads the same attribute the same way for the same
        reason. Silently skipping the first publish is right; a strip that claimed a device
        the page had not yet asked the machine about would not be.
        """
        if getattr(self, "_backend", None) is None:
            return
        self.summary_label.setText(self.truth_summary())

    def set_stem_format(self, fmt: str) -> None:
        """Show the format the file actually holds -- including one no radio can hold.

        `save_format` is a free `str` in a file a user can hand-edit, so the value a refused
        write reverts to may be `"OGG"`. Indexing the group by it raised `KeyError` out of a
        Qt slot (PySide6 prints that to stderr and the loop carries on), which left the
        rejected pick on screen over a file that still said something else. Nothing checked
        is the state that value loaded into, and `_checked_format` reads it as `"?"`.

        Exclusivity is lifted for the write because an exclusive `QButtonGroup` will not let
        its checked button be cleared, and "clear the group" is exactly what an unknown
        value asks for.
        """
        self.format_group.setExclusive(fmt in self.format_radios)
        for name, radio in self.format_radios.items():
            radio.blockSignals(True)
            radio.setChecked(name == fmt)
            radio.blockSignals(False)
        self.format_group.setExclusive(True)
        self.refresh_summary()

    def set_gpu(self, on: bool) -> None:
        self.gpu_check.blockSignals(True)
        self.gpu_check.setChecked(on)
        self.gpu_check.blockSignals(False)
        self.refresh_summary()

    def refresh_default_output(self) -> None:
        """Say where an empty Output box would send the stems.

        The box's own text is the override and this line is the fallback, so the two
        together are the whole answer; neither one of them alone is, which is why this is
        a visible label and not a placeholder (a placeholder disappears the moment the
        field is focused, and the moment of focusing it is exactly when the user is
        deciding).
        """
        self.default_output_label.setText(
            tr("default_set", path=self.ui.export_path) if self.ui.export_path
            else tr("default_unset"))

    # -- batch lifecycle ------------------------------------------------------ #

    def begin_batch(self, total: int) -> None:
        """Open the run the root is about to start. `total` is `pending_files()`'s length.

        The console line is for a *multi*-file batch only. One file is what 5.6 does and
        what Plan 2 shipped, and wrapping every single run in batch vocabulary would be a
        change to what the user reads without being a change to what happens -- which is
        the kind of edit this suite's two-sided rule exists to catch.
        """
        self._batch_total = max(1, total)
        self._files_done = 0
        self._batch_active = True
        self.progress.setValue(0)
        self.refresh_queue()
        if self._batch_total > 1:
            self.console.appendPlainText(
                f"batch started: {self._batch_total} files, one after another; a file that "
                f"fails is reported and the run moves on -- STOP stops the rest")

    def _on_progress(self, pct: int) -> None:
        """One file's percent, spread over the batch.

        This page's docstring used to refuse batch mode because `run_job` "would reset per
        file and read as a saw-tooth". It does reset per file -- so the number published
        is not the file's, it is `(files done + this file's fraction) / total`.
        `session.progress_percent`'s own `total_files` term stays untouched: that is the
        engine's arithmetic for the models inside one file, and this is the GUI's
        aggregation of the per-file signal. Two different denominators, one bar.
        """
        value = max(0, min(100, int(
            (self._files_done + pct / 100.0) * 100 / max(1, self._batch_total))))
        self.progress.setValue(value)
        self.progress_label.setText(f"{value}%")

    def end_file(self) -> None:
        """One file is over, however it ended. Called by the root before it advances.

        Republishes the queue because the count label is the only place a user can see
        *which* file the engine is on, and a label that stayed at "0 / 30" for the whole
        run would be a progress indicator that does not progress.
        """
        self._files_done += 1
        self.refresh_queue()

    def end_batch(self, *, written: int, failures: list[str], stopped: bool) -> None:
        """Release the busy state and give the tally.

        A run releases busy here when the queue had more than one file in it, and on its
        own terminal when it did not -- see `_terminal_releases`, which is the rule and not
        this docstring. What batch mode forces is that a terminal in the *middle* of a
        queue must not re-enable START: `Job.start` would answer that click with "a job is
        already running", and a refusal on a button the user was told was enabled is the
        failure this fork has been asked to fix twice now.

        `size` is the queue's length and `total` is how far it got, and they are not the
        same number: "was this a batch" is answered by the first, which is why a batch
        stopped on its very first file still gets the stopped line, while a one-file run
        never gets batch vocabulary at all.
        """
        size, total = self._batch_total, self._files_done
        self._batch_active = False
        self._batch_total = 0
        self.set_busy(False)
        self.refresh_queue()
        if size <= 1:
            # One file is not a batch, in either direction: `_on_done` / `_on_failed` /
            # `_on_cancelled` already said everything there is to say about it, and
            # appending batch vocabulary -- including "the rest were not run", when there
            # was no rest -- would change what a user of the path Plan 2 shipped has
            # always read.
            return
        if stopped:
            self.progress.setValue(0)
            line = f"batch stopped after {total} file(s); the rest were not run"
            self.console.appendPlainText(line)
            # The one ending the user caused gets the same surface as the two they did
            # not: `_notify_for_file` stayed silent for every file of this batch, so
            # without this a STOP is a console line nobody is looking at.
            self._notify(CANCELLED_TITLE, line)
            return
        self.console.appendPlainText(
            f"batch finished: {total} file(s) attempted, {written} file(s) written"
            + (f", {len(failures)} of {size} file(s) failed" if failures else ""))
        for line in failures:
            self.console.appendPlainText(f"  failed: {line}")
        if failures:
            self._notify(FAILED_TITLE, f"{len(failures)} of {size} file(s) failed -- "
                                       f"see the console lines above")
        else:
            # The batch gets one surface, at the end, in the batch's own words: a run of
            # forty files that never said anything until the console went quiet is a run
            # the user has to go and inspect.
            self._notify(DONE_TITLE, f"{total} file(s) done, {written} file(s) written")

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

    def _pick_more_inputs(self) -> None:
        """Many files, appended to what is already queued. A refused pick changes nothing.

        The union is built from `pending_files()` -- the typed file included -- and handed
        to `add_extra_files` as a replacement, so the queue never grows a second holder of
        the path that is already in `input_edit`.
        """
        start = str(Path(self.input_edit.text()).parent) if self.input_edit.text().strip() \
            else str(Path.home())
        chosen = dialogs.Pickers.input_files(self, start)
        if chosen is None:
            return
        self.add_extra_files([*[str(p) for p in self.pending_files()], *chosen])

    def _pick_input_folder(self) -> None:
        start = str(Path(self.input_edit.text()).parent) if self.input_edit.text().strip() \
            else str(Path.home())
        chosen = dialogs.Pickers.input_folder(self, start)
        if chosen is None:
            return
        found = [str(p) for p in audio_files_in(Path(chosen))]
        self.add_extra_files([*[str(p) for p in self.pending_files()], *found])
        self.console.appendPlainText(
            f"{len(found)} audio file(s) found under {chosen}"
            + ("" if found else " -- the engine reads none of what is in there"))

    def _on_default_output_toggled(self, checked: bool) -> None:
        """Ask the root to persist it. A widget does not write a settings file.

        Empty text with the box ticked is the one combination that would otherwise store a
        default meaning "the working directory", because `Path("")` is a relative path that
        exists -- so it is refused here, on the way out, rather than reaching the file and
        then the engine's `mkdir`.
        """
        text = self.output_edit.text().strip()
        if checked and not text:
            # Restored from the stored value rather than from the click: the file is the
            # only thing that knows what the default is.
            self.set_default_output(self.ui.export_path)
            self.console.appendPlainText(
                tr("default_not_set"))
            return
        self.default_output_requested.emit(text if checked else "")

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

    def _notify_for_file(self, title: str, text: str) -> None:
        """A terminal's surface, unless a batch is speaking -- then the tally speaks.

        Called with `_batch_total` still holding the batch's size: the root's own handler
        is connected after this page's, so `end_batch()` has not run yet. The console keeps
        every line either way -- the engine's own words, and a failure's traceback frames,
        belong in the permanent record -- but one InfoBar per file over forty files is a
        queue the user has to click through to find out whether the run finished, which is
        the failure `app/dialogs.py`'s notification queue exists to prevent.
        """
        if self._batch_total > 1:
            return
        self._notify(title, text)

    def _terminal_releases(self) -> bool:
        """True when this terminal really is the end of the run.

        Either there is no batch (`_batch_total` 0 or 1 -- a single file is what Plan 2
        shipped and what these handlers used to release on, unconditionally), or this file
        is the queue's last. Checked as `_files_done + 1 >= _batch_total` because
        `end_file()` has not run yet when a terminal arrives: the root's slot is connected
        after this page's, and walking the queue is its job, so this terminal is file
        `_files_done + 1`.

        A run releases busy, not a file. The other half of the rule is that a terminal
        *outside* a batch still releases, because `Job.start` answers a double click with
        `failed` and a page that only ever un-disabled START from `end_batch` would then be
        disabled with no run in flight -- the stuck-button failure in a new costume.
        """
        return self._batch_total <= 1 or (self._files_done + 1) >= self._batch_total

    def _file_ended(self) -> None:
        """The one place a terminal decides whether the run is over."""
        if self._terminal_releases():
            self.set_busy(False)

    def _on_done(self, written: list) -> None:
        """One file's outcome. The busy state is the run's -- see `_terminal_releases`."""
        self._file_ended()
        for path in written:
            self.console.appendPlainText(f"wrote {path} ({path.stat().st_size} bytes)")
        self._notify_for_file(DONE_TITLE, f"{len(written)} file(s) written")

    def _on_failed(self, text: str) -> None:
        # The console is the permanent record and the surface is the one that gets your
        # attention: both get the engine's own words, never a paraphrase. Task 5 showed
        # these lines in the console only; the category comes from `_classify_failure`,
        # which answers None for text nobody recognises -- and that stays `Failed`.
        self._file_ended()
        self.console.appendPlainText(f"failed: {text}")
        self._notify_for_file(FAILED_TITLE, text)

    def _on_cancelled(self, text: str) -> None:
        # Zeroed only when the run ended here: mid-batch that would erase the position the
        # tally is about to report, and `end_batch(stopped=True)` zeroes it in the same
        # breath as it says how far the run got.
        self._file_ended()
        if self._terminal_releases():
            self.progress.setValue(0)
        self._notify_for_file(CANCELLED_TITLE, text)

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
