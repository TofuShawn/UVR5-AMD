"""The settings page: 50 `Settings` fields, 33 selections, and the refresh command.

The groups below are not a taste call -- they follow which arch's model-data branch
reads each field, which is exactly the criterion that fixed `Settings` at 50 fields.
`plan.py` cites the source line for every one of them, so each entry is checkable.

Why the widgets are generated rather than written out
-----------------------------------------------------
`Settings` is 50 typed fields and `Selections` 33 more, and both classes re-parse every
value through their own `_coerce` on construction. So the page reads those two type
tables and builds one widget per field from them: the field list is the only place a
name can come from, and the validator is the type system the file already uses. A
hand-written form would drift from `Settings` on the next field added -- and drift here
is silent data loss, because a field with no widget keeps whatever the file said
forever while looking saved.

Layout: stacked `Expander`s inside one `ScrollView`, measured rather than preferred
-----------------------------------------------------------------------------------
The task allowed a `Pivot`/`TabView` of per-arch tabs if it laid out better. It cannot:
probed on the pinned wheel, `Pivot` exposes `addItem`/`items`/`itemAt` and **no content
setter at all**, and `TabView` exposes `addTab` and likewise none -- `PivotItem` and
`TabViewItem` are `Object`s carrying `text`/`selected`/`enabled`, not widgets, so
neither control can *hold* a field. That leaves `Expander` as the only container in
this binding that shows content, and content is what 83 fields are.

The scroll view is a measured necessity, not decoration. With every section collapsed
the page wants 320 px; expanded, the panels are 630 (General), 284 (VR), 436 (MDX),
502 (Demucs) and **1262 px** (the 33 selections) -- 3114 px of form against a 600 px
viewport, so without a scroll area the bottom sections would be laid out below the
window and unreachable. A `ScrollView` alone is not enough either: it defaults to
`widgetResizable(False)` and never learns that an opened section grew, which put one
section's fields on top of the refresh button in the real-session screenshot. `_fit_body`
is what closes that gap, by the binding's own numbers (each section's `sizeHint()`, the
body's `sizeHint()`, the viewport's width), and each section's expansion animation is
off so those numbers are true on the tick the user clicks.

`apply()` validates before it writes, and `set_field_value_raw()` exists to prove it:
see their docstrings.
"""
from __future__ import annotations

import dataclasses
import re
import typing
import weakref
from pathlib import Path

import catalog
import fluentqt
import model_refresh
import selections as selections_module
import settings as settings_module
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout

ARCH_FIELD_GROUPS: dict[str, tuple[str, ...]] = {
    # read on the VR branch (plan.py, UVR.py:1621/1628 + 3421-3451)
    "vr": ("aggression_setting", "window_size", "batch_size", "post_process_threshold",
           "is_tta", "is_post_process", "is_high_end_process", "crop_size"),
    # read on the MDX / MDX-C / ONNX branch (UVR.py:1586-1606 + 3568-3645)
    "mdx": ("mdxnet_stems", "mdx_segment_size", "overlap_mdx", "overlap_mdx23",
            "mdx_batch_size", "compensate", "is_mdx_c_seg_def", "is_mdx23_combine_stems",
            "denoise_option", "is_match_frequency_pitch", "is_deverb_vocals",
            "deverb_vocal_opt"),
    # read on the demucs branch (UVR.py:1644-1663 + 3498-3528)
    "demucs": ("demucs_stems", "segment", "overlap", "shifts", "margin_demucs",
               "is_split_mode", "is_chunk_demucs", "is_demucs_combine_stems",
               "semitone_shift", "is_invert_spec", "margin",
               "is_demucs_pre_proc_model_activate", "is_demucs_pre_proc_model_inst_mix"),
    # everything else: format, device, stems-only, secondary switches, ensemble leftovers
    "common": ("chosen_process_method", "device_set", "gpu_conversion", "is_use_directml",
               "is_normalization", "save_format", "mp3_bit_set", "wav_type_set",
               "is_primary_stem_only", "is_secondary_stem_only",
               "is_primary_stem_only_Demucs", "is_secondary_stem_only_Demucs",
               "vr_is_secondary_model_activate", "mdx_is_secondary_model_activate",
               "demucs_is_secondary_model_activate", "ensemble_main_stem",
               "is_save_inst_set_vocal_splitter"),
}

#: Display order of the arch sections. `common` first: it holds the device and format
#: choices that decide what the other three mean.
SECTION_ORDER: tuple[str, ...] = ("common", "vr", "mdx", "demucs")

#: Strict numeric text -- deliberately narrower than Python's parsers, which also accept
#: PEP 515 separators (`float('1_000')` == 1000.0). See `_read`.
_NUMERIC_TEXT = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")

_BOOL, _INT, _FLOAT, _STR = bool, int, float, str


def all_field_names() -> frozenset[str]:
    return frozenset(name for group in ARCH_FIELD_GROUPS.values() for name in group)


_MISSING = {f.name for f in dataclasses.fields(settings_module.Settings)} - all_field_names()
if _MISSING:
    raise RuntimeError(
        f"settings page has no widget for {sorted(_MISSING)}: a field the GUI cannot "
        f"edit is a field that silently keeps its loaded value forever")


def widget_kind(typ: type) -> type:
    return _BOOL if typ is bool else _INT if typ is int else _FLOAT if typ is float else _STR


def _weak_slot(page_ref, method: str):
    """A Qt slot that reaches the page through a weak reference only -- a crash fix.

    Measured on the pinned wheel: a signal connected to a callable that keeps a strong
    reference to the page (a `lambda: self...` or a bound method) forms a Python->C++
    cycle, and a page destroyed that way -- which is what happens when a dialog closes
    and its last reference goes -- took the process down with an access violation inside
    `destroyQCoreApplication`/`QWidget::destroy` in 4 runs out of 4, at one page as well
    as at sixteen. Disconnecting the signal: 0 of 3 crashes. Reconnecting through a weak
    reference: 0 of 3. Keeping the pages alive instead would only hide the hazard until
    the next dialog a user closes, and `tests/test_settings_page.py` pins the property
    (the page must be collectable once the test lets go of it).

    The reference cannot dangle into a call: every signal connected here comes from a
    child of the page, so a dead page has no child left to emit. The guard is for the
    teardown ordering, not for the user.
    """

    def slot(*_args) -> None:
        page = page_ref()
        if page is not None:
            getattr(page, method)()

    return slot


def _deferred_slot(page_ref, method: str):
    """A slot that schedules `page.method()` for the next turn of the event loop.

    Needed because Qt coalesces layout updates: measured, `Expander.expandedChanged`
    fires *before* the expander's own `sizeHint()` has caught up, so a fit done inside
    the emission sized the body to the previous state (338 px for a form that wanted
    953), and the section then grew past it. One turn later the same numbers are right
    (953 / 1258 / 1711 / 2194 / 3457 px for one, two, three, four and five open
    sections, each with nothing overlapping the next). Like `_weak_slot`, it holds the
    page weakly, so the deferral cannot resurrect a closed dialog.
    """

    def slot(*_args) -> None:
        QTimer.singleShot(0, _weak_slot(page_ref, method))

    return slot


class SettingsPage(fluentqt.Card):
    """Every editable value the engine reads, in one dialog-sized surface.

    The page holds no state of its own beyond the widgets: `self.st` / `self.sel` are the
    objects it was built from, and after a clean `apply()` they are the objects that went
    into the file, so a caller that saves and then runs uses what the user just chose.
    No `Job` is constructed here (R-T4-onejob) and no network call happens except through
    `refresh_tables`, which routes everything to `model_refresh.refresh`.
    """

    def __init__(self, *, settings_obj=None, selections_obj=None, parent=None):
        super().__init__(parent)
        self.st = settings_obj if settings_obj is not None else settings_module.load()
        self.sel = (selections_obj if selections_obj is not None
                    else selections_module.load())
        self._widgets: dict[str, object] = {}
        #: values written by `set_field_value_raw`, which a widget may not be able to hold
        self._raw: dict[str, object] = {}
        self._types = typing.get_type_hints(settings_module.Settings)
        self._stypes = typing.get_type_hints(selections_module.Selections)
        self._scroller: fluentqt.ScrollView | None = None
        self._body: fluentqt.Card | None = None
        self._sections: list = []
        self._build()

    # -- construction ---------------------------------------------------------- #

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        # One scrollable body: see the module docstring for the heights that make it
        # mandatory. The body is a `Card` and not a bare `QWidget` because a plain
        # surface keeps the native palette while its Fluent children follow the theme
        # (spec §5.3), and Task 5's guard fails the commit over exactly that.
        self._scroller = fluentqt.ScrollView(self)
        self._body = fluentqt.Card()
        form = QVBoxLayout(self._body)
        page_ref = weakref.ref(self)
        for arch in SECTION_ORDER:
            form.addWidget(self._section(
                "General" if arch == "common" else arch.upper(),
                ARCH_FIELD_GROUPS[arch], self._types, self.st))
        form.addWidget(self._section(
            "Secondary models and splitters",
            tuple(sorted(self._stypes)), self._stypes, self.sel))
        self.refresh_button = fluentqt.Button("Refresh model tables", self._body)
        self.refresh_button.clicked.connect(_weak_slot(page_ref, "_on_refresh_clicked"))
        form.addWidget(self.refresh_button)
        # Where a refresh says what it fetched and wrote. The page has no other voice,
        # and `model_refresh.refresh` reports through a printer it must be given.
        self.status = fluentqt.TextEdit(self._body)
        self.status.setReadOnly(True)
        self.status.setMaxVisibleLines(3)
        form.addWidget(self.status)
        self._scroller.setContentWidget(self._body)
        root.addWidget(self._scroller, 1)
        self._fit_body()

    def _section(self, title: str, names, types: dict, holder) -> "fluentqt.Expander":
        """`Expander(title, parent)` cannot be constructed at all -- the same Shiboken
        overload defect as `FluentMenu` (spec §5.2.1) -- so the title is set after
        construction. The verb is `setHeaderText`, not `setTitle`: Expander has no
        `setTitle` (measured, and pinned by Task 1), while `ContentDialog` on the other
        side has `setContent` and no `setContentWidget`. The content here is a `Card`
        because `setContentWidget` takes a widget and a bare QWidget would be a
        native-palette surface (Task 5's guard).

        Each non-bool field is a `Label` and its widget in one `QHBoxLayout` row, added
        to the form as a *layout*: 83 fields with no names next to them are not a
        settings page, and a per-row container widget would be a bare `QWidget` owned by
        a `Card`, which the surface guard rightly refuses. A `CheckBox` needs no row --
        its own text *is* the field name.

        The animation is off because the scroll area needs the new height at all: with
        it on, `sizeHint()` only catches up when the transition finishes (measured: 44 px
        straight after `setExpanded(True)`, 1307 px after the animation), and the panel
        then paints over its neighbours. Off, the numbers are right one turn of the event
        loop later, which is what `_deferred_slot` waits for.
        """
        group = fluentqt.Expander(self)
        group.setHeaderText(title)
        group.setAnimationEnabled(False)
        group.expandedChanged.connect(_deferred_slot(weakref.ref(self), "_fit_body"))
        self._sections.append(group)
        host = fluentqt.Card()
        form = QVBoxLayout(host)
        for name in names:
            widget = self._field(name, types[name], getattr(holder, name))
            if isinstance(widget, fluentqt.CheckBox):
                form.addWidget(widget)
            else:
                row = QHBoxLayout()
                row.addWidget(fluentqt.Label(name, host))
                row.addWidget(widget, 1)
                form.addLayout(row)
        group.setContentWidget(host)
        return group

    def _field(self, name: str, typ: type, value):
        """The one widget this field has, chosen by its declared type and nothing else."""
        kind = widget_kind(typ)
        if kind is _BOOL:
            w = fluentqt.CheckBox(name, self)
            w.setChecked(bool(value))
        elif kind in (_INT, _FLOAT):
            # NumberBox is a LineEdit subclass in this binding, not a spin box: it
            # takes floats natively (setValue(0.5) -> value() == 0.5) and has no
            # setSingleStep. Float fields therefore need display precision rather than a
            # second widget type -- `post_process_threshold` and the twelve scales.
            w = fluentqt.NumberBox(self)
            if kind is _INT:
                w.setRange(-2_147_483_648, 2_147_483_647)
            else:
                w.setRange(-1e6, 1e6)
                w.setDisplayPrecision(6)
            w.setValue(value)
        else:
            # A `LineEdit` and not a `ComboBox`, because a combo needs a value list and
            # this page is not allowed to invent one: `Catalog` knows model *names*, not
            # the `save_format` / `mp3_bit_set` / stems tokens, and the sentinel-or-number
            # fields (`batch_size`, `overlap`, `compensate`, ...) are stored as the text
            # the tkinter var held (ruling R-T9-types), which no list here could be
            # authoritative about. What the field type refuses, `apply()` reports.
            w = fluentqt.LineEdit(self)
            w.setText(str(value))
        w.setObjectName(name)
        self._widgets[name] = w
        return w

    # -- reading and writing one field ----------------------------------------- #

    def _on_refresh_clicked(self) -> None:
        """The button's meaning, in one word: ask. Confirmation stays `refresh`'s rule.

        And the refusal has to be *said*. This was `self.refresh_tables(confirmed=False)`
        on one line, which let the exception out of a Qt slot -- and on this wheel
        (PySide6 6.9.3, measured) an exception that escapes a slot does not reach the
        user: the binding prints the traceback to stderr and the event loop carries on, so
        pressing "Refresh model tables" looked exactly like a button that did nothing.
        Three ways this call fails, and each one is the user's to act on:

        * `RefreshError` -- what `confirmed=False` always raises (R-T10-refresh), and the
          only reason a click has ever had in this app;
        * `OSError` -- a stalled or refused fetch (`urllib.error.URLError` is an `OSError`
          subclass, and `fetch_json` lets it through) or a catalog directory that will not
          take a write;
        * `catalog.CatalogError` -- the writable dir resolving inside a version-controlled
          root, which is `Catalog.write_table`'s refusal, not this page's.

        All three land in the status box that was built for exactly this. Nothing here
        weakens the confirmation rule: the page still passes `confirmed=False`, and a real
        refresh still needs the CLI flag -- Task 7 shipped the params form, the failure
        surfaces and the pickers, and no confirmation dialog (it is not in that brief's
        interfaces, and inventing a yes/no surface for a network call is not this page's
        call to make).
        """
        try:
            self.refresh_tables(confirmed=False)
        except model_refresh.RefreshError as exc:
            self._log(f"refresh refused: {exc}")
        except (OSError, catalog.CatalogError) as exc:
            self._log(f"refresh failed: {type(exc).__name__}: {exc}")

    def _fit_body(self) -> None:
        """Give every section the height it now reports, and the body the height that
        follows -- at the viewport's width.

        This is what makes the scroll area honest, and both halves are measured on the
        pinned wheel. `ScrollView` defaults to `widgetResizable(False)`, so it never
        resizes the widget it was given: an opened section grew to 1307 px inside a body
        still 338 px tall and painted its fields straight over the refresh button (the
        overlap in `task-6-shots/04_settings_all_expanded.png`, taken before this method
        existed). Setting each section's minimum height to its own `sizeHint()` makes the
        body's `sizeHint()` the sum (measured: 338 -> 1601 px), and sizing the body to
        that at the viewport's width gives the scroll bar something to scroll
        (`scrollableHeight()` 0 -> 1120 px). Nothing here invents a number.
        """
        if self._scroller is None or self._body is None:
            return
        for group in self._sections:
            group.setMinimumHeight(group.sizeHint().height())
        natural = self._body.sizeHint()
        # The scroll area's own `sizeHint()` is a scroll area's, not its content's --
        # measured 464 px against a body that wants 501 -- so a dialog sized from it
        # opens with a horizontal scrollbar. Asking the scroller for the body's width is
        # how the page says "as wide as the widest row needs", with no constant.
        self._scroller.setMinimumWidth(natural.width())
        width = max(self._scroller.viewport().width(), natural.width())
        self._body.setMinimumSize(width, natural.height())
        self._body.resize(width, natural.height())

    def resizeEvent(self, event) -> None:
        """Re-fit on every resize: the body's width follows the viewport, and the dialog
        can be dragged wider than it opened."""
        super().resizeEvent(event)
        self._fit_body()

    def widget_names(self) -> list[str]:
        """Every field the page can edit, in build order -- what the coverage tests read."""
        return list(self._widgets)

    def field_value(self, name: str):
        """What this field would save right now, after the type system has said yes."""
        return self._read(name)

    def _read(self, name: str):
        widget = self._widgets[name]
        if name in self._raw:
            return self._raw[name]
        if isinstance(widget, fluentqt.CheckBox):
            return widget.isChecked()
        if isinstance(widget, fluentqt.NumberBox):
            # Two readings, and the choice matters. `value()` is the only lossless one of
            # what the box displays: with `setDisplayPrecision(6)` the *text* of
            # 0.123456789 is '0.123457', so reading text would rewrite a scale the user
            # never touched. But `value()` is *stale* for text that is not a number (a
            # `NumberBox` keeps its last good value when you type junk into it), and
            # pre-parsing the text here would also step on `_coerce`'s digit-separator
            # refusal, which only fires on text. So: `value()` for strict numeric text,
            # the text itself otherwise, and the type system gets the last word either
            # way.
            text = widget.text().strip()
            return widget.value() if _NUMERIC_TEXT.fullmatch(text) else text
        return widget.text()

    def set_field_value(self, name: str, value) -> None:
        """The typed write: the widget receives the value as its declared type has it."""
        widget = self._widgets[name]
        self._raw.pop(name, None)
        if isinstance(widget, fluentqt.CheckBox):
            widget.setChecked(bool(value))
        elif isinstance(widget, fluentqt.NumberBox):
            widget.setValue(float(value))
        else:
            widget.setText(str(value))

    def set_field_value_raw(self, name: str, value: object) -> None:
        """The untyped write: record `value` exactly as given, for `apply()` to judge.

        This is the page's seam for *a value that did not come through a widget's own
        type* -- junk text, `None`, a container, the text `'False'` in a bool slot. A
        widget cannot display several of those (a `CheckBox` has no text form for
        `'False'`, and a `NumberBox` refuses to change `value()` for text that is not a
        number), so the value is stored beside the widget and read back by `_read`; the
        text-bearing widgets are updated as well, so what is on screen matches what was
        refused. `apply()` has to reject every one of them, which is what the two error
        classes in its docstring are for.
        """
        self._raw[name] = value
        widget = self._widgets[name]
        if isinstance(widget, (fluentqt.NumberBox, fluentqt.LineEdit)):
            widget.setText(str(value))

    def load_from(self, settings_obj, selections_obj) -> None:
        """Push two state objects into every widget -- the reverse of `apply`.

        Iterates the widgets, not the field tuples, so a field the page has no widget for
        could not be silently skipped here; the import-time guard is what keeps that set
        complete.
        """
        self.st = settings_obj
        self.sel = selections_obj
        self._raw.clear()
        for name in self._widgets:
            holder = self.st if name in self._types else self.sel
            self.set_field_value(name, getattr(holder, name))

    # -- save ------------------------------------------------------------------ #

    def apply(self, path: Path | str) -> list[str]:
        """Validate through the type system, then write both halves; return the messages.

        Empty means the file now holds what the page showed. Non-empty from a *refused
        value* means **nothing was written**: the two dataclasses are constructed before
        either `save()` runs, because a `settings.json` carrying one new half and one old
        half is a config the next load cannot explain. Non-empty from a *refused write*
        says which half got through -- see the note below the two `save` calls, because
        that is the one case this promise cannot keep.

        The validator is `Settings(**values)` / `Selections(**values)` themselves. Both
        re-parse every field through their `_coerce` in `__post_init__`, so the page owns
        no rules of its own and cannot drift from the format -- and both error classes
        have to be caught here, because the two modules raise different ones for the same
        shape: a junk or `None` scale raises `ValueError` from `_coerce`, while
        `selections.load()`'s gate raises `TypeError` for it; a container in a `str` field
        and a non-bool in a bool slot raise `TypeError` from either side. Catching only
        `ValueError` would let a settings-page save escape as a traceback through a
        button. The message comes from the coercion itself, which names the field, so the
        user is told *which* value to fix.

        An `OSError` is the third way out, and it is caught for the same reason as the
        other two: `apply()` runs inside the save button's slot, on this wheel an exception
        that escapes a slot is printed to stderr and the app carries on, and a locked
        `settings.json` is the ordinary Windows case rather than an exotic one. The dialog
        stays open either way; what changes is whether the user is told.
        """
        target = Path(path)
        values = {name: self._read(name) for name in self._types}
        svalues = {name: self._read(name) for name in self._stypes}
        try:
            st = settings_module.Settings(**values)
            sel = selections_module.Selections(**svalues)
        except (TypeError, ValueError) as exc:
            return [f"not saved: {exc}"]
        # Two writers, one file, and the second can fail after the first succeeded --
        # `Settings.save` and `selections.save` are both read-modify-write merges of the
        # same `settings.json`, and on Windows a file open in an editor makes
        # `PermissionError` (an `OSError`) the *normal* case, not the exotic one. Neither
        # of them may escape: this runs inside the save button's slot, and an exception
        # that leaves a Qt slot prints to stderr and carries on, which is how a refused
        # save reached the user as silence. `ValueError` is in the catch for the other way
        # the shared file can fight back -- `Settings.save` re-reads it first, so a
        # truncated file raises `JSONDecodeError`, and its `allow_nan=False` raises
        # `ValueError` for a value `_coerce` was never shown.
        #
        # This is the one window where "write nothing on failure" is not airtight, and it
        # is I/O-sized, not validation-sized: validation is finished above, so the only
        # way to land in it is a disk that says no between the two writes. It is reported
        # rather than papered over -- the message says which half is on disk -- and the
        # two writers are not folded into a transaction, which would be more mechanism
        # than a settings file needs.
        try:
            st.save(target)
        except (OSError, ValueError) as exc:
            return [f"not saved: {target} could not be written "
                    f"({type(exc).__name__}: {exc})"]
        try:
            selections_module.save(sel, target)
        except (OSError, ValueError) as exc:
            return [f"only half saved: the 50 settings are on disk in {target} and the "
                    f"33 selections are not ({type(exc).__name__}: {exc}) -- fix the file "
                    f"and press Save again"]
        self.st, self.sel = st, sel
        self._raw.clear()
        return []

    # -- the one network command ----------------------------------------------- #

    def refresh_tables(self, *, confirmed: bool) -> list[Path]:
        """Refresh the four baked model tables, or refuse -- by `model_refresh`, not here.

        The page does no fetching, no URL list of its own and no path derivation: it hands
        `model_refresh.refresh` the catalog from `build_catalog()` and passes
        `confirm=confirmed` straight through, so the button's `confirmed=False` produces
        that module's own `RefreshError` before anything is requested (R-T10-refresh).
        Writing the tables stays `Catalog.write_table`'s, which is what refuses a
        `writable_dir` inside a version-controlled root.
        """
        return model_refresh.refresh(model_refresh.build_catalog(),
                                     confirm=confirmed, printer=self._log)

    def _log(self, text: str) -> None:
        """Where a refresh says what it fetched and wrote: the page's own status box."""
        self.status.appendPlainText(text)
