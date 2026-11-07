"""Where a run stops and a human has to answer.

Three surfaces, all of them consequences of decisions already made:

* tier 3 (`UnknownModelParamsError`) -- the fork's GUI-free path refuses with a
  message naming the hash and the file it wanted; the GUI is the place that message
  turns into a form (spec §4.2, R-T10-params tier 3).
* the four abnormal states the design requires distinct surfaces for (§8): VRAM
  exhaustion, a missing weight, ORT's silent CPU fallback, and a user cancel. Ruling
  R-T7-infobar adds the *discipline*: one at a time, in a band the page reserves for it,
  and gone on its own. Measured on 1.11.2, `InfoBar` has no `show_message`, no
  `setDuration` and no auto-dismiss timer of any kind, and `Flyout` -- which does carry
  `closePolicy`/`aboutToShow`/`closed` -- cannot host one from Python: `Flyout(view)`
  constructs but `show()` leaves it invisible (it wants the C++
  `Flyout(view, position, anchor, ...)` factory), and `Flyout(view, anchor)` raises
  `TypeError: Flyout() takes at most 1 argument (2 given)`, the same Shiboken overload
  hole as `Expander('title', parent)`. So the dismissal is owned here.
* file and directory picking -- `QFileDialog`, because Fluent-Qt's Python binding
  has no file dialog and `FluentMenu` cannot even construct (spec §5.2).

Nothing in this module runs a job or touches the catalog except through
`Catalog.write_params`, and nothing here invents a Fluent verb: `setContent` (not
`setContentWidget`), `setTitle`, `setPrimaryButtonText`, `setCloseButtonText`, `result()`
and `closed` are the names Task 1 pinned.
"""
from __future__ import annotations

import collections
import math
from pathlib import Path

import fluentqt
from PySide6.QtCore import QObject, QTimer
from PySide6.QtWidgets import QFileDialog, QVBoxLayout

#: The per-arch param fields the legacy popups collected. This is the field list read
#: out of `UVR.py`'s popup bodies before Plan 2 Task 8 deleted them, so it is the last
#: copy -- and the line numbers cited throughout this block are line numbers *in that
#: deleted copy*, `UVR.py` at commit `720ca46`, re-derivable only from git. The guard
#: that used to re-read them with `ast` now compares this table against
#: `tests/test_dialogs.py::POPUP_EVIDENCE`, the frozen record of what the walk produced,
#: and `test_the_popups_the_table_was_read_from_are_gone` says why the walk is gone (two
#: of the four verbs the brief named do not exist: there is no `pop_up_mdx_param` -- the
#: MDX popup is `pop_up_mdx_model` at `:4560` of that copy -- and there is no
#: `pop_up_onnx_param` at all, because an `.onnx` model goes through that same body,
#: which reads the graph at `:4570-4575` and writes the `mdx` row).
#:
#: Every key here is a key the engine reads out of `spec.model_data`, because a key a
#: popup collected that `plan.py` never looks up is a *silent* tier-3 failure: the form
#: appears to save and `resolve()` is still blind. That half is still checked live,
#: against `plan.py` and `catalog.py`, by
#: `test_every_field_the_dialog_collects_is_a_key_the_engine_reads`.
PARAM_FIELDS: dict[str, tuple[tuple[str, type], ...]] = {
    # `pop_up_vr_param.pull_data` (`UVR.py:4874-4889`). Engine reads: `vr_model_param`
    # (`plan.py:485`, which then opens `lib_v5/vr_network/modelparams/<name>.json` -- the
    # `dimension_c`/`dim_f`/`n_fft` family lives *inside that file*, not in this answer),
    # `primary_stem` (`plan.py:516/560`, `catalog.py:644`), `nout`+`nout_lstm`
    # (`plan.py:497-498`), and the karaoke trio via the constants at
    # `gui_data/constants.py:83-85` (`plan.py:599-602`).
    "vr": (("vr_model_param", str), ("primary_stem", str), ("nout", int),
           ("nout_lstm", int), ("is_karaoke", bool), ("is_bv_model", bool),
           ("is_bv_model_rebalanced", float)),
    # `pop_up_mdx_model.pull_data` (`UVR.py:4706-4715`). Engine reads: `compensate`
    # (`plan.py:546/552` -- indexed, not `.get()`, so a row without it raises),
    # `mdx_dim_f_set`/`mdx_dim_t_set`/`mdx_n_fft_scale_set`/`primary_stem`
    # (`plan.py:557-560`), the karaoke trio (`plan.py:599-602`).
    "mdx": (("compensate", float), ("mdx_dim_f_set", int), ("mdx_dim_t_set", int),
            ("mdx_n_fft_scale_set", int), ("primary_stem", str),
            ("is_karaoke", bool), ("is_bv_model", bool),
            ("is_bv_model_rebalanced", float)),
    # `pop_up_mdx_c_param.pull_data` (`UVR.py:4749-4751`). Engine reads: `config_yaml`
    # (`plan.py:516`, which puts it straight into a path at `:519` and raises `PlanError`
    # at `:521-525` when that path is not a file; `catalog.py:651-652` builds
    # `config_path` the same way), `is_roformer` (`plan.py:592`). Unreachable in
    # this fork -- `mdx_c` is a blocked arch (`catalog.py:104`, refused at `:427`), so no
    # spec can carry it; kept because this is the last copy of the body.
    "mdx_c": (("config_yaml", str), ("is_roformer", bool)),
    # `pop_up_apollo_param.pull_data` (`UVR.py:4807-4808`): `config_yaml` and nothing
    # else. Apollo is the arch whose tier-3 path is still wired in `UVR.py` itself
    # (`get_model_data_from_popup`, `:702-715`). Engine reads `config_yaml`
    # (`catalog.py:651-652`, which builds `config_path` from it).
    #
    # REACHABLE THROUGH THIS FORM? Not on this machine today, and that was measured
    # rather than assumed (`tests/test_dialogs.py::_apollo_tier3` records the reading):
    # the *code* path is live -- `catalog.py:257-259` gives apollo `params_required=True`
    # with `params_table=None`, so `resolve()` has no tier 2 to fall back on and raises
    # the tier-3 error at `catalog.py:663` -- but nothing reaches it, because
    # `Catalog.names()` lists weight files only (`catalog.py:734-738`) and no root has
    # one (`models/Apollo_Models` holds `model_configs/*.yaml` and `model_data/*.json`
    # and nothing else), so `app/work_page.py:243` has an empty Model list to offer and
    # `app/controller.py:120` refuses before `session.assemble` ever resolves. The suffix
    # rule below is still made -- `test_a_typed_config_yaml_reaches_the_engine_with_its_
    # extension` supplies a throwaway weight to prove the row arms itself the moment one
    # is installed -- and note that even then the run stops later, at `plan.py:311-314`:
    # apollo's `process_method` is None, so no branch of `separate.py` reads one (the
    # module's own words, `plan.py:98`). The params file is what the dialog is for; the
    # plan is not this form's to promise.
    "apollo": (("config_yaml", str),),
}

#: The suffix a typed config name needs before the engine can use it. Upstream does not
#: store what the user typed: its mdx_c and apollo popups both write
#: `f"{var.get()}{YAML}"`, and that `YAML` is `gui_data/constants.py:128` -- the literal
#: `'.yaml'`. The two popup bodies are gone with Plan 2 Task 8's tkinter layer, so the
#: evidence is the frozen table at
#: `tests/test_dialogs.py::POPUP_EVIDENCE` (read from `UVR.py` at commit `720ca46`,
#: lines `:4750` and `:4808` of that copy) and the live check is
#: `test_the_config_yaml_suffix_is_the_one_upstreams_popup_appends`, which compares this
#: copy against `gui_data.constants.YAML` without importing it.
#:
#: WHY THIS FILE HOLDS ITS OWN LITERAL RATHER THAN `from gui_data.constants import YAML`:
#: not coupling, and an earlier draft of this note said it was, which was false --
#: `app/work_page.py`, `app/controller.py`, `app/session.py` and `app/settings_page.py`
#: all `import catalog`, and `catalog` imports `gui_data.constants`, so that module is
#: already in the app's process. The reason is *what a failure means*. The guard above
#: compares two sources and is supposed to fire when the app stops writing what
#: upstream's popup wrote. If this line read the constant instead, an upstream rename
#: with zero behavioural effect -- `YAML` to `YAML_SUFFIX`, say -- would take the whole
#: GUI half of the suite red for the wrong reason, the exact pattern this fork pins
#: against elsewhere (`tests/test_engine_purity.py` refuses to count guard 2's files for
#: the same cause). A literal plus a comparison fails only on a real mismatch.
#:
#: The name arrives bare because upstream's dropdowns that feed those variables are built
#: with `os.path.splitext(x)[0]` (the lambda at `:4802` of the same copy, called with
#: `YAML` at `:4803`), while `catalog.py:651-652` puts the value straight into
#: `config_path` and `plan.py:516-519` into an `mdx_c_configs` path that `:521-525`
#: refuses with `PlanError` when the file is not there. Collecting it verbatim therefore
#: writes a JSON whose config nobody can find.
CONFIG_YAML_SUFFIX = ".yaml"

#: Values upstream's own popup pre-sets, so a box the tkinter dialog also pre-sets is not
#: left for the user to guess: `primary_stem` 'Vocals' (`UVR.py:4860`, and `VOCAL_STEM` at
#: `:4565`), `compensate` 1.035 (`:4608`), `is_bv_model_rebalanced` 0 (`:4610`, `:4864`).
#: Everything else starts blank -- including the three MDX dims, which upstream derives
#: from the checkpoint or ONNX graph it loaded (`:4572-4585`). This GUI does not open
#: weight files, so pre-filling them would be inventing a size the engine then runs with.
FIELD_DEFAULTS: dict[str, object] = {
    "primary_stem": "Vocals",
    "compensate": 1.035,
    "is_bv_model_rebalanced": 0.0,
}

#: The one pair that is collected *together or not at all*: `UVR.py:4868-4870` disables
#: the two boxes until the "VR 5.1 model" checkbox is on and `:4872-4889` omits both keys
#: when it is off, and `plan.py:497` reads them only `if "nout" in params and "nout_lstm"
#: in params` -- presence of both is what sets `is_vr_51_model` (`:499`). So a blank pair
#: is left out of the answer, and half a pair is refused: the engine could not see it.
OPTIONAL_KEYS = frozenset({"nout", "nout_lstm"})

FAILURE_KINDS = ("vram", "missing_weight", "cpu_fallback", "cancelled")

_TITLE = {"vram": "Out of video memory",
          "missing_weight": "Weight file not found",
          "cpu_fallback": "Not running on the GPU",
          "cancelled": "Stopped by user"}

#: How long a notification stays up. The app's own number, not the wheel's: `InfoBar`
#: exposes no duration at all, so without this line a bar would sit on the page until the
#: user pressed its close button -- and Task 5's pile was exactly that.
DEFAULT_DISMISS_MS = 8000


class Notifier(QObject):
    """At most one notification on screen, and every one of them goes away.

    A `QObject` rather than a plain class for one measured reason: a slot connected to a
    bound method is held *weakly* by the binding when the receiver is a `QObject`, so the
    `InfoBar` cannot reach back and hold the queue that holds it. That cycle is what the
    crash in `app/controller.py`'s docstring is about, and these bars are transient
    surfaces named nothing like `Dialog`, so no AST guard in the suite would have looked.

    The band (a `QVBoxLayout` the host reserves) is where a bar goes when it is this
    queue's turn: a layout sibling cannot overlap the console or another notification,
    which is the property R-T7-infobar asks for. With no band the bar is a top-level
    window instead -- still one at a time.
    """

    def __init__(self, dismiss_ms: int = DEFAULT_DISMISS_MS):
        super().__init__()
        self.dismiss_ms = dismiss_ms
        self._pending: collections.deque = collections.deque()
        self._showing: tuple | None = None           # (bar, band, dismiss_ms)
        self._timer = QTimer(self)                  # one timer for the whole queue: it
        self._timer.setSingleShot(True)             # always aims at the current bar
        self._timer.timeout.connect(self._retire)

    # -- posting -------------------------------------------------------------- #

    def post(self, title: str, text: str, *, parent=None, band=None,
             dismiss_ms: int | None = None) -> fluentqt.InfoBar:
        """Build the bar now and return it; showing it is the queue's decision."""
        if band is None:
            band = _band_for(parent)
        bar = fluentqt.InfoBar(parent)
        bar.setTitle(title)
        bar.setMessage(text)
        # Measured at 1.11.2: an InfoBar's own sizeHint is 600x50 while a parented one
        # keeps Qt's default until something sizes it, and a 0-height bar in a band is
        # the (0,0)-bug in a new costume. Height comes from the hint; width comes from
        # the band, which is what a full-width line wants.
        bar.setFixedHeight(bar.sizeHint().height())
        bar.closed.connect(self._retire)
        ms = self.dismiss_ms if dismiss_ms is None else dismiss_ms
        self._pending.append((bar, band, ms))
        self._pump()
        return bar

    def current(self) -> "fluentqt.InfoBar | None":
        return None if self._showing is None else self._showing[0]

    def pending_count(self) -> int:
        return len(self._pending)

    def clear(self) -> None:
        """Drop everything queued and close what is showing -- the test seam M5 asks for.

        `DEFAULT_NOTIFIER` is process state with a lifetime longer than any page's, so a
        suite that posts a bar and then destroys its page inherits a queue full of dead
        wrappers; a test that counts bars has to be able to say "start from nothing"
        instead of monkeypatching the deque behind the class's back.
        """
        self._timer.stop()
        while self._pending:
            bar, _band, _ms = self._pending.popleft()
            _hide_and_release(bar)
        if self._showing is not None:
            bar, band, _ms = self._showing
            self._showing = None
            _hide_and_release(bar, band=band)

    # -- the one-at-a-time discipline ----------------------------------------- #

    def _pump(self) -> None:
        if self._showing is not None:
            return
        while self._pending:
            bar, band, ms = self._pending.popleft()
            if not _alive(bar):
                # Its page was destroyed while this bar waited its turn. Dropping it is
                # the whole of the handling: the entry cannot be shown, and keeping it
                # would make the *next* notification wait behind a widget that no longer
                # exists -- which is how the queue used to stall silently (M5).
                continue
            if band is not None and not _alive(band):
                # The page took its notification band with it and this bar did not belong
                # to the page (`post`'s `band` and `parent` are separate arguments). The
                # message is still true, so it is shown top-level like any bandless one;
                # touching the dead layout here is the *other* way this queue froze.
                band = None
            self._showing = (bar, band, ms)
            if band is not None:
                band.insertWidget(0, bar)
            bar.show()
            if ms > 0:
                self._timer.start(ms)
            return

    def _retire(self) -> None:
        """Close the current bar, give the slot back, and let the next one through.

        Idempotent and identity-checked, because two things call it: the timer and the
        bar's own close button (`closed` fires on that press -- measured; neither `close()`
        nor `setIsOpen(False)` emits it, which is why the timer cannot lean on them).

        Nothing here may raise on its way to `self._pump()`: the bar can belong to a page
        that has already been destroyed, and until this was guarded the `RuntimeError` Qt
        raises for a deleted C++ object came out of `_retire` *before* the pump, so one
        dead bar froze the queue and every notification after it went nowhere.
        """
        if self._showing is None:
            return
        bar, band, _ms = self._showing
        self._showing = None
        self._timer.stop()
        _hide_and_release(bar, band=band)
        self._pump()


def _alive(obj) -> bool:
    """False once the C++ object behind a wrapper is gone -- i.e. its page was destroyed.

    Asked of *either* kind a queue entry holds (the bar is a `QWidget`, the band a
    `QVBoxLayout`), which is why the touch is `objectName()` and not `isVisible()`: a
    layout has no visibility to ask about, and the band is exactly as dead as the page
    that owned it. Any call at all raises once the peer is gone, so this reads the one
    every `QObject` answers.

    `shiboken6.isValid` would be the direct question, and the wheel itself uses it
    (`.venv/.../fluentqt/foundation.py`), but it would put a binding-internals import in
    the module the surface guards read; Qt's own answer is precisely this `RuntimeError`.
    A dead bar is dropped, not shown, not counted; a dead band costs the placement, not
    the message.
    """
    try:
        obj.objectName()
    except RuntimeError:
        return False
    return True


def _hide_and_release(bar, band=None) -> None:
    """Take a bar off screen if it is still a widget, and hand it back to Python.

    The two liveness checks are independent, and that is not paranoia: a pending bar was
    never inserted into its band, so a bar the queue still owns can outlive the layout it
    was meant to sit in.
    """
    if not _alive(bar):
        return
    if band is not None and _alive(band):
        band.removeWidget(bar)
    bar.hide()
    # Measured: `deleteLater()` is not honoured inside `processEvents()` (Qt only
    # processes deferred deletes when control returns to a *real* loop), so a
    # deleteLater'd bar stays alive past the last reference and a leak here would be
    # invisible to the suite. Dropping the parent hands ownership back to Python, and
    # the widget goes now.
    bar.setParent(None)


def _band_for(parent) -> "QVBoxLayout | None":
    """The notification band a host offers, if it has one.

    `WorkPage` keeps one; a parent that does not is treated as a floating surface. Read
    by attribute on purpose: `report_failure(kind, text, parent=...)` is the interface the
    brief names, and a host that reserves space should not have to pass its layout twice.
    """
    return getattr(parent, "notification_band", None)


#: The app-wide queue. One at a time is a property of the *screen*, not of a page, and a
#: run can produce a failure while a Done bar is still up.
#:
#: The consequence is that this queue outlives the pages whose bars it holds, and a held
#: `(bar, band, ms)` tuple keeps a wrapper whose C++ object a closed dialog already
#: destroyed. `_alive` (below) is what makes that survivable -- a dead bar is dropped on
#: the way in and on the way out, a dead band costs only the placement, so neither can
#: raise and freeze the pump (M5) -- and `Notifier.clear()` is the seam that lets a test
#: empty the process-wide state it would otherwise inherit.
DEFAULT_NOTIFIER = Notifier()


def notify(title: str, text: str, parent=None, *, band=None,
           dismiss_ms: int | None = None) -> fluentqt.InfoBar:
    """Say something that is not one of the four abnormal states (`Done`, or a refusal
    the failure vocabulary has no category for)."""
    return DEFAULT_NOTIFIER.post(title, text, parent=parent, band=band,
                                 dismiss_ms=dismiss_ms)


def report_failure(kind: str, text: str, parent=None, *, band=None,
                   dismiss_ms: int | None = None) -> fluentqt.InfoBar:
    """The four abnormal states, each with its own title and the engine's own words as
    its message.

    `kind` comes from `work_page._classify_failure`, which answers `None` for text it
    does not recognise -- and `None` is not a kind here, because guessing a category is
    how a crash gets labelled a cancel. A caller that has nothing to classify calls
    `notify`.
    """
    if kind not in FAILURE_KINDS:
        raise ValueError(f"unknown failure kind {kind!r}; known: {FAILURE_KINDS}")
    return notify(_TITLE[kind], text, parent, band=band, dismiss_ms=dismiss_ms)


class ModelParamsDialog(fluentqt.ContentDialog):
    """The tier-3 refusal, as a form.

    Named `*Dialog` so the lifetime guard in `tests/test_work_page.py` scans it, and
    built as a `ContentDialog` subclass because the answer it collects belongs to the
    field list that produced it: `collected_params` coerces by the *declared* type, and
    a `NumberBox` answers a float for a key the engine indexes as an int.

    No connection here captures `self`: the dialog is answered through Fluent's own
    buttons and read back after `exec()` returns, which is the only shape in which the
    controller can call `collected_params(dlg)` on the line after asking.
    """

    def __init__(self, spec, catalog_obj, fields, parent=None):
        super().__init__(parent)
        self.spec = spec
        #: The catalog that asked. Kept so the object the answer is written into travels
        #: with the form that collected it -- `write_answer(dlg.catalog, dlg.spec, ...)`.
        self.catalog = catalog_obj
        self.fields = tuple(fields)
        self.editors: dict[str, object] = {}
        host = fluentqt.Card(self)                   # a widget, never a layout: Fluent's
        body = QVBoxLayout(host)                    # setContent takes exactly one
        self.message_label = fluentqt.Label(spec.missing_message, host)
        body.addWidget(self.message_label)
        for name, typ in self.fields:
            editor = _make_editor(name, typ, host)
            if typ is not bool:
                body.addWidget(fluentqt.Label(name, host))
            body.addWidget(editor)
            self.editors[name] = editor
        self.setContent(host)                        # `setContent`, not
        self.setTitle(f"{spec.arch} params for {spec.name}")
        self.setPrimaryButtonText("Save params")     # both verbs pinned by Task 1's
        self.setCloseButtonText("Cancel")            # name assertions


def _make_editor(name: str, typ: type, parent):
    if typ is bool:
        return fluentqt.CheckBox(name, parent)       # the label is the field's own name
    if typ in (int, float):
        editor = fluentqt.NumberBox(parent)
        default = FIELD_DEFAULTS.get(name)
        if default is not None:
            editor.setValue(default)
        return editor
    editor = fluentqt.LineEdit(parent)
    default = FIELD_DEFAULTS.get(name)
    if default is not None:
        editor.setText(str(default))
    return editor


def _is_blank(editor) -> bool:
    """No answer given yet. A checkbox always answers (`False` is a decision), a cleared
    `NumberBox` reads back NaN -- which is also how it paints: an empty field, measured --
    and an empty line edit is empty."""
    if isinstance(editor, fluentqt.CheckBox):
        return False
    if isinstance(editor, fluentqt.NumberBox):
        return math.isnan(editor.value())
    return not editor.text().strip()


def model_params_dialog(spec, cat, *, offer: bool = True, parent=None):
    """Collect the params a tier-3 model is missing, or say why nothing can be collected.

    `spec.missing_message` is shown verbatim: the hash and the wanted filename are the
    payload, and paraphrasing them is what made the CLI path unusable for a user.

    This builds the form; it does not run it. `offer=False` hands it back unshown, which
    is what a caller that means to `exec()` it needs (a `ContentDialog` is modal by
    construction -- measured -- and the tier-3 branch runs before any worker thread
    exists, so a nested loop cannot interleave with a job; see `Controller._offer_params`).
    """
    fields = PARAM_FIELDS.get(spec.arch)
    if fields is None:
        notify(f"{spec.arch}: no params template",
               "app/dialogs.py has no field list for this arch, so there is nothing for "
               "a form to collect. Write the JSON the message above names, then press "
               "START again.", parent=parent)
        return None
    dlg = ModelParamsDialog(spec, cat, fields, parent)
    if offer:
        dlg.show()
    return dlg


def collected_params(dlg) -> dict | None:
    """Read the editors back, but only if the user pressed the primary button.

    `result()` reports Fluent's own code -- `ResultNone` / `ResultPrimary` /
    `ResultSecondary`, measured as plain ints 0/1/2 on this wheel, which is why this is
    `!=` and not `is`. A dismissal is not an error and must not be reported as one: the
    caller returns quietly and the run stays unresolved. A half-filled answer is neither:
    `unanswered_fields` refuses it, because a JSON with a hole in it saves cleanly and then
    fails inside `plan.build`, with a message that blames the engine.

    Every string is collected as typed except `config_yaml`, which is the one answer that
    is a *file name* and so goes through `_config_yaml` -- the same thing upstream's
    `pull_data` does before writing its JSON.
    """
    if dlg is None or dlg.result() != fluentqt.ContentDialog.ResultPrimary:
        return None
    if unanswered_fields(dlg):
        return None
    out: dict[str, object] = {}
    for name, typ in dlg.fields:
        editor = dlg.editors[name]
        if typ is bool:
            out[name] = bool(editor.isChecked())
            continue
        if isinstance(editor, fluentqt.NumberBox):
            if _is_blank(editor):                     # optional and left alone
                continue
            value = editor.value()
            out[name] = int(value) if typ is int else float(value)
        else:
            text = editor.text().strip()
            if not text:
                continue
            out[name] = _config_yaml(text) if name == "config_yaml" else text
    return out


def _config_yaml(value: str) -> str:
    """Put the extension back on a typed config name -- once.

    Idempotent because a user who types the whole file name is not wrong: upstream's
    popup pre-selects from stems, but nothing stops a person writing `apollo_base.yaml`,
    and appending again would hand the engine `apollo_base.yaml.yaml`. The test is the
    same case-sensitive `endswith` upstream's own list filter uses
    (`UVR.py:4802`, `x.endswith(ext)`, fed with `YAML` at `:4803`), so a config named
    `X.YAML` on disk keeps the behaviour the engine's directory listing would have it.
    """
    return value if value.endswith(CONFIG_YAML_SUFFIX) else f"{value}{CONFIG_YAML_SUFFIX}"


def unanswered_fields(dlg) -> list[str]:
    """The fields that have no answer yet, in table order -- empty when the answer is
    complete.

    `OPTIONAL_KEYS` is the one exception worth a test: upstream disables those two boxes
    and omits both keys together, and `plan.py:497` only reads them as a pair, so a form
    with one filled and one not must refuse rather than drop the number the user typed.
    """
    if dlg is None:
        return []
    blank = {name for name, _typ in dlg.fields if _is_blank(dlg.editors[name])}
    missing = blank - OPTIONAL_KEYS
    if (blank & OPTIONAL_KEYS) and (OPTIONAL_KEYS - blank):
        missing |= blank & OPTIONAL_KEYS
    return [name for name, _typ in dlg.fields if name in missing]


def write_answer(cat, spec, params: dict) -> Path:
    """Persist a tier-3 answer where `resolve()` looks first (writable_dir, tier 1)."""
    return cat.write_params(spec.arch, spec.hash, params)


#: The extensions this app offers *and* scans, as the app's own copy. Held rather than
#: imported for the reason `CONFIG_YAML_SUFFIX`'s note earlier in this file gives: the pin
#: that matters is a *comparison* against `gui_data.constants.FFMPEG_EXT`, in
#: `tests/test_dialogs.py::test_the_app_audio_extensions_are_the_engines_own_list` (both
#: directions), so an upstream rename with no behavioural effect cannot redden the GUI half
#: of the suite for the wrong reason while a real change in what ffmpeg opens cannot pass
#: unnoticed here.
#:
#: One list, two users: `Pickers.AUDIO_FILTER` shows it to the file dialog and
#: `work_page.audio_files_in()` walks a folder with it. The filter used to be nine patterns
#: typed into a string (`*.wav *.mp3 *.flac *.ogg *.m4a *.opus *.aac *.aiff *.wma`, measured
#: at cd177fe); the folder scan is new in Plan 3, so the two have never disagreed -- the
#: reason they share one list is that two lists *would* disagree, and a folder pick that
#: accepts a file the multi-select dialog hides reads as the picker being broken. Lower-case
#: and dotted because the engine constant carries `"MP3"` with no dot at
#: `gui_data/constants.py:832` and both cases of `.wav`/`.WAV`; sorted so the filter string
#: is stable across runs.
AUDIO_EXTENSIONS: tuple[str, ...] = (
    ".aac", ".aiff", ".alac", ".eac3", ".flac", ".m4a", ".mkv", ".mov", ".mp2",
    ".mp3", ".mp4", ".mpc", ".mpc8", ".mpeg", ".ogg", ".opus", ".tta", ".wav",
    ".wave", ".webm", ".wma",
)


class Pickers:
    """The native dialogs this binding has no equivalent of (spec §5.2).

    `FluentMenu` cannot construct and the wheel ships no file dialog from Python, so
    Qt's own is used -- here, in one place, so the suite can say what was asked for
    without hanging on a modal dialog it cannot answer.
    """

    #: Derived from the module-level `AUDIO_EXTENSIONS` above, never typed out beside it:
    #: a filter that is a second copy of a list is a filter that drifts from the scan
    #: using the first. Qt wants each pattern as `*.wav`, hence the prefix -- which is the
    #: other half of why this is derived: a hand-typed filter string is where the dot and
    #: the star get forgotten, and a filter that matches nothing looks exactly like a
    #: folder full of the wrong files.
    AUDIO_FILTER = "Audio (" + " ".join(f"*{e}" for e in AUDIO_EXTENSIONS) + ")"

    @staticmethod
    def input_file(parent=None, start: str = "") -> "str | None":
        """The chosen path, or None. A refused pick is not a path."""
        chosen, _filter = QFileDialog.getOpenFileName(
            parent, "Choose an audio file", start or str(Path.home()),
            Pickers.AUDIO_FILTER)
        return chosen or None

    @staticmethod
    def input_files(parent=None, start: str = "") -> "list[str] | None":
        """Many paths, or None. Batch mode's entry point.

        `None` and `[]` are folded into one answer deliberately: `getOpenFileNames` returns
        an empty list on cancel, and a caller that had to tell a shrug from a refusal would
        be a caller with two bugs available. The single-file picker above stays, because a
        dialog whose many results are used as one path is how a batch accidentally starts
        at file 1 of 40.
        """
        chosen, _filter = QFileDialog.getOpenFileNames(
            parent, "Choose audio files", start or str(Path.home()), Pickers.AUDIO_FILTER)
        return chosen or None

    @staticmethod
    def input_folder(parent=None, start: str = "") -> "str | None":
        """A folder, to be scanned by `work_page.audio_files_in` -- which is where the
        extension rules live, so this picker stays a directory chooser and not a filter."""
        chosen = QFileDialog.getExistingDirectory(
            parent, "Choose a folder of audio files", start or str(Path.home()))
        return chosen or None

    @staticmethod
    def output_dir(parent=None, start: str = "") -> "str | None":
        chosen = QFileDialog.getExistingDirectory(
            parent, "Choose an output directory", start or str(Path.home()))
        return chosen or None
