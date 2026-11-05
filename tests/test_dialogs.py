"""The tier-3 params dialog, the four abnormal-state surfaces, and the two pickers.

Three promises this file holds each other to:

* the catalog's refusal becomes a form, and the answer lands where `resolve()` looks
  first (R-T10-writable) -- a popup that writes somewhere else reads as "it did not
  save";
* every abnormal state spec §8 names has its own surface, and a surface that cannot say
  what happened says nothing: `_classify_failure`'s `None` is never guessed into a
  category here, and a dismissed dialog is never reported as an error;
* notifications do not pile up (ruling R-T7-infobar): one at a time, in a band the page
  reserves for it, and gone on its own -- `InfoBar` on this wheel has no auto-dismiss
  timer of any kind, so the queue in `app/dialogs.py` owns the dismissal.

Nothing here opens a real `QFileDialog`: it is modal and would hang the suite, so the
pickers are tested by replacing Qt's two static verbs. The decision under test is which
dialog the app asks for and what it does with the answer, and that decision is the only
thing the app owns.
"""
from __future__ import annotations

import ast
import gc
import json
import sys
import time
import weakref
from pathlib import Path

import backends
import catalog
import fluentqt
import gui_data.constants as const
import pytest
import separate_cli
import selections as selections_module
import settings as settings_module
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QFileDialog, QWidget

from app import controller, dialogs, session, work_page

REPO = Path(__file__).resolve().parents[1]

#: The two button texts the form sets, so a test that presses one fails loudly when the
#: dialog stops offering it.
SAVE = "Save params"
CANCEL = "Cancel"

#: A complete, honest VR answer for the throwaway weight `_tier3_spec` builds. Every key
#: is one `plan.py` reads (`vr_model_param` at :485, `primary_stem` at :560, the karaoke
#: trio at :599-602), and `3band_44100_mid` is a real params file in
#: `lib_v5/vr_network/modelparams/`, so the second pass assembles for true.
VR_ANSWER = {"vr_model_param": "3band_44100_mid", "primary_stem": "Instrumental",
             "is_karaoke": False, "is_bv_model": False, "is_bv_model_rebalanced": 0.0}


# -- fixtures and helpers ----------------------------------------------------- #

@pytest.fixture()
def queue(monkeypatch):
    """A fresh notification queue for one test.

    `DEFAULT_NOTIFIER` is module state on purpose -- the ruling is "at most one at a
    time" for the whole app -- so a test that counts bars must not inherit another
    test's pending pile. Swapping the module attribute is enough: the free functions read
    it at call time, so `app/dialogs.py` carries no test-only reset hook.
    """
    fresh = dialogs.Notifier()
    monkeypatch.setattr(dialogs, "DEFAULT_NOTIFIER", fresh)
    return fresh


def _tier3_spec(tmp_path, arch: str = "vr"):
    """A weight that exists but has no params anywhere: the real tier-3 outcome."""
    root = tmp_path / "root"
    weight_dir = {"vr": "VR_Models", "mdx": "MDX_Net_Models"}[arch]
    (root / weight_dir).mkdir(parents=True)
    (root / weight_dir / "mystery.pth").write_bytes(b"z" * 1024)
    cat = separate_cli.make_catalog(root, tmp_path / "w")
    with pytest.raises(catalog.UnknownModelParamsError) as excinfo:
        cat.resolve("mystery", required=True)
    assert excinfo.value.spec.arch == arch
    return excinfo.value.spec, cat


def spec_weights_root(tmp_path):
    return tmp_path / "root"


def _cpu_backend():
    return backends.Backend(name="cpu", torch_device="cpu",
                            ort_providers=["CPUExecutionProvider"])


def _page(qapp, tmp_path, monkeypatch, *, weights=None):
    """A work page over a throwaway root, with the notification band live.

    `backends.resolve` is patched so the start-up device line is the CPU one and costs
    nothing; the band, `_notify` and the queue are the real ones.
    """
    monkeypatch.setattr(backends, "resolve", lambda force=None: _cpu_backend())
    root = weights if weights is not None else tmp_path / "nope"
    cat = separate_cli.make_catalog(root, tmp_path / "w")
    ctrl = controller.Controller(cat=cat, st=settings_module.Settings(),
                                 selections=selections_module.Selections())
    return ctrl, ctrl.work_page


def _settle(qapp, rounds: int = 8) -> None:
    for _ in range(rounds):
        qapp.processEvents()


def _current_bar(page) -> "fluentqt.InfoBar | None":
    """The band's only widget, or None while nothing is showing."""
    item = page.notification_band.itemAt(0)
    return None if item is None else item.widget()


def _click_close(bar) -> None:
    """Press the bar's own close button -- the binding's name for it, measured."""
    button = bar.findChild(fluentqt.Button, "InfoBarCloseButton")
    assert button is not None, "InfoBar lost its close button on this wheel"
    button.click()


def _arm(dlg, text: str) -> None:
    """Arm a click on one of the dialog's own buttons.

    `exec()` runs a nested loop, so a timer scheduled *before* the call fires inside it
    (measured on this wheel: `exec()` returned 1 for a timer that clicked the primary
    button). Every test that lets the controller reach `exec()` has to arm its answer
    first -- and `fire` raising on a missing button is what stops a renamed button from
    becoming a hung suite instead of a red test.
    """
    def fire() -> None:
        for child in dlg.findChildren(fluentqt.Button):
            if child.text() == text:
                child.click()
                return
        raise AssertionError(f"the dialog has no {text!r} button")

    QTimer.singleShot(0, fire)


def _fill(dlg, values: dict) -> None:
    """Type an answer into the form the way a user fills a field."""
    for name, value in values.items():
        editor = dlg.editors[name]
        if isinstance(editor, fluentqt.CheckBox):
            editor.setChecked(bool(value))
        elif isinstance(editor, fluentqt.NumberBox):
            editor.setValue(value)
        else:
            editor.setText(str(value))


def _answer_when_offered(monkeypatch, values: dict, button: str) -> list:
    """Make the controller's dialog appear answered, the way a user answers it.

    Wraps the factory instead of the dialog: the controller holds the instance, so a
    test cannot reach in and press a button on something it never got back.
    """
    real = dialogs.model_params_dialog
    offered: list = []

    def wrapper(spec, cat, **kwargs):
        dlg = real(spec, cat, **kwargs)
        if dlg is not None:
            if values:
                _fill(dlg, values)
            _arm(dlg, button)
            offered.append(dlg)
        return dlg

    monkeypatch.setattr(dialogs, "model_params_dialog", wrapper)
    return offered


def _run_button_started(page, monkeypatch) -> list:
    """Record what reaches the one `Job` instead of launching the engine."""
    started: list = []
    monkeypatch.setattr(page.job, "start", lambda request: started.append(request))
    return started


# -- the brief's three -------------------------------------------------------- #

def test_dialog_reads_the_message_the_catalog_composed(qapp, tmp_path):
    spec, cat = _tier3_spec(tmp_path)
    assert spec.name in spec.missing_message and spec.hash in spec.missing_message
    dlg = dialogs.model_params_dialog(spec, cat, offer=True)
    assert dlg is not None


def test_answering_the_dialog_writes_the_file_the_cli_asked_for(qapp, tmp_path, monkeypatch):
    """The tier-3 contract in one line: the answer must land where `resolve()` looks
    first, or the popup reads as "it did not save" (R-T10-writable)."""
    spec, cat = _tier3_spec(tmp_path)
    params = {"primary_stem": "Vocals", "secondary_stem": "Instrumental",
              "norm_output": False, "process_method": "VR Arch",
              "dimension_c": 128, "dimension_ft": 9, "dimension_stft": 2048,
              "band_count": 1, "dim_f": 1024, "dim_t": 750, "model_samplerate": 44100,
              "n_fft": 2048, "hop_size": 512, "mp": None, "is_c_ckpt": False,
              "is_mdx23": False, "vr_arch_type": "UVR-VR-BVE-458n-255-96k-1",
              "window_size": 512, "aggression_setting": 5, "batch_size": 5,
              "post_process_threshold": 0.2, "high_end_process": None,
              "model_type": "UVR-MDX-NET-Inst_HQ_3", "mdx_scale": 1.0,
              "denoise": False, "match_input": False, "segment_size": 128,
              "overlap": 64, "is_denoise": False}
    written = dialogs.write_answer(cat, spec, params)
    # `wanted_json` is the public name for exactly this path: Catalog.resolve() builds
    # `wanted = writable_dir / <arch tier1_dir> / f"{digest}.json"` and stores it on the
    # spec when no model_data was found. `tier1_dir` is a field of the *arch table entry*,
    # not of ModelSpec -- asserting spec.tier1_dir here would be an AttributeError.
    assert written == spec.wanted_json
    assert written.is_file()
    cat2 = separate_cli.make_catalog(spec_weights_root(tmp_path), tmp_path / "w")
    assert cat2.resolve(spec.name, required=True).needs_user_params is False


def test_four_abnormal_states_have_four_distinct_surfaces(qapp, queue):
    bars = []
    for kind in ("vram", "missing_weight", "cpu_fallback", "cancelled"):
        assert kind in dialogs.FAILURE_KINDS
        bar = dialogs.report_failure(kind, "detail")
        assert bar is not None
        bars.append(bar)
    titles = [bar.title() for bar in bars]
    assert len(titles) == len(set(titles)) == 4, titles
    assert all(bar.message() == "detail" for bar in bars), (
        "the engine's own words are the message; a title is not a translation of it")


# -- what the form collects --------------------------------------------------- #

def test_the_form_offers_exactly_the_fields_the_engine_reads(qapp, tmp_path):
    """The row for the spec's arch, one editor per key, and nothing else.

    This is the silent-failure class the last-copy rule is about: a form that collects a
    key nobody reads -- or that omits one the engine indexes -- saves a JSON that still
    leaves `resolve()` blind and `plan.build` refusing.
    """
    spec, cat = _tier3_spec(tmp_path)
    dlg = dialogs.model_params_dialog(spec, cat, offer=False)
    assert list(dlg.editors) == [name for name, _type in dialogs.PARAM_FIELDS["vr"]]
    assert all(isinstance(editor, QWidget) for editor in dlg.editors.values())
    assert dlg.message_label.text() == spec.missing_message, (
        "the catalog's own words are the payload; paraphrasing them is what made the "
        "CLI path unusable for a user")
    assert dlg.isVisible() is False, "offer=False builds the form without showing it"
    assert dlg.isModal() is True, (
        "measured: a ContentDialog is modal by construction, and the tier-3 branch runs "
        "before any worker thread exists, so a nested loop cannot interleave with a run")

    shown = dialogs.model_params_dialog(spec, cat, offer=True)
    _settle(qapp)
    assert shown.isVisible() is True
    assert shown.primaryButtonText() == SAVE and shown.closeButtonText() == CANCEL
    assert shown.secondaryButtonText() == "", "a third answer nobody asked for"
    shown.done(0)


def test_the_answer_keeps_the_types_the_engine_indexes(qapp, tmp_path):
    """`params.get('nout')` reaches `plan.py:497-498`, which builds a model capacity from
    it. A `NumberBox` answers a float, so the reader coerces by the declared type or the
    JSON holds `32.0` where the engine wants `32`."""
    spec, cat = _tier3_spec(tmp_path)
    dlg = dialogs.model_params_dialog(spec, cat, offer=False)
    _fill(dlg, {"vr_model_param": "3band_44100_mid", "primary_stem": "Vocals",
                "nout": 32, "nout_lstm": 128, "is_karaoke": False,
                "is_bv_model": True, "is_bv_model_rebalanced": 0.5})
    dlg.setResult(fluentqt.ContentDialog.ResultPrimary)
    params = dialogs.collected_params(dlg)
    assert params == {"vr_model_param": "3band_44100_mid", "primary_stem": "Vocals",
                      "nout": 32, "nout_lstm": 128, "is_karaoke": False,
                      "is_bv_model": True, "is_bv_model_rebalanced": 0.5}
    assert type(params["nout"]) is int and type(params["nout_lstm"]) is int
    assert type(params["is_karaoke"]) is bool
    assert type(params["is_bv_model_rebalanced"]) is float
    assert json.loads(json.dumps(params)) == params


def test_the_capacity_keys_stay_optional_as_upstreams_are(qapp, tmp_path):
    """`nout`/`nout_lstm` are gated behind the "VR 5.1 model" checkbox upstream
    (`UVR.py:4868-4870` disables the boxes, `:4872-4889` omits the keys when it is off)
    and `plan.py:497` reads them only `if "nout" in params` -- and reading them *sets*
    `is_vr_51_model` (`:499`). A form that always collected them would turn every VR
    model into a 5.1 model, so a blank pair is left out of the answer rather than
    refused or invented."""
    spec, cat = _tier3_spec(tmp_path)
    dlg = dialogs.model_params_dialog(spec, cat, offer=False)
    _fill(dlg, VR_ANSWER)
    dlg.setResult(fluentqt.ContentDialog.ResultPrimary)
    assert dialogs.unanswered_fields(dlg) == []
    params = dialogs.collected_params(dlg)
    assert params == VR_ANSWER
    assert "nout" not in params and "nout_lstm" not in params

    _fill(dlg, {"nout": 32})
    assert dialogs.unanswered_fields(dlg) == ["nout_lstm"], (
        "half a pair is a hole plan.py cannot see: `:497` asks for both keys and would "
        "otherwise ignore the one number the user typed")
    _fill(dlg, {"nout_lstm": 128})
    assert dialogs.collected_params(dlg) == VR_ANSWER | {"nout": 32, "nout_lstm": 128}


def test_an_unanswered_field_is_not_written(qapp, tmp_path):
    """A blank field is not an answer. Upstream presets the boxes it can and refuses to
    write when the one it cannot preset is still `NONE_SELECTED`
    (`UVR.py:4891-4897`); this form keeps both halves -- the refusal names the fields,
    because a JSON that saves with a hole in it is the silent tier-3 failure."""
    spec, cat = _tier3_spec(tmp_path)
    dlg = dialogs.model_params_dialog(spec, cat, offer=False)
    _fill(dlg, VR_ANSWER)
    dlg.editors["vr_model_param"].setText("")
    dlg.setResult(fluentqt.ContentDialog.ResultPrimary)
    assert dialogs.unanswered_fields(dlg) == ["vr_model_param"]
    assert dialogs.collected_params(dlg) is None
    assert spec.wanted_json.is_file() is False, "nothing was written yet"


def test_the_mdx_form_refuses_until_the_derived_dims_are_typed(qapp, tmp_path):
    """The MDX row's three dim keys are what upstream reads out of the checkpoint or the
    ONNX graph (`UVR.py:4572-4585`); the GUI here does not open weights, so they start
    blank and the answer stays refused until the user types them -- rather than being
    invented at zero, which `plan.py:557-559` would hand to the engine as a real size."""
    spec, cat = _tier3_spec(tmp_path, arch="mdx")
    dlg = dialogs.model_params_dialog(spec, cat, offer=False)
    assert list(dlg.editors) == [name for name, _type in dialogs.PARAM_FIELDS["mdx"]]
    _fill(dlg, {"primary_stem": "Vocals"})
    dlg.setResult(fluentqt.ContentDialog.ResultPrimary)
    assert dialogs.unanswered_fields(dlg) == ["mdx_dim_f_set", "mdx_dim_t_set",
                                              "mdx_n_fft_scale_set"]
    _fill(dlg, {"compensate": 1.035, "mdx_dim_f_set": 3072, "mdx_dim_t_set": 8,
                "mdx_n_fft_scale_set": 6144})
    assert dialogs.unanswered_fields(dlg) == []
    params = dialogs.collected_params(dlg)
    assert params["mdx_dim_f_set"] == 3072 and type(params["mdx_dim_f_set"]) is int
    assert params["compensate"] == 1.035 and type(params["compensate"]) is float


def test_an_arch_with_no_template_says_so_instead_of_opening_an_empty_form(
        qapp, tmp_path, queue):
    """`mdx_c`/`onnx` are blocked archs (`catalog.py:103-107`) and `demucs` answers from
    its name table, so the only way here is a table that lost a row. The answer is the
    catalog's own instruction -- write the JSON and re-resolve -- not a form with no
    fields whose Save button would write an empty file."""
    spec, cat = _tier3_spec(tmp_path)
    assert dialogs.PARAM_FIELDS.get("demucs") is None
    no_template = catalog.ModelSpec(name=spec.name, arch="demucs",
                                    process_method=None, primary_stem=None,
                                    secondary_stem=None, weights_path=spec.weights_path,
                                    config_path=None, hash=spec.hash, model_data=None,
                                    wanted_json=spec.wanted_json)
    assert dialogs.model_params_dialog(no_template, cat, offer=True) is None
    _settle(qapp)
    bar = queue.current()
    assert bar is not None and "demucs" in bar.title()
    assert "write the json" in bar.message().lower()


# -- the controller's tier-3 branch ------------------------------------------- #

def test_start_on_an_unresolved_model_writes_the_answer_and_runs_again(
        qapp, tmp_path, monkeypatch):
    """The whole tier-3 contract in one click: START -> refusal -> form -> answer -> file
    where tier 1 is read -> the run that was refused goes to the one `Job`."""
    spec, cat = _tier3_spec(tmp_path)
    monkeypatch.setattr(backends, "resolve", lambda force=None: _cpu_backend())
    ctrl = controller.Controller(cat=cat, st=settings_module.Settings(),
                                 selections=selections_module.Selections())
    page = ctrl.work_page
    page.arch_bar.setSelectedIndex(page.arch_names().index("vr"))
    page.model_combo.setCurrentText(spec.name)
    audio = tmp_path / "in.wav"
    audio.write_bytes(b"RIFF")
    page.input_edit.setText(str(audio))
    page.output_edit.setText(str(tmp_path / "out"))

    started = _run_button_started(page, monkeypatch)
    reported = []
    monkeypatch.setattr(dialogs, "report_failure",
                        lambda *a, **kw: reported.append(a))
    offered = _answer_when_offered(monkeypatch, VR_ANSWER, SAVE)

    page.run_requested.emit()
    _settle(qapp)

    assert len(offered) == 1, "tier 3 must ask once, through the dialog"
    assert spec.wanted_json.is_file(), "the answer did not land where resolve() looks"
    assert json.loads(spec.wanted_json.read_text(encoding="utf-8")) == VR_ANSWER
    assert started, "re-resolving a tier-1 answer must hand the job to the one Job"
    assert reported == [], f"a run that went ahead was reported as a failure: {reported}"
    assert page.start_button.isEnabled() is False, "the run took the busy state"


def test_a_dismissed_dialog_writes_nothing_and_is_not_reported_as_a_failure(
        qapp, tmp_path, monkeypatch):
    """A change of mind is not an error: closing the form leaves the run unresolved and
    quiet -- no file, no retry, no failure surface."""
    spec, cat = _tier3_spec(tmp_path)
    monkeypatch.setattr(backends, "resolve", lambda force=None: _cpu_backend())
    ctrl = controller.Controller(cat=cat, st=settings_module.Settings(),
                                 selections=selections_module.Selections())
    page = ctrl.work_page
    page.arch_bar.setSelectedIndex(page.arch_names().index("vr"))
    page.model_combo.setCurrentText(spec.name)
    audio = tmp_path / "in.wav"
    audio.write_bytes(b"RIFF")
    page.input_edit.setText(str(audio))

    started = _run_button_started(page, monkeypatch)
    reported = []
    monkeypatch.setattr(dialogs, "notify", lambda *a, **kw: reported.append(a))
    monkeypatch.setattr(dialogs, "report_failure", lambda *a, **kw: reported.append(a))
    _answer_when_offered(monkeypatch, {}, CANCEL)

    page.run_requested.emit()
    _settle(qapp)

    assert spec.wanted_json.is_file() is False
    assert started == []
    assert reported == [], f"a dismissal was reported as something: {reported}"
    assert page.start_button.isEnabled() is True, "a dismissed run leaves the page idle"


def test_an_incomplete_answer_is_refused_in_words_that_name_the_fields(
        qapp, tmp_path, monkeypatch):
    """The other half of truthfulness: pressing Save on a half-empty form must not read
    as success. The console gets the field names, because that is what the user has to
    type next."""
    spec, cat = _tier3_spec(tmp_path)
    monkeypatch.setattr(backends, "resolve", lambda force=None: _cpu_backend())
    ctrl = controller.Controller(cat=cat, st=settings_module.Settings(),
                                 selections=selections_module.Selections())
    page = ctrl.work_page
    page.arch_bar.setSelectedIndex(page.arch_names().index("vr"))
    page.model_combo.setCurrentText(spec.name)
    audio = tmp_path / "in.wav"
    audio.write_bytes(b"RIFF")
    page.input_edit.setText(str(audio))

    started = _run_button_started(page, monkeypatch)
    _answer_when_offered(monkeypatch, {"vr_model_param": ""}, SAVE)

    page.run_requested.emit()
    _settle(qapp)

    text = page.console.toPlainText()
    assert spec.wanted_json.is_file() is False
    assert started == []
    assert "vr_model_param" in text and "not written" in text.lower(), text


def test_a_refusal_leaves_no_dialog_attached_to_the_page(qapp, tmp_path, monkeypatch):
    """The form is built with the page as its parent, so a dialog that stays parented is a
    `Card`, seven editors and a Catalog reference the page holds for the rest of the
    session -- one per refusal, and a user who mistypes an answer gets several. The
    controller detaches it on every exit path, which is what this counts after an answered
    run *and* after a dismissed one."""
    spec, cat = _tier3_spec(tmp_path)
    monkeypatch.setattr(backends, "resolve", lambda force=None: _cpu_backend())
    ctrl = controller.Controller(cat=cat, st=settings_module.Settings(),
                                 selections=selections_module.Selections())
    page = ctrl.work_page
    page.arch_bar.setSelectedIndex(page.arch_names().index("vr"))
    page.model_combo.setCurrentText(spec.name)
    audio = tmp_path / "in.wav"
    audio.write_bytes(b"RIFF")
    page.input_edit.setText(str(audio))
    _run_button_started(page, monkeypatch)

    _answer_when_offered(monkeypatch, VR_ANSWER, SAVE)
    page.run_requested.emit()
    _settle(qapp)
    assert spec.wanted_json.is_file()
    assert page.findChildren(fluentqt.ContentDialog) == []

    spec.wanted_json.unlink()                       # the same refusal, answered differently
    _answer_when_offered(monkeypatch, {}, CANCEL)
    page.run_requested.emit()
    _settle(qapp)
    assert spec.wanted_json.is_file() is False
    assert page.findChildren(fluentqt.ContentDialog) == []


# -- ruling R-T7-infobar ------------------------------------------------------ #

def test_two_failures_in_a_row_do_not_overlap(qapp, tmp_path, monkeypatch, queue):
    """The property the ruling names. Measured on 1.11.2: `InfoBar` has no `setDuration`
    and no auto-dismiss of any kind, and a parented one lays out at (0,0), so N failures
    used to pile N bars on the same pixel and never let go. One bar goes in the band;
    the rest wait their turn."""
    _, page = _page(qapp, tmp_path, monkeypatch)
    page.show()
    _settle(qapp)

    first = dialogs.report_failure("vram", "first", parent=page)
    second = dialogs.report_failure("missing_weight", "second", parent=page)
    third = dialogs.report_failure("cpu_fallback", "third", parent=page)
    _settle(qapp)

    assert page.notification_band.count() == 1, (
        "the band holds the current notification and nothing else")
    assert _current_bar(page) is first
    assert second.isVisible() is False and third.isVisible() is False

    _click_close(first)
    _settle(qapp)
    assert _current_bar(page) is second, "the queue must release the slot on `closed`"
    assert second.isVisible() is True
    assert page.notification_band.count() == 1

    _click_close(second)
    _settle(qapp)
    assert _current_bar(page) is third


def test_a_notification_never_covers_the_console(qapp, tmp_path, monkeypatch, queue):
    """The other half of the ruling: the pile used to hide the run output. The band is a
    layout sibling of the console, so overlap is excluded by construction -- and this
    asks about the painted rectangles rather than trusting the reasoning."""
    _, page = _page(qapp, tmp_path, monkeypatch)
    page.resize(760, 900)
    page.show()
    _settle(qapp)

    dialogs.report_failure("vram", "HIP out of memory. Tried to allocate 60.00 GiB.",
                           parent=page)
    _settle(qapp)
    bar = _current_bar(page)
    assert bar is not None and bar.isVisible() is True
    overlap = bar.geometry().intersected(page.console.geometry())
    assert overlap.isEmpty(), f"the notification sits on the console: {overlap}"
    assert bar.width() > 0 and bar.height() > 0, "a zero-sized bar is the (0,0) bug again"
    assert bar.title() == "Out of video memory"


def test_a_notification_goes_away_on_its_own(qapp, tmp_path, monkeypatch, queue):
    """Transient means transient. The queue owns one reusable single-shot timer: the
    ruling suggested `QTimer.singleShot`, but that reference cannot be cancelled, so a
    user dismissal would leave a timer holding the retired bar (measured: a bar kept by
    a pending timer is not collectable) and could retire its successor early."""
    _, page = _page(qapp, tmp_path, monkeypatch)
    page.show()
    _settle(qapp)

    first = dialogs.report_failure("cancelled", "stopped by the user", parent=page,
                                   dismiss_ms=20)
    assert _current_bar(page) is first
    deadline = time.monotonic() + 3.0
    while _current_bar(page) is first and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.005)

    assert _current_bar(page) is None, "the auto-dismiss timer never retired the bar"
    assert page.notification_band.count() == 0
    assert first.isVisible() is False


def test_every_notification_takes_its_turn(qapp, tmp_path, monkeypatch, queue):
    """Nothing may leave the queue without releasing the slot, or the app stops
    notifying for ever. The close button and the timer are the only two ways out, so the
    timer path -- the one that emits no signal at all -- is exercised here."""
    _, page = _page(qapp, tmp_path, monkeypatch)
    page.show()
    _settle(qapp)

    dialogs.notify("Done", "1 file written", parent=page, dismiss_ms=10)
    dialogs.notify("Done", "2 files written", parent=page, dismiss_ms=10)
    seen: set[str] = set()
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and len(seen) < 2:
        bar = _current_bar(page)
        if bar is not None:
            seen.add(bar.message())
        qapp.processEvents()
        time.sleep(0.005)

    assert seen == {"1 file written", "2 files written"}, seen
    deadline = time.monotonic() + 3.0
    while _current_bar(page) is not None and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.005)
    assert _current_bar(page) is None
    assert queue.current() is None
    assert queue.pending_count() == 0


def test_report_failure_refuses_a_kind_it_does_not_have(qapp, queue):
    """Truthfulness in the other direction: a caller cannot invent a category. An unknown
    kind is a bug in the caller, and the message names the four real ones."""
    with pytest.raises(ValueError) as excinfo:
        dialogs.report_failure("gpu_on_fire", "text")
    assert "gpu_on_fire" in str(excinfo.value)
    for kind in dialogs.FAILURE_KINDS:
        assert kind in str(excinfo.value)
    assert queue.current() is None


# -- the page's four terminals ------------------------------------------------- #

def test_each_terminal_lands_on_its_own_surface(qapp, tmp_path, monkeypatch, queue):
    """Task 5 left `_notify` as the seam; Task 7 fills it. The engine's own words pick
    the category, a cancel is a *signal* and not a guess, and text nobody recognises
    stays uncategorised -- four different titles out of four different events."""
    _, page = _page(qapp, tmp_path, monkeypatch)
    page.show()
    _settle(qapp)

    cases = [
        ("torch.cuda.OutOfMemoryError: HIP out of memory. Tried to allocate 8.00 GiB.",
         "Out of video memory"),
        ("OSError: [Errno 2] No such file or directory: 'mystery.pth'",
         "Weight file not found"),
        ("[warn] backend degraded: Running on CPU while the machine asked for a GPU",
         "Not running on the GPU"),
        ("RuntimeError: MIOpen has no kernel for this conv", "Failed"),
    ]
    for text, title in cases:
        page.job.failed.emit(text)
        _settle(qapp)
        bar = _current_bar(page)
        assert bar is not None, f"{text!r} produced no surface"
        assert bar.title() == title, (text, bar.title())
        assert bar.message() == text, "the surface must carry the engine's own words"
        _click_close(bar)
        _settle(qapp)

    page.job.cancelled.emit("stopped by the user")
    _settle(qapp)
    bar = _current_bar(page)
    assert bar.title() == "Stopped by user"
    assert bar.message() == "stopped by the user"
    _click_close(bar)
    _settle(qapp)

    written = tmp_path / "a.wav"
    written.write_bytes(b"RIFF")
    page.job.succeeded.emit([written])
    _settle(qapp)
    assert _current_bar(page).title() == "Done"


def test_the_console_keeps_the_engine_text_after_the_surface_gone(
        qapp, tmp_path, monkeypatch, queue):
    """The InfoBar goes away; the run record does not. A user who missed the toast can
    still read what the engine said."""
    _, page = _page(qapp, tmp_path, monkeypatch)
    text = "torch.cuda.OutOfMemoryError: HIP out of memory. Tried to allocate 8.00 GiB."
    page.job.failed.emit(text)
    _settle(qapp)
    _click_close(_current_bar(page))
    _settle(qapp)

    assert _current_bar(page) is None
    assert text in page.console.toPlainText()


# -- the VRAM obligation: triggered for real, never simulated ----------------- #

#: The one request this test makes of the hardware: 60 GiB on the 24 GiB card is a
#: refusal the ROCm allocator produces by itself. A card bigger than the request cannot
#: refuse, and the test says so out loud rather than passing quietly -- the number to
#: raise is here and nowhere else.
VRAM_REQUEST_BYTES = 60 * 2 ** 30

#: Carried by every failure exit below. Ruling R-T7-vram keeps the real trigger and rules
#: a `skipif` out, so off-GPU this suite is red *by contract*; the contract is what has to
#: be legible at the point of failure, not in a file nobody opens.
GPU_VENV_REQUIREMENT = (
    "This case is this file's only GPU-resident test, and it needs the venv this fork "
    "runs pytest in.\n"
    "  * build it: `.\\bootstrap.ps1 -Backend rocm` -- the ROCm torch wheel -- and run "
    "the suite with that venv's python: "
    "`./.venv/Scripts/python.exe -m pytest -q`;\n"
    "  * it is the same device the app asks for by flag (`separate_cli.py --device "
    "rocm`), so a venv that can run the app can run this test;\n"
    "  * `tests/test_golden_rocm.py` requires this venv too and fails without it (its "
    "autouse `_rocm_is_in_the_room`), so a cuda/cpu venv is red here on purpose -- "
    "ruling R-T7-vram: a skip would turn honest coverage of spec §8's VRAM surface into "
    "silent green;\n"
    "  * on a machine that is meant to have no GPU, deselect this one case "
    "`-k 'not vram'` rather than weaken an assertion in it."
)


def _torch_for_vram_test():
    """The ROCm torch, or a failure that names the venv that would have had it."""
    try:
        import torch
    except ImportError as exc:
        pytest.fail(f"`import torch` failed under {sys.executable} ({exc}). "
                    + GPU_VENV_REQUIREMENT)
    return torch


def test_the_vram_surface_fires_on_a_real_out_of_memory_on_the_gpu(qapp, tmp_path,
                                                                   monkeypatch, queue):
    """Spec §8's VRAM case, from a real allocation on the device `--device rocm` answers
    with, given back before the test ends.

    The plan's self-review recorded this as the one requirement of the four abnormal
    states with no honest test, and the fork's standing rule is that an absent case is
    blocked, never simulated. So this asks the ROCm allocator for 60 GiB on a 24 GiB card
    and lets it refuse. Nothing about the error is manufactured: the classification, the
    wording and the surface all come from the exception the allocator raised itself.

    COUPLING, stated where a maintainer meets it: this test needs a ROCm venv and a GPU
    in the room, and ruling R-T7-vram keeps that coupling instead of skipping it, because
    this fork runs `pytest` on the `bootstrap.ps1 -Backend rocm` venv and
    `tests/test_golden_rocm.py` is already red without it. Its cost is measured on this
    fork's RX 7900 XTX machine, not assumed: 0.9 s of call time inside the 222 s full
    suite (torch is imported by then), 1.4 s within `tests/test_dialogs.py` alone, 2.9 s
    run by itself -- the last two include the ~2 s `import torch`. Cheap enough, so the
    surface keeps being tested by the real refusal rather than by a smaller stand-in.
    Every exit below fails *diagnostically* -- it names the requirement, the command that
    answers it, and the machine's own numbers -- so a red run on a CPU-only box reads as
    a contract, not as a mystery `assert torch.cuda.is_available()`. Deselect with
    `-k 'not vram'` there; do not soften an assertion here.
    """
    torch = _torch_for_vram_test()

    backend = backends.resolve(force="rocm")
    assert backend.torch_device == "cuda", (
        f"`--device rocm` must answer with the torch device an allocator can run out of; "
        f"this venv ({sys.executable}) reported {backend!r}. "
        + GPU_VENV_REQUIREMENT)

    if not torch.cuda.is_available():
        pytest.fail(
            f"torch {torch.__version__} (hip={torch.version.hip}) under "
            f"{sys.executable} reports no reachable GPU, so `--device rocm` has no "
            f"allocator for this test to run out of. " + GPU_VENV_REQUIREMENT)

    free_before = torch.cuda.mem_get_info()[0]
    try:
        tensor = torch.empty(VRAM_REQUEST_BYTES, dtype=torch.uint8, device="cuda")
    except torch.cuda.OutOfMemoryError as exc:
        text = str(exc)
    except Exception as exc:
        raise AssertionError(
            f"the allocator answered {type(exc).__name__} ({exc}) to a request of "
            f"{VRAM_REQUEST_BYTES:,} bytes ({VRAM_REQUEST_BYTES / 2 ** 30:g} GiB) instead "
            f"of torch.cuda.OutOfMemoryError, so the needles asserted below were never "
            f"handed the out-of-memory text they exist to catch.") from exc
    else:
        # Not a defect in the surface: the card is bigger than the request. Give the
        # memory back *before* failing, so no later GPU test inherits the block.
        total = torch.cuda.mem_get_info()[1]
        device_name = torch.cuda.get_device_name(0)
        del tensor
        torch.cuda.empty_cache()
        pytest.fail(
            f"the request of {VRAM_REQUEST_BYTES:,} bytes "
            f"({VRAM_REQUEST_BYTES / 2 ** 30:g} GiB) SUCCEEDED on '{device_name}' "
            f"({total / 2 ** 30:.2f} GiB total), so nothing raised and the VRAM surface "
            f"went untested. Raise VRAM_REQUEST_BYTES above this card's capacity -- the "
            f"honest alternative is `-k 'not vram'`, not a pass.")

    # The failed request is the allocator's own bookkeeping, not a tensor: give it back
    # now, before the surface is built, so the card is clean for whatever runs next.
    torch.cuda.empty_cache()
    _settle(qapp)

    assert "out of memory" in text.lower(), text
    assert work_page._classify_failure(text) == "vram", (
        f"the ROCm wheel's own words are {text!r}, and the page's needles missed them")

    _, page = _page(qapp, tmp_path, monkeypatch)
    page.show()
    _settle(qapp)
    page.job.failed.emit(text)
    _settle(qapp)

    bar = _current_bar(page)
    assert bar is not None and bar.title() == "Out of video memory", bar
    assert "Tried to allocate" in bar.message()
    assert text in page.console.toPlainText()

    free_after = torch.cuda.mem_get_info()[0]
    assert free_after >= free_before - 2 ** 20, (
        f"the allocation was not given back: {free_before} -> {free_after} free bytes on "
        f"'{torch.cuda.get_device_name(0)}' -- this test must leave the card as it found "
        f"it, or the suite's later GPU cases fail for the wrong reason")
    assert torch.cuda.memory_allocated() == 0, torch.cuda.memory_stats()


# -- PARAM_FIELDS is the last copy of upstream's popups ------------------------ #

def _engine_param_keys() -> set[str]:
    """Every key `plan.py`/`catalog.py` take out of a params row, read from the source.

    Not from this file's comments, so the test below fails when the engine starts or
    stops reading a key rather than when somebody edits a table.
    """
    def resolve(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            value = getattr(const, node.id, None)
            return value if isinstance(value, str) else None
        return None

    holders = ("params", "model_data")
    keys: set[str] = set()
    for name in ("plan.py", "catalog.py"):
        tree = ast.parse((REPO / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            args = None
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "get"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id in holders):
                args = node.args[:1]
            elif (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                  and node.value.id in holders):
                args = [node.slice]
            elif (isinstance(node, ast.Compare)
                  and any(isinstance(op, (ast.In, ast.NotIn)) for op in node.ops)
                  and isinstance(node.comparators[0], ast.Name)
                  and node.comparators[0].id in holders):
                args = [node.left]
            for arg in args or ():
                key = resolve(arg)
                if key:
                    keys.add(key)
    return keys


def _popup_collected_keys(function_name: str) -> set[str]:
    """The keys one `UVR.py` popup writes into its JSON, read as text with `ast`.

    `UVR.py` cannot be imported on this machine (it dies at `import pyglet`), and its
    bodies go away in Task 8, so the source is the last evidence available.
    """
    tree = ast.parse((REPO / "UVR.py").read_text(encoding="utf-8"))
    funcs = [n for n in ast.walk(tree)
             if isinstance(n, ast.FunctionDef) and n.name == function_name]
    assert len(funcs) == 1, f"{function_name} found {len(funcs)} times in UVR.py"
    keys: set[str] = set()
    for node in ast.walk(funcs[0]):
        if isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    keys.add(key.value)
                elif isinstance(key, ast.Name):
                    value = getattr(const, key.id, None)
                    assert isinstance(value, str), f"{key.id} is not a string constant"
                    keys.add(value)
    return keys


@pytest.mark.parametrize("arch, popup", (
    ("vr", "pop_up_vr_param"),
    ("mdx", "pop_up_mdx_model"),
    ("mdx_c", "pop_up_mdx_c_param"),
    ("apollo", "pop_up_apollo_param"),
))
def test_the_rows_are_upstreams_own_keys_and_nothing_invented(arch, popup):
    """Key *sets*, compared against the popup body that writes them.

    There is no `pop_up_mdx_param` and no `pop_up_onnx_param` in `UVR.py`: the MDX popup
    is `pop_up_mdx_model` (:4560) and an `.onnx` model goes through that same body, which
    reads the graph at :4570-4575 and writes the mdx row -- so `onnx` has no row here.
    """
    assert _popup_collected_keys(popup) == {name for name, _type
                                           in dialogs.PARAM_FIELDS[arch]}


def test_every_field_the_dialog_collects_is_a_key_the_engine_reads():
    """The silent-failure class this table exists to avoid: a key that reads from a popup
    the engine never looks up saves a JSON that still leaves `resolve()` blind."""
    read = _engine_param_keys()
    orphans = {f"{arch}:{key}" for arch, group in dialogs.PARAM_FIELDS.items()
               for key, _type in group if key not in read}
    assert not orphans, (
        f"these keys are collected by app/dialogs.py and read by nobody: "
        f"{sorted(orphans)}; the engine reads {sorted(read)}")


def test_the_table_covers_every_arch_that_can_reach_tier_three():
    """`Catalog.resolve()` hands back a tier-3 spec only for an arch in `ARCHS` whose
    `params_required` is true (a blocked arch raises at `catalog.py:427` first, and
    Demucs answers from its name table), so the reachable set is exactly this.
    """
    reach = {arch for arch, spec in catalog.ARCHS.items() if spec.params_required}
    assert reach <= set(dialogs.PARAM_FIELDS), sorted(reach - set(dialogs.PARAM_FIELDS))
    assert set(dialogs.PARAM_FIELDS) <= set(catalog.ARCHS) | set(catalog.BLOCKED_ARCHS)


def test_the_engine_reads_every_vr_and_mdx_key_the_brief_dropped():
    """The brief's table was transcribed from the wrong body: it offered
    `dimension_c`/`dim_f`/`n_fft`/`model_type` for VR and MDX, which are keys of a
    `modelparams/*.json` file, not of the tier-1 answer. `plan.py` asks the answer for
    the *name* of that file instead, so the brief's row could never resolve."""
    read = _engine_param_keys()
    assert "vr_model_param" in read and "mdx_dim_f_set" in read
    for dropped in ("dimension_c", "dim_f", "n_fft", "model_type", "hop_size"):
        assert dropped not in read, f"{dropped} is read after all -- re-check the table"


def test_a_json_answer_written_through_the_form_resolves_at_tier_one(qapp, tmp_path):
    """The round trip the popup exists for, in the shape the CLI's refusal asks for:
    answer -> `write_params` -> a *fresh* Catalog reads tier 1 with nothing missing."""
    spec, cat = _tier3_spec(tmp_path)
    dlg = dialogs.model_params_dialog(spec, cat, offer=False)
    _fill(dlg, VR_ANSWER)
    dlg.setResult(fluentqt.ContentDialog.ResultPrimary)
    dialogs.write_answer(cat, spec, dialogs.collected_params(dlg))

    again = separate_cli.make_catalog(spec_weights_root(tmp_path),
                                      tmp_path / "w").resolve(spec.name, required=True)
    assert again.needs_user_params is False
    assert again.provenance.label == "tier1-per-model-json@writable_dir"
    assert again.model_data == VR_ANSWER
    assert again.primary_stem == "Instrumental"


# -- config_yaml is a file name, not a value ---------------------------------- #

def _apollo_tier3(tmp_path):
    """An apollo tier-3 refusal, built in a throwaway root.

    This is deliberately not the machine's own catalog: apollo cannot reach tier 3 here
    today, because `Catalog.names()` lists weight *files* (`catalog.py:734-738`) and no
    root has one (`models/Apollo_Models` carries `model_configs/*.yaml` and
    `model_data/*.json` and nothing else), so `app/work_page.py:243` has nothing to offer
    and `app/controller.py:120` refuses before `session.assemble` ever resolves. The
    weight is what the gate is, so the fixture supplies one; `params_table=None`
    (`catalog.py:257-259`) then leaves no tier 2 to answer from, and `resolve()` raises at
    `catalog.py:663` exactly as it would for a VR model.
    """
    root = tmp_path / "root"
    (root / "Apollo_Models" / "model_configs").mkdir(parents=True)
    (root / "Apollo_Models" / "apollo_demo.pth").write_bytes(b"z" * 1024)
    cat = separate_cli.make_catalog(root, tmp_path / "w")
    assert "apollo_demo.pth" in cat.names("apollo"), (
        "the name below is a name the Model list would hand the user")
    with pytest.raises(catalog.UnknownModelParamsError) as excinfo:
        cat.resolve("apollo_demo.pth", required=True)
    assert excinfo.value.spec.arch == "apollo"
    return excinfo.value.spec, cat


def test_a_typed_config_yaml_reaches_the_engine_with_its_extension(qapp, tmp_path):
    """The bug this round fixes: upstream's popup never writes what the user typed --
    `UVR.py:4808` (apollo) and `:4750` (mdx_c) both write `f"{var.get()}{YAML}"`, with
    `YAML` = `gui_data/constants.py:128` -- because the value is a *stem*: the dropdown
    that fills that variable is built with `os.path.splitext(x)[0]` (`UVR.py:4802`).
    Collected verbatim, the engine reads a stem as if it were a file name:
    `catalog.py:651-652` turns it into `config_path` (apollo's consumer, and the one this
    test measures) and `plan.py:516-519` does the same for the mdx_c row, whose `:521-525`
    `PlanError` names a yaml the user had spelled correctly apart from the extension
    nobody showed them."""
    spec, cat = _apollo_tier3(tmp_path)
    dlg = dialogs.model_params_dialog(spec, cat, offer=False)
    assert list(dlg.editors) == ["config_yaml"], "the apollo row is one field, no more"

    _fill(dlg, {"config_yaml": "apollo_base"})
    dlg.setResult(fluentqt.ContentDialog.ResultPrimary)
    params = dialogs.collected_params(dlg)
    assert params == {"config_yaml": "apollo_base" + const.YAML}, (
        "the value must be the one upstream's popup would have written, with the suffix "
        "it appends -- not the stem the form was handed")
    assert params["config_yaml"] == "apollo_base.yaml"

    written = dialogs.write_answer(cat, spec, params)
    assert json.loads(written.read_text(encoding="utf-8")) == params

    # The config itself, where the engine looks: `catalog.py:652` derives `config_path`
    # from the root that answered, and after this write the answer came from
    # `writable_dir`, so that is the root the path is read against.
    config = (cat.writable_dir / "Apollo_Models" / "model_configs"
              / params["config_yaml"])
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text("training:\n  target_instrument: Vocals\n", encoding="utf-8")

    again = separate_cli.make_catalog(spec_weights_root(tmp_path),
                                      tmp_path / "w").resolve("apollo_demo.pth",
                                                              required=True)
    assert again.needs_user_params is False
    assert again.provenance.label == "tier1-per-model-json@writable_dir"
    assert again.config_path == config, (
        f"the engine derived {again.config_path!r}; the suffix the form appends is the "
        f"whole difference between that path naming a file and naming nothing")
    assert again.config_path.is_file() is True
    assert config.parent.joinpath("apollo_base").is_file() is False, (
        "the stem on its own is not a readable config -- that is the value the fix "
        "prevents from ever being written")


def test_a_config_yaml_that_already_carries_the_extension_is_kept_as_typed(
        qapp, tmp_path):
    """The idempotence half the ruling names: a user who types the whole file name is not
    wrong, and appending a second time would write `apollo_base.yaml.yaml` -- a path even
    more certainly absent than the one the fix is about. Leading and trailing space is
    stripped first, so `" apollo_base.yaml "` is the same answer."""
    spec, cat = _apollo_tier3(tmp_path)
    dlg = dialogs.model_params_dialog(spec, cat, offer=False)
    dlg.setResult(fluentqt.ContentDialog.ResultPrimary)

    for typed in ("apollo_base.yaml", "  apollo_base.yaml  "):
        _fill(dlg, {"config_yaml": typed})
        assert dialogs.collected_params(dlg) == {"config_yaml": "apollo_base.yaml"}, (
            f"{typed!r} was answered with the suffix twice over")

    _fill(dlg, {"config_yaml": "apollo_base"})
    assert dialogs.collected_params(dlg) == {"config_yaml": "apollo_base.yaml"}, (
        "the bare stem is still the case the rule exists for")


def _popup_yaml_suffix(function_name: str) -> str:
    """What one popup's `config_yaml` value appends, resolved the way Python would.

    Same source of truth as `_popup_collected_keys`: `UVR.py` cannot be imported here, so
    the body is read with `ast`, and the appended name is looked up in
    `gui_data.constants` -- the module upstream itself imports.
    """
    tree = ast.parse((REPO / "UVR.py").read_text(encoding="utf-8"))
    funcs = [n for n in ast.walk(tree)
             if isinstance(n, ast.FunctionDef) and n.name == function_name]
    assert len(funcs) == 1, f"{function_name} found {len(funcs)} times in UVR.py"
    for node in ast.walk(funcs[0]):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if not (isinstance(key, ast.Constant) and key.value == "config_yaml"):
                continue
            assert isinstance(value, ast.JoinedStr), (
                f"{function_name} builds config_yaml as {type(value).__name__}, not an "
                f"f-string; re-read the body before trusting the suffix")
            last = value.values[-1]
            assert isinstance(last, ast.FormattedValue), ast.dump(value)
            name = last.value
            assert isinstance(name, ast.Name), ast.dump(name)
            suffix = getattr(const, name.id, None)
            assert isinstance(suffix, str), f"{name.id} is not a string constant"
            return suffix
    raise AssertionError(f"{function_name} writes no config_yaml")


@pytest.mark.parametrize("popup", ("pop_up_apollo_param", "pop_up_mdx_c_param"))
def test_the_config_yaml_suffix_is_the_one_upstreams_popup_appends(popup):
    """.yaml is read out of the two sources that decide it, not assumed by this file: the
    f-string in the popup body that writes the value, and the constant that f-string
    interpolates. If either moves, `app/dialogs.py`'s copy has to move with it -- which is
    the only way this GUI can still be said to write what upstream's popup wrote."""
    suffix = _popup_yaml_suffix(popup)
    assert suffix == const.YAML, (
        f"{popup} appends a constant that resolves to {suffix!r} while "
        f"gui_data.constants.YAML is {const.YAML!r}")
    assert dialogs.CONFIG_YAML_SUFFIX == suffix, (
        f"app/dialogs.py appends {dialogs.CONFIG_YAML_SUFFIX!r} but upstream's popup "
        f"appends {suffix!r}; the engine reads the path the value names")


# -- lifetimes ---------------------------------------------------------------- #

def test_the_params_dialog_is_collected_once_the_test_lets_go(qapp, tmp_path):
    """The shape `tests/test_work_page.py:705-765` pins for the settings dialog, copied
    here because Task 6's carry names this task as the place the mistake would be made: a
    dialog that survives its own last reference is a cycle, and the cyclic collector that
    frees it finalises a Qt object behind Qt's back -- `0xc0000374` at unconfigure, after
    every test has passed."""
    spec, cat = _tier3_spec(tmp_path)
    dlg = dialogs.model_params_dialog(spec, cat, offer=False)
    assert dlg.__class__.__name__.endswith("Dialog"), (
        "the AST guard in tests/test_work_page.py only scans classes named *Dialog, so a "
        "transient surface under another name is unguarded")

    held = weakref.ref(dlg)
    catalog_held = weakref.ref(cat)
    del dlg, cat
    gc.collect()

    assert held() is None, (
        "the params dialog outlived its own last reference: something inside it holds "
        "one back -- its own connections, or the module-level queue")
    assert catalog_held() is None, (
        "the dialog keeps the Catalog alive after the test dropped it")


def test_a_retired_notification_is_collected(qapp, tmp_path, monkeypatch, queue):
    """The other transient surface, which no rule names: an InfoBar the queue let go of
    must be collectable, or `report_failure` leaks a widget per failure for the life of
    the session. Measured on this wheel, `deleteLater()` is not enough -- Qt does not
    process deferred deletes inside `processEvents()`, so the retire path drops the
    parent and lets Python delete instead."""
    _, page = _page(qapp, tmp_path, monkeypatch)
    page.show()
    bar = dialogs.report_failure("vram", "HIP out of memory", parent=page)
    held = weakref.ref(bar)
    _click_close(bar)
    _settle(qapp)
    del bar
    gc.collect()

    assert held() is None, "the queue still holds a notification it retired"
    assert queue.current() is None
    assert page.notification_band.count() == 0


# -- pickers ------------------------------------------------------------------ #

def test_the_pickers_ask_qt_for_the_two_dialogs_the_binding_lacks(qapp, monkeypatch,
                                                                  tmp_path):
    """Spec §5.2: Fluent's Python binding has no file dialog and `FluentMenu` cannot even
    construct, so `QFileDialog` stands in -- and this is the one place that says which
    dialog, with what caption, on what filter."""
    calls = []

    def fake_open(parent, caption, directory, selected_filter, *args, **kwargs):
        calls.append(("open", parent, caption, directory, selected_filter))
        return str(tmp_path / "song.wav"), selected_filter

    def fake_dir(parent, caption, directory, *args, **kwargs):
        calls.append(("dir", parent, caption, directory))
        return str(tmp_path / "out")

    monkeypatch.setattr(QFileDialog, "getOpenFileName", fake_open)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", fake_dir)

    assert dialogs.Pickers.input_file(None, str(tmp_path)) == str(tmp_path / "song.wav")
    assert dialogs.Pickers.output_dir(None, str(tmp_path)) == str(tmp_path / "out")

    assert calls[0][1:4] == (None, "Choose an audio file", str(tmp_path))
    audio_filter = calls[0][4]
    for suffix in (".wav", ".mp3", ".flac", ".ogg"):
        assert suffix in audio_filter, audio_filter
    assert calls[1] == ("dir", None, "Choose an output directory", str(tmp_path))


def test_a_refused_pick_is_not_written_into_the_field(qapp, monkeypatch, tmp_path):
    """Cancel returns an empty string, and an empty string must not become the input
    path: the field keeps what it had and the controller's refusal stays truthful."""
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: ("", ""))
    assert dialogs.Pickers.input_file(None, str(tmp_path)) is None
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: "")
    assert dialogs.Pickers.output_dir(None, str(tmp_path)) is None

    _, page = _page(qapp, tmp_path, monkeypatch)
    page.input_edit.setText("kept.wav")
    page.browse_input.click()
    page.browse_output.click()

    assert page.input_edit.text() == "kept.wav"
    assert page.output_edit.text() == ""


def test_the_browse_buttons_go_through_the_pickers(qapp, monkeypatch, tmp_path):
    """One implementation of "pick a file" -- the same one the test above covers -- and
    each picker starts where its field already is."""
    seen = []
    monkeypatch.setattr(dialogs.Pickers, "input_file",
                        lambda parent=None, start="": (seen.append(("in", parent, start))
                                                       or str(tmp_path / "a.wav")))
    monkeypatch.setattr(dialogs.Pickers, "output_dir",
                        lambda parent=None, start="": (seen.append(("out", parent, start))
                                                       or str(tmp_path / "b")))
    _, page = _page(qapp, tmp_path, monkeypatch)
    page.input_edit.setText("kept.wav")
    page.browse_input.click()
    page.output_edit.setText("already/here")
    page.browse_output.click()

    assert [entry[0] for entry in seen] == ["in", "out"], seen
    assert seen[0][1] is page and seen[1][1] is page, "the picker must be parented"
    assert seen[0][2] == str(Path("kept.wav").resolve()) or seen[0][2] == "kept.wav"
    assert seen[1][2] == "already/here"
    assert page.input_edit.text() == str(tmp_path / "a.wav")
    assert page.output_edit.text() == str(tmp_path / "b")


# -- the surface rule, applied to what this task adds ------------------------- #

def test_the_params_dialog_content_is_a_card(qapp, tmp_path):
    """The container rule with the dialog's own verb: `ContentDialog.setContent` takes one
    widget, the app hands it a `Card`, and `dlg.content()` reads back what the dialog
    actually installed -- which a class-keyed licence cannot see."""
    spec, cat = _tier3_spec(tmp_path)
    dlg = dialogs.model_params_dialog(spec, cat, offer=False)
    assert type(dlg.content()).__name__ == "Card"


def test_the_failure_surfaces_do_not_use_a_plain_widget(qapp, tmp_path, monkeypatch,
                                                        queue):
    """`app/dialogs.py` builds a `QWidget` only where Qt itself needs one; the four
    failure surfaces and the params form are Fluent all the way down. The census over the
    shown dialog and the band is what a reviewer asked for here."""
    _, page = _page(qapp, tmp_path, monkeypatch)
    page.show()
    dialogs.report_failure("vram", "HIP out of memory", parent=page)
    _settle(qapp)

    from tests.test_ui_surfaces import _offenders

    bar = _current_bar(page)
    assert bar is not None
    assert _offenders(bar.findChildren(QWidget)) == []
    assert _offenders(page.findChildren(QWidget)) == []
