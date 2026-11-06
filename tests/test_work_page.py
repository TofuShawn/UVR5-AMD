"""The work page and the composition root, tested offscreen.

Nothing here clicks a `Browse` button: `QFileDialog` opens modally and would hang the
suite. The path fields are set directly, which is the same state a chosen file leaves.
"""
from __future__ import annotations

import ast
import collections
import sys
import threading
import time
from pathlib import Path

import backends
import catalog
import fluentqt
import pytest
import selections as selections_module
import settings as settings_module
from PySide6.QtGui import QColor, QFontMetrics, QImage, QPainter, QPen

from app import controller, dialogs, job, main, session, work_page


@pytest.fixture()
def page(qapp, tmp_path):
    import separate_cli
    cat = separate_cli.make_catalog(tmp_path / "nope", tmp_path / "w")
    return work_page.WorkPage(cat=cat, settings_obj=None, selections_obj=None)


def _cpu_backend():
    return backends.Backend(name="cpu", torch_device="cpu",
                            ort_providers=["CPUExecutionProvider"])


class _Rp:
    """The four attributes the real `session.model_count` reads (it is not patched in
    this file, so the progress arithmetic stays production code on this path too)."""

    is_secondary_model_activated = False
    secondary_model_4_stem: list = []
    pre_proc_model = None
    vocal_split_model = None


class SlowEngine:
    """A run that reports progress in small steps, so a cooperative STOP can land."""

    def __init__(self, rp, process_data):
        self.pd = process_data

    def seperate(self):
        for _ in range(300):
            self.pd["set_progress_bar"](0.001, 0.0)
            time.sleep(0.01)
        self.pd["set_progress_bar"](0.5, 0.0)


class StuckEngine:
    """A run whose cancel provably cannot land inside a few milliseconds: it parks in a
    sleep that fires no progress hook (`separate.py:1383` is the only polling point)."""

    entered = threading.Event()

    def __init__(self, rp, process_data):
        self.pd = process_data

    def seperate(self):
        StuckEngine.entered.set()
        time.sleep(0.3)
        self.pd["set_progress_bar"](0.5, 0.0)


def _request(audio, out, engine=SlowEngine):
    return job.JobRequest(
        session=session.Session(spec=None, backend=None, rp=_Rp(),
                                engine_class=engine, compute_lines=()),
        audio=audio, export_path=out)


def _deliver(qapp):
    """`QThread.wait()` does not pump the receiver's loop; see tests/test_job.py."""
    for _ in range(3):
        qapp.processEvents()


def _spy_notify(page, monkeypatch):
    """Capture the notifications the page would show.

    The terminal handlers speak through `_notify`, which since Task 7 hands the message to
    the queue in `app/dialogs.py` (Task 5 built a local `InfoBar` in that one body).
    Spying here asserts *which* surface a terminal reaches -- the distinction a cancel is
    all about -- without pinning the widget Task 7 owns; the widget, the band and the
    one-at-a-time queue are `tests/test_dialogs.py`.
    """
    shown: list[tuple[str, str]] = []
    monkeypatch.setattr(page, "_notify", lambda title, text: shown.append((title, text)))
    return shown


def _vr_page(qapp, tmp_path, monkeypatch):
    """A page on the VR arch with one model picked, and a selection frame carrying a
    secondary model -- the state a real user reaches by picking, not by clicking."""
    import separate_cli
    monkeypatch.setattr(backends, "resolve", lambda force=None: _cpu_backend())
    cat = separate_cli.make_catalog(tmp_path / "nope", tmp_path / "w")
    sel = selections_module.Selections(vr_voc_inst_secondary_model="UVR-DeNoise-Lite",
                                       vr_voc_inst_secondary_model_scale="0.5")
    ctrl = controller.Controller(cat=cat, st=settings_module.Settings(), selections=sel)
    pg = ctrl.work_page
    pg.arch_bar.setSelectedIndex(pg.arch_names().index("vr"))
    pg.model_combo.setCurrentText("UVR-DeNoise-Lite")
    return ctrl, pg


# -- the brief's six ---------------------------------------------------------- #

def test_arch_selector_offers_every_arch_the_catalog_knows_including_blocked_ones(page):
    """Not a filtered list: a blocked arch must still appear so the refusal comes from
    `resolve()` with `BLOCKED_ARCHS`' own reason, which is the payload (the CLI made
    exactly this choice for `--arch`, see separate_cli.py's ARCH_CHOICES comment)."""
    names = page.arch_names()
    assert names == sorted(page.catalog.archs())


def test_model_combo_is_populated_from_the_catalog_not_hardcoded(page, monkeypatch):
    seen = []
    monkeypatch.setattr(page.catalog, "known_names",
                        lambda arch: seen.append(arch) or ["A", "B"])
    page.refresh_models()
    assert seen == [page.current_arch()]
    assert page.model_combo.count() == 3          # NO_MODEL + the two names


def test_device_summary_says_only_what_the_backend_reports():
    text = work_page.device_summary(_cpu_backend())
    assert "cpu" in text
    assert "rocm" not in text and "cuda" not in text and "GPU" not in text


def test_the_unflagged_page_shows_the_unflagged_resolution(qapp, tmp_path, monkeypatch):
    """The review lesson, in GUI form: nothing was clicked, so whatever the machine
    resolves to with no argument is what the line must say. Monkeypatching `resolve`
    rather than reading the label twice keeps this from being a tautology."""
    import separate_cli
    monkeypatch.setattr(backends, "resolve", lambda force=None: _cpu_backend())
    cat = separate_cli.make_catalog(tmp_path / "nope", tmp_path / "w")
    fresh = work_page.WorkPage(cat=cat, settings_obj=None, selections_obj=None)
    assert "cpu" in fresh.device_label.text()
    assert fresh.compute_label.text() == "", (
        "the compute line is plan-derived and must stay empty until a plan exists")


def test_busy_state_flips_start_and_stop_together(page):
    page.set_busy(True)
    assert page.start_button.isEnabled() is False
    assert page.stop_button.isEnabled() is True
    page.set_busy(False)
    assert page.start_button.isEnabled() is True
    assert page.stop_button.isEnabled() is False


def test_failure_text_is_classified_or_refused_never_guessed(page):
    """Three cases, and the middle one is the point. `_classify_failure` returning a
    category for text it does not understand would let Task 7's dialog label a crash as
    a user cancel, and a cancel is a *different signal* (`Job.cancelled`), so no failure
    string is ever a cancel."""
    assert work_page._classify_failure("torch.cuda.OutOfMemoryError: CUDA out of memory. "
                                       "Tried to allocate 2.00 GiB") == "vram"
    assert work_page._classify_failure("some new engine exception we have never seen") is None
    assert work_page._classify_failure("stopped by the user") is None
    assert work_page._classify_failure("No such file or directory: x.pth") == "missing_weight"


# -- the rest of the page's contract ------------------------------------------ #

def test_the_job_signals_are_wired_to_the_widgets_they_drive(page):
    """The bar's only source is `Job.progress`, and the console's is `Job.console`.
    Emitting on the `Job` (not on a widget) is what Task 4's thread model guarantees,
    so this is the marshalling layer the page *is*."""
    page.job.progress.emit(42)
    assert page.progress.value() == 42
    page.job.console.emit("loading 1_HP-UVR")
    assert "loading 1_HP-UVR" in page.console.toPlainText()
    page.job.progress.emit(100)
    assert page.progress.value() == 100


def test_show_compute_keeps_the_engines_words_and_drops_only_the_cli_prefix(page):
    page.show_compute(("[cli] compute device: cpu (is_gpu_conversion=-1 from Settings; "
                       "pass --device rocm to use the GPU)",))
    text = page.compute_label.text()
    assert text.startswith("compute device: cpu (is_gpu_conversion=-1 from Settings;")
    assert "[cli]" not in text


def test_show_compute_translates_a_cli_flag_into_the_affordance_the_gui_offers(page):
    """I5(b). The clause `describe_compute_device()` ends with tells a *CLI* user to type
    `--device rocm`; this GUI has no such flag, the affordance is the `gpu_conversion`
    checkbox on the settings page, and a line that names a flag nobody can pass is the
    kind of text that gets ignored. Translated here, in the page's own translation layer:
    `separate_cli.py`'s bytes are the CLI-parity property spec §8 pins and stay as they
    are (`tests/test_separate_cli.py` is the gate, and
    `tests/test_cli_device_reporting.py` still checks the CLI's own wording)."""
    page.show_compute(("[cli] compute device: cpu (is_gpu_conversion=-1 from Settings; "
                       "pass --device rocm to use the GPU)",))
    text = page.compute_label.text()
    assert "--device" not in text, text
    assert "Settings" in text and "GPU" in text, text
    assert text.startswith("compute device: cpu (is_gpu_conversion=-1 from Settings;"), text


def test_a_refusal_clears_the_compute_line_the_previous_run_put_there(qapp, tmp_path,
                                                                     monkeypatch):
    """I5(a). `compute_label` was only ever *set*, so every refusal path left the last
    run's `compute device:` line on screen under a "not started" console: the page went
    on asserting what the engine will compute on for a run it refused to make. Driven
    through the composition root, because that is who owns the line."""
    import separate_cli

    monkeypatch.setattr(backends, "resolve", lambda force=None: _cpu_backend())
    cat = separate_cli.make_catalog(tmp_path / "nope", tmp_path / "w")
    ctrl = controller.Controller(cat=cat, st=settings_module.Settings(),
                                 selections=selections_module.Selections())
    pg = ctrl.work_page
    pg.show_compute(("[cli] compute device: cuda (torch.version.hip='6.2')",))
    assert pg.compute_label.text() != ""

    pg.input_edit.setText("")                           # the cheapest honest refusal
    pg.run_requested.emit()

    assert "not started" in pg.console.toPlainText()
    assert pg.compute_label.text() == "", (
        "a refusal left the previous run's compute line asserting what will run")


def test_the_classify_of_a_failure_reads_the_exception_line_not_the_traceback(page):
    """M6. `_classify_failure` looked for a bare `"weight"` anywhere in the text
    `Job.failed` carries -- and that text is `type: message\\n` plus four frames of engine
    source (`app/job.py:313-314`), where `lib_v5/spec_utils.py:165-188` uses
    `weight = np.zeros_like(...)` on a perfectly normal path. So any crash inside that
    function was reported as "Weight file not found". Classification is now done on the
    first line, which is the only part that *is* the exception."""
    crash = ("IndexError: list index out of range\n"
             '  File "E:/Project/UVR/UVR/lib_v5/spec_utils.py", line 165, in crop\n'
             "    weight = np.zeros_like(spec[..., :bins], dtype=np.float32)\n")
    assert work_page._classify_failure(crash) is None, work_page._classify_failure(crash)
    assert work_page._classify_failure("OSError: [WinError 2] No such file or directory: "
                                       "'Kim_Inst.pth'\nTraceback...") == "missing_weight"
    assert work_page._classify_failure(
        "torch.cuda.OutOfMemoryError: CUDA out of memory") == "vram"
    # ... and the word in the *message* still means the weight, not the traceback frame.
    assert work_page._classify_failure("RuntimeError: cannot load weight Kim_Inst.pth"
                                       ) == "missing_weight"


def _gpu_backend_with_cpu_ort():
    return backends.Backend(name="rocm", torch_device="cuda",
                            ort_providers=["CPUExecutionProvider"])


def test_a_warning_from_the_run_reaches_the_console(qapp, tmp_path, monkeypatch):
    """I4. The engine's degradations are `RuntimeWarning`s (`backends.py:43-60`,
    `separate.py:80-100`); nothing in `app/**` listened, so spec §8's fourth abnormal
    state had no producer at all and `cpu_fallback` was kept alive by a string that
    exists nowhere in the shipped code."""
    import separate_cli

    monkeypatch.setattr(backends, "resolve", lambda force=None: _gpu_backend_with_cpu_ort())
    cat = separate_cli.make_catalog(tmp_path / "nope", tmp_path / "w")
    pg = work_page.WorkPage(cat=cat, settings_obj=None, selections_obj=None)
    surfaces = []
    monkeypatch.setattr(dialogs, "report_failure",
                        lambda kind, text, **kw: surfaces.append((kind, text)))

    pg.job.warning.emit("RuntimeWarning: ONNX Runtime dropped an execution provider "
                        "for model.onnx: it was asked for ['DmlExecutionProvider']")

    assert "dropped an execution provider" in pg.console.toPlainText()
    assert surfaces == [("cpu_fallback",
                         "RuntimeWarning: ONNX Runtime dropped an execution provider "
                         "for model.onnx: it was asked for "
                         "['DmlExecutionProvider']")], surfaces


def test_a_warning_on_a_cpu_only_machine_is_not_called_a_fallback(qapp, tmp_path,
                                                                  monkeypatch):
    """The state decides, not the wording: a machine that *is* cpu has not fallen back
    from anything, and calling its warning "Not running on the GPU" would be the same
    invention this item is removing."""
    import separate_cli

    monkeypatch.setattr(backends, "resolve", lambda force=None: _cpu_backend())
    cat = separate_cli.make_catalog(tmp_path / "nope", tmp_path / "w")
    pg = work_page.WorkPage(cat=cat, settings_obj=None, selections_obj=None)
    shown = []
    monkeypatch.setattr(pg, "_notify", lambda title, text: shown.append((title, text)))

    pg.job.warning.emit("RuntimeWarning: something the page has no category for")

    assert shown == [("Warning", "RuntimeWarning: something the page has no category for")]


def test_the_fallback_state_comes_from_the_backend_and_not_from_a_needle(qapp, tmp_path,
                                                                        monkeypatch):
    """`_classify_failure`'s `Running on CPU` needle appears in the shipped code nowhere
    but a test fixture; `is_cpu_fallback` is the reviewer's condition instead -- a GPU
    backend whose ONNX Runtime providers are CPU only."""
    import separate_cli

    monkeypatch.setattr(backends, "resolve", lambda force=None: _gpu_backend_with_cpu_ort())
    cat = separate_cli.make_catalog(tmp_path / "nope", tmp_path / "w")
    pg = work_page.WorkPage(cat=cat, settings_obj=None, selections_obj=None)
    assert pg.is_cpu_fallback() is True
    assert "rocm" in pg.device_label.text()

    monkeypatch.setattr(backends, "resolve", lambda force=None: _cpu_backend())
    plain = work_page.WorkPage(cat=cat, settings_obj=None, selections_obj=None)
    assert plain.is_cpu_fallback() is False
    assert work_page._classify_failure("Running on CPU") is None, (
        "the invented needle is still a producer for the degraded surface")


# -- M1: the arch's engine parameters, on the page spec §10 puts them on -------- #

def test_the_model_parameters_expander_is_not_a_dead_control(qapp, tmp_path, monkeypatch):
    """Task 5 shipped this Expander as a placeholder -- "no content until Task 6" -- and
    Task 6 put the fields on the settings page without removing it, so the primary page
    carried a control that opened onto nothing and no test named it. Spec §10 assigns the
    arch's engine parameters to the work page, so it now shows a read-only summary of
    them plus one action that goes where they are editable."""
    import separate_cli
    from app import settings_page

    monkeypatch.setattr(backends, "resolve", lambda force=None: _cpu_backend())
    cat = separate_cli.make_catalog(tmp_path / "nope", tmp_path / "w")
    pg = work_page.WorkPage(cat=cat, settings_obj=settings_module.Settings(),
                            selections_obj=None)
    assert pg.params.contentWidget() is pg.params_host, "the expander is still empty"
    assert pg.params_summary.isVisibleTo(pg)
    pg.arch_bar.setSelectedIndex(pg.arch_names().index("vr"))
    text = pg.params_summary.text()
    for name in settings_page.ARCH_FIELD_GROUPS["vr"]:
        assert name in text, (name, text)
    assert repr(settings_module.Settings().window_size) in text, text
    # An arch the parameter table has no group for says so instead of borrowing numbers.
    pg.arch_bar.setSelectedIndex(pg.arch_names().index("apollo"))
    assert "apollo" in pg.params_summary.text()
    for name in settings_page.ARCH_FIELD_GROUPS["vr"]:
        assert name not in pg.params_summary.text(), name


def test_the_parameter_summary_follows_the_arch_and_reads_the_settings_object(
        qapp, tmp_path, monkeypatch):
    """One source, no second table: the names come from the same
    `settings_page.ARCH_FIELD_GROUPS` the editable page is built from, and the values
    from the `Settings` the controller assembled with -- so the summary cannot disagree
    with what a run will actually use."""
    import separate_cli
    from app import controller, settings_page

    monkeypatch.setattr(backends, "resolve", lambda force=None: _cpu_backend())
    cat = separate_cli.make_catalog(tmp_path / "nope", tmp_path / "w")
    ctrl = controller.Controller(cat=cat, st=settings_module.Settings(segment="70"),
                                 selections=selections_module.Selections())
    pg = ctrl.work_page
    pg.arch_bar.setSelectedIndex(pg.arch_names().index("demucs"))
    _deliver(qapp)
    text = pg.params_summary.text()
    assert "segment = '70'" in text, text
    for name in settings_page.ARCH_FIELD_GROUPS["demucs"]:
        assert name in text, (name, text)
    for name in settings_page.ARCH_FIELD_GROUPS["vr"]:
        assert name not in text, (name, text)

    # and a settings save, which hands the controller new objects, moves the summary too
    pg.arch_bar.setSelectedIndex(pg.arch_names().index("vr"))
    _deliver(qapp)
    pg.settings = settings_module.Settings(window_size=1024)
    pg.refresh_parameter_summary()
    assert "window_size = 1024" in pg.params_summary.text(), pg.params_summary.text()


def test_the_parameter_summaries_one_action_opens_the_editable_page(qapp, tmp_path,
                                                                    monkeypatch):
    """One editable surface, restated: the work page shows the numbers and the button is
    the only way to change them -- it emits the same `settings_requested` the Settings
    button emits, so the dialog stays the controller's one dialog (R-T4-onejob, and the
    one-dialog rule in `Controller._on_settings_requested`)."""
    import separate_cli

    monkeypatch.setattr(backends, "resolve", lambda force=None: _cpu_backend())
    cat = separate_cli.make_catalog(tmp_path / "nope", tmp_path / "w")
    pg = work_page.WorkPage(cat=cat, settings_obj=None, selections_obj=None)
    asked = []
    pg.settings_requested.connect(lambda: asked.append(True))
    pg.params_edit.click()
    assert asked == [True], asked
    # The summary's action is the *same signal*, so the dialog stays the controller's
    # one dialog (R-T4-onejob and `Controller._on_settings_requested`'s one-at-a-time).
    ctrl = controller.Controller(cat=cat, st=settings_module.Settings(),
                                 selections=selections_module.Selections())
    ctrl.work_page.params_edit.click()
    ctrl.work_page.settings_button.click()
    assert ctrl.settings_dialog is not None
    first = ctrl.settings_dialog
    ctrl.work_page.params_edit.click()
    assert ctrl.settings_dialog is first, "the summary's action stacked a second dialog"


def test_a_cancel_is_reported_as_a_cancel_not_as_a_failure(page, qapp, monkeypatch):
    """`Job.cancelled` and `Job.failed` are different signals and the page must not
    blend them: a STOP leaves the bar at zero and says 'Stopped', never 'Failed'."""
    shown = _spy_notify(page, monkeypatch)
    page.set_busy(True)
    page.progress.setValue(60)
    page.job.cancelled.emit("stopped by the user")
    assert page.progress.value() == 0
    assert page.start_button.isEnabled() is True
    assert shown == [("Stopped", "stopped by the user")], shown

    page.job.failed.emit("RuntimeError: MIOpen has no kernel for this conv")
    assert shown[-1][0] == "Failed", shown


# -- the composition root ----------------------------------------------------- #

def test_start_assembles_with_no_device_flag_and_the_saved_selections(qapp, tmp_path,
                                                                      monkeypatch):
    """Two facts this test exists to pin: the GUI asks for the *un-flagged* device (the
    path a user who changes nothing takes), and it passes `to_plan_dict()`, which is the
    line that makes secondary models reachable from a GUI for the first time."""
    seen = {}

    def fake_assemble(**kwargs):
        seen.update(kwargs)
        return session.Session(spec=None, backend=None, rp=_Rp(), engine_class=SlowEngine,
                               compute_lines=("[cli] compute device: cuda "
                                              "(torch.version.hip='6.2')",))

    monkeypatch.setattr(session, "assemble", fake_assemble)
    ctrl, pg = _vr_page(qapp, tmp_path, monkeypatch)
    audio = tmp_path / "in.wav"
    audio.write_bytes(b"RIFF")
    pg.input_edit.setText(str(audio))
    pg.output_edit.setText(str(tmp_path / "out"))

    pg.run_requested.emit()

    assert seen["force_device"] is None, seen
    assert seen["model"] == "UVR-DeNoise-Lite" and seen["arch"] == "vr", seen
    assert seen["selections"]["vr"]["voc_inst_secondary_model"] == "UVR-DeNoise-Lite", seen
    assert pg.compute_label.text() == ("compute device: cuda (torch.version.hip='6.2')")
    assert pg.start_button.isEnabled() is False and pg.stop_button.isEnabled() is True

    pg.job.request_cancel()
    assert pg.job.wait(10_000)
    _deliver(qapp)
    assert pg.start_button.isEnabled() is True, "the terminal must release the busy state"


def test_a_run_that_was_never_started_is_refused_in_words_not_tracebacks(qapp, tmp_path,
                                                                         monkeypatch):
    """Empty path, no model, unreadable catalog: each is a refusal the user can act on,
    and none of them may reach `Job.start()`."""
    calls = []
    monkeypatch.setattr(session, "assemble", lambda **kwargs: calls.append(kwargs))
    ctrl, pg = _vr_page(qapp, tmp_path, monkeypatch)

    pg.run_requested.emit()                                   # no input file typed
    assert calls == []
    assert "input" in pg.console.toPlainText().lower()

    pg.input_edit.setText(str(tmp_path / "missing.wav"))      # a file that is not there
    pg.run_requested.emit()
    assert calls == []
    assert "missing.wav" in pg.console.toPlainText()

    pg.input_edit.setText(str(tmp_path / "in.wav"))           # exists, but no model picked
    (tmp_path / "in.wav").write_bytes(b"RIFF")
    pg.model_combo.setCurrentText(catalog.NO_MODEL)
    pg.run_requested.emit()
    assert calls == []
    assert "model" in pg.console.toPlainText().lower()
    assert pg.start_button.isEnabled() is True, "a refusal must leave the page idle"


def test_an_assembly_refusal_is_shown_as_the_catalog_written_it(qapp, tmp_path, monkeypatch):
    """`assemble()` raises typed errors so the caller's surface can name them: a
    `CatalogError` here is a model whose weight is not on disk, and its own text is the
    payload -- the GUI must show it, not a generic crash."""
    def refuse(**kwargs):
        raise catalog.CatalogError("weight for 'Kim_Inst' is not in any model root")

    monkeypatch.setattr(session, "assemble", refuse)
    ctrl, pg = _vr_page(qapp, tmp_path, monkeypatch)
    audio = tmp_path / "in.wav"
    audio.write_bytes(b"RIFF")
    pg.input_edit.setText(str(audio))

    pg.run_requested.emit()

    assert "weight for 'Kim_Inst' is not in any model root" in pg.console.toPlainText()
    assert pg.start_button.isEnabled() is True and pg.job.is_running() is False


def test_a_save_format_the_engine_cannot_write_is_refused_before_the_run(
        qapp, tmp_path, monkeypatch):
    """C1(b) on the GUI side, through the *real* `assemble`. `save_format` was a free-text
    box, so 'mp3' reached the engine, which converted nothing and deleted the wav it made:
    the app printed "Done -- 0 file(s) written" over an empty folder. START now refuses and
    says what the legal values are, and nothing is started."""
    import separate_cli

    monkeypatch.setattr(backends, "resolve", lambda force=None: _cpu_backend())
    cat = separate_cli.make_catalog(tmp_path / "nope", tmp_path / "w")
    started: list = []
    ctrl = controller.Controller(cat=cat, st=settings_module.Settings(save_format="mp3"),
                                 selections=selections_module.Selections())
    pg = ctrl.work_page
    monkeypatch.setattr(pg.job, "start", lambda request: started.append(request))
    pg.model_combo.clear()
    pg.model_combo.addItems(["1_HP-UVR"])
    audio = tmp_path / "in.wav"
    audio.write_bytes(b"RIFF")
    pg.input_edit.setText(str(audio))

    pg.run_requested.emit()

    text = pg.console.toPlainText()
    assert "not started" in text, text
    assert "save_format" in text and "mp3" in text, text
    assert "FLAC" in text, f"the refusal must name the legal values: {text}"
    assert started == [], "a refusal may not hand a job to the worker"
    assert pg.start_button.isEnabled() is True and pg.stop_button.isEnabled() is False


def test_tier_three_hands_the_refusal_to_the_params_dialog(qapp, tmp_path, monkeypatch):
    """Tier 3 (a model with no params anywhere) is `app/dialogs.py`'s form now, so this
    pins the *handoff* rather than the dialog: the controller passes the spec it was
    handed, the catalog it holds, and the page as parent. The `None` answer -- an arch with
    no field list, which the dialog module reports by itself -- still leaves the catalog's
    own `missing_message` on the console whole, because that line names the model, the hash
    and the JSON it wanted.

    Task 5's version of this test asserted the dismissal it shipped ("until Task 7 opens
    the dialog"); the form, the answer and the retry are `tests/test_dialogs.py`.
    """
    spec = catalog.ModelSpec(name="1_HP-UVR", arch="vr", process_method="VR Architecture",
                             primary_stem="Vocals", secondary_stem="Instrumental",
                             weights_path=None, config_path=None, hash="deadbeef",
                             model_data=None)

    def refuse(**kwargs):
        raise catalog.UnknownModelParamsError(spec)

    monkeypatch.setattr(session, "assemble", refuse)
    ctrl, pg = _vr_page(qapp, tmp_path, monkeypatch)
    handed: list = []

    def no_form(held_spec, held_catalog, **kwargs):
        handed.append((held_spec, held_catalog, kwargs.get("parent")))
        return None

    monkeypatch.setattr(dialogs, "model_params_dialog", no_form)
    audio = tmp_path / "in.wav"
    audio.write_bytes(b"RIFF")
    pg.input_edit.setText(str(audio))

    pg.run_requested.emit()

    assert [entry[0] for entry in handed] == [spec], handed
    assert handed[0][1] is ctrl.catalog and handed[0][2] is pg
    text = pg.console.toPlainText()
    assert "deadbeef" in text and "1_HP-UVR" in text, text
    assert pg.job.is_running() is False


def test_stop_pressing_cancel_is_the_job_that_owns_the_flag(qapp, tmp_path, monkeypatch):
    """`cancel_requested` must reach `Job.request_cancel()` -- the engine has no stop
    flag of its own, so this single hop is the whole STOP path."""
    ctrl, pg = _vr_page(qapp, tmp_path, monkeypatch)
    shown = _spy_notify(pg, monkeypatch)
    audio = tmp_path / "in.wav"
    audio.write_bytes(b"RIFF")
    pg.job.start(_request(audio, tmp_path / "out", SlowEngine))
    assert pg.job.is_running() is True
    pg.cancel_requested.emit()
    assert pg.job.wait(10_000)
    _deliver(qapp)
    assert pg.job.is_running() is False
    assert shown == [("Stopped", "stopped by the user")], shown
    assert pg.start_button.isEnabled() is True


# -- ruling R-T6-archtabs: the arch row has to be readable at the default size -- #
#
# What this section guards is the *paint*, not the layout. The first version of this test
# compared `itemHeaderGeometry(i).width()` with `fontMetrics().horizontalAdvance()` and
# called the difference proof. It is not proof: the header rect is *defined* as the painted
# advance plus the control's own padding -- measured on the shipped `Pivot` at the default
# window, rect vs paint-font advance came back 72/40, 85/53, 60/28, 48/13 for `apollo`,
# `demucs`, `mdx`, `vr`, i.e. ~32 px of slack on every one of them -- so `given >= required`
# holds by construction for any label the control has not squeezed into overflow, and a
# `SelectorBar` hands back the *same* 72/85/60/48 rects while painting `apollo demucs m… vr`.
# What follows measures the ink the widget put on the screen instead, and
# `test_the_ink_check_rejects_the_selector_bar_the_ruling_replaced` keeps that measurement
# honest by running it against the control the ruling rejected.
#
# Why the comparison is against a *rendered reference* and not against `horizontalAdvance`,
# which is the shape this review asked for: both labels disagree with the paint, in opposite
# directions, and were measured here (see the report for the table).
#   * `bar.fontMetrics()` is 9 pt and the control paints at 14 px, so the advances it
#     reports (34/44/23/10) are ~18 % below every item's real ink width (40/52/27/13). Every
#     leg passes with slack, and the elided `mdx` still passes: its ink is 23 px wide and
#     the number to beat is 23. That is the reviewer's own case, and it is a pass.
#   * the paint font's advances (40/53/28/13) are *above* the ink the good control lays
#     down, because an advance carries the trailing side bearing: `demucs` paints 52 px of
#     ink for a 53 px advance and `mdx` 27 for 28. Asserting `ink >= advance` fails the
#     shipped, correct `Pivot`.
# Comparing the item's ink with the same string rendered in the same font and colour has
# neither problem: the widths come out equal to the pixel and the elided item falls 4 px and
# 52 ink pixels short. No pixel constant appears anywhere.
#
# Both numbers come from the paint itself:
#
#   * `bg`   -- the most common colour *inside that one header rect*, which is the bar's
#     own background as it was painted there (measured #2c2c2c under the dark theme);
#   * `ink`  -- every pixel of the rect that is not `bg`, less the selection indicator,
#     whose rect the control publishes (`selectedIndicatorGeometry(index)`, measured at
#     y=37..39 inside a header rect that runs to y=40, i.e. overlapping). The indicator
#     is the selected item's own accent bar, 40x3 px, and counting it as ink would credit
#     the item with pixels that are not its text;
#   * the *reference* -- the same label painted by `QPainter.drawText` with the font the
#     control uses for its items (`fluentqt.font_for_role(bar.itemFontRole())`; measured
#     `FontRole.Body` -> "FluentQt UI Text" 14 px) in the colour the item actually painted,
#     on a field of the colour the item actually painted behind it.
#
# Measured for the four real arch labels on the shipped `Pivot`, item vs reference:
# `apollo` 40x14/260 px against 40x14/260, `demucs` 52x12/296 against 52x12/295, `mdx`
# 27x12/157 against 27x12/154, `vr` 13x8/65 against 13x8/62 -- same width to the pixel,
# mass within 3 px and always in the item's favour. The same measurement on a
# `SelectorBar` gives `mdx` 23x9/85 against a reference 27x12/137: the ellipsis is
# shorter *and* lighter than the label it replaced, which is exactly what a user reads
# as an unreadable tab, and what the geometry version scored as a pass.

#: The arch labels the ruling photographed, used by the negative control.
ARCH_PROBE_LABELS = ("apollo", "demucs", "mdx", "vr")


def _ink_of_rect(image, x0, y0, w, h, exclude=()):
    """The pixels this header rect actually paints, plus the two colours that calibrate
    the measurement. `exclude` drops the selection indicator's own rect (see above)."""
    seen = collections.Counter()
    points = []
    for y in range(y0, min(y0 + h, image.height())):
        for x in range(x0, min(x0 + w, image.width())):
            px, py = x - x0, y - y0
            if any(ex0 <= px < ex0 + ew and ey0 <= py < ey0 + eh
                   for ex0, ey0, ew, eh in exclude):
                continue
            colour = image.pixel(x, y)
            seen[colour] += 1
            points.append((px, py, colour))
    if not points:
        return set(), 0, 0
    bg = seen.most_common(1)[0][0]
    ink = [(px, py, colour) for px, py, colour in points if colour != bg]
    if not ink:
        return set(), bg, 0
    ink_colour = collections.Counter(c for _, _, c in ink).most_common(1)[0][0]
    return {(px, py) for px, py, _ in ink}, bg, ink_colour


def _span(points):
    """(width, height, pixel count) of an ink point set."""
    if not points:
        return 0, 0, 0
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    return max(xs) - min(xs) + 1, max(ys) - min(ys) + 1, len(points)


def _reference_span(text, font, bg_rgb, ink_rgb):
    """What `text` looks like when this font paints it in this ink on this field."""
    image = QImage(400, 120, QImage.Format_RGB888)
    image.fill(QColor.fromRgb(bg_rgb))
    painter = QPainter(image)
    painter.setFont(font)
    painter.setPen(QPen(QColor.fromRgb(ink_rgb)))
    painter.drawText(8, 8 + QFontMetrics(font).ascent(), text)
    painter.end()
    points, _, _ = _ink_of_rect(image, 0, 0, image.width(), image.height())
    return _span(points)


def _header_rect(bar, index):
    """The rect the bar reserves for item `index`'s header. Both controls publish one,
    under different names, and each name exists only on its own class (measured)."""
    if hasattr(bar, "itemHeaderGeometry"):
        return bar.itemHeaderGeometry(index)
    return bar.itemGeometry(index)


def _legibility_deficits(bar, labels):
    """One message per label the bar does not paint in full; empty means every item's
    ink is at least the ink its own text produces.

    Two legs, because elision shows up two ways and a single one can be fooled. An item
    ellipsized to `m…` loses the ascender row of the `d`, so it is *shorter* (23 px of
    ink against a 27 px reference); an item whose tail is clipped loses only width; an
    item painted in a lighter colour than the reference would lose mass but not width.
    `width` is the tightest of the two on real elisions (exact equality for a full
    label, a 15 % shortfall for the ellipsized one) and `mass` is the one that cannot be
    satisfied by drawing fewer, taller strokes.
    """
    pixmap = bar.grab()
    image = pixmap.toImage().convertToFormat(QImage.Format_RGB888)
    # The reference below is painted into a `QImage` at this process's own ratio, and on a
    # scaled desktop `prepare_high_dpi_application()` can make that 1.5 or 2 -- the grab is
    # already in device pixels then, so every item's rect would be measured against a
    # reference in logical ones and the comparison would be a size test, not a legibility
    # one. Measured offscreen on this machine: ratio 1.0, and `image.width()` equals the
    # widget's own width. If this ever fails, scale `_reference_span` rather than deleting
    # the comparison.
    assert image.devicePixelRatio() == 1.0, (
        f"the grab is at {image.devicePixelRatio()} while the reference paint is at 1.0: "
        "the ink comparison is no longer comparing like with like")
    scale = image.width() / max(bar.width(), 1)
    font = fluentqt.font_for_role(bar.itemFontRole())
    deficits = []
    for index, text in enumerate(labels):
        rect = _header_rect(bar, index)
        x0, y0 = round(rect.x() * scale), round(rect.y() * scale)
        w, h = round(rect.width() * scale), round(rect.height() * scale)
        exclude = ()
        if index == bar.selectedIndex():
            indicator = bar.selectedIndicatorGeometry(index)
            exclude = ((round((indicator.x() - rect.x()) * scale),
                        round((indicator.y() - rect.y()) * scale),
                        round(indicator.width() * scale),
                        round(indicator.height() * scale)),)
        points, bg, ink_rgb = _ink_of_rect(image, x0, y0, w, h, exclude)
        got = _span(points)
        want = _reference_span(text, font, bg, ink_rgb)
        if got[2] == 0:
            deficits.append(f"{text!r}: the bar painted no ink at all in its "
                            f"{w}x{h} header rect")
            continue
        if got[0] < want[0]:
            deficits.append(f"{text!r}: the bar paints {got[0]}px of ink for a label that "
                            f"is {want[0]}px wide in this font -- the tail is elided or "
                            f"clipped (rect {_header_rect(bar, index).width()}px)")
        if got[2] < want[2]:
            deficits.append(f"{text!r}: the bar paints {got[2]} ink pixels for a label "
                            f"that needs {want[2]} in this font and colour")
    return deficits


def test_every_arch_label_is_painted_in_full_at_the_default_window(qapp, monkeypatch):
    """Task 5's launch photograph showed the third tab ellipsized to `m…`, and the
    geometry test Task 6 shipped with could not see it (see the block comment above for
    the measurements). So this test grabs the bar and measures what it painted.

    Nothing here is a pixel constant: the ink is compared with the same text painted by
    the same font, and the two colours the comparison is calibrated against are read out
    of the paint itself. The legs, each a different mechanism:

      * the bar's labels are the catalog's own arch strings, in the order the page
        offered them -- a guard renamed or filtered would not be the thing measured;
      * nothing is in overflow (`hiddenItemIndexes`), which is the *squeeze* failure,
        and a real one: under a hard squeeze the header rects do go to 0;
      * every item's painted ink is at least its own label's ink -- the *intrinsic*
        elision the ruling exists to prevent, invisible to the first two legs;
      * plus the two layout legs geometry can honestly answer: the bar cannot be
        squeezed below the width its own labels need, and it cannot be taller than its
        content (Task 5's photograph also showed the page's slack parked in the nav row).
    """
    monkeypatch.setattr(backends, "resolve", lambda force=None: _cpu_backend())
    app_obj, window = main.build_app([])
    window.show()
    for _ in range(3):
        qapp.processEvents()

    page = window.controller.work_page
    bar = page.arch_bar
    texts = [item.header for item in bar.items()]
    assert texts == page.arch_names(), (
        "the bar's labels are the catalog's own arch strings, in the order the page "
        "offered them -- no invented names, no abbreviation, no filtering")
    assert not bar.hiddenItemIndexes(), (
        "an arch is in overflow, so the user cannot see or reach it at the default width")
    deficits = _legibility_deficits(bar, texts)
    assert not deficits, f"an arch label is not fully painted: {deficits}"

    fm = QFontMetrics(fluentqt.font_for_role(bar.itemFontRole()))
    assert bar.minimumWidth() >= sum(fm.horizontalAdvance(t) for t in texts), (
        "the bar may be squeezed below the width its own labels need -- that is what "
        "lets a resize drop an item into overflow at all")
    assert bar.height() == bar.sizeHint().height(), (
        f"the bar is {bar.height()}px tall but only needs {bar.sizeHint().height()}px; "
        f"the page's slack is landing in the navigation row")


def test_the_ink_check_rejects_the_selector_bar_the_ruling_replaced(qapp, monkeypatch):
    """The negative control, and the reason the test above is a guard rather than a
    ceremony: the same measurement, the same four labels and the same header rects run
    against the control R-T6-archtabs removed.

    Ruling R-T6-archtabs replaced this page's `SelectorBar` with a `Pivot` after
    measuring that the former ellipsizes `mdx` to `m…` at 520, 728, 1200 and 2000 px
    alike -- its item rects do not depend on the bar's width, so no resize can rescue
    it. `SelectorBar` passes every *geometry* test (the reviewer's numbers, reproduced
    below: the same 72/85/60/48 rects, each wider than the label it reserves) while
    painting a shortened label, which is exactly the blind spot the old assertion had.
    So this asserts two things about one measurement: that it reports the elision, and
    that the width proxy it replaced does not.

    Like `tests/test_fluent_binding.test_the_measured_binding_holes_are_still_holes`,
    this pins a defect of the pinned wheel rather than of the app: if a future Fluent-Qt
    stops eliding here, this test fails and the answer is to re-take the measurement,
    not to delete the ink check the ruling depends on.

    It is measured in the same initialised theme as the shipped bar, and that is not
    cosmetic: in a process where `fluentqt.initialize_resources()` has never run this
    control picks different rects for itself -- measured, 116/116/74/60 for the first three
    labels instead of 72/85/60 -- and the elision disappears. The ruling was taken on the
    app's own conditions, so the control that defends it has to be taken there too.
    """
    monkeypatch.setattr(backends, "resolve", lambda force=None: _cpu_backend())
    app_obj, window = main.build_app([])
    window.show()
    for _ in range(3):
        qapp.processEvents()
    pivot = window.controller.work_page.arch_bar
    labels = [item.header for item in pivot.items()]
    assert labels == list(ARCH_PROBE_LABELS), (
        "the two controls must be measured over the same labels, or this compares "
        "nothing -- re-read the arch list the catalog publishes")

    bar = fluentqt.SelectorBar()
    for text in labels:
        bar.addItem(fluentqt.SelectorBarItem(text))
    bar.resize(pivot.width(), bar.sizeHint().height())
    bar.show()
    for _ in range(3):
        qapp.processEvents()

    # The proxy the ruling replaced: on this control it is satisfied by construction --
    # every header rect is wider than the label inside it, because the rect is *defined*
    # as the advance plus the control's padding -- and the third label is still elided.
    # The advance is taken from the font the item is painted with, so this is the
    # friendliest possible reading of the geometry test, not the widget's own 9 pt font.
    fm = QFontMetrics(fluentqt.font_for_role(bar.itemFontRole()))
    for index, text in enumerate(labels):
        assert _header_rect(bar, index).width() > fm.horizontalAdvance(text), (
            "the geometry proxy started biting; re-read the ruling before trusting a "
            "rect-vs-advance comparison again")

    deficits = _legibility_deficits(bar, labels)
    assert any("mdx" in line for line in deficits), (
        f"the ink check no longer rejects the control the ruling rejected: {deficits}")
    assert _legibility_deficits(pivot, labels) == [], (
        "the same measurement must pass for the control the app actually ships")


# -- the settings dialog (Task 6) --------------------------------------------- #

@pytest.fixture()
def appdata(tmp_path, monkeypatch):
    """`Controller.save_settings` writes `settings.config_file()`, which reads `APPDATA`.
    Repointed here for the same reason `tests/test_settings.py` does it: the shared file
    is what every other test's `load()` would otherwise pick up."""
    base = tmp_path / "Roaming"
    base.mkdir()
    monkeypatch.setenv("APPDATA", str(base))
    return base / "UVR-fork"


def _settings_page(qapp, tmp_path, monkeypatch):
    ctrl, pg = _vr_page(qapp, tmp_path, monkeypatch)
    pg.settings_requested.emit()
    return ctrl, pg, ctrl.settings_dialog.page


def test_the_settings_button_opens_a_page_covering_every_editable_field(qapp, tmp_path,
                                                                        monkeypatch):
    """One click, the whole engine surface: 50 settings + 33 selections, in a dialog the
    controller owns. And no second `Job` appears while it is open (R-T4-onejob) -- the
    dialog edits state, it does not run anything."""
    ctrl, pg, page = _settings_page(qapp, tmp_path, monkeypatch)
    assert page is not None and ctrl.settings_dialog.isVisible() is True
    assert len(page.widget_names()) == 83, len(page.widget_names())
    assert not hasattr(page, "job"), "the settings dialog built a second Job"
    assert ctrl.work_page.job is pg.job, "the page still owns the one Job"
    assert pg.settings_button.text() == "Settings"


def test_opening_the_settings_twice_reuses_the_one_dialog(qapp, tmp_path, monkeypatch):
    """83 widgets is not something to stack. A second click raises what is already open."""
    ctrl, pg, page = _settings_page(qapp, tmp_path, monkeypatch)
    pg.settings_requested.emit()
    assert ctrl.settings_dialog.page is page, "a second dialog was built over the first"


def test_saving_from_the_dialog_writes_the_shared_file_and_hands_the_state_on(qapp, tmp_path,
                                                                              monkeypatch,
                                                                              appdata):
    """The pairing this root exists to keep: what the page validated becomes the objects
    the next `START` assembles from, and the file the next launch loads."""
    ctrl, pg, page = _settings_page(qapp, tmp_path, monkeypatch)
    page.set_field_value("window_size", 768)
    page.set_field_value("vr_voc_inst_secondary_model", "UVR-BVE-458n")
    ctrl.settings_dialog.save_button.click()

    assert settings_module.load().window_size == 768
    assert selections_module.load().vr_voc_inst_secondary_model == "UVR-BVE-458n"
    assert ctrl.settings.window_size == 768
    assert ctrl.selections.vr_voc_inst_secondary_model == "UVR-BVE-458n"
    assert pg.selections is ctrl.selections
    assert ctrl.settings_dialog.isVisible() is False
    assert str(settings_module.config_file()) in pg.console.toPlainText()


def test_a_refused_save_writes_nothing_and_keeps_the_dialog_open(qapp, tmp_path, monkeypatch,
                                                                 appdata):
    """The other half of `apply`'s promise, seen from the button: the file is not created,
    the dialog stays so the field can be fixed, and the console names it."""
    ctrl, pg, page = _settings_page(qapp, tmp_path, monkeypatch)
    before = ctrl.settings.window_size
    page.set_field_value("window_size", 640)                  # a clean change ...
    page.set_field_value_raw("save_format", ["FLAC"])         # ... and one refused value
    ctrl.settings_dialog.save_button.click()

    assert not (appdata / "settings.json").exists(), "a refused save touched the file"
    assert ctrl.settings.window_size == before, "the controller adopted refused values"
    assert ctrl.settings_dialog.isVisible() is True
    assert "save_format" in pg.console.toPlainText(), pg.console.toPlainText()


def test_a_locked_settings_file_is_reported_in_the_console_and_keeps_the_dialog_open(
        qapp, tmp_path, monkeypatch, appdata):
    """The I/O half of the save path, seen from the button -- the case Task 6 left silent.

    A `settings.json` open in an editor is the ordinary way this fails on Windows, and
    before `apply()` caught `OSError` it disappeared the way the refresh refusal did:
    PySide6 prints an exception that escapes a slot and carries on, so the dialog stayed
    open, the console stayed empty, and the user could only press Save again. Here the same
    click has to leave a line that names the file's problem, keep the dialog open, and not
    adopt what never reached the disk.
    """
    import sys

    escapes = []
    monkeypatch.setattr(sys, "excepthook",
                        lambda exc_type, value, tb: escapes.append(value))
    ctrl, pg, page = _settings_page(qapp, tmp_path, monkeypatch)
    before = ctrl.settings.window_size
    page.set_field_value("window_size", 999)

    def locked(self, target=None):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(settings_module.Settings, "save", locked)
    ctrl.settings_dialog.save_button.click()

    assert not escapes, f"the save error escaped the slot again: {escapes!r}"
    text = pg.console.toPlainText()
    assert "could not be written" in text and "PermissionError" in text, text
    assert ctrl.settings_dialog.isVisible() is True, (
        "a save that failed closed the dialog -- the user has nowhere to press again")
    assert (appdata / "settings.json").exists() is False
    assert ctrl.settings.window_size == before, "the controller adopted an unwritten value"


def test_the_settings_dialog_is_collected_once_the_test_lets_go(qapp, tmp_path, monkeypatch):
    """The crash this pins is heap corruption (`0xc0000374`) during pytest's forced
    `gc_collect_harder()` at unconfigure, *after* every test had passed. The controller
    holds the dialog in `settings_dialog`, and the dialog had held the controller back
    strongly, so the pair was a reference cycle that only the **cyclic** collector could
    free -- and it freed the dialog's Qt-child page behind Qt's own back. Measured on the
    pinned wheel: with four dialogs accumulated, 6 crashes out of 6; a single dialog was
    clean, which is exactly why it hid from a one-test run and only bit the full suite.

    The fix has two halves and the assertion has to cover both, which is why there are
    two checks here rather than one:

      * `SettingsDialog._ctrl_ref` is a weakref, so the dialog never closes a cycle with
        the controller that owns it;
      * the save button is connected through `settings_page._weak_slot`, not to a
        callable that captures the dialog. That half is invisible to any inspection of
        `vars(dialog)`: it puts the cycle *inside* the dialog, and `vars(dialog)` still
        names no controller.

    Measured on this wheel, and the two halves split along an unexpected line, so each
    gets its own guard:
      * `connect(lambda: self.save())` -- the dialog is **still alive after
        `gc.collect()`** and the lifetime assertion below fails (PySide stores a plain
        callable strongly, so the C++ button holds the lambda, the lambda holds the
        dialog, and a top-level dialog is never deleted while Qt lives). On that same
        mutation everything Task 6's test inspected still looked clean:
        `isinstance(_ctrl_ref, ReferenceType)` True, `_ctrl_ref() is ctrl` True, no
        controller among `vars(dialog)`'s values -- measured, not argued, and exactly the
        gap this finding named.
      * `connect(self.save)` -- a *bound method*: the dialog really is collected
        (measured: the lifetime assertion passes), because the binding keeps the receiver
        of a method slot weakly. So this half is caught by the AST guard below, which
        rejects both forms, and the `isinstance` leg above stays for the controller
        back-reference, which is collectable either way and so invisible to the drop.

    So: built with `parent=None` (no Qt parent can keep it alive), every reference the
    test held dropped, `gc.collect()`, and the object must be gone. Same shape as
    `tests/test_settings_page.py:270-291` for the page, and the shape Tasks 7 and 8 copy
    for every dialog they add.
    """
    import gc
    import weakref

    ctrl, pg = _vr_page(qapp, tmp_path, monkeypatch)
    dialog = controller.SettingsDialog(ctrl, settings_obj=settings_module.Settings(),
                                       selections_obj=selections_module.Selections())
    assert isinstance(dialog._ctrl_ref, weakref.ReferenceType), (
        "half one of the fix: the dialog reaches the controller through a weak reference")

    held = weakref.ref(dialog)
    controller_held = weakref.ref(ctrl)
    del dialog, ctrl, pg
    gc.collect()

    assert held() is None, (
        "the dialog survived its own last reference: something inside it holds a strong "
        "reference back -- the save button must connect through "
        "`settings_page._weak_slot`, exactly as `SettingsPage` connects its own buttons")
    assert controller_held() is None, (
        "the controller outlived the test: the dialog keeps it alive, so `_ctrl_ref` is "
        "not a weak reference after all")


def test_no_dialog_in_the_app_holds_itself_in_a_slot_connection():
    """The lifetime rule above, made mechanical for every dialog the app will have.

    Tasks 7 and 8 add dialogs, and the one-line mistake that reintroduces the crash is a
    bound method handed to `connect()`. `WorkPage` and `Job` may keep theirs -- they are
    alive for the whole session, so their cycles never finalise behind Qt's back -- but a
    dialog is built, shown, closed and dropped, which is precisely the object whose cycle
    kills the process. Ruling: inside a `class *Dialog`, a slot is either a module-level
    function, a signal's `emit`, or a wrapper built from a weak reference
    (`settings_page._weak_slot` / `_deferred_slot`).

    AST rather than introspection so it says *where*: a structural scan of a live dialog
    cannot see the difference between `connect(self.save)` and `connect(_weak_slot(...))`
    until the process is already dying.
    """
    repo_root = Path(__file__).resolve().parents[1]
    offenders = []
    for path in sorted((repo_root / "app").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.ClassDef) and node.name.endswith("Dialog")):
                continue
            for call in ast.walk(node):
                if not (isinstance(call, ast.Call)
                        and isinstance(call.func, ast.Attribute)
                        and call.func.attr == "connect"):
                    continue
                for arg in list(call.args) + [kw.value for kw in call.keywords]:
                    if isinstance(arg, ast.Lambda) and any(
                            isinstance(n, ast.Name) and n.id == "self"
                            for n in ast.walk(arg)):
                        offenders.append(f"{path.name}:{arg.lineno} {node.name} "
                                         f"connects a lambda over `self`")
                    elif (isinstance(arg, ast.Attribute)
                          and isinstance(arg.value, ast.Name) and arg.value.id == "self"):
                        offenders.append(f"{path.name}:{arg.lineno} {node.name} "
                                         f"connects the bound method `self.{arg.attr}`")
    assert not offenders, (
        "a dialog holds one of its own widgets through a strong self-reference, which is "
        "the reference cycle that takes the process down at shutdown -- connect through "
        "`settings_page._weak_slot` instead:\n" + "\n".join(offenders))


# -- teardown: ruling R-T4-shutdown ------------------------------------------- #

def test_closing_the_window_drains_the_job_before_the_widgets_go(qapp, tmp_path, monkeypatch):
    """The `QThread` is a child of the `Job` and its destructor hard-aborts the process
    while the thread is running, so `closeEvent` has to come through `shutdown()` and a
    True return is what lets the window go. The cancelled terminal still arrives."""
    ctrl, pg = _vr_page(qapp, tmp_path, monkeypatch)
    shown = _spy_notify(pg, monkeypatch)
    window = main.AppWindow(ctrl)
    window.setContentWidget(pg)
    audio = tmp_path / "in.wav"
    audio.write_bytes(b"RIFF")
    pg.job.start(_request(audio, tmp_path / "out", SlowEngine))
    pg.set_busy(True)                                   # what a controller START does
    assert pg.job.is_running() is True
    window.show()

    window.close()

    assert window.isVisible() is False, "a drained job must let the window close"
    assert pg.job.is_running() is False, "close must leave the Job drained"
    _deliver(qapp)
    assert shown == [("Stopped", "stopped by the user")], shown
    assert pg.start_button.isEnabled() is True, "the terminal releases the busy state"


def test_a_close_is_refused_while_the_thread_is_still_live(qapp, tmp_path, monkeypatch):
    """The other half of `shutdown()`'s return: False means the thread is still running,
    so destroying the widgets now is exactly the abort this rule exists to avoid. The
    window stays open and says so; nothing is disposed."""
    StuckEngine.entered.clear()
    ctrl, pg = _vr_page(qapp, tmp_path, monkeypatch)
    window = main.AppWindow(ctrl, drain_ms=5)
    window.setContentWidget(pg)
    audio = tmp_path / "in.wav"
    audio.write_bytes(b"RIFF")
    pg.job.start(_request(audio, tmp_path / "out", StuckEngine))
    pg.set_busy(True)                                   # what a controller START does
    assert StuckEngine.entered.wait(5.0), "the engine never reached the worker thread"
    window.show()

    window.close()

    assert window.isVisible() is True, "a live thread must refuse the close"
    assert pg.job.is_running() is True
    assert "winding down" in pg.console.toPlainText()
    assert pg.start_button.isEnabled() is False, "a refused close keeps the busy state"

    assert pg.job.shutdown(10_000) is True, "the engine must finish after the bound"
    _deliver(qapp)


# -- I10: the OS accessibility switches, read before the first widget --------- #

@pytest.fixture()
def motion_mode_restored():
    """`set_motion_mode` is process-wide state on the wheel's own policy object, so a
    test that changes it hands back what it found -- otherwise the surface tests that run
    after this one are measured under a mode the app never chose, and the failure they
    would report is a duration of 0 with nothing in the traceback to say why."""
    before = fluentqt.current_motion_mode()
    yield
    fluentqt.set_motion_mode(before)


def test_the_os_animation_switch_decides_the_motion_mode(monkeypatch,
                                                         motion_mode_restored):
    """Spec §5.3 keeps reduce-motion, and until this item nothing in `app/**` called
    `fluentqt.set_motion_mode` at all: the wheel resolves its own default (`Full`), so
    `motion_policy().resolvedDuration(250)` answered 250 and the GUI animated on a machine
    whose user had turned animation effects off. That is an unmet requirement dressed as a
    deferred one, which is what this item is about.

    The measurement is the duration, not the enum, because the duration is the fact the
    user sees. `Disabled` and not `Reduced` for the off position: the OS flag is binary, so
    there is no producer for "animate, but quicker" -- and 50 ms still moves (ruling stated
    in `app/main.py`'s docstring, with the human-verified half: no `WM_SETTINGCHANGE`
    listener, so a mid-session change needs a restart).
    """
    for answer, mode, duration in ((False, fluentqt.MotionMode.Disabled, 0),
                                   (True, fluentqt.MotionMode.Full, 250),
                                   (None, fluentqt.MotionMode.Full, 250)):
        monkeypatch.setattr(main, "animations_enabled", lambda a=answer: a)
        chosen = main.system_motion_mode()
        fluentqt.set_motion_mode(chosen)
        assert chosen is mode, f"animations_enabled()={answer} chose {chosen!r}"
        got = fluentqt.motion_policy().resolvedDuration(250)
        assert got == duration, (
            f"animations_enabled()={answer}: the wheel still resolves 250 ms to {got}, "
            f"so the page animates anyway")


def test_the_call_that_refuses_is_not_read_as_animations_off(monkeypatch):
    """`SystemParametersInfoW(SPI_GETANIMATION)` with `uiParam=0` returns rc 0 and leaves
    `iMinAnimate` at 0 -- a failure that looks exactly like an answer, on a machine whose
    animations are on. `animations_enabled()` therefore answers None for a refused call,
    and None must not reach the `Disabled` branch.
    """
    class _RefusingUser32:
        def SystemParametersInfoW(self, *_args):
            return 0

    class _Windll:                                   # stands in for `ctypes.windll`
        user32 = _RefusingUser32()

    monkeypatch.setattr(main.ctypes, "windll", _Windll(), raising=True)
    assert main.animations_enabled() is None
    assert main.high_contrast_enabled() is False
    assert main.system_motion_mode() is fluentqt.MotionMode.Full


def test_the_machine_is_actually_read(qapp):
    """The two calls above are exercised against fakes, so this one asks the real OS and
    pins only that an answer comes back in the domain -- the shape of the probe, not its
    value, which is the user's setting and none of this suite's business. Measured on the
    machine this was written on: animations on, contrast theme off."""
    if sys.platform != "win32":
        pytest.skip("the SPI calls this reads are Windows-only")
    assert main.animations_enabled() in (True, False)
    assert isinstance(main.high_contrast_enabled(), bool)


def test_a_contrast_theme_is_not_relabeled_light_or_dark(monkeypatch):
    """Spec §5.3's other retained demand, and `system_theme()` mapped two of the wheel's
    three modes: `Theme.HighContrast` exists and `set_theme` takes it (both measured), so a
    user on a contrast theme was being told their accessibility state was a colour
    preference. The registry value still decides when the flag is not set."""
    monkeypatch.setattr(main, "high_contrast_enabled", lambda: True)
    assert main.system_theme() is fluentqt.Theme.HighContrast
    monkeypatch.setattr(main, "high_contrast_enabled", lambda: False)
    assert main.system_theme() in (fluentqt.Theme.Light, fluentqt.Theme.Dark)


def test_build_app_publishes_both_before_anything_is_shown(qapp, monkeypatch,
                                                           motion_mode_restored):
    """The composition root is where these belong: `set_motion_mode` after the first hover
    leaves that hover animated, because the duration is resolved per call. So the app-wide
    entry point has to be what asks the machine -- and the fact this pins is that a real
    `build_app` leaves the process resolving 0 ms on a machine that asked for no motion."""
    monkeypatch.setattr(backends, "resolve", lambda force=None: _cpu_backend())
    monkeypatch.setattr(main, "animations_enabled", lambda: False)
    monkeypatch.setattr(main, "high_contrast_enabled", lambda: False)
    app_obj, window = main.build_app([])
    try:
        assert app_obj is qapp, "build_app must not make a second QApplication"
        assert fluentqt.motion_policy().resolvedDuration(250) == 0, (
            "the app-wide entry point did not publish the machine's answer")
        assert fluentqt.current_theme() in (fluentqt.Theme.Light, fluentqt.Theme.Dark)
    finally:
        window.controller.request_close(5_000)
