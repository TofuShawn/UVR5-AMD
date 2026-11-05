"""The work page and the composition root, tested offscreen.

Nothing here clicks a `Browse` button: `QFileDialog` opens modally and would hang the
suite. The path fields are set directly, which is the same state a chosen file leaves.
"""
from __future__ import annotations

import threading
import time

import backends
import catalog
import pytest
import selections as selections_module
import settings as settings_module

from app import controller, job, main, session, work_page


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

    Task 5's terminal handlers speak through `_notify` (a local `InfoBar`); Task 7
    replaces that one body with `app/dialogs.py`. Spying here asserts *which* surface a
    terminal reaches -- the distinction a cancel is all about -- without pinning the
    widget Task 7 owns.
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


def test_tier_three_is_dismissed_until_task_7_opens_the_dialog(qapp, tmp_path, monkeypatch):
    """Tier 3 (a model with no params anywhere) is Task 7's dialog. Until then the page
    dismisses it with the catalog's own `missing_message`, which already names the model,
    the hash and the JSON it wanted."""
    spec = catalog.ModelSpec(name="1_HP-UVR", arch="vr", process_method="VR Architecture",
                             primary_stem="Vocals", secondary_stem="Instrumental",
                             weights_path=None, config_path=None, hash="deadbeef",
                             model_data=None)

    def refuse(**kwargs):
        raise catalog.UnknownModelParamsError(spec)

    monkeypatch.setattr(session, "assemble", refuse)
    ctrl, pg = _vr_page(qapp, tmp_path, monkeypatch)
    audio = tmp_path / "in.wav"
    audio.write_bytes(b"RIFF")
    pg.input_edit.setText(str(audio))

    pg.run_requested.emit()

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

def test_every_arch_label_is_given_its_full_text_width_at_the_default_window(qapp,
                                                                            monkeypatch):
    """Task 5's launch photograph showed the third tab ellipsized to `m…` and the bar
    claiming a band of empty card. Both are geometry, so both are asserted as geometry,
    and no number in here is a pixel constant: the required width is the binding's own
    `fontMetrics` advance for the label, the given width is the binding's own
    `itemHeaderGeometry`, and the window width is the one `build_app` resizes to.

    The control is a `Pivot`, chosen over `SelectorBar` by measurement (see
    `app/work_page.py`): `SelectorBar` ellipsizes `mdx` to `m…` at every width and its
    `itemGeometry` lies about it (reports 60 px for a 23 px label and still clips), so a
    `SelectorBar` could never pass a geometry test honestly. `Pivot`'s
    `itemHeaderGeometry` matches what it paints -- verified offscreen against the actual
    pixels -- which is what makes required-vs-given a real proof here rather than a
    rubber stamp.

    Three separate failures this catches, each one a different mechanism:
      * an item whose header rect is narrower than its text -- the ellipsis in the
        photograph;
      * a bar whose floor is below the sum of its labels, which is what lets a squeeze
        drop an item into overflow at all;
      * a bar taller than its own content row, i.e. the page's slack parked in the
        navigation instead of in the form.
    """
    monkeypatch.setattr(backends, "resolve", lambda force=None: _cpu_backend())
    app_obj, window = main.build_app([])
    window.show()
    for _ in range(3):
        qapp.processEvents()

    page = window.controller.work_page
    bar = page.arch_bar
    fm = bar.fontMetrics()
    texts = [item.header for item in bar.items()]
    assert texts == page.arch_names(), (
        "the bar's labels are the catalog's own arch strings, in the order the page "
        "offered them -- no invented names, no abbreviation, no filtering")
    assert not bar.hiddenItemIndexes(), (
        "an arch is in overflow, so the user cannot see or reach it at the default width")

    for index, text in enumerate(texts):
        required = fm.horizontalAdvance(text)
        given = bar.itemHeaderGeometry(index).width()
        assert given >= required, f"{text!r} needs {required}px of text, the bar gives {given}px"
    assert bar.minimumWidth() >= sum(fm.horizontalAdvance(t) for t in texts), (
        "the bar may be squeezed below the width its own labels need")
    assert bar.height() == bar.sizeHint().height(), (
        f"the bar is {bar.height()}px tall but only needs {bar.sizeHint().height()}px; "
        f"the page's slack is landing in the navigation row")


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


def test_the_settings_dialog_reaches_the_controller_only_through_a_weakref(qapp, tmp_path,
                                                                           monkeypatch):
    """The crash this pins is heap corruption (`0xc0000374`) during pytest's forced
    `gc_collect_harder()` at unconfigure, *after* every test had passed. The controller
    holds the dialog in `settings_dialog`, and the dialog had held the controller back
    strongly, so the pair was a reference cycle that only the **cyclic** collector could
    free -- and it freed the dialog's Qt-child page behind Qt's own back. Measured on the
    pinned wheel: with four dialogs accumulated, 6 crashes out of 6; a single dialog was
    clean, which is exactly why it hid from a one-test run and only bit the full suite.
    Held weakly the dialog's lifetime is the Qt parent chain's alone and no cycle forms.

    Asserted structurally, not by `gc.collect()`: the dialog is a live child of the work
    page, so it is not collectable while the page is, and a collectability assertion
    would pass for the wrong reason. What has to be true is that nothing on the dialog
    *is* the controller by strong reference.
    """
    import weakref

    ctrl, pg, page = _settings_page(qapp, tmp_path, monkeypatch)
    dialog = ctrl.settings_dialog
    assert isinstance(dialog._ctrl_ref, weakref.ReferenceType)
    assert dialog._ctrl_ref() is ctrl
    strong = [name for name, value in vars(dialog).items() if value is ctrl]
    assert not strong, f"the dialog strongly references the controller via {strong}"


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
