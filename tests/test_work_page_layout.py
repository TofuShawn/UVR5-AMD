"""Plan 3 Task 3: the work page's layout, its four controls, and what they persist.

The requirement this file exists to hold is a sentence the user said out loud after three
days of screenshots: the page has to be as easy to use as official UVR 5.6 and better where
5.6 is ugly, and "one screen" is not the goal -- "沒必要執著在於一屏". So the page is now
two labelled segments in a scroll body over a fixed action row, and the three choices that
decide where a stem lands and what computes it (output folder, format, GPU) are on the page
instead of behind a Settings button in an 83-field dialog.

What is *not* here is as load-bearing as what is. There is no "what to extract" column:
`separate_cli.effective_settings` overwrites `chosen_process_method` from the resolved
model (`separate_cli.py:282-283`) and `plan.py:457` branches on the result, so a radio that
could not change which model runs would be a control that lies, and the Model list stays
the only such choice. That argument is `test_no_stem_scope_control_exists_on_the_page`.

The stubs come from `tests/test_work_page.py` (`_ui_ctrl`, `_cpu_backend`) and
`tests/test_work_page_batch.py` (`_stub_assemble`), cross-module the way
`tests/test_ui_surfaces.py:28` already imports from `tests/test_fluent_binding.py`.
"""
from __future__ import annotations

import json
from pathlib import Path

import backends
import fluentqt
import pytest
import settings as settings_module
import ui_prefs
from app import controller, dialogs, session, work_page
from tests.test_work_page import SlowEngine, _Rp, _cpu_backend, _ui_ctrl
from tests.test_work_page_batch import _stub_job_start, _wav_files


# -- the layout --------------------------------------------------------------- #

def test_building_the_page_writes_nothing_to_disk(qapp, tmp_path, monkeypatch):
    """A control initialised *from* a stored value must not write on the way in.

    This is the bug the `ToggleSwitch` hole would otherwise have caused: a radio created
    with `setChecked(True)` fires `toggled`, and if that is wired straight to the
    persisting slot, opening the window rewrites `settings.json` and reports a change
    nobody made -- in a fork that treats a config write as a user action.

    Pinned as bytes *and* `st_mtime_ns`, with an absent file allowed as one of the two
    before-states: the session-scoped `APPDATA` sandbox (`tests/conftest.py:30-68`) is
    shared by every test that writes a setting, so "the file does not exist yet" would be
    a statement about test order rather than about this constructor. A rewrite with
    identical content still moves the mtime, which is why both are compared.
    """
    target = settings_module.config_file()
    before = (target.read_bytes(), target.stat().st_mtime_ns) if target.exists() else None
    ctrl = _ui_ctrl(qapp, tmp_path, monkeypatch)
    assert ctrl.settings == settings_module.Settings()
    if before is None:
        assert not target.exists(), f"building the page wrote {target}"
    else:
        assert (target.read_bytes(), target.stat().st_mtime_ns) == before, (
            f"building the page rewrote {target}")


def test_the_page_is_two_segments_over_a_fixed_action_row(qapp, tmp_path, monkeypatch):
    """The two segments say what each half is for, and the action row does not scroll.

    Four assertions, each a different failure: labels absent (the page is still a wall of
    widgets); the action row *inside* the scroll area (START, the progress bar and the
    reason drift off the bottom during a batch, which is 5.6's complaint rather than its
    fix); the console inside it too (reading engine text while scrolling is not possible);
    and the form controls outside it (the whole reason a scroll body is allowed).
    """
    ctrl = _ui_ctrl(qapp, tmp_path, monkeypatch)
    page = ctrl.work_page        # the root is kept: its slots are what the page's signals
                                 # write through, and a throwaway reference silences them
    assert page.segment_one.text().startswith("1 ·"), page.segment_one.text()
    assert page.segment_two.text().startswith("2 ·"), page.segment_two.text()
    assert page.action_row_host.parent() is page
    for control in (page.input_edit, page.model_combo, page.arch_bar, page.output_edit,
                    page.file_list, page.params, page.gpu_check, page.start_button,
                    page.stop_button, page.settings_button, page.reason_label,
                    page.progress, page.count_label, page.console, page.summary_label,
                    page.device_label, page.compute_label):
        assert page.isAncestorOf(control) or control.parent() is page, control


def test_no_scroll_view_is_built_by_the_page(qapp, tmp_path, monkeypatch):
    """Pinned because the alternative was found by a segfault, not by reading.

    Two `WorkPage`s each holding a `ScrollView` crash at interpreter exit inside
    `QWidget::destroy` under `destroyQCoreApplication` -- measured on the pinned wheel,
    with the suite reporting 739 passed and rc 139, and with eight of the same builds
    going clean when the scroll view is replaced by a plain `Card` around the same body.
    `setContentWidget` transfers ownership of the content widget to the binding, so a
    scroll area built once per test is one whose destruction Qt and Shiboken both try to
    perform. `settings_page` gets away with its scroll area because the suite builds that
    page inside a dialog, rarely, and never eight times in one process.

    So there is no scroll area here, and the reason it is not needed is a number rather
    than an assertion of intent: the page's own `sizeHint()` at the width `app/main.py`
    launches at, with the queue showing and the parameters collapsed, against the height
    that window asks for. If that stops being true, the scroll area belongs at the
    *window* level -- one per process, owned by the chrome -- and this is where whoever
    gets there finds out why it is not already there.
    """
    ctrl = _ui_ctrl(qapp, tmp_path, monkeypatch)
    page = ctrl.work_page
    assert page.findChildren(fluentqt.ScrollView) == []
    page.resize(900, 200)
    assert page.sizeHint().height() < 900, page.sizeHint().height()


def test_no_stem_scope_control_exists_on_the_page(qapp, tmp_path, monkeypatch):
    """The 5.6 left column's Vocal/Instrumental/4-Stems radios are deliberately absent,
    and this test is what keeps the absence a decision instead of an oversight: adding a
    `RadioButton` whose label promises a stem set the engine derives from the *model*
    would be a control that lies. The Model list is the only such choice on the page, and
    the format radios are the only others -- `session.SAVE_FORMATS`, three, closed.
    """
    ctrl = _ui_ctrl(qapp, tmp_path, monkeypatch)
    page = ctrl.work_page        # the root is kept: its slots are what the
                                 # page's signals write through, and a
                                 # throwaway reference silences them
    assert list(page.format_radios) == list(session.SAVE_FORMATS)
    stems = ("vocal", "instrumental", "accompaniment", "other", "4 stem", "four stem",
             "drums", "bass")
    texts = [r.text() for r in page.format_radios.values()]
    assert not [t for t in texts if any(s in t.lower() for s in stems)], texts
    assert page.arch_bar in page.findChildren(fluentqt.Pivot), \
        "the arch switch survived the re-layout"


# -- the four controls, and what they persist --------------------------------- #

@pytest.mark.parametrize("fmt", list(session.SAVE_FORMATS))
def test_the_format_choice_is_a_closed_set_written_to_disk(qapp, tmp_path, monkeypatch,
                                                           fmt):
    """Three radios, and exactly one checked afterwards.

    `QButtonGroup(exclusive=True)` rather than Qt's auto-exclusive-by-parent: the two
    halves of the requirement are "the engine can never be handed `'mp3'` or `'WAV '`"
    (C1's free-text path, which `session.check_save_format` still refuses at the last
    moment) and "the page cannot show two formats at once", and neither is safe to rely on
    a Fluent child widget's parent behaviour to provide.
    """
    ctrl = _ui_ctrl(qapp, tmp_path, monkeypatch)
    page = ctrl.work_page
    # Start from the *other* side of the change, so every parametrised case really is one:
    # `Settings.save_format` defaults to WAV, and asserting "a write happened" on a click
    # that already had WAV checked would be asserting it on the default value instead.
    other = next(f for f in session.SAVE_FORMATS if f != fmt)
    page.format_radios[other].setChecked(True)
    assert ctrl.settings.save_format == other
    page.format_radios[fmt].setChecked(True)
    assert ctrl.settings.save_format == fmt
    raw = json.loads(settings_module.config_file().read_text(encoding="utf-8"))
    assert raw["save_format"] == fmt
    assert [f for f, r in page.format_radios.items() if r.isChecked()] == [fmt]
    assert f"{fmt} output" in page.truth_summary()
    assert page.summary_label.text() == page.truth_summary(), \
        "the strip is republished from what is stored, not from what was clicked"
    assert f"settings saved: save_format = {fmt}" in page.console.toPlainText()


def test_re_selecting_the_current_format_writes_nothing(qapp, tmp_path, monkeypatch):
    """A click that changes nothing must not rewrite `settings.json`. The signal is
    `toggled`, whose guard is therefore the emit condition and not the write, and a group
    that re-emitted would make every stray click on an already-selected box a config write
    with a console line claiming a change."""
    ctrl = _ui_ctrl(qapp, tmp_path, monkeypatch)
    page = ctrl.work_page
    emits: list[str] = []
    page.format_requested.connect(emits.append)
    page.format_radios[page._checked_format()].setChecked(True)
    assert emits == []
    page.gpu_check.setChecked(page.gpu_check.isChecked())
    assert "settings saved" not in page.console.toPlainText()


def test_the_gpu_choice_writes_the_one_field_the_plan_reads(qapp, tmp_path, monkeypatch):
    """Not decorative, and the citation is the proof: `plan.py:329` reads
    `settings.gpu_conversion`, and the GUI passes `force_device=None`, so
    `effective_settings`' device override (`separate_cli.py:284-286`) never fires and this
    checkbox is the only thing in the GUI that can set it. The assertion that reaches the
    engine is that `assemble` is handed *this* object -- which is what makes a click on the
    box and the next run the same fact rather than two facts that happen to agree."""
    ctrl = _ui_ctrl(qapp, tmp_path, monkeypatch)
    page = ctrl.work_page
    started: list = []
    monkeypatch.setattr(session, "assemble",
                        lambda **kw: started.append(kw) or session.Session(
                            spec=None, backend=_cpu_backend(), rp=_Rp(),
                            engine_class=SlowEngine, compute_lines=()))
    monkeypatch.setattr(page, "selected_model", lambda: "UVR-MDX-NET-Inst_HQ_3")
    _stub_job_start(page, monkeypatch, [], failing=set())
    page.input_edit.setText(str(_wav_files(tmp_path, ["one.wav"])[0]))
    # Both directions, from the other side of the default: `gpu_conversion` defaults to
    # False, so a lone `setChecked(False)` emits no `toggled`, writes nothing, and asserts
    # the default back at itself -- which is what this test used to do.
    page.gpu_check.setChecked(True)
    assert ctrl.settings.gpu_conversion is True
    assert json.loads(settings_module.config_file().read_text(encoding="utf-8"))[
        "gpu_conversion"] is True
    page.gpu_check.setChecked(False)
    assert ctrl.settings.gpu_conversion is False
    assert json.loads(settings_module.config_file().read_text(encoding="utf-8"))[
        "gpu_conversion"] is False
    page.run_requested.emit()
    assert started, "START assembled"
    assert started[-1]["st"] is ctrl.settings
    assert started[-1]["st"].gpu_conversion is False
    assert started[-1]["force_device"] is None


def test_a_refused_write_reverts_the_control_and_says_which_one(qapp, tmp_path,
                                                                monkeypatch):
    """The everyday Windows failure is a `settings.json` open in an editor. If the radio
    kept showing the value that could not be stored, the page would be promising a format
    the next run will not use -- and the next run is the one that writes the stems.
    The revert is asserted by reading the widgets back, because that is what the user
    sees, and the console line is asserted with the field name in it, because two
    controls can now fail and "it did not save" does not say what did not.
    """
    ctrl = _ui_ctrl(qapp, tmp_path, monkeypatch)
    page = ctrl.work_page
    monkeypatch.setattr(settings_module.Settings, "save",
                        lambda self, path=None: (_ for _ in ()).throw(
                            PermissionError(13, "denied")))
    before = ctrl.settings.save_format
    page.format_radios["FLAC"].setChecked(True)
    assert ctrl.settings.save_format == before, "a refused write changes nothing"
    assert [f for f, r in page.format_radios.items() if r.isChecked()] == [before]
    text = page.console.toPlainText()
    assert "not saved" in text and "PermissionError" in text, text
    page.gpu_check.setChecked(not ctrl.settings.gpu_conversion)
    assert ctrl.settings.gpu_conversion is settings_module.Settings().gpu_conversion
    assert page.gpu_check.isChecked() is settings_module.Settings().gpu_conversion


def test_an_open_settings_dialog_does_not_survive_a_work_page_write(qapp, tmp_path,
                                                                    monkeypatch):
    """One writer at a time. `SettingsPage.apply` builds all 50 values from its own
    widgets, so a dialog left open across a work-page change holds a draft that would undo
    that change on its next Save -- silently, from a button the user pressed for another
    reason. Closing it is visible; losing an edit to a stale Save is not recoverable."""
    ctrl = _ui_ctrl(qapp, tmp_path, monkeypatch)
    ctrl._on_settings_requested()
    assert ctrl.settings_dialog is not None
    ctrl.work_page.format_radios["MP3"].setChecked(True)
    assert ctrl.settings_dialog is None
    assert "settings dialog closed" in ctrl.work_page.console.toPlainText()
    # And the dialog the user opens next is built from the object that was just written,
    # so it shows the truth rather than the previous run's numbers -- `field_value` is the
    # settings page's own "what this would save right now", which is exactly the half a
    # stale draft used to get wrong.
    ctrl._on_settings_requested()
    assert ctrl.settings_dialog.page.field_value("save_format") == "MP3"


def test_the_summary_strip_repeats_nothing_it_can_get_wrong(qapp, tmp_path, monkeypatch):
    """The line a user skims before pressing START, and the reason it is derived rather
    than cached: five of its six terms come out of live widgets, so changing one cannot
    leave the other five asserting the old run. This is also the surface that would have
    told whoever sent the first screenshot that the page was on `apollo` with no weights
    installed for it -- the state that made the app look broken rather than unconfigured.
    """
    ctrl = _ui_ctrl(qapp, tmp_path, monkeypatch)
    page = ctrl.work_page        # the root is kept: its slots are what the
                                 # page's signals write through, and a
                                 # throwaway reference silences them
    files = _wav_files(tmp_path, ["one.wav", "two.wav"])
    page.add_extra_files([str(f) for f in files])
    page.format_radios["FLAC"].setChecked(True)
    page.gpu_check.setChecked(True)
    assert "GPU" in page.truth_summary(), page.truth_summary()
    page.gpu_check.setChecked(False)
    text = page.truth_summary()
    assert "FLAC output" in text and "CPU" in text, text
    assert "2 file(s)" in text, text
    assert page.summary_label.text() == text
    page.input_edit.setText("")
    page.format_radios["WAV"].setChecked(True)
    assert "WAV output" in page.summary_label.text()
    assert "FLAC" not in page.summary_label.text()
    # The count follows the queue rather than the last click: emptying it is the one
    # change a user can see happen, and "2 file(s)" left on screen over an empty queue is
    # the summary becoming the thing the page has always been accused of being.
    page.clear_files_button.click()
    assert page.pending_files() == []
    assert "no input" in page.summary_label.text(), page.summary_label.text()


def test_the_controls_are_read_only_at_build_and_live_afterwards(qapp, tmp_path,
                                                                 monkeypatch):
    """Two-sided guards, per this suite's rule for prohibitions: the checkbox is
    initialised from the saved value *without* writing (Task 3's `test_building_the_page_
    writes_nothing_to_disk`), and afterwards the same widget does write. A page that
    blocked signals forever would pass the first and fail this one."""
    ctrl = _ui_ctrl(qapp, tmp_path, monkeypatch,
                    ui_obj=ui_prefs.UiPrefs(export_path=""))
    page = ctrl.work_page
    assert page.gpu_check.isChecked() is settings_module.Settings().gpu_conversion
    page.gpu_check.click()
    assert ctrl.settings.gpu_conversion is not settings_module.Settings().gpu_conversion
    assert json.loads(settings_module.config_file().read_text())["gpu_conversion"] == \
        ctrl.settings.gpu_conversion


def test_a_settings_save_republishes_the_two_choices_the_work_page_shows(qapp, tmp_path,
                                                                      monkeypatch):
    """The work page now *displays* two `Settings` fields, so a save that came through the
    dialog has to reach the widgets that display them. `save_settings` adopted the new
    object and refreshed the parameter summary but left the radios and the GPU box showing
    the previous run's values -- and those two widgets are what the truth strip reads, so
    the page spent the rest of the session asserting a format the next run would not use.
    The republish blocks signals: showing a stored value is not a user action, and a
    `toggled` here would ask the root to write back what it just read."""
    ctrl = _ui_ctrl(qapp, tmp_path, monkeypatch)
    page = ctrl.work_page
    ctrl._on_settings_requested()
    dlg = ctrl.settings_dialog
    dlg.page.set_field_value("save_format", "FLAC")
    dlg.page.set_field_value("gpu_conversion", True)
    before = settings_module.config_file().read_bytes()
    ctrl.save_settings(dlg.page, dlg)
    assert ctrl.settings.save_format == "FLAC"
    assert page._checked_format() == "FLAC"
    assert page.gpu_check.isChecked() is True
    assert "FLAC output" in page.summary_label.text(), page.summary_label.text()
    assert settings_module.config_file().read_bytes() != before, "the save reached the disk"
    assert "settings saved: save_format" not in page.console.toPlainText(), \
        "a republish is not a second write"


def test_a_settings_save_mid_run_says_which_run_it_applies_to(qapp, tmp_path, monkeypatch):
    """The dialog is non-modal, so START can be pressed while it is open and Save can land
    while file 2 is running. The batch keeps the plan it assembled -- that is the whole
    reason it assembles once -- so the console has to say the saved values are the *next*
    run's. Without the line, "settings saved to ..." reads as a change to the run in flight,
    which is the same deferred-write lie the work page's own controls are frozen against."""
    from tests.test_work_page_batch import _stub_assemble, _wav_files

    ctrl = _ui_ctrl(qapp, tmp_path, monkeypatch)
    page = ctrl.work_page
    _stub_assemble(monkeypatch, page)
    files = _wav_files(tmp_path, ["a.wav", "b.wav"])
    page.input_edit.setText(str(files[0]))
    page.add_extra_files([str(files[1])])
    monkeypatch.setattr(page.job, "start", lambda request: None)   # still in flight
    ctrl._on_settings_requested()
    dlg = ctrl.settings_dialog
    dlg.page.set_field_value("save_format", "FLAC")
    page.run_requested.emit()
    assert page._busy is True
    ctrl.save_settings(dlg.page, dlg)
    text = page.console.toPlainText()
    assert "the run in flight keeps the plan it started with" in text, text
    assert page._checked_format() == "FLAC", "the page shows what the next run will use"


def test_the_page_still_refuses_an_illegal_format_at_the_engine_gate(qapp, tmp_path,
                                                                    monkeypatch):
    """The radios close the *widget* hole; `session.check_save_format` still closes the
    file hole, and a hand-edited `"save_format": "ogg"` in `settings.json` must not become
    a page with no radio checked and a run that deletes its own output. So the page has to
    say so out loud rather than render a blank group."""
    target = settings_module.config_file()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"save_format": "OGG"}), encoding="utf-8")
    monkeypatch.setattr(backends, "resolve", lambda force=None: _cpu_backend())
    import separate_cli
    cat = separate_cli.make_catalog(tmp_path / "nope", tmp_path / "w")
    ctrl = controller.Controller(cat=cat, st=settings_module.load(target),
                                 selections=None, ui_obj=ui_prefs.UiPrefs())
    page = ctrl.work_page
    assert not [r for r in page.format_radios.values() if r.isChecked()], \
        "nothing is claimed to be selected when the file says something illegal"
    assert page._checked_format() == "?"
    assert "? output" in page.truth_summary()


def test_a_refused_write_from_an_illegal_stored_format_reverts_to_nothing(qapp, tmp_path,
                                                                       monkeypatch):
    """The two holes above meet in one gesture, and the revert path is where they collided:
    the file says `"OGG"`, the user clicks WAV, the write is refused, and the controller
    republishes the value it could not change -- `"OGG"`, which has no radio. Indexing the
    group by it raised `KeyError` out of a Qt slot, which PySide6 prints to stderr while
    the event loop carries on: the radio kept showing the rejected WAV and the summary kept
    saying WAV. Reverting to *nothing checked* is the only honest answer, and it is the same
    state the illegal value loaded into."""
    target = settings_module.config_file()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"save_format": "OGG"}), encoding="utf-8")
    monkeypatch.setattr(backends, "resolve", lambda force=None: _cpu_backend())
    import separate_cli
    cat = separate_cli.make_catalog(tmp_path / "nope", tmp_path / "w")
    ctrl = controller.Controller(cat=cat, st=settings_module.load(target),
                                 selections=None, ui_obj=ui_prefs.UiPrefs())
    page = ctrl.work_page
    monkeypatch.setattr(settings_module.Settings, "save",
                        lambda self, path=None: (_ for _ in ()).throw(
                            PermissionError(13, "denied")))
    page.format_radios["WAV"].setChecked(True)
    assert ctrl.settings.save_format == "OGG", "a refused write changes nothing"
    assert not [r for r in page.format_radios.values() if r.isChecked()], \
        "the rejected WAV is not left on screen over a file that still says OGG"
    assert page._checked_format() == "?"
    assert "? output" in page.summary_label.text(), page.summary_label.text()
    assert "not saved" in page.console.toPlainText()
