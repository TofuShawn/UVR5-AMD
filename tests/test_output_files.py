"""The run-output list: what this session wrote, and the three things you do with it.

Why it exists at all: the engine writes the stems and then the GUI forgets them. "Where
did it go?" is answered today by the Output box (a folder) and the console (engine text),
which is fine for one file and useless for a forty-file batch. The user asked for the
list directly ("輸出檔列表 + 打開/所在資料夾/刪除"), so the tests here are about the three
verbs and the one irreversible one.

The delete path is the part that earns the file:

* `confirm_delete` is a function so this suite can replace it. `exec()` enters a nested
  event loop; a test that let a ContentDialog run would sit in it.
* a refused unlink must keep both the file *and* its row. A row removed while the file
  is still on disk is the list lying about the disk, which is the failure class this
  page keeps being rewritten against.
* deleting is disabled for the duration of a run, because `session.run_job` decides what
  it wrote by diffing the export folder before and after: a stem deleted mid-run would
  be attributed to the file being separated.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import backends
import selections as selections_module
import separate_cli
import settings as settings_module
import ui_prefs
from app import controller, dialogs


def _cpu_backend() -> backends.Backend:
    return backends.Backend(name="cpu", torch_device="cpu",
                            ort_providers=["DmlExecutionProvider",
                                           "CPUExecutionProvider"])


@pytest.fixture
def ctrl(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(backends, "resolve", lambda force=None: _cpu_backend())
    cat = separate_cli.make_catalog(tmp_path / "nope", tmp_path / "writable")
    return controller.Controller(cat=cat, st=settings_module.Settings(),
                                 selections=selections_module.Selections(),
                                 ui_obj=ui_prefs.UiPrefs())


def _stem(tmp_path, name="song (Vocals)_1.wav") -> Path:
    path = tmp_path / name
    path.write_bytes(b"stem")
    return path


def _select(page, path: Path) -> None:
    row = page.written_paths().index(str(path))
    page.output_list.setCurrentIndex(page._output_model.index(row, 0))


def test_the_list_starts_empty_and_its_buttons_need_a_selection(ctrl, tmp_path):
    page = ctrl.work_page
    assert page.written_paths() == []
    assert page.output_list.placeholderText() == "stems written by this run appear here"
    assert page.open_output_button.isEnabled() is False
    assert page.reveal_output_button.isEnabled() is False
    assert page.delete_output_button.isEnabled() is False

    stem = _stem(tmp_path)
    page.add_written([stem])
    assert page.written_paths() == [str(stem)]
    # The row is the file name and the tooltip the full path: at 1280 the list is
    # ~200 px wide, and a row of "Z:\\...\\song (Vocals)_1.wav" clipped to the shared
    # prefix, leaving three identical-looking rows (measured).
    item = page._output_model.item(0)
    assert item.text() == stem.name
    assert item.toolTip() == str(stem)
    assert page.open_output_button.isEnabled() is False, "a row exists, but none is selected"

    _select(page, stem)
    assert page.selected_output() == str(stem)
    for button in (page.open_output_button, page.reveal_output_button,
                   page.delete_output_button):
        assert button.isEnabled() is True


def test_open_hands_the_file_to_the_desktop_and_reveal_hands_it_the_folder(
        ctrl, tmp_path, monkeypatch):
    page = ctrl.work_page
    stem = _stem(tmp_path)
    page.add_written([stem])
    opened = []

    def fake_open(url):
        opened.append(url.toLocalFile())
        return True

    monkeypatch.setattr(controller.QDesktopServices, "openUrl", fake_open)

    ctrl._on_open_output(str(stem))
    ctrl._on_reveal_output(str(stem))
    # `QUrl.fromLocalFile` normalises Windows separators to '/', so the claim is about
    # the path, not the spelling.
    assert [Path(p) for p in opened] == [stem, stem.parent]


def test_a_desktop_that_refuses_is_reported_in_the_console(ctrl, tmp_path, monkeypatch):
    """`openUrl` answering False is the "button does nothing" case: a file type with no
    association, or a folder the shell will not open. Silence here is the bug."""
    page = ctrl.work_page
    stem = _stem(tmp_path)
    page.add_written([stem])
    monkeypatch.setattr(controller.QDesktopServices, "openUrl", lambda url: False)

    ctrl._on_open_output(str(stem))
    assert f"could not open {stem}" in page.console.toPlainText()


def test_delete_asks_first_and_a_refusal_changes_nothing(ctrl, tmp_path, monkeypatch):
    page = ctrl.work_page
    stem = _stem(tmp_path)
    page.add_written([stem])
    asked = []

    def refuse(target, *, parent=None):
        asked.append(str(target))
        return False

    monkeypatch.setattr(dialogs, "confirm_delete", refuse)
    ctrl._on_delete_output(str(stem))

    assert asked == [str(stem)], "the confirmation must name the file it will remove"
    assert stem.is_file()
    assert page.written_paths() == [str(stem)]


def test_a_confirmed_delete_removes_the_file_and_its_row(ctrl, tmp_path, monkeypatch):
    page = ctrl.work_page
    stem = _stem(tmp_path)
    keep = _stem(tmp_path, "other.wav")
    page.add_written([stem, keep])
    monkeypatch.setattr(dialogs, "confirm_delete", lambda target, *, parent=None: True)

    ctrl._on_delete_output(str(stem))

    assert not stem.exists()
    assert keep.is_file()
    assert page.written_paths() == [str(keep)], page.written_paths()
    assert f"deleted {stem.name}" in page.console.toPlainText()


def test_a_refused_unlink_keeps_both_the_file_and_the_row(ctrl, tmp_path, monkeypatch):
    """A file held open by a player, or a read-only folder: the row stays, because a row
    that vanished over a file still on disk is the list lying about the disk."""
    page = ctrl.work_page
    folder = tmp_path / "locked.wav"
    folder.mkdir()                      # directories cannot be unlinked on Windows
    page.add_written([folder])
    monkeypatch.setattr(dialogs, "confirm_delete", lambda target, *, parent=None: True)

    ctrl._on_delete_output(str(folder))

    assert folder.is_dir()
    assert page.written_paths() == [str(folder)]
    text = page.console.toPlainText()
    assert f"could not delete {folder.name}" in text, text


def test_the_verbs_are_disabled_while_a_run_is_in_flight(ctrl, tmp_path):
    page = ctrl.work_page
    stem = _stem(tmp_path)
    page.add_written([stem])
    _select(page, stem)
    assert page.open_output_button.isEnabled() is True

    page.set_busy(True)
    assert page.open_output_button.isEnabled() is False
    assert page.delete_output_button.isEnabled() is False

    page.set_busy(False)
    assert page.delete_output_button.isEnabled() is True


def test_a_finished_file_reaches_the_list_through_the_job_signal(ctrl, tmp_path,
                                                               monkeypatch):
    """The one wiring that makes the list about *this run* rather than whatever a test
    calls `add_written` with: `Job.succeeded` carries the paths, `_on_file_done` appends
    them, and the queue walk is stubbed so this test is about the wiring only."""
    page = ctrl.work_page
    stem = _stem(tmp_path)
    ctrl._batch = controller._Batch(sess=None, files=[stem])
    monkeypatch.setattr(ctrl, "_advance", lambda: None)

    page.job.succeeded.emit([stem])
    assert page.written_paths() == [str(stem)]
