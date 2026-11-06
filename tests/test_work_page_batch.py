"""Plan 3 Task 2: the batch queue, tested where the queue lives -- the composition root.

Batch mode is deliberately *not* in `app/job.py`. `Job` is one file (`:133-150` refuses an
overlap and explains why: the export directory, the VRAM, and `run_job`'s before/after
snapshot of the former), `session.run_job` is one file, and the engine is one file. What
runs many files is the thing that owns all three, so the queue is `Controller`'s business
and the only change to `job.py` is nothing.

The stubs are the suite's own: `_ui_ctrl`, `_Rp`, `SlowEngine` and `_spy_notify` come from
`tests/test_work_page.py` (the cross-module import is `tests/test_ui_surfaces.py:28`'s
established pattern), so this file adds no second version of a CPU backend or a fake
plan. Only `Job.start` is replaced -- its replacement records the file it was handed and
fires that file's terminal, so the batch advances because a *signal* arrived, which is the
whole requirement, and never because the test called `_advance()`.
"""
from __future__ import annotations

from pathlib import Path

from app import dialogs, session, work_page
from tests.test_work_page import SlowEngine, _Rp, _spy_notify, _ui_ctrl


def _stub_assemble(monkeypatch, page, model="UVR-MDX-NET-Inst_HQ_3"):
    """A picked model and a built plan, neither of which needs a real catalog.

    `session.assemble` is the one step of `_on_run` that cannot answer anything on a
    `make_catalog(nope, w)` catalog -- an empty one raises `UnknownModelParamsError` and
    hands the test a modal dialog. The `Session` handed back is the real dataclass with
    the `_Rp` this suite already uses, so the progress arithmetic stays production code on
    this path too.
    """
    monkeypatch.setattr(page, "selected_model", lambda: model)
    monkeypatch.setattr(session, "assemble",
                        lambda **kw: session.Session(spec=None, backend=None, rp=_Rp(),
                                                     engine_class=SlowEngine,
                                                     compute_lines=()))


def _wav_files(tmp_path, names):
    out = []
    for n in names:
        p = tmp_path / n
        p.write_bytes(b"x")
        out.append(p)
    return out


def _stub_job_start(page, monkeypatch, started: list, failing=(), terminal=None):
    """`Job.start` -> record the file, write a stem, fire that file's terminal.

    Synchronous, and that is a property of Qt rather than of this stub: with no `QThread`
    running, a signal emitted on the main thread is delivered directly, so the whole queue
    drains inside the one `run_requested.emit()`.
    """
    def start(request):
        audio = Path(request.audio)
        started.append(audio)
        export = Path(request.export_path)
        export.mkdir(parents=True, exist_ok=True)
        # The engine reports 100 before it returns, so a file that ended at 0% would be a
        # stub bug rather than a page bug -- and this is the one number the old suite
        # already pinned for a single-file run.
        page.job.progress.emit(100)
        if audio in failing:
            page.job.failed.emit("RuntimeError: stub engine failure")
            return
        stem = export / f"{audio.stem}_(Vocals).wav"
        stem.write_bytes(b"x")
        (terminal or (lambda: page.job.succeeded.emit([stem])))()
    monkeypatch.setattr(page.job, "start", start)


def _queued_files(page, files):
    page.input_edit.setText(str(files[0]))
    page.add_extra_files([str(f) for f in files[1:]])
    return files


def test_pending_files_is_typed_file_first_then_extras_deduped(qapp, tmp_path, monkeypatch):
    """The queue has one source. `input_edit` stays file #1 because every existing test in
    this suite sets it the way a user types it, and a second list that also holds it is
    how a batch comes to run the same file twice."""
    page = _ui_ctrl(qapp, tmp_path, monkeypatch).work_page
    a, b = _wav_files(tmp_path, ["a.wav", "b.wav"])
    page.input_edit.setText(str(a))
    page.add_extra_files([str(b), str(a)])
    assert page.pending_files() == [a, b]
    # Clearing the typed box leaves *both* extras: the user queued them deliberately
    # through the picker, so dropping one because it happened to also be typed would lose
    # a file the moment they edited the box. The dedup is about not running a file twice,
    # not about forgetting it.
    page.input_edit.setText("")
    assert page.pending_files() == [b, a]


def test_audio_files_in_uses_the_engines_own_extension_list():
    """Recursive, and matched against `gui_data.constants.FFMPEG_EXT` rather than a
    home-grown tuple: that constant carries `"MP3"` with no dot at :832 and both cases of
    `.wav`/`.WAV`, so a tidy five-suffix list would quietly skip files the engine can
    read -- which in batch mode reads as "it lost half my folder", with no error."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "song.MP3").write_bytes(b"x")
        (root / "deep").mkdir()
        (root / "deep" / "take.flac").write_bytes(b"x")
        (root / "notes.txt").write_bytes(b"x")
        got = work_page.audio_files_in(root)
        assert {p.name for p in got} == {"song.MP3", "take.flac"}
        assert got == sorted(got), "path order, so the queue a user sees is stable"


def test_blockers_are_the_words_the_refusal_uses(qapp, tmp_path, monkeypatch):
    """One table, two readers: the label and the refusal. Before this, `_on_run` owned the
    checks and the page could only guess at them, which is how an explanation and a
    refusal drifted apart -- and the everyday form of that drift is the button that looks
    like it did nothing. The strings are the pre-existing refusal's verbatim, so nothing
    a user read last week changed wording. The page builds on an empty catalog, where the
    combo holds only `NO_MODEL`, so the first blocker is the model and it stays first."""
    page = _ui_ctrl(qapp, tmp_path, monkeypatch).work_page
    assert page.blockers() == ["pick a model from the Model list first",
                               "choose an input audio file first"]
    assert not page.start_button.isEnabled()
    assert page.reason_label.text() == ("pick a model from the Model list first; "
                                        "choose an input audio file first")

    _stub_assemble(monkeypatch, page)
    missing = tmp_path / "gone.wav"                 # named, never created
    page.input_edit.setText(str(missing))
    assert page.blockers() == [f"input audio not found: {missing}"]
    assert not page.start_button.isEnabled()


def test_a_batch_of_missing_files_names_three_and_counts_the_rest(qapp, tmp_path,
                                                                  monkeypatch):
    """400 stale paths must not print 400 paths into a label."""
    page = _ui_ctrl(qapp, tmp_path, monkeypatch).work_page
    _stub_assemble(monkeypatch, page)
    page.input_edit.setText("")
    page.add_extra_files([str(tmp_path / f"gone{n}.wav") for n in range(12)])
    blockers = page.blockers()
    assert len(blockers) == 1 and blockers[0].startswith("input audio not found:")
    assert "(+9 more)" in blockers[0], blockers
    assert "gone11.wav" not in blockers[0]


def test_batch_runs_sequentially_continues_on_error_and_tallies(qapp, tmp_path, monkeypatch):
    """Three files, the second fails, all three are attempted -- and the failing one is
    named, because "2 of 3 written" without a *which* is a mystery the user has to re-run
    to solve."""
    files = _wav_files(tmp_path, ["a.wav", "b.wav", "c.wav"])
    ctrl = _ui_ctrl(qapp, tmp_path, monkeypatch)
    page = ctrl.work_page
    _stub_assemble(monkeypatch, page)
    started: list[Path] = []
    _stub_job_start(page, monkeypatch, started, failing={files[1]})
    shown = _spy_notify(page, monkeypatch)
    _queued_files(page, files)
    page.run_requested.emit()

    assert [p.name for p in started] == ["a.wav", "b.wav", "c.wav"]
    text = page.console.toPlainText()
    assert ("batch finished: 3 file(s) attempted, 2 file(s) written, "
            "1 of 3 file(s) failed") in text
    assert "b.wav: RuntimeError: stub engine failure" in text
    # After the tally the label goes back to what the *next* press would run: the queue is
    # unchanged by a run, and "3 / 3" lingering over an idle page reads as a run in
    # progress. `test_progress_spreads_...` pins the live form of the same label.
    assert page.count_label.text() == "3 file(s) queued", page.count_label.text()
    assert page.start_button.isEnabled(), "the batch releases busy, not file 2"
    assert page.stop_button.isEnabled() is False
    assert shown == [(work_page.FAILED_TITLE,
                      "1 of 3 file(s) failed -- see the console lines above")], shown


def test_a_single_file_batch_reports_exactly_as_it_did_before(qapp, tmp_path, monkeypatch):
    """The regression this guards is a silent one: batch code paths that also run for
    `len(files) == 1` would turn every old single-file run into "batch finished: 1 file(s)
    attempted". One file is what 5.6 does and what Plan 2 shipped, so its wording stays."""
    files = _wav_files(tmp_path, ["only.wav"])
    ctrl = _ui_ctrl(qapp, tmp_path, monkeypatch)
    page = ctrl.work_page
    _stub_assemble(monkeypatch, page)
    started: list[Path] = []
    _stub_job_start(page, monkeypatch, started)
    shown = _spy_notify(page, monkeypatch)
    _queued_files(page, files)
    page.run_requested.emit()

    assert [p.name for p in started] == ["only.wav"]
    # Matched on the sentences, not the word: this test's own tmp directory is named after
    # it, so a bare `"batch" not in text` would fail on a path and prove nothing.
    text = page.console.toPlainText()
    assert "batch started" not in text and "batch finished" not in text, text
    assert shown == [(work_page.DONE_TITLE, "1 file(s) written")], shown
    assert page.progress.value() == 100


def test_stop_after_the_first_file_runs_no_more(qapp, tmp_path, monkeypatch):
    """STOP is cooperative at the engine's own granularity, and in a batch it also means
    "do not start the rest" -- the part of a batch a user can actually act on mid-run.
    What is written stays written, and the console says how far it got."""
    files = _wav_files(tmp_path, ["a.wav", "b.wav"])
    ctrl = _ui_ctrl(qapp, tmp_path, monkeypatch)
    page = ctrl.work_page
    _stub_assemble(monkeypatch, page)
    started: list[Path] = []
    _stub_job_start(page, monkeypatch, started,
                    terminal=lambda: page.job.cancelled.emit("stopped by the user"))
    _queued_files(page, files)
    page.run_requested.emit()

    assert [p.name for p in started] == ["a.wav"], started
    assert ("batch stopped after 1 file(s); the rest were not run"
            in page.console.toPlainText())
    assert page.start_button.isEnabled()


def test_a_stop_pressed_before_any_terminal_stops_the_queue(qapp, tmp_path, monkeypatch):
    """The flag is set by the *button*, not read off the terminal. A cancellation lands
    when the engine's next progress hook fires (`separate.py:1383`), so a STOP during a
    model load has no terminal to come from yet -- and a queue that only learned about
    stopping from terminals would start file 2 anyway."""
    files = _wav_files(tmp_path, ["a.wav", "b.wav"])
    ctrl = _ui_ctrl(qapp, tmp_path, monkeypatch)
    page = ctrl.work_page
    _stub_assemble(monkeypatch, page)
    started: list[Path] = []

    def start(request):                       # file 1 is still loading: no terminal yet
        started.append(Path(request.audio))
    monkeypatch.setattr(page.job, "start", start)
    asked: list[bool] = []
    monkeypatch.setattr(page.job, "request_cancel", lambda: asked.append(True))

    _queued_files(page, files)
    page.run_requested.emit()
    assert [p.name for p in started] == ["a.wav"]

    page.cancel_requested.emit()
    assert asked == [True], "STOP still reaches the one Job it always reached"
    page.job.cancelled.emit("stopped by the user")    # the engine finally answers
    assert [p.name for p in started] == ["a.wav"], "file 2 must not start"
    assert "stopped after 1 file(s)" in page.console.toPlainText()


def test_progress_spreads_one_files_percent_over_the_batch(qapp, tmp_path, monkeypatch):
    """The page's docstring used to refuse batch because `run_job` "would reset per file
    and read as a saw-tooth". It does reset per file -- so the published number is not the
    file's, it is `(files done + this file's fraction) / total`. `session.progress_percent`
    keeps its own `total_files` term untouched; this is the GUI aggregating the signal."""
    page = _ui_ctrl(qapp, tmp_path, monkeypatch).work_page
    page.begin_batch(4)
    assert page.count_label.text() == "0 / 4 file(s)", page.count_label.text()
    page._on_progress(50)
    assert page.progress.value() == 12, page.progress.value()      # 0 done, half of 1/4
    page.end_file()
    assert page.count_label.text() == "1 / 4 file(s)", "the label walks with the queue"
    page._on_progress(100)
    assert page.progress.value() == 50
    page.end_file()
    page.end_file()
    page.end_file()
    page._on_progress(100)
    assert page.progress.value() == 100
    page.end_batch(written=4, failures=[], stopped=False)
    assert page.count_label.text().endswith("file(s) queued")


def test_the_queue_is_reachable_from_the_page_and_survives_a_refusal(qapp, tmp_path,
                                                                    monkeypatch):
    """Two halves of one promise. The buttons fill the queue (no modal dialog is clicked --
    the suite's rule about `QFileDialog`), and a START that *refuses* leaves the queue
    exactly as it was: a refusal that ate the user's 40-file list would be worse than the
    missing batch mode it arrived with. Driven through the root, because the root is who
    listens to `run_requested` -- a bare page emits into nothing."""
    ctrl = _ui_ctrl(qapp, tmp_path, monkeypatch)
    page = ctrl.work_page
    files = _wav_files(tmp_path, ["a.wav", "b.wav", "c.wav"])
    page.add_extra_files([str(f) for f in files])          # what "Add files…" would do
    assert len(page.pending_files()) == 3
    assert page.count_label.text() == "3 file(s) queued", page.count_label.text()

    page.run_requested.emit()                              # no model picked -> refused
    assert ("not started: pick a model from the Model list first"
            in page.console.toPlainText())
    assert len(page.pending_files()) == 3, "a refusal must not eat the queue"
    assert page.start_button.isEnabled() is False
    assert ctrl._batch is None, "a refusal starts nothing"


def test_the_buttons_are_wired_to_the_pickers_not_to_nothing(qapp, tmp_path, monkeypatch):
    """The click path, without a modal dialog: the suite never opens `QFileDialog`, so
    what is pinned here is that each button reaches the picker the queue then reads --
    which is the half that rots silently when a layout is rebuilt and a `connect` is
    dropped."""
    import fluentqt

    page = _ui_ctrl(qapp, tmp_path, monkeypatch).work_page
    asked: list[str] = []
    files = _wav_files(tmp_path, ["a.wav", "b.wav", "c.wav"])
    page.input_edit.setText(str(files[0]))
    monkeypatch.setattr(dialogs.Pickers, "input_files",
                        lambda parent=None, start="": (asked.append("files") or
                                                       [str(files[1])]))
    monkeypatch.setattr(dialogs.Pickers, "input_folder",
                        lambda parent=None, start="": (asked.append("folder") or
                                                       str(tmp_path)))
    page.add_files_button.click()
    assert [p.name for p in page.pending_files()] == ["a.wav", "b.wav"], asked
    page.add_folder_button.click()
    assert [p.name for p in page.pending_files()] == ["a.wav", "b.wav", "c.wav"], \
        "the scan overlaps the two already queued, and dedup keeps it at three"
    assert asked == ["files", "folder"], asked
    page.clear_files_button.click()
    assert [p.name for p in page.pending_files()] == ["a.wav"], "the typed file stays"
