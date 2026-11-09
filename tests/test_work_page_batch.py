"""Plan 3 Task 2: the batch queue, tested where the queue lives -- the composition root.

Batch mode is deliberately *not* in `app/job.py`. `Job` is one file (`:141-158` refuses an
overlap and explains why: the export directory, the VRAM, and `run_job`'s before/after
snapshot of the former), `session.run_job` is one file, and the engine is one file. What
runs many files is the thing that owns all three, so the queue is `Controller`'s business;
the only change to `job.py` is `BUSY_REFUSAL`, the sentence its refusal emits, held there so
the root can pre-empt that refusal without a second copy of the words.

The stubs are the suite's own: `_ui_ctrl`, `_Rp`, `SlowEngine` and `_spy_notify` come from
`tests/test_work_page.py` (the cross-module import is `tests/test_ui_surfaces.py:28`'s
established pattern), so this file adds no second version of a CPU backend or a fake
plan. Only `Job.start` is replaced -- its replacement records the file it was handed and
fires that file's terminal, so the batch advances because a *signal* arrived, which is the
whole requirement, and never because the test called `_advance()`.
"""
from __future__ import annotations

from pathlib import Path

from app import dialogs, i18n, job, session, work_page
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
    ctrl = _ui_ctrl(qapp, tmp_path, monkeypatch)
    page = ctrl.work_page
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


def test_audio_files_in_is_recursive_case_insensitive_and_sorted():
    """What the scan does with the list `app/dialogs.py::AUDIO_EXTENSIONS` holds: walk
    subfolders, match the suffix case-insensitively, skip what is not audio, and return path
    order so the queue a user sees is stable. The list's *identity* is pinned elsewhere and
    in both directions -- `tests/test_dialogs.py::
    test_the_app_audio_extensions_are_the_engines_own_list` compares the held copy against
    `gui_data.constants.FFMPEG_EXT`, which carries `"MP3"` with no dot at :832 and both cases
    of `.wav`/`.WAV`. This test is the behaviour half: a scan that lower-cased the pattern
    but not the filename would pass that pin and still lose `song.MP3`, which in batch mode
    reads as "it lost half my folder", with no error."""
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
    ctrl = _ui_ctrl(qapp, tmp_path, monkeypatch)
    page = ctrl.work_page
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
    ctrl = _ui_ctrl(qapp, tmp_path, monkeypatch)
    page = ctrl.work_page
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
    assert shown == [(i18n.tr(work_page.FAILED_TITLE),
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
    assert shown == [(i18n.tr(work_page.DONE_TITLE), "1 file(s) written")], shown
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


def test_a_stopped_batch_gets_a_surface_and_not_only_a_console_line(qapp, tmp_path,
                                                                   monkeypatch):
    """The other two batch endings speak through `_notify`; a STOP did not.

    `_notify_for_file` returns early while `_batch_total > 1` (one bar per file over forty
    files is the pile R-T7-infobar exists to prevent), and `end_batch(stopped=True)`
    returned before either tally branch -- so the one ending a user *caused* was the only
    one with no surface at all. The console keeps its line; the band gets the batch's own
    words, and exactly one bar, because the per-file terminals stayed silent."""
    files = _wav_files(tmp_path, ["a.wav", "b.wav", "c.wav"])
    ctrl = _ui_ctrl(qapp, tmp_path, monkeypatch)
    page = ctrl.work_page
    _stub_assemble(monkeypatch, page)
    started: list[Path] = []
    _stub_job_start(page, monkeypatch, started,
                    terminal=lambda: page.job.cancelled.emit("stopped by the user"))
    shown = _spy_notify(page, monkeypatch)
    _queued_files(page, files)
    page.run_requested.emit()

    assert [p.name for p in started] == ["a.wav"], started
    assert shown == [(i18n.tr(work_page.CANCELLED_TITLE),
                      "batch stopped after 1 file(s); the rest were not run")], shown
    assert ("batch stopped after 1 file(s); the rest were not run"
            in page.console.toPlainText())


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
    ctrl = _ui_ctrl(qapp, tmp_path, monkeypatch)
    page = ctrl.work_page
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


def test_a_refused_start_does_not_drain_the_queue_as_fake_failures(qapp, tmp_path,
                                                                  monkeypatch):
    """`Job.start` reports a refusal on `failed` -- the same signal a file's engine error
    arrives on (`job.py:156-158`). Consuming that as a file terminal advances the queue,
    which starts the next file, which refuses again: a three-file batch ends as "3 file(s)
    attempted, 0 written, 3 of 3 failed" with a named failure per file, for runs that never
    existed. Measured before the fix, and the lie is the point -- the user is told which
    files are broken when nothing ran at all.

    Driven with `is_running()` held True, so the condition is the production one. The guard
    pre-empts it: no request is handed to a `Job` that would answer on `failed`, so nothing
    is attempted, one reason is printed, busy is released, and the queue is intact for the
    next press. The negative control at the end calls the *real* `start` to prove the
    sentence the root prints is the one the `Job` itself emits."""
    files = _wav_files(tmp_path, ["a.wav", "b.wav", "c.wav"])
    ctrl = _ui_ctrl(qapp, tmp_path, monkeypatch)
    page = ctrl.work_page
    _stub_assemble(monkeypatch, page)
    _queued_files(page, files)
    monkeypatch.setattr(page.job, "is_running", lambda: True)
    started: list[Path] = []
    real_start = page.job.start

    def start(request):
        started.append(Path(request.audio))
        real_start(request)
    monkeypatch.setattr(page.job, "start", start)

    page.run_requested.emit()

    assert started == [], "the guard pre-empts the refusal instead of consuming it"
    text = page.console.toPlainText()
    assert f"not started: {job.BUSY_REFUSAL}" in text, text
    assert "b.wav" not in text and "c.wav" not in text, text
    assert "batch finished" not in text, text
    assert ctrl._batch is None
    assert page.start_button.isEnabled(), "the refusal releases busy instead of stranding it"
    assert page.stop_button.isEnabled() is False
    assert len(page.pending_files()) == 3, "a refusal must not eat the queue"

    # Negative control, last so its terminal cannot disturb the assertions above: the real
    # `start` answers a live job with exactly this sentence on `failed`, and the root ignores
    # it now that no batch is live (`_batch is None`) rather than tallying a file.
    heard: list[str] = []
    page.job.failed.connect(heard.append)
    real_start(job.JobRequest(session=None, audio=files[0], export_path=tmp_path))
    assert heard == [job.BUSY_REFUSAL], heard
    assert ctrl._batch is None


def test_the_choices_a_run_froze_cannot_be_changed_mid_run(qapp, tmp_path, monkeypatch):
    """The plan is assembled once for the whole batch (`_on_run`), and `export_for` is asked
    per file. So a change made while file 2 is running has two possible meanings and both
    used to be available: the Output box would move the *rest* of one batch into a second
    folder -- which is also the directory `run_job` diffs to decide what it wrote -- and a
    format or GPU click would print "settings saved" while every remaining file still ran
    under the plan assembled before it. A console line claiming a change the run does not
    honour is the same lie as a radio showing a value that was never stored.

    The fix is that those controls are read-only for the duration, so there is no deferred
    write to explain. The queue is deliberately *not* frozen: lining up the next run while
    this one finishes is legitimate, and `batch.files` is already a copy."""
    files = _wav_files(tmp_path, ["a.wav", "b.wav"])
    ctrl = _ui_ctrl(qapp, tmp_path, monkeypatch)
    page = ctrl.work_page
    _stub_assemble(monkeypatch, page)
    _queued_files(page, files)
    monkeypatch.setattr(page.job, "start", lambda request: None)   # no terminal: still running
    frozen = [page.output_edit, page.browse_output, page.default_output_check,
              page.gpu_check, page.model_combo, page.arch_bar,
              *page.format_radios.values()]

    page.run_requested.emit()
    assert all(not w.isEnabled() for w in frozen), \
        [type(w).__name__ for w in frozen if w.isEnabled()]
    assert page.stop_button.isEnabled(), "STOP is the one thing a run in flight must offer"
    assert page.input_edit.isEnabled() and page.add_files_button.isEnabled(), \
        "the next run's queue is not this run's plan"

    page.job.succeeded.emit([])                     # file 1 ends, file 2 starts
    assert ctrl._batch is not None and ctrl._batch.index == 1
    assert all(not w.isEnabled() for w in frozen), "still mid-batch, still frozen"

    page.job.succeeded.emit([])                     # the batch ends
    assert ctrl._batch is None
    assert all(w.isEnabled() for w in frozen), \
        [type(w).__name__ for w in frozen if not w.isEnabled()]
    assert page.start_button.isEnabled()


def test_the_buttons_are_wired_to_the_pickers_not_to_nothing(qapp, tmp_path, monkeypatch):
    """The click path, without a modal dialog: the suite never opens `QFileDialog`, so
    what is pinned here is that each button reaches the picker the queue then reads --
    which is the half that rots silently when a layout is rebuilt and a `connect` is
    dropped."""
    import fluentqt

    ctrl = _ui_ctrl(qapp, tmp_path, monkeypatch)
    page = ctrl.work_page
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
