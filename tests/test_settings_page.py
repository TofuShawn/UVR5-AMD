"""The settings page: 50 engine fields, 33 selections, and the one network command.

Every test here writes to its **own** `tmp_path`, never to the shared `%APPDATA%` file.
`tests/test_settings.py:959-961` asserts a saved settings file holds exactly 50
top-level keys, and that is true only while *its* fixture is fresh: a page test that
seeded a `selections` block into the same directory would fail for a reason that looks
like a bug in `apply()`. `apply(path)` takes the destination, so there is no reason to
share. The one test that must go through the controller's save path (which writes
`settings.config_file()`) repoints `APPDATA` at its own temporary base for the same
reason.

Nothing here clicks the refresh button in a way that could reach the network: the
un-confirmed path is asserted to fetch nothing, and the confirmed path runs against a
temporary catalog with a faked fetch.
"""
from __future__ import annotations

import dataclasses
import json

import pytest
import catalog
import fluentqt
import model_refresh
import selections as selections_module
import settings as settings_module

from app import settings_page


def test_every_settings_field_is_reachable_from_the_page():
    """50 fields, and the count is `Settings`' own invariant, not this test's invention.
    A field with no widget is a field the GUI cannot save -- silent data loss."""
    fields = {f.name for f in dataclasses.fields(settings_module.Settings)}
    assert len(fields) == 50, len(fields)
    covered = set(settings_page.all_field_names())
    assert fields <= covered, sorted(fields - covered)


def test_arch_groups_are_disjoint_and_cover_exactly_the_50():
    groups = settings_page.ARCH_FIELD_GROUPS
    assert set(groups) == {"vr", "mdx", "demucs", "common"}
    names = [n for group in groups.values() for n in group]
    assert len(names) == len(set(names)), [n for n in names if names.count(n) > 1]
    fields = {f.name for f in dataclasses.fields(settings_module.Settings)}
    assert set(names) == fields, (sorted(fields - set(names)), sorted(set(names) - fields))


def test_apply_round_trips_the_three_field_kinds_it_sets(qapp, tmp_path):
    """Name fix (wave 2 minor): this was `..._every_field` while the body sets three
    widgets -- one int, one str enum, one `Selections` name -- and the *all* fields claim
    is already made honestly one test above, where `set(names) == fields` pins every
    `Settings` field to a widget. Three samples across the three widget kinds is what
    this proves; the old name promised an exhaustive round-trip it never ran."""
    st = settings_module.Settings()
    sel = selections_module.Selections()
    page = settings_page.SettingsPage(settings_obj=st, selections_obj=sel)
    page.set_field_value("window_size", 1024)
    page.set_field_value("save_format", "FLAC")
    page.set_field_value("vr_voc_inst_secondary_model", "UVR-BVE-458n")
    assert page.apply(tmp_path / "settings.json") == []
    assert settings_module.load(tmp_path / "settings.json").window_size == 1024
    assert selections_module.load(tmp_path / "settings.json").vr_voc_inst_secondary_model == "UVR-BVE-458n"


def test_a_value_the_type_system_refuses_is_reported_and_nothing_is_written(qapp, tmp_path):
    page = settings_page.SettingsPage(settings_obj=settings_module.Settings(),
                                      selections_obj=selections_module.Selections())
    page.set_field_value_raw("window_size", "not a number")
    target = tmp_path / "settings.json"
    messages = page.apply(target)
    assert messages and "window_size" in messages[0], messages
    assert not target.exists(), "a page that could not validate must not touch the file"


def test_refresh_without_confirmation_raises_and_fetches_nothing(qapp, tmp_path, monkeypatch):
    """`model_refresh.refresh` already refuses unconfirmed work (R-T10-refresh); this
    test's job is to prove the page routes through it rather than around it."""
    import model_refresh
    fetched = []
    monkeypatch.setattr(model_refresh, "fetch_json", lambda url: fetched.append(url) or {})
    page = settings_page.SettingsPage(settings_obj=settings_module.Settings(),
                                      selections_obj=selections_module.Selections())
    with pytest.raises(model_refresh.RefreshError):
        page.refresh_tables(confirmed=False)
    assert fetched == []


# -- the rest of the page's contract ------------------------------------------ #

def _page(qapp):
    return settings_page.SettingsPage(settings_obj=settings_module.Settings(),
                                      selections_obj=selections_module.Selections())


def test_every_selection_field_is_reachable_from_the_page(qapp):
    """The 33 is `Selections`' own invariant, and the same argument as the 50: a slot
    with no widget is a slot the GUI cannot fill, so it keeps whatever the file said."""
    fields = {f.name for f in dataclasses.fields(selections_module.Selections)}
    assert len(fields) == 33, len(fields)
    assert fields == set(selections_module.SELECTION_FIELD_NAMES)
    covered = set(_page(qapp).widget_names())
    assert fields <= covered, sorted(fields - covered)


def test_the_page_holds_exactly_one_widget_per_field_and_no_other_kind(qapp, tmp_path):
    """83 fields, 83 widgets -- and every widget is registered under a name that *is* a
    field. A stray entry means a widget the user can edit that nothing reads back, or a
    field whose widget was overwritten by a second one."""
    page = _page(qapp)
    names = set(page.widget_names())
    expected = ({f.name for f in dataclasses.fields(settings_module.Settings)}
                | {f.name for f in dataclasses.fields(selections_module.Selections)})
    assert names == expected, (sorted(names - expected), sorted(expected - names))
    assert len(page._widgets) == len(names) == 83
    for name, widget in page._widgets.items():
        # Four kinds, not three: I7 turned the fields the engine itself lists values for
        # into non-editable `ComboBox`es, so a `LineEdit` is now only what is left for a
        # field whose domain the engine derives (model names, `mdxnet_stems` from a
        # checkpoint, `chosen_process_method` from the resolved spec).
        assert isinstance(widget, (fluentqt.CheckBox, fluentqt.NumberBox,
                                   fluentqt.LineEdit, fluentqt.ComboBox)), (
            name, type(widget).__name__)


def test_the_untouched_page_saves_every_field_back_as_it_was_loaded(qapp, tmp_path):
    """The whole point of building the widgets from the type system: a user who changes
    nothing must not lose anything. Every one of the 83 values goes widget -> validator
    -> file -> loader and has to arrive equal to what it was."""
    st, sel = settings_module.Settings(), selections_module.Selections()
    page = settings_page.SettingsPage(settings_obj=st, selections_obj=sel)
    target = tmp_path / "settings.json"
    assert page.apply(target) == []
    assert settings_module.load(target) == st
    assert selections_module.load(target) == sel


def test_apply_reports_the_valueerror_the_scale_gate_raises(qapp, tmp_path):
    """`Selections._coerce` raises `ValueError` for junk text in a float scale (and for
    `None`), while `selections.load()`'s gate raises `TypeError` for the same shape. The
    page constructs, so it sees the first -- and must not let either class escape."""
    page = _page(qapp)
    page.set_field_value_raw("vr_voc_inst_secondary_model_scale", "loud")
    target = tmp_path / "settings.json"
    messages = page.apply(target)
    assert messages and "vr_voc_inst_secondary_model_scale" in messages[0], messages
    assert not target.exists()


def test_apply_reports_the_typeerror_the_bool_and_str_gates_raise(qapp, tmp_path):
    """The other half of the pair: a `str` field refuses a container with `TypeError`
    (it will not render `[]`), and `Selections`' bool branch refuses the truthy text
    `'False'` with `TypeError` because a truthified flag would activate a model nobody
    chose. Both must come back as messages, not as a traceback through a button."""
    for name, junk in (("batch_size", []), ("is_set_vocal_splitter", "False"),
                       ("vr_model", {"a": 1})):
        page = _page(qapp)
        page.set_field_value_raw(name, junk)
        target = tmp_path / "settings.json"
        messages = page.apply(target)
        assert messages and name in messages[0], (name, messages)
        assert not target.exists(), f"{name}: a refused page still wrote the file"


def test_apply_validates_both_halves_before_writing_either(qapp, tmp_path):
    """The settings half is clean and the selections half is not: nothing may be on disk
    afterwards, because a file holding one new half and one old half is a config the next
    load cannot explain."""
    page = _page(qapp)
    page.set_field_value("window_size", 2048)
    page.set_field_value_raw("mdx_bass_secondary_model_scale", None)
    target = tmp_path / "settings.json"
    assert page.apply(target)
    assert not target.exists()
    assert not (tmp_path / "settings.json").is_file()


def test_a_file_that_refuses_to_be_written_is_reported_as_a_message(qapp, tmp_path):
    """`OSError` out of a save is the everyday Windows case -- a `settings.json` open in an
    editor, a share that went away, a directory where the file should be -- and until this
    catch it escaped the page the same way the refresh refusal did: printed by the binding,
    swallowed by the event loop, with the dialog still open and nothing on screen to say
    why. The target here is a directory, which makes `Settings.save`'s own read-modify-write
    raise a real `OSError` rather than a patched one, and the whole of the assertion is that
    `apply` comes back with a message instead of with that exception (a test that let it
    propagate would show as an error, not a failure).

    `test_a_locked_settings_file_is_reported_in_the_console_...` in
    `tests/test_work_page.py` is the same failure taken through the button, where it really
    does cross a slot boundary.
    """
    page = _page(qapp)
    page.set_field_value("window_size", 1337)

    messages = page.apply(tmp_path)          # a directory, so the write cannot land

    assert messages, "a refused write came back as the empty 'all saved' list"
    assert len(messages) == 1, messages
    assert "not saved" in messages[0] and "could not be written" in messages[0], messages[0]
    assert any(name in messages[0] for name in ("PermissionError", "IsADirectoryError",
                                                "NotADirectoryError")), messages[0]
    assert page.st.window_size != 1337, (
        "a save that failed still adopted its values into the page's state")


def test_a_failure_between_the_two_writes_says_which_half_is_on_disk(qapp, tmp_path,
                                                                    monkeypatch):
    """The one place `apply`'s "write nothing on failure" promise cannot hold, and the
    reason it is *reported* instead of hidden: the settings half goes in, then the
    selections writer raises. Validation is finished by then, so only I/O can land here.

    The message has to name the split -- what is on disk and what is not -- because a file
    with one new half and one old half is the config the next load cannot explain, and a
    user who is told "not saved" would look for a field to fix rather than for a locked
    file. And the page must not adopt the values it could not write: `st` / `sel` stay the
    objects the file agrees with, which is what keeps the controller from assembling a run
    from settings that never reached the disk.
    """
    def refuse_selections(sel, target):
        raise PermissionError(13, "Permission denied: settings.json")

    page = _page(qapp)
    before = page.sel
    page.set_field_value("window_size", 1337)
    target = tmp_path / "settings.json"
    monkeypatch.setattr(selections_module, "save", refuse_selections)

    messages = page.apply(target)

    assert messages and messages[0].startswith("only half saved"), messages
    assert "selections" in messages[0] and "settings" in messages[0], messages[0]
    assert target.is_file(), "the settings half was written but the test says it was not"
    assert settings_module.load(target).window_size == 1337
    assert selections_module.load(target) == before, (
        "the selections block on disk changed while its writer was refused")
    assert page.st.window_size != 1337 and page.sel is before


def test_load_from_pushes_a_new_state_into_every_widget(qapp, tmp_path):
    """The reverse of `apply`: the controller hands the page the state it should show,
    and every field has to arrive in its widget -- not just the ones a user can see."""
    page = _page(qapp)
    st = settings_module.Settings(window_size=2048, save_format="MP3", is_tta=True,
                                  post_process_threshold=0.35)
    sel = selections_module.Selections(vr_model="UVR-DeNoise-Lite",
                                       demucs_bass_secondary_model_scale=0.25,
                                       is_set_vocal_splitter=True)
    page.load_from(st, sel)
    assert page.field_value("window_size") == 2048
    assert page.field_value("save_format") == "MP3"
    assert page.field_value("is_tta") is True
    assert page.field_value("post_process_threshold") == 0.35
    assert page.field_value("vr_model") == "UVR-DeNoise-Lite"
    assert page.field_value("demucs_bass_secondary_model_scale") == 0.25
    assert page.field_value("is_set_vocal_splitter") is True


def test_a_refused_value_is_reported_in_words_that_name_the_field(qapp, tmp_path):
    """The message is what a user acts on: it has to say which field and what the type
    system objected to, and it must not be a bare exception class name."""
    page = _page(qapp)
    page.set_field_value_raw("crop_size", "banana")
    messages = page.apply(tmp_path / "settings.json")
    assert len(messages) == 1, messages
    assert "crop_size" in messages[0] and "banana" in messages[0], messages[0]
    assert "int" in messages[0], messages[0]


def test_the_confirmed_refresh_writes_the_four_tables_through_the_catalog(qapp, tmp_path,
                                                                         monkeypatch):
    """`confirmed=True` is the only way anything here reaches the network, and even then
    the page does no fetching and no path-deriving of its own: it calls `refresh`, which
    owns both the confirmation rule and the `Catalog.write_table` destination."""
    model_root = tmp_path / "weights"
    writable = tmp_path / "appdata" / "model_data"
    for rel in ("VR_Models/model_data", "MDX_Net_Models/model_data",
                "Demucs_Models/model_data"):
        (model_root / rel).mkdir(parents=True, exist_ok=True)
    writable.mkdir(parents=True)
    cat = catalog.Catalog(model_root, writable, fallback_root=None)
    monkeypatch.setattr(model_refresh, "build_catalog", lambda *a, **k: cat)
    fetched = []
    monkeypatch.setattr(model_refresh, "fetch_json",
                        lambda url: fetched.append(url) or {"marker": url})

    page = _page(qapp)
    written = page.refresh_tables(confirmed=True)

    assert fetched == [s.url for s in model_refresh.TABLES], fetched
    assert len(written) == 4 and all(p.is_file() for p in written), written
    assert "wrote" in page.status.toPlainText(), page.status.toPlainText()


def test_the_refresh_button_is_the_only_way_in_to_the_refresh(qapp, monkeypatch):
    """`clicked` is wired to `confirmed=False`, so the button asks and `refresh` refuses
    until something else confirms. A page that passed `confirmed=True` from the click
    would be the official build's start-up fetch again, wearing a button."""
    seen = {}
    page = _page(qapp)
    monkeypatch.setattr(page, "refresh_tables",
                        lambda *, confirmed: seen.update(confirmed=confirmed) or [])
    page.refresh_button.click()
    assert seen == {"confirmed": False}, seen


def test_clicking_refresh_says_no_to_the_user_instead_of_to_the_console(qapp, monkeypatch):
    """The app's only network button, and the whole of what a click does today: `refresh`
    refuses an unconfirmed refresh with `RefreshError` (R-T10-refresh) and the page has to
    turn that refusal into a line in its own status box.

    It did not. `_on_refresh_clicked` called `refresh_tables(confirmed=False)` and let the
    exception leave the Qt slot, and on this wheel that is not a crash and not an error the
    user sees -- PySide6 6.9.3 prints the traceback to stderr and the event loop carries on,
    so the press looked like a button with nothing on it. Measured, not assumed: with the
    catch removed the two assertions below fail in exactly that shape (the status box stays
    empty and the exception lands in `sys.excepthook`), which is what they are for.

    The catch is not a softening of the rule: nothing is fetched here, `confirmed=False`
    still travels to `refresh`, and the message is `model_refresh`'s own words.
    """
    import sys

    fetched = []
    monkeypatch.setattr(model_refresh, "fetch_json", lambda url: fetched.append(url) or {})
    escapes = []
    monkeypatch.setattr(sys, "excepthook",
                        lambda exc_type, value, tb: escapes.append(value))

    page = _page(qapp)
    page.refresh_button.click()

    assert not escapes, f"the refusal left the slot again: {escapes!r}"
    assert not fetched, "a refresh that was refused must still fetch nothing"
    status = page.status.toPlainText()
    assert "refresh refused" in status, status
    assert "confirm" in status, status
    assert "RefreshError" not in status, (
        f"the user gets a class name, not a reason: {status}")


def test_a_refresh_that_breaks_after_the_refusal_is_reported_too(qapp, monkeypatch):
    """The other two branches of the same slot, which shipped untested.

    Task 6 added the catch for `RefreshError` and, in the same `try`, the one for
    `OSError`/`catalog.CatalogError` -- a stalled or refused fetch (`urllib.error.URLError`
    is an `OSError`, and `model_refresh.fetch_json` lets it through) and a writable
    directory that will not take a table (`Catalog.write_table`'s refusal). Nothing in the
    suite reached those two lines, so the branch that turns a failed fetch into a line in
    the status box was only ever *read*, never run.

    It matters for the same reason the refusal did: an exception that leaves a Qt slot is
    printed by the binding and swallowed by the event loop, so the user sees a button that
    did nothing. And this message keeps the class name on purpose -- here the class *is*
    the information a user needs ("OSError"/"URLError" says the machine could not reach the
    network), unlike `RefreshError`, whose text alone already reads as a refusal.
    """
    import sys

    escapes = []
    monkeypatch.setattr(sys, "excepthook",
                        lambda exc_type, value, tb: escapes.append(value))

    def raise_when_called(exc):
        def raiser(*, confirmed):
            raise exc
        return raiser

    page = _page(qapp)
    monkeypatch.setattr(page, "refresh_tables",
                        raise_when_called(OSError(
                            "[WinError 121] The wait operation on the timed out")))
    page.refresh_button.click()

    assert not escapes, f"the fetch failure left the slot again: {escapes!r}"
    status = page.status.toPlainText()
    assert "refresh failed" in status and "OSError" in status, status
    assert "timed out" in status, status

    page.status.clear()
    monkeypatch.setattr(page, "refresh_tables",
                        raise_when_called(catalog.WritableLocationError(
                            "writable_dir sits inside a version-controlled root")))
    page.refresh_button.click()

    assert not escapes, escapes
    status = page.status.toPlainText()
    assert "refresh failed" in status, status
    assert "WritableLocationError" in status, status
    assert "version-controlled" in status, status


def test_an_opened_section_pushes_everything_below_it_down(qapp):
    """The overlap a real session caught and the offscreen numbers did not: a section
    that grew to 1307 px inside a 338 px body painted its fields straight over the
    refresh button, because `ScrollView` never resizes its content (`widgetResizable`
    defaults to False). So the assertion is about the geometry the user gets, not about
    a method having been called: nothing may share a band with anything below it, and
    the scroll area has to be able to reach the bottom.

    Every number here is the binding's own: the heights come from `sizeHint()`, and the
    comparison is between the rects the layout actually assigned.
    """
    page = _page(qapp)
    page.resize(700, 520)
    page.show()
    qapp.processEvents()
    sections = page._sections
    last = sections[-1]
    assert last.sizeHint().height() == 44, "collapsed, a section is only its header"
    last.setExpanded(True)
    # The fit is deferred one turn of the event loop (`_deferred_slot`), so this drives
    # the real path -- click, settle, look -- rather than calling the method by hand.
    for _ in range(3):
        qapp.processEvents()

    assert last.sizeHint().height() > 44, "the section did not grow to its content"
    assert page._body.height() >= page._body.sizeHint().height()
    assert page._body.height() >= sum(g.sizeHint().height() for g in sections), (
        "the body is still sized for the previous state -- the fit ran before the "
        "layout had caught up, which is the one-step-behind bug `_deferred_slot` fixes")
    rects = [(g.y(), g.height()) for g in sections] + [(page.refresh_button.y(), 0)]
    for (y0, h0), (y1, _) in zip(rects, rects[1:]):
        assert y1 >= y0 + h0, f"sections overlap: {rects}"
    assert page._scroller.scrollableHeight() > 0, (
        "the body is not taller than the viewport, so the bottom of the form is unreachable")


def test_the_refresh_button_holds_its_page_weakly(qapp):
    """The bug this pins kills the *process*, at interpreter shutdown, after every test
    has already passed: an access violation inside `destroyQCoreApplication` ->
    `QWidget::destroy`, 4 runs out of 4, when the page's button was connected to a
    callable holding a strong reference to the page. Measured here instead of there,
    because a reference cycle is observable while the process is still alive.

    A page that is collectable once the test lets go of it is the safe shape; the
    `lambda: self.refresh_tables(...)` version stays alive through `gc.collect()` (and
    segfaults on the way out), which is what `_refresh_slot` exists to prevent.
    """
    import gc
    import weakref

    page = _page(qapp)
    ref = weakref.ref(page)
    del page
    gc.collect()
    assert ref() is None, (
        "the page survived its own last reference: something in it holds a strong "
        "reference back -- see app/settings_page._refresh_slot")


def test_the_page_writes_the_shared_file_it_was_given_and_nothing_else(qapp, tmp_path):
    """One file, two writers: `apply` goes through `Settings.save` and
    `selections.save`, so the top-level shape stays what the loaders read -- 50 settings
    keys plus a `selections` block, and no page-private format."""
    page = _page(qapp)
    target = tmp_path / "nested" / "settings.json"
    assert page.apply(target) == []
    raw = json.loads(target.read_text(encoding="utf-8"))
    assert set(raw) - {"selections"} == {f.name for f in
                                         dataclasses.fields(settings_module.Settings)}
    assert set(raw["selections"]) == set(selections_module.SELECTION_FIELD_NAMES)


# -- I7: the fields whose domain the engine already publishes ------------------ #

#: Each enum field's legal values, computed *here* from `gui_data.constants` so the
#: assertion below is "the app's copy moved", not "both copies say the same literal".
#: This is the `R-T7-suffixchoice` shape (`app/dialogs.py:111-128`: the app keeps its own
#: literal and the *test* resolves the engine's constant, as
#: `tests/test_dialogs.py::test_the_config_yaml_suffix_is_the_one_upstreams_popup_appends`
#: does), for the same reason -- and with the same lookup helper, so an upstream rename
#: that changes no behaviour answers with the guard's own sentence instead of a bare
#: `AttributeError` that takes the whole GUI half of the suite red for the wrong reason.
#:
#: The `str()` on the numeric lists is not a convenience: these fields are stored as the
#: *text* the tkinter var held (ruling R-T9-types), so the page's domain is the text form
#: of the engine's numbers -- and `str()` is the conversion the combobox itself performed
#: when it wrote a picked value back out.
def _engine_const(const, name: str):
    """One named value of `gui_data.constants`, looked up by name and never guessed."""
    value = getattr(const, name, None)
    assert value is not None, (
        f"gui_data/constants.py has no `{name}` -- this table reads the engine's own "
        f"domain from it and `app/settings_page.py::ENUM_FIELDS` is the app's copy of "
        f"what that name held, so if upstream renamed it the copy and this lookup move "
        f"together.")
    return value


ENUM_SOURCES = {
    "save_format": lambda c: tuple(_engine_const(c, n) for n in ("WAV", "FLAC", "MP3")),
    "mp3_bit_set": lambda c: tuple(_engine_const(c, "MP3_BIT_RATES")),
    "wav_type_set": lambda c: tuple(_engine_const(c, "WAV_TYPE")),
    "device_set": lambda c: tuple(str(v) for v in _engine_const(c, "GPU_DEVICE_NUM_OPTS")),
    "denoise_option": lambda c: tuple(_engine_const(c, "MDX_DENOISE_OPTION")),
    "deverb_vocal_opt": lambda c: tuple(_engine_const(c, "DEVERB_MAPPER")),
    "compensate": lambda c: tuple(str(v) for v in _engine_const(c, "VOL_COMPENSATION")),
    "batch_size": lambda c: tuple(str(v) for v in _engine_const(c, "BATCH_SIZE")),
    "mdx_batch_size": lambda c: tuple(str(v) for v in _engine_const(c, "BATCH_SIZE")),
    "segment": lambda c: tuple(str(v) for v in _engine_const(c, "DEMUCS_SEGMENTS")),
    "overlap": lambda c: tuple(str(v) for v in _engine_const(c, "DEMUCS_OVERLAP")),
    "overlap_mdx": lambda c: tuple(str(v) for v in _engine_const(c, "MDX_OVERLAP")),
    "overlap_mdx23": lambda c: tuple(str(v) for v in _engine_const(c, "MDX23_OVERLAP")),
    "semitone_shift": lambda c: tuple(str(v) for v in _engine_const(c, "SEMITONE_SEL")),
    "demucs_stems": lambda c: tuple(dict.fromkeys(
        (*_engine_const(c, "DEMUCS_4_STEM_OPTIONS"),
         *_engine_const(c, "DEMUCS_6_STEM_OPTIONS"),
         *_engine_const(c, "DEMUCS_2_STEM_OPTIONS")))),
    "vocal_splitter_primary_label": lambda c: (
        "", _engine_const(c, "VOCAL_STEM_ONLY"), _engine_const(c, "INST_STEM_ONLY")),
    "vocal_splitter_secondary_label": lambda c: (
        "", _engine_const(c, "VOCAL_STEM_ONLY"), _engine_const(c, "INST_STEM_ONLY")),
}


def test_the_enum_table_covers_exactly_the_fields_the_review_named():
    """The key set is part of the fix: an engine-listed field that silently stays free
    text is the drift this branch has been chasing all plan."""
    assert set(settings_page.ENUM_FIELDS) == set(ENUM_SOURCES), (
        sorted(set(settings_page.ENUM_FIELDS) ^ set(ENUM_SOURCES)))


def test_every_enum_field_is_a_non_editable_combo_holding_the_engine_list(qapp):
    """I7's point is the *widget*, not the validator: a box that cannot hold a wrong
    value beats a check that rejects one afterwards, because the unchecked path (typed
    text, `''`, `'WAV '`) is exactly what produced the no-output runs in C1."""
    import gui_data.constants as const

    page = _page(qapp)
    for name, derive in ENUM_SOURCES.items():
        widget = page._widgets[name]
        assert isinstance(widget, fluentqt.ComboBox), (
            f"{name} is still a {type(widget).__name__}: a free-text box for a field the "
            f"engine lists values for is the bug I7 is about")
        assert not widget.isEditable(), f"{name}: an editable combo can be typed into"
        offered = {widget.itemText(i) for i in range(widget.count())}
        assert offered == set(derive(const)), (
            f"{name}: the combo offers {sorted(offered)}, the engine's constant holds "
            f"{sorted(derive(const))}")
        assert set(settings_page.ENUM_FIELDS[name]) == set(derive(const)), (
            f"{name}: app/settings_page.py's own copy of the domain is "
            f"{settings_page.ENUM_FIELDS[name]}, the engine's is {derive(const)}")


def test_no_default_value_falls_outside_its_own_enum_list(qapp):
    """A combo whose items do not contain the loaded default would show item 0 and save
    item 0: the intent the file held is dropped, which is the same silent data loss the
    `_MISSING` import guard refuses for a field with no widget at all."""
    st, sel = settings_module.Settings(), selections_module.Selections()
    page = settings_page.SettingsPage(settings_obj=st, selections_obj=sel)
    for name in settings_page.ENUM_FIELDS:
        holder = st if name in page._types else sel
        value = str(getattr(holder, name))
        assert value in settings_page.ENUM_FIELDS[name], (
            f"{name}'s default {value!r} is not in {settings_page.ENUM_FIELDS[name]} -- "
            f"the page would display and save a different value from the one it loaded")
        assert page.field_value(name) == value, name


def test_a_value_outside_the_domain_stays_visible_instead_of_being_silently_replaced(
        qapp):
    """The file is still a place a bad value can arrive from (hand-edited, or written by
    an older build), and the page's rule is the one `apply()` already follows for a junk
    number: show what is there, refuse to pretend. Run-time is where it gets refused --
    `app/session.py`'s `check_save_format` for `save_format`, the controller's trailing
    `except Exception` for the `DEVERB_MAPPER` key -- so the widget's job is only to stop
    the value being *created* here."""
    page = _page(qapp)
    page.load_from(settings_module.Settings(save_format="mp3"),
                   selections_module.Selections())
    assert page.field_value("save_format") == "mp3", page.field_value("save_format")
    widget = page._widgets["save_format"]
    assert "mp3" in [widget.itemText(i) for i in range(widget.count())]


def test_typing_a_wrong_value_is_not_an_option(qapp):
    """The domain check the LineEdit used to rely on is in the widget now: a
    non-editable `ComboBox` has no keyboard path to a value that is not one of its
    items, so `setCurrentText` of a stranger leaves the selection inside the domain."""
    page = _page(qapp)
    widget = page._widgets["deverb_vocal_opt"]
    assert isinstance(widget, fluentqt.ComboBox)
    widget.setCurrentText("nope")                        # what a user typing would do
    assert page.field_value("deverb_vocal_opt") in settings_page.ENUM_FIELDS[
        "deverb_vocal_opt"], page.field_value("deverb_vocal_opt")


def test_the_blend_scales_are_bounded_to_their_real_domain(qapp):
    """I7(c): twelve blend weights between 0 and 1, bounded by `setRange(-1e6, 1e6)` --
    a range no other guard backs up, because nothing in the engine bounds them either.
    The names come from `Selections`' own fields, so a thirteenth scale inherits the
    bound instead of being forgotten."""
    page = _page(qapp)
    assert len(settings_page.BLEND_SCALE_FIELDS) == 12, settings_page.BLEND_SCALE_FIELDS
    for name in settings_page.BLEND_SCALE_FIELDS:
        widget = page._widgets[name]
        assert isinstance(widget, fluentqt.NumberBox), name
        assert (widget.minimum(), widget.maximum()) == (0.0, 1.0), (
            f"{name} is a blend weight; {widget.minimum()}..{widget.maximum()} is not "
            f"its domain")
    other = page._widgets["post_process_threshold"]
    assert other.maximum() > 1.0, "the bound moved onto a field that is not a scale"


def test_the_splitters_save_filter_is_reachable_from_the_page(qapp, tmp_path):
    """I7(a): `plan.py:467-475` matches the two splitter labels against 'Vocals Only' /
    'Instrumental Only' -- strings that appeared nowhere in `app/**`, so the engine's
    save-filter branch was unreachable from this GUI however the user clicked. The combo
    now offers exactly those two, and what the page saves is what the plan reads."""
    import gui_data.constants as const

    page = _page(qapp)
    for name in ("vocal_splitter_primary_label", "vocal_splitter_secondary_label"):
        offered = {page._widgets[name].itemText(i)
                   for i in range(page._widgets[name].count())}
        assert offered == {"", const.VOCAL_STEM_ONLY, const.INST_STEM_ONLY}, (name, offered)

    page.set_field_value("vocal_splitter_primary_label", const.VOCAL_STEM_ONLY)
    page.set_field_value("set_vocal_splitter", "UVR-BVE-458n")
    page.set_field_value("is_set_vocal_splitter", True)
    target = tmp_path / "settings.json"
    messages = page.apply(target)
    assert messages == [], messages
    reloaded = selections_module.load(target)
    assert reloaded.vocal_splitter_primary_label == const.VOCAL_STEM_ONLY
    plan_dict = reloaded.to_plan_dict()
    assert plan_dict["vocal_splitter"]["primary_stem_only_label"] == const.VOCAL_STEM_ONLY
