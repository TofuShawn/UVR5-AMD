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


def test_apply_round_trips_every_field(qapp, tmp_path):
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
        assert isinstance(widget, (fluentqt.CheckBox, fluentqt.NumberBox, fluentqt.LineEdit))


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
