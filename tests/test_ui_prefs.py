"""The `ui` block: front-end state the engine never reads, in the fork's own file.

Three loaders now read one JSON file (`settings.load`, `selections.load`,
`ui_prefs.load`) and three writers claim a piece of it. This file is the proof that they
coexist: each writer must preserve the other two's bytes, and the parse must be
`selections._coerce`'s, not a fourth opinion about what a string is -- the rule finding
I9 established for the two halves that already share the file, applied to the third.

The block holds one field today. It still goes through the whole machinery, because the
machinery is two lines and the alternative is the exact bug I9 named: a value the
constructor accepts and the loader refuses.
"""
from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest
import settings as settings_module
import selections as selections_module
import ui_prefs


def test_load_of_a_missing_file_is_the_empty_prefs(tmp_path):
    assert ui_prefs.load(tmp_path / "none.json") == ui_prefs.UiPrefs()
    assert ui_prefs.UiPrefs().export_path == ""


def test_round_trip_keeps_the_other_two_blocks(tmp_path):
    target = tmp_path / "settings.json"
    st = settings_module.Settings(gpu_conversion=True, crop_size=352)
    sel = selections_module.Selections(vr_voc_inst_secondary_model_scale="0.7")
    st.save(target)
    selections_module.save(sel, target)
    written = ui_prefs.save(ui_prefs.UiPrefs(export_path=str(target.parent / "out")), target)
    assert written == target

    raw = json.loads(target.read_text(encoding="utf-8"))
    assert raw["ui"] == {"export_path": str(target.parent / "out")}
    # The two earlier writers' bytes are still there, and still readable by their own
    # loaders -- a merge that rewrites the top level would silently delete one of them.
    assert "selections" in raw and "save_format" in raw
    assert settings_module.load(target).crop_size == 352
    assert selections_module.load(target) == sel
    assert ui_prefs.load(target).export_path == str(target.parent / "out")


def test_the_other_two_writers_preserve_the_ui_block(tmp_path):
    """The reverse of `test_round_trip_keeps_the_other_two_blocks`, and the half that was
    missing: three writers claim one file, so proving the *newest* one merges is not
    proving the other two do. `Settings.save` is the one a work-page format click reaches
    (`Controller._apply_setting`), so a `save` that rewrote the top level would delete the
    configured output folder on the next format change -- silently, and only for users who
    had configured one. Both directions are asserted by reading the file back with each
    block's own loader, which is what the next launch does."""
    target = tmp_path / "settings.json"
    prefs = ui_prefs.UiPrefs(export_path=str(tmp_path / "out"))
    ui_prefs.save(prefs, target)
    settings_module.Settings(save_format="FLAC").save(target)
    assert ui_prefs.load(target) == prefs, "Settings.save dropped the ui block"
    sel = selections_module.Selections(vr_voc_inst_secondary_model_scale="0.7")
    selections_module.save(sel, target)
    assert ui_prefs.load(target) == prefs, "selections.save dropped the ui block"
    raw = json.loads(target.read_text(encoding="utf-8"))
    assert raw["save_format"] == "FLAC"
    assert raw["selections"]["vr_voc_inst_secondary_model_scale"] == \
        sel.vr_voc_inst_secondary_model_scale


def test_constructing_from_text_and_reading_the_file_agree(tmp_path):
    """`UiPrefs(export_path=X)` == `load()` of a file that says X, including the
    whitespace: `_coerce`'s `str` branch passes the very object given, so a path a user
    pasted with a trailing space survives both doors -- and a parse that stripped it
    here would be a fourth opinion about what a string is."""
    target = tmp_path / "settings.json"
    for value in ("E:/out", "E:/out ", "", "  "):
        ui_prefs.save(ui_prefs.UiPrefs(export_path=value), target)
        assert ui_prefs.load(target) == ui_prefs.UiPrefs(export_path=value)


def test_unknown_keys_in_the_ui_block_are_dropped_and_a_wrong_type_raises(tmp_path):
    target = tmp_path / "settings.json"
    target.write_text(json.dumps({"ui": {"not_a_field": 1}}), encoding="utf-8")
    assert ui_prefs.load(target) == ui_prefs.UiPrefs()
    target.write_text(json.dumps({"ui": {"export_path": ["not", "a", "path"]}}),
                      encoding="utf-8")
    with pytest.raises(TypeError) as exc:
        ui_prefs.load(target)
    assert "ui.export_path" in str(exc.value)
    assert str(target) in str(exc.value)


def test_a_number_in_the_str_field_is_stringified_not_refused(tmp_path):
    # `_coerce`'s domain, exactly: it stringifies a number in a `str` field because that
    # is the one rewrite a widget performs on its own numeric content. A hand-edited
    # `"export_path": 7` therefore loads as `"7"` here just as `Selections(...)` would
    # take it -- refusing it would be a second, stricter parse.
    target = tmp_path / "settings.json"
    target.write_text(json.dumps({"ui": {"export_path": 7}}), encoding="utf-8")
    assert ui_prefs.load(target).export_path == "7"


def test_a_file_that_is_not_an_object_is_a_type_error_naming_the_file(tmp_path):
    target = tmp_path / "settings.json"
    target.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(TypeError) as exc:
        ui_prefs.load(target)
    assert "must hold an object" in str(exc.value)


def test_importing_ui_prefs_pulls_in_no_qt():
    """The block is readable by a front end that has no Qt at all.

    Two-sided, per this suite's rule for prohibitions: the refusal alone would be
    satisfied by deleting the module, so `test_the_module_is_the_whole_public_surface_it_claims`
    below asks that it be there and say something.
    """
    code = ("import sys, ui_prefs;"
            "bad = sorted(m for m in sys.modules if m.startswith(('PySide6', 'fluentqt', 'tkinter')));"
            "sys.exit(1 if bad else 0)")
    assert subprocess.run([sys.executable, "-c", code], cwd=str(Path(ui_prefs.__file__).parent)
                          ).returncode == 0


def test_the_module_is_the_whole_public_surface_it_claims():
    """The two-sided half: not just "no Qt", but "the block is there and complete".

    A purity assertion alone is satisfied by deleting the module, so this one names the
    three things a caller may use and nothing else -- `save` is exercised by every other
    test in this file against a `tmp_path`, never here, because a test that writes to the
    real `config_file()` leaves its result for the rest of the session.
    """
    assert callable(ui_prefs.load) and callable(ui_prefs.save)
    assert ui_prefs.UiPrefs() == ui_prefs.UiPrefs(export_path="")
    assert [f.name for f in dataclasses.fields(ui_prefs.UiPrefs)] == ["export_path"]
