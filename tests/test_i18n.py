"""Guards for the front-end i18n layer: completeness, coverage, and the locale seam.

Three properties matter and each gets a test: every catalog row carries all three
locales (a key cannot be English-only by accident); every `tr("...")` literal in `app/`
resolves to a real key (a typo would otherwise raise KeyError inside a Qt slot at runtime);
and the persisted choice round-trips without disturbing the other settings.json blocks.
The work-page tests build under a non-English locale to prove the chrome actually switches,
then restore English so the rest of the suite's pinned copy is unaffected.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from PySide6.QtCore import QLocale

from app import i18n
from tests.test_work_page import _ui_ctrl

APP = Path(__file__).resolve().parent.parent / "app"


def test_every_catalog_row_carries_all_three_locales():
    assert [code for code, _ in i18n.LANGUAGES] == ["en", "zh-Hant", "zh-Hans"]
    for key, row in i18n.CATALOG.items():
        assert isinstance(row, tuple) and len(row) == 3, key
        for text in row:
            assert isinstance(text, str) and text.strip(), (key, text)


def test_every_tr_literal_in_the_app_resolves_to_a_catalog_key():
    used = set()
    for path in sorted(APP.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "tr" and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)):
                used.add(node.args[0].value)
    assert used, "no tr() call sites found -- the scan is broken"
    missing = used - set(i18n.CATALOG)
    assert not missing, sorted(missing)


def test_tr_switches_with_the_locale_and_falls_back_to_english():
    try:
        i18n.set_locale("en")
        assert i18n.tr("seg1") == "1 · What to process"
        i18n.set_locale("zh-Hant")
        assert i18n.tr("seg1") == "1 · 處理什麼"
        i18n.set_locale("zh-Hans")
        assert i18n.tr("seg1") == "1 · 处理什么"
        assert i18n.tr("count_queued", n=2) == "已排队 2 个文件"
        i18n.set_locale("not-a-locale")
        assert i18n.current_locale() == "en"
        assert i18n.tr("seg1") == "1 · What to process"
    finally:
        i18n.set_locale("en")


@pytest.mark.parametrize("tags,expected", [
    (["zh-TW"], "zh-Hant"), (["zh-HK"], "zh-Hant"), (["zh-Hant-TW"], "zh-Hant"),
    (["zh-CN"], "zh-Hans"), (["zh-SG"], "zh-Hans"), (["zh-Hans-CN"], "zh-Hans"),
    (["en-US"], "en"), (["fr-FR"], "en"),
])
def test_resolve_system_maps_regions_to_the_right_script(monkeypatch, tags, expected):
    class _Stub:
        def uiLanguages(self):
            return tags

        def name(self):
            return tags[0]
    monkeypatch.setattr(i18n.QLocale, "system", staticmethod(lambda: _Stub()))
    assert i18n.resolve_system() == expected


def test_language_round_trips_without_touching_the_other_blocks(tmp_path):
    target = tmp_path / "settings.json"
    target.write_text(json.dumps({"save_format": "WAV", "ui": {"export_path": "X"}}),
                      encoding="utf-8")
    assert i18n.load_language(target) == "system"
    i18n.save_language("zh-Hans", target)
    raw = json.loads(target.read_text(encoding="utf-8"))
    assert raw["i18n"] == {"language": "zh-Hans"}
    assert raw["ui"] == {"export_path": "X"}
    assert raw["save_format"] == "WAV"
    assert i18n.load_language(target) == "zh-Hans"


def test_the_language_combo_lives_in_settings_and_persists(qapp, tmp_path, monkeypatch):
    """Moved off the run band: it changes nothing about a run. The page still persists
    immediately and still says a restart is what applies it."""
    ctrl = _ui_ctrl(qapp, tmp_path, monkeypatch)
    ctrl._on_settings_requested()
    page = ctrl.shell._settings_page
    codes = page._lang_codes
    page.language_combo.setCurrentIndex(codes.index("zh-Hans"))
    assert i18n.load_language() == "zh-Hans"
    assert not hasattr(ctrl.work_page, "language_combo"), (
        "the run band still carries the language choice")
    # restoring keeps the shared sandbox clean for the tests that follow
    page.language_combo.setCurrentIndex(codes.index("system"))
    assert i18n.load_language() == "system"


@pytest.mark.parametrize("locale", ["zh-Hant", "zh-Hans"])
def test_the_chrome_actually_renders_in_the_selected_locale(qapp, tmp_path, monkeypatch,
                                                            locale):
    try:
        i18n.set_locale(locale)
        ctrl = _ui_ctrl(qapp, tmp_path, monkeypatch)
        page = ctrl.work_page
        assert page.input_edit.placeholderText() == i18n.tr("input_ph")
        assert page.gpu_check.text() == i18n.tr("gpu_check")
        assert page.orientation_label.text() == i18n.tr("orientation")
    finally:
        i18n.set_locale("en")