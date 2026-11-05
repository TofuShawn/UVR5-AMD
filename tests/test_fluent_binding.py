"""The upgrade tripwire for a bus-factor-1 dependency.

Fluent-Qt is one author, 6 months old at the time this fork adopted it, and has
removed public API between minor versions without a compatibility layer. This test
constructs every Fluent class the app uses, by *name*, from a pinned set -- it does
not scan `dir(fluentqt)`, because a scan would report green when a class is deleted
and simply disappear from the listing.

No `importorskip` anywhere: a venv without FluentQt is a broken venv for this plan,
and a silent skip is exactly the failure mode the fork rules against (absent means
blocked, loudly -- same reasoning as the ROCm-only pin store).
"""
from __future__ import annotations

import fluentqt
import pytest

#: Every Fluent class `app/**` is allowed to depend on. Adding a class to the app
#: means adding it here in the same commit, so the list is a reviewed surface.
UI_CLASSES = frozenset({
    "Window", "Card", "FluentWidget", "Label", "Button", "ToggleButton",
    "HyperlinkButton", "ComboBox", "MultiSelectComboBox", "AutoSuggestBox",
    "LineEdit", "PasswordBox", "TextEdit", "NumberBox", "Slider", "ProgressBar",
    "ProgressRing", "CheckBox", "ToggleSwitch", "RadioButton", "Expander",
    "ContentDialog", "InfoBar", "ToolTip", "SelectorBar", "SelectorBarItem",
    "Pivot", "PivotItem", "TabView", "TabViewItem", "Divider", "ScrollView",
})

#: Classes that need more than an optional parent to construct.
_NEEDS_ARGS = {
    "Label": ("text",),
    "Button": ("text",),
    "ToggleButton": ("text",),
    "HyperlinkButton": ("text",),
    "SelectorBarItem": ("text",),
    "PivotItem": ("text",),
    "TabViewItem": ("text",),
}


def test_every_declared_class_exists():
    missing = sorted(name for name in UI_CLASSES if not hasattr(fluentqt, name))
    assert not missing, f"fluentqt {getattr(fluentqt, '__version__', '?')} no longer exports {missing}"


def test_every_declared_class_constructs(qapp):
    fluentqt.initialize_resources()
    failures = []
    for name in sorted(UI_CLASSES):
        try:
            getattr(fluentqt, name)(*_NEEDS_ARGS.get(name, ()))
        except Exception as exc:                      # noqa: BLE001 - report, don't mask
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    assert not failures, "cannot construct:\n" + "\n".join(failures)


def test_version_identity_is_the_pinned_triple(qapp):
    info = fluentqt.binding_build_info()
    assert info["fluentqt_version"] == "1.11.2", info
    assert info["pyside6_version"] == info["shiboken6_version"] == "6.9.3", info
    assert info["qt_compile_version"] == info["qt_runtime_version"], (
        f"Qt compile {info['qt_compile_version']} != runtime {info['qt_runtime_version']}")


def test_the_measured_binding_holes_are_still_holes(qapp):
    """Pinned defects, not wishes. Each passing assertion means the app must keep its
    workaround; when one starts failing, delete the workaround in the same commit.
    """
    with pytest.raises(Exception) as excinfo:          # NameError from the binding
        fluentqt.FluentMenu()
    assert "not defined" in str(excinfo.value), (
        "FluentMenu constructs now -- it may replace QMenu in app/work_page.py")

    # The same Shiboken overload defect, in its *string-first* form. The app therefore
    # builds an Expander with `Expander(parent)` + `setHeaderText()`, never `Expander(title)`.
    # Measured on 1.11.2, the two string-first forms die differently: the one-argument
    # call reaches the binding's own `__init__` lookup and fails there (`NameError:
    # ... '_fluentqt' is not defined`), while the two-argument call never gets that far --
    # Shiboken sees one `(parent)` overload and refuses the extra argument up front
    # (`TypeError: Expander() takes at most 1 argument (2 given)`). Both are holes, so
    # each case pins *its own* message: a fixed binding must fail this needle, not
    # merely raise a different exception on the way to the same refusal.
    from PySide6.QtWidgets import QWidget

    host = QWidget()
    for label, args, needle in (("Expander('title')", ("t",), "not defined"),
                                ("Expander('title', parent)", ("t", host), "takes at most 1 argument")):
        with pytest.raises(Exception) as excinfo:
            fluentqt.Expander(*args)
        assert needle in str(excinfo.value), f"{label} works now -- app/settings_page.py may use it"
    fluentqt.Expander()                                  # the form the app relies on
    fluentqt.Expander(host)
    fluentqt.SelectorBarItem("vr")                       # text-only, no parent

    nav_signals = [n for n in ("itemSelected", "currentItemChanged", "selectionChanged",
                               "addItem", "menuItems") if hasattr(fluentqt.NavigationView, n)]
    assert not nav_signals, (
        f"NavigationView gained {nav_signals} -- the nav shell becomes possible")

    assert not hasattr(fluentqt.ComboBox, "textChanged"), (
        "ComboBox gained textChanged -- drop the fluentLineEdit workaround")

    # Method names that are *not* what the C++ docs suggest. If these flip, the app's
    # call sites get updated -- not these assertions.
    assert not hasattr(fluentqt.InfoBar, "show_message")
    assert not hasattr(fluentqt.ContentDialog, "setContentWidget")
    assert hasattr(fluentqt.ContentDialog, "setContent")
    assert hasattr(fluentqt.SelectorBar, "selectedIndex")
    assert not hasattr(fluentqt.NumberBox, "setSingleStep")

    # The title/content verbs are *asymmetric* between the two containers, and the
    # asymmetry is measured, not guessed: Expander speaks setHeaderText/setContentWidget
    # and has no setTitle at all, while ContentDialog speaks setContent and has no
    # setContentWidget. Both call sites in Tasks 5-7 depend on this split.
    assert hasattr(fluentqt.Expander, "setHeaderText")
    assert not hasattr(fluentqt.Expander, "setTitle")
    assert hasattr(fluentqt.Expander, "setContentWidget")
    assert not hasattr(fluentqt.Expander, "setContent")


def test_plain_qwidget_is_not_a_fluent_surface(qapp):
    """The rule the container guard in Task 5 enforces, proven here at the library
    level: token overrides are refused for a plain QWidget (spec §5.3)."""
    from PySide6.QtWidgets import QWidget

    plain = QWidget()
    assert fluentqt.set_widget_theme_overrides(plain, {"light": {"textPrimary": "#000000"}}) is False
    card = fluentqt.Card()
    assert fluentqt.set_widget_theme_overrides(card, {"light": {"textPrimary": "#000000"}}) is True
