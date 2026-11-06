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

    Wave 2 minor: the exception **type** is now pinned per case (a truly fixed binding
    fails on the type, not merely on the message needle), the symbols that may vanish
    are read through `getattr(..., None)` with a real message so a disappearance is
    reported rather than crashing as an unrelated `AttributeError`, and the bare
    `hasattr` asserts each carry a message naming the call site. Crucially the failure
    texts no longer say "works now -- app/... may use it": a binding that merely
    *reworded* while staying broken used to trip that claim and invite the reader to
    delete a still-needed workaround. A changed needle now says "re-measure".
    """
    # --- FluentMenu: the Shiboken overload hole, a NameError raised at __init__ eval. ---
    FluentMenu = getattr(fluentqt, "FluentMenu", None)
    assert FluentMenu is not None, (
        "fluentqt no longer exposes the name FluentMenu at all. This hole used to be "
        "'it constructs to a NameError'; a symbol that is *gone* is a different defect, "
        "so re-read app/work_page.py's QMenu usage before concluding anything.")
    with pytest.raises(NameError) as excinfo:
        FluentMenu()
    assert "not defined" in str(excinfo.value), (
        "FluentMenu() still raises NameError but not the `_fluentqt ... is not defined` "
        "the app's QMenu workaround assumes -- the binding changed shape. Re-measure "
        "before touching app/work_page.py; a different message is NOT a fixed binding.")

    # The same Shiboken overload defect, in its *string-first* form. The app therefore
    # builds an Expander with `Expander(parent)` + `setHeaderText()`, never
    # `Expander(title)`. Measured on 1.11.2 the two string-first forms die with
    # *different types*: the one-argument call reaches the binding's own `__init__`
    # lookup and raises NameError; the two-argument call never gets there and Shiboken
    # raises TypeError up front. Both are holes, so each pins its own type AND its own
    # needle -- a fixed binding (or one that only reworded) must fail this, not merely
    # raise some other exception on the way to the same refusal.
    from PySide6.QtWidgets import QWidget

    host = QWidget()
    for label, args, exc_type, needle in (
            ("Expander('title')", ("t",), NameError, "not defined"),
            ("Expander('title', parent)", ("t", host), TypeError,
             "takes at most 1 argument")):
        with pytest.raises(exc_type) as excinfo:
            fluentqt.Expander(*args)
        assert needle in str(excinfo.value), (
            f"{label} still raises {exc_type.__name__} but not with `{needle}` -- the "
            "binding reworded or changed shape; re-measure before changing "
            "app/settings_page.py's Expander call, do not assume the workaround is dead.")
    fluentqt.Expander()                                  # the form the app relies on
    fluentqt.Expander(host)
    fluentqt.SelectorBarItem("vr")                       # text-only, no parent

    # NavigationView: no selection signal means the nav shell stays impossible. Read the
    # class through getattr so a vanished class is reported, not crashed.
    NavigationView = getattr(fluentqt, "NavigationView", None)
    assert NavigationView is not None, (
        "fluentqt.NavigationView is gone entirely -- this hole assumed the class exists "
        "but exposes no selection API; a missing class needs a re-read of app/main.py's "
        "single-window structure, not a green pass.")
    nav_signals = [n for n in ("itemSelected", "currentItemChanged", "selectionChanged",
                               "addItem", "menuItems") if hasattr(NavigationView, n)]
    assert not nav_signals, (
        f"NavigationView gained {nav_signals} -- the nav shell becomes possible; revisit "
        "app/main.py before assuming a NavigationView nav.")

    assert not hasattr(fluentqt.ComboBox, "textChanged"), (
        "ComboBox gained textChanged -- drop the fluentLineEdit workaround in the app.")

    # Method names that are *not* what the C++ docs suggest. If these flip, the app's
    # call sites get updated -- not these assertions. Each now names its call site.
    assert not hasattr(fluentqt.InfoBar, "show_message"), (
        "InfoBar.show_message exists now -- app/dialogs.py drives InfoBar with a "
        "different verb; re-check the call site rather than assuming the hole closed.")
    assert not hasattr(fluentqt.ContentDialog, "setContentWidget"), (
        "ContentDialog.setContentWidget exists -- but app/dialogs.py uses setContent; "
        "check that call site before migrating.")
    assert hasattr(fluentqt.ContentDialog, "setContent"), (
        "ContentDialog.setContent disappeared -- app/dialogs.py builds its body through "
        "this exact verb; the workaround call site is now broken.")
    assert hasattr(fluentqt.SelectorBar, "selectedIndex"), (
        "SelectorBar.selectedIndex disappeared -- app/work_page.py's arch row reads it.")
    assert not hasattr(fluentqt.NumberBox, "setSingleStep"), (
        "NumberBox.setSingleStep exists now -- revisit app/settings_page.py's manual "
        "step handling; it used to be applied by hand.")

    # The title/content verbs are *asymmetric* between the two containers, and the
    # asymmetry is measured, not guessed: Expander speaks setHeaderText/setContentWidget
    # and has no setTitle at all, while ContentDialog speaks setContent and has no
    # setContentWidget. Both call sites in Tasks 5-7 depend on this split.
    assert hasattr(fluentqt.Expander, "setHeaderText"), (
        "Expander.setHeaderText disappeared -- app/settings_page.py titles every section "
        "with it (Expander has no setTitle).")
    assert not hasattr(fluentqt.Expander, "setTitle"), (
        "Expander.setTitle exists -- the container asymmetry this pins may have moved; "
        "re-read app/settings_page.py before switching verbs.")
    assert hasattr(fluentqt.Expander, "setContentWidget"), (
        "Expander.setContentWidget disappeared -- app/settings_page.py puts each "
        "section's body in through this verb (ContentDialog uses setContent instead).")
    assert not hasattr(fluentqt.Expander, "setContent"), (
        "Expander.setContent exists -- the asymmetry this test records has moved; "
        "re-measure before touching app/settings_page.py.")

    # Wave 2 minor: the app docstrings in `app/dialogs.py` (:11-12, :22-23) and
    # `app/work_page.py` (:431-432) credit *Task 1* (this file) with pinning the exact
    # InfoBar and ContentDialog verbs they rely on. Task 1 only pinned `show_message`
    # and the setContent/setContentWidget split, so the credits were claims about pins
    # that did not exist. They are made true here -- adding the pin, rather than editing
    # the (wave-1-owned) app docstrings to drop the credit. Each was measured on the
    # pinned wheel at HEAD before being asserted.
    assert not hasattr(fluentqt.InfoBar, "setDuration"), (
        "InfoBar.setDuration exists now -- app/dialogs.py and app/work_page.py both "
        "say Task 1 pinned its absence and dismiss bars by queue on that basis; if the "
        "wheel gained an auto-dismiss, revisit R-T7-infobar before trusting the queue.")
    assert hasattr(fluentqt.ContentDialog, "setTitle"), (
        "ContentDialog.setTitle disappeared -- app/dialogs.py's title call site is "
        "credited to this pin.")
    assert hasattr(fluentqt.ContentDialog, "setPrimaryButtonText"), (
        "ContentDialog.setPrimaryButtonText disappeared -- app/dialogs.py names the "
        "confirm button through it.")
    assert hasattr(fluentqt.ContentDialog, "setCloseButtonText"), (
        "ContentDialog.setCloseButtonText disappeared -- app/dialogs.py names the "
        "cancel button through it.")
    assert hasattr(fluentqt.ContentDialog, "result"), (
        "ContentDialog.result disappeared -- app/dialogs.py reads the yes/no verdict "
        "from exactly this verb.")
    assert hasattr(fluentqt.ContentDialog, "closed"), (
        "ContentDialog.closed disappeared -- app/dialogs.py connects dismissal to it.")


def test_plain_qwidget_is_not_a_fluent_surface(qapp):
    """The rule the container guard in Task 5 enforces, proven here at the library
    level: token overrides are refused for a plain QWidget (spec §5.3)."""
    from PySide6.QtWidgets import QWidget

    # getattr so a vanished verb is reported as the §5.3 mechanism breaking, not as an
    # unrelated AttributeError (wave 2 minor).
    set_overrides = getattr(fluentqt, "set_widget_theme_overrides", None)
    assert set_overrides is not None, (
        "fluentqt.set_widget_theme_overrides is gone -- Ruling A keeps the app at zero "
        "token overrides and this test proves the verb still *refuses* a plain QWidget; "
        "a missing verb needs a re-read of spec §5.3, not a green pass.")
    plain = QWidget()
    assert set_overrides(plain, {"light": {"textPrimary": "#000000"}}) is False
    card = fluentqt.Card()
    assert set_overrides(card, {"light": {"textPrimary": "#000000"}}) is True
