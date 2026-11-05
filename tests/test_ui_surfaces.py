"""Every surface the app paints must be a Fluent widget.

A plain QWidget keeps the native palette while its Fluent children switch to dark
tokens, which reads as white-on-white. On this machine that bug is invisible under a
real session (the desktop theme is dark, so the native background is #1e1e1e and the
contrast failure hides) and obvious offscreen (#efefef) -- so the guard runs offscreen
and asserts the *class*, not the pixels. Spec §5.3.

Two guards live here:

* `test_every_widget_in_the_tree_is_fluent_or_a_known_container` -- the runtime tree.
* `test_app_only_touches_the_reviewed_fluent_surface` -- the compile-time one:
  `tests/test_fluent_binding.UI_CLASSES` declares itself "every Fluent class `app/**`
  is allowed to depend on", and Task 1's review noted nothing enforced it, so the list
  quietly stopped meaning anything. The AST scan makes it a reviewed surface again.
"""
from __future__ import annotations

import ast
from pathlib import Path

from PySide6.QtWidgets import QDialog, QWidget

from app import main
from tests.test_fluent_binding import UI_CLASSES

#: Bare QWidgets and plain Qt widgets the pinned wheel builds *inside* its own controls.
#: Measured offscreen at FluentQt 1.11.2 over the whole built tree (52 descendants, 15 of
#: them non-Fluent -- full census in task-5-report.md), keyed by the nearest Fluent
#: ancestor's **binding** class name: `Window` hosts its content area and, through
#: `TitleBar`, its caption buttons in `QWidget`s; `Expander` clips through
#: `fluentExpanderClip`; and the `QComboBox` / `QTextEdit` bases install Qt's own popup and
#: scroll-area parts (`QFrame`, `QListView`, `QScrollBar`, `QTextEdit`, `qt_scrollarea_*`).
#:
#: Keyed by the *owner*, so this is not a class-level amnesty: the app parents everything to
#: the `Card` it builds, so an app-authored bare `QWidget` gets owner `Card`, appears nowhere
#: in this table, and fails -- which is the bug class spec §5.3 exists to catch. A new entry
#: needs a measurement of the wheel, not a guess.
FLUENT_INTERNALS: dict[str, set[str]] = {
    "Window": {"QWidget"},
    "TitleBar": {"QWidget"},
    "Expander": {"QWidget"},
    "ComboBox": {"QWidget", "QFrame", "QListView", "QScrollBar"},
    "TextEdit": {"QWidget", "QScrollBar", "QTextEdit"},
    # Task 6's settings page scrolls 83 fields through one of these. Measured at
    # FluentQt 1.11.2 with every section expanded: 4 `QWidget`s (the
    # `qt_scrollarea_viewport` plus the binding's own bar hosts) and no `QScrollBar`
    # class of Qt's at all, because the wheel paints its own.
    "ScrollView": {"QWidget"},
}

#: Module-level `fluentqt` names that are functions or enums rather than widget
#: classes. Anything outside `UI_CLASSES` and this set is not a reviewed surface.
#:
#: Listed here is *not* licence to call one: Ruling A keeps the app at zero token
#: overrides, so `set_widget_theme_overrides` may appear in this test but nowhere in
#: `app/**` -- `test_no_theme_override_call_sites_in_the_app` below is what holds that.
UI_MODULE_FUNCTIONS = frozenset({
    "initialize_resources", "prepare_high_dpi_application", "set_theme",
    "set_widget_theme_overrides", "motion_policy", "set_motion_mode",
    "binding_build_info", "Theme", "ButtonStyle",
})


def _is_fluent(widget) -> bool:
    """True when any class in the widget's own ancestry comes from the binding."""
    return any((klass.__module__ or "").startswith(("fluentqt", "_fluentqt"))
               for klass in type(widget).__mro__)


def _binding_name(widget) -> str:
    """The widget's own binding class name, so the app's subclasses are keyed by the
    Fluent class they are (`AppWindow` -> `Window`, `WorkPage` -> `Card`)."""
    for klass in type(widget).__mro__:
        if (klass.__module__ or "").startswith(("fluentqt", "_fluentqt")):
            return klass.__name__
    return type(widget).__name__


def _owner(widget) -> str | None:
    """The nearest Fluent-backed ancestor -- the control that built this Qt part."""
    parent = widget.parent()
    while parent is not None:
        if _is_fluent(parent):
            return _binding_name(parent)
        parent = parent.parent()
    return None


def test_every_widget_in_the_tree_is_fluent_or_a_known_container(qapp):
    app_obj, window = main.build_app([])
    offenders = []
    for widget in window.findChildren(QWidget):
        if _is_fluent(widget) or isinstance(widget, QDialog):
            continue
        if type(widget).__name__ not in FLUENT_INTERNALS.get(_owner(widget) or "", ()):
            offenders.append(f"{_owner(widget)} owns a {type(widget).__name__} "
                             f"({widget.objectName() or '<unnamed>'}) that is not Fluent")
    assert not offenders, "\n".join(offenders)


def test_the_window_content_itself_is_a_fluent_surface(qapp):
    """The table above cannot cover this case: a bare `QWidget` set as the window's
    content has owner `Window`, which legitimately hosts bare `QWidget`s. It is also
    the one bare surface that would paint the whole client area with the native
    palette, so it gets its own assertion (spec §5.3)."""
    app_obj, window = main.build_app([])
    content = window.contentWidget()
    assert content is not None and _is_fluent(content), (
        f"window content is {type(content).__name__}, not a Fluent surface")


def test_the_settings_page_adds_no_bare_surface_of_its_own(qapp):
    """Task 6 puts 83 fields behind a `ScrollView` and five `Expander`s inside a dialog.
    The dialog is opened here rather than assumed: the table above licenses only the
    wheel's own internals, and this is what makes an app-authored bare `QWidget` in the
    scrollable body -- the one place the page could hide one -- fail the commit."""
    app_obj, window = main.build_app([])
    window.controller.work_page.settings_requested.emit()
    dialog = window.controller.settings_dialog
    assert _is_fluent(dialog.page), f"the settings page is a {type(dialog.page).__name__}"
    offenders = []
    for widget in dialog.findChildren(QWidget):
        if _is_fluent(widget) or isinstance(widget, QDialog):
            continue
        if type(widget).__name__ not in FLUENT_INTERNALS.get(_owner(widget) or "", ()):
            offenders.append(f"{_owner(widget)} owns a {type(widget).__name__} "
                             f"({widget.objectName() or '<unnamed>'}) that is not Fluent")
    assert not offenders, "\n".join(offenders)


def test_app_only_touches_the_reviewed_fluent_surface():
    repo_root = Path(__file__).resolve().parents[1]
    offenders = []
    for path in sorted((repo_root / "app").glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                    and node.value.id == "fluentqt"):
                if node.attr not in UI_CLASSES and node.attr not in UI_MODULE_FUNCTIONS:
                    offenders.append(f"{path.name}:{node.lineno} fluentqt.{node.attr}")
    assert not offenders, (
        "app/** depends on a Fluent name nobody reviewed; add it to UI_CLASSES in "
        "tests/test_fluent_binding.py in the same commit (and to the binding's "
        "construct test) if it is really needed:\n" + "\n".join(offenders))


def test_no_theme_override_call_sites_in_the_app():
    """Ruling A, checked instead of promised: native Fluent means `set_theme(...)` and
    nothing else, so the two override verbs stay out of `app/**` entirely."""
    repo_root = Path(__file__).resolve().parents[1]
    banned = ("apply_theme_overrides", "set_widget_theme_overrides")
    offenders = []
    for path in sorted((repo_root / "app").glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Attribute) and node.attr in banned:
                offenders.append(f"{path.name}:{node.lineno} {node.attr}")
    assert not offenders, "token overrides are back in the app:\n" + "\n".join(offenders)
