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

import catalog
import fluentqt
import separate_cli
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
    "ComboBox": {"QWidget", "QFrame", "QListView", "QScrollBar"},
    "TextEdit": {"QWidget", "QScrollBar", "QTextEdit"},
}

#: The `ScrollView` Task 6 scrolls the settings body through and the `Expander` it puts
#: each section in need a **name**-keyed licence, not a class-keyed one. Their internals
#: are `QWidget`s -- measured offscreen on the built dialog, every section expanded:
#: `qt_scrollarea_viewport`, `qt_scrollarea_hcontainer`, `qt_scrollarea_vcontainer` (Qt's
#: own scroll-area parts) and `FluentScrollViewTransparentCorner`, the binding's corner
#: host, and for the Expander the single clip widget `fluentExpanderClip` (no `QScrollBar`
#: class of Qt's at all, because the wheel paints its own) -- and `QWidget` is exactly the
#: class this guard exists to refuse. Licensing the *class* would license the mistake the
#: docstring above says it catches: `SettingsPage._body` sits at body ->
#: viewport(`QWidget`) -> `ScrollView`, and a section's content host sits at
#: content -> `fluentExpanderClip`(QWidget) -> `Expander`, so swapping the `Card` for a
#: bare `QWidget` reads as "the container owns a QWidget" and the suite stays green.
#: Naming each host means an app-authored widget -- which is never named `qt_scrollarea_*`
#: or `fluentExpanderClip` -- cannot pass for one.
NAMED_INTERNALS: dict[str, frozenset[str]] = {
    "ScrollView": frozenset({
        "qt_scrollarea_viewport",
        "qt_scrollarea_hcontainer",
        "qt_scrollarea_vcontainer",
        "FluentScrollViewTransparentCorner",
    }),
    "Expander": frozenset({"fluentExpanderClip"}),
}


def _is_licensed_internal(widget) -> bool:
    """True when the wheel, not the app, built this Qt part inside the control that owns
    it. A `ScrollView` and an `Expander` only license their named hosts; a
    `ContentDialog` licenses the one panel it builds for its button bar, which is the
    same class the app's content host is -- so that one is told apart by `content()`,
    the dialog's own verb for the widget the caller handed it. Every other owner is
    licensed by class, as measured."""
    owner_widget = _owner_widget(widget)
    if owner_widget is None:
        return False
    owner = _binding_name(owner_widget)
    klass = type(widget).__name__
    if klass in FLUENT_INTERNALS.get(owner, ()):
        return True
    if klass == "QWidget" and widget.objectName() in NAMED_INTERNALS.get(owner, ()):
        return True
    return (owner == "ContentDialog" and klass == "QWidget"
            and widget.parent() is owner_widget
            and widget is not owner_widget.content())


def _offenders(widgets) -> list[str]:
    """The bare surfaces in this subtree that neither the Fluent ancestry, the dialog
    allowance, nor the wheel's own measured internals explain."""
    return [f"{_owner(w)} owns a {type(w).__name__} "
            f"({w.objectName() or '<unnamed>'}) that is not Fluent"
            for w in widgets
            if not (_is_fluent(w) or isinstance(w, QDialog) or _is_licensed_internal(w))]


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
    # Wave 2 item 0: `app/main.py` reads the machine's animation switch and publishes
    # it through `set_motion_mode(fluentqt.MotionMode.Full|Disabled)` before any widget
    # exists (spec §5.3 reduce-motion). `MotionMode` is the enum both verbs speak; the
    # honest fix is to review it here, not to drop the wiring or cast the enum to an int.
    "MotionMode",
})

#: Qt base classes the app may never *construct* itself. `QWidget(` keeps the native
#: palette (spec §5.3), and a raw `QDialog(`/`QFrame(` is the same mistake one level up:
#: `ContentDialog` is the binding's dialog, so building a bare `QDialog` bypasses the
#: census entirely -- the sibling hole the `ContentDialog` licence in `_is_licensed_
#: internal` only half-closes. The app reaches Qt layouts (`QVBoxLayout`) and the
#: application object, but never instantiates a bare widget; measured 0 constructions
#: across `app/**` at HEAD, so this refusal is a ratchet, not a re-litigation.
BARE_QT_CONSTRUCTIBLES = frozenset({"QWidget", "QDialog", "QFrame"})

#: GUI imports spec §4 calls an error anywhere in `app/**`. The tkinter layer is retired
#: (Task 8) and the app draws through `fluentqt`, so a re-added `import tkinter` /
#: `sv_ttk` / `tkinterdnd2` / `PIL.ImageTk` here is a GUI coupling on the front-end side
#: the branch promised to leave. (`PySide6` is deliberately NOT listed: the app imports
#: it for the application object, layouts and QSettings -- the reviewed-surface rule for
#: Qt widgets is `UI_CLASSES` + `BARE_QT_CONSTRUCTIBLES`, not a blanket PySide6 ban.)
RETIRED_GUI_IMPORTS = ("tkinter", "gui_data.sv_ttk", "gui_data.tkinterdnd2",
                       "PIL.ImageTk")

#: The UI_CLASSES names the app does NOT instantiate, kept on the list purely for the
#: binding tripwire (`test_fluent_binding` constructs every declared class, so a wheel
#: that deletes one fails *there*, by name). Wave 2 minor: `UI_CLASSES` was policed
#: forward only -- app->list -- so 16 of 32 names went unchecked for whether the app
#: uses them, and a decorative class drifting out would redden the suite for no
#: app-facing reason. This is the explicit exemption the reverse assertion below
#: requires: "unused-but-reviewed", stated, so a name leaving it (now used) or a name
#: used but absent from it must be an edit in the same commit.
DECORATIVE_UI_CLASSES = frozenset({
    "AutoSuggestBox", "Divider", "FluentWidget", "HyperlinkButton",
    "MultiSelectComboBox", "PasswordBox", "ProgressRing", "RadioButton",
    "SelectorBar", "SelectorBarItem", "Slider", "TabView", "TabViewItem",
    "ToggleButton", "ToggleSwitch", "ToolTip",
    # `CheckBox` left this set in Plan 3 Task 1, where `app/work_page.py` started using it
    # for the default-output-folder toggle -- and it left it for the reason this list
    # exists: the binding's `ToggleSwitch` has `setIsOn`/`toggled` and *no* `setText`,
    # `setChecked` or `isChecked` (measured on the pinned wheel), so a switch cannot be
    # initialised from a stored value, which is precisely what that toggle has to do.
})


def _app_py_files(repo_root):
    """Every `app/**/*.py`, recursively. The AST scans used to glob `app/*.py` while
    their docstrings said `app/**`; `app` is flat today so the two agreed, but the
    one-word fix (`rglob`) makes the scan match the promise the moment a subpackage
    lands."""
    return sorted((repo_root / "app").rglob("*.py"))


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
    """The nearest Fluent-backed ancestor's binding name -- the control that built this
    Qt part."""
    owner = _owner_widget(widget)
    return None if owner is None else _binding_name(owner)


def _owner_widget(widget):
    """The nearest Fluent-backed ancestor itself, so a licence can ask that widget what
    it made. `ContentDialog.content()` is the question the name-only version cannot
    ask -- which is the difference between licensing a bare content host and refusing
    one."""
    parent = widget.parent()
    while parent is not None:
        if _is_fluent(parent):
            return parent
        parent = parent.parent()
    return None


def test_every_widget_in_the_tree_is_fluent_or_a_known_container(qapp):
    app_obj, window = main.build_app([])
    offenders = _offenders(window.findChildren(QWidget))
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
    scrollable body -- the one place the page could hide one -- fail the commit.

    The two direct assertions are the part the licence cannot do. `ScrollView` hosts its
    content in `qt_scrollarea_viewport`, a `QWidget`, so the body's nearest Fluent
    ancestor is the `ScrollView` itself -- which means "the body is a bare QWidget"
    arrives here as "ScrollView owns a QWidget" and would be licensed by class if the
    licence were by class. It is by name (see `NAMED_INTERNALS`), and these two
    lines ask about the body directly instead of through the owner chain: the widget the
    page put in the scroll area, and whatever the scroll area reports as its content,
    must both be Fluent surfaces. Measured with a bare `QWidget` substituted for the
    `Card`: both fail, and the census below names it too.
    """
    app_obj, window = main.build_app([])
    window.controller.work_page.settings_requested.emit()
    dialog = window.controller.settings_dialog
    page = dialog.page
    assert _is_fluent(page), f"the settings page is a {type(page).__name__}"
    assert _is_fluent(page._body), (
        f"the scrollable body of the settings page is a {type(page._body).__name__}, "
        f"not a Fluent surface -- it is the largest painted rectangle on the page")
    content = page._scroller.contentWidget()
    assert _is_fluent(content), (
        f"the ScrollView's content widget is a {type(content).__name__}, so the page's "
        f"body is not the Card `setContentWidget` was given")
    offenders = _offenders(dialog.findChildren(QWidget))
    assert not offenders, "\n".join(offenders)


def test_app_only_touches_the_reviewed_fluent_surface():
    repo_root = Path(__file__).resolve().parents[1]
    offenders = []
    referenced = set()
    for path in _app_py_files(repo_root):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                    and node.value.id == "fluentqt"):
                referenced.add(node.attr)
                if node.attr not in UI_CLASSES and node.attr not in UI_MODULE_FUNCTIONS:
                    offenders.append(f"{path.name}:{node.lineno} fluentqt.{node.attr}")
    assert not offenders, (
        "app/** depends on a Fluent name nobody reviewed; add it to UI_CLASSES in "
        "tests/test_fluent_binding.py in the same commit (and to the binding's "
        "construct test) if it is really needed:\n" + "\n".join(offenders))

    # The reverse direction (wave 2 minor). Forward, UI_CLASSES means "every Fluent class
    # app may depend on"; reverse, it means the same only if the 16 names app does NOT
    # depend on are *deliberately* kept for the binding tripwire. Without this half, a
    # decorative class could be quietly trimmed (or an upstream deletion would look like
    # an app bug), and the list would stop meaning "reviewed surface". `referenced` is
    # the fluentqt names app actually touches; intersect with UI_CLASSES for the used set.
    used_in_app = referenced & set(UI_CLASSES)
    not_used = set(UI_CLASSES) - used_in_app
    assert not_used == set(DECORATIVE_UI_CLASSES), (
        "UI_CLASSES has drifted from `used-in-app ∪ decorative` -- the two-way contract "
        "this guard enforces. Names app now uses but the decorative list still claims "
        f"unused: {sorted(not_used & set(DECORATIVE_UI_CLASSES) - not_used)}; names app "
        f"does NOT use but the decorative list forgot to exempt: "
        f"{sorted(not_used - set(DECORATIVE_UI_CLASSES))}; decorative names that app now "
        f"uses (drop them from the exemption): "
        f"{sorted(set(DECORATIVE_UI_CLASSES) & used_in_app)}. Update DECORATIVE_UI_CLASSES "
        "in the same commit as the app usage and say why.")


def test_the_app_never_constructs_a_bare_qt_widget():
    """The compile-time half of spec §5.3: the census refuses a bare surface in the
    *built* tree; this refuses a bare `QWidget(`/`QDialog(`/`QFrame(` in the *source*,
    which also closes the `ContentDialog` sibling the runtime licence cannot see (a raw
    QDialog is never walked by `findChildren` on a Fluent owner). Layouts and the
    application object are not widgets and stay legal."""
    repo_root = Path(__file__).resolve().parents[1]
    offenders = []
    for path in _app_py_files(repo_root):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id in BARE_QT_CONSTRUCTIBLES):
                offenders.append(f"{path.name}:{node.lineno} {node.func.id}(...)")
    assert not offenders, (
        "app/** constructs a bare Qt widget; use the fluentqt control "
        "(ContentDialog, not QDialog; Card, not QWidget):\n" + "\n".join(offenders))


def test_the_app_never_imports_the_retired_gui_stack():
    """Spec §4: the tkinter GUI is retired, so importing it from the new front end is an
    error, not a fallback. `import tkinter` / `sv_ttk` / `tkinterdnd2` / `PIL.ImageTk`
    anywhere in `app/**` (at any depth, since the scan is over the whole AST) means the
    retired stack is being reached for again. Measured 0 today; this is the ratchet."""
    repo_root = Path(__file__).resolve().parents[1]
    offenders = []
    for path in _app_py_files(repo_root):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                names = [node.module]
            for nm in names:
                if any(nm == pkg or nm.startswith(pkg + ".")
                       for pkg in RETIRED_GUI_IMPORTS):
                    offenders.append(f"{path.name}:{node.lineno} import {nm}")
    assert not offenders, (
        "app/** imports the retired tkinter GUI stack (spec §4 says this is an error):\n"
        + "\n".join(offenders))


def test_no_theme_override_call_sites_in_the_app():
    """Ruling A, checked instead of promised: native Fluent means `set_theme(...)` and
    nothing else, so the two override verbs stay out of `app/**` entirely."""
    repo_root = Path(__file__).resolve().parents[1]
    banned = ("apply_theme_overrides", "set_widget_theme_overrides")
    offenders = []
    for path in _app_py_files(repo_root):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Attribute) and node.attr in banned:
                offenders.append(f"{path.name}:{node.lineno} {node.attr}")
    assert not offenders, "token overrides are back in the app:\n" + "\n".join(offenders)


def test_the_params_dialog_adds_no_bare_surface_of_its_own(qapp, tmp_path):
    """Task 7 puts a form inside a `ContentDialog`, and the dialog builds exactly one
    `QWidget` of its own -- the panel that holds its button bar (measured offscreen at
    1.11.2: one direct child, unnamed, three Fluent `Button`s inside it). `QWidget` is the
    class this guard exists to refuse, and the app's own content host is the *same class
    with the same parent*, so a class-keyed licence would swallow the mistake Task 6's
    carry names: a bare `QWidget` passed to `setContent` in place of the `Card` arrives at
    the census as "ContentDialog owns a QWidget" and the suite stays green.

    So the licence asks the dialog a question a class cannot answer -- `content()` is
    Fluent's own verb for the widget the caller handed it, and it is the second half of
    what this test asserts: the app's body must be a Fluent surface *and* a bare widget
    put in the same place must be an offender.
    """
    from app import dialogs

    cat = separate_cli.make_catalog(tmp_path / "nope", tmp_path / "w")
    spec = catalog.ModelSpec(name="mystery", arch="vr", process_method="VR Architecture",
                             primary_stem="Vocals", secondary_stem="Instrumental",
                             weights_path=None, config_path=None, hash="deadbeef",
                             model_data=None)
    dlg = dialogs.model_params_dialog(spec, cat, offer=False)
    body = dlg.content()
    assert body is not None and _is_fluent(body), (
        f"the form's body is a {type(body).__name__} -- it is the largest painted "
         "rectangle inside the dialog")
    assert _binding_name(body) == "Card", _binding_name(body)
    offenders = _offenders(dlg.findChildren(QWidget))
    assert not offenders, "\n".join(offenders)

    naked = fluentqt.ContentDialog()
    host = QWidget()
    naked.setContent(host)
    assert not _is_licensed_internal(host), (
        "a bare content host is licensed as the dialog's own panel: the licence stopped "
        "asking `content()`")
    assert _offenders(naked.findChildren(QWidget)), (
        "the census let a bare QWidget through as a content host")


def test_an_app_widget_parented_into_an_expander_is_not_its_clip(qapp):
    """Task 6's carry, landed: the `Expander` licence used to be `{"QWidget"}` by class,
    which is the same hole the `ScrollView` entry had -- a section's `Card` content host
    swapped for a bare `QWidget` still reads as "Expander owns a QWidget", because
    `_owner()` reports the nearest Fluent ancestor and not who built the widget. The
    binding's own part is named (`fluentExpanderClip`, measured); an app widget is not.
    """
    expander = fluentqt.Expander()
    clip = expander.findChild(QWidget, "fluentExpanderClip")
    assert clip is not None and type(clip).__name__ == "QWidget", (
        "the Expander's internals moved -- re-measure the licence")
    assert _is_licensed_internal(clip)

    bare = QWidget(expander)
    assert not _is_licensed_internal(bare)
    assert _offenders(expander.findChildren(QWidget)) == [
        "Expander owns a QWidget (<unnamed>) that is not Fluent"]
