"""The one window's shell: a labeled left navigation rail and a main content region.

`spec §5.2` measured that Fluent-Qt's `NavigationView` exposes no item/selection API, so
the app cannot use it as an off-the-shelf nav shell. This rebuilds the Wintoys/Fluent
Settings look from Fluent parts: flush-left nav rows (icon + label) with a short accent
bar marking the active one, that swap the main content between the work page and a
full-page settings view *in the one window* -- replacing the old second top-level window
(a non-modal `QDialog`) that fragmented a single-window tool.

The accent bar is custom-painted (a Fluent `Card` subclass) because stock Fluent-Qt has no
nav-item indicator and Ruling A forbids token overrides; it reads the system accent via
`fluentqt.accent_color()`. The settings page is built lazily on first open -- 83 widgets a
user who never opens Settings should not pay for at start-up -- and `reset_settings()`
discards it when a settings value is changed elsewhere.
"""
from __future__ import annotations

import weakref

import fluentqt
from PySide6.QtCore import (QAbstractAnimation, QEasingCurve, QParallelAnimationGroup,
                            QPropertyAnimation, QRectF, Qt, Signal)
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QGraphicsOpacityEffect, QHBoxLayout, QVBoxLayout

from app import settings_page


def apply_fonts(root) -> None:
    """Reach the widgets Fluent themed as "FluentQt UI ..." and give them our bundled
    chain: Latin to Inter, CJK to the subset Noto Sans SC. Fluent sets that family on each
    widget, so an application font never reaches them; this rewrites only that default and
    leaves a deliberate override (the mono data lines) alone."""
    from PySide6.QtWidgets import QWidget
    chain = ["Inter", "Noto Sans SC"]
    for widget in root.findChildren(QWidget):
        font = widget.font()
        if font.family().startswith("FluentQt UI"):
            font.setFamilies(chain)
            widget.setFont(font)


def _motion(ms: int) -> int:
    """A duration that respects the system reduce-motion switch: 0 means "snap"."""
    try:
        return int(fluentqt.motion_policy().resolvedDuration(ms))
    except Exception:                                 # no policy on an odd platform
        return 0


class NavItem(fluentqt.Card):
    """One flush-left navigation row: icon + label, with a short accent bar when active.

    The whole row is clickable; the leading icon is a Fluent `Button` for its glyph but is
    made transparent to the mouse so a click anywhere in the row lands here.
    """

    clicked = Signal()

    def __init__(self, glyph, text: str, parent=None):
        super().__init__(parent)
        self._active = False
        self.setCursor(Qt.PointingHandCursor)
        row = QHBoxLayout(self)
        row.setContentsMargins(20, 6, 12, 6)
        row.setSpacing(12)
        icon = fluentqt.Button("", self)
        icon.setIconGlyph(glyph)
        icon.setFluentStyle(fluentqt.Button.ButtonStyle.Subtle)
        icon.setFixedSize(30, 30)
        icon.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        row.addWidget(icon)
        self.label = fluentqt.Label(text, self)
        self.label.setFluentTypography(fluentqt.FontRole.Body)
        row.addWidget(self.label)
        row.addStretch(1)

    def setActive(self, on: bool) -> None:
        self._active = bool(on)
        self.update()

    def setCollapsed(self, collapsed: bool) -> None:
        self.label.setVisible(not collapsed)
        self.layout().setContentsMargins(13 if collapsed else 20, 6, 12, 6)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mouseReleaseEvent(event)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self._active:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setBrush(fluentqt.accent_color())
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(QRectF(4, self.height() / 2 - 10, 3, 20), 1.5, 1.5)


class AppShell(fluentqt.Card):
    """Main content + left rail; owns the full-page settings view and its Save button."""

    def __init__(self, ctrl, work_page, *, settings_obj, selections_obj, parent=None):
        super().__init__(parent)
        self._ctrl_ref = weakref.ref(ctrl)
        self.work_page = work_page
        self._settings_obj = settings_obj
        self._selections_obj = selections_obj
        self._settings_page = None
        self._settings_save = None
        self.setAppearance(fluentqt.Card.Appearance.Canvas)

        self._content = fluentqt.Card(self)
        self._content.setAppearance(fluentqt.Card.Appearance.Canvas)
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.addWidget(work_page, 1)

        self.rail = fluentqt.Card(self)
        self.rail.setAppearance(fluentqt.Card.Appearance.Layer)
        self.rail.setBorderVisible(True)
        self._collapsed = False
        self._showing_settings = False
        self.rail.setFixedWidth(190)
        rail_col = QVBoxLayout(self.rail)
        rail_col.setContentsMargins(0, 8, 0, 10)
        rail_col.setSpacing(2)
        self.menu_button = fluentqt.Button("", self.rail)
        self.menu_button.setIconGlyph(fluentqt.Icons.GlobalNav)
        self.menu_button.setFluentStyle(fluentqt.Button.ButtonStyle.Subtle)
        self.menu_button.setFixedSize(30, 30)
        self.menu_button.setToolTip("Collapse / expand")
        self.menu_button.clicked.connect(self._toggle_rail)
        menu_row = QHBoxLayout()
        menu_row.setContentsMargins(13, 0, 8, 6)
        menu_row.addWidget(self.menu_button)
        menu_row.addStretch(1)
        rail_col.addLayout(menu_row)
        self.work_button = NavItem(fluentqt.Icons.Home, "Work", self.rail)
        self.settings_button = NavItem(fluentqt.Icons.Settings, "Settings", self.rail)
        self.work_button.clicked.connect(self.show_work)
        self.settings_button.clicked.connect(self._request_settings)
        rail_col.addWidget(self.work_button)
        rail_col.addStretch(1)
        rail_col.addWidget(self.settings_button)

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        row.addWidget(self.rail)
        row.addWidget(self._content, 1)
        self._set_active(work=True)
        apply_fonts(self)

    # -- the rail -------------------------------------------------------------- #

    def _set_active(self, *, work: bool) -> None:
        self.work_button.setActive(work)
        self.settings_button.setActive(not work)

    def _toggle_rail(self) -> None:
        self._collapsed = not self._collapsed
        target = 56 if self._collapsed else 190
        items = (self.work_button, self.settings_button)
        if self._collapsed:                       # hide labels at once when collapsing
            for item in items:
                item.setCollapsed(True)
        duration = _motion(180)
        if duration <= 0:
            self.rail.setFixedWidth(target)
            for item in items:
                item.setCollapsed(self._collapsed)
            return
        group = QParallelAnimationGroup(self)
        for prop in (b"minimumWidth", b"maximumWidth"):
            anim = QPropertyAnimation(self.rail, prop, group)
            anim.setDuration(duration)
            anim.setStartValue(self.rail.width())
            anim.setEndValue(target)
            anim.setEasingCurve(QEasingCurve.OutCubic)
            group.addAnimation(anim)
        def _done() -> None:
            self.rail.setFixedWidth(target)
            for item in items:                    # labels appear once expanded
                item.setCollapsed(self._collapsed)
        group.finished.connect(_done)
        group.start(QAbstractAnimation.DeleteWhenStopped)

    def _fade_in(self, widget) -> None:
        duration = _motion(160)
        if duration <= 0:
            return
        effect = QGraphicsOpacityEffect(widget)
        widget.setGraphicsEffect(effect)
        anim = QPropertyAnimation(effect, b"opacity", widget)
        anim.setDuration(duration)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        anim.finished.connect(lambda: widget.setGraphicsEffect(None))
        anim.start(QAbstractAnimation.DeleteWhenStopped)

    def _request_settings(self) -> None:
        self.work_page.settings_requested.emit()

    # -- page switching -------------------------------------------------------- #

    @property
    def settings_open(self) -> bool:
        return self._showing_settings

    def has_settings(self) -> bool:
        return self._settings_page is not None

    def show_work(self) -> None:
        if self._settings_page is not None:
            self._settings_page.setVisible(False)
        if self._settings_save is not None:
            self._settings_save.setVisible(False)
        self.work_page.setVisible(True)
        self._set_active(work=True)
        self._showing_settings = False

    def show_settings(self) -> None:
        self._ensure_settings()
        self.work_page.setVisible(False)
        self._settings_page.setVisible(True)
        self._settings_save.setVisible(True)
        self._set_active(work=False)
        self._showing_settings = True

    def _ensure_settings(self) -> None:
        if self._settings_page is not None:
            return
        # Read the *current* objects from the controller, not the pair captured at
        # construction: a value changed on the work page replaces `ctrl.settings`, and a
        # rebuilt page must show the new truth rather than the pre-change draft.
        ctrl = self._ctrl_ref()
        settings_obj = ctrl.settings if ctrl is not None else self._settings_obj
        selections_obj = ctrl.selections if ctrl is not None else self._selections_obj
        self._settings_page = settings_page.SettingsPage(
            settings_obj=settings_obj, selections_obj=selections_obj)
        self._settings_save = fluentqt.Button("Save settings", self._content)
        self._settings_save.setIconGlyph(fluentqt.Icons.Save)
        self._settings_save.setFluentLayout(fluentqt.Button.ButtonLayout.IconBefore)
        self._settings_save.clicked.connect(self._save)
        self._content_layout.addWidget(self._settings_page, 1)
        self._content_layout.addWidget(self._settings_save)
        apply_fonts(self._settings_page)

    def _save(self) -> None:
        ctrl = self._ctrl_ref()
        if ctrl is None or self._settings_page is None:
            return
        if not ctrl.save_settings(self._settings_page):
            self.show_work()

    def reset_settings(self) -> None:
        """Discard the settings view so the next open rebuilds from the current values."""
        if self._settings_page is not None:
            self._settings_page.setParent(None)
            self._settings_page.deleteLater()
            self._settings_page = None
        if self._settings_save is not None:
            self._settings_save.setParent(None)
            self._settings_save.deleteLater()
            self._settings_save = None
        self.show_work()
