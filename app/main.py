"""Launch the app: QApplication, theme, window, work page.

Theme follows the system (spec §5.3, ruling A: zero token overrides). Reading it is a
registry lookup on Windows and a palette guess elsewhere; `set_theme` is the only
call, and no `apply_theme_overrides` happens anywhere in this app.
"""
from __future__ import annotations

import sys
from pathlib import Path

import fluentqt
from PySide6.QtCore import QSettings
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QApplication


def system_theme() -> "fluentqt.Theme":
    prefs = QSettings("HKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\CurrentVersion"
                      "\\Themes\\Personalize", QSettings.NativeFormat)
    value = prefs.value("AppsUseLightTheme")
    if value is None:                                  # non-Windows or unreadable
        return fluentqt.Theme.Dark if (
            QApplication.palette().color(QPalette.Window).lightness() < 128) else fluentqt.Theme.Light
    return fluentqt.Theme.Light if bool(int(value)) else fluentqt.Theme.Dark


class AppWindow(fluentqt.Window):
    """The window, plus the one thing a close has to decide before the widgets go.

    Ruling R-T4-shutdown -- `closeEvent` (not an app-level `aboutToQuit`) is the chosen
    hook here, because an about-to-quit handler cannot refuse the close, and a `Job` whose
    worker thread is still running must not be destroyed: `QThread`'s destructor
    hard-aborts the process, and the `Job`'s thread is a child of the `Job`. So this
    handler asks the controller to drain the job with a bounded wait and lets the window
    go only when that returns True. On False it ignores the close and the page's console
    says what is still happening -- see `Controller.request_close`.
    """

    def __init__(self, ctrl, *, drain_ms: int = 5000):
        super().__init__()
        self.controller = ctrl
        self.drain_ms = drain_ms

    def closeEvent(self, event) -> None:
        if self.controller.request_close(self.drain_ms):
            super().closeEvent(event)
            return
        event.ignore()


def build_app(argv=None):
    # `instance() or ...` is not caution: Task 1's session-scoped `qapp` fixture already
    # owns the one QApplication this process may have, and Task 5's surface test asks for
    # both. A second QApplication is a Qt error regardless of what Shiboken tolerates.
    app = QApplication.instance() or QApplication(argv if argv is not None else sys.argv[:1])
    fluentqt.prepare_high_dpi_application()
    fluentqt.initialize_resources()
    fluentqt.set_theme(system_theme())
    from app import controller

    ctrl = controller.Controller()
    window = AppWindow(ctrl)
    window.setWindowTitle("UVR5 AMD/ROCm")
    window.setContentWidget(ctrl.work_page)   # `setContentWidget` on Window is real;
    window.resize(760, 900)                    # `Window.setContent` is not (measured).
    return app, window


def main() -> int:
    app, window = build_app()
    window.show()
    return app.exec()


if __name__ == "__main__":
    # `python app/main.py` puts `app/` on sys.path, not the repository root, and the
    # engine surface this app calls (`backends`, `catalog`, `separate_cli`, `plan`) lives
    # at the root. The launch step in the plan is that exact command, so the script path
    # is made to match the package path rather than asking users for `-m app.main`.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    raise SystemExit(main())
