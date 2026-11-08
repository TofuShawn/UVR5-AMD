"""Launch the app: QApplication, theme, window, work page.

Run it as `python -m app.main` from the repository root -- the package form, because the
engine surface this app calls (`backends`, `catalog`, `separate_cli`, `plan`) lives at
the root and `-m` puts the working directory on `sys.path`. `python app/main.py` does
not: it puts `app/` there instead, and this module used to patch that with a `sys.path`
insert. Plan 2 Task 8 standardised on the package form and deleted the patch, so the
launch command and the import path now agree (R-T8-launch).

Theme follows the system (spec §5.3, ruling A: zero token overrides). Reading it is a
registry lookup on Windows and a palette guess elsewhere; `set_theme` is the only
call, and no `apply_theme_overrides` happens anywhere in this app. The registry value
answers Light or Dark, and a contrast theme is neither -- `high_contrast_enabled()` is
read first, because re-labelling one of Windows' own accessibility states as a preference
the user did not pick is the same class of untruth as the motion half below.

Motion follows the system too, for the same reason and by the same route -- spec §5.3 keeps
reduce-motion as a *retained* demand, and until now nothing in `app/**` called
`fluentqt.set_motion_mode`, so the wheel resolved its default `Full` and the GUI animated
on a machine whose user had turned animation effects off (measured:
`motion_policy().resolvedDuration(250) == 250`). `animations_enabled()` reads the same
Windows switch the theme comes from, and `build_app` publishes the answer before any
widget exists, because a policy that arrives after the first hover is a policy the first
hover did not see.

Two things about that mapping are rulings, not oversights, and both are human-verified the
way §9.6 says Mica is:

* the OS flag is binary, so `Reduced` has no producer -- the choice is `Full` when
  animations are on and `Disabled` when they are not, because "don't move things" is not a
  request for *faster* movement (`resolvedDuration(250)` answers 250 and 0 respectively;
  `Reduced`'s 50 still moves).
* nothing here re-reads the flag after launch. Windows broadcasts
  `WM_SETTINGCHANGE`, and acting on it needs an event filter over the window; the setting
  is read at start-up, so a mid-session change needs a restart. Verified by hand, not by
  the suite.

One thing this wiring owes the suite, and it belongs to the wave that owns the guard files:
`tests/test_ui_surfaces.py`'s `UI_MODULE_FUNCTIONS` already reviews `set_motion_mode` and
`motion_policy` but not the enum they speak (`MotionMode`), so
`test_app_only_touches_the_reviewed_fluent_surface` fails on `main.py`'s three
`fluentqt.MotionMode` attributes until `"MotionMode"` joins that frozenset. The app half is
done; the list is a reviewed-surface decision this wave must not make for wave 2.
"""
from __future__ import annotations

import ctypes
import sys

import fluentqt
from PySide6.QtCore import QSettings
from PySide6.QtGui import QFont, QFontDatabase, QPalette
from PySide6.QtWidgets import QApplication

#: `SystemParametersInfoW` verbs this file needs. Both take a struct by reference and
#: both want its size in `uiParam`: measured, `SPI_GETANIMATION` with `uiParam=0` returns
#: rc 0 and leaves `iMinAnimate` at 0, i.e. it *fails in the direction that looks like an
#: answer* -- a machine with its animations on would be told to stop animating. The size
#: is passed, and a zero return is reported as "no answer" rather than as False.
SPI_GETANIMATION = 0x0048
SPI_GETHIGHCONTRAST = 0x0042
HCF_HIGHCONTRASTON = 0x00000001


class _ANIMATIONINFO(ctypes.Structure):
    _fields_ = [("cbSize", ctypes.c_uint), ("iMinAnimate", ctypes.c_int)]


class _HIGHCONTRASTW(ctypes.Structure):
    _fields_ = [("cbSize", ctypes.c_uint), ("dwFlags", ctypes.c_ulong),
                ("lpszDefaultScheme", ctypes.c_wchar_p)]


def _system_parameters_info(verb: int, struct) -> bool:
    """One `SystemParametersInfoW` get, filled in place. False when the call refuses."""
    struct.cbSize = ctypes.sizeof(struct)
    return bool(ctypes.windll.user32.SystemParametersInfoW(       # type: ignore[attr-defined]
        verb, struct.cbSize, ctypes.byref(struct), 0))


def animations_enabled() -> "bool | None":
    """Windows' Settings > Accessibility > Visual effects > Animation effects switch.

    `None` on a platform that has no such call, or when the call itself refuses -- and
    `None` is *not* folded into False, because False means "the user asked for no motion"
    and this means "the machine was not asked".
    """
    if sys.platform != "win32":
        return None
    info = _ANIMATIONINFO()
    if not _system_parameters_info(SPI_GETANIMATION, info):
        return None
    return bool(info.iMinAnimate)


def high_contrast_enabled() -> bool:
    """Windows' contrast-theme flag, and False is the answer for every case that is not
    "the flag is set" -- the same `None`-as-False collapse `animations_enabled()` refuses,
    because a theme is a preference this app already has to pick something about."""
    if sys.platform != "win32":
        return False
    hc = _HIGHCONTRASTW()
    if not _system_parameters_info(SPI_GETHIGHCONTRAST, hc):
        return False
    return bool(hc.dwFlags & HCF_HIGHCONTRASTON)


def system_motion_mode() -> "fluentqt.MotionMode":
    """The wheel's mode for the OS's animation switch (`app/main.py`'s docstring §I10)."""
    enabled = animations_enabled()
    if enabled is None:
        return fluentqt.MotionMode.Full          # nothing was asked of us: the default
    return fluentqt.MotionMode.Full if enabled else fluentqt.MotionMode.Disabled


def system_theme() -> "fluentqt.Theme":
    if high_contrast_enabled():
        # Spec §5.3 keeps `HighContrast`, and the wheel has the mode (measured:
        # `Theme.HighContrast` is a member and `set_theme` accepts it) -- what this app
        # used to map was only Light/Dark, so a contrast theme was silently re-labelled
        # one of the two the user had just opted out of. It outranks the light/dark
        # registry value rather than combining with it.
        return fluentqt.Theme.HighContrast
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


def _load_fonts(app) -> None:
    """Load the bundled UI fonts: Inter (UI), JetBrains Mono (data) and a subset Noto
    Sans SC (CJK). A missing file is tolerated -- the system font is the fallback."""
    from pathlib import Path
    fonts = Path(__file__).resolve().parent / "assets" / "fonts"
    for name in ("Inter.ttf", "NotoSansSC.ttf", "JetBrainsMono.ttf"):
        path = fonts / name
        if path.is_file():
            QFontDatabase.addApplicationFont(str(path))
    base = QFont("Inter")
    base.setFamilies(["Inter", "Noto Sans SC"])
    app.setFont(base)


def build_app(argv=None):
    # `instance() or ...` is not caution: Task 1's session-scoped `qapp` fixture already
    # owns the one QApplication this process may have, and Task 5's surface test asks for
    # both. A second QApplication is a Qt error regardless of what Shiboken tolerates.
    app = QApplication.instance() or QApplication(argv if argv is not None else sys.argv[:1])
    fluentqt.prepare_high_dpi_application()
    fluentqt.initialize_resources()
    _load_fonts(app)
    # Both system reads happen before anything is built: `set_theme` after the first
    # widget paints leaves it painted in the other theme, and `set_motion_mode` after the
    # first hover leaves that hover animated -- the wheel resolves the duration per call,
    # so the policy has to be in place before the first one is asked for.
    fluentqt.set_theme(system_theme())
    fluentqt.set_motion_mode(system_motion_mode())
    from app import controller

    ctrl = controller.Controller()
    window = AppWindow(ctrl)
    window.setWindowTitle("UVR5 PySide6 GUI")
    # The custom Fluent TitleBar does not render `setWindowTitle` (measured): the OS title
    # is set for the taskbar, but the drawn top bar stays blank. Give the TitleBar a
    # content widget so the app's name is visible where a user looks first.
    _title_bar = window.titleBar()
    _title = fluentqt.Label("UVR5 PySide6 GUI", _title_bar)
    _title.setFluentTypography(fluentqt.FontRole.Caption)
    _title_bar.setContentWidget(_title)
    window.setContentWidget(ctrl.shell)   # `setContentWidget` on Window is real;
    window.resize(1280, 980)                     # `Window.setContent` is not (measured).
    # 900x960, not the 760x900 Plan 2 shipped with nor the 900x1100 the first two-column cut
    # asked for: the work page is two cards side by side over a full-width run band and
    # console now, and the extra width is what lets the left card hold two path rows plus
    # the queue without stretching every control to the full window. The height is the
    # page's own fullest sizeHint (944 with a full console and the parameters open,
    # measured) plus a little margin -- and 960 still fits a 1080p screen's usable area,
    # which 1100 did not. There is no scroll body: two pages each holding a ScrollView
    # segfault at interpreter exit (see `work_page._build`), so the page is laid out to fit
    # instead, and `test_no_scroll_view_is_built_by_the_page` pins the build-state hint
    # under 900 so a future control that outgrows the window is caught here.
    return app, window


def main() -> int:
    # Resolve the UI language before any widget is built: a label constructed in one
    # language and repainted in another is a half-translated page. `build_app` itself does
    # not set the locale, so tests that build pages directly stay on the English source.
    from app import i18n
    choice = i18n.load_language()
    i18n.set_locale(i18n.resolve_system() if choice == "system" else choice)
    app, window = build_app()
    window.show()
    # Fluent sets its own per-widget family when a widget is realised, so re-apply the
    # bundled chain once the event loop is running -- labels keep Inter for Latin and the
    # Noto subset for CJK, and the LineEdits do not fall back to Fluent's own family.
    from PySide6.QtCore import QTimer
    from app.shell import apply_fonts
    QTimer.singleShot(0, lambda: apply_fonts(window))
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
