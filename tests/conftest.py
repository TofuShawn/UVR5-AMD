"""Put the repository root on sys.path, and make Qt testable without a display.

The engine modules (`separate`, `lib_v5.*`, `uvr_paths`) live at the repo root and
are imported by name from tests, so the root must be importable no matter which
directory pytest is invoked from.

`QT_QPA_PLATFORM=offscreen` is set before anything imports Qt and is a *default*,
not an override: a developer who runs one UI test under the real `windows` platform
may do so, but the suite must not need a display. Offscreen is also the only
environment where the container-theme bug (a plain QWidget surface keeping the
native palette while Fluent children switch to dark tokens) is *visible* -- on this
machine the desktop theme is dark, so a real-session test would hide it. See spec
§5.3.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _sandbox_config_root(tmp_path_factory):
    """Keep the app-building guards off the developer's real user config.

    Item 11 (wave 2). `tests/test_ui_surfaces.py` and `tests/test_work_page.py` build
    the app through `app/main.py` -> `Controller()` -> `settings.load()` /
    `selections.load()`, which read the config file `settings.config_dir()` resolves to:
    `%APPDATA%/UVR-fork/settings.json`, falling back to `~/.config/UVR-fork` when
    `APPDATA` is unset. Those loaders *raise* on invalid JSON, so a hand-edited or stale
    config in the real profile turned the container/legibility guards red for a reason
    that had nothing to do with the widget tree -- an environment-dependent guard, the
    class of flake the plan treats as a defect. Repoint `APPDATA` *and* `Path.home` (the
    fallback root and cache base) at a throwaway session dir so those guards resolve the
    same way on every machine, mirroring the per-test `appdata` fixture at
    `tests/test_settings.py:431-436`. Per-test monkeypatches still win inside their own
    tests (function scope shadows this for the duration, then restores to this baseline).

    Restores both precisely on teardown: `os.environ['APPDATA']` back to its prior value
    or removed, and `Path.home` back to the class's own descriptor (a session run must
    not leak a patched `home` into anything that imports pytest afterwards).
    """
    import os
    from pathlib import Path

    base = tmp_path_factory.mktemp("sandbox-config")
    roaming = base / "AppData" / "Roaming"
    roaming.mkdir(parents=True)
    original_home = Path.__dict__["home"]
    old_appdata = os.environ.get("APPDATA")
    os.environ["APPDATA"] = str(roaming)
    Path.home = classmethod(lambda cls: base)
    try:
        yield base
    finally:
        Path.home = original_home
        if old_appdata is None:
            os.environ.pop("APPDATA", None)
        else:
            os.environ["APPDATA"] = old_appdata


@pytest.fixture(scope="session")
def qapp():
    """One QApplication for the whole session (Qt forbids a second in-process app)."""
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app
    # Destroy surviving top-level windows before Qt tears the QApplication down, so
    # Shiboken's visitAllPyObjects during destroyQCoreApplication has no live widget to
    # crash on (the interpreter-exit segfault a larger widget tree can surface).
    for widget in app.topLevelWidgets():
        widget.deleteLater()
    app.processEvents()
