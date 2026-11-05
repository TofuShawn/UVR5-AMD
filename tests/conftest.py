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


@pytest.fixture(scope="session")
def qapp():
    """One QApplication for the whole session (Qt forbids a second in-process app)."""
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app
