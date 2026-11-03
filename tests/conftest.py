"""Put the repository root on sys.path.

The engine modules (`separate`, `lib_v5.*`, `uvr_paths`) live at the repo root and
are imported by name from tests, so the root must be importable no matter which
directory pytest is invoked from.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
