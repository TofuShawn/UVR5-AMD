"""Build the distributable source zip: repo runtime + bootstrap + install notes.

The ROCm torch stack is ~2-3 GB and freezes badly (PyInstaller/Nuitka miss torch/HIP
dlls), so the fork ships *source* plus `bootstrap.ps1`: the user runs one bootstrap to
fetch the backend at install time, and the app itself stays network-free. This script
assembles exactly the tracked runtime tree (no .venv / .git / docs / tests / dev tools),
adds a `START_HERE.txt`, and zips it under `dist/`.

Run from the repo root:  python tools/make_dist.py

No launcher is shipped. `bootstrap.ps1` writes `start.bat` itself, and only after its
install checks pass -- a zip whose bootstrap failed must not leave a launcher pointing
at a half-built venv. Both `run.bat` (this machine's convenience double-click) and the
generated `start.bat` are gitignored, so neither can ride along in the tracked tree.
"""
from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: Tracked prefixes that are development-only and never ship.
_EXCLUDE_PREFIX = ("tests/", "tools/", "docs/", ".github/", ".superpowers/")
#: Tracked top-level files that are not part of the PySide6 runtime.
_EXCLUDE_TOP = {"UVR.py", "install_packages.sh", ".gitignore"}

_START_HERE = """\
UVR5 PySide6 GUI (AMD/ROCm fork) -- source distribution
======================================================
1. Install Python 3.12 (the ROCm wheels are cp312-only).
2. Fetch the backend once (network at install time; the app itself never networks):
     powershell -NoProfile -ExecutionPolicy Bypass -File bootstrap.ps1 -Backend rocm
   Use -Backend cpu on a machine with no AMD GPU.
   bootstrap.ps1 writes start.bat as its last step, and only if the install passed.
3. Double-click start.bat. After the app exits it asks whether to delete itself.

Weights: start.bat defaults UVR_MODEL_ROOT to the bundled models/ (tables + one VR
weight). To use your own installed weights, set UVR_MODEL_ROOT before launching, or
drop weight files under models/<Arch>_Models/. Only models whose weights are present
are offered in the Model list.
"""


def _tracked() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=REPO, check=True,
                         capture_output=True, text=True)
    return [p for p in out.stdout.split("\0") if p]


def _version() -> str:
    ns: dict = {}
    exec((REPO / "__version__.py").read_text(encoding="utf-8"), ns)
    return ns["VERSION"]


def main() -> Path:
    dist = REPO / "dist"
    dist.mkdir(exist_ok=True)
    target = dist / f"UVR5-PySide6-GUI-{_version()}-win-src.zip"
    members = [p for p in _tracked()
               if not p.startswith(_EXCLUDE_PREFIX) and p not in _EXCLUDE_TOP]
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in sorted(members):
            zf.write(REPO / rel, rel)
        zf.writestr("START_HERE.txt", _START_HERE)
    print(f"wrote {target} ({target.stat().st_size / 1e6:.1f} MB, "
          f"{len(members) + 1} entries)")
    return target


if __name__ == "__main__":
    main()