"""`start.bat` is written by the installer, never shipped, and can remove itself.

The design being pinned: a zip whose bootstrap failed must not leave a launcher that
double-clicks into a half-built venv, so the write is the last step of `bootstrap.ps1`
and comes after every check that can throw. The file asks Y/N before deleting itself,
because a batch cannot delete the file cmd is still reading -- `(goto)` drops that
batch context first, and the tests below require both halves to stay.

Static reads, like `tests/test_no_self_update.py`: PowerShell has no importer here, and
what needs guarding is the *order* in the file, which no runtime probe would show.
"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BOOTSTRAP = (REPO / "bootstrap.ps1").read_text(encoding="utf-8")
MAKE_DIST = (REPO / "tools" / "make_dist.py").read_text(encoding="utf-8")


def test_the_launcher_is_written_after_every_check_that_can_throw():
    """Position, not presence: a `start.bat` written early survives a failed install."""
    probe = BOOTSTRAP.index("torch import/probe failed")
    rocm = BOOTSTRAP.index('if ($Backend -eq "rocm")')
    write = BOOTSTRAP.index('Join-Path $PSScriptRoot "start.bat"')
    done = BOOTSTRAP.index('Write-Host "done:')
    assert probe < rocm < write < done, (probe, rocm, write, done)


def test_the_launcher_runs_the_app_and_offers_to_delete_itself():
    for needle in ('".venv\\Scripts\\python.exe" -m app.main',
                   '.venv is missing',
                   'choice /c YN',
                   'del "%~f0"',
                   '(goto) 2>nul'):
        assert needle in BOOTSTRAP, needle
    # ASCII, because a BOM would print before `@echo off` takes effect.
    assert "-Encoding ASCII" in BOOTSTRAP


def test_the_launcher_is_gitignored_and_never_packed():
    """Two-sided: generated (above) and *not* shipped (here)."""
    assert "/start.bat" in (REPO / ".gitignore").read_text(encoding="utf-8")
    assert "_LAUNCHER" not in MAKE_DIST, "make_dist still emits its own launcher"
    assert 'writestr("run.bat"' not in MAKE_DIST
    assert "start.bat" in MAKE_DIST, "the install notes must name the generated launcher"
    assert "ls-files" in MAKE_DIST, "the zip must be built from tracked files"
