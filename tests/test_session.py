"""The CLI's observable order is part of its contract.

`separate_cli.run()` prints the backend line *before* `plan.build()` and the compute
line *after* (separate_cli.py:481-497, and the comment at :490 explains why the pair
is honest only in that order). A refactor that assembles everything first and prints
afterwards would lose the backend line whenever the plan refuses -- which is exactly
the case a user needs it in, because "plan refused" with no device line reads as
"nothing was resolved".

Both tests were rewritten on 2026-11-05 after the first drafts were measured against
the real engine, because neither ran:

* `1_HP-UVR` cannot be resolved from a `tmp_path` root at all -- the catalog looks for
  a weight file and raises `CatalogError` first -- and with the installed root it
  resolves *and plans successfully*, so `pytest.raises(plan.PlanError)` had nothing to
  catch. The refusal below is a real one, produced deterministically.
* the device lines are compared against the installed root (`make_catalog(None, ...)`)
  because that is the only root where the CLI gets far enough to print them.
"""
from __future__ import annotations

import io

import backends
import catalog
import plan
import settings as settings_module
import pytest


def _refusing_cat(tmp_path):
    """A catalog holding one VR weight whose tier-1 params row is empty.

    This is the deterministic way to reach `plan.PlanError` *after* the device line:
    a bare weight file makes `resolve()` raise `UnknownModelParamsError` (tier 3), so
    `write_params(..., {})` stands in for the GUI's answer; the row then resolves at
    tier 1 (`needs_user_params False`) and `plan.build` refuses at plan.py:488 because
    `params["vr_model_param"]` is absent and `modelparams/None.json` does not exist.
    Nothing here is simulated for the test's convenience: it is the same refusal a user
    gets from a real weight with a stubbed params file.
    """
    import separate_cli

    root = tmp_path / "root"
    (root / "VR_Models").mkdir(parents=True)
    (root / "VR_Models" / "mystery.pth").write_bytes(b"z" * 1024)
    cat = separate_cli.make_catalog(root, tmp_path / "w")
    with pytest.raises(catalog.UnknownModelParamsError) as excinfo:
        cat.resolve("mystery", required=True)
    spec = excinfo.value.spec
    cat.write_params(spec.arch, spec.hash, {})
    return cat


def test_device_line_is_printed_before_a_plan_refusal(tmp_path):
    import app.session as session

    cat = _refusing_cat(tmp_path)
    out = io.StringIO()
    with pytest.raises(plan.PlanError):
        session.assemble(st=settings_module.Settings(), cat=cat, model="mystery",
                         force_device="cpu", printer=out.write)
    printed = out.getvalue()
    assert "backend:" in printed, printed
    assert "compute device:" not in printed, printed


def test_assemble_output_matches_the_clis_own_lines(tmp_path):
    """Dry-run through the CLI and through assemble() must print the same device lines."""
    import app.session as session
    import separate_cli

    argv = ["--audio", str(tmp_path / "missing.wav"), "--dry-run", "--model", "1_HP-UVR",
            "--device", "cpu", "--writable-dir", str(tmp_path / "w")]
    cli_out = io.StringIO()
    assert separate_cli.main(argv, out=cli_out) == 0, cli_out.getvalue()
    cli_lines = [ln for ln in cli_out.getvalue().splitlines()
                 if ln.startswith("[cli] backend:") or ln.startswith("[cli] torch ")]
    assert cli_lines, "the CLI printed no device line at all -- the gate is vacuous"

    asm_out = io.StringIO()
    # `None` is the installed weights root (uvr_paths.MODEL_ROOT from UVR_MODEL_ROOT);
    # a tmp_path root cannot resolve a real model, so the CLI would fail before printing.
    cat = separate_cli.make_catalog(None, tmp_path / "w2")
    session.assemble(st=settings_module.Settings(), cat=cat, model="1_HP-UVR",
                     force_device="cpu", printer=asm_out.write)
    assert asm_out.getvalue().splitlines() == cli_lines


def test_importing_session_pulls_in_no_qt():
    """`separate_cli.run()` reaches `app.session` from inside its body, and
    `tests/test_engine_purity.py` blocks `app` for a module that is on its own guarded
    import list. The deferral is therefore load-bearing, and the only thing that makes
    it a designed exception rather than a blind spot is this assertion: the shared
    assembly path must be Qt-free, or the CLI stops being runnable headless the moment
    a wheel is missing. Checked in a child process, because this suite has already
    imported Qt by the time it gets here.
    """
    import subprocess
    import sys

    code = ("import sys; import app.session; "
            "bad = [m for m in sys.modules if m == 'PySide6' or m.startswith('PySide6.')"
            " or m == 'fluentqt' or m.startswith('fluentqt.')]; "
            "sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, f"app.session imported Qt modules: {r.stdout} {r.stderr}"


def test_dry_run_never_imports_the_engine(tmp_path):
    """`separate_cli.py:22-27` promises the no-run path never pulls the engine into the
    process, which is what keeps `--dry-run` cheap and usable on a machine whose venv is
    not the ROCm one. Task 2 moved the assembly into `session.assemble()`, and an
    engine-class lookup there would break the promise by importing `separate` (ruling
    R-T2-lazy). Child process, because by now the suite has imported torch itself.

    The assertion is about `separate`, not about `torch`, because the brief's literal
    `torch` wording is not the engine's real behaviour and was measured here before
    being narrowed: at HEAD 59d3799 -- before Task 2 touched anything -- the same child
    (`--model 1_HP-UVR --device cpu --dry-run`) exits 0 with `torch` in `sys.modules`
    and `separate` absent. `torch` arrives from `separate_cli.py:337`, inside
    `describe_compute_device`, which HEAD and `assemble()` both call *before* the
    dry-run return because the `compute device:` line is pinned output
    (`tests/test_cli_device_reporting.py`); with no `--device` flag it also arrives
    from `backends._hip_version()`. What `--dry-run` really promises, and what R-T2-lazy
    restores, is that it never reaches `separate` -- the module that loads the model
    classes and the whole ROCm-side import graph. Asserting `torch` here would be a
    red test against code that was never broken, so it asserts the engine instead.
    """
    import os
    import subprocess
    import sys
    from pathlib import Path

    repo = str(Path(__file__).resolve().parents[1])
    code = (
        "import sys; sys.argv=['separate_cli','--model','1_HP-UVR','--device','cpu',"
        f"'--dry-run','--writable-dir',r'{tmp_path / 'w'}'];"
        "import separate_cli; rc = separate_cli.main();"
        "sys.exit(1 if ('separate' in sys.modules or rc != 0) else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=repo, capture_output=True,
                       text=True, env={**os.environ, "PYTHONPATH": repo})
    assert r.returncode == 0, (
        f"--dry-run imported the engine or failed:\nout={r.stdout}\nerr={r.stderr}")


def test_no_engine_class_names_the_arch_and_the_known_list(tmp_path, monkeypatch):
    """The branch existed in the old `run()` with no test. Ruling R-T2-noclass keeps its
    text and exit code, so pin the text: it must read
    `arch 'vr' has no engine class in separate.py; known: ...`, and must NOT arrive as a
    `plan refused:` line, because that prefix belongs to `PlanError` only.
    """
    import separate_cli
    from app import session

    cat = _refusing_cat(tmp_path)
    spec = cat.resolve("mystery", required=True)      # tier 1 now answers it
    # `run_job` looks the class up through `separate_cli.engine_classes()`, so that is
    # the seam: no fourth arch is added to the shipped table to make this fail.
    monkeypatch.setattr(separate_cli, "engine_classes",
                        lambda: {"mdx": object, "demucs": object})
    sess = session.Session(spec=spec, backend=backends.resolve(force="cpu"),
                           rp=None, compute_lines=())  # rp is unused: the raise comes first
    with pytest.raises(session.NoEngineClass) as excinfo:
        session.run_job(sess, audio=tmp_path / "in.wav", export_path=tmp_path / "out")
    assert excinfo.value.arch == "vr"
    text = str(excinfo.value)
    assert text.startswith("arch 'vr' has no engine class in separate.py; known:")
    assert "mdx" in text and "demucs" in text
