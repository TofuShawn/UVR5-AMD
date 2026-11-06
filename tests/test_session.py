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
from pathlib import Path

import backends
import catalog
import plan
import selections as selections_module
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
    """`separate_cli`'s "No GUI, and no import of one" bullet (`separate_cli.py:21-40`)
    promises the no-run path never pulls the engine into the process, which is what keeps
    `--dry-run` cheap and usable on a machine whose venv is not the ROCm one. Task 2 moved
    the assembly into `session.assemble()`, and an engine-class lookup there would break
    the promise by importing `separate` (ruling R-T2-lazy). Child process, because by now
    the suite has imported torch itself.

    The assertion is about `separate`, not about `torch`, because the brief's literal
    `torch` wording is not the engine's real behaviour and was measured here before
    being narrowed: at HEAD 59d3799 -- before Task 2 touched anything -- the same child
    (`--model 1_HP-UVR --device cpu --dry-run`) exits 0 with `torch` in `sys.modules`
    and `separate` absent. `torch` arrives from `separate_cli.py:349`, inside
    `describe_compute_device`, which HEAD and `assemble()` both call *before* the
    dry-run return because the `compute device:` line is pinned output
    (`tests/test_cli_device_reporting.py`); with no `--device` flag it also arrives
    from `backends._hip_version()`. What `--dry-run` really promises, and what R-T2-lazy
    restores, is that it never reaches `separate` -- the module that loads the model
    classes and the whole ROCm-side import graph. Asserting `torch` here would be a
    red test against code that was never broken, so it asserts the engine instead.
    Plan 2 Task 8 rewrote that bullet to say this measurement rather than the false
    "without touching torch" claim, and re-measured the other no-run command too:
    `--list-models` returns before `run()` reaches the device line, so it is the one
    command that really is torch-free. If either line moves, the citation above and the
    bullet it quotes are the two places to fix.
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


# -- C1: the formats the engine can write, and the ones it cannot -------------- #

class _Plan:
    """Just enough of a `RunPlan` for `run_job`: no field it reads is faked here, so
    this stays the shape `model_count` and the progress wrapper already tolerate."""

    is_secondary_model_activated = False
    secondary_model_4_stem: list = []
    pre_proc_model = None
    vocal_split_model = None
    save_format = "FLAC"


def _sess(engine_class):
    from app import session
    return session.Session(spec=None, backend=None, rp=_Plan(),
                           engine_class=engine_class, compute_lines=())


def _converted_output_engine(suffix: str):
    """An engine shaped like `separate.py:1541-1570`: it writes the wav, converts it to
    `suffix`, and removes the wav unconditionally.
    """

    class Converted:
        def __init__(self, rp, process_data):
            self.pd = process_data

        def seperate(self):
            out = Path(self.pd["export_path"])
            wav = out / "song_(Vocals).wav"
            wav.write_bytes(b"RIFF")
            (out / f"song_(Vocals){suffix}").write_bytes(b"data")
            wav.unlink()                              # separate.py:1566-1570

    return Converted


@pytest.mark.parametrize("suffix", [".flac", ".mp3"])
def test_run_job_sees_a_converted_stem_in_every_format_the_engine_can_write(
        tmp_path, suffix):
    """C1(a). A FLAC or MP3 run is legitimate -- both are engine-supported and both are
    selectable on the shipped settings page -- but `run_job` diffed `rglob("*.wav")`
    only, so it returned [] and the GUI said "Done - 0 file(s) written" over two files
    on disk. The suffix list now comes from `session.OUTPUT_SUFFIXES` and covers all
    three.
    """
    from app import session

    written = session.run_job(_sess(_converted_output_engine(suffix)),
                              audio=tmp_path / "in.wav", export_path=tmp_path / "out")
    assert [p.name for p in written] == [f"song_(Vocals){suffix}"], written
    assert written[0].is_file()


def test_run_job_still_sees_the_plain_wav_output(tmp_path):
    """The other half of the list: `save_format == 'WAV'` writes the wav and removes
    nothing (`separate.py:1542` only converts when the format is not WAV)."""
    from app import session

    class WavOnly:
        def __init__(self, rp, process_data):
            self.pd = process_data

        def seperate(self):
            (Path(self.pd["export_path"]) / "song_(Vocals).wav").write_bytes(b"RIFF")

    written = session.run_job(_sess(WavOnly), audio=tmp_path / "in.wav",
                              export_path=tmp_path / "out")
    assert [p.name for p in written] == ["song_(Vocals).wav"]


class _NoOutputEngine:
    """An engine that returns normally having written nothing -- the case I2 is about."""

    def __init__(self, rp, process_data):
        self.pd = process_data

    def seperate(self):
        return None


def test_run_job_refuses_when_the_engine_returned_without_writing(tmp_path):
    """I2: the emptiness check used to live only in `separate_cli.run()`, so the two
    front ends reached *different verdicts from the same return value* -- the CLI exited
    1, the GUI emitted `succeeded` and printed "Done - 0 file(s) written". One assembly
    path now carries one verdict (`app/session.py:1-9`), inherited by both callers."""
    from app import session

    with pytest.raises(session.NoStemWritten) as excinfo:
        session.run_job(_sess(_NoOutputEngine), audio=tmp_path / "in.wav",
                        export_path=tmp_path / "out")
    assert "without writing any stem" in str(excinfo.value)


def test_the_empty_written_cli_bytes_do_not_move(tmp_path, monkeypatch):
    """The CLI's refusal sentence is the one the finding says must not move:
    `[cli] the engine returned without writing any stem to <dir>; that is an engine
    failure, not a success.` on stderr with exit 1. It now arrives from
    `session.NoStemWritten` rather than from `run()`'s own `if not written`, and the
    `except` in `run()` re-prints it verbatim.
    """
    import separate_cli
    from app import session

    export = tmp_path / "UVR_out"

    def raiser(*_args, **_kwargs):
        raise session.NoStemWritten(export)

    monkeypatch.setattr(session, "run_job", raiser)
    audio = tmp_path / "in.wav"
    audio.write_bytes(b"RIFF")
    err = io.StringIO()
    code = separate_cli.main(["--audio", str(audio), "--model", "1_HP-UVR", "--device",
                              "cpu", "--writable-dir", str(tmp_path / "w")],
                             out=io.StringIO(), err=err)
    assert code == separate_cli.EXIT_ENGINE, code
    assert err.getvalue() == (f"[cli] the engine returned without writing any stem to "
                              f"{export}; that is an engine failure, not a success.\n"), \
        err.getvalue()


@pytest.mark.parametrize("bad", ["mp3", "WAV ", "OGG", "", "wav"])
def test_assemble_refuses_a_save_format_the_engine_cannot_write(tmp_path, bad):
    """C1(b). The engine's legal domain is three literals (`gui_data/constants.py:521-523`)
    and `Settings.save_format` is a free-text field, so `'mp3'` took the
    convert-nothing branch of `separate.py:save_format()` and deleted the wav anyway:
    no output at all. Refuse before the run, in the one assembly path both front ends
    use, and raise a `ValueError` subclass so the CLI's existing handler and the GUI's
    existing `_refuse` both pick it up.

    `'wav'` in lowercase and `'WAV '` are the two a hand-edited `settings.json` or a
    mistyped box actually produces: the match is exact, because the engine's own
    comparison is (`separate.py:1542` reads `if not save_format == WAV`).
    """
    import separate_cli
    from app import session

    st = settings_module.Settings(save_format=bad)
    cat = separate_cli.make_catalog(tmp_path / "nope", tmp_path / "w")
    with pytest.raises(session.InvalidSaveFormat) as excinfo:
        session.assemble(st=st, cat=cat, model="1_HP-UVR", force_device="cpu",
                         printer=lambda _line: None)
    assert type(excinfo.value) is not ValueError
    assert isinstance(excinfo.value, ValueError)
    for token in session.SAVE_FORMATS:
        assert token in str(excinfo.value), str(excinfo.value)
    assert repr(bad) in str(excinfo.value) or bad == ""


def test_assemble_accepts_every_format_it_lists(tmp_path):
    """The refusal must not eat the legal values: a FLAC run is a real run."""
    import separate_cli
    from app import session

    cat = separate_cli.make_catalog(None, tmp_path / "w")
    for legal in session.SAVE_FORMATS:
        st = settings_module.Settings(save_format=legal)
        session.assemble(st=st, cat=cat, model="1_HP-UVR", force_device="cpu",
                         printer=lambda _line: None)


def test_a_bad_save_format_refuses_from_the_cli_too(tmp_path):
    """Both front ends, one verdict: the CLI exits with the usage code and prints the
    same sentence the GUI routes to its console."""
    import separate_cli

    settings_path = tmp_path / "settings.json"
    settings_path.write_text('{"save_format": "mp3"}', encoding="utf-8")
    out, err = io.StringIO(), io.StringIO()
    code = separate_cli.main(["--audio", "in.wav", "--model", "1_HP-UVR", "--device",
                              "cpu", "--settings", str(settings_path),
                              "--writable-dir", str(tmp_path / "w")],
                             out=out, err=err)
    assert code == separate_cli.EXIT_USAGE, code
    assert "save_format" in err.getvalue() and "mp3" in err.getvalue(), err.getvalue()


def test_output_suffixes_are_pinned_to_the_engines_own_literals():
    """The drift guard for `session.SAVE_FORMATS` / `OUTPUT_SUFFIXES`, in the shape
    `app/dialogs.py:111-128` established: the app holds its own literals and a test
    compares them against the engine's, so a rename upstream that changes no behaviour
    cannot turn the GUI red while a real change in the domain cannot pass unnoticed.

    Two sides are checked. `gui_data/constants.py` for the three format tokens, and
    `separate.py`'s `save_format()` body for the three file suffixes it writes -- the
    latter by `ast`, because importing `separate` pulls the whole ROCm graph (ruling
    R-T2-lazy).
    """
    import ast

    from app import session

    import gui_data.constants as const
    for token, value in (("WAV", const.WAV), ("FLAC", const.FLAC), ("MP3", const.MP3)):
        assert token in session.SAVE_FORMATS, (
            f"gui_data.constants.{token} is {value!r} but app/session.py does not offer "
            f"it -- the app's copy of the output-format domain has drifted")
    assert set(session.SAVE_FORMATS) == {const.WAV, const.FLAC, const.MP3}, (
        f"the app lists {session.SAVE_FORMATS}, the engine defines "
        f"{(const.WAV, const.FLAC, const.MP3)}")

    tree = ast.parse((Path(__file__).resolve().parents[1] / "separate.py")
                     .read_text(encoding="utf-8"))
    body = next((node for node in ast.walk(tree)
                 if isinstance(node, ast.FunctionDef) and node.name == "save_format"),
                None)
    assert body is not None, "separate.py no longer has a save_format() to diff against"
    suffixes = {c.value for c in ast.walk(body)
                if isinstance(c, ast.Constant) and isinstance(c.value, str)
                and c.value.startswith(".")}
    assert suffixes == set(session.OUTPUT_SUFFIXES), (
        f"separate.py's save_format() writes {sorted(suffixes)} but app/session.py "
        f"diffs {sorted(session.OUTPUT_SUFFIXES)} -- a fourth format would be "
        f"invisibly dropped by run_job again")


# -- I6: one default export directory for both front ends ---------------------- #

def test_default_export_path_is_beside_the_input(tmp_path):
    from app import session

    assert session.default_export_path(tmp_path / "song.wav") == tmp_path / "UVR_out"
    assert session.default_export_path(str(tmp_path / "song.wav")) == tmp_path / "UVR_out"


def test_both_front_ends_default_to_the_same_export_directory(tmp_path, monkeypatch):
    """Reviewer B: no test referenced `UVR_out` at all, while the default was spelled
    twice -- `app/controller.py:138` and `separate_cli.py:510` -- which is the
    divergence the comment above one of the two copies claimed to prevent."""
    import separate_cli
    from app import session

    audio = tmp_path / "in.wav"
    audio.write_bytes(b"RIFF")
    seen = {}

    def raiser(*_args, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop after the path")

    monkeypatch.setattr(session, "run_job", raiser)
    err = io.StringIO()
    separate_cli.main(["--audio", str(audio), "--model", "1_HP-UVR", "--device", "cpu",
                       "--writable-dir", str(tmp_path / "w")],
                      out=io.StringIO(), err=err)
    assert seen["export_path"] == session.default_export_path(audio), seen
    assert seen["export_path"].name == "UVR_out"


# -- I3: a mis-typed saved value must not be a silent no-op -------------------- #

def test_start_after_a_bad_deverb_setting_reports_instead_of_silently_doing_nothing(
        qapp, tmp_path, monkeypatch):
    """`plan.py:368` does a raw `DEVERB_MAPPER[settings.deverb_vocal_opt]` subscript on
    every plan, so any other string -- including a cleared field -- raises `KeyError`,
    which was not in the controller's `except` tuple. On PySide6 6.9.3 an exception that
    escapes a Qt slot prints to stderr and the event loop carries on, which the codebase
    itself documents as "looked exactly like a button that did nothing"
    (`app/settings_page.py:281-291`). START must now *say* what happened."""
    import separate_cli
    from app import controller, session

    monkeypatch.setattr(backends, "resolve",
                        lambda force=None: backends.Backend(
                            name="cpu", torch_device="cpu",
                            ort_providers=["CPUExecutionProvider"]))
    cat = separate_cli.make_catalog(tmp_path / "nope", tmp_path / "w")
    (cat.writable_dir).mkdir(parents=True, exist_ok=True)

    def explode(**_kwargs):
        raise KeyError("nope")

    monkeypatch.setattr(session, "assemble", explode)
    ctrl = controller.Controller(cat=cat, st=settings_module.Settings(
        deverb_vocal_opt="nope"), selections=selections_module.Selections())
    pg = ctrl.work_page
    audio = tmp_path / "in.wav"
    audio.write_bytes(b"RIFF")
    pg.model_combo.clear()
    pg.model_combo.addItems(["1_HP-UVR"])
    pg.input_edit.setText(str(audio))
    pg.output_edit.setText(str(tmp_path / "out"))

    pg.run_requested.emit()

    text = pg.console.toPlainText()
    assert "not started" in text, text
    assert "KeyError" in text, text
    assert pg.start_button.isEnabled(), "a refusal must not leave the page busy"
    assert pg.compute_label.text() == "", (
        "I5(a): a refusal may not leave the previous run's compute line on screen")
