"""The CLI's device report, and the one path nobody had tested: no `--device`.

Every pre-existing CLI device test in `tests/test_separate_cli.py` passes
`--device cpu` or `--device rocm`. A flag makes `effective_settings()` set
`gpu_conversion` (`separate_cli.py:262-263`), so the flag *always* agreed with the
banner, and the defect lived exactly where no test went: the un-flagged run. There,
`gpu_conversion` comes from the user's persisted `Settings` -- measured on this box
as `False` at `gui_data/constants.py:667` -- `plan.build()` maps it to
`is_gpu_conversion = -1` (`plan.py:329`), and `separate.py:253` then skips the whole
GPU block, so `self.device` stays `cpu` (`:233`) and `self.run_type` stays
`['CPUExecutionProvider']` (`:241`) while `describe_device()` printed
`backend: name=rocm torch_device=cuda`. A user copying the README's first example got
CPU audio under a GPU banner.

The ruling is that the CLI must not override a saved "GPU off", so the fix is a
second line stating the device the engine will act on. What these tests pin:

1. the un-flagged run reports **cpu**, and its wording names the setting that caused
   it (`is_gpu_conversion=-1 from Settings; pass --device rocm to use the GPU`);
2. `--device rocm` reports the GPU **and** puts `gpu_conversion` through to the plan
   (`is_gpu_conversion = 0` in the printed fields);
3. the report never claims a GPU device while `is_gpu_conversion < 0`, for every
   backend the machine can hand back -- which is the "two lines must never disagree"
   rule, checked against a re-derivation of `separate.py`'s own branch conditions
   rather than against a fixed string;
4. the child process a user actually runs (`python separate_cli.py …` with no flag)
   prints the same thing the in-process call does;
5. `separate.check_ort_providers`, the ONNX/DirectML half: a session that dropped the
   provider the run asked for is reported, and a CPU-only request says nothing.

The real weights root is required for 1-4: the plan only builds against a resolvable
weight. Following the fork's policy, a missing root is a `pytest.fail` naming the
variable to set, never a skip and never a fabricated spec.
"""
from __future__ import annotations

import ast
import io
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

import backends
import separate_cli
import uvr_paths
from tests.test_separate_cli import REPO, vr_run_fixture

CLI = REPO / "separate_cli.py"
COMPUTE_PREFIX = "[cli] compute device: "
BACKEND_PREFIX = "[cli] backend: "


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def cli_dry_run(tmp_path, *extra: str) -> str:
    """The un-/pre-flagged default run, as far as it goes: resolve + plan + print."""
    root, model, audio, out = vr_run_fixture(tmp_path)
    out_dir = tmp_path / "dry-out"
    argv = [str(part) for part in
            ["--audio", audio, "--out", out_dir, "--model-root", root,
             "--writable-dir", tmp_path / "writable",
             "--arch", "vr", "--model", model, "--dry-run", *extra]]
    printed, errors = io.StringIO(), io.StringIO()
    code = separate_cli.main(argv, out=printed, err=errors)
    assert code == 0, f"the dry run exited {code}:\n{printed.getvalue()}{errors.getvalue()}"
    return printed.getvalue() + errors.getvalue()


def compute_line(text: str) -> str:
    """The single `compute device:` line, with the exact count checked.

    Two such lines would mean the report answers the same question twice and the
    answers may differ, which is the failure this whole file exists to prevent.
    """
    lines = [line for line in text.splitlines() if line.startswith(COMPUTE_PREFIX)]
    assert len(lines) == 1, (
        f"expected exactly one compute-device line, got {len(lines)} in:\n{text}")
    return lines[0]


def backend_line(text: str) -> str:
    lines = [line for line in text.splitlines() if line.startswith(BACKEND_PREFIX)]
    assert len(lines) == 1, f"expected one backend line in:\n{text}"
    return lines[0]


def reported_device(line: str) -> str:
    """The bare device word: `cuda:0 (...)` -> `cuda`, `cpu (...)` -> `cpu`."""
    return line[len(COMPUTE_PREFIX):].split(" ")[0].split(":")[0]


def plan_field(text: str, name: str) -> str:
    """One `--dry-run` plan field's printed repr, straight from the report."""
    for line in text.splitlines():
        marker = f"[cli]   {name} = "
        if line.startswith(marker):
            return line[len(marker):]
    raise AssertionError(f"plan field {name!r} is not in the report:\n{text}")


def engine_device(is_gpu_conversion: int, torch_device: str, *, cuda_available: bool,
                  mps_available: bool) -> str:
    """`separate.py`'s own decision, re-derived -- the oracle, not a fixed string.

    Mirrors `separate.py:233` (starts at cpu), `:253` (the GPU block only runs when
    `is_gpu_conversion >= 0`), `:254-255` (mps first) and `:270-272` (cuda only when
    the backend asked for a cuda device *and* torch can open one). If the CLI's line
    ever names a different device than this does, the test fails with both.
    """
    if is_gpu_conversion < 0:
        return "cpu"
    if mps_available:
        return "mps"
    if torch_device == "cuda" and cuda_available:
        return "cuda"
    return "cpu"


def fake_torch(monkeypatch, *, cuda: bool, hip: str | None = None,
               mps: bool = False) -> types.SimpleNamespace:
    """A torch stub in `sys.modules`, the pattern `tests/test_separate_cli.py` uses.

    Needed because the report's truth depends on the live interpreter: on a CPU-only
    torch build no CLI flag can make `compute device: cuda` correct, so that
    combination can only be tested by replacing torch.
    """
    fake = types.SimpleNamespace(
        __version__="0.0+fake",
        version=types.SimpleNamespace(hip=hip),
        cuda=types.SimpleNamespace(is_available=lambda: cuda,
                                   # `describe_device()` names the device when torch
                                   # says one is open, so the stub has to answer that
                                   # too or the CLI line it feeds raises AttributeError.
                                   get_device_name=lambda index: "Fake AMD GPU"),
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: mps)))
    monkeypatch.setitem(sys.modules, "torch", fake)
    return fake


# --------------------------------------------------------------------------- #
# 1-2. the un-flagged path, and the flagged one, as the user runs them
# --------------------------------------------------------------------------- #

def test_un_flagged_default_run_reports_cpu(tmp_path):
    """C1, the exact copy-paste case: no `--device`, Settings says GPU off.

    `gpu_conversion` is False in the built-in defaults on this machine, so a run with
    no flag must say cpu -- even though `backends.resolve()` on the same machine
    answers rocm/cuda. Both lines are checked: the backend line keeps saying what the
    machine offers, the new line says what this run uses.
    """
    text = cli_dry_run(tmp_path)
    line = compute_line(text)

    if bool(separate_cli.settings_module.Settings().gpu_conversion):
        pytest.fail("this box's default Settings has gpu_conversion=True, so the "
                    "un-flagged run is the GPU case; the assertion below is only "
                    "meaningful while the default is off, as measured at HEAD.")

    assert line.startswith(COMPUTE_PREFIX + "cpu ("), line
    assert "is_gpu_conversion=-1" in line, line
    assert "from Settings" in line, line
    assert "pass --device rocm to use the GPU" in line, line
    assert plan_field(text, "is_gpu_conversion") == "-1", text
    assert reported_device(line) == "cpu"


def test_un_flagged_backend_line_still_names_the_gpu_the_machine_offers(tmp_path):
    """The two lines are separate claims, and both must survive.

    Deleting the rocm half of the report would fix C1 by destroying the useful part:
    on a GPU-off Settings box the backend line is still how a user learns the machine
    has ROCm at all. So the expectation is *both*, not *only cpu*. If this box has no
    ROCm build the backend line says cpu and the pair agrees trivially -- asserted
    below, not assumed.
    """
    text = cli_dry_run(tmp_path)
    backend = backend_line(text)
    line = compute_line(text)

    if "name=rocm" in backend or "name=cuda" in backend:
        assert "torch_device=cuda" in backend, backend
        assert reported_device(line) == "cpu", (
            f"the backend line offers a GPU while Settings declines it; the compute "
            f"line must then say cpu, or the pair disagrees:\n{backend}\n{line}")
    else:
        assert reported_device(line) == "cpu", line


def test_device_rocm_reports_the_gpu_and_reaches_the_plan(tmp_path, monkeypatch):
    """The flag works today; pinned so the new line cannot drift from it.

    Two separate claims, both needed: that the report says GPU, and that
    `effective_settings()` actually flipped `gpu_conversion` so the plan (and the
    engine) sees 0 rather than -1.

    torch is stubbed so the assertion holds on a machine without a ROCm build: the
    point is what `--device rocm` *does* (it sets the flag, the plan says 0, the line
    names cuda), which a CPU-only interpreter would otherwise report as cpu for the
    honest reason `describe_compute_device` states.
    """
    fake_torch(monkeypatch, cuda=True, hip="6.4-fake")
    text = cli_dry_run(tmp_path, "--device", "rocm")
    line = compute_line(text)

    assert plan_field(text, "is_gpu_conversion") == "0", text
    assert reported_device(line) == "cuda", line
    assert "torch.version.hip=6.4-fake" in line, line

    st = separate_cli.settings_module.Settings()
    changed = separate_cli.effective_settings(
        st, types.SimpleNamespace(process_method=None), "rocm")
    assert changed.gpu_conversion is True, (
        "--device rocm must set gpu_conversion; without it the plan says -1 and the "
        "engine stays on CPU no matter what this line prints")
    assert separate_cli.effective_settings(
        st, types.SimpleNamespace(process_method=None), "cpu").gpu_conversion is False


# --------------------------------------------------------------------------- #
# 3. the invariant: never claim a GPU the plan will not use
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("backend_name,torch_device", [
    ("rocm", "cuda"), ("cuda", "cuda"), ("cpu", "cpu")])
def test_the_report_never_claims_a_gpu_when_is_gpu_conversion_is_negative(
        monkeypatch, backend_name, torch_device):
    """The rule, stated as a rule: `is_gpu_conversion < 0` => the line names cpu.

    Tested across every backend a machine can hand back *and* with torch reporting a
    working GPU, because that is the combination that broke: a reachable ROCm build,
    a GPU-shaped Backend, and a plan that tells the engine to stay on CPU. The device
    word is what may not name a GPU -- the sentence may still mention `rocm`, since
    telling the user which flag turns the GPU on is the point of the line.
    """
    fake_torch(monkeypatch, cuda=True, hip="6.4-fake")
    monkeypatch.setattr(sys, "platform", "win32")
    backend = backends.Backend(name=backend_name, torch_device=torch_device,
                               ort_providers=["CPUExecutionProvider"])
    line = separate_cli.describe_compute_device(-1, backend, None)[0]
    assert line.startswith(COMPUTE_PREFIX + "cpu ("), line
    assert reported_device(line) == "cpu", line
    assert "(torch.version.hip=" not in line, line


@pytest.mark.parametrize("conversion,torch_device,cuda,mps,expected", [
    (-1, "cuda", True, False, "cpu"),      # Settings says off; GPU reachable anyway
    (-1, "cpu", False, False, "cpu"),
    (0, "cuda", True, False, "cuda"),      # the ROCm case on this box
    (0, "cuda", False, False, "cpu"),      # GPU asked for, driver not there
    (0, "cpu", False, False, "cpu"),       # CPU-only torch build, setting on
    (0, "cuda", False, True, "mps"),       # separate.py:254 takes mps first
])
def test_the_compute_line_matches_separate_py_own_branch(
        monkeypatch, conversion, torch_device, cuda, mps, expected):
    """Line-by-line agreement with the engine, for the combinations that can occur.

    `mps` is only reachable when `sys.platform == "darwin"` (`separate.py:43` gates
    it on `is_macos`), which the patch below reproduces; on Windows the mps probe
    never runs, and that is tested by leaving the platform alone in the other rows.
    """
    fake_torch(monkeypatch, cuda=cuda, hip="6.4-fake" if cuda or mps else None,
               mps=mps)
    monkeypatch.setattr(sys, "platform", "darwin" if mps else "win32")
    backend = backends.Backend(name="cpu" if torch_device == "cpu" else "rocm",
                               torch_device=torch_device,
                               ort_providers=["CPUExecutionProvider"])
    line = separate_cli.describe_compute_device(conversion, backend, None)[0]
    oracle = engine_device(conversion, torch_device, cuda_available=cuda,
                           mps_available=mps and sys.platform == "darwin")
    assert reported_device(line) == oracle == expected, line


def test_device_set_index_reaches_the_reported_device(monkeypatch):
    """`separate.py:272` suffixes the device with the index unless it is `Default`.

    A report that says `cuda` while the engine binds `cuda:1` is the same class of
    lie as saying `cuda` while it computes on CPU, in milder form, so the suffix is
    part of the line.
    """
    fake_torch(monkeypatch, cuda=True, hip="6.4-fake")
    monkeypatch.setattr(sys, "platform", "win32")
    gpu = backends.Backend(name="rocm", torch_device="cuda",
                           ort_providers=["CPUExecutionProvider"])
    assert separate_cli.describe_compute_device(0, gpu, "rocm", "Default")[0] \
        == "[cli] compute device: cuda (torch.version.hip=6.4-fake)"
    indexed = separate_cli.describe_compute_device(0, gpu, "rocm", "1")[0]
    assert indexed.startswith("[cli] compute device: cuda:1 (torch.version.hip="), indexed
    assert reported_device(indexed) == "cuda"


def test_the_cpu_and_gpu_lines_cannot_both_appear(monkeypatch):
    """One line per run, and it contains exactly one device word for the device.

    Guards the shape the README quotes: a reader must be able to tell from this line
    alone which device runs, so it may not name two of them.
    """
    fake_torch(monkeypatch, cuda=False, hip=None)
    monkeypatch.setattr(sys, "platform", "win32")
    cpu = backends.Backend(name="cpu", torch_device="cpu",
                           ort_providers=["CPUExecutionProvider"])
    line = separate_cli.describe_compute_device(-1, cpu, None)[0]
    assert line.count("device:") == 1, line
    assert " (torch.version.hip=" not in line, line


# --------------------------------------------------------------------------- #
# 4. the child process, un-flagged -- how the bug was actually shipped
# --------------------------------------------------------------------------- #

def test_child_process_un_flagged_prints_cpu(tmp_path):
    """`python separate_cli.py …` with no `--device`, the README's first example.

    The in-process tests above could pass while the shipped line differed, because
    the defect was in what a user reads. This is the same command the README tells
    people to copy, run as a child, asserting the *wording* rather than only the
    device: the README agent quotes it verbatim and cannot see this code.
    """
    root, model, audio, _ = vr_run_fixture(tmp_path)
    proc = subprocess.run(
        [sys.executable, str(CLI), "--audio", audio, "--out", tmp_path / "out",
         "--model-root", root, "--writable-dir", tmp_path / "writable",
         "--arch", "vr", "--model", model, "--dry-run"],
        cwd=REPO, capture_output=True, text=True, timeout=300)
    printed = proc.stdout + proc.stderr
    assert proc.returncode == 0, printed
    lines = [l for l in printed.splitlines() if l.startswith(COMPUTE_PREFIX)]
    assert len(lines) == 1, printed
    assert lines[0] == (COMPUTE_PREFIX + "cpu (is_gpu_conversion=-1 from Settings; "
                        "pass --device rocm to use the GPU)"), lines[0]
    assert "is_gpu_conversion = -1" in printed, printed
    assert "pass --device" in printed


def test_help_states_the_single_model_scope():
    """Minor 4: `selections={}` is a documented limit, not a silent one.

    Asserted on the rendered `--help` text rather than on the parser's internals,
    because what a user reads is the help output: a secondary model, a pre-proc model
    and the vocal splitter are unreachable from the CLI.
    """
    proc = subprocess.run([sys.executable, str(CLI), "--help"], cwd=REPO,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    low = proc.stdout.lower()
    assert "single-model" in low, proc.stdout
    for term in ("secondary model", "pre-proc", "vocal splitter"):
        assert term in low, f"--help does not mention that {term} is unreachable"
    assert "compute device" in low, (
        "--help should name the line that tells the user which device will run")


# --------------------------------------------------------------------------- #
# 5. I1: the ONNX provider the run asked for must be the provider it got
# --------------------------------------------------------------------------- #

class _FakeSession:
    """`ort.InferenceSession` is not needed to test one method call."""

    def __init__(self, providers):
        self._providers = list(providers)

    def get_providers(self):
        return list(self._providers)


def test_a_dropped_directml_provider_is_reported():
    """EP drop at session creation: requested DML, attached CPU, and it must be loud.

    `InferenceSession(path, providers=self.run_type)` does not raise when it cannot
    honour the list -- it keeps what it loaded. `mdx`/`onnx` have no golden baseline,
    so this unit test plus the runtime check are the whole guard on that path, and
    the SNR suite cannot catch a silent DML-to-CPU downgrade.
    """
    import separate

    backends._warned.clear()                # the reporter de-dupes per process
    try:
        with pytest.warns(RuntimeWarning) as caught:
            attached = separate.check_ort_providers(
                _FakeSession(["CPUExecutionProvider"]),
                ["DmlExecutionProvider", "CPUExecutionProvider"],
                "UVR-MDX-NET-Inst_HQ_3.onnx")
    finally:
        backends._warned.clear()

    message = str(caught[0].message)
    assert attached == ["CPUExecutionProvider"], attached
    assert "dropped an execution provider" in message, message
    assert "DmlExecutionProvider" in message, message
    assert "CPUExecutionProvider" in message, message
    assert "enable_fallback" in message, (
        f"the run-time half of the downgrade must be named, since this check cannot "
        f"see it: {message}")
    assert "UVR-MDX-NET-Inst_HQ_3.onnx" in message, message


def test_a_cpu_only_request_says_nothing():
    """The machine that never asked for DirectML must not be warned at.

    `backends.resolve()` returns CPU-only providers when the vendor scan finds no AMD
    GPU, so a silent path here is required: warning on every run of a correct
    configuration is how these notices get ignored.
    """
    import warnings

    import separate

    backends._warned.clear()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            separate.check_ort_providers(
                _FakeSession(["CPUExecutionProvider"]), ["CPUExecutionProvider"],
                "cpu-only.onnx")
            separate.check_ort_providers(
                _FakeSession(["DmlExecutionProvider", "CPUExecutionProvider"]),
                ["DmlExecutionProvider", "CPUExecutionProvider"], "kept.onnx")
    finally:
        backends._warned.clear()
    assert caught == [], [str(w.message) for w in caught]


def test_check_ort_providers_is_called_after_the_session_is_built():
    """The call site, pinned structurally.

    `separate.py:592` builds the ONNX session; a helper nobody calls is the same as
    no helper, and this file cannot be imported for a behavioural test of that line
    without the weights the `mdx` path does not have here.
    """
    tree = ast.parse((REPO / "separate.py").read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute)
             and node.func.attr == "InferenceSession"]
    assert calls, "the ONNX InferenceSession call vanished from separate.py"
    source = (REPO / "separate.py").read_text(encoding="utf-8").splitlines()
    checked = [c.lineno for c in calls
               if any("check_ort_providers(" in line
                      for line in source[c.lineno - 1:c.lineno + 6])]
    assert len(checked) == len(calls), (
        f"InferenceSession built at line(s) {[c.lineno for c in calls]} but only "
        f"{checked} are followed by a get_providers() check")
