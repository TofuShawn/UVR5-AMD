"""Task 6: `backends.py` is the single device/provider authority.

Two halves. The first pins the *decision table* -- every probe is monkeypatched,
so what these tests check is what `resolve()` concludes from
(hip, cuda, amd-vendor, force), not the real WMI/torch/ORT hardware; Step 7 of
the brief is the one place the hardware itself is asserted. The failure paths of
the vendor probe are the exception: they run the real `_is_amd_device()` with
`subprocess.run` replaced, because "a probe failure warns instead of quietly
dropping DirectML" is only testable at that boundary.

The second half pins the *consumer*: `separate.SeperateAttributes` must read the
authority in every mode (no-GPU, GPU, mps), take its providers from it, and warn
when a resolved GPU claim cannot be honoured.

`_no_real_probing` is autouse: `resolve()` must never shell out to WMI or import
onnxruntime during the unit tests. `test_resolve_probes_at_most_once` overrides
`_is_amd_device` again with its own counting probe -- that is intentional and it
is the only test that must.
"""
import subprocess
import sys
import types
import warnings

import pytest

import backends
import separate
from tests.golden import runner, settings

#: The real vendor probe, captured before the autouse fixture replaces the module
#: attribute. The failure-path tests below call it directly -- they are testing
#: that probe, so they must not be handed the fixture's stub.
_REAL_VENDOR_PROBE = backends._is_amd_device


@pytest.fixture(autouse=True)
def _no_real_probing(monkeypatch):
    """resolve() must never shell out to WMI or import onnxruntime in tests."""
    monkeypatch.setattr(backends, "_available_ort", lambda: {"DmlExecutionProvider",
                                                             "CUDAExecutionProvider",
                                                             "CPUExecutionProvider"})
    monkeypatch.setattr(backends, "_is_amd_device", lambda: True)


def _real_probe_with_fake_scan(monkeypatch, platform, result_or_error, calls=None):
    """The real `_is_amd_device()` against a replaced `subprocess.run`.

    Returns the callable to invoke, so a test can call it as often as it likes:
    direct calls bypass `_once`'s memo, which is keyed on the callable and would
    otherwise hand the second caller a cached answer. Pass `calls` to also record
    what the fake intercepted -- that is how the "it really does shell out"
    control stays non-vacuous.

    Note `backends.subprocess` *is* the stdlib module, so this patches
    `subprocess.run` process-wide for the duration of the test -- harmless here
    (nothing else runs concurrently, and monkeypatch restores it), but it is the
    reason to load anything from disk before calling this helper.
    """
    monkeypatch.setattr(backends, "_is_amd_device", _REAL_VENDOR_PROBE)
    monkeypatch.setattr(sys, "platform", platform)

    def fake_run(*args, **kwargs):
        if calls is not None:
            calls.append(args)
        if isinstance(result_or_error, Exception):
            raise result_or_error
        return result_or_error

    monkeypatch.setattr(backends.subprocess, "run", fake_run)
    return backends._is_amd_device


def _backend_warnings(records):
    """The RuntimeWarnings among `records` -- the device layer's channel."""
    return [str(w.message) for w in records if issubclass(w.category, RuntimeWarning)]


def test_rocm_reports_cuda_device_string(monkeypatch):
    monkeypatch.setattr(backends, "_hip_version", lambda: "7.2.42141")
    monkeypatch.setattr(backends, "_cuda_available", lambda: True)
    b = backends.resolve()
    assert b.name == "rocm"
    assert b.torch_device == "cuda"
    assert b.ort_providers[0] == "DmlExecutionProvider"


def test_nvidia_keeps_cuda_provider(monkeypatch):
    monkeypatch.setattr(backends, "_hip_version", lambda: None)
    monkeypatch.setattr(backends, "_cuda_available", lambda: True)
    monkeypatch.setattr(backends, "_is_amd_device", lambda: False)
    b = backends.resolve()
    assert b.name == "cuda"
    assert b.ort_providers[0] == "CUDAExecutionProvider"


def test_cpu_fallback(monkeypatch):
    monkeypatch.setattr(backends, "_hip_version", lambda: None)
    monkeypatch.setattr(backends, "_cuda_available", lambda: False)
    monkeypatch.setattr(backends, "_is_amd_device", lambda: False)
    b = backends.resolve()
    assert b.name == "cpu"
    assert b.torch_device == "cpu"
    assert b.ort_providers == ["CPUExecutionProvider"]


def test_cpu_torch_still_gets_directml_for_onnx(monkeypatch):
    monkeypatch.setattr(backends, "_hip_version", lambda: None)
    monkeypatch.setattr(backends, "_cuda_available", lambda: False)
    monkeypatch.setattr(backends, "_is_amd_device", lambda: True)
    b = backends.resolve()
    assert b.torch_device == "cpu"
    assert b.ort_providers[0] == "DmlExecutionProvider"


def test_force_cpu_overrides_torch_but_not_onnx(monkeypatch):
    """`force` exists so a user can pin torch to CPU; it must not silently
    pin ONNX to CPU too, and it must not leave name == "rocm" claiming a
    device we are not using."""
    monkeypatch.setattr(backends, "_hip_version", lambda: "7.2.42141")
    monkeypatch.setattr(backends, "_cuda_available", lambda: True)
    b = backends.resolve(force="cpu")
    assert b.name == "cpu"
    assert b.torch_device == "cpu"
    assert b.ort_providers[0] == "DmlExecutionProvider"


def test_force_rejects_unknown_backend():
    with pytest.raises(ValueError, match="expected rocm|cuda|cpu"):
        backends.resolve(force="opencl")


def test_resolve_probes_at_most_once(monkeypatch):
    """SeperateAttributes is constructed once per model per stem, so resolve()
    runs many times in a single run. The WMI query costs ~0.5 s and ORT provider
    enumeration is not free; both must be memoised.

    The memoisation has to survive the probe symbol itself being replaced:
    `monkeypatch.setattr` swaps `_is_amd_device` for a fresh, un-decorated
    function, so an `@lru_cache` living *on* `_is_amd_device` would be bypassed
    and the counter would hit 5. `resolve()` therefore keys its memo on the
    probe callable, not on the module attribute -- see backends._once.

    Both probes named by the cost argument are counted, and each must be called
    *exactly* once: `<= 1` would also be satisfied by a refactor that stopped
    consulting the vendor scan at all, which is a different bug.
    """
    calls = {"wmi": 0, "ort": 0}

    def counting_amd():
        calls["wmi"] += 1
        return True

    def counting_ort():
        calls["ort"] += 1
        return {"DmlExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"}

    monkeypatch.setattr(backends, "_is_amd_device", counting_amd)
    monkeypatch.setattr(backends, "_available_ort", counting_ort)
    for _ in range(5):
        backends.resolve()
    assert calls == {"wmi": 1, "ort": 1}, calls


# --------------------------------------------------------------------------- #
# Vendor-probe failure paths (fix round 1, finding 1)
# --------------------------------------------------------------------------- #
# `except Exception: return False` used to answer three different events --
# no powershell, a WMI timeout, WMI access denied -- exactly as "this is not an
# AMD box", and the only consequence was that DirectML vanished and MDX-C ran on
# CPU with no message anywhere. The conservative fallback stays; it now speaks.

@pytest.mark.parametrize("failure", [
    FileNotFoundError(2, "The system cannot find the file specified: 'powershell'"),
    subprocess.TimeoutExpired(cmd="powershell", timeout=10),
    PermissionError(13, "WMI access denied"),
], ids=["no-powershell", "wmi-timeout", "wmi-access-denied"])
def test_vendor_probe_failure_warns_instead_of_passing_as_no_amd_gpu(monkeypatch, failure):
    probe = _real_probe_with_fake_scan(monkeypatch, "win32", failure)

    with pytest.warns(RuntimeWarning) as record:
        assert probe() is False

    # The caught exception has to be *in* the message, or the user cannot act on it.
    text = str(record[0].message)
    assert type(failure).__name__ in text, text
    assert "CPUExecutionProvider" in text and "DirectML" in text, text


def test_vendor_probe_warns_when_powershell_runs_but_wmi_refuses(monkeypatch):
    """A non-zero exit is not "no AMD GPU": stdout is empty for a different reason.

    Without this branch the `returncode`-carrying result of the old
    ``.stdout``-only read made a locked-down WMI look exactly like an Intel box.
    """
    probe = _real_probe_with_fake_scan(
        monkeypatch, "win32",
        types.SimpleNamespace(returncode=1, stdout="", stderr="Get-CimInstance : Access denied"))

    with pytest.warns(RuntimeWarning) as record:
        assert probe() is False

    assert "exit 1" in str(record[0].message) and "Access denied" in str(record[0].message)


def test_degradation_warning_survives_the_engines_ambient_ignore_filter(monkeypatch):
    """Loud, not merely present in the source.

    ``separate.py`` runs ``warnings.filterwarnings("ignore")`` at import -- the
    head of ``warnings.filters`` becomes ``('ignore', None, Warning, None, 0)`` --
    so a plain ``warnings.warn`` from the device layer would still be "no message
    anywhere" in the shipped product. `backends.warn_degradation` tightens the
    filter only inside its own ``catch_warnings`` to escape that, and de-dupes by
    message so a per-model-per-stem run says it once rather than eight times.

    (The exception text below is unique in this file on purpose: the de-dupe keys
    on the message, so a repeated message would correctly warn nobody twice.)
    """
    probe = _real_probe_with_fake_scan(monkeypatch, "win32", FileNotFoundError(2, "powershell not on PATH"))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        with pytest.warns(RuntimeWarning) as record:
            assert probe() is False
        assert "powershell not on PATH" in str(record[0].message)

        with warnings.catch_warnings(record=True) as again:
            warnings.simplefilter("always")
            assert probe() is False
        assert again == [], [str(w.message) for w in again]


def test_vendor_probe_answers_a_working_scan_without_warning(monkeypatch):
    """Non-vacuous control: the probe really does shell out and read the vendor id."""
    calls = []
    probe = _real_probe_with_fake_scan(
        monkeypatch, "win32",
        types.SimpleNamespace(returncode=0, stdout="PCI\\VEN_1002&DEV_744C\n", stderr=""),
        calls=calls)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert probe() is True

    assert len(calls) == 1, "the probe must issue exactly one scan"
    assert [str(w.message) for w in caught] == [], [str(w.message) for w in caught]


def test_vendor_probe_is_gated_on_windows_instead_of_discovering_it_by_exception(monkeypatch):
    """`sys.platform` says the WMI scan cannot run here, so it is not attempted --
    and no warning is raised, because off Windows there is no DirectML to lose."""
    probe = _real_probe_with_fake_scan(monkeypatch, "darwin", AssertionError("must not shell out"))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert probe() is False

    assert [str(w.message) for w in caught] == [], [str(w.message) for w in caught]


# --------------------------------------------------------------------------- #
# force= claims the engine has to check (fix round 1, finding 2, part 1)
# --------------------------------------------------------------------------- #

def test_force_cuda_keeps_the_cuda_claim_when_no_gpu_is_available(monkeypatch):
    """`force='cuda'` on a CUDA-less box: `resolve` reports what was asked for.

    The authority cannot see the driver itself -- `_cuda_available` is patched
    False here -- so it hands the engine a claim plus the device that claim
    needs. Deciding not to believe it, loudly, is `separate.py`'s job; the test
    below covers that half.
    """
    monkeypatch.setattr(backends, "_hip_version", lambda: None)
    monkeypatch.setattr(backends, "_cuda_available", lambda: False)
    monkeypatch.setattr(backends, "_is_amd_device", lambda: False)
    b = backends.resolve(force="cuda")
    assert b.name == "cuda"
    assert b.torch_device == "cuda"
    assert b.ort_providers[0] == "CUDAExecutionProvider"


def test_force_rocm_asks_for_the_cuda_device_on_a_cpu_only_box(monkeypatch):
    monkeypatch.setattr(backends, "_hip_version", lambda: None)
    monkeypatch.setattr(backends, "_cuda_available", lambda: False)
    b = backends.resolve(force="rocm")
    assert b.name == "rocm"
    assert b.torch_device == "cuda"


# --------------------------------------------------------------------------- #
# Engine wiring: separate.SeperateAttributes is the authority's only consumer
# --------------------------------------------------------------------------- #

def _attributes(monkeypatch, tmp_path, *, is_gpu_conversion, backend,
                cuda_available=True, mps_available=False, device_set=None):
    """Build a real ``SeperateAttributes`` with the probes and the authority pinned.

    Frozen golden settings (``tests.golden.settings``) supply the ~50 ``ModelData``
    attributes, exactly as the harness does, so this exercises the shipped
    constructor rather than a re-implementation of it.
    """
    attrs = dict(settings.make("vr", "cpu"))
    attrs["is_gpu_conversion"] = is_gpu_conversion
    if device_set is not None:
        attrs["device_set"] = device_set
    process_data = runner.make_process_data(tmp_path / "input.wav", tmp_path / "out")

    monkeypatch.setattr(separate, "resolve_backend", lambda *a, **k: backend)
    monkeypatch.setattr(separate, "cuda_available", cuda_available)
    monkeypatch.setattr(separate, "mps_available", mps_available)
    return separate.SeperateAttributes(types.SimpleNamespace(**attrs), process_data)


def _backend_claims(cuda_available=True, providers=("DmlExecutionProvider", "CPUExecutionProvider")):
    return backends.Backend(name="rocm", torch_device="cuda", ort_providers=list(providers))


def test_no_gpu_mode_has_a_backend_but_keeps_cpu_providers(monkeypatch, tmp_path):
    """The user's explicit "GPU off" switch still wins -- but it is now a *visible*
    choice at one site, and self.backend exists so nothing downstream has to guess.

    Regression lock for the ruled-out half of finding 2: no-GPU mode must NOT be
    handed DirectML (which is what resolve(force="cpu") would have done).
    """
    backend = _backend_claims()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        attrs = _attributes(monkeypatch, tmp_path, is_gpu_conversion=-1, backend=backend)

    assert attrs.backend is backend
    assert str(attrs.device) == "cpu"
    assert attrs.run_type == ["CPUExecutionProvider"], (
        "no-GPU mode must stay on CPU providers; the authority is read but not applied"
    )
    assert _backend_warnings(caught) == []


def test_gpu_mode_honours_the_authority_without_warning(monkeypatch, tmp_path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        attrs = _attributes(monkeypatch, tmp_path, is_gpu_conversion=0, backend=_backend_claims())

    assert str(attrs.device) == "cuda"
    assert attrs.run_type == ["DmlExecutionProvider", "CPUExecutionProvider"]
    assert isinstance(attrs.run_type, list), "InferenceSession(providers=...) needs a list"
    assert attrs.is_other_gpu is False, "ROCm is not 'other gpu' -- see brief rule 1"
    assert _backend_warnings(caught) == []


def test_gpu_mode_keeps_the_device_set_index_suffix(monkeypatch, tmp_path):
    attrs = _attributes(monkeypatch, tmp_path, is_gpu_conversion=0, backend=_backend_claims(),
                        device_set="1")
    assert str(attrs.device) == "cuda:1"


def test_gpu_mode_takes_providers_from_the_authority_even_when_torch_is_cpu(monkeypatch, tmp_path):
    """CPU torch + AMD box in GPU mode: DirectML is still the right ONNX provider.

    Reading providers only inside the ``torch_device == 'cuda'`` branch left the
    hardcoded CPU default as a second source of provider truth.
    """
    cpu_backend = backends.Backend(name="cpu", torch_device="cpu",
                                   ort_providers=["DmlExecutionProvider", "CPUExecutionProvider"])
    attrs = _attributes(monkeypatch, tmp_path, is_gpu_conversion=0, backend=cpu_backend,
                        cuda_available=False)
    assert attrs.run_type == ["DmlExecutionProvider", "CPUExecutionProvider"]
    assert str(attrs.device) == "cpu"


def test_unhonoured_gpu_claim_warns_instead_of_silently_running_cpu(monkeypatch, tmp_path):
    """The missing ``else`` (finding 2): hip/cuda build, no driver -- or
    force='cuda'/'rocm' on a box without one. Device stays CPU, but not quietly,
    and self.backend.name no longer contradicts the run unnoticed."""
    with pytest.warns(RuntimeWarning, match="no GPU is available"):
        attrs = _attributes(monkeypatch, tmp_path, is_gpu_conversion=0,
                            backend=_backend_claims(), cuda_available=False)

    assert str(attrs.device) == "cpu"
    assert attrs.backend.name == "rocm"


def test_mps_mode_has_a_backend_and_is_left_alone(monkeypatch, tmp_path):
    """macOS stays out of this fork's scope (brief rule 3) but still gets
    self.backend, so the attribute exists in all three modes."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        attrs = _attributes(monkeypatch, tmp_path, is_gpu_conversion=0,
                            backend=_backend_claims(), mps_available=True)

    assert attrs.backend is not None
    assert attrs.device == "mps"
    assert attrs.is_other_gpu is True
    assert attrs.run_type == ["CPUExecutionProvider"]
    assert _backend_warnings(caught) == []

