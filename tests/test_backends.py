"""Task 6: `backends.py` is the single device/provider authority.

Two halves. The first pins the *decision table* -- every probe is monkeypatched,
so what these tests check is what `resolve()` concludes from (hip, cuda, force)
and what onnxruntime reports, not the real torch/ORT hardware. The second half
pins the *consumer*: `separate.SeperateAttributes` must read the authority in
every mode (no-GPU, GPU, mps), take its providers from it, and warn when a
resolved GPU claim cannot be honoured.

The default install is CPU torch + `onnxruntime-directml`, so the decision this
file protects is: a machine with no torch GPU still puts its ONNX models on
DirectML. That used to be gated on a WMI scan for AMD's PCI vendor id; the gate
is gone (DirectML serves any DX12 GPU), and `test_directml_needs_no_vendor_scan`
is the regression lock for the gate not coming back.

`_no_real_probing` is autouse: `resolve()` must never import onnxruntime during
the unit tests. `test_resolve_probes_at_most_once` installs its own counting
probe -- that is intentional and it is the only test that must.
"""
import types
import warnings

import pytest

import backends
import separate
from tests.golden import runner, settings


@pytest.fixture(autouse=True)
def _no_real_probing(monkeypatch):
    """resolve() must never import onnxruntime in tests."""
    monkeypatch.setattr(backends, "_available_ort", lambda: {"DmlExecutionProvider",
                                                             "CUDAExecutionProvider",
                                                             "CPUExecutionProvider"})


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
    b = backends.resolve()
    assert b.name == "cuda"
    assert b.ort_providers[0] == "CUDAExecutionProvider"


def test_cpu_torch_still_gets_directml_for_onnx(monkeypatch):
    """The shipped default: CPU torch, ONNX on the DX12 GPU.

    This is the assertion the whole DML+CPU install rests on -- a user who never
    installs the HIP SDK must still get MDX on the GPU rather than on the CPU.
    """
    monkeypatch.setattr(backends, "_hip_version", lambda: None)
    monkeypatch.setattr(backends, "_cuda_available", lambda: False)
    b = backends.resolve()
    assert b.name == "cpu"
    assert b.torch_device == "cpu"
    assert b.ort_providers[0] == "DmlExecutionProvider"


def test_cpu_is_the_answer_only_when_onnxruntime_has_no_directml(monkeypatch):
    monkeypatch.setattr(backends, "_hip_version", lambda: None)
    monkeypatch.setattr(backends, "_cuda_available", lambda: False)
    monkeypatch.setattr(backends, "_available_ort",
                        lambda: frozenset({"CPUExecutionProvider"}))
    b = backends.resolve()
    assert b.torch_device == "cpu"
    assert b.ort_providers == ["CPUExecutionProvider"]


def test_directml_needs_no_vendor_scan(monkeypatch):
    """Regression lock: the WMI vendor probe is gone.

    It answered "is this an AMD card?" and cost ~0.5 s plus three ways to fail
    (no powershell, WMI timeout, access denied), while DirectML runs on any DX12
    GPU. If a future change reintroduces a vendor gate, resolve() no longer
    selects DirectML on a plain CPU-torch box and this test fails. Asserting the
    *symbol* is gone is the second half: a re-added probe whose answer nobody
    reads would still make the module shell out.
    """
    assert not hasattr(backends, "_is_amd_device")
    monkeypatch.setattr(backends, "_hip_version", lambda: None)
    monkeypatch.setattr(backends, "_cuda_available", lambda: False)
    assert backends.resolve().ort_providers[0] == "DmlExecutionProvider"


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


def test_force_dml_puts_onnx_on_directml_and_torch_on_cpu(monkeypatch):
    """The no-new-dependency fallback: DirectML accelerates the ONNX models on any DX12
    GPU while torch (demucs/apollo) stays on CPU, so a machine without ROCm is usable."""
    monkeypatch.setattr(backends, "_hip_version", lambda: "7.2.42141")
    monkeypatch.setattr(backends, "_cuda_available", lambda: True)
    b = backends.resolve(force="dml")
    assert b.name == "dml"
    assert b.torch_device == "cpu"
    assert b.ort_providers[0] == "DmlExecutionProvider"


def test_force_dml_without_the_provider_falls_back_to_cpu_and_says_so(monkeypatch):
    monkeypatch.setattr(backends, "_available_ort",
                        lambda: frozenset({"CPUExecutionProvider"}))
    with pytest.warns(RuntimeWarning):
        b = backends.resolve(force="dml")
    assert b.name == "dml"
    assert b.torch_device == "cpu"
    assert b.ort_providers == ["CPUExecutionProvider"]


def test_force_rejects_unknown_backend():
    with pytest.raises(ValueError, match="expected rocm|cuda|cpu"):
        backends.resolve(force="opencl")


def test_resolve_probes_at_most_once(monkeypatch):
    """SeperateAttributes is constructed once per model per stem, so resolve()
    runs many times in a single run. ORT provider enumeration is not free and must
    be memoised; the memo has to survive the probe symbol itself being replaced
    (``monkeypatch.setattr`` swaps the module attribute, so an ``@lru_cache`` on
    the function object would be bypassed and the counter would hit 5).
    """
    calls = {"ort": 0}

    def counting_ort():
        calls["ort"] += 1
        return {"DmlExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"}

    monkeypatch.setattr(backends, "_available_ort", counting_ort)
    for _ in range(5):
        backends.resolve()
    assert calls == {"ort": 1}, calls


def test_degradation_warning_survives_the_engines_ambient_ignore_filter():
    """Loud, not merely present in the source.

    ``separate.py`` runs ``warnings.filterwarnings("ignore")`` at import -- the
    head of ``warnings.filters`` becomes ``('ignore', None, Warning, None, 0)`` --
    so a plain ``warnings.warn`` from the device layer would still be "no message
    anywhere" in the shipped product. `backends.warn_degradation` tightens the
    filter only inside its own ``catch_warnings`` to escape that, and de-dupes by
    message so a per-model-per-stem run says it once rather than eight times.
    """
    message = "probe text unique to this test"

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        with pytest.warns(RuntimeWarning) as record:
            backends.warn_degradation(message)
        assert message in str(record[0].message)

        with warnings.catch_warnings(record=True) as again:
            warnings.simplefilter("always")
            backends.warn_degradation(message)
        assert again == [], [str(w.message) for w in again]


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

