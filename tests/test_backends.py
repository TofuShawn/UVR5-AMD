"""Task 6: `backends.py` is the single device/provider authority.

Every probe is monkeypatched here, so what these tests actually pin is the
*decision table* -- given (hip, cuda, amd-vendor, force), which Backend comes
back -- not the real WMI/torch/ORT hardware. Step 7 of the brief is the one
place the hardware itself is asserted.

`_no_real_probing` is autouse: `resolve()` must never shell out to WMI or import
onnxruntime during the unit tests. `test_resolve_probes_at_most_once` overrides
`_is_amd_device` again with its own counting probe -- that is intentional and it
is the only test that must.
"""
import pytest

import backends


@pytest.fixture(autouse=True)
def _no_real_probing(monkeypatch):
    """resolve() must never shell out to WMI or import onnxruntime in tests."""
    monkeypatch.setattr(backends, "_available_ort", lambda: {"DmlExecutionProvider",
                                                             "CUDAExecutionProvider",
                                                             "CPUExecutionProvider"})
    monkeypatch.setattr(backends, "_is_amd_device", lambda: True)


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
    probe callable, not on the module attribute -- see backends._memo.
    """
    calls = {"wmi": 0}

    def counting_amd():
        calls["wmi"] += 1
        return True

    monkeypatch.setattr(backends, "_is_amd_device", counting_amd)
    for _ in range(5):
        backends.resolve()
    assert calls["wmi"] <= 1
