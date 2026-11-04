"""backends.py -- the single device/provider authority.

Scope by design: this module resolves *which torch device* and *which ONNX
Runtime execution providers* a run should use, and nothing else. Stem selection,
cropping and inference stay in ``separate.py``.

The one non-obvious rule (the plan's central premise): AMD's ROCm torch wheel
presents itself as CUDA -- ``torch.cuda.is_available()`` is True and the device
type is ``"cuda"``. ``torch.version.hip`` is the *only* discriminator. So "rocm"
is decided by the torch build, never by the PCI vendor, and the vendor scan is
allowed to choose only the ONNX provider (DirectML), which also serves an AMD
iGPU sitting behind an NVIDIA dGPU.
"""
from __future__ import annotations

import functools
import subprocess
from dataclasses import dataclass

_ROCM = "rocm"
_CUDA = "cuda"
_CPU = "cpu"
_VALID_FORCE = (_ROCM, _CUDA, _CPU)


@dataclass(frozen=True)
class Backend:
    name: str
    torch_device: str
    ort_providers: list[str]


def _hip_version():
    import torch
    return getattr(torch.version, "hip", None)


def _cuda_available() -> bool:
    import torch
    return torch.cuda.is_available()


def _is_amd_device() -> bool:
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty PNPDeviceID"],
            capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return False
    return "VEN_1002" in out          # PCI vendor id 0x1002 == AMD


def _available_ort():
    import onnxruntime as ort
    return frozenset(ort.get_available_providers())


@functools.lru_cache(maxsize=None)
def _once(probe):
    """Memoise an environment probe by the *callable*, not by a module name.

    ``_is_amd_device`` shells out to WMI (~0.5 s) and ``_available_ort`` walks
    the ORT provider registry; neither changes for the life of the process, but
    ``SeperateAttributes`` (and therefore ``resolve``) is constructed once per
    model per stem, so they must be probed once.

    The memo lives here rather than as an ``@lru_cache`` directly on each probe,
    because ``monkeypatch.setattr(backends, "_is_amd_device", fn)`` in the tests
    replaces the *module attribute* -- the decorator on the old function object
    is left behind and the swap silently defeats it (a defect found by running
    the brief's own reference implementation against its own tests). Keying on
    the callable identity means: production sees the single stable module-level
    function and probes it once; each test's freshly-injected probe is a distinct
    object, so no value is ever reused across tests that disagree.
    """
    return probe()


def resolve(force: str | None = None) -> Backend:
    if force is not None and force not in _VALID_FORCE:
        raise ValueError(f"unknown backend {force!r}; expected rocm|cuda|cpu")

    # ROCm speaks the CUDA API, so a hip build IS the cuda device. Therefore
    # "rocm" is decided by the torch build, never by the PCI vendor: an AMD GPU
    # driving an NVIDIA torch build falls back to cpu and says so. The vendor
    # scan is only ever allowed to choose the ONNX provider.
    if force:
        name = force
    else:
        name = _ROCM if _hip_version() else (_CUDA if _cuda_available() else _CPU)

    torch_device = _CPU if name == _CPU else _CUDA

    avail = _once(_available_ort)
    if _once(_is_amd_device) and "DmlExecutionProvider" in avail:
        providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
    elif name == _CUDA and "CUDAExecutionProvider" in avail:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    return Backend(name=name, torch_device=torch_device, ort_providers=providers)
