"""backends.py -- the single device/provider authority.

Scope by design: this module resolves *which torch device* and *which ONNX
Runtime execution providers* a run should use, and nothing else. Stem selection,
cropping and inference stay in ``separate.py``.

The one non-obvious rule (the plan's central premise): AMD's ROCm torch wheel
presents itself as CUDA -- ``torch.cuda.is_available()`` is True and the device
type is ``"cuda"``. ``torch.version.hip`` is the *only* discriminator. So "rocm"
is decided by the torch build, never by the PCI vendor.

DirectML is the default accelerator for ONNX models and it is *vendor-agnostic*:
it runs on any DX12 GPU (AMD, NVIDIA, Intel), so it is chosen whenever the torch
build is not already driving a CUDA/iGPU device. That is what makes the shipped
default -- CPU torch + ``onnxruntime-directml`` -- usable on a machine with no
HIP SDK at all: MDX runs on the GPU, VR/Demucs run on the CPU. The earlier WMI
vendor scan existed only to keep DirectML on AMD cards; it cost ~0.5 s per
process and could fail (no powershell, WMI hung, access denied), and it answered
a question the decision no longer asks.
"""
from __future__ import annotations

import functools
import warnings
from dataclasses import dataclass

_ROCM = "rocm"
_CUDA = "cuda"
_CPU = "cpu"
_DML = "dml"
_VALID_FORCE = (_ROCM, _CUDA, _CPU, _DML)


@dataclass(frozen=True)
class Backend:
    name: str
    torch_device: str
    ort_providers: list[str]


#: Distinct degradation notices already shown. One per message per process:
#: ``SeperateAttributes`` (and so ``resolve``) is constructed once per model per
#: stem, and an ensemble run must not print the same complaint eight times.
_warned: set[str] = set()


def warn_degradation(message: str) -> None:
    """Emit ``message`` as a RuntimeWarning that an ambient filter cannot swallow.

    ``warnings.warn`` is the engine-pure channel (this module may not import
    tkinter/``gui_data`` or write to ``process_data['write_to_console']``), but it
    is not enough on its own: ``separate.py`` runs
    ``warnings.filterwarnings("ignore")`` at import time -- observed as
    ``('ignore', None, Warning, None, 0)`` at the head of ``warnings.filters`` --
    so a plain warning from the device layer reproduces the very "no message
    anywhere" these notices exist to remove. The filter is tightened inside
    ``catch_warnings`` only, which restores the process's filter list afterwards:
    nothing else in the application changes behaviour.
    """
    if message in _warned:
        return
    _warned.add(message)
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.warn(message, RuntimeWarning, stacklevel=3)


def _hip_version():
    import torch
    return getattr(torch.version, "hip", None)


def _cuda_available() -> bool:
    import torch
    return torch.cuda.is_available()


def _available_ort():
    import onnxruntime as ort
    return frozenset(ort.get_available_providers())


@functools.lru_cache(maxsize=None)
def _once(probe):
    """Memoise an environment probe by the *callable*, not by a module name.

    ``_available_ort`` walks the ORT provider registry; it does not change for the
    life of the process, but ``SeperateAttributes`` (and therefore ``resolve``) is
    constructed once per model per stem, so it must be probed once.

    The memo lives here rather than as an ``@lru_cache`` directly on each probe,
    because ``monkeypatch.setattr`` in the tests replaces the *module attribute*
    -- the decorator on the old function object is left behind and the swap
    silently defeats it. Keying on the callable identity means: production sees
    the single stable module-level function and probes it once; each test's
    freshly-injected probe is a distinct object, so no value is ever reused
    across tests that disagree.
    """
    return probe()


def resolve(force: str | None = None) -> Backend:
    if force is not None and force not in _VALID_FORCE:
        raise ValueError(f"unknown backend {force!r}; expected rocm|cuda|cpu|dml")

    # ROCm speaks the CUDA API, so a hip build IS the cuda device. Therefore
    # "rocm" is decided by the torch build, never by the PCI vendor: an AMD GPU
    # driving a NVIDIA torch build falls back to cpu and says so.
    if force:
        name = force
    else:
        name = _ROCM if _hip_version() else (_CUDA if _cuda_available() else _CPU)

    # DirectML is an ONNX-only accelerator here (no torch-directml dependency): torch
    # stays on CPU while the ONNX models run on the DX12 GPU. That is the default
    # install's route, and it is the same on AMD, NVIDIA and Intel.
    torch_device = _CPU if name in (_CPU, _DML) else _CUDA

    avail = _once(_available_ort)
    if name == _DML:
        if "DmlExecutionProvider" in avail:
            providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        else:
            warn_degradation(
                "DirectML was requested but onnxruntime reports no "
                "DmlExecutionProvider (install onnxruntime-directml on Windows). "
                "ONNX models fall back to CPUExecutionProvider.")
            providers = ["CPUExecutionProvider"]
    elif name == _CUDA and "CUDAExecutionProvider" in avail:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif "DmlExecutionProvider" in avail:
        # CPU torch (the shipped default) and ROCm torch both land here: Windows has no
        # ROCm onnxruntime, and DirectML serves any DX12 GPU, so leaving ONNX on CPU
        # would be a free 10x left on the floor for every MDX run.
        providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    return Backend(name=name, torch_device=torch_device, ort_providers=providers)
