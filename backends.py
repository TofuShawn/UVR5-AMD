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
import sys
import warnings
from dataclasses import dataclass

_ROCM = "rocm"
_CUDA = "cuda"
_CPU = "cpu"
_VALID_FORCE = (_ROCM, _CUDA, _CPU)
_AMD_PCI_VENDOR = "VEN_1002"          # PCI vendor id 0x1002 == AMD


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


def _probe_failed(detail: str) -> None:
    """Say out loud that DirectML is being given up, and why.

    The fallback itself stays conservative (CPU is always correct, just slow),
    but it is no longer silent: a ROCm run computing MDX-C on CPU is the exact
    failure this fork exists to kill, and it is invisible to the golden suite,
    which covers ``vr`` and ``htdemucs`` only.
    """
    warn_degradation(
        "Could not read the display-device list, so this run cannot tell whether "
        f"the machine has an AMD GPU ({detail}). ONNX models will fall back to "
        "CPUExecutionProvider instead of DirectML -- correct, but far slower, and "
        "the slowdown is not the GPU's fault. Check that powershell and the WMI "
        "service (Winmgmt) are available."
    )


def _is_amd_device() -> bool:
    """True when some display device reports AMD's PCI vendor id.

    This answer only ever chooses the ONNX provider (see ``resolve``), so a wrong
    ``False`` costs speed, not correctness -- which is exactly why each way it can
    fail is reported instead of folded into "not an AMD box".
    """
    if sys.platform != "win32":
        # The WMI scan below is a Windows-only probe, so the platform is tested
        # here rather than discovering non-Windows through the FileNotFoundError
        # that ``powershell`` would raise on macOS/Linux -- a missing interpreter
        # is not the same event as a locked-down WMI. Nothing to warn about:
        # DirectML does not exist off Windows, so CPU is the right answer.
        return False

    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty PNPDeviceID"],
            capture_output=True, text=True, timeout=10)
    except Exception as exc:
        # FileNotFoundError (no powershell), subprocess.TimeoutExpired (WMI hung),
        # OSError/permission errors -- all three used to return False silently.
        _probe_failed(f"{type(exc).__name__}: {exc}")
        return False

    if result.returncode != 0:
        # powershell started but WMI refused it (access denied, Winmgmt stopped).
        # stdout is empty here, so without this branch the machine is mislabelled
        # as "no AMD GPU" rather than "GPU identity unknown".
        stderr = (result.stderr or "").strip()
        _probe_failed(f"powershell exit {result.returncode}: {stderr[:200] or '<no stderr>'}")
        return False

    return _AMD_PCI_VENDOR in result.stdout


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
