<#
.SYNOPSIS
    One-command environment bootstrap for the UVR5 ROCm fork.

.DESCRIPTION
    Creates (or reuses) the Python 3.12 virtualenv at .\.venv, installs the torch
    trio for the requested backend, then the repo's requirements.txt and pytest.

    Backend notes, all verified against the live indexes on 2026-11-03:

    rocm  PyTorch's own rocm6.3 / 6.4 / 7.0 / 7.1 channels publish ZERO win_amd64
          torch wheels. The only Windows ROCm source is the AMD Radeon repo, and
          that page is a FLAT directory listing, not a PEP 503 simple index: pip
          with `--index-url` asks for <url>/torch/ and gets HTTP 404. `--find-links`
          plus an exact local-version pin is the form that works. The pin is also
          the safety net: `2.9.1+rocm7.2.1` exists only on the AMD index, so a
          silent fallback to a PyPI/download.pytorch.org CPU wheel is impossible,
          and the post-install HIP guard below fails loudly if one ever appears.
          The trio is installed BEFORE requirements.txt on purpose - onnx2pytorch,
          pytorch-lightning and diffq all depend on torch, so installing the
          requirements first pulls torch 2.15.0+cpu off PyPI.
          Expect ~1.95 GB of downloads: the torch wheel is 821 MB and it depends on
          the rocm[libraries]==7.2.1 meta package, which brings rocm-sdk-core
          (645 MB) and rocm-sdk-libraries-custom (490 MB) - that is the HIP/MIOpen
          runtime, so it is not optional and not a mistake.

    cuda  NVIDIA path; PyTorch's cu124 index is used as an extra index so the
          non-torch dependencies still resolve from PyPI.

    cpu   THE DEFAULT, and not "CPU only": Plain PyPI torch plus the
          onnxruntime-directml wheel that requirements.txt pins, so ONNX models
          (MDX) run on any DX12 GPU -- AMD, NVIDIA or Intel -- while torch models
          (VR, Demucs, roformer) run on the CPU. No HIP SDK, no 2 GB download, no
          GPU vendor check. Choose this unless you specifically want the torch
          models on an AMD GPU, which is what `rocm` buys.

    Requirements.txt deliberately contains no torch line (see its header comment),
    so the backend chosen here is the only torch in the environment. Step 4 now
    prints the onnxruntime provider list: `DmlExecutionProvider` present is what
    makes the default install a GPU install for the ONNX models.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File .\bootstrap.ps1 -Backend rocm
    .\bootstrap.ps1 -Backend rocm -Fresh
#>
[CmdletBinding()]
param(
    [ValidateSet("rocm", "cuda", "cpu")]
    [string]$Backend = "cpu",

    # Delete and recreate .\.venv instead of reusing it.
    [switch]$Fresh,

    # The AMD index currently ships exactly one Windows ROCm build, and it is cp312-only.
    [string]$RocmIndexUrl   = "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/",
    [string]$RocmTorch      = "torch==2.9.1+rocm7.2.1",
    [string]$RocmTorchVision = "torchvision==0.24.1+rocm7.2.1",
    [string]$RocmTorchAudio = "torchaudio==2.9.1+rocm7.2.1"
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$VenvDir = Join-Path $PSScriptRoot ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

# Every pip call goes through here so the "-m pip" prefix can never be lost: the
# venv's python.exe happily treats a bare "install" as a script filename and fails
# with "can't open file 'install'" instead of a pip error.
function Invoke-Pip {
    param([string[]]$PipArgs, [string]$Label)
    $Arguments = @("-m", "pip") + $PipArgs
    Write-Host (">> " + ($Arguments -join " ")) -ForegroundColor Cyan
    & $VenvPython @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "bootstrap step '$Label' failed with exit code $LASTEXITCODE"
    }
}

# --- 1. venv ---------------------------------------------------------------------
if ($Fresh -and (Test-Path -LiteralPath $VenvDir)) {
    Write-Host ">> removing existing venv (-Fresh)" -ForegroundColor Yellow
    Remove-Item -LiteralPath $VenvDir -Recurse -Force
}

if (-not (Test-Path -LiteralPath $VenvPython)) {
    Write-Host ">> creating Python 3.12 venv at $VenvDir" -ForegroundColor Cyan
    py -3.12 -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) {
        throw "venv creation failed (exit $LASTEXITCODE). ROCm wheels are cp312-only; " +
              "check 'py -0p' lists -V:3.12."
    }
} else {
    Write-Host ">> reusing existing venv at $VenvDir (use -Fresh to rebuild)" -ForegroundColor DarkGray
}

# Activate for the rest of this session so plain `python` is the venv's.
. (Join-Path $VenvDir "Scripts\Activate.ps1")

Invoke-Pip @("install", "--upgrade", "pip") "upgrade pip"

# --- 2. torch trio ---------------------------------------------------------------
switch ($Backend) {
    "rocm" {
        Invoke-Pip @("install", "--find-links", $RocmIndexUrl,
                     $RocmTorch, $RocmTorchVision, $RocmTorchAudio) "install ROCm torch trio"
    }
    "cuda" {
        Invoke-Pip @("install", "--extra-index-url", "https://download.pytorch.org/whl/cu124",
                     "torch", "torchvision", "torchaudio") "install CUDA torch trio"
    }
    "cpu" {
        Invoke-Pip @("install", "torch", "torchvision", "torchaudio") "install CPU torch trio"
    }
}

# --- 3. the rest of the stack ----------------------------------------------------
Invoke-Pip @("install", "-r", "requirements.txt") "install requirements.txt"
Invoke-Pip @("install", "pytest") "install pytest"

# pip check is advisory here: a broken pin is reported loudly but must not hide the
# version/HIP output below, which is the part that tells you whether you have a GPU.
& $VenvPython -m pip check
if ($LASTEXITCODE -ne 0) {
    Write-Warning "pip check reported broken requirements (exit $LASTEXITCODE) - see above."
}

# --- 4. prove the backend is real ------------------------------------------------
@'
import torch
import onnxruntime as ort
print("torch.__version__        :", torch.__version__)
print("torch.version.hip        :", torch.version.hip)
print("torch.version.cuda       :", torch.version.cuda)
print("torch.cuda.is_available  :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device name              :", torch.cuda.get_device_name(0))
    print("device properties        :", torch.cuda.get_device_properties(0))
providers = ort.get_available_providers()
print("onnxruntime providers    :", providers)
if "DmlExecutionProvider" in providers:
    print("ONNX route               : DirectML (any DX12 GPU)")
    if not torch.cuda.is_available():
        print("torch route              : CPU -- use -Backend rocm to put VR/Demucs on an AMD GPU")
else:
    print("ONNX route               : CPU -- onnxruntime-directml is missing from this venv")
'@ | Set-Content -LiteralPath (Join-Path $env:TEMP "uvr_bootstrap_torch_probe.py") -Encoding UTF8

& $VenvPython (Join-Path $env:TEMP "uvr_bootstrap_torch_probe.py")
if ($LASTEXITCODE -ne 0) { throw "torch import/probe failed (exit $LASTEXITCODE)" }

if ($Backend -eq "rocm") {
    $hip = (& $VenvPython -c "import torch;print(torch.version.hip or '')").Trim()
    if ($LASTEXITCODE -ne 0 -or -not $hip) {
        throw ("torch.version.hip is empty - this is a CPU wheel, not ROCm. The AMD index " +
               "was skipped (check the --find-links URL and that Python is 3.12). Do NOT " +
               "point this at download.pytorch.org: that silently defeats the fork.")
    }
    # Test the exit code, not the captured text: a healthy `sys.exit(0)` prints
    # nothing, and `-not ""` would warn on a GPU that is perfectly visible.
    & $VenvPython -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)"
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "HIP runtime present but no GPU visible to torch - check the AMD driver and \\Device\\hip_0."
    }
}

Write-Host "done: $Backend" -ForegroundColor Green
