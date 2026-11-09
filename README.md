# UVR5 AMD/ROCm — PySide6 GUI

A fork of Ultimate Vocal Remover 5 with a new Fluent-Qt (PySide6) front end and a
Windows AMD GPU path. The separation engine is UVR5's: the same models and the same
architectures (MDX-Net, VR, Demucs, RoFormer). The tkinter GUI in `UVR.py` is
retired to a header whose `__main__` refuses.

Latest build: [v5.6.1-amd.1](https://github.com/TofuShawn/UVR5-AMD/releases/tag/v5.6.1-amd.1)
— source zip, installed with `bootstrap.ps1` (below); upstream's own README is in
[README.upstream.md](README.upstream.md).

## Hardware

| GPU | What runs where |
| --- | --- |
| **RDNA 4** — Radeon RX 9000 (gfx120x) | ROCm: MDX-Net plus VR/Demucs/RoFormer on the GPU |
| **RDNA 3** — Radeon RX 7000 (gfx110x) | ROCm: MDX-Net plus VR/Demucs/RoFormer on the GPU |
| RDNA 2 — RX 6000 and older | No ROCm on Windows: MDX-Net on DirectML, VR/Demucs/RoFormer on CPU |
| NVIDIA | CUDA torch (optional): ONNX on the CUDA provider, torch models on CUDA |
| Intel / any DX12 GPU | MDX-Net on DirectML only; torch models on CPU |

Windows ROCm supports **RDNA 3 and RDNA 4 only** — AMD's system-requirements page
lists no RDNA 1/2 consumer GPUs. Platform: **Windows 10/11, Python 3.12** (the
Windows ROCm wheels are `cp312`-only).

## Install

Default — CPU torch + DirectML, no GPU SDK required:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\bootstrap.ps1
```

Performance pack — AMD's Windows ROCm stack, ~1.95 GB:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\bootstrap.ps1 -Backend rocm
```

`bootstrap.ps1` creates `.venv` with Python 3.12, installs the torch trio for the
chosen backend first (so nothing pulls a different torch later), then
`requirements.txt`, then prints the torch and onnxruntime providers.

## Run

`bootstrap.ps1` writes `start.bat` as its **last** step, and only when its checks
passed — a failed install leaves no launcher behind. Double-click it; after the app
exits it asks whether to delete itself. Or run the commands directly:

```powershell
.venv\Scripts\python.exe -m app.main              # the GUI
.venv\Scripts\python.exe separate_cli.py --help   # the GUI-free CLI
```

From the repository root: `-m` puts the engine modules on `sys.path`.

## Models

Weights are **not** shipped. `uvr_paths.py` resolves the model root from
`UVR_MODEL_ROOT`, falling back to `REPO/models`; there is no Settings field and no
GUI picker, and the app treats the root as **read-only**:

```powershell
$env:UVR_MODEL_ROOT = "C:\Users\you\UVR\models"
.venv\Scripts\python.exe -m app.main
```

The GUI does not download models yet. Nothing phones home: the model tables refresh
only from `separate_cli.py --refresh-model-tables --confirm`, and the settings
page's refresh button deliberately refuses (it passes `confirmed=False`), so the
click prints the refusal and touches nothing.

## What the window does

- Two-segment work page: **Input** (file, batch add files/folder, queue, output
  folder, default-folder toggle) and **Model & output** (architecture, model,
  format radios), plus a Device group and a model-parameters expander.
- Device switch: **Auto / ROCm / DirectML / CPU**; picking DirectML or ROCm turns
  the GPU path on, CPU turns it off.
- Batch queue with continue-on-error: a failed file is named and the run moves on.
- Run-output list for the stems this session wrote: **Open**, **Show in folder**,
  **Delete** (confirmed; a refused unlink keeps the file and its row).
- Full-page settings (50 engine fields, 33 selections), English / 繁體中文 /
  简体中文, bundled Inter / JetBrains Mono / Noto Sans SC.

## Measured

Conditions for every number: **RX 7900 XTX (gfx1100)**, Windows, `torch
2.9.1+rocm7.2.1`, golden 30 s clip, **warm** (a process's first GPU run is excluded:
MIOpen's cold search is ~20× slower).

| Check | Result |
| --- | --- |
| VR `1_HP-UVR`, warm | 1.61–1.66 s |
| `htdemucs`, warm | 2.93–3.04 s |
| CPU vs ROCm, same input | SNR ≥ 40 dB (golden gate) |
| Suite | 804 passed, 1 skipped (2026-11-09) |

The timings come from `tools/task7_timing.py`: n=4 GPU runs per session, ~2%
within a session and 12.6% across sessions. The gate's audio is a committed CC0
excerpt (`tests/fixtures/`, provenance beside it), not a commercial recording.

## Testing

`tests/golden/pins.json` asserts `torch` / `torch_hip` against the live environment,
so the suite passes **only in a `-Backend rocm` venv** — by design, not by accident.
The GUI and the CLI do not care which venv they run in.

## Known limitations

- No installer: a source zip plus `bootstrap.ps1`, because the ROCm stack is too
  large and fragile to freeze.
- No auto-update; the update channel is deleted, not disabled.
- In-app model download exists as a module but is not wired into the GUI.
- Upstream ships no `LICENSE` file; the fork's licence is MIT, stated here rather
  than by a file that does not exist.

## Licence

MIT, following upstream Ultimate Vocal Remover.
