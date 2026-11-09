# UVR5 AMD/ROCm fork — read this before your first run

This branch (`fork/amd-rocm-pyside`) runs UVR's engine on an AMD GPU through torch's ROCm build,
behind a Fluent-Qt (PySide6) window or a GUI-free CLI. The tkinter GUI is retired to a 389-line
header in `UVR.py` whose first statement refuses; `THIRD_PARTY_NOTICES.md` names what ships, and
§License below carries this fork's own licence — MIT, with the `LICENSE` file still to be added.

## 1. Install: DirectML + CPU by default, ROCm as the optional performance pack

The default install needs no GPU SDK. `bootstrap.ps1` defaults to `-Backend cpu`, which is
CPU torch plus the `onnxruntime-directml` wheel `requirements.txt` pins; the app then runs
the ONNX models (MDX) on any DX12 GPU — AMD, NVIDIA or Intel — and the torch models (VR,
Demucs, roformer) on the CPU. `backends.resolve()` needs no vendor check for this: DirectML
is chosen whenever no CUDA/iGPU torch device is driving ONNX.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\bootstrap.ps1   # == -Backend cpu
.venv\Scripts\python.exe -m app.main
```

**DirectML** in the work page's device box (or `--device dml` on the CLI) turns the GPU path
on for the ONNX models; **Auto** leaves the Use GPU switch alone; **CPU** turns it off. The
installer's last step prints the onnxruntime provider list, so "DmlExecutionProvider" there
is the proof that the default install is a GPU install.

`bootstrap.ps1 -Backend rocm` is the performance pack: ~1.95 GB of ROCm torch, and the only
route that puts the **torch** models on an AMD GPU. The suite runs only in that venv —
`tests/golden/pins.json` asserts `torch`/`torch_hip` against the live environment, so a
`-Backend cpu` venv fails it **by design** rather than skipping its GPU-resident test
(ruling R-T7-vram). The GUI and the CLI themselves do not care which venv they are in.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\bootstrap.ps1 -Backend rocm
.venv\Scripts\python.exe -m pytest -q   # 3 min 40 s here (RX 7900 XTX, warm MIOpen cache)
```

## 2. Running it

```powershell
.venv\Scripts\python.exe -m app.main              # the GUI
.venv\Scripts\python.exe separate_cli.py --help   # the GUI-free CLI
```

`-m app.main` from the repository root: `-m` puts the root — where the engine modules live — on
`sys.path`, and `app/main.py` no longer patches that. In the CLI, omitting `--device` means CPU
(`Settings.gpu_conversion` defaults off) and every run prints one `[cli] compute device:` line.

**Batch and the default output folder are the GUI's.** **Add files…** / **Add folder…** queue
several files and runs them one after another — a file the engine fails on is named in the console
and the run moves on, and **STOP** stops the rest. Ticking **Use this as my default output folder**
stores it under `ui.export_path` in `settings.json`, which is the fallback when the Output box is
empty; the CLI keeps its own rule (`UVR_out` beside the input) and never reads that block.
**Output format** and **Use GPU** are on the work page too, and each writes its one `Settings`
field the moment it is changed — the page is two labelled segments now, not one screenful.

**Where the weights come from.** The model root is `uvr_paths.py:13`, which defaults to
`REPO/models` and is overridden **only** by the `UVR_MODEL_ROOT` environment variable — there is
no Settings/Selections field for it and no GUI picker. A fresh `bootstrap.ps1 -Backend rocm` user
who launches `python -m app.main` with nothing set therefore sees the small set that ships *inside
this repo tree* (the VR `UVR-DeNoise-Lite` weight and `lib_v5/mixer.ckpt` are git-tracked), not
their existing library. To point the fork at an install's model library — the fork's central
"no silent migration" promise is that it never moves your files for you — set the variable to the
**Windows** path before launching:

```powershell
$env:UVR_MODEL_ROOT = "C:\Users\you\UVR\models"   # the read-only library you already have
.venv\Scripts\python.exe -m app.main
```

This fork ships **no weight downloader**: nothing on any code path fetches a `.pth`/`.onnx`/`.th` —
you place the file yourself and, if it is a name the baked tables do not know, supply its params as
§3 describes. The whole app is bound to a **Windows-only, Python 3.12** interpreter: the only ROCm
torch wheel is AMD's `cp312` Windows build (`bootstrap.ps1` pins `py -3.12`), so neither the GUI nor
the suite is reachable on another OS or Python version.

## 3. A GUI-free run cannot open the params dialog: an unknown model is a hard error

Upstream resolves params from the per-model `<hash>.json`, then a baked table, then **asks a human
through a popup**. A CLI has no popup, so that case exits non-zero (`3`) naming the hash and the exact
`<hash>.json` it wanted — the only self-service path: write that file
(`Catalog.write_params("vr", "<hash>", params)`, or by hand) into `%APPDATA%\UVR-fork\model_data`,
which the catalog reads before the baked tables. Weight and model trees are **read-only**.

## 4. No auto-update; one command touches the network

Nothing phones home on any code path. The four baked model tables refresh only when asked, and
`separate_cli.py --refresh-model-tables --confirm` is the one command here that reaches the
network; without `--confirm` it refuses, exits non-zero and touches nothing. The tables shipped
in this repo are staler than a live install's, so this is what makes newer model names resolvable.

**Say plainly what the GUI's button does not do.** The settings page's "Refresh model tables"
button — this fork's only network-facing control in the GUI — cannot fetch: it hands the refresh
`confirmed=False`, which `model_refresh.refresh` refuses before any request, so the click prints
that refusal in the status box and touches nothing. A refresh that reaches the network runs from
the CLI with `--confirm`, as above. This is a known deviation from spec §10, which lists the
button as in scope; the confirm step was deliberately left unbuilt rather than shipping a GUI
control that phones home on a single click.

## 5. Speed numbers carry their conditions

Conditions on every figure: **RX 7900 XTX (gfx1100)**, Windows, `torch 2.9.1+rocm7.2.1`
(`hip 7.2.53211-158bd99533`), golden 30 s crop, frozen `pins.json` settings (VR `1_HP-UVR`), **warm**
(a process's first GPU run excluded: MIOpen's cold search is ~20× slower); committed driver
`tools/task7_timing.py`, which prints every number below.

* Warm GPU, absolute — driver `tools/task7_timing.py`: VR **1.67–1.88 s**, htdemucs
  **2.74 s** (n=4 GPU runs per session; ~2% within a session, 12.6% across it, so ~2%
  is never a cross-session claim).
* Healthy-CPU ratio — driver `tools/task7_timing.py`, published only as a **range with
  its n**: **5.24–5.90×** (VR, n=3 sessions), **4.05×** (htdemucs, n=2); its divisor
  (`torch 2.15.0+cpu`, measured elsewhere) is not reproducible from this repository, so
  the range's top may never stand alone — a test asserts that.
* The arches fall into three tiers, and only the first is numerically verified here.
  **Golden-covered** (`vr`, `htdemucs`): SNR ≥ 40 dB CPU-vs-ROCm measured on this branch.
  **Blocked-by-table** (`mdx_c`, `onnx`, `roformer`): refused as `BlockedArchError` (exit 4),
  never simulated. **Selectable-but-unverified**: `mdx` is in `SUPPORTED_ARCHS`, is **not** in
  `BLOCKED_ARCHS`, and has a real 66,759,214-byte weight installed here
  (`MDX_Net_Models/UVR-MDX-NET-Inst_HQ_3.onnx`), so it *runs* — it simply has no golden SNR
  figure on this branch; `apollo` is offered by `archs()` yet lists **zero** names, so it
  selects nothing. Demucs v1/v2 are **not catalog arches at all**: naming one is
  `UnknownModelName` (exit 6), not a blocked (exit 4) refusal.

The upstream v5.6 document follows, kept intact but for §License; it describes the **official** app.

---

# Ultimate Vocal Remover GUI v5.6
<img src="https://raw.githubusercontent.com/Anjok07/ultimatevocalremovergui/master/gui_data/img/UVR_v5.6.png?raw=true" />

[![Release](https://img.shields.io/github/release/anjok07/ultimatevocalremovergui.svg)](https://github.com/anjok07/ultimatevocalremovergui/releases/latest)
[![Downloads](https://img.shields.io/github/downloads/anjok07/ultimatevocalremovergui/total.svg)](https://github.com/anjok07/ultimatevocalremovergui/releases)

## About

This application uses state-of-the-art source separation models to remove vocals from audio files. UVR's core developers trained all of the models provided in this package (except for the Demucs v3 and v4 4-stem models).

- **Core Developers**
    - [Anjok07](https://github.com/anjok07)
    - [aufr33](https://github.com/aufr33)

- **Support the Project**
    - [Donate](https://www.buymeacoffee.com/uvr5)

## Installation

These bundles contain the UVR interface, Python, PyTorch, and other dependencies needed to run the application effectively. No prerequisites are required.

### Windows Installation

- Please Note:
    - This installer is intended for those running Windows 10 or higher. 
    - Application functionality for systems running Windows 7 or lower is not guaranteed.
    - Application functionality for Intel Pentium & Celeron CPUs systems is not guaranteed.
    - You must install UVR to the main C:\ drive. Installing UVR to a secondary drive will cause instability.

- Download the UVR installer for Windows via the link below:
    - [Main Download Link](https://github.com/Anjok07/ultimatevocalremovergui/releases/download/v5.6/UVR_v5.6.0_setup.exe)
    - [Main Download Link mirror](https://www.mediafire.com/file_premium/jiatpgp0ljou52p/UVR_v5.6.0_setup.exe/file)
- If you use an **AMD Radeon or Intel Arc graphics card**, you can try the OpenCL version:
    - [OpenCL Version - Main Download Link](https://github.com/Anjok07/ultimatevocalremovergui/releases/download/v5.6/UVR_v5.6.0_setup_opencl.exe)
- Update Package instructions for those who have UVR already installed:
    - If you already have UVR installed you can install this package over it or download it straight from the application or [click here for the patch](https://github.com/Anjok07/ultimatevocalremovergui/releases/download/v5.6/UVR_Patch_10_6_23_4_27.exe).

<details id="WindowsManual">
  <summary>Windows Manual Installation</summary>

### Manual Windows Installation

- Download and extract the repository [here](https://github.com/Anjok07/ultimatevocalremovergui/archive/refs/heads/master.zip)
- Download and install Python [here](https://www.python.org/ftp/python/3.9.8/python-3.9.8-amd64.exe)
   - Make sure to check "Add python.exe to PATH" during the install
- Run the following commands from the extracted repo directory:

```
python.exe -m pip install -r requirements.txt
```

If you have a compatible Nvidia GPU, run the following command:

```
python.exe -m pip install --upgrade torch --extra-index-url https://download.pytorch.org/whl/cu117
```

If you do not have FFmpeg or Rubber Band installed and want to avoid going through the process of installing them the long way, follow the instructions below.

**FFmpeg Installation**

- Download the precompiled build [here](https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip)
- From the archive, extract the following file to the UVR application directory:
   - ```ffmpeg-5.1.2-essentials_build/bin/ffmpeg.exe```

**Rubber Band Installation**

In order to use the Time Stretch or Change Pitch tool, you'll need Rubber Band.

- Download the precompiled build [here](https://breakfastquay.com/files/releases/rubberband-3.1.2-gpl-executable-windows.zip)
- From the archive, extract the following files to the UVR application directory:
   - ```rubberband-3.1.2-gpl-executable-windows/rubberband.exe```
   - ```rubberband-3.1.2-gpl-executable-windows/sndfile.dll```

</details>

### MacOS Installation
- Please Note:
    - The MacOS Sonoma mouse clicking issue has been fixed.
    - MPS (GPU) acceleration for Mac M1 has been expanded to work with Demucs v4 and all MDX-Net models.
    - This bundle is intended for those running macOS Big Sur and above.
    - Application functionality for systems running macOS Catalina or lower is not guaranteed.
    - Application functionality for older or budget Mac systems is not guaranteed.
    - Once everything is installed, the application may take up to 5-10 minutes to start for the first time (depending on your Macbook).

- Download the UVR dmg for MacOS via one of the links below:
    - Mac M1 (arm64) users:
       - [Main Download Link](https://github.com/Anjok07/ultimatevocalremovergui/releases/download/v5.6/Ultimate_Vocal_Remover_v5_6_MacOS_arm64.dmg)
       - [Main Download Link mirror](https://www.mediafire.com/file_premium/u3rk54wsqadpy93/Ultimate_Vocal_Remover_v5_6_MacOS_arm64.dmg/file)

    - Mac Intel (x86_64) users:
       - [Main Download Link](https://github.com/Anjok07/ultimatevocalremovergui/releases/download/v5.6/Ultimate_Vocal_Remover_v5_6_MacOS_x86_64.dmg)
       - [Main Download Link mirror](https://www.mediafire.com/file_premium/2gf1werx5ly5ylz/Ultimate_Vocal_Remover_v5_6_MacOS_x86_64.dmg/file)

<details id="CannotOpen">
  <summary>MacOS Users: Having Trouble Opening UVR?</summary>

> Due to Apples strict application security, you may need to follow these steps to open UVR.
>
> First, run the following command via Terminal.app to allow applications to run from all sources (it's recommended that you re-enable this once UVR opens properly.)
> 
> ```bash
> sudo spctl --master-disable
> ```
> 
> Second, run the following command to bypass Notarization: 
> 
> ```bash
> sudo xattr -rd com.apple.quarantine /Applications/Ultimate\ Vocal\ Remover.app
> ```

</details>

<details id="MacInstall">
  <summary>Manual MacOS Installation</summary>

### Manual MacOS Installation

- Download and save this repository [here](https://github.com/Anjok07/ultimatevocalremovergui/archive/refs/heads/master.zip)
- Download and install Python 3.10 [here](https://www.python.org/ftp/python/3.10.9/python-3.10.9-macos11.pkg)
- From the saved directory run the following - 

```
pip3 install -r requirements.txt
```

- If your Mac is running with an M1, please run the following command next. If not, skip this step. - 

```
cp /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/_soundfile_data/libsndfile_arm64.dylib /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/_soundfile_data/libsndfile.dylib
```

**FFmpeg Installation**

- Once everything is done installing, download the correct FFmpeg binary for your system [here](http://www.osxexperts.net) and place it into the main application directory.

**Rubber Band Installation**

In order to use the Time Stretch or Change Pitch tool, you'll need Rubber Band.

- Download the precompiled build [here](https://breakfastquay.com/files/releases/rubberband-3.1.2-gpl-executable-windows.zip)
- From the archive, extract the following files to the UVR/lib_v5 application directory:
   - ```rubberband-3.1.2-gpl-executable-macos/rubberband```

This process has been tested on a MacBook Pro 2021 (using M1) and a MacBook Air 2017 and is confirmed to be working on both.

</details>

### Linux Installation (Updated Instructions)

<details id="LinuxInstall">
  <summary>See Linux Installation Instructions</summary>

<br />

**These installation instructions are for Debian & Arch-based Linux systems.**

---

#### **Step 1: Download the Repository**
- Download and save this repository from [GitHub](https://github.com/Anjok07/ultimatevocalremovergui/archive/refs/heads/master.zip).
- Extract the downloaded file to a directory of your choice.

---

#### **Step 2: Install Dependencies**
Use the following commands based on your system type:

**For Debian-based systems (Ubuntu, Mint, etc.):**
```bash
sudo apt update && sudo apt upgrade
sudo apt-get install -y ffmpeg python3-pip python3-tk
```

**For Arch-based systems (EndeavourOS):**
```bash
sudo pacman -Syu
sudo pacman -S ffmpeg python-pip tk
```

---

#### **Step 3: Set Up a Virtual Environment (Recommended)**
Setting up a virtual environment (venv) ensures that the program's dependencies do not interfere with system-wide Python packages.

1. **Navigate to the extracted repository directory:**
   ```bash
   cd /path/to/ultimatevocalremovergui
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   ```

3. **Activate the virtual environment:**
   - For **Debian-based and Arch-based systems:**
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies in the virtual environment:**
   ```bash
   pip install -r requirements.txt
   ```

---

#### **Step 4: Run the Application**
While the virtual environment is activated, start the application:
```bash
python UVR.py
```

---

#### **Important Notes**
1. **Avoid Modifying System Files:**  
   Previous instructions suggested deleting the `/usr/lib/python3.11/EXTERNALLY-MANAGED` file, which is dangerous and can break Python package management. Do **NOT** delete this file.

2. **Why Use Virtual Environments?**  
   Virtual environments isolate the program's dependencies, preventing conflicts with system Python packages. More information is available [here](https://stackoverflow.com/questions/75602063/pip-install-r-requirements-txt-is-failing-this-environment-is-externally-mana/75696359#75696359).

3. **Known Issues and Discussions:**  
   - [Issue #1578](https://github.com/Anjok07/ultimatevocalremovergui/issues/1578)  
   - [Pull Request #1068](https://github.com/Anjok07/ultimatevocalremovergui/pull/1068)

---

If you encounter issues, refer to the [GitHub Issues](https://github.com/Anjok07/ultimatevocalremovergui/issues) page for help. 

</details>

### Other Application Notes
- Nvidia RTX 1060 6GB is the minimum requirement for GPU conversions.
- Nvidia GPUs with at least 8GBs of V-RAM are recommended.
- AMD Radeon GPU supported is limited at this time.
   - There is currently a working branch for AMD GPU users [here](https://github.com/Anjok07/ultimatevocalremovergui/tree/v5.6-amd-gpu)
- This application is only compatible with 64-bit platforms. 
- This application relies on the Rubber Band library for the Time-Stretch and Pitch-Shift options.
- This application relies on FFmpeg to process non-wav audio files.
- The application will automatically remember your settings when closed.
- Conversion times will significantly depend on your hardware. 
- These models are computationally intensive. 

### Performance:
- Model load times are faster.
- Importing/exporting audio files is faster.

## Troubleshooting

### Common Issues

- If FFmpeg is not installed, the application will throw an error if the user attempts to convert a non-WAV file.
- Memory allocation errors can usually be resolved by lowering the "Segment" or "Window" sizes.

#### MacOS Sonoma Left-click Bug
There's a known issue on MacOS Sonoma where left-clicks aren't registering correctly within the app. This was impacting all applications built with Tkinter on Sonoma and has since been resolved. Please download the latest version via the following link if you are still experiencing issues - [link](https://github.com/Anjok07/ultimatevocalremovergui/releases/tag/v5.6)

This issue was being tracked [here](https://github.com/Anjok07/ultimatevocalremovergui/issues/840).

### Issue Reporting

Please be as detailed as possible when posting a new issue. 

If possible, click the "Settings Button" to the left of the "Start Processing" button and click the "Error Log" button for detailed error information that can be provided to us.

## License

The **Ultimate Vocal Remover GUI** code is MIT-licensed, and this fork is offered under that
same MIT licence. **There is no `LICENSE` file in this tree**: upstream deleted its own in
`63a4cb6` ("Delete LICENSE", 2022-12-18), and the link this sentence carried pointed at a
file that has not existed since, so it is removed rather than restated — adding the file
means naming a copyright holder, which is the maintainer's decision, not this branch's.
The licence statement of record for `fork/amd-rocm-pyside` is the "own licence" section of
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), which also lists the third-party terms;
Qt via PySide6 is LGPL-3.0, so its dynamic-linking exception is the one to read first.

- **Please Note:** For all third-party application developers who wish to use our models, please honor the MIT license by providing credit to UVR and its developers.

## Credits
- [ZFTurbo](https://github.com/ZFTurbo) - Created & trained the weights for the new MDX23C models. 
- [DilanBoskan](https://github.com/DilanBoskan) - Your contributions at the start of this project were essential to the success of UVR. Thank you!
- [Bas Curtiz](https://www.youtube.com/user/bascurtiz) - Designed the official UVR logo, icon, banner, and splash screen.
- [tsurumeso](https://github.com/tsurumeso) - Developed the original VR Architecture code. 
- [Kuielab & Woosung Choi](https://github.com/kuielab) - Developed the original MDX-Net AI code. 
- [Adefossez & Demucs](https://github.com/facebookresearch/demucs) - Developed the original Demucs AI code. 
- [KimberleyJSN](https://github.com/KimberleyJensen) - Advised and aided the implementation of the training scripts for MDX-Net and Demucs. Thank you!
- [Hv](https://github.com/NaJeongMo/Colab-for-MDX_B) - Helped implement chunks into the MDX-Net AI code. Thank you!

## Contributing

- For anyone interested in the ongoing development of **Ultimate Vocal Remover GUI**, please send us a pull request, and we will review it. 
- This project is 100% open-source and free for anyone to use and modify as they wish. 
- We only maintain the development and support for the **Ultimate Vocal Remover GUI** and the models provided. 

## References
- [1] Takahashi et al., "Multi-scale Multi-band DenseNets for Audio Source Separation", https://arxiv.org/pdf/1706.09588.pdf
