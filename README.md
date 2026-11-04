# UVR5 AMD/ROCm fork — read this before your first run

This branch (`fork/amd-rocm-pyside`) runs UVR's engine on an AMD GPU through torch's ROCm
build, without the tkinter GUI. Three things here are policy, not preference.

## 1. `pytest` is supported only on a `-Backend rocm` venv

One golden pin store (`tests/golden/pins.json`) asserts `torch`/`torch_hip` against the live
environment, so a venv built `-Backend cuda` or `-Backend cpu` fails the suite **by design**,
even on byte-identical audio. That red is not something you broke.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\bootstrap.ps1 -Backend rocm
.venv\Scripts\python.exe -m pytest -q   # 3 min here (RX 7900 XTX, warm MIOpen cache)
```

## 2. A GUI-free run cannot open the model-params dialog: an unknown model is a hard error

Upstream resolves params from the per-model `<hash>.json`, then a baked table, then **asks a
human through a popup**. A CLI has no popup: the third case exits non-zero naming the hash and
the exact file it wanted, and that message is the only self-service path there is. It refuses
during resolution, before any device is chosen, so this example needs no `--device`:

```
$ .venv/Scripts/python.exe separate_cli.py --arch vr --model mystery-model -a song.flac -o out
[cli] "mystery-model" (arch vr) is not in the catalog: hash 3f2a…c1; no per-model JSON at
      C:\Users\you\AppData\Roaming\UVR-fork\model_data\VR_Models\model_data\3f2a…c1.json …
      Tier 3 needs user-supplied params — a CLI cannot open the params dialog, so write that
      file (Catalog.write_params) or pass the params explicitly.                              (exit 3)
```

Write that JSON — `Catalog.write_params("vr", "<hash>", params)` or by hand — and the same
command resolves. Weights dirs (`F:/Software/Ultimate Vocal Remover/models` and the tracked
`models/` tree) are **read-only**: new per-model JSON goes to `%APPDATA%\UVR-fork\model_data`,
which the catalog reads *before* the baked tables.

## 3. Speed claims carry their conditions

Conditions: **Radeon RX 7900 XTX (gfx1100)**, Windows, `torch 2.9.1+rocm7.2.1`,
`torch.version.hip 7.2.53211-158bd99533`, the golden 30 s crop through
`tests.golden.runner.separate()` with the frozen `pins.json` settings (`Default` device,
VR `1_HP-UVR`), **warm** — a process's first GPU run is excluded, MIOpen's cold kernel search
being ~20× slower (36.84 s → 1.65 s). Re-derive any figure below with the committed driver:

```
UVR_MODEL_ROOT="F:/Software/Ultimate Vocal Remover/models" .venv/Scripts/python.exe tools/task7_timing.py
```

* Headline: **absolute warm GPU times** — **VR 1.67–1.88 s**, **htdemucs 2.74 s** for the
  30 s crop (n=4 GPU runs per session; warm = best of the 3 after cold). Repeats *within* a
  session agree to ~2%; *across* sessions VR spans 1.67–1.88 s (12.6%), so ~2% is never a
  cross-session claim — the driver itself says ~10% session-to-session is normal.
* The healthy-CPU ratio appears **only as a range with its n**: **5.24–5.90×** (VR, n=3
  published sessions) and **4.05×** (htdemucs, n=2 — the driver recorded 2.74 s warm in both,
  so both compute to 4.05 and there is no low bound to state). Its divisor (9.845 s / 11.108 s)
  came from `torch 2.15.0+cpu` elsewhere and this repository cannot re-measure it, so quoting
  the top of the VR range as a single number is a policy violation — a test asserts its absence.
* **Numerically verified**: SNR ≥ 40 dB, CPU output vs ROCm output, for `vr` and `htdemucs`
  only — single-model runs, one 30 s crop, one device pair (the RX 7900 XTX and this wheel).
  **Not verified**: no `mdx`, `onnx`, `roformer` or `mdx_c` baseline exists, and demucs **v1/v2
  on ROCm is unverified** — no v1/v2 weight is installed here (`Catalog.names("demucs")` gives
  `955717e8-8726e21a.th` and `htdemucs.yaml`, both `demucs_version="v4"`).
* Same-wheel CPU ratios (VR ~4.3–4.7×, htdemucs ~18–20×) are printed by the driver and are
  reproducible here, but are **not** headlined: this ROCm wheel's CPU path runs ~5× slower than
  a healthy build's, so that demucs figure measures a CPU regression, not a GPU result.

## No auto-update; tables refresh only by explicit command

Nothing phones home on any code path, and the official update package would overwrite a forked
install anyway. The four baked tables upstream fetches on startup refresh only when asked:

```
.venv/Scripts/python.exe separate_cli.py --refresh-model-tables --confirm
```

Without `--confirm` it refuses, exits non-zero and touches no network (`model_refresh.py`'s own
rule, which the CLI delegates to). The shipped repo tables are staler than a live install's
(MDX 61 vs 86 entries; name mapper 20 vs 46, the missing 26 being the Roformer/MDX23C class), so
a refresh is what makes newer names resolvable.

## Running separations without the GUI

**Without `--device`, a run computes on CPU**: `Settings.gpu_conversion` defaults to `False`
(the GUI's GPU checkbox, off) and `--device` is the only thing that overrides it.

```
.venv/Scripts/python.exe separate_cli.py --list-models vr
.venv/Scripts/python.exe separate_cli.py -a song.flac -o out --arch vr --model 1_HP-UVR --device rocm
.venv/Scripts/python.exe separate_cli.py -a song.flac -o out --model NAME --device rocm --dry-run
```

`--device` goes to `backends.resolve()` and is never a silent downgrade: the run prints the
backend it chose (`name=rocm torch_device=cuda …`) and a `[cli] compute device:` line stating
the device it will actually use, and says out loud when the requested device cannot be opened.
Exit codes: 0 ok, 1 engine failure, 2 usage, 3 unknown model params, 4 blocked arch, 5 missing
weight, 6 unknown name, 7 plan refused, 8 refresh refused, 9 input missing, 10 writable-location
refused (the catalog write dir resolved inside a read-only or git-tracked weights tree).
**Blocked:** `mdx_c`, `onnx` and `roformer` are refused rather than simulated — no weights of
those classes exist on this machine (configs only), so this fork publishes no result for them.
Internals (the `Settings`/`Catalog`/`RunPlan` split, the purity ratchet) live in module
docstrings and commit messages, deliberately not here.

The upstream v5.6 document follows, kept intact; everything in it about installers, update
packages and Nvidia/OpenCL setup describes the **official** application, not this fork.

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

The **Ultimate Vocal Remover GUI** code is [MIT-licensed](LICENSE). 

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
