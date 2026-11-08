# Third-party notices

Everything below was read off this machine's installed wheels, not off a package index,
on 2026-11-06. Versions are the ones `requirements.txt` and
`tests/golden/pins.json` pin for the branch, and each entry names the upstream project's
**own** licence file. Where that file ships inside the wheel it is given by the path it
has *here*; where it does not — the font/icon asset rows below — the path is the one it
has in the upstream **source** tree, and the recipient of this fork gets those faces
compiled into the native `_fluentqt.pyd` with no separate licence text they can open.
Those two paths are a citation, not a file this repository ships.

## This fork's own licence

The Ultimate Vocal Remover GUI is MIT-licensed, and this fork is distributed under that
same MIT licence: it derives from upstream's code and adds an AMD/ROCm engine path, a
GUI-free CLI and a Fluent-Qt front end. Attribution for the models themselves is asked
for in upstream's own terms — if you use a model published by a third party, honour that
model's licence.

One caveat a reader should not have to discover later: **this tree contains no `LICENSE`
file.** Upstream removed it in commit `63a4cb6` ("Delete LICENSE", 2022-12-18). The fork's
`README.md` says so plainly under §License — it no longer links to a file that is not there
— and points back here, so the paragraph above is the licence statement of record for this
branch. Writing the file is a maintainer's decision, because it means naming a copyright
holder; this notice does not invent one.

## UI layer

| Component | Version | Licence | Upstream's own licence file |
|---|---|---|---|
| Fluent-Qt (Shiboken6 bindings for the Fluent-Qt C++ Qt Widgets library) | 1.11.2 | MIT | `fluentqt-1.11.2.dist-info/licenses/LICENSE` |
| Qt via PySide6-Essentials + shiboken6 | 6.9.3 each | LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only (commercial alternative available) | `pyside6_essentials-6.9.3.dist-info/licenses/LicenseRef-Qt-Commercial.txt`, plus the SPDX expression in each `METADATA` |
| Inter (the text faces inside Fluent-Qt) | 4.1 derivatives | SIL Open Font License 1.1 | `third_party/fonts/inter/LICENSE.txt` in the Fluent-Qt source tree, cited by its own `THIRD_PARTY_NOTICES.md` |
| Fluent UI System Icons | 1.1.328 derivatives | MIT | `third_party/icons/fluentui-system-icons/LICENSE.txt` and `NOTICE.txt`, cited by the same file |

**Fluent-Qt** is the only UI dependency this app builds against, and its MIT licence
covers its own source only. Its wheel carries its own notices, which are the authority
for the two asset rows below:
`fluentqt-1.11.2.dist-info/licenses/THIRD_PARTY_NOTICES.md` and
`…/licenses/TRADEMARKS.md`.

**Inter.** The faces the widgets draw with are renamed derivatives of the hinted static
Inter 4.1 TrueType fonts, copyright *The Inter Project Authors*, licensed under the SIL
Open Font License 1.1. Per Fluent-Qt's own notices they use Fluent-Qt-specific family
names and **do not use Inter's Reserved Font Name**, so the OFL's RFN clause is not
engaged. On this install they are not loose files: `fluentqt/res/fonts/` does not exist
in the 1.11.2 wheel — the faces are compiled into the Qt resource inside
`fluentqt/_fluentqt.pyd`, and the app reaches them only through
`fluentqt.initialize_resources()`. This fork redistributes no Inter font file of its own.

**Fluent UI System Icons.** The icon face and name-to-codepoint catalog are renamed
derivatives of Microsoft's Fluent UI System Icons 1.1.328 under the MIT licence.
Fluent-Qt's generator preserves the upstream catalog entries and adds its own semantic
shortcuts **without copying any proprietary Segoe font outlines**; nothing here ships a
Segoe face. The icon aliases the Python side sees are `fluentqt/_icon_aliases.json`.

One asset row the wheel's own notices list but this table does not: a **Noto Sans SC**
derivative. It is referenced only by Fluent-Qt's WebAssembly (browser) target, is absent
from the native Windows wheel this fork installs, and is therefore never linked into
`_fluentqt.pyd` here — nothing in this repository reaches it.

**Qt / PySide6.** `PySide6-Essentials 6.9.3` and `shiboken6 6.9.3` (pulled in by
Fluent-Qt, which hard-pins both) declare `LGPL-3.0-only OR GPL-2.0-only OR
GPL-3.0-only` in their wheel metadata; the only licence *file* the wheels carry is
`LicenseRef-Qt-Commercial.txt`, and the LGPL/GPL texts are Qt upstream's, at
<https://www.qt.io/qt-licensing>. This app takes the LGPL route, and the reason it can is
how it is distributed: it runs from source in a `pip`-created venv, with Qt's libraries as
separate, replaceable DLLs (`PySide6/Qt6Core.dll`, `Qt6Gui.dll`, `Qt6Widgets.dll`) that
the interpreter loads at run time. That is dynamic linking, not a static or single-binary
work, so LGPLv3 §4's combined-works terms are met by construction: nothing needs
re-linking to swap Qt out, and replacing the `PySide6`/`shiboken6` directories replaces
the whole Qt runtime. No Qt corresponding source is mirrored in this repository; the
source for the exact 6.9.3 build is The Qt Company's, and the fork's own source — the
only part a recipient would need to relink — is this repository.

## Engine layer

| Component | Version | Licence | Upstream's own licence file |
|---|---|---|---|
| torch, ROCm build | 2.9.1+rocm7.2.1 | BSD-3-Clause, with the third-party texts concatenated inside | `torch-2.9.1+rocm7.2.1.dist-info/licenses/LICENSE` and `…/NOTICE` |
| onnxruntime-directml | 1.24.4 | MIT | `onnxruntime/LICENSE`, with `onnxruntime/ThirdPartyNotices.txt` beside it |

**ROCm torch** is not PyPI's torch. It is AMD's Windows ROCm build, installed by
`bootstrap.ps1 -Backend rocm` from the flat directory listing at
`https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/`, because PyTorch's own ROCm
channels publish no `win_amd64` wheels for the release train this fork uses. The licence
file is torch's BSD-3 text plus the concatenated third-party licences of the build;
the HIP / ROCm runtime components it links (rocBLAS, MIOpen, cuDNN-equivalents, and the
`rocm-sdk-*` packages the bootstrap installs alongside) stay under AMD's and their own
upstream's terms, which that `NOTICE` file is the index to. `torch.version.hip` in
`tests/golden/pins.json` is the identity check that a run really used this build.

**onnxruntime-directml** is Microsoft's ONNX Runtime with the DirectML execution
provider, which is how this fork runs ONNX models on an AMD card: `backends.resolve()`
selects `DmlExecutionProvider` from a PCI vendor scan rather than trusting a device
name. The dist-info for the DirectML build carries no `licenses/` directory — the MIT
text is the in-package `onnxruntime/LICENSE` that its `RECORD` lists, and
`ThirdPartyNotices.txt` next to it covers the components ONNX Runtime itself embeds.

## Everything else

The remaining runtime dependencies are listed in `requirements.txt` and stay under their
own upstream licences, which the installed wheels carry; they are not re-transcribed
here, because a copy would be a second source of truth free to drift. That set includes
librosa, scipy, numba/llvmlite, soundfile, pydub, einops, julius, omegaconf,
ml-collections, pytorch-lightning, onnx, natsort, matchering, psutil and the demucs and
`lib_v5` model code vendored in this repository.

The vendored `demucs/` and `lib_v5/` trees carry their upstream headers and licences
in place; nothing in this notice alters them. Model weights are separate works with
their own terms. No tooling on this branch ever *installs* a weight: there is no
downloader on any code path, and the model root (`UVR_MODEL_ROOT`, defaulting to
`REPO/models`) is read-only by policy. That is not the same as "the tree holds none" —
two weights inherited from the upstream base are still git-tracked here,
`models/VR_Models/UVR-DeNoise-Lite.pth` (17,922,277 bytes) and `lib_v5/mixer.ckpt`, and
`catalog.py` builds VR weight paths under the repo root, so the repository tree *is* a
working weights root for a fresh clone. Whether to keep shipping that tracked `.pth` is
the maintainer's open decision; this notice states the fact and does not resolve it.

## Names

"Fluent", "WinUI", "Windows" and related marks belong to Microsoft; "Qt" is a trademark
of The Qt Company; "AMD", "Radeon" and "ROCm" belong to Advanced Micro Devices; "PyTorch"
belongs to its Foundation contributors; "Inter" is the Reserved Font Name the font faces
above deliberately do not use. Their appearance here identifies the upstreams being
redistributed, and implies no endorsement or affiliation. Fluent-Qt's own
`TRADEMARKS.md`, shipped in the wheel, is the fuller statement.

## Bundled UI fonts (SIL Open Font License 1.1)

The GUI bundles three fonts under `app/assets/fonts/`, all under the SIL Open Font
License 1.1 (the full text ships as `app/assets/fonts/OFL.txt`):

- **Inter** (c) Rasmus Andersson -- https://github.com/rsms/inter
- **JetBrains Mono** (c) JetBrains s.r.o. -- https://github.com/JetBrains/JetBrainsMono
- **Noto Sans SC** (subset to this app's glyphs) (c) Google -- https://fonts.google.com/noto
