from __future__ import annotations
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name, "").strip()
    return Path(raw) if raw else default


MODEL_ROOT = _env_path("UVR_MODEL_ROOT", REPO / "models")

# Fixed golden input. Tasks 4, 7 and 13 all read this.
INPUT_FILE = _env_path(
    "UVR_INPUT_FILE",
    Path.home() / "Documents/UVR5/1_Circle With Me_(Instrumental).flac",
)

WEIGHT_GLOBS = ("*.pth", "*.th", "*.pt", "*.ckpt", "*.onnx")

_NAMED = {
    "vr": ["VR_Models/1_HP-UVR.pth"],
    "demucs": ["Demucs_Models/v3_v4_repo/955717e8-8726e21a.th"],
    "onnx": ["MDX_Net_Models/UVR-MDX-NET-Inst_HQ_3.onnx"],
}

# Keys whose weight filename varies per model -> search by extension.
_SEARCH_DIRS = {
    "mdx_c": ["MDX_Net_Models"],
    "roformer": ["Apollo_Models", "MDX_Net_Models", "VR_Models"],
}

# Roformer/Apollo checkpoints are identifiable by name; upstream files are e.g.
# "BS-Roformer-Viperx-1297.ckpt", "mel-band-roformer-karlschware.pth", apollo .bin.
_ROFORMER_TOKENS = ("roformer", "apollo", "melband", "mel-band", "viperx", "bs_roformer")

# Folders the engine actually loads roformer-class weights from (UVR.py:6075-6080:
# VR_Models -> VR nets, MDX_Net_Models -> onnx / mdx_c / mdx23c / roformer,
# Apollo_Models -> apollo). VR_Models is deliberately NOT one of them.
_ROFORMER_HOME = ("Apollo_Models", "MDX_Net_Models")


def _named(rel: str) -> Path | None:
    p = MODEL_ROOT / rel
    return p if p.is_file() else None


def _is_family(key: str, directory: str, hit: Path) -> bool:
    """Refuse a loose weight that clearly belongs to a different architecture.

    A plain extension glob over VR_Models would happily return 1_HP-UVR.pth and
    register it as MODELS["roformer"] — the same false positive that made the
    mdx_c *config directory* look like an available weight. Likewise a roformer
    checkpoint dropped in the shared MDX_Net_Models folder is not an MDX-C one.
    """
    stem = hit.stem.lower()
    looks_roformer = any(token in stem for token in _ROFORMER_TOKENS)
    if key == "roformer":
        return True if directory in _ROFORMER_HOME else looks_roformer
    return not looks_roformer


def _searched(key: str, directories: list[str]) -> Path | None:
    claimed = set(MODELS.values())
    for directory in directories:
        base = MODEL_ROOT / directory
        if not base.is_dir():
            continue
        for pattern in WEIGHT_GLOBS:
            for hit in sorted(base.rglob(pattern)):
                # an onnx file is an ONNX-model weight, not an MDX-C/roformer one
                if not hit.is_file() or hit.suffix == ".onnx":
                    continue
                if hit in claimed:
                    continue
                if _is_family(key, directory, hit):
                    return hit
    return None


MODELS: dict[str, Path] = {}
for _key, _rels in _NAMED.items():
    for _r in _rels:
        _p = _named(_r)
        if _p:
            MODELS[_key] = _p
            break

for _key, _dirs in _SEARCH_DIRS.items():
    _p = _searched(_key, _dirs)
    if _p:
        MODELS[_key] = _p

# Config directories are inputs to a model, never a weight.
CONFIGS: dict[str, Path] = {}
_mdx_c_cfg = MODEL_ROOT / "MDX_Net_Models/model_data/mdx_c_configs"
if _mdx_c_cfg.is_dir():
    CONFIGS["mdx_c"] = _mdx_c_cfg
_roformer_cfg = MODEL_ROOT / "Apollo_Models/model_configs"
if _roformer_cfg.is_dir():
    CONFIGS["roformer"] = _roformer_cfg
