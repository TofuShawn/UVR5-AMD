"""`ModelCatalog` — model metadata, hash lookup and the writable `model_data` dir.

Task 10 takes the four `*_MAPPER` tables and the model-params resolution chain off
the tkinter App (`UVR.py:1712-1716`, `:740-777`) and gives them an owner with no
GUI dependency: this module imports no `tkinter` and no `gui_data` GUI module (only
`gui_data.constants`, the same exception `separate.py:18` uses), and it never calls
`settings.config_dir()` — `writable_dir` arrives as a constructor argument (ruling
R-T10-writable), because the composition root `app/controller.py` is the only place
that sees `Settings`, `Catalog` and `Backend` together.

Entry points
------------
* `hash_of(weights_path) -> str` — `UVR.py:793-799` (md5 of the last
  `10000 * 1024` = 10,240,000 bytes, whole-file below that).
* `Catalog(model_root, writable_dir, fallback_root=REPO/models)`
  with `.resolve(name, arch=None, *, required=False) -> ModelSpec`,
  `.names(arch)`, `.known_names(arch)`, `.baked_table(arch, kind)`,
  `.write_params(arch, model_hash, params)`, `.hash_of(path)`,
  `.secondary_for(spec, requested)`, `.demucs_pre_proc_for(name)`,
  `.vocal_split_for(name)`.
* `secondary_slot_for(process_method, primary_stem) -> str | None`
* `ModelSpec`, `Provenance`, `Tier`, `RootRole`, `CatalogError` and subclasses.

The three decisions this task was allowed to make
-------------------------------------------------
1. **How params-absent (tier 3) is represented.** A resolved `ModelSpec` whose
   params could not be found comes back with `model_data is None` and
   `needs_user_params=True`; it does **not** raise and it never opens a dialog.
   Reason: upstream already has this exit — `is_dry_check` short-circuits
   `get_model_data_from_popup` to `None` (`UVR.py:760-761`) and
   `is_get_hash_dir_only` short-circuits `change_model_data` (`:754`) — so a
   returned object mirrors the precedent instead of inventing a parallel flag, and
   Task 11 can port `ModelData` mechanically. A CLI that would rather stop than
   continue passes `required=True` and gets the typed `UnknownModelParamsError`,
   whose `str()` is `ModelSpec.missing_message`: it names the model, the hash and
   the exact file it wanted, which is what Task 13's acceptance criteria print.
2. **How `resolve()` reports provenance.** Every spec carries `provenance` — a
   `Tier` (`PER_MODEL_JSON` = tier 1, `BAKED_TABLE` = tier 2,
   `NEEDS_USER_PARAMS` = tier 3, `NOT_APPLICABLE` = this arch has no params table)
   plus the `RootRole` (`WRITABLE_DIR` / `MODEL_ROOT` / `FALLBACK_ROOT`), the root
   `Path` that answered and the `source` file. Reason: the two model roots on this
   machine disagree (`models/` in the repo is a stale subset — fewer MDX hashes,
   fewer model names, and one differing `primary_stem` value), so "which root and
   which tier answered" has to be observable instead of inferred from a `KeyError`.
   Reads fall back from the active root to the repo tree; **writes only ever go to
   `writable_dir`**, and on a tier-1 conflict `writable_dir` wins (R-T10-writable).
3. **How the hash memo is keyed.** `UVR.py:315`'s `model_hash_table` is a
   process-global dict keyed by model *path* alone, so swapping a weight under the
   same name serves the old hash forever. That key is deliberately not ported
   forward: `Catalog.hash_of` memoises on `(resolved path, st_size, st_mtime_ns)`
   and the memo dies with the `Catalog` instance. Reason: a stale hash silently
   re-points every `model_data/<hash>.json` lookup.

Blocked architectures (`mdx_c`, `onnx`, `roformer`) refuse rather than simulate —
`is_blocked(arch)` returns the reason and `names()`/`resolve()` raise
`BlockedArchError`. Apollo has no tier-2 table at all (`APOLLO_HASH_JSON` is
defined at `UVR.py:253` and its loader at `:1714` is commented out), so
`baked_table("apollo", "params")` raises `NoBakedTableError` instead of handing
back an empty dict that reads like "the table is empty".
"""
from __future__ import annotations

import enum
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import uvr_paths
from gui_data.constants import (DEMUCS_ARCH_TYPE, DEMUCS_UVR_MODEL, MDX_ARCH_TYPE,
                                NO_MODEL, VR_ARCH_TYPE, secondary_stem)

__all__ = ["TAIL_WINDOW_BYTES", "Tier", "RootRole", "Provenance", "ModelSpec",
           "Catalog", "CatalogError", "BlockedArchError", "UnknownArchError",
           "UnknownModelNameError", "MissingWeightError", "NoBakedTableError",
           "WritableLocationError", "UnknownModelParamsError", "hash_of",
           "secondary_slot_for", "BLOCKED_ARCHS", "SUPPORTED_ARCHS", "ARCHS"]

#: `UVR.py:795` -- `f.seek(- 10000 * 1024, 2)`. 10,240,000 bytes, i.e. 10,000 KiB.
#: Writing `10 * 1024 * 1024` (10 MiB) instead changes the digest of every weight
#: larger than the window and breaks every existing `<hash>.json` lookup.
TAIL_WINDOW_BYTES = 10000 * 1024

#: The repo tree's `models/` is version-controlled (23 tracked files, including
#: committed tier-1 artifacts), so it is the default *fallback* for reads and never
#: a target for writes.
REPO_MODELS = uvr_paths.REPO / "models"

VR_TABLE = "VR_Models/model_data/model_data.json"
MDX_TABLE = "MDX_Net_Models/model_data/model_data.json"
MDX_NAMES = "MDX_Net_Models/model_data/model_name_mapper.json"
DEMUCS_NAMES = "Demucs_Models/model_data/model_name_mapper.json"

#: Fork policy (`tests/golden/pins.json` `archs_blocked`): no weights of these
#: classes exist on this machine, so the catalog refuses instead of fabricating.
BLOCKED_ARCHS = {
    "mdx_c": "no MDX-C checkpoint on this machine — only model_data/mdx_c_configs yaml",
    "onnx": "onnx execution is out of scope for this fork's engine (DirectML path not wired)",
    "roformer": "no BS/Mel-Band/Apollo roformer weights on this machine (configs only)",
}

#: The four suffixes a VR/MDX/Apollo weight can arrive under (`uvr_paths.WEIGHT_GLOBS`
#: minus `.onnx`, which MDX re-adds).
_VR_SUFFIXES = (".pth", ".th", ".pt", ".ckpt")
_MDX_SUFFIXES = (".onnx", ".ckpt", ".pth", ".pt")
_DEMUCS_SUFFIXES = (".th", ".yaml", ".ckpt", ".pt")
_APOLLO_SUFFIXES = (".bin", ".ckpt", ".pth", ".safetensors", ".pt")


class Tier(enum.Enum):
    """Which rung of the resolution chain answered."""

    PER_MODEL_JSON = "tier1-per-model-json"
    BAKED_TABLE = "tier2-baked-table"
    NEEDS_USER_PARAMS = "tier3-needs-user-params"
    NOT_APPLICABLE = "no-params-table-for-this-arch"


class RootRole(enum.Enum):
    """Which root answered; see the module docstring's decision 2."""

    WRITABLE_DIR = "writable_dir"
    MODEL_ROOT = "model_root"
    FALLBACK_ROOT = "fallback_root"


class CatalogError(Exception):
    """Base class for every catalog failure; all are printable by a CLI."""


class BlockedArchError(CatalogError):
    """The arch has no weights on this machine (fork policy: refuse, not simulate)."""

    def __init__(self, arch: str, reason: str):
        super().__init__(f"architecture {arch!r} is blocked: {reason}")
        self.arch, self.reason = arch, reason


class UnknownArchError(CatalogError):
    """Not a catalog architecture, blocked or otherwise."""


class UnknownModelNameError(CatalogError):
    """No architecture's tables or files declare this name."""


class MissingWeightError(CatalogError):
    """The name is declared but the weight file is not on disk in any root."""


class NoBakedTableError(CatalogError):
    """This arch has no baked tier-2 table (Apollo, Demucs params) — a state, not `{}`."""


class WritableLocationError(CatalogError):
    """`writable_dir` sits inside a read-only or version-controlled root."""


class UnknownModelParamsError(CatalogError):
    """Tier 3, raised only when the caller asked for `required=True`.

    Carries the `ModelSpec` so the CLI can print the same string the GUI would show.
    """

    def __init__(self, spec: "ModelSpec"):
        super().__init__(spec.missing_message)
        self.spec = spec


@dataclass(frozen=True)
class Provenance:
    """The answer's address: which tier, which root, which file."""

    tier: Tier
    root_role: RootRole
    root: Path
    source: Path | None = None

    @property
    def label(self) -> str:
        return f"{self.tier.value}@{self.root_role.value}"


@dataclass(frozen=True)
class ModelSpec:
    """One model's resolved metadata. Field order is the plan's interface."""

    name: str
    arch: str
    process_method: str | None
    primary_stem: str | None
    secondary_stem: str | None
    weights_path: Path | None
    config_path: Path | None
    hash: str | None
    model_data: dict | None
    provenance: Provenance | None = None
    needs_user_params: bool = False
    wanted_json: Path | None = None
    searched_tables: tuple[Path, ...] = ()

    @property
    def missing_message(self) -> str:
        """The tier-3 outcome, as a single line a GUI can show and a CLI can print.

        Task 13's acceptance criteria require it to name the model, its hash and the
        file it wanted; upstream's equivalent is the blocking
        `get_model_data_from_popup` (`UVR.py:759-777`).
        """
        wanted = self.wanted_json or "<no per-model JSON location configured>"
        tables = ", ".join(str(p) for p in self.searched_tables) or "<none>"
        return (f'"{self.name}" (arch {self.arch}) is not in the catalog: '
                f"hash {self.hash}; no per-model JSON at {wanted} and no baked-table "
                f"entry among {tables}. Tier 3 needs user-supplied params — a CLI "
                f"cannot open the params dialog, so write that file "
                f"(Catalog.write_params) or pass the params explicitly.")


@dataclass(frozen=True)
class ArchSpec:
    """The static knowledge: where an arch's weights, tables and tier-1 files live."""

    key: str
    process_method: str | None
    weight_dirs: tuple[str, ...]
    tier1_dir: str
    params_table: str | None
    names_table: str | None
    config_dir: str | None
    params_required: bool
    suffixes: tuple[str, ...]
    # 'file' -> the name IS the file (stem for VR); 'vr'/'mdx'/'demucs'/'apollo'
    name_style: str


ARCHS: dict[str, ArchSpec] = {
    "vr": ArchSpec("vr", VR_ARCH_TYPE, ("VR_Models",), "VR_Models/model_data",
                   VR_TABLE, None, None, True, _VR_SUFFIXES, "vr"),
    # 'mdx' is the MDX-Net/MDX23C class the baked tables cover; `mdx_c` (blocked)
    # is the *config-driven* variant and is deliberately a different key.
    "mdx": ArchSpec("mdx", MDX_ARCH_TYPE, ("MDX_Net_Models",), "MDX_Net_Models/model_data",
                    MDX_TABLE, MDX_NAMES, None, True, _MDX_SUFFIXES, "mdx"),
    "demucs": ArchSpec("demucs", DEMUCS_ARCH_TYPE,
                       ("Demucs_Models/v3_v4_repo", "Demucs_Models"),
                       "Demucs_Models/model_data", None, DEMUCS_NAMES, None, False,
                       _DEMUCS_SUFFIXES, "demucs"),
    # UVR.py:1059-1148 — ApolloModelData has its own one-parameter get_model_data
    # (:1102): tier-1 file or nothing. Its loader for APOLLO_HASH_JSON (:253) is
    # commented out at :1714, so `params_table` is None rather than a missing file.
    "apollo": ArchSpec("apollo", None, ("Apollo_Models",), "Apollo_Models/model_data",
                       None, None, "Apollo_Models/model_configs", True,
                       _APOLLO_SUFFIXES, "apollo"),
}

SUPPORTED_ARCHS = tuple(sorted(ARCHS))
#: The stems that pick which secondary-model slot applies (`UVR.py:7059-7070`).
_SECONDARY_SLOTS = {
    "voc_inst": "voc_inst_secondary_model",
    "other": "other_secondary_model",
    "drums": "drums_secondary_model",
    "bass": "bass_secondary_model",
}


def _stem_group(primary_stem: str | None) -> str | None:
    if primary_stem in {"Vocals", "Instrumental"}:
        return "voc_inst"
    if primary_stem in {"Other", "No Other"}:
        return "other"
    if primary_stem in {"Drums", "No Drums"}:
        return "drums"
    if primary_stem in {"Bass", "No Bass"}:
        return "bass"
    return None


def secondary_slot_for(process_method: str | None, primary_stem: str | None) -> str | None:
    """Which of the four saved secondary selections applies — `UVR.py:7046-7070`.

    Upstream first picks the *variable group* by process method
    (`vr_secondary_model_vars` / `mdx_...` / `demucs_...`) and then the slot by the
    main model's `primary_stem`. The group is `Settings`' concern (Task 9 owns the
    50 fields); this returns the key within whichever group the caller holds.
    Returns None for an arch without secondary models (Apollo: upstream's
    `ApolloModelData` never sets a process method) or an unknown stem.
    """
    if process_method not in {VR_ARCH_TYPE, MDX_ARCH_TYPE, DEMUCS_ARCH_TYPE}:
        return None
    group = _stem_group(primary_stem)
    return None if group is None else _SECONDARY_SLOTS[group]


def hash_of(weights_path: Path) -> str:
    """md5 of the last `TAIL_WINDOW_BYTES`, whole-file below that — `UVR.py:793-799`.

    Deliberately *not* verbatim: upstream's fallback is a bare `except:` whose
    whole-file branch re-opens the file with an unclosed `open(...)` (and Apollo
    duplicates the same block at `:1087-1092`). Here the window seek is narrowed to
    `OSError`/`ValueError` and both branches go through one `with`. This is the same
    digest `tests/golden/settings.py:_vr_model_hash` produces; that copy moves here
    when Task 11 retires `ModelData`.
    """
    path = Path(weights_path)
    try:
        with path.open("rb") as handle:
            handle.seek(-TAIL_WINDOW_BYTES, 2)
            return hashlib.md5(handle.read()).hexdigest()
    except (OSError, ValueError):
        # the file is smaller than the window (or unreadable as a seekable stream):
        # upstream's whole-file branch, without the leaked handle
        return hashlib.md5(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


@dataclass
class Catalog:
    r"""Model metadata for one machine, from one explicit pair of roots.

    `model_root` is the active weights dir (`uvr_paths.MODEL_ROOT`, i.e.
    `UVR_MODEL_ROOT` when set) and `writable_dir` is the fork's user-writable
    catalog location — per ruling R-T10-writable `%APPDATA%\UVR-fork\model_data`,
    supplied by the composition root, never reached for internally. `fallback_root`
    defaults to the repo's version-controlled `models/` tree: a user with a
    read-only or empty weights dir still resolves *something* from the shipped
    tables, and `provenance.root_role` says which root answered.

    Reads check `writable_dir` first, then `model_root`, then `fallback_root`.
    Writes go **only** to `writable_dir`, creating directories as needed — upstream
    has no `os.makedirs` for any model directory (`UVR.py` has one at `:7238`, for
    the export path), which is why `pop_up_apollo_model_sub_json_dump` (`:5227`)
    fails outright when `Apollo_Models` is absent.
    """

    model_root: Path
    writable_dir: Path
    fallback_root: Path | None = REPO_MODELS

    _hash_memo: dict[tuple[str, int, int], str] = field(default_factory=dict, repr=False)
    _table_memo: dict[tuple[str, str, str], dict] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.model_root = Path(self.model_root)
        self.writable_dir = Path(self.writable_dir)
        self.fallback_root = None if self.fallback_root is None else Path(self.fallback_root)
        for parent, label in ((self.model_root, "model_root"),
                              (self.fallback_root, "fallback_root"),
                              (REPO_MODELS, "the version-controlled models/ tree")):
            if parent is not None and _is_within(self.writable_dir.resolve(), parent.resolve()):
                raise WritableLocationError(
                    f"writable_dir {self.writable_dir} sits inside {label} ({parent}); "
                    f"tier-1 writes there dirty a read-only or git-tracked tree "
                    f"— use e.g. %APPDATA%\\UVR-fork\\model_data")

    # -- roots -------------------------------------------------------------- #

    def _roots(self, *, weights: bool = False) -> list[tuple[RootRole, Path]]:
        out: list[tuple[RootRole, Path]] = []
        if not weights:
            out.append((RootRole.WRITABLE_DIR, self.writable_dir))
        out.append((RootRole.MODEL_ROOT, self.model_root))
        if self.fallback_root is not None:
            out.append((RootRole.FALLBACK_ROOT, self.fallback_root))
        return out

    def _find(self, rel: str, *, weights: bool = False) -> tuple[RootRole, Path, Path] | None:
        for role, root in self._roots(weights=weights):
            candidate = root / rel
            if candidate.is_file():
                return role, root, candidate
        return None

    def _arch(self, arch: str) -> ArchSpec:
        if arch in BLOCKED_ARCHS:
            raise BlockedArchError(arch, BLOCKED_ARCHS[arch])
        try:
            return ARCHS[arch]
        except KeyError:
            raise UnknownArchError(
                f"unknown architecture {arch!r}; the catalog knows "
                f"{', '.join(SUPPORTED_ARCHS)} and blocks {', '.join(sorted(BLOCKED_ARCHS))}"
            ) from None

    # -- tables ------------------------------------------------------------- #

    def is_blocked(self, arch: str) -> str | None:
        """The reason this fork refuses `arch`, or None if it is supported."""
        return BLOCKED_ARCHS.get(arch)

    def archs(self) -> tuple[str, ...]:
        return SUPPORTED_ARCHS

    def has_baked_table(self, arch: str, kind: str = "params") -> bool:
        spec = self._arch(arch)
        rel = self._table_rel(spec, kind)
        return rel is not None and self._find(rel) is not None

    @staticmethod
    def _table_rel(spec: ArchSpec, kind: str) -> str | None:
        if kind == "params":
            return spec.params_table
        if kind == "names":
            return spec.names_table
        raise UnknownArchError(f"no such table kind {kind!r}; use 'params' or 'names'")

    def baked_table(self, arch: str, kind: str = "params") -> dict:
        """The tier-2 table for `arch`, as the JSON file declares it.

        Raises `NoBakedTableError` when the arch has no such table — Apollo's params
        table does not exist (`UVR.py:253` is defined, `:1714` is commented out), and
        an empty dict here would read as "the table is empty" instead.
        """
        spec = self._arch(arch)
        rel = self._table_rel(spec, kind)
        if rel is None:
            raise NoBakedTableError(
                f"arch {arch!r} has no baked {kind} table: its metadata comes from "
                f"{'the per-model tier-1 JSON only' if arch == 'apollo' else 'the engine'}")
        memo_key = (arch, kind, "")
        if memo_key in self._table_memo:
            return self._table_memo[memo_key]
        found = self._find(rel)
        if found is None:
            raise NoBakedTableError(
                f"no {kind} table for arch {arch!r} on disk: looked for "
                f"{rel} under " + ", ".join(str(r) for _, r in self._roots()))
        table = _load_json(found[2])
        self._table_memo[memo_key] = table
        return table

    def _table_with_source(self, arch: str, kind: str) -> tuple[dict, RootRole, Path, Path] | None:
        spec = self._arch(arch)
        rel = self._table_rel(spec, kind)
        if rel is None:
            return None
        found = self._find(rel)
        if found is None:
            return None
        role, root, path = found
        return self.baked_table(arch, kind), role, root, path

    # -- weights ------------------------------------------------------------ #

    def hash_of(self, weights_path: Path) -> str:
        """`hash_of` with a `(path, size, mtime)` memo — see the docstring's decision 3."""
        path = Path(weights_path)
        stat = path.stat()
        key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
        if key not in self._hash_memo:
            self._hash_memo[key] = hash_of(path)
        return self._hash_memo[key]

    def _name_to_file(self, name: str, spec: ArchSpec) -> str | None:
        """Map a model name to a file name under the arch dir, per `UVR.py`'s rules."""
        if spec.name_style == "vr":
            # UVR.py:501 -- os.path.join(VR_MODELS_DIR, f"{model_name}.pth")
            return None
        table = self._table_with_source(spec.key, "names")
        entries = table[0] if table else {}
        if spec.name_style == "mdx":
            # UVR.py:700-705 -- substring match on the display name; the mapper key
            # carries no extension for the onnx class, and '.ckpt' keys get none added.
            for file_name, display in entries.items():
                if name in str(display):
                    return file_name if Path(file_name).suffix else file_name + ".onnx"
        elif spec.name_style == "demucs":
            # UVR.py:716-719 -- exact match on the display name
            for file_name, display in entries.items():
                if name == str(display):
                    return file_name
        if name in entries:
            return name
        return None

    def _weight_candidates(self, name: str, spec: ArchSpec) -> list[str]:
        mapped = self._name_to_file(name, spec)
        raw = Path(name).name
        out: list[str] = []
        for candidate in (mapped, raw):
            if not candidate:
                continue
            for directory in spec.weight_dirs:
                if spec.name_style == "vr":
                    stem = Path(candidate).stem
                    options = [f"{stem}{suffix}" for suffix in spec.suffixes]
                    options.append(candidate)
                else:
                    options = [candidate]
                    if Path(candidate).suffix not in spec.suffixes:
                        options += [candidate + s for s in spec.suffixes if not Path(candidate).suffix]
                for option in options:
                    rel = f"{directory}/{option}"
                    if rel not in out:
                        out.append(rel)
        return out

    def _locate_weight(self, name: str, spec: ArchSpec) -> tuple[RootRole, Path, Path]:
        for rel in self._weight_candidates(name, spec):
            found = self._find(rel, weights=True)
            if found is not None:
                return found
        raise MissingWeightError(
            f"no {spec.key} weight file for {name!r}; looked in "
            f"{', '.join(str(d) for d in spec.weight_dirs)} under "
            + ", ".join(str(r) for _, r in self._roots(weights=True)))

    def _locate_any(self, name: str) -> tuple[ArchSpec, RootRole, Path, Path]:
        errors: list[str] = []
        for key in ("vr", "mdx", "demucs", "apollo"):
            spec = ARCHS[key]
            try:
                role, root, path = self._locate_weight(name, spec)
                return spec, role, root, path
            except (MissingWeightError, CatalogError) as exc:
                errors.append(f"{key}: {exc}")
        raise UnknownModelNameError(f"nothing in the catalog declares {name!r}; "
                                    + " | ".join(errors))

    # -- resolution --------------------------------------------------------- #

    def resolve(self, name: str, arch: str | None = None, *,
                required: bool = False) -> ModelSpec:
        """Resolve a model name to its metadata, reporting which root and tier answered.

        Tier 1 is `<root>/<arch>/model_data/<hash>.json`, searched `writable_dir`
        first, then `model_root`, then `fallback_root` — so a per-model JSON the user
        saved through the popup wins over the older weights-dir copy (see
        R-T10-writable: silently preferring the weights-dir copy reads as "the popup
        failed to save"). Tier 2 is the baked table, in the same order so an explicit
        refresh (Task 12) written into `writable_dir` wins. Tier 3 does not block:
        `model_data is None`, `needs_user_params=True`, and with `required=True` the
        typed `UnknownModelParamsError`.
        """
        if arch is not None:
            spec = self._arch(arch)
            role, root, weights = self._locate_weight(name, spec)
        else:
            spec, role, root, weights = self._locate_any(name)

        digest = self.hash_of(weights)
        model_data: dict | None = None
        tier = Tier.NOT_APPLICABLE
        answer_role, answer_root, source = role, root, weights
        wanted = self.writable_dir / spec.tier1_dir / f"{digest}.json"
        searched: list[Path] = []

        if spec.params_required:
            tier1_rel = f"{spec.tier1_dir}/{digest}.json"
            hit = self._find(tier1_rel)
            if hit is not None:
                answer_role, answer_root, source = hit
                model_data = _load_json(source)
                tier = Tier.PER_MODEL_JSON
            else:
                table_hit = self._table_with_source(spec.key, "params")
                if table_hit is not None:
                    table, answer_role, answer_root, source = table_hit
                    searched.append(source)
                    # UVR.py:747-749 -- `if self.model_hash in hash` (substring)
                    for key, params in table.items():
                        if digest in key:
                            model_data = params
                            tier = Tier.BAKED_TABLE
                            break
            if model_data is None:
                tier = Tier.NEEDS_USER_PARAMS
        else:
            # Demucs: the name table answers, and the params come from the engine.
            table_hit = self._table_with_source(spec.key, "names")
            if table_hit is not None:
                answer_role, answer_root, source = table_hit[1], table_hit[2], table_hit[3]
            tier = Tier.BAKED_TABLE if table_hit else Tier.NOT_APPLICABLE
            model_data = {"stem_count": 2 if DEMUCS_UVR_MODEL in name else 4}

        primary = (model_data or {}).get("primary_stem")
        try:
            secondary = secondary_stem(primary) if primary else None
        except (KeyError, TypeError, ValueError):
            secondary = None
        config_path = None
        if spec.config_dir and model_data and model_data.get("config_yaml"):
            config_path = answer_root / spec.config_dir / model_data["config_yaml"]

        result = ModelSpec(
            name=name, arch=spec.key, process_method=spec.process_method,
            primary_stem=primary, secondary_stem=secondary, weights_path=weights,
            config_path=config_path, hash=digest, model_data=model_data,
            provenance=Provenance(tier, answer_role, answer_root, source),
            needs_user_params=model_data is None and spec.params_required,
            wanted_json=wanted if model_data is None else None,
            searched_tables=tuple(searched),
        )
        if required and result.needs_user_params:
            raise UnknownModelParamsError(result)
        return result

    # -- writes ------------------------------------------------------------- #

    def write_params(self, arch: str, model_hash: str, params: dict) -> Path:
        """Save a tier-1 `<hash>.json` — into `writable_dir`, never the weights root.

        `mkdir -p` is mandatory here: upstream writes into `UVR.py:247/:249/:252`
        without creating them, so the Apollo write at `:5227` raises FileNotFoundError
        on a machine where that dir was never installed.
        """
        spec = self._arch(arch)
        target = self.writable_dir / spec.tier1_dir / f"{model_hash}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(params, indent=4), encoding="utf-8")
        self._hash_memo.clear()
        return target

    # -- listings ----------------------------------------------------------- #

    def names(self, arch: str) -> list[str]:
        """Model names that resolve *right now* — weights present in one of the roots.

        Blocked archs raise `BlockedArchError`: listing `mdx_c` from a directory full
        of config yaml would offer models this machine cannot run.
        """
        spec = self._arch(arch)
        found: list[str] = []
        for role, root in self._roots(weights=True):
            for directory in spec.weight_dirs:
                base = root / directory
                if not base.is_dir():
                    continue
                for path in sorted(base.rglob("*")):
                    if not path.is_file() or path.suffix not in spec.suffixes:
                        continue
                    token = path.stem if spec.name_style == "vr" else path.name
                    if token not in found:
                        found.append(token)
        return sorted(found)

    def known_names(self, arch: str) -> list[str]:
        """Names the baked tables declare, installed or not.

        The difference against another root is the staleness the plan measured: the
        repo tree's MDX tables carry fewer hashes and none of the newer class names.
        """
        spec = self._arch(arch)
        table = self._table_with_source(spec.key, "names")
        if table is None:
            return self.names(arch)
        return sorted(table[0])

    def uninstalled_names(self, arch: str) -> list[str]:
        return sorted(set(self.known_names(arch)) - set(self.names(arch)))

    # -- secondary model decisions (UVR.py:7046 / :7088 / :7103) -------------- #

    def secondary_for(self, spec: ModelSpec, requested: str | None) -> ModelSpec | None:
        """`process_determine_secondary_model` (`UVR.py:7046-7086`) without tkinter.

        `requested` is the name held in the slot `secondary_slot_for` selects; the
        stem-only flags and the 0-1 scale stay with `Settings` (Task 9). Upstream
        returns None when the slot says "No Model Selected" *and* when the chosen
        weight turns out to be invalid (`:7081-7082`), so a bad secondary model is
        dropped rather than raised through a run.
        """
        if not requested or requested == NO_MODEL:
            return None
        try:
            return self.resolve(requested)
        except CatalogError:
            return None

    def demucs_pre_proc_for(self, requested: str | None, *, is_activated: bool = True,
                            primary_stem: str | None = None) -> ModelSpec | None:
        """`process_determine_demucs_pre_proc_model` (`UVR.py:7088-7101`).

        One of the seven methods `ModelData` calls through `root.` — porting the
        other two but not this one would leave the dependency alive.
        """
        if not requested or requested == NO_MODEL or not is_activated:
            return None
        try:
            return self.resolve(requested, arch="demucs") if primary_stem else self.resolve(requested)
        except CatalogError:
            return None

    def vocal_split_for(self, requested: str | None, *,
                        is_activated: bool = True) -> ModelSpec | None:
        """`process_determine_vocal_split_model` (`UVR.py:7103-7114`)."""
        if not requested or requested == NO_MODEL or not is_activated:
            return None
        try:
            return self.resolve(requested)
        except CatalogError:
            return None
