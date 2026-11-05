"""Model selections: the inputs `plan.build()` takes that are not `Settings` fields.

Two groups live here. The 26 values R-T11-selections kept out of `Settings` (12
secondary-model names, 12 scales, the demucs pre-proc model, the vocal splitter), and
the 7 that address them (4 main-model picks, the splitter's activation switch, and the
two stem-only checkbox *labels* -- `plan.py:452-456` says those labels are checkbox
text with no `Settings` field, and `plan.build` reads them out of the vocal-splitter
entry).

Upstream's source of truth for the names is the tk var set in `UVR.py` (e.g.
`self.vr_voc_inst_secondary_model_var`); the mapping is reproduced field by field
below, not re-derived.
"""
from __future__ import annotations

import dataclasses
import json
import typing
from pathlib import Path

import catalog
import settings as settings_module

_GROUPS = ("voc_inst", "other", "drums", "bass")          # catalog._stem_group's names
_ARCHS = ("vr", "mdx", "demucs")


@dataclasses.dataclass(frozen=True)
class Selections:
    # main model per arch (these feed Catalog.resolve, not plan.build)
    vr_model: str = catalog.NO_MODEL
    mdx_net_model: str = catalog.NO_MODEL
    demucs_model: str = catalog.NO_MODEL
    apollo_model: str = catalog.NO_MODEL
    # secondary models and their scales, per arch and per stem group
    vr_voc_inst_secondary_model: str = catalog.NO_MODEL
    vr_voc_inst_secondary_model_scale: float = 0.5
    vr_other_secondary_model: str = catalog.NO_MODEL
    vr_other_secondary_model_scale: float = 0.5
    vr_drums_secondary_model: str = catalog.NO_MODEL
    vr_drums_secondary_model_scale: float = 0.5
    vr_bass_secondary_model: str = catalog.NO_MODEL
    vr_bass_secondary_model_scale: float = 0.5
    mdx_voc_inst_secondary_model: str = catalog.NO_MODEL
    mdx_voc_inst_secondary_model_scale: float = 0.5
    mdx_other_secondary_model: str = catalog.NO_MODEL
    mdx_other_secondary_model_scale: float = 0.5
    mdx_drums_secondary_model: str = catalog.NO_MODEL
    mdx_drums_secondary_model_scale: float = 0.5
    mdx_bass_secondary_model: str = catalog.NO_MODEL
    mdx_bass_secondary_model_scale: float = 0.5
    demucs_voc_inst_secondary_model: str = catalog.NO_MODEL
    demucs_voc_inst_secondary_model_scale: float = 0.5
    demucs_other_secondary_model: str = catalog.NO_MODEL
    demucs_other_secondary_model_scale: float = 0.5
    demucs_drums_secondary_model: str = catalog.NO_MODEL
    demucs_drums_secondary_model_scale: float = 0.5
    demucs_bass_secondary_model: str = catalog.NO_MODEL
    demucs_bass_secondary_model_scale: float = 0.5
    # demucs pre-proc, and the vocal splitter with its two checkbox labels
    demucs_pre_proc_model: str = catalog.NO_MODEL
    set_vocal_splitter: str = catalog.NO_MODEL
    is_set_vocal_splitter: bool = False
    vocal_splitter_primary_label: str = ""
    vocal_splitter_secondary_label: str = ""

    def to_plan_dict(self) -> dict:
        """The nested `selections` argument `plan.build` documents at plan.py:741-750.

        An all-default instance returns `{}` -- which is precisely what
        `separate_cli.run()` passes today, so a GUI run that touches nothing behaves
        like a CLI run. Once *anything* is selected the frame carries all three arch
        keys (`build`'s docstring shows the shape with them present, and
        `test_to_plan_dict_matches_the_shape_plan_build_documents` pins that): a slot
        with no model in it is simply absent from its group, and `plan.py:277`
        (`selections.get(group) or {}`) plus `UVR.py:7075/:7084` read an empty group,
        a missing group and `NO_MODEL` identically -- so the always-present keys are
        shape, not behaviour.
        """
        frame: dict[str, dict] = {}
        for arch in _ARCHS:
            group: dict[str, typing.Any] = {}
            for stem in _GROUPS:
                name = getattr(self, f"{arch}_{stem}_secondary_model")
                if name and name != catalog.NO_MODEL:
                    group[f"{stem}_secondary_model"] = name
                    group[f"{stem}_secondary_model_scale"] = getattr(
                        self, f"{arch}_{stem}_secondary_model_scale")
            frame[arch] = group
        out: dict[str, typing.Any] = {}
        if self.demucs_pre_proc_model and self.demucs_pre_proc_model != catalog.NO_MODEL:
            out["demucs_pre_proc_model"] = self.demucs_pre_proc_model
        if self.set_vocal_splitter and self.set_vocal_splitter != catalog.NO_MODEL:
            out["vocal_splitter"] = {
                "name": self.set_vocal_splitter,
                "is_activated": bool(self.is_set_vocal_splitter),
                "primary_stem_only_label": self.vocal_splitter_primary_label,
                "secondary_stem_only_label": self.vocal_splitter_secondary_label,
            }
        if not out and not any(frame.values()):
            return {}
        out.update(frame)
        return out


SELECTION_FIELD_NAMES: frozenset[str] = frozenset(
    f.name for f in dataclasses.fields(Selections))

_KEY = "selections"


def _read_raw(path: Path) -> dict:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"{path}: settings.json must hold an object, got {type(raw).__name__}")
    return raw


def load(path: Path | None = None) -> Selections:
    """Read the `selections` block of the shared settings file; defaults if absent.

    Same contract as `settings.load`: unknown keys are dropped, but a known key whose
    value cannot be its declared type **raises** -- silently keeping a corrupt value is
    how a wrong-type field reaches the engine. `isinstance` alone would not do it,
    because `True` is an `int` and `1` is not a `bool`.
    """
    target = Path(path) if path is not None else settings_module.config_file()
    block = _read_raw(target).get(_KEY) or {}
    types = typing.get_type_hints(Selections)
    values: dict[str, typing.Any] = {}
    for f in dataclasses.fields(Selections):
        if f.name not in block:
            continue
        value, typ = block[f.name], types[f.name]
        ok = (typ is bool and isinstance(value, bool)
              or typ is int and isinstance(value, int) and not isinstance(value, bool)
              or typ is float and isinstance(value, (int, float)) and not isinstance(value, bool)
              or typ is str and isinstance(value, str))
        if not ok:
            raise TypeError(f"{target}: selections.{f.name} must be {typ.__name__}, "
                            f"got {type(value).__name__}")
        values[f.name] = value
    return Selections(**values)


def save(sel: Selections, path: Path | None = None) -> None:
    """Merge this block into the shared file without rewriting the settings half."""
    target = Path(path) if path is not None else settings_module.config_file()
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = _read_raw(target)
    raw[_KEY] = dataclasses.asdict(sel)
    target.write_text(json.dumps(raw, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                      encoding="utf-8")
