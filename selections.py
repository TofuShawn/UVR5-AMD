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
import math
import typing
from pathlib import Path
from types import MappingProxyType

import catalog
import settings as settings_module

_GROUPS = ("voc_inst", "other", "drums", "bass")          # catalog._stem_group's names
_ARCHS = ("vr", "mdx", "demucs")

#: Memoised `{field name: declared type}`, sealed, filled on the first `_field_types()`
#: call. Same reason as `settings._FIELD_TYPES`: `__post_init__` asks for it on every
#: construction, and the class is closed at import.
_FIELD_TYPES: typing.Mapping[str, type] | None = None


def _coerce(field_name: str, value: object, typ: type) -> typing.Any:
    """Parse one selection into the field's declared type. This module's one table.

    The mechanism is `settings._coerce`'s (settings.py:247-332), reused rather than
    re-invented so that a value entering a typed field parses the same way wherever it
    came from -- file, widget text, or a bare constructor argument. Without it the
    annotations on `Selections` are decoration: the Task 5/6 UI builds this class from
    widget text, so `Selections(vr_voc_inst_secondary_model_scale='0.7')` *held* the
    `str` `'0.7'` in a field annotated `float`, `save()` wrote `"0.7"`, and the next
    `load()` raised `TypeError` (see its float gate) -- the app poisoning its own
    config. The bare constructor is made safe here instead of adding a
    `from_widget_values()` factory beside the unsafe one, exactly as
    `Settings.__post_init__` (settings.py:443-481) settled for the sibling class.

    Three branches, one of them a deliberate divergence from the sibling table:

      * `bool` -- **only a real `bool` passes; anything else raises.** This is the one
        place the table is *stricter* than `settings._to_bool`
        (settings.py:211-229), which parses `0`/`1` and the tokens `'true'`/`'False'`
        because those are genuine `data.pkl` and `tk` encodings for *those* flags. Two
        reasons it cannot do that here: this module's `load()` calls this very function,
        so a constructor accepting `1` or `'False'` would let the class hold what its own
        file cannot be read back as; and
        `to_plan_dict()` re-coerces this field with `bool(...)` (see its
        `vocal_splitter` entry), so the *truthy text* `'False'` would not raise, not
        store `False`, and not stay quiet -- it would silently activate a second model
        in the plan. No truthifying, no guessing.
      * `float` -- text parses (`'0.7'` -> `0.7`) and an `int` widens, because JSON has
        one number type; unparseable text **raises** rather than becoming `0.0`; a
        `bool` raises (a bool is not a scale -- it is the very confusion
        `test_a_wrongly_typed_selection_raises_instead_of_silently_passing` pins);
        PEP 515 separators in text raise (`float('1_0')` parses, and no widget text or
        JSON number can contain the underscore); and `nan`/`inf` raise, since neither
        strict JSON (this module's `save()` writes `allow_nan=False`) nor the class's
        own round trip carries them -- `nan != nan` would make
        `constructed == load(save(constructed))` False by itself. Mirrors
        settings.py:311-323.
      * `str` -- a `str` passes through as the **very object given**, so model names
        and the two checkbox labels stay verbatim (whitespace included); `bool`/`int`/
        `float` are stringified, because that is the one rewrite a widget performs on
        its own numeric content; anything else raises instead of rendering as `'[]'`.
        Same as settings.py:324-331.

    `None` raises in every branch: no widget stores a null in these 33 fields.
    """
    if value is None:
        raise ValueError(f"Selections.{field_name}: null is not a value any widget stores")
    if typ is bool:
        if isinstance(value, bool):
            return value
        raise TypeError(f"Selections.{field_name}: {value!r} is a "
                        f"{type(value).__name__}, not a bool; this module's loader "
                        f"refuses one too, and a truthy non-bool would reach "
                        f"to_plan_dict()'s bool(...) and activate a model nobody chose")
    if typ is float:
        if isinstance(value, bool):
            raise ValueError(f"Selections.{field_name}: a bool is not a float")
        if isinstance(value, str) and "_" in value:
            raise ValueError(f"Selections.{field_name}: {value!r} uses a Python digit "
                             f"separator, which no widget text or JSON number can contain")
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Selections.{field_name}: {value!r} is not a float ({exc})") from exc
        if not math.isfinite(parsed):
            raise ValueError(f"Selections.{field_name}: {value!r} is not a finite float "
                             f"(neither this module's round trip nor strict JSON carries "
                             f"NaN or Infinity)")
        return parsed
    if typ is str:
        if isinstance(value, str):
            return value
        if isinstance(value, (bool, int, float)):
            return str(value)
        raise TypeError(f"Selections.{field_name}: {type(value).__name__} is not text a "
                        f"widget can hold, and a str field does not render objects")
    raise TypeError(f"Selections.{field_name}: unsupported type {typ!r}")


@dataclasses.dataclass(frozen=True)
class Selections:
    """The 33 model selections `plan.build()` takes that are not `Settings` fields.

    Frozen, all-defaulted, and coerced at construction: `__post_init__` re-parses
    every field through `_coerce`, so the declared types are enforced rather than
    advertised, and `Selections(**{...scale: '0.7'})` equals what `load()` returns for
    the same logical value. Same contract `Settings` carries (settings.py:443-481).
    """
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

    def __post_init__(self) -> None:
        """Re-parse every field into its declared type, in place (fix round 1).

        `Settings` gained this for the same reason (settings.py:443-481), and the
        argument carries over verbatim: Plan 2's UI builds both classes from widget
        text, and only one of them had a gate. Task 6's `apply(path)` validates the
        settings half by constructing `Settings(**values)` and letting
        `__post_init__` be the validator; the selections half had no such gate, so the
        settings page could write a `settings.json` whose selections block
        `load()` then refuses to read back. Putting it here rather than in a Qt widget
        makes Task 6 correct by construction (controller ruling R-T3-coerce).

        See `_coerce` for what each branch accepts and refuses; in short -- a numeric
        string in a scale field becomes the float, a non-numeric one **raises**
        instead of becoming `0.0`, and the splitter switch takes only a real `bool`:
        `'False'`, `0`, `[]` and `''` all raise, none of them truthified into
        activating a second model.

        `object.__setattr__` because the instance is frozen -- `self.x = ...` here
        would raise `FrozenInstanceError` -- and only when the parse produced a
        different object, so a `str` field keeps its identity and a value already of
        the declared type is left alone. That makes the coercion idempotent, which is
        what the finding's round trip needs: every value `load()` accepts is already in
        range, so `Selections(**asdict(load(p))) == load(p)` and
        `load(save(constructed)) == constructed` both hold.
        """
        types = _field_types()
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            coerced = _coerce(f.name, value, types[f.name])
            if coerced is not value:
                object.__setattr__(self, f.name, coerced)

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


def _field_types() -> typing.Mapping[str, type]:
    """`{field name: declared type}`, resolved once, cached, and handed out sealed.

    `settings._field_types()` (settings.py:511-527) is the model, including the
    `MappingProxyType`: `__post_init__` calls this for every instance, and handing out
    the mutable cache would let one caller rewrite what every later construction
    coerces *to* -- which is exactly the round-1 defect that fix closed there.
    """
    global _FIELD_TYPES
    if _FIELD_TYPES is None:
        _FIELD_TYPES = MappingProxyType(typing.get_type_hints(Selections))
    return _FIELD_TYPES


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
    how a wrong-type field reaches the engine.

    The parsing is `_coerce`'s, so the two halves of `settings.json` agree on the domain
    (finding I9). This loader used to run its own `isinstance` gate, which refused the
    text `'0.7'` in a float field that `_coerce` parses and `settings.load` (settings.py
    :546) accepts: the promise at the top of `_coerce`'s docstring -- the same parse
    "wherever it came from" -- held for the constructor and not for the file, and the
    asymmetry was reachable. `Controller.__init__` calls this before a window exists, so
    a hand-edited scale was an app that would not launch with no surface to say so on.

    What stays here rather than being left to `__post_init__` is the *wrapping*: this
    raises `TypeError` naming the file and the key, because a corrupt config is a
    file-level diagnosis, while `_coerce` speaks for a constructor argument. The domain
    they agree on is `_coerce`'s, including its stricter bool branch -- `1` and `'False'`
    are refused here exactly as `Selections(...)` refuses them, so nothing this accepts
    can fail to survive its own round trip.
    """
    target = Path(path) if path is not None else settings_module.config_file()
    block = _read_raw(target).get(_KEY) or {}
    types = _field_types()
    values: dict[str, typing.Any] = {}
    for f in dataclasses.fields(Selections):
        if f.name not in block:
            continue
        value, typ = block[f.name], types[f.name]
        try:
            values[f.name] = _coerce(f.name, value, typ)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{target}: selections.{f.name} must be {typ.__name__}, "
                            f"got {type(value).__name__} ({exc})") from exc
    return Selections(**values)


def save(sel: Selections, path: Path | None = None) -> None:
    """Merge this block into the shared file without rewriting the settings half."""
    target = Path(path) if path is not None else settings_module.config_file()
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = _read_raw(target)
    raw[_KEY] = dataclasses.asdict(sel)
    target.write_text(json.dumps(raw, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                      encoding="utf-8")
