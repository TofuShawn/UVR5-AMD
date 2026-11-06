"""Front-end preferences: what the GUI remembers, and the engine never reads.

A third block in the same `settings.json`, not a 51st `Settings` field. The fork decided
this in Plan 1 and said it out loud: `settings.py:27-29` excludes `input_paths`,
`export_path`, `lastDir` and `DualBatch_inputPaths` from the 50 with the note that they
are "per-run inputs (Task 11) or UI convenience state (Plan 2)", under the criterion at
`settings.py:30-31` -- *the engine reads it inside `ModelData`, or it is not a `Settings`
field*. `export_path` is upstream's own key for exactly the thing this block holds
(`gui_data/constants.py:686`, `'export_path': ''`), so the name is inherited rather than
invented, and a `settings.json` carried over from an official install keeps its meaning.

JSON, not TOML: Python's stdlib `tomllib` parses TOML and cannot write it, and a writer
would be a new third-party dependency in a fork that hand-audits every pin. The file is
already the shared home of two writers (`settings.save`, `selections.save`), each of which
read-modifies-writes the top-level object rather than rewriting it; this is the third, and
it keeps the same discipline, because a writer that rewrote the file would delete a block
it was never given.

One field, and still the whole `selections` machinery -- because the machinery is two
imported functions and the alternative is a fourth private opinion about parsing.
`selections.py:41-53` states the rule this module obeys: a value entering a typed field
parses the same way wherever it came from, or the annotations are decoration.
"""
from __future__ import annotations

import dataclasses
import json
import typing
from pathlib import Path
from types import MappingProxyType

import settings as settings_module
from selections import _coerce, _read_raw

_KEY = "ui"

#: Memoised `{field name: declared type}`, sealed, filled on first use -- the same
#: reasoning as `settings._FIELD_TYPES` and `selections._FIELD_TYPES`: `__post_init__`
#: asks on every construction and the class is closed at import.
_FIELD_TYPES: typing.Mapping[str, type] | None = None


def _field_types() -> typing.Mapping[str, type]:
    global _FIELD_TYPES
    if _FIELD_TYPES is None:
        _FIELD_TYPES = MappingProxyType(typing.get_type_hints(UiPrefs))
    return _FIELD_TYPES


@dataclasses.dataclass(frozen=True)
class UiPrefs:
    """`export_path` -- the folder a GUI run falls back to when the Output box is empty.

    Empty means `session.default_export_path()`: `UVR_out` beside the input file, which is
    also what the CLI does with no flag, so an untouched install behaves identically
    through both front ends. A non-empty value is a *GUI* override and `separate_cli` never
    reads this module -- moving a scripted run's output because someone ticked a box in the
    other front end is the kind of surprise this fork has already been burned by once.

    Coerced at construction like its two siblings (`Settings.__post_init__`,
    `Selections.__post_init__`), so `UiPrefs(export_path=7)` and the file that says `7`
    agree, and the value this class holds is always the one its own `save()` can write and
    `load()` can read back.
    """

    export_path: str = ""

    def __post_init__(self) -> None:
        types = _field_types()
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            coerced = _coerce(f"UiPrefs.{f.name}", value, types[f.name])
            if coerced is not value:
                object.__setattr__(self, f.name, coerced)


def load(path: Path | None = None) -> UiPrefs:
    """Read the `ui` block of the shared settings file; defaults if absent.

    Same contract as `settings.load` and `selections.load`: unknown keys are dropped, a
    file that is not a JSON object raises naming the path, and a known key whose value
    cannot be its declared type raises `TypeError` with the file *and* the block-qualified
    key -- a corrupt config is a file-level diagnosis, which is why the message is built
    here and not left to `__post_init__`'s constructor-flavoured one.
    """
    target = Path(path) if path is not None else settings_module.config_file()
    block = _read_raw(target).get(_KEY) or {}
    types = _field_types()
    values: dict[str, typing.Any] = {}
    for f in dataclasses.fields(UiPrefs):
        if f.name not in block:
            continue
        value, typ = block[f.name], types[f.name]
        try:
            values[f.name] = _coerce(f"UiPrefs.{f.name}", value, typ)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{target}: ui.{f.name} must be {typ.__name__}, "
                            f"got {type(value).__name__} ({exc})") from exc
    return UiPrefs(**values)


def save(prefs: UiPrefs, path: Path | None = None) -> Path:
    """Merge this block into the shared file without rewriting the other two.

    Returns the target, because the caller's console line has to name the file it wrote --
    and on Windows the everyday failure is a `settings.json` open in an editor, where the
    name is the thing the user needs in order to fix it.
    """
    target = Path(path) if path is not None else settings_module.config_file()
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = _read_raw(target)
    raw[_KEY] = dataclasses.asdict(prefs)
    target.write_text(
        json.dumps(raw, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return target
