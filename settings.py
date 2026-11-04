"""settings.py -- the settings the engine actually reads, as typed immutable data.

Task 9 of the ROCm-engine plan; the first task on the *app* side of the seam.
Tasks 1-8 rebuilt the engine (`lib_v5/`, `demucs/`, `separate.py`, `backends.py`)
in place, and this module is what replaces the tkinter App's `root.*_var.get()`
reads that `ModelData` (`UVR.py:365-804`) performs. Task 11's `RunPlan` consumes
the `Settings` instance; the engine never imports this module (the dependency
rule points one way, and Task 8's purity ratchet guards the engine, so
`settings.py` is deliberately *not* in that ratchet's glob).

Field list -- 50, and not one more
----------------------------------
`ModelData` contains 77 `root.` references across 63 distinct names: 49 `*_var`,
4 `*_MAPPER`, 7 genuinely-called methods, 3 plain attributes
(`wav_type_set`, `vr_model_params`, `mdx_model_params`). `Settings` is the 49
`*_var` reads **plus** `wav_type_set`. `tests/test_settings.py` re-derives that
set from `UVR.py` at test time; the count here is a comment, not a source.

Excluded on purpose (rulings R-T9-params / R-T9-paths):
  * `vr_hash_MAPPER`, `mdx_hash_MAPPER`, `mdx_name_select_MAPPER`,
    `demucs_name_select_MAPPER`, and the secondary/chain/vocal-splitter decision
    methods -- model knowledge, which is `ModelCatalog` (Task 10).
  * `vr_model_params` / `mdx_model_params` -- the modal dialog's out-parameter
    channel (`UVR.py:5165` / `:5425` write, `:773` / `:776` read back), not
    lazily-cached user state, and nothing in `separate.py` or `lib_v5/**` ever
    sees them.
  * `input_paths`, `export_path`, `lastDir`, `fileOneEntry*`,
    `DualBatch_inputPaths` -- real `DEFAULT_DATA` keys that `ModelData` never
    reads. They are per-run inputs (Task 11) or UI convenience state (Plan 2).
  The criterion for all of them: *the engine reads it inside `ModelData`, or it
  is not a `Settings` field.*

Defaults
--------
Not one default in this file is a literal. Every value is resolved from
`gui_data.constants.DEFAULT_DATA` at import time by `_data_default`, because the
table cannot be read as data: of its 129 entries, 39 values are not literals
(29 bare constant references such as `batch_size = DEF_OPT` and
`chosen_process_method = MDX_ARCH_TYPE`, 7 subscripts such as
`segment = DEMUCS_SEGMENTS[0]`, plus `input_paths = []`,
`DualBatch_inputPaths = []`, `model_hash_table = {}`), so
`ast.literal_eval` on the dict raises `ValueError: malformed node ... <ast.Name>`.
Two of the 50 fields have **no `DEFAULT_DATA` key at all** --
`ensemble_main_stem` (`UVR.py:1669`, `tk.StringVar(value=CHOOSE_STEM_PAIR)`) and
`mdxnet_stems` (`:1673`, `value=ALL_STEMS`) -- so those two come from the App
constructor via `_widget_default`, which is stated in the code, not hidden.

The one field whose name differs from its table key is `gpu_conversion`, backed
by the key `is_gpu_conversion`; `data_key` keeps that mapping visible instead of
pretending the key matches the field.

Types (ruling R-T9-types): by stored domain, not by the consumer's coercion
--------------------------------------------------------------------------
`ModelData` coerces after reading -- `int(root.crop_size_var.get())`,
`float(root.semitone_shift_var.get())` -- but what this type persists is what the
widget holds and what `data.pkl` held. Measured buckets, confirmed by which
tkinter class backs each var (the 24 `tk.BooleanVar` in `UVR.py` are exactly the
`bool` fields, the 25 `tk.StringVar` are the rest):

    24 bool   the `is_*` / `*_activate` group
    8 int/float  aggression_setting, crop_size, margin, margin_demucs,
                 mdx_segment_size, shifts, window_size (int); post_process_threshold (float)
   18 str     everything else, including `wav_type_set`

Consequences worth naming:
  * `overlap_mdx23` and `semitone_shift` are `str`, because the table stores the
    *text* `'8'` and `'0'` (`DEFAULT_DATA['overlap_mdx23'] = '8'`) even though
    `UVR.py:395-396` immediately does `int(float(...))` / `float(...)`. A float
    default would equal no entry in the combobox list that produced the value,
    and Plan 2 would show a silently unselected dropdown.
  * `batch_size`, `mdx_batch_size`, `compensate`, `overlap`, `overlap_mdx` are
    `str` because their domain is *sentinel-or-number*: every raw read is a
    sentinel comparison first (`== DEF_OPT` at `:391`/`:496`, `== AUTO_SELECT` at
    `:584`, `== DEFAULT` at `:393-394`), so a field typed `int`/`float` could not
    hold the value the GUI persists. The coercion is Task 11's job.
  * `overlap` is the one field whose table value is a float
    (`DEMUCS_OVERLAP[0]` is `0.25`, not a string) on a field typed `str`. The
    `tk.StringVar` at `UVR.py:7378` normalises it to the text `'0.25'` before
    `ModelData` ever sees it, and that text is what `:7678` persists, so the
    field stays `str` and the default is the stringified table value -- resolved,
    not typed in.
  * `window_size` and `crop_size` are `tk.StringVar` (`:7356`/`:7359`) yet typed
    `int`, because `int` is what the engine consumes (`:495`/`:497`). This is the
    one place the rule is applied on the consumer's side, and it is why the
    legacy import coerces rather than copying -- and why `__post_init__` coerces:
    a UI hands this type the widget *text*, and an instance that kept `'352'` in
    an `int` field would compare unequal to the `352` the same file loads back.

What the coercion refuses (fix round 2)
---------------------------------------
`_coerce` is the only place a value changes type, and its docstring carries the
boundaries it rules on, all three of them tested in `tests/test_settings.py`: a `str`
field takes text, or the number/`bool` a `tk.StringVar` would have stringified, and
**raises** on anything that has to be *rendered* -- `Settings(device_set=[])` used to
hold `'[]'`; an `int` field admits a whole-number float (`352.0` -> `352`, exact and
lossless) but refuses Python's digit separators (`'1_000'` parses to `1000`, which is
not the text that was typed); and no numeric field takes `nan`/`inf`, numeric or text,
because such a value alone breaks `constructed == load(save(constructed))` (nan != nan)
and `json.dumps` writes it as a bare `NaN`, which is not JSON an ECMA-404 parser
accepts. `save()` passes `allow_nan=False`, so the writer holds that line too. The
field-type table `__post_init__` coerces against is handed out as a `MappingProxyType`,
so no caller can rewrite what every later construction reads.

`device_set` (ruling R-T9-device): one raw string
-------------------------------------------------
`Settings.device_set` is the combobox value verbatim. The same widget is filled
from two different value sets, so the persisted string has at least four shapes:
the `DEFAULT` sentinel (`"Default"`, `constants.py:31`), a bare index `'0'..'8'`
(`GPU_DEVICE_NUM_OPTS`, `constants.py:522`), and `"<device name>:<index>"` built
at `UVR.py:3546` (CUDA/ROCm) or `:3551` (DirectML). There is no `device_index`
field and no `:` split here -- `ModelData` performs that at `:410`/`:956` and the
split moves to Task 11's `RunPlan`.

This module imports no `torch` on purpose. `UVR.py:3560-3561` resets a stored
value that is not in the *currently detected* device list; that is right for a
combobox and wrong for a persisted file, because it would silently rewrite a
saved multi-GPU choice on a one-card machine. Worse, which list is authoritative
depends on `is_use_directml` (`:3559`), so one stored string can be valid with
DirectML on and invalid with it off -- `Settings` stores both raw and reconciles
nothing at load. Reconciliation needs the live device list, which is
`backends.py`'s knowledge; Task 11 owns the explicit, logged reset.

Persistence
-----------
JSON at `%APPDATA%\\UVR-fork\\settings.json` (`APPDATA` unset or empty falls back
to `~/.config/UVR-fork`), never CWD-relative: upstream does `os.chdir(BASE_PATH)`
at `UVR.py:83` and opens a bare `'data.pkl'` at `:172`/`:183`, so a relative path
here would silently read the install directory's file.

`import_legacy_data_pickle` is the one-time, read-only bridge from the official
build's `data.pkl`. That format stores `tk.StringVar` numbers as text and can
carry booleans as `0`/`1`, so it coerces. For `is_gpu_conversion` in particular,
`0` means GPU **on**: `UVR.py:384` computes `0 if var.get() else -1` and the
golden harness mirrors it (`tests/golden/settings.py:254`), so a naive
`bool(0) == False` is the wrong import. Any legacy key this module does not
understand is dropped, not guessed, and a key the legacy file does not mention
keeps its `DEFAULT_DATA` default.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import pickle
import typing
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from gui_data.constants import ALL_STEMS, CHOOSE_STEM_PAIR, DEFAULT_DATA

__all__ = ["Settings", "config_dir", "config_file", "load", "import_legacy_data_pickle",
           "data_key", "field_for_data_key"]

#: Directory under `%APPDATA%` (or `~/.config`) holding the fork's settings.
_DIR_NAME = "UVR-fork"
_FILE_NAME = "settings.json"

#: The single field whose name is not its `DEFAULT_DATA` key.
_KEY_OVERRIDES = {"gpu_conversion": "is_gpu_conversion"}

#: Fields with no `DEFAULT_DATA` key; their default is the App constructor's.
_APP_CTOR_FIELDS = ("ensemble_main_stem", "mdxnet_stems")

_TYPES: dict[str, type] = {"bool": bool, "int": int, "float": float, "str": str}

#: Memoised `{field name: type}`, sealed, filled on the first `_field_types()` call.
#: A read-only view, because the callers below hand it out to every construction; see
#: `_field_types`.
_FIELD_TYPES: typing.Mapping[str, type] | None = None

_TRUE_TOKENS = frozenset({"True", "true", "1"})
_FALSE_TOKENS = frozenset({"False", "false", "0", ""})

#: `UVR.py:384` -- `self.is_gpu_conversion = 0 if root.is_gpu_conversion_var.get() else -1`
_GPU_ON = 0
_GPU_CPU = -1


def config_dir() -> Path:
    """Where the fork keeps its settings: `%APPDATA%`/UVR-fork, else `~/.config`.

    An unset, empty or whitespace-only `APPDATA` all fall back, the same way
    `uvr_paths._env_path` treats `UVR_MODEL_ROOT`. Without the `or ""` a missing
    variable raised `KeyError` on a POSIX box; without `.strip()` a
    whitespace-only value built the relative-looking path `'  /UVR-fork'`, and
    `os.chdir(BASE_PATH)` at `UVR.py:83` decides where that actually lands.
    """
    base = (os.environ.get("APPDATA") or "").strip() or str(Path.home() / ".config")
    return Path(base) / _DIR_NAME


def config_file() -> Path:
    return config_dir() / _FILE_NAME


def data_key(field_name: str) -> str:
    """The `DEFAULT_DATA` / legacy `data.pkl` key behind a field name."""
    return _KEY_OVERRIDES.get(field_name, field_name)


def field_for_data_key(key: str) -> str | None:
    """The field a legacy/data key maps to, or `None` (the key is then dropped)."""
    for f in dataclasses.fields(Settings):
        if data_key(f.name) == key:
            return f.name
    return None


def _to_bool(field_name: str, value: object) -> bool:
    """Parse the representations this type's own domains actually contain.

    JSON writes real booleans; a legacy `data.pkl` may hold `0`/`1`; a hand-edited
    file may hold `"true"`. Anything else is a guess, so it raises.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value in (0, 1):
            return bool(value)
        raise ValueError(f"Settings.{field_name}: {value!r} is not a 0/1 flag")
    if isinstance(value, str):
        if value in _TRUE_TOKENS:
            return True
        if value in _FALSE_TOKENS:
            return False
        raise ValueError(f"Settings.{field_name}: {value!r} is not a boolean token")
    raise TypeError(f"Settings.{field_name}: {type(value).__name__} is not a stored flag")


def _reject_digit_separators(field_name: str, value: object) -> None:
    """Refuse PEP 515 underscore separators in a numeric field's text.

    `int('1_000')` and `float('1_0.5')` both parse, so without this a number written
    with separators would silently arrive as a *different number* than the text the user
    typed. Underscores are a Python source-code affordance: no `tk.StringVar` produces
    them and no JSON number may contain one, so nothing that can reach this module from
    a widget or a file loses anything by the refusal. Whitespace stays accepted --
    `int(' 1024 ')` -- because padded widget text is real. (fix round 2, M2, ruled.)
    """
    if isinstance(value, str) and "_" in value:
        raise ValueError(f"Settings.{field_name}: {value!r} uses a Python digit "
                         f"separator, which no widget text or JSON number can contain")


def _coerce(field_name: str, value: object, typ: type) -> typing.Any:
    """Parse one value into the field's declared type. No other magic.

    This is the single coercion table: `load()`, `import_legacy_data_pickle()`,
    `_data_default()`/`_widget_default()` and `Settings.__post_init__()` all route
    through it, so a value that reaches an `int` field as text parses the same way
    wherever it came from. `Settings` performs no coercion beyond
    bool/int/float/str parsing of its own JSON and its own constructor arguments: it
    never resolves a sentinel, never splits a device string, never validates against
    hardware, and never rewrites a string it was already given.

    What counts as *uncoercible* is decided here, once. The boundary is deliberately
    narrower than Python's own parsers at the three points where their generosity would
    have stored a value this format cannot round-trip (all three ruled in fix round 2,
    each tested in `tests/test_settings.py`):

      * `str` -- a `str` passes through **as the very object given** (identity, not an
        equal copy: R-T9-types and R-T9-device hold the text verbatim, whitespace
        included). A `bool`/`int`/`float` is stringified, because that is the one
        rewrite a `tk.StringVar` performs on its own content -- `DEFAULT_DATA['overlap']`
        is the float `0.25` and the widget's text is `'0.25'`. Anything else **raises
        `TypeError`**: rendering a `list`, `dict`, `set`, `tuple`, `bytes` or an
        arbitrary object produced values like `'[]'` and `'<object object at 0x...>'`,
        which contradict `load()`'s promise never to keep a corrupt value and are
        inconsistent with `_data_default`, which already refuses a non-scalar table
        value for the same reason. Every one of the 50 fields' `DEFAULT_DATA` values is
        a scalar, so no real legacy path is lost. (A non-finite float handed to a `str`
        field still becomes its text, `'nan'` -- strict JSON quotes that, and the same
        characters arriving *as* text are not rejectable without banning widget content,
        so it is not the hole M2 closes.)
      * `int` -- text parses (`'352'`, `' 1024 '`); a whole-number `float` is admissible
        because the map is exact and JSON has one number type, so a file writing `352.0`
        for an integer setting means 352 (`352.0` -> `352`, `352.5` raises); a `bool` is
        never an `int`; digit separators raise (see `_reject_digit_separators`); and a
        non-finite float raises rather than producing `int(nan)`'s `ValueError` or
        `int(inf)`'s raw `OverflowError`.
      * `float` -- same rules, plus: **`nan`, `inf` and `-inf` raise**, in either their
        numeric or their text form (`'nan'`, `'Infinity'`). They are not values this
        format can store. A non-finite field broke the module's own round trip
        (`constructed == load(save(constructed))` is False for nan, since nan != nan),
        and `json.dumps` writes one as a bare `NaN`, which is not ECMA-404 JSON -- an
        editor, a preset tool, or a harness in another language can refuse the file our
        `save()` produced. `save()` now passes `allow_nan=False` so the writer refuses
        too, for the one route `_coerce` cannot police: a forced attribute on a frozen
        instance.
    """
    if value is None:
        raise ValueError(f"Settings.{field_name}: null is not a value any widget stores")
    if typ is bool:
        return _to_bool(field_name, value)
    if typ is int:
        if isinstance(value, bool):
            raise ValueError(f"Settings.{field_name}: a bool is not an int")
        _reject_digit_separators(field_name, value)
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"Settings.{field_name}: {value!r} is not a finite number")
        try:
            # tk.StringVar stored these as text, so "512" is legitimate input.
            parsed = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Settings.{field_name}: {value!r} is not an int ({exc})") from exc
        if isinstance(value, float) and value != parsed:
            raise ValueError(f"Settings.{field_name}: {value!r} is not a whole number")
        return parsed
    if typ is float:
        if isinstance(value, bool):
            raise ValueError(f"Settings.{field_name}: a bool is not a float")
        _reject_digit_separators(field_name, value)
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Settings.{field_name}: {value!r} is not a float ({exc})") from exc
        if not math.isfinite(parsed):
            raise ValueError(f"Settings.{field_name}: {value!r} is not a finite float "
                             f"(neither this module's round trip nor strict JSON carries "
                             f"NaN or Infinity)")
        return parsed
    if typ is str:
        # Numbers arrive as text here: the stored domain of a tk.StringVar is str.
        if isinstance(value, str):
            return value
        if isinstance(value, (bool, int, float)):
            return str(value)
        raise TypeError(f"Settings.{field_name}: {type(value).__name__} is not text a "
                        f"widget can hold, and a str field does not render objects")
    raise TypeError(f"Settings.{field_name}: unsupported type {typ!r}")


def _data_default(key: str, type_name: str, field_name: str) -> typing.Any:
    """Resolve a field default out of `DEFAULT_DATA`, in the constants namespace.

    Called in the class body, so the value is the *resolved* constant
    (`DEF_OPT` -> `'Default'`, `DEMUCS_SEGMENTS[0]` -> `'Default'`,
    `DEMUCS_OVERLAP[0]` -> `'0.25'`), never the name. `KeyError` is a bug in the
    field list -- the key really is not in the table. The `isinstance` test below
    uses `is`/`isinstance` rather than `==`, because
    `DEFAULT_DATA['is_gpu_conversion']` is `False` and `== 0` would read that as a
    match; it rejects a mutable table value, since a `[]` or `{}` here would hand a
    frozen dataclass a shared mutable default -- the per-model `dict` field
    R-T9-params exists to prevent. Every value it resolves, for all 50 fields, is a
    scalar, and `tests/test_settings.py` checks that too.
    """
    if key not in DEFAULT_DATA:
        raise KeyError(f"DEFAULT_DATA has no key {key!r} for Settings.{field_name}")
    raw = DEFAULT_DATA[key]
    if raw is dataclasses.MISSING or not isinstance(raw, (bool, int, float, str)):
        raise TypeError(f"DEFAULT_DATA[{key!r}] is not a scalar; Settings.{field_name}"
                        f" must not gain a mutable or absent default")
    return _coerce(field_name, raw, _TYPES[type_name])


def _widget_default(value: object, type_name: str, field_name: str) -> typing.Any:
    """Default for a field `DEFAULT_DATA` does not carry, from the App constructor.

    Only `ensemble_main_stem` (`UVR.py:1669`) and `mdxnet_stems` (`:1673`) use
    this; the value passed in is the same constant the tkinter App initialises the
    var with, referenced by name so it stays traceable.
    """
    if field_name not in _APP_CTOR_FIELDS:
        raise KeyError(f"Settings.{field_name} has a DEFAULT_DATA key; use _data_default")
    return _coerce(field_name, value, _TYPES[type_name])


@dataclass(frozen=True)
class Settings:
    """The 50 settings `ModelData` reads, typed by their stored domain.

    Frozen because every one of the 49 vars is read via `.get()` inside
    `ModelData` -- never `.set()` and never traced -- so a snapshot is the right
    shape. Order follows the measured `*_var` list (alphabetical by var name),
    then `wav_type_set`, so the block diffs against the derivation script's
    output line for line; positional construction is therefore discouraged in
    favour of keywords.

    The declared types are enforced, not advertised: `__post_init__` re-parses
    every field through `_coerce`, so `Settings(crop_size='352')` holds the `int`
    `352` and equals what `load()` returns for the same file, while the `str`
    fields keep their text verbatim. See that method for the three constraints.
    """

    aggression_setting: int = _data_default("aggression_setting", "int", "aggression_setting")
    batch_size: str = _data_default("batch_size", "str", "batch_size")
    chosen_process_method: str = _data_default("chosen_process_method", "str", "chosen_process_method")
    compensate: str = _data_default("compensate", "str", "compensate")
    crop_size: int = _data_default("crop_size", "int", "crop_size")
    demucs_is_secondary_model_activate: bool = _data_default("demucs_is_secondary_model_activate", "bool", "demucs_is_secondary_model_activate")
    demucs_stems: str = _data_default("demucs_stems", "str", "demucs_stems")
    denoise_option: str = _data_default("denoise_option", "str", "denoise_option")
    deverb_vocal_opt: str = _data_default("deverb_vocal_opt", "str", "deverb_vocal_opt")
    device_set: str = _data_default("device_set", "str", "device_set")
    # No DEFAULT_DATA key: UVR.py:1669 initialises it with CHOOSE_STEM_PAIR.
    ensemble_main_stem: str = _widget_default(CHOOSE_STEM_PAIR, "str", "ensemble_main_stem")
    is_chunk_demucs: bool = _data_default("is_chunk_demucs", "bool", "is_chunk_demucs")
    is_demucs_combine_stems: bool = _data_default("is_demucs_combine_stems", "bool", "is_demucs_combine_stems")
    is_demucs_pre_proc_model_activate: bool = _data_default("is_demucs_pre_proc_model_activate", "bool", "is_demucs_pre_proc_model_activate")
    is_demucs_pre_proc_model_inst_mix: bool = _data_default("is_demucs_pre_proc_model_inst_mix", "bool", "is_demucs_pre_proc_model_inst_mix")
    is_deverb_vocals: bool = _data_default("is_deverb_vocals", "bool", "is_deverb_vocals")
    # The DEFAULT_DATA key behind this field is 'is_gpu_conversion', not 'gpu_conversion'.
    gpu_conversion: bool = _data_default("is_gpu_conversion", "bool", "gpu_conversion")
    is_high_end_process: bool = _data_default("is_high_end_process", "bool", "is_high_end_process")
    is_invert_spec: bool = _data_default("is_invert_spec", "bool", "is_invert_spec")
    is_match_frequency_pitch: bool = _data_default("is_match_frequency_pitch", "bool", "is_match_frequency_pitch")
    is_mdx23_combine_stems: bool = _data_default("is_mdx23_combine_stems", "bool", "is_mdx23_combine_stems")
    is_mdx_c_seg_def: bool = _data_default("is_mdx_c_seg_def", "bool", "is_mdx_c_seg_def")
    is_normalization: bool = _data_default("is_normalization", "bool", "is_normalization")
    is_post_process: bool = _data_default("is_post_process", "bool", "is_post_process")
    is_primary_stem_only_Demucs: bool = _data_default("is_primary_stem_only_Demucs", "bool", "is_primary_stem_only_Demucs")
    is_primary_stem_only: bool = _data_default("is_primary_stem_only", "bool", "is_primary_stem_only")
    is_save_inst_set_vocal_splitter: bool = _data_default("is_save_inst_set_vocal_splitter", "bool", "is_save_inst_set_vocal_splitter")
    is_secondary_stem_only_Demucs: bool = _data_default("is_secondary_stem_only_Demucs", "bool", "is_secondary_stem_only_Demucs")
    is_secondary_stem_only: bool = _data_default("is_secondary_stem_only", "bool", "is_secondary_stem_only")
    is_split_mode: bool = _data_default("is_split_mode", "bool", "is_split_mode")
    is_tta: bool = _data_default("is_tta", "bool", "is_tta")
    is_use_directml: bool = _data_default("is_use_directml", "bool", "is_use_directml")
    margin: int = _data_default("margin", "int", "margin")
    margin_demucs: int = _data_default("margin_demucs", "int", "margin_demucs")
    mdx_batch_size: str = _data_default("mdx_batch_size", "str", "mdx_batch_size")
    mdx_is_secondary_model_activate: bool = _data_default("mdx_is_secondary_model_activate", "bool", "mdx_is_secondary_model_activate")
    mdx_segment_size: int = _data_default("mdx_segment_size", "int", "mdx_segment_size")
    # No DEFAULT_DATA key: UVR.py:1673 initialises it with ALL_STEMS.
    mdxnet_stems: str = _widget_default(ALL_STEMS, "str", "mdxnet_stems")
    mp3_bit_set: str = _data_default("mp3_bit_set", "str", "mp3_bit_set")
    overlap: str = _data_default("overlap", "str", "overlap")
    overlap_mdx: str = _data_default("overlap_mdx", "str", "overlap_mdx")
    overlap_mdx23: str = _data_default("overlap_mdx23", "str", "overlap_mdx23")
    post_process_threshold: float = _data_default("post_process_threshold", "float", "post_process_threshold")
    save_format: str = _data_default("save_format", "str", "save_format")
    segment: str = _data_default("segment", "str", "segment")
    semitone_shift: str = _data_default("semitone_shift", "str", "semitone_shift")
    shifts: int = _data_default("shifts", "int", "shifts")
    vr_is_secondary_model_activate: bool = _data_default("vr_is_secondary_model_activate", "bool", "vr_is_secondary_model_activate")
    window_size: int = _data_default("window_size", "int", "window_size")
    # `wav_type_set` is a plain App attribute (UVR.py:409), not a *_var, and its
    # table value is 'PCM_16'.
    wav_type_set: str = _data_default("wav_type_set", "str", "wav_type_set")

    def __post_init__(self) -> None:
        """Re-parse every field into its declared type, in place (fix round 1).

        Without this, the annotations are decoration: Plan 2's UI builds `Settings`
        from `StringVar` text, so `Settings(crop_size='352')` held the *str* `'352'`
        in a field annotated `int`, while `load()` returned the *int* `352` for the
        same logical value -- and `constructed == loaded` was False. A frozen
        dataclass with declared types should not be constructible holding an off-type
        value at all, so the bare constructor is made safe rather than a
        `from_widget_values()` factory added beside the unsafe one (ruled).

        Three constraints, all rulings, none invented here:
          * it routes through the existing `_coerce`, so there is one coercion table;
          * `str` fields pass through **unchanged** -- `_coerce` returns the very
            object it was given for a `str`, so `batch_size='Default'`,
            `overlap='0.25'` and `semitone_shift='0'` keep their text exactly as
            R-T9-types stores them and `device_set` stays verbatim (whitespace and
            all) per R-T9-device. This coerces types; it never reformats a string, and
            since fix round 2 (M1) it never *renders* one either: a container or an
            object in a `str` field raises instead of becoming `'[]'`;
          * a value that cannot be parsed into the declared type **raises**, it does
            not silently become `0`. That includes `Settings(is_tta=-1)`: the legacy
            `-1`-means-CPU encoding is decoded in the import path
            (`_legacy_gpu_conversion`), so a bare one reaching the constructor is a
            caller error, not a second encoding to guess at. It also includes a
            non-finite float (fix round 2, M2) -- `nan` in a field would have made
            `constructed == load(save(constructed))` False all by itself.

        `object.__setattr__` because the instance is frozen; writing `self.x = ...`
        here would raise `FrozenInstanceError`.
        """
        types = _field_types()
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            coerced = _coerce(f.name, value, types[f.name])
            # Only write when the parse produced a different object, so a `str`
            # field keeps its identity and nothing is rewritten that already fits.
            if coerced is not value:
                object.__setattr__(self, f.name, coerced)

    def save(self, path: Path | None = None) -> None:
        """Write the snapshot as strict JSON, keyed by field name. Creates the directory.

        `allow_nan=False` (fix round 2, M2) is the writer's half of the guarantee that a
        file this method produces is readable by anything else: Python's default would
        turn a non-finite float into a bare `NaN`, which `json.loads` reads back and an
        ECMA-404 parser refuses. `_coerce` already keeps such a value out of every field
        that arrives through it, so this only fires when one was forced onto a frozen
        instance behind the type system's back -- and then it raises instead of writing.
        """
        target = Path(path) if path is not None else config_file()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(dataclasses.asdict(self), indent=2, ensure_ascii=False,
                       allow_nan=False) + "\n",
            encoding="utf-8",
        )


def _field_types() -> typing.Mapping[str, type]:
    """`{field name: declared type}`, resolved once, cached, and handed out sealed.

    `__post_init__` calls this for every instance, and the class is closed at import
    (50 fields, no dynamic annotation), so re-running `get_type_hints` per construction
    would only slow the suite down.

    The return is a `MappingProxyType`, not the cached dict (fix round 2, M3). Handing
    out the mutable cache meant one caller doing `settings._field_types()["crop_size"] =
    str` changed what *every later construction* coerces to -- round 1's defect,
    reintroduced by a mutation rather than by a bug. A proxy keeps the whole read
    interface (`[]`, `get`, `len`, iteration, equality with a dict) and refuses the write.
    """
    global _FIELD_TYPES
    if _FIELD_TYPES is None:
        _FIELD_TYPES = MappingProxyType(typing.get_type_hints(Settings))
    return _FIELD_TYPES


def load(path: Path | None = None) -> Settings:
    """Read `%APPDATA%`/UVR-fork/settings.json, or defaults if it does not exist.

    Unknown keys are dropped (forward compatibility, and the reason a stale
    `input_paths` in an old file cannot break a load); a known key that cannot be
    parsed into its declared type raises, because silently keeping a corrupt value
    is how a wrong-type field reaches the engine.
    """
    target = Path(path) if path is not None else config_file()
    if not target.exists():
        return Settings()
    raw = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"{target}: settings.json must hold an object, got {type(raw).__name__}")
    types = _field_types()
    known = {f.name: types[f.name] for f in dataclasses.fields(Settings)}
    values = {name: _coerce(name, raw[name], typ) for name, typ in known.items() if name in raw}
    return Settings(**values)


def _legacy_gpu_conversion(value: object) -> bool:
    """Decode `is_gpu_conversion` in either of the two encodings it arrives in.

    The widget's own encoding is boolean; the *derived* one -- `UVR.py:384` and
    `tests/golden/settings.py:254` -- is `0` for GPU-on and `-1` for CPU, so `0`
    must not become `False`. A boolean is taken at face value; any other number
    that is neither sentinel is truth-tested.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return _to_bool("gpu_conversion", value)
    if isinstance(value, (int, float)):
        if value == _GPU_ON:
            return True
        if value == _GPU_CPU:
            return False
        return bool(value)
    raise TypeError(f"Settings.gpu_conversion: {type(value).__name__} is not a stored flag")


def import_legacy_data_pickle(pkl: Path) -> Settings:
    """One-time, read-only import of the official build's `data.pkl`.

    The legacy document is keyed by the `DEFAULT_DATA` names (the App persisted
    `self.<name>_var.get()` under those keys, `UVR.py:7678`-`:7758`), so keys map
    through `field_for_data_key`, which is where the `is_gpu_conversion` ->
    `gpu_conversion` rename is undone. Differences from `load()`:
      * numbers stored as text by `tk.StringVar` are coerced to `int`/`float`;
      * flags may arrive as `0`/`1`, decoded per `_legacy_gpu_conversion`;
      * a key the legacy file does not mention keeps its `DEFAULT_DATA` default --
        never an invented placeholder;
      * a key this module does not have a field for (`vr_model`, `input_paths`,
        `export_path`, every `*_MAPPER`, ...) is dropped, not guessed.
    Nothing is written: the caller saves when it is ready to adopt the fork's
    JSON format.
    """
    source = Path(pkl)
    raw = pickle.loads(source.read_bytes())
    if not isinstance(raw, dict):
        raise TypeError(f"{source}: data.pkl must hold a settings mapping, "
                        f"got {type(raw).__name__}")
    types = _field_types()
    values: dict[str, typing.Any] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            continue
        name = field_for_data_key(key)
        if name is None:
            continue
        if name == "gpu_conversion":
            values[name] = _legacy_gpu_conversion(value)
        else:
            values[name] = _coerce(name, value, types[name])
    return Settings(**values)
