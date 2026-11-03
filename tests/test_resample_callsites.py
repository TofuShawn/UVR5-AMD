"""Ratchet: no positional ``librosa`` arguments may survive on ``resample``, ``load`` or ``stft``.

librosa >= 0.10 turned the whole tail of every entry point's signature into
keyword-only parameters, so calls that were legal under the py3.9/librosa-0.7 build now
raise ``TypeError`` at *runtime* -- not at import time, which is why the import smoke
tests cannot catch them. Verified against the installed librosa 1.0.0::

    load(path, *, sr, mono, offset, duration, dtype, res_type)
    stft(y, *, n_fft, hop_length, win_length, window, center, dtype, pad_mode, out)
    resample(y, *, orig_sr, target_sr, res_type, fix, scale, axis, **kwargs)

Only the first parameter (the buffer, or the path for ``load``) may be passed
positionally. These calls sit on the VR pipeline's sample-rate ladder and its
spectrogram window, so a surviving positional site would either crash the run or, if
"fixed" by deleting the extra argument, silently corrupt the golden audio baselines the
next task captures -- ``n_fft`` is the only argument that sets the window size, and
``sr``/``orig_sr`` are the only thing pinning the band rates.

The guard is purely static (AST) so it runs anywhere, without torch, models or audio
fixtures. ``librosa.istft`` is deliberately **not** policed (nothing after ``y`` was ever
positional there), and the predicate is keyed on a ``librosa`` name so foreign resamplers
never fall under it.

A second, narrower rule covers ``res_type``. Deleting it is *not* a ``TypeError`` -- every
librosa entry point that takes it has a default, and in 1.0 that default is ``soxr_hq`` --
so a "cleanup" that drops the keyword keeps the suite green while swapping the resampler
under the audio. That is the corruption the plan's Task 3 constraint forbids ("Do **not**
change any ``res_type`` value while converting: UVR's ``sinc_fastest`` / ``sinc_best`` /
``sinc_medium`` / ``polyphase`` / ``kaiser_fast`` strings are what make output audio match
the reference build"). The rule is deliberately *per site* rather than *per entry point*:
``res_type`` is required exactly where it already appears and forbidden everywhere else:
six of the 25 guarded calls bind it and the other 19 never have, so both halves are
policed -- a pinned site losing its binding, and a site acquiring one it never had
(``test_res_type_is_pinned_at_the_sites_that_bind_it``).
"""

import ast
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]

TARGETS = ["lib_v5/spec_utils.py", "separate.py"]

# librosa 1.0 keyword-only entry points policed by this guard: the value is the maximum
# number of positional arguments a call may pass (the buffer/path itself).
_MAX_POSITIONAL = {"resample": 1, "load": 1, "stft": 1}

# Parameters that used to arrive positionally at the converted sites. A "fix" that just
# deletes them keeps librosa quiet while destroying the sample rate, the channel count or
# the FFT window, so the guard also requires them to be bound by keyword.
REQUIRED_KEYWORDS = {
    "resample": ("orig_sr", "target_sr"),
    "load": ("sr", "mono"),
    "stft": ("n_fft", "hop_length"),
}

# The only guarded calls that bind ``res_type``, keyed by
# ``(file, dotted enclosing def/class, librosa entry point, ordinal inside that scope)``
# -- an identity that survives line-number drift (this branch has already moved
# ``separate.py:1213``->``1191`` and ``1544``->``1522`` once) -- and valued by the exact
# expression each site must keep passing. Every one of them is a *name*, never a string
# literal, because the resampler is chosen by the module constant or by the model JSON:
#   * ``wav_resolution``                   -- lib_v5/spec_utils.py:41 (``sinc_fastest`` off
#     macOS) or separate.py:1177/:1179 (``bp['res_type']`` from modelparams/*.json),
#   * ``wav_resolution_float_resampling``  -- lib_v5/spec_utils.py:42,
#   * ``wav_resolution`` at separate.py:1516 -- upstream's own ``'polyphase'`` constant,
#     where ``#bp['res_type']`` is deliberately commented out.
# Pinning the expression text is what also rejects a hardcoded replacement: the moment a
# site reads ``res_type='soxr_hq'`` the unparsed value no longer matches this table.
RES_TYPE_SITES = {
    ("lib_v5/spec_utils.py", "cmb_spectrogram_to_wave", "resample", 0): "wav_resolution",
    ("lib_v5/spec_utils.py", "cmb_spectrogram_to_wave", "resample", 1): "wav_resolution",
    ("lib_v5/spec_utils.py", "change_pitch_semitones", "resample", 0): "wav_resolution_float_resampling",
    ("separate.py", "SeperateVR.loading_mix", "load", 0): "wav_resolution",
    ("separate.py", "SeperateVR.loading_mix", "resample", 0): "wav_resolution",
    ("separate.py", "loading_mix", "resample", 0): "wav_resolution",
}

# The seven load/stft sites converted alongside the five resample ones, in their pre-fix
# form. Kept verbatim so the guard can be shown to actually reject them (see
# test_guard_rejects_the_legacy_call_forms) -- the repo itself is green, so without this
# the wider coverage would never have been observed failing.
LEGACY_POSITIONAL_SOURCE = '''
import librosa
import numpy as np

def legacy(audio_file, bp, wav_resolution, wave, wave_left, wave_right, n_fft, hop_length, nfft, hl):
    X_wave, _ = librosa.load(audio_file, bp['sr'], False, dtype=np.float32, res_type=wav_resolution)
    spec_left = librosa.stft(wave_left, n_fft, hop_length=hop_length)
    spec_right = librosa.stft(wave_right, n_fft, hop_length=hop_length)
    spec_left = librosa.stft(wave_left, n_fft, hop_length=hop_length)
    spec_right = librosa.stft(wave_right, n_fft, hop_length=hop_length)
    spec_left = librosa.stft(wave_left, nfft, hop_length=hl)
    spec_right = librosa.stft(wave_right, nfft, hop_length=hl)
    X_wave = librosa.resample(X_wave, bp['sr'], 44100, res_type=wav_resolution)
    return X_wave
'''


def _guarded_call_records(source):
    """Yield ``(scope, name, node)`` for every guarded ``librosa.<name>(...)`` call.

    *scope* is the dotted path of enclosing ``def``/``class`` bodies (``<module>`` at top
    level), which is what lets ``res_type`` be pinned to a *site* without citing a line
    number. The predicate requires ``func`` to be an attribute of a bare ``ast.Name``
    called ``librosa`` with a guarded ``attr``, so ``librosa.istft``, ``resampy.resample``,
    ``samplerate.resample``, ``scipy.signal.resample_poly`` and a locally re-bound ``stft``
    (``lib_v5.spec_utils.stft`` is one) are never collected.
    """
    tree = ast.parse(source)
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    def scope_of(node):
        parts = []
        current = node
        while current is not None:
            if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                parts.append(current.name)
            current = parents.get(current)
        return ".".join(reversed(parts)) or "<module>"

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if (
                isinstance(f, ast.Attribute)
                and isinstance(f.value, ast.Name)
                and f.value.id == "librosa"
                and f.attr in _MAX_POSITIONAL
            ):
                yield scope_of(node), f.attr, node


def _calls_in_source(source):
    """Yield ``(name, node)`` for every guarded ``librosa.<name>(...)`` call in *source*.

    See ``_guarded_call_records`` for the predicate; the foreign-resampler and ``istft``
    exclusions are what keeps ``resampy``/``samplerate``/``scipy.signal`` calls -- some of
    which legitimately take positional args -- out of this guard's reach.
    """
    for _, name, node in _guarded_call_records(source):
        yield name, node


def _librosa_calls(rel_path):
    """Guarded calls in *rel_path*, resolved against the repo root, not the cwd."""
    return _calls_in_source((REPO / rel_path).read_text(encoding="utf-8"))


def _scoped_librosa_calls(rel_path):
    """Same collector as ``_librosa_calls`` but keeping each call's enclosing scope."""
    return list(_guarded_call_records((REPO / rel_path).read_text(encoding="utf-8")))


def _res_type_census(path, records):
    """Map every guarded call in *path* to its ``res_type`` binding, keyed by site.

    Keys are ``(path, scope, entry point, ordinal within that scope)``; the ordinal is
    assigned in source order (``ast.walk`` is breadth-first, so it is re-sorted here) and
    counts *all* guarded calls of that entry point in that scope, res_type-bearing or not.
    Values are ``(unparsed_expression_or_None, lineno)``; ``None`` means the call binds no
    ``res_type`` at all, which is correct for most sites and must stay that way.
    """
    counters = {}
    census = {}
    for scope, name, node in sorted(records, key=lambda r: (r[2].lineno, r[2].col_offset)):
        ordinal = counters.get((scope, name), 0)
        counters[(scope, name)] = ordinal + 1
        bound = [kw for kw in node.keywords if kw.arg == "res_type"]
        census[(path, scope, name, ordinal)] = (
            ast.unparse(bound[0].value) if bound else None,
            node.lineno,
        )
    return census


def _res_type_offenders(path, records):
    """Problems from moving a ``res_type`` binding: dropped, rewritten, or invented.

    Three separate lists so a failure says which way the call drifted:
      * ``lost``       -- a site that binds ``res_type`` today stopped doing so; librosa
        then silently uses its ``soxr_hq`` default and the audio fingerprint changes.
      * ``rewritten``  -- the keyword is still there but no longer the pinned expression.
      * ``invented``   -- a site that never had one now does: same silent substitution,
        just pushed onto calls the engine deliberately left on librosa's default.
    """
    census = _res_type_census(path, records)
    lost, rewritten, invented = [], [], []

    for key, expected in sorted(RES_TYPE_SITES.items()):
        if key[0] != path:
            continue
        actual, lineno = census.get(key, (None, None))
        if key not in census:
            lost.append(
                f"{path}: {key[1]}.{key[2]} #{key[3]} vanished -- the site that bound "
                f"res_type={expected} is gone (renamed scope, new entry point, or an "
                "extra call renumbered it)"
            )
        elif actual is None:
            lost.append(
                f"{path}:{lineno}: librosa.{key[2]} in {key[1]} dropped "
                f"res_type={expected}; librosa 1.0 now falls back to its soxr_hq "
                "default and the resampled audio changes"
            )
        elif actual != expected:
            rewritten.append(
                f"{path}:{lineno}: librosa.{key[2]} in {key[1]} re-bound res_type: "
                f"{expected} -> {actual}"
            )

    for key, (actual, lineno) in sorted(census.items(), key=lambda kv: (kv[0][2], kv[1][1])):
        if key[0] != path or key in RES_TYPE_SITES:
            continue
        if actual is not None:
            invented.append(
                f"{path}:{lineno}: librosa.{key[2]} in {key[1]} now passes "
                f"res_type={actual}, but this site never bound one -- forcing a "
                "resampler onto it is as much a substitution as dropping it"
            )

    return lost, rewritten, invented


def _offenders(calls, location=None):
    """Split *calls* into (positional offenders, keyword-binding offenders).

    *location* is an optional path used to prefix each problem with ``path:lineno`` so a
    failure points straight at the line that introduced it.
    """
    too_positional = []
    unbound_keywords = []
    for name, node in calls:
        where = f"{location}:{node.lineno}: " if location else ""
        if len(node.args) > _MAX_POSITIONAL[name]:
            too_positional.append(f"{where}librosa.{name}({len(node.args)} positional)")
            continue
        bound = {kw.arg for kw in node.keywords if kw.arg}
        missing = [k for k in REQUIRED_KEYWORDS[name] if k not in bound]
        if missing:
            unbound_keywords.append(f"{where}librosa.{name} does not bind {missing}")
    return too_positional, unbound_keywords


@pytest.mark.parametrize("path", TARGETS)
def test_no_positional_librosa_args(path):
    """Every guarded call passes its buffer/path and nothing else, positionally."""
    too_positional, _ = _offenders(_librosa_calls(path), location=path)
    assert not too_positional, (
        "librosa>=0.10 requires keyword-only sr/mono/n_fft/hop_length/orig_sr/target_sr:\n"
        + "\n".join(too_positional)
    )


@pytest.mark.parametrize("path", TARGETS)
def test_keyword_only_parameters_are_named(path):
    """The conversion must *name* the args it un-positionalised, not drop them."""
    _, unbound_keywords = _offenders(_librosa_calls(path), location=path)
    assert not unbound_keywords, (
        "positional args cannot be deleted to satisfy librosa 1.0 -- they carry the "
        "band sample rate, the stereo flag and the FFT window size:\n"
        + "\n".join(unbound_keywords)
    )


def test_res_type_is_pinned_at_the_sites_that_bind_it():
    """``res_type`` must survive at exactly the six sites that already pass it -- nowhere else.

    ``_MAX_POSITIONAL``/``REQUIRED_KEYWORDS`` cannot express this: ``res_type`` has a
    librosa default (``soxr_hq``), so deleting it from ``separate.py:1182`` or from any of
    the five ``resample`` sites keeps both ratchets green while libsamplerate/resampy is
    quietly replaced by soxr. The six distinct strings UVR can feed these sites
    (``sinc_fastest``, ``sinc_best``, ``sinc_medium``, ``polyphase``, ``kaiser_fast``,
    ``kaiser_best`` -- from ``lib_v5/spec_utils.py:41-42``, ``separate.py:1516`` and
    ``modelparams/*.json``'s ``bp['res_type']``) each produce a different output
    fingerprint, which is exactly what Task 4's golden baselines and Task 7's CPU-vs-ROCm
    SNR gate compare.

    The rule is "required exactly where it already appears", so its two directions are both
    policed: a pinned site losing ``res_type`` (or having it re-bound to another
    expression), and a site that never had one acquiring it.
    """
    problems = []
    census = {}
    for path in TARGETS:
        records = _scoped_librosa_calls(path)
        census.update(_res_type_census(path, records))
        lost, rewritten, invented = _res_type_offenders(path, records)
        problems.extend(lost + rewritten + invented)

    assert not problems, (
        "res_type is what pins the resampler; a site that stops binding it falls back to"
        " librosa's soxr_hq default and silently changes the audio (and a site that gains a"
        " res_type it never had is the same substitution in reverse).\n"
        + "\n".join(problems)
        + "\n\npinned sites: "
        + "\n".join(f"  {key} -> {expr}" for key, expr in sorted(RES_TYPE_SITES.items()))
        + "\nactual census: "
        + "\n".join(
            f"  {key} -> {expr or '<no res_type>'} (line {lineno})"
            for key, (expr, lineno) in sorted(
                census.items(), key=lambda kv: (kv[0][0], kv[1][1])
            )
        )
    )


def test_guard_actually_sees_every_site():
    """Fail loudly if the AST predicate loses touch with the code -- an empty loop passes.

    Pinned per file *and* per function name, so a detector that silently stopped matching
    ``load`` or ``stft`` is caught, not just one that stopped matching ``resample``.
    """
    found = {path: {} for path in TARGETS}
    for path in TARGETS:
        for name, _ in _librosa_calls(path):
            found[path][name] = found[path].get(name, 0) + 1

    expected = {
        "lib_v5/spec_utils.py": {"resample": 3, "load": 8, "stft": 7},
        "separate.py": {"resample": 4, "load": 3},
    }
    assert found == expected, (
        "the guard's librosa call detector lost touch with the code; "
        f"counted {found}, expected {expected}"
    )


def test_guard_rejects_the_legacy_call_forms():
    """The widened guard really is red on the pre-fix calls it was written for."""
    too_positional, _ = _offenders(_calls_in_source(LEGACY_POSITIONAL_SOURCE))
    assert len(too_positional) == 8, (
        "expected the 7 legacy load/stft sites plus 1 legacy resample site to be "
        f"rejected; got {too_positional}"
    )
    assert any("librosa.load(3 positional)" in m for m in too_positional)
    assert any("librosa.stft(2 positional)" in m for m in too_positional)
    assert any("librosa.resample(3 positional)" in m for m in too_positional)


def test_guard_ignores_istft_and_foreign_resamplers():
    """Only ``resample``/``load``/``stft`` are policed, and only when reached via librosa.

    ``librosa.istft`` is out of scope by design, and the separate ``stft``/``istft``
    helpers in spec_utils plus ``resampy``/``samplerate``/``scipy.signal`` resamplers must
    never be swept up -- some of them legitimately take positional arguments, and forcing
    them to keyword form would change audio output.
    """
    foreign = """
import librosa
import resampy
import samplerate
from scipy import signal

def other(y, spec, sr_a, sr_b):
    librosa.istft(spec, hop_length=sr_a)
    resampy.resample(y, sr_a, sr_b)
    samplerate.resample(y, 2.0, sr_a)
    signal.resample_poly(y, 2, 3)
    stft(y, 2048, 1024)
    return None
"""
    assert list(_calls_in_source(foreign)) == [], (
        "the guard collected a call it must not police"
    )

    scanned = "\n".join((REPO / path).read_text(encoding="utf-8") for path in TARGETS)
    collected = sum(len(list(_librosa_calls(path))) for path in TARGETS)
    textual = sum(
        scanned.count(f"librosa.{name}(")
        for name in ("resample", "load", "stft", "istft")
    )

    assert "librosa.istft(" in scanned, "istft vanished; revisit this test's premise"
    assert "signal.resample_poly(" in scanned
    assert collected < textual, (
        f"guard collected {collected} of {textual} textual librosa call sites; the "
        "unpoliced istft calls should make these differ -- they do not, so the "
        "predicate is matching more than it should"
    )
