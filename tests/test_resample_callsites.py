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


def _calls_in_source(source):
    """Yield ``(name, node)`` for every guarded ``librosa.<name>(...)`` call in *source*.

    The predicate requires ``func`` to be an attribute of a bare ``ast.Name`` called
    ``librosa`` with a guarded ``attr``, so ``librosa.istft``, ``resampy.resample``,
    ``samplerate.resample``, ``scipy.signal.resample_poly`` and a locally re-bound
    ``stft`` (``lib_v5.spec_utils.stft`` is one) are never collected.
    """
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            f = node.func
            if (
                isinstance(f, ast.Attribute)
                and isinstance(f.value, ast.Name)
                and f.value.id == "librosa"
                and f.attr in _MAX_POSITIONAL
            ):
                yield f.attr, node


def _librosa_calls(rel_path):
    """Guarded calls in *rel_path*, resolved against the repo root, not the cwd."""
    return _calls_in_source((REPO / rel_path).read_text(encoding="utf-8"))


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
