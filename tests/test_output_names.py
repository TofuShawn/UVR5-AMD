"""The ``.wav`` suffix helper, pinned in both directions.

``separate.py`` derives every sibling output name (overwrite-protect stamp,
``_deverbed``/``_reverb_only``, ``.flac``/``.mp3``) from a stem path. The old code
used ``str.replace(".wav", ...)``, which rewrites *every* ``.wav`` in the string --
so an output folder or stem containing ``.wav`` was mangled. This file pins the
helper that replaced it:

* the behavioural half proves the pathological path is fixed *and* the ordinary
  path is byte-identical to the old behaviour (a fix that also moved normal names
  would be a regression, not a fix);
* the static half proves no raw ``.replace(".wav", ...)`` survives in the engine
  and that the helper is actually wired at the five call sites -- so the
  prohibition cannot be satisfied by merely deleting the sites.
"""
from __future__ import annotations

import ast
import pathlib

from output_names import replace_wav_suffix

REPO = pathlib.Path(__file__).resolve().parents[1]


# --- behavioural half ------------------------------------------------------- #

def test_normal_path_moves_only_the_trailing_suffix():
    assert replace_wav_suffix("/out/song.wav", ".flac") == "/out/song.flac"
    assert replace_wav_suffix("/out/song.wav", "_deverbed.wav") == "/out/song_deverbed.wav"
    assert replace_wav_suffix("/out/song.wav", "_x.wav") == "/out/song_x.wav"


def test_a_wav_in_a_directory_name_is_not_touched():
    """The bug: ``str.replace`` rewrote the directory part too."""
    assert replace_wav_suffix("/archive.wav/song.wav", ".mp3") == "/archive.wav/song.mp3"
    assert replace_wav_suffix("/a.wav.b/song.wav", "_deverbed.wav") == "/a.wav.b/song_deverbed.wav"


def test_a_wav_inside_the_stem_is_not_touched():
    assert replace_wav_suffix("/out/song.wav.take2.wav", ".flac") == "/out/song.wav.take2.flac"


def test_non_wav_path_is_returned_unchanged():
    assert replace_wav_suffix("/out/song.flac", ".mp3") == "/out/song.flac"
    assert replace_wav_suffix("/out/song", "_deverbed.wav") == "/out/song"


def test_case_sensitivity_matches_the_old_behaviour():
    # the ``replace`` it replaced was case-sensitive; keep that
    assert replace_wav_suffix("/out/song.WAV", ".flac") == "/out/song.WAV"


# --- static half ------------------------------------------------------------ #

def _separate_tree():
    return ast.parse((REPO / "separate.py").read_text(encoding="utf-8"))


def test_separate_py_no_longer_calls_str_replace_on_dot_wav():
    offenders, helper_loads = [], 0
    for node in ast.walk(_separate_tree()):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "replace" and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == ".wav"):
            offenders.append(f"line {node.lineno}")
        if isinstance(node, ast.Name) and node.id == "replace_wav_suffix":
            helper_loads += 1
    assert not offenders, (
        "separate.py still does `something.replace(\".wav\", ...)`, which rewrites a "
        "'.wav' inside a directory or stem too:\n  " + "\n  ".join(offenders))
    assert helper_loads == 5, (
        f"expected the suffix helper at its five derivation sites (overwrite-protect, "
        f"_deverbed, _reverb_only, .flac, .mp3); found {helper_loads} load(s). A raw "
        "replace may have been deleted instead of routed through the helper.")
