"""Derive a sibling output name by moving only the trailing ``.wav``.

The engine derives every sibling name from a stem path: the overwrite-protect
stamp (``separate.py:539``), the deverb pair (``:576-577``), and the FLAC/MP3
conversions (``:1557``/``:1561``). Each used to be ``path.replace(".wav", ...)``,
and ``str.replace`` rewrites *every* occurrence -- so a run whose output directory
(or stem) contained ``.wav`` wrote to a mangled path. This helper moves the suffix
and nothing else: ``"/a.wav/song.wav"`` with ``".mp3"`` is ``"/a.wav/song.mp3"``,
not ``"/a.mp3/song.mp3"``.

Deliberately case-sensitive, exactly like the ``replace`` it replaced: a ``.WAV``
stem is left alone. Pure and stdlib-only, so it is importable and unit-tested
without torch.
"""
from __future__ import annotations

_WAV = ".wav"


def replace_wav_suffix(path: str, new_suffix: str) -> str:
    """Move a trailing ``.wav`` to *new_suffix*; leave everything else untouched.

    *new_suffix* is appended after the ``.wav`` is stripped, so the callers' existing
    arguments read the same: ``"_deverbed.wav"``, ``".flac"``, ``f"_{stamp}.wav"``.
    """
    return path[: -len(_WAV)] + new_suffix if path.endswith(_WAV) else path
