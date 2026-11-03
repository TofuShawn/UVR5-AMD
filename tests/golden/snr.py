"""Signal-to-noise ratio helper for the golden audio gate.

The gate is *cross-device equivalence* (CPU torch vs ROCm torch, same venv, same
frozen settings), not fidelity to official UVR output. Read
``tests.golden.runner``'s module docstring before changing any threshold here:
the modernised dependency stack (librosa >= 0.10, numpy 2.x) already resamples
differently from the reference build, so these numbers fingerprint *this* stack.
Loosening or reinterpreting the 40 dB gate to make a historical recording match
would destroy the only correctness proof the fork has.
"""
from __future__ import annotations

import numpy as np


def _pair(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    n = min(a.size, b.size)
    assert n > 0, "empty signal"
    return a[:n], b[:n]


def snr_db(reference, candidate) -> float:
    """Return the SNR in dB of ``candidate`` measured against ``reference``.

    Identical signals give ``inf``; a reference that is pure silence gives
    ``-inf``.
    """
    ref, cand = _pair(reference, candidate)
    err = ref - cand
    p_sig = float(np.mean(ref ** 2))
    p_err = float(np.mean(err ** 2))
    if p_err == 0.0:
        return float("inf")
    if p_sig == 0.0:
        return float("-inf")
    return 10.0 * float(np.log10(p_sig / p_err))


def max_abs_dev(reference, candidate) -> float:
    """Largest sample-wise absolute deviation; reported alongside the gate."""
    ref, cand = _pair(reference, candidate)
    return float(np.max(np.abs(ref - cand)))
