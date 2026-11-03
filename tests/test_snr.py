"""Discrimination tests for the golden SNR gate.

The gate is only meaningful if it provably separates three cases:
identical signals (inf), a ~45 dB perturbation (pass) and an audible error
(fail). Without the negative control a bug in ``snr_db`` would report every
comparison as green.
"""
import numpy as np

from tests.golden.snr import snr_db, max_abs_dev


def test_identical_signals_are_infinite_snr():
    x = np.sin(np.linspace(0, 1000, 44100)).astype("float32")
    assert snr_db(x, x) == float("inf")


def test_small_noise_clears_the_40db_gate():
    rng = np.random.default_rng(0)
    x = np.sin(np.linspace(0, 1000, 44100)).astype("float32")
    y = x + rng.standard_normal(x.size).astype("float32") * 10 ** (-45 / 20)
    assert snr_db(x, y) > 40


def test_audible_difference_fails_the_gate():
    rng = np.random.default_rng(0)
    x = np.sin(np.linspace(0, 1000, 44100)).astype("float32")
    y = x + rng.standard_normal(x.size).astype("float32") * 0.05
    assert snr_db(x, y) < 40
    assert max_abs_dev(x, y) > 0.1
