"""Ratchet: no positional ``librosa.resample()`` sr arguments may survive.

librosa >= 0.10 turned ``orig_sr``/``target_sr`` into keyword-only parameters, so a
call like ``librosa.resample(wave, sr_orig, sr_target, res_type=...)`` now raises
``TypeError: resample() takes 1 positional argument but 3 were given`` at runtime --
not at import time, which is why the import smoke tests cannot catch it. These calls
sit on the sample-rate ladder of the VR band pipeline, so a surviving positional site
would corrupt the golden audio baselines captured by the next task.

The guard is purely static (AST) so it runs without torch, models or audio.
"""

import ast
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]

TARGETS = ["lib_v5/spec_utils.py", "separate.py"]

SR_KEYWORDS = ("orig_sr", "target_sr")


def _resample_calls(rel_path):
    """Yield every ``librosa.resample(...)`` ast.Call node in *rel_path*.

    Deliberately keyed on ``librosa.resample`` so that unrelated resamplers in the
    same files -- ``resampy.resample``, ``samplerate.resample`` and
    ``scipy.signal.resample_poly`` -- are never swept into the assertion.
    """
    tree = ast.parse((REPO / rel_path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if (
                isinstance(f, ast.Attribute)
                and f.attr == "resample"
                and isinstance(f.value, ast.Name)
                and f.value.id == "librosa"
            ):
                yield node


@pytest.mark.parametrize("path", TARGETS)
def test_no_positional_resample(path):
    for node in _resample_calls(path):
        assert not node.args[1:], (
            f"{path}:{node.lineno} passes positional sr args to librosa.resample; "
            "librosa>=0.10 requires orig_sr= / target_sr="
        )


@pytest.mark.parametrize("path", TARGETS)
def test_resample_binds_both_sample_rates_by_keyword(path):
    """The fix must name both rates, not merely drop the extra positional args."""
    for node in _resample_calls(path):
        bound = {kw.arg for kw in node.keywords if kw.arg}
        missing = [name for name in SR_KEYWORDS if name not in bound]
        assert not missing, (
            f"{path}:{node.lineno} does not bind {missing} as keyword arguments"
        )


def test_guard_actually_sees_every_resample_site():
    """Fail loudly if the AST predicate matches nothing -- an empty loop passes."""
    found = {path: len(list(_resample_calls(path))) for path in TARGETS}

    assert found == {"lib_v5/spec_utils.py": 3, "separate.py": 4}, (
        "the guard's librosa.resample detector lost touch with the code; "
        f"counted {found} (expected 3 in spec_utils, 4 in separate)"
    )


def test_guard_ignores_foreign_resamplers():
    """Other resamplers in these files must not be policed by this guard.

    separate.py calls ``signal.resample_poly`` (scipy) and that call legitimately
    takes positional arguments; a guard that swept it up would demand a change that
    breaks audio output. Count textual resample calls and prove the AST collector
    sees strictly fewer.
    """
    scanned = "\n".join((REPO / path).read_text(encoding="utf-8") for path in TARGETS)
    naive = scanned.count("resample(") + scanned.count("resample_poly(")
    collected = sum(len(list(_resample_calls(path))) for path in TARGETS)

    assert "signal.resample_poly(" in scanned
    assert all(
        node.func.value.id == "librosa"
        for path in TARGETS
        for node in _resample_calls(path)
    )
    assert collected < naive, (
        f"guard collected {collected} of {naive} textual resample calls; expected to "
        "skip the non-librosa ones"
    )
