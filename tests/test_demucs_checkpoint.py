"""Task 6 Step 6: the demucs checkpoint is loadable from the ENGINE, not the harness.

Before this task the only ``torch.serialization.safe_globals`` allowlist that let
a demucs ``.th`` unpickle under torch >= 2.6 lived inside
``tests/golden/runner.py:_inference_context``. The consequence was exactly the
fork's worst failure mode: the golden suite was green while the real product
(Task 13's GUI-free CLI, which calls ``SeperateDemucs`` directly and never touches
``runner.py``) crashed loading the same checkpoint.

These tests pin the fix from both directions:

* ``test_unguarded_htdemucs_load_is_rejected`` -- a bare
  ``demucs.pretrained.get_model`` on the fixture *fails*, i.e. the weights-only
  crash is real and still present in the vendored loader.
* ``test_htdemucs_loads_through_the_engine_guard`` -- the same load, wrapped in
  the engine's ``separate.demucs_checkpoint_load``, *succeeds*. This is the
  shipped-CLI path; it deliberately does NOT call ``runner._inference_context``,
  so it cannot be kept green by the harness quietly carrying the allowlist again.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

import separate
import uvr_paths
from demucs.pretrained import get_model as _gm

#: The v3/v4 repo model the engine loads through ``states.py:46``.
#: ``uvr_paths.MODELS['demucs']`` is the ``.th`` weight; the engine's
#: ``model_path`` is the sibling ``htdemucs.yaml`` bag descriptor in the same
#: directory (see ``separate.py`` else-branch and the harness's own derivation in
#: ``tests/golden/settings._demucs_model_path``).
_demucs_dir = uvr_paths.MODELS["demucs"].parent
_demucs_name = (_demucs_dir / "htdemucs.yaml").stem


def test_unguarded_htdemucs_load_is_rejected():
    """The vendored loader alone cannot read the checkpoint under torch >= 2.6.

    This is the crash the harness used to paper over. If it ever *stops* raising,
    the port is no longer load-bearing and this test should be reconsidered rather
    than deleted silently.
    """
    with pytest.raises(Exception) as excinfo:
        _gm(name=_demucs_name, repo=_demucs_dir)

    # Name the failure mode so a *different* exception (a missing fixture, say)
    # cannot masquerade as the weights-only rejection.
    text = str(excinfo.value)
    assert "Weights only load failed" in text or "safe_globals" in text or "Unpickling" in type(
        excinfo.value).__name__ + text, (
        f"expected a weights_only unpickling rejection, got "
        f"{type(excinfo.value).__name__}: {text[:300]}"
    )


def test_htdemucs_loads_through_the_engine_guard():
    """The engine's own scoped allowlist is enough -- no harness context."""
    with separate.demucs_checkpoint_load():
        model = _gm(name=_demucs_name, repo=_demucs_dir)

    assert model is not None
    # htdemucs is a 4-stem bag; loading it for real is the whole point.
    sources = list(getattr(model, "sources", []))
    assert len(sources) == 4, f"expected 4 demucs sources, got {sources}"


def test_engine_guard_is_scoped_not_process_wide():
    """The allowlist must not leak past the ``with`` block.

    If ``demucs_checkpoint_load`` were a process-wide
    ``torch.serialization.add_safe_globals``, a VR/MDX load that forgot the
    context manager would silently start trusting globals too -- the exact class
    of silent-widening this guard checks for.
    """
    with separate.demucs_checkpoint_load():
        pass
    with pytest.raises(Exception):
        _gm(name=_demucs_name, repo=_demucs_dir)
