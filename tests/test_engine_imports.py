# tests/test_engine_imports.py
import importlib
import pytest

MODULES = [
    "separate",
    "lib_v5.spec_utils",
    "lib_v5.mdxnet",
    "lib_v5.vr_network.nets",
    "lib_v5.bs_roformer",
    "lib_v5.mel_band_roformer",
]

@pytest.mark.parametrize("name", MODULES)
def test_engine_module_imports(name):
    importlib.import_module(name)
