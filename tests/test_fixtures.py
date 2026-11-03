from uvr_paths import MODEL_ROOT, MODELS, CONFIGS, INPUT_FILE

REQUIRED = ["vr", "demucs", "onnx"]


def test_input_file_exists():
    assert INPUT_FILE.is_file(), (
        f"golden input missing at {INPUT_FILE}. Set UVR_INPUT_FILE. "
        "Tasks 4, 7 and 13 all key off this one file."
    )


def test_model_root_exists():
    assert MODEL_ROOT.is_dir(), (
        f"no model root at {MODEL_ROOT}. Run with "
        'UVR_MODEL_ROOT="<path to models>" .venv/Scripts/python.exe -m pytest'
    )


def test_every_required_arch_has_a_weight():
    missing = [k for k in REQUIRED if k not in MODELS]
    assert not missing, (
        f"missing weights for {missing} under {MODEL_ROOT}. "
        "Set UVR_MODEL_ROOT or fetch them before continuing."
    )


def test_model_entries_are_files_not_directories():
    # the mdx_c_configs directory is a CONFIG, and a loose .exists() predicate
    # would have let it masquerade as an available weight
    assert all(p.is_file() for p in MODELS.values()), MODELS
    assert not any(p.is_dir() for p in MODELS.values()), MODELS
