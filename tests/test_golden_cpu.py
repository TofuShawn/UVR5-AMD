"""Task 4 golden harness tests: capture CPU baselines, prove the harness is real.

A capture that merely "ran" proves almost nothing, so four kinds of assertion
live here:

1. *Completeness* -- the frozen settings supply every ``model_data.<attr>`` that
   ``separate.py`` actually reads for the arch being run. Derived from the
   source with AST, not from a hand-maintained list, so Task 11 cannot add a
   read without this test failing first.
2. *Coverage honesty* -- the arch list under gate is exactly the set of archs
   whose weights exist on this machine (VR + Demucs). A missing arch must be a
   reported gap, never a silently-green row.
3. *Reproducibility* -- two consecutive CPU captures of the same arch are
   byte-identical, and the seeding that makes them so is asserted directly.
   Without the pinned ``random`` seed the demucs shift trick makes even
   CPU-vs-CPU comparisons differ, which would render the 40 dB cross-device
   gate meaningless.
4. *Discrimination* -- the gate must reject audio that really is different:
   the VR and Demucs vocal stems of the same crop score far below 40 dB
   against each other, while a stem always scores ``inf`` against itself.
"""
from __future__ import annotations

import ast
import hashlib
import json
import pathlib
import shutil

import pytest
import soundfile as sf

import uvr_paths
from tests.golden import runner, settings
from tests.golden.snr import max_abs_dev, snr_db

APPLY = settings.APPLY
REPO = pathlib.Path(uvr_paths.REPO)

#: ``separate.py`` dispatches on these constants; they name the arch blocks
#: whose ``model_data.*`` reads are conditional.
ARCH_CONSTANT_TO_KEY = {"VR_ARCH_TYPE": "vr", "DEMUCS_ARCH_TYPE": "demucs",
                        "MDX_ARCH_TYPE": "mdx"}


# --------------------------------------------------------------------------- #
# AST derivation of the ModelData contract
# --------------------------------------------------------------------------- #

def _model_data_attrs(node) -> set[str]:
    return {
        child.attr
        for child in ast.walk(node)
        if isinstance(child, ast.Attribute)
        and isinstance(child.value, ast.Name)
        and child.value.id == "model_data"
    }


def _seperate_attributes_contract() -> tuple[set[str], dict[str, set[str]]]:
    """(attrs read on every path, {arch_key: attrs read only on that arch}).

    Split by *statement*, not by name: ``model_data.primary_stem`` is read both
    unconditionally (``separate.py:120``) and again inside the MDX/Demucs
    blocks, and a name-set subtraction would wrongly drop it from the base.
    """
    tree = ast.parse((REPO / "separate.py").read_text(encoding="utf-8"))
    classes = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}

    init = next(
        f for f in classes["SeperateAttributes"].body
        if isinstance(f, ast.FunctionDef) and f.name == "__init__"
    )

    def arch_key(stmt) -> str | None:
        test = getattr(stmt, "test", None)
        if (isinstance(stmt, ast.If) and isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Attribute)
                and isinstance(test.left.value, ast.Name)
                and test.left.value.id == "model_data"
                and test.left.attr == "process_method"
                and isinstance(test.comparators[0], ast.Name)):
            return ARCH_CONSTANT_TO_KEY.get(test.comparators[0].id)
        return None

    always: set[str] = set()
    blocks: dict[str, set[str]] = {}
    for stmt in init.body:
        key = arch_key(stmt)
        if key is None:
            always |= _model_data_attrs(stmt)
        else:
            blocks.setdefault(key, set()).update(_model_data_attrs(stmt))

    for name in ("SeperateVR", "SeperateDemucs"):
        always |= _model_data_attrs(classes[name])
    return always, blocks


def _required_attrs(arch: str) -> set[str]:
    always, blocks = _seperate_attributes_contract()
    return always | blocks.get(arch, set())


def test_seperate_attributes_contract_is_derivable():
    """Tripwire: the AST walk must see the reads the source actually contains.

    Guards the guard -- if ``SeperateAttributes.__init__`` is restructured so
    this walk silently finds nothing, the coverage test below would pass on an
    empty requirement set. Note that some names (``process_method``,
    ``primary_stem``, ``secondary_stem``) appear in both the base set and an
    arch block: they are read unconditionally at ``separate.py:105/120/122``
    *and* re-read inside the arch blocks, which is why the split is done per
    statement rather than by subtracting block names from the base.
    """
    always, blocks = _seperate_attributes_contract()
    for arch in APPLY:
        required = _required_attrs(arch)
        assert {"process_method", "model_path", "primary_stem", "secondary_stem"} <= required
    assert {"process_method", "primary_stem", "secondary_stem"} <= always
    assert {"DENOISER_MODEL", "overlap", "wav_type_set", "device_set"} <= always
    assert {"vr_model_param", "window_size", "aggression_setting"} <= blocks["vr"]
    assert {"demucs_source_map", "segment", "shifts"} <= blocks["demucs"]
    assert "vr_model_param" not in always, "arch-conditional read leaked into the base set"
    assert "vr_model_param" not in blocks["demucs"]
    assert "demucs_source_map" not in blocks["vr"]


@pytest.mark.parametrize("arch", APPLY)
def test_frozen_settings_supply_every_attr_the_engine_reads(arch):
    required = _required_attrs(arch)
    provided = set(settings.make(arch, "cpu"))
    missing = sorted(required - provided)
    assert not missing, (
        f"separate.py reads model_data.{missing} on the {arch} path but "
        "tests/golden/settings.py does not supply it"
    )


def test_coverage_guard_detects_a_missing_attr():
    """Negative control: dropping a required attr must be reported, not waved."""
    required = _required_attrs("vr")
    assert len(required) > 40, "requirement set implausibly small"
    provided = set(settings.make("vr", "cpu")) - {"vr_model_param"}
    assert "vr_model_param" in required - provided


# --------------------------------------------------------------------------- #
# Coverage honesty
# --------------------------------------------------------------------------- #

def test_apply_list_covers_only_archs_with_real_weights():
    assert APPLY, "the golden gate would cover nothing"
    for arch in APPLY:
        assert arch in runner.ARCH_CLASS, f"{arch} has no Seperate* binding"
        assert arch in uvr_paths.MODELS, f"{arch} is in APPLY but has no weight"
        assert uvr_paths.MODELS[arch].is_file(), f"{arch} weight is not a file"
    for arch, cls in runner.ARCH_CLASS.items():
        assert arch in APPLY, f"{cls.__name__} is bound but not gated"


def test_blocked_archs_are_reported_not_simulated():
    """MDX-C / Roformer have no weights here; nothing may pretend otherwise.

    An arch may enter ``APPLY`` only together with a real weight file, so a
    future row cannot claim coverage the machine does not provide.
    """
    for absent in ("mdx_c", "roformer"):
        if absent not in uvr_paths.MODELS:
            assert absent not in APPLY, f"{absent} is in APPLY without a weight"
            assert absent not in runner.ARCH_CLASS, f"{absent} is bound without a weight"
    # The configs really are present, which is exactly the false positive the
    # fixture manifest was written to avoid (a directory is not a weight).
    assert "mdx_c" in uvr_paths.CONFIGS
    assert "mdx_c" not in uvr_paths.MODELS


# --------------------------------------------------------------------------- #
# Capture
# --------------------------------------------------------------------------- #

def _sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _previous_pins() -> dict:
    """stem -> sha256 from the manifest of an earlier capture run, if any."""
    if not settings.MANIFEST_FILE.is_file():
        return {}
    manifest = json.loads(settings.MANIFEST_FILE.read_text(encoding="utf-8"))
    return {
        record["arch"]: {
            stem: entry["sha256"] for stem, entry in record["stems"].items()
        }
        for record in manifest.get("captures", [])
    }


@pytest.fixture(scope="module")
def captures(tmp_path_factory) -> dict:
    """Run every gated arch once on CPU, re-run it to prove determinism, then
    publish the baselines plus the manifest.

    Module-scoped because these are real separations: VR ~10 s and htdemucs
    ~11 s per run on this machine for the 30-second crop.
    """
    previous = _previous_pins()
    crop_duration = sf.info(str(runner.golden_input())).frames / 44100

    records: dict[str, dict] = {}
    for arch in APPLY:
        record = runner.capture_baseline(arch, "cpu", tmp_path_factory.mktemp(f"{arch}-cpu"))
        record["previous_pins"] = previous.get(arch, {})

        repeat_dir = tmp_path_factory.mktemp(f"{arch}-cpu-repeat")
        record["repeat_sha256"] = {
            stem: _sha256(path) for stem, path in runner.separate(arch, "cpu", repeat_dir).items()
        }
        record["crop_duration_seconds"] = round(crop_duration, 6)
        shutil.rmtree(repeat_dir, ignore_errors=True)
        records[arch] = record

    manifest_path = runner.write_manifest(list(records.values()))
    return {"records": records, "manifest": manifest_path}


@pytest.mark.parametrize("arch", APPLY)
def test_cpu_baseline_files_are_non_empty_and_full_length(captures, arch):
    record = captures["records"][arch]
    assert record["stems"], f"{arch} wrote no stems"
    for stem, entry in record["stems"].items():
        assert entry["bytes"] > 44, f"{arch}/{stem} is empty or header-only"
        assert entry["sample_rate"] == 44100, f"{arch}/{stem} unexpected rate"
        assert entry["channels"] == 2, f"{arch}/{stem} is not stereo"
        ratio = entry["duration_seconds"] / record["crop_duration_seconds"]
        assert 0.99 <= ratio <= 1.01, (
            f"{arch}/{stem} runs {entry['duration_seconds']}s against a "
            f"{record['crop_duration_seconds']}s crop ({ratio:.3f}x)"
        )


@pytest.mark.parametrize("arch", APPLY)
def test_cpu_capture_is_byte_reproducible(captures, arch):
    """The pinned seed must make two runs identical, or the gate measures noise."""
    record = captures["records"][arch]
    for stem, entry in record["stems"].items():
        assert stem in record["repeat_sha256"], f"{arch}: repeat run lost {stem}"
        assert entry["sha256"] == record["repeat_sha256"][stem], (
            f"{arch}/{stem} differs between two identical CPU runs -- "
            "unpinned randomness. Fixing this is a prerequisite for the 40 dB gate."
        )


def test_separate_pins_the_random_seed(monkeypatch, tmp_path):
    """``demucs/apply.py:220`` draws ``random.randint``; the harness must seed it.

    Asserted structurally (what ``separate`` does) rather than by running an
    unseeded capture, so the check costs nothing and cannot be mistaken for a
    baseline refresh.
    """
    seen: list[int] = []
    monkeypatch.setattr(runner.random, "seed", lambda value: seen.append(value))
    runner.separate("vr", "cpu", tmp_path)
    assert seen == [settings.RANDOM_SEED], (
        "separate() must seed random with settings.RANDOM_SEED immediately "
        "before running the engine"
    )


@pytest.mark.parametrize("arch", ["vr", "demucs"])
def test_gate_discriminates_on_real_audio(captures, arch):
    """Negative control with the actual baselines: different audio scores low.

    ``vr_Vocals`` and ``demucs_Vocals`` are both valid separations of the same
    crop by different networks, and the gate must call them *not* equivalent.
    Without this, a ``snr_db`` that always returned a large number would keep
    every Task 7 assertion green.

    The two stems are compared on SNR *and* on peak deviation: the fixture mix
    is instrumental, so the vocal stems are quiet and decibels alone would be an
    unfair measure -- see the ``tests.golden.runner`` docstring note for Task 7.
    """
    other = "demucs" if arch == "vr" else "vr"
    a_path = pathlib.Path(captures["records"][arch]["stems"]["Vocals"]["path"])
    b_path = pathlib.Path(captures["records"][other]["stems"]["Vocals"]["path"])
    a, _ = sf.read(str(a_path), dtype="float64")
    b, _ = sf.read(str(b_path), dtype="float64")

    assert snr_db(a, b) < 40.0, (
        f"{arch} vs {other} vocal stems scored above the gate -- the SNR "
        "helper is not discriminating on real audio"
    )
    assert max_abs_dev(a, b) > 0.01
    # ...and a signal is still identical to itself.
    assert snr_db(a, a.copy()) == float("inf")


@pytest.mark.parametrize("arch", APPLY)
def test_pinned_fingerprint_did_not_move(captures, arch):
    """Against the previous run's manifest: the baseline must not drift silently.

    First capture on a fresh checkout has nothing to compare with; after that,
    a difference means a dependency or engine change moved the numbers, and
    refreshing the pin must be a deliberate act (delete the manifest, re-run).
    """
    record = captures["records"][arch]
    previous = record["previous_pins"]
    if not previous:
        assert settings.MANIFEST_FILE.is_file(), "no manifest after capture"
        return
    for stem, pinned in previous.items():
        assert stem in record["stems"], f"{arch}: {stem} disappeared from the baseline"
        assert record["stems"][stem]["sha256"] == pinned, (
            f"{arch}/{stem} drifted away from the pinned fingerprint. If the "
            "change is intentional, delete tests/golden/_artifacts/baseline/"
            "manifest.json and re-capture; do not re-record it quietly."
        )


def test_manifest_records_the_crop_environment_and_timings(captures):
    manifest = json.loads(captures["manifest"].read_text(encoding="utf-8"))

    crop = manifest["input"]
    assert crop["crop_duration_seconds"] == pytest.approx(settings.CROP_SECONDS, abs=0.01)
    assert crop["crop_sample_rate"] == 44100
    assert crop["crop_channels"] == 2
    assert crop["crop_sha256"], "crop must be fingerprinted for Tasks 7 and 11"

    env = manifest["environment"]
    for key in ("python", "torch", "numpy", "librosa", "scipy", "soundfile"):
        assert env[key], f"manifest is missing the {key} version"

    assert manifest["archs_covered"] == APPLY
    assert set(manifest["archs_blocked"]) == {"mdx_c", "roformer", "onnx"}
    assert manifest["random_seed"] == settings.RANDOM_SEED
    for arch in APPLY:
        record = next(r for r in manifest["captures"] if r["arch"] == arch)
        assert record["elapsed_seconds"] > 0
        assert record["model_data_attrs"], "settings snapshot missing"
        assert record["model_data_attrs"]["process_method"] == settings.make(arch, "cpu")["process_method"]


def test_manifest_snapshot_keeps_the_vr_network_params_readable(captures):
    """The ModelParameters object must survive JSON-ification, not become repr()."""
    manifest = json.loads(captures["manifest"].read_text(encoding="utf-8"))
    vr = next(r for r in manifest["captures"] if r["arch"] == "vr")
    params = vr["model_data_attrs"]["vr_model_param"]
    assert isinstance(params, dict) and "ModelParameters" in params
    band = params["ModelParameters"]["band"]
    assert str(4) in band and band["4"]["res_type"] == "sinc_medium", (
        "the VR res_type that the Task 3 guard protects must stay visible "
        "in the manifest"
    )
