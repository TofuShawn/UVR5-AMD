"""Task 4 golden harness tests: capture CPU baselines, prove the harness is real.

A capture that merely "ran" proves almost nothing, so five kinds of assertion
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
5. *Pin discipline* -- the fingerprint tripwire has to be able to stay red.
   The version of this file before the Task 5 fix round read the pins out of the
   manifest, then overwrote that same manifest with the new shas, so any real
   drift tripped exactly one run and healed itself (task-5-report.md section 6.4).
   ``test_pinned_fingerprint_did_not_move`` now scores the capture against the
   tracked ``tests/golden/pins.json``, which nothing may rewrite without
   ``UVR_REPIN=1``, and it is backed by a negative control that perturbs one
   sample by one PCM_16 LSB to show the comparison is not vacuous.
"""
from __future__ import annotations

import ast
import hashlib
import json
import pathlib
import shutil

import numpy as np
import pytest
import soundfile as sf
import torch

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


#: The pin store as it stood at collection time -- before any fixture in this
#: session could have touched it. ``test_repin_is_an_explicit_opt_in`` compares
#: the file against this snapshot, which is what makes "the default path never
#: moves a pin" an asserted property instead of a promise.
PINS_AT_COLLECTION = (settings.PINS_FILE.read_bytes()
                      if settings.PINS_FILE.is_file() else None)

#: ``reason`` stamped into a deliberately re-pinned store. It is the lineage note
#: a future reader needs to know *which* stack these fingerprints are a fingerprint
#: of, and which move they already absorbed.
REPIN_REASON = (
    "explicit operator re-pin (UVR_REPIN=1) for the Task 5 ruling: CPU baselines "
    "re-captured with device forced to 'cpu' on the torch wheel this fork ships "
    "(2.9.1+rocm7.2.1), so Task 7's gate varies device only, never device + torch "
    "version. Lineage: Task 4 pinned torch 2.15.0+cpu (stem shas 01ee2cdfd480, "
    "e12c9f904e0a, c9ad85d18ce3, a1e429496cd7, d86869223c88, b15c877ae5f4). The "
    "2.15 -> 2.9.1 move shifted CPU output by exactly 1 PCM_16 LSB on "
    "0.002-0.23% of samples at SNR 53.45-125.88 dB (task-5-report.md section 6.3) "
    "and the self-healing fixture that used to read its own manifest as the pin "
    "store absorbed it, so the 2.15 WAVs no longer exist on disk; the delta stored "
    "beside this note is therefore 2.9.1-vs-2.9.1 and is zero by construction."
)


def _disagreements(diff: dict) -> list[str]:
    """Flatten one ``compare_fingerprints`` result into readable complaints."""
    out = [f"  {stem}: pinned {info['pinned'][:16]}... -> measured {info['measured'][:16]}..."
           for stem, info in diff["changed"].items()]
    out += [f"  {stem}: pinned, but this run no longer writes it"
            for stem in diff["missing"]]
    out += [f"  {stem}: this run writes it, but nothing pins it"
            for stem in diff["unpinned"]]
    return out


@pytest.fixture(scope="module")
def captures(tmp_path_factory) -> dict:
    """Run every gated arch once on CPU, re-run it to prove determinism, then
    decide -- loudly -- what happens to the pins.

    Module-scoped because these are real separations: VR ~8 s and htdemucs ~55 s
    per run on this machine for the 30-second crop.

    The pin policy this fixture implements (the Task 5 fix round; before it the
    fixture re-pinned the very manifest it had just compared against, so drift
    announced itself once and was then absorbed):

    * ``tests/golden/pins.json`` is read before anything is written and is the
      only authority; ``baseline/manifest.json`` is a report about this run.
    * A capture that reproduces the pinned bytes gets *published* into
      ``baseline/`` -- those WAVs are what Tasks 7 and 11 read, so a fresh
      checkout with an empty git-ignored ``_artifacts/`` rebuilds itself.
    * A capture that does not, or a ``baseline/`` whose files no longer match
      their pins, publishes **nothing**: the WAVs carrying the old numbers stay
      on disk as the evidence, and the guard tests below go red.
    * Only ``UVR_REPIN=1`` writes the pin store (and republishes). It stamps the
      engine identity and the measured audio delta it is absorbing.
    """
    pins_document = runner.read_pins()
    pinned = runner.pins_by_arch(pins_document)
    pins_existed = pins_document is not None
    disk_before = runner.baseline_fingerprints()
    crop_duration = sf.info(str(runner.golden_input())).frames / 44100

    records: dict[str, dict] = {}
    for arch in APPLY:
        record = runner.capture_baseline(arch, "cpu", tmp_path_factory.mktemp(f"{arch}-cpu"))
        record["pinned_before_run"] = pinned.get(arch, {})
        # Drift: what this run measured vs the pins.
        record["drift"] = runner.compare_fingerprints(
            pinned.get(arch, {}), runner.capture_fingerprints(record))
        # Integrity: the published WAVs as they stood *before* this run touched
        # anything -- a file edited or lost between runs is a different failure
        # from a capture that moved, and drift alone cannot see it.
        record["baseline_integrity"] = runner.compare_fingerprints(
            pinned.get(arch, {}), disk_before.get(arch, {}))

        repeat_dir = tmp_path_factory.mktemp(f"{arch}-cpu-repeat")
        record["repeat_sha256"] = {
            stem: _sha256(path) for stem, path in runner.separate(arch, "cpu", repeat_dir).items()
        }
        record["crop_duration_seconds"] = round(crop_duration, 6)
        shutil.rmtree(repeat_dir, ignore_errors=True)
        records[arch] = record

    # Old pinned WAV vs this capture, scored in audio terms *before* anything is
    # published over it -- the number a re-pin is otherwise free to swallow.
    absorbed: dict[str, dict] = {}
    for arch, record in records.items():
        delta = runner.measure_audio_delta(
            {stem: runner.baseline_target(arch, stem) for stem in record["stems"]},
            {stem: entry["staged_path"] for stem, entry in record["stems"].items()},
        )
        if delta:
            absorbed[arch] = delta

    drift_moved = [a for a, r in records.items() if _disagreements(r["drift"])]
    baseline_broken = [a for a, r in records.items()
                       if r["baseline_integrity"]["changed"] or r["baseline_integrity"]["unpinned"]]

    repin = runner.should_repin()
    publish = repin or (pins_existed and not drift_moved and not baseline_broken)
    if publish:
        for record in records.values():
            runner.publish_baseline(record)
    if repin:
        runner.write_pins(list(records.values()), reason=REPIN_REASON,
                          absorbed=absorbed, supersedes=pins_document)
        # The store this run just wrote is what the guards score against now; the
        # pre-run comparisons stay in the record as the report of what was absorbed.
        after = runner.pins_by_arch(runner.read_pins())
        disk_after = runner.baseline_fingerprints()
        for name, entry in records.items():
            entry["drift_at_start"] = entry["drift"]
            entry["baseline_integrity_at_start"] = entry["baseline_integrity"]
            entry["drift"] = runner.compare_fingerprints(
                after.get(name, {}), runner.capture_fingerprints(entry))
            entry["baseline_integrity"] = runner.compare_fingerprints(
                after.get(name, {}), disk_after.get(name, {}))

    if repin:
        action = "repinned-and-published"
    elif not pins_existed:
        action = "blocked-no-pin-store"
    elif drift_moved:
        action = "blocked-capture-moved-away-from-pins"
    elif baseline_broken:
        action = "blocked-published-baseline-disagrees-with-pins"
    else:
        action = "published-pinned-output"

    manifest_path = runner.write_manifest(list(records.values()), extra={
        "pin_policy": {
            "pin_store": str(settings.PINS_FILE),
            "pin_store_existed_before_run": pins_existed,
            "repin_opt_in_variable": settings.REPIN_ENV_VAR,
            "repin_opt_in_given": repin,
            "action": action,
            "published": publish,
            "drifted_archs": drift_moved,
            "baseline_integrity_archs": baseline_broken,
            "absorbed_audio_delta": absorbed,
        },
    })
    return {
        "records": records,
        "manifest": manifest_path,
        "pins_existed_before_run": pins_existed,
        "repinned": repin,
        "published": publish,
        "action": action,
        "absorbed_delta": absorbed,
    }


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
    """A CPU fingerprint that moved makes the suite RED and keeps it red.

    This is the fork's tripwire against the plan's named top risk -- silent
    numeric drift after a dependency upgrade -- so it has to be able to *stay*
    failing. Reading the pins out of ``baseline/manifest.json`` did not do that:
    the fixture rewrote that file with the fresh shas at the end of every run, so
    a real move (torch 2.15.0+cpu -> 2.9.1+rocm7.2.1, 1 PCM_16 LSB on every stem)
    tripped once and healed itself on the next run. The pin store is now the
    tracked ``tests/golden/pins.json``, and the default path cannot write it.

    Two directions of drift, both fatal:
      * the capture no longer reproduces its pin (a dependency or engine change
        moved the numbers);
      * the published ``baseline/*.wav`` no longer match their pin (a reference
        file was edited, truncated or lost -- Task 7 would silently score against
        it, and capture-vs-pin cannot see this at all).
    """
    record = captures["records"][arch]
    if not captures["pins_existed_before_run"] and not captures["repinned"]:
        pytest.fail(
            f"no pin store at {settings.PINS_FILE}: {arch}'s fingerprints are "
            "unpinned, so drift could not be detected at all. Creating a pin "
            "store is a deliberate act -- read what the capture measured, then "
            f"run once with {settings.REPIN_ENV_VAR}=1. A normal run must never "
            "invent pins."
        )

    complaints = _disagreements(record["drift"])
    assert not complaints, (
        f"\n{arch}: the CPU capture moved away from its pinned fingerprint:\n"
        + "\n".join(complaints)
        + "\n\nPins were captured under "
          f"{json.dumps((runner.read_pins() or {}).get('identity', {}), sort_keys=True)}"
          f"; this run is {json.dumps(runner.engine_identity('cpu'), sort_keys=True)}."
          "\n\nA red tripwire here is a finding, not a step. Do not 'fix' it by "
          "re-pinning. Read the delta first: the run report at "
          f"{settings.MANIFEST_FILE} carries this run's measurement plus the audio "
          "comparison against the previous WAVs. If the change is understood and "
          "acceptable, an operator says so out loud: "
          f"{settings.REPIN_ENV_VAR}=1 pytest ..., which moves "
          f"{settings.PINS_FILE} and stamps what it absorbed. Until then nothing "
          "in this harness rewrites a pin or overwrites the baseline WAVs that "
          "still carry the old numbers."
    )

    integrity = record["baseline_integrity"]
    complaints = _disagreements(integrity)
    assert not integrity["changed"] and not integrity["unpinned"], (
        "\n" + arch + ": the published baselines no longer match their pins:\n"
        + "\n".join(complaints)
        + "\nA baseline WAV that disagrees with its pin is exactly what Task 7 "
          "would score against, and this run refused to publish over it so the "
          "disagreement stays visible. Restore the file, or, if the reference set "
          "really is meant to change, say so out loud with "
          f"{settings.REPIN_ENV_VAR}=1."
    )
    assert not integrity["missing"] or captures["published"], (
        f"{arch}: pinned stems absent from {settings.BASELINE_DIR} and this run "
        f"refused to publish them (action={captures['action']}); the gate has no "
        "reference audio to score against until that is resolved."
    )


def test_repin_is_an_explicit_opt_in_and_a_normal_run_moves_no_pin(monkeypatch, captures):
    """Nothing short of ``UVR_REPIN=1`` may rewrite the pin store.

    Two halves: the gate itself only answers "yes" to the exact string, and the
    session that produced ``captures`` -- a plain ``pytest`` run with no opt-in --
    left ``pins.json`` byte-for-byte as it found it. That second half is the
    regression test for Task 5's self-heal.
    """
    for value in (None, "", "0", "no", "yes", "true", "2", "01", "1x", "re-pin"):
        if value is None:
            monkeypatch.delenv(settings.REPIN_ENV_VAR, raising=False)
        else:
            monkeypatch.setenv(settings.REPIN_ENV_VAR, value)
        assert runner.should_repin() is False, (
            f"{settings.REPIN_ENV_VAR}={value!r} must not be read as consent to "
            "re-pin; only the exact '1' is"
        )

    monkeypatch.setenv(settings.REPIN_ENV_VAR, "1")
    assert runner.should_repin() is True

    # ...and the writer refuses independently of whoever asked it, so a future
    # caller cannot reach past the gate by accident.
    monkeypatch.delenv(settings.REPIN_ENV_VAR, raising=False)
    with pytest.raises(RuntimeError, match="explicit opt-in"):
        runner.write_pins([], reason="this call must be refused")

    if captures["repinned"]:
        assert settings.PINS_FILE.is_file(), "an opted-in re-pin must leave a store"
    elif PINS_AT_COLLECTION is None:
        pytest.fail(f"{settings.PINS_FILE} did not exist when the session started "
                    "and this run did not create it deliberately")
    else:
        assert settings.PINS_FILE.read_bytes() == PINS_AT_COLLECTION, (
            "a plain pytest run rewrote the pin store -- the self-heal is back"
        )


def test_tripwire_is_not_vacuous_on_a_one_lsb_perturbation(captures, tmp_path):
    """One PCM_16 LSB -- the exact drift class Task 5 measured -- must be caught.

    A sha comparison is only as good as its smallest observable change, so this
    perturbs every real baseline by one bit on one sample and shows both halves
    of the guard reacting: ``compare_fingerprints`` names the stem, and
    ``measure_audio_delta`` reports 1 differing sample at a deviation of one LSB.
    Costs one temp-dir WAV copy per stem, no separation.
    """
    checked = 0
    for arch in APPLY:
        record = captures["records"][arch]
        for stem, entry in sorted(record["stems"].items()):
            source = pathlib.Path(entry["path"])
            data, rate = sf.read(str(source), dtype="int16")
            flat = data.reshape(-1)
            movable = np.flatnonzero(np.abs(flat) < 32767)
            assert movable.size, f"{arch}/{stem} is pinned to the int16 rail"
            index = int(movable[0])
            flat[index] = flat[index] + 1 if flat[index] >= 0 else flat[index] - 1

            perturbed = tmp_path / f"{arch}_{stem}_perturbed.wav"
            sf.write(str(perturbed), data, rate, subtype="PCM_16", format="WAV")
            diff = runner.compare_fingerprints(
                {stem: entry["sha256"]}, {stem: _sha256(perturbed)})
            assert stem in diff["changed"], (
                f"{arch}/{stem}: a single-LSB change slipped past the fingerprint "
                "comparison -- the tripwire would be decorative"
            )

            row = runner.measure_audio_delta({stem: source}, {stem: perturbed})[stem]
            assert row["samples_differing"] == 1, row
            assert abs(row["max_abs_dev"] - runner.PCM_16_LSB) < 1e-12, row
            assert row["deviation_above_1_pcm16_lsb"] is False, row
            # The one-LSB class stays far above the gate, exactly as Task 5 found.
            assert row["snr_db"] > 40.0 and row["snr_below_40db_gate"] is False, row
            checked += 1
    assert checked == sum(len(r["stems"]) for r in captures["records"].values())



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
    # Identity, not just versions: which torch, which HIP, and which device the
    # harness *asked for*. Under the ROCm wheel "cpu" is a real choice (that wheel
    # keeps its CPU kernels), so "no CUDA wheel installed" must never be mistaken
    # for "this fingerprint was produced on CPU".
    assert env["capture_device"] == ["cpu"]
    assert env["torch"] == torch.__version__
    assert env["torch_hip"] == getattr(torch.version, "hip", None)

    assert manifest["archs_covered"] == APPLY
    assert set(manifest["archs_blocked"]) == {"mdx_c", "roformer", "onnx"}
    assert manifest["random_seed"] == settings.RANDOM_SEED
    for arch in APPLY:
        record = next(r for r in manifest["captures"] if r["arch"] == arch)
        assert record["elapsed_seconds"] > 0
        assert record["model_data_attrs"], "settings snapshot missing"
        assert record["model_data_attrs"]["process_method"] == settings.make(arch, "cpu")["process_method"]

    # The pins have to say which stack they belong to, or a future reader cannot
    # tell a stale fingerprint from a current one -- the exact confusion that
    # made Task 4's torch 2.15 pins look authoritative after the ROCm install.
    pins = runner.read_pins()
    assert pins is not None, f"no pin store at {settings.PINS_FILE}"
    identity = pins["identity"]
    assert (identity["device"], identity["torch"], identity["torch_hip"]) == (
        "cpu", env["torch"], env["torch_hip"]), (
        "the pin store describes a different engine stack than the run judging "
        "it; re-pin deliberately, do not compare across stacks"
    )
    assert identity["crop_sha256"] == crop["crop_sha256"], "pins for another crop are not these pins"
    assert identity["random_seed"] == settings.RANDOM_SEED
    assert set(pins["archs_blocked"]) == {"mdx_c", "roformer", "onnx"}
    assert pins["gate"]["threshold_db"] == 40.0
    assert manifest["pin_policy"]["pin_store"] == str(settings.PINS_FILE)
    assert manifest["pin_policy"]["repin_opt_in_given"] is captures["repinned"]
    assert manifest["pin_policy"]["action"] == captures["action"]


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
