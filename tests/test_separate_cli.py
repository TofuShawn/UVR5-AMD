"""Task 13: the GUI-free CLI, tested as the fork's headline claim.

Six contracts, listed in the order of cost rather than importance: test 4 is the
one that matters most, and it is last in its section because it is the slowest.

1. **Tier 3 is a hard error that tells you what to do.** A weight whose params no
   table knows must exit non-zero *naming its own hash and the exact `<hash>.json`
   path* -- because a GUI-free run cannot open the params dialog, that string is the
   only self-service path a CLI user has. Asserting the two substrings is the point;
   asserting the exit code alone would pass with any error message.
2. **Blocked, not simulated -- proved by absence.** A refused arch must write no
   output. A CLI that prints a refusal and then runs anyway passes a
   message-only test, so every refusal here checks the output directory is empty.
   No test fabricates a roformer/onnx/mdx_c run: those weights do not exist on this
   machine, so the only honest assertion available is that nothing happened.
3. **`process_data`'s 11 keys, pinned from both sides.** A hand-written frozenset,
   compared by exact equality against what the CLI builds *and* against an AST
   re-derivation of the keys `separate.py` indexes. The message is
   `test_engine_purity.pin_drift`'s, because the failure worth predicting is a
   future engine key arriving: it must read as "NOT IN the pin", not as a silent gap.
4. **GUI-free for real, not by inspection.** The purity guard proves the *import
   graph* is tkinter-free. These two children prove the *run* is: the same armed
   blocker, then a real `vr` separation to a temp directory. If anything in the call
   path reaches for a widget, `tkinter` raises and the child exits non-zero.
5. **Device honesty.** The CLI prints the `Backend` `backends.resolve()` returned --
   forwarding Task 6's reporting, not duplicating it -- and a GPU request the
   interpreter cannot honour is stated in the output.
6. **No network in this file's own import graph.** AST allow-list plus the
   `urlopen`-raises fixture from `tests/test_model_refresh.py`, so even a test that
   forgot to inject a fake fails loudly instead of fetching.
"""
from __future__ import annotations

import ast
import io
import os
import subprocess
import sys
import types
import urllib.request
from pathlib import Path

import pytest

import catalog
import separate_cli
import uvr_paths
from tests.test_engine_purity import (CHILD_TIMEOUT_SECONDS, ENGINE_IMPORTS,
                                      guard_script, pin_drift)

REPO = Path(__file__).resolve().parents[1]
CLI = REPO / "separate_cli.py"


# --------------------------------------------------------------------------- #
# no network, ever, in this file
# --------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def _forbid_real_network(monkeypatch):
    """The `tests/test_model_refresh.py` pattern.

    Nothing here should fetch, and a test that fetched would be one that could pass
    on this machine and fail on another.
    """
    def forbidden(*args, **kwargs):
        raise AssertionError(f"a test tried to fetch for real: {args!r} {kwargs!r}")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def run_cli(*argv: str | Path, timeout: int = 120) -> subprocess.CompletedProcess:
    """The CLI as a user invokes it: a real child process, real exit code."""
    return subprocess.run([sys.executable, str(CLI), *[str(a) for a in argv]],
                          cwd=REPO, capture_output=True, text=True, timeout=timeout)


def synthetic_root(tmp_path: Path, arch: str = "vr", name: str = "mystery-model"):
    """A weights root holding one weight whose params nobody knows (tier 3).

    The bytes are random, so the digest is random, so no tier-1 `<hash>.json` and no
    tier-2 table row can possibly match -- which is what makes this a real tier-3
    case rather than a fixture that happens to look like one.
    """
    root = tmp_path / "model-root"
    spec = catalog.ARCHS[arch]
    directory = root / spec.weight_dirs[0]
    directory.mkdir(parents=True, exist_ok=True)
    weight = directory / f"{name}{spec.suffixes[0]}"
    weight.write_bytes(os.urandom(4096))
    return root, weight


def wav_files(directory: Path) -> list[Path]:
    return sorted(Path(directory).rglob("*.wav"))


# --------------------------------------------------------------------------- #
# 1. tier 3
# --------------------------------------------------------------------------- #

def test_tier3_failure_names_the_hash_and_the_exact_json_it_wanted(tmp_path):
    root, weight = synthetic_root(tmp_path)
    writable = tmp_path / "writable"
    out = tmp_path / "out"
    digest = catalog.hash_of(weight)

    proc = run_cli("--arch", "vr", "--model", weight.stem,
                   "--audio", tmp_path / "input.flac", "--out", out,
                   "--model-root", root, "--writable-dir", writable)
    printed = proc.stdout + proc.stderr

    assert proc.returncode != 0, f"tier 3 exited 0 -- the CLI invented params:\n{printed}"
    assert proc.returncode == separate_cli.EXIT_TIER3, printed
    # The payload: the digest, the file it wanted, and where that file goes.
    assert digest in printed, f"hash {digest} missing from the refusal:\n{printed}"
    assert f"{digest}.json" in printed, printed
    assert str(writable) in printed, (
        f"the refusal must name the writable directory the user can write into:\n{printed}")
    # Refused *and* did not run.
    assert not wav_files(out), "tier-3 refusal still wrote stems"
    assert not writable.exists() or not list(writable.rglob("*.json")), (
        "the CLI wrote params it did not have")


# --------------------------------------------------------------------------- #
# 2. blocked, not simulated
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("arch", sorted(catalog.BLOCKED_ARCHS))
def test_blocked_arch_refuses_and_writes_no_output(tmp_path, arch):
    root = tmp_path / "model-root"
    out = tmp_path / "out"
    out.mkdir()
    writable = tmp_path / "writable"

    proc = run_cli("--arch", arch, "--model", "Whatever",
                   "--audio", tmp_path / "input.flac", "--out", out,
                   "--model-root", root, "--writable-dir", writable)
    printed = proc.stdout + proc.stderr

    assert proc.returncode == separate_cli.EXIT_BLOCKED_ARCH, printed
    # The catalog's *pinned* reason, verbatim -- a paraphrase loses the payload.
    assert catalog.BLOCKED_ARCHS[arch] in printed, (
        f"the refusal for {arch} did not carry the pinned reason text:\n{printed}")
    assert not wav_files(out), (
        f"{arch} was refused in words and then separated anyway: "
        f"{wav_files(out)} -- blocked means nothing is written")
    assert not any(out.rglob("*")), f"{arch} wrote into the output dir: {list(out.rglob('*'))}"


def test_an_unknown_arch_name_is_argparse_s_business_and_lists_the_choices(tmp_path):
    """`--arch mdx_c` reaches the catalog; a made-up name reaches argparse, usefully.

    `--arch` deliberately does *not* restrict `choices` to the runnable set: an
    argparse error for `mdx_c` would print usage text and lose
    `BLOCKED_ARCHS`' reason, which is the payload a user needs.
    """
    proc = run_cli("--arch", "graffiti", "--model", "Whatever", "--out", tmp_path / "out")
    assert proc.returncode == 2, (
        f"a name that is not an architecture at all is argparse's business:\n"
        f"{proc.stdout}{proc.stderr}")
    for arch in sorted(catalog.BLOCKED_ARCHS):
        assert arch in (proc.stdout + proc.stderr), (
            "argparse's usage text must list the blocked archs too, or a user cannot "
            f"discover that {arch} is a real thing this fork refuses")


# --------------------------------------------------------------------------- #
# 3. the process_data contract, pinned from both sides
# --------------------------------------------------------------------------- #

#: Hand-written, on purpose: everything else in this test derives its expectation
#: from `separate.py` or from `build_process_data`, and a pin rebuilt from either
#: side is a self-audit. Copied from `separate.py:138-152` and cross-checked against
#: `tests/golden/runner.py:make_process_data`.
PINNED_PROCESS_DATA_KEYS = frozenset({
    "audio_file", "audio_file_base", "cached_model_source_holder",
    "cached_source_callback", "export_path", "is_4_stem_ensemble",
    "is_ensemble_master", "list_all_models", "process_iteration",
    "set_progress_bar", "write_to_console",
})


def engine_process_data_keys() -> frozenset[str]:
    """AST re-derivation: every `process_data['…']` the engine's own `__init__` indexes.

    Read from the `SeperateAttributes.__init__` body only -- the class every
    `Seperate*` subclass inherits without defining its own `__init__`, so this is the
    whole unconditional contract, not one arch's extras.
    """
    tree = ast.parse((REPO / "separate.py").read_text(encoding="utf-8"))
    cls = next((n for n in tree.body
                if isinstance(n, ast.ClassDef) and n.name == "SeperateAttributes"), None)
    assert cls is not None, "separate.py no longer defines SeperateAttributes"
    init = next((n for n in cls.body
                 if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
    assert init is not None, "SeperateAttributes has no __init__ to index process_data"
    keys = set()
    for node in ast.walk(init):
        if (isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Name) and node.value.id == "process_data"
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)):
            keys.add(node.slice.value)
    return frozenset(keys)


def test_process_data_keys_are_pinned_against_the_engine_and_the_cli(tmp_path):
    built = separate_cli.build_process_data(tmp_path / "a song.flac", tmp_path / "out")

    assert frozenset(built) == PINNED_PROCESS_DATA_KEYS, pin_drift(
        "the CLI's process_data", built.keys(), PINNED_PROCESS_DATA_KEYS,
        "PINNED_PROCESS_DATA_KEYS")
    engine = engine_process_data_keys()
    assert engine == PINNED_PROCESS_DATA_KEYS, pin_drift(
        "separate.py's process_data subscripts", engine,
        PINNED_PROCESS_DATA_KEYS, "PINNED_PROCESS_DATA_KEYS")
    # Three of the eleven are callables and the engine calls them; a None here is an
    # AttributeError deep inside a separation, after the model is already loaded.
    for name in ("set_progress_bar", "write_to_console", "cached_source_callback",
                 "cached_model_source_holder", "process_iteration"):
        assert callable(built[name]), f"{name} is not callable"
    assert built["cached_source_callback"]("VR Arc") == (None, None), (
        "a source cache that reports a hit would short-circuit real inference")
    assert built["is_ensemble_master"] is False and built["is_4_stem_ensemble"] is False
    assert built["list_all_models"] == []
    assert built["audio_file_base"] == "a song"
    assert Path(built["export_path"]).is_dir(), (
        "build_process_data owns mkdir of the export path, like the golden harness")


# --------------------------------------------------------------------------- #
# 4. GUI-free for real: the armed blocker, then a separation
# --------------------------------------------------------------------------- #

def blocked_child_source(argv: list[str]) -> str:
    """The purity guard's own script, then the CLI as `__main__` inside that child.

    `guard_script(ENGINE_IMPORTS)` is reused, not reinvented: it installs the finder
    over `GUI_PACKAGES`, purges `sys.modules`, and self-asserts that
    `import tkinter` raises before anything else happens -- so a child that reached
    the CLI already knows its blocker works. Importing all of `ENGINE_IMPORTS`
    first is deliberate: `separate_cli` joined that list in this slice, and the
    claim the README makes is about the whole call path, including the torch import
    `separate` brings.
    """
    return (
        guard_script(ENGINE_IMPORTS)
        + "import runpy, sys\n"
        + f"sys.argv = [{str(CLI)!r}] + {argv!r}\n"
        + f"runpy.run_path({str(CLI)!r}, run_name='__main__')\n"
    )


def run_under_blocker(argv: list[str],
                      timeout: int = CHILD_TIMEOUT_SECONDS) -> subprocess.CompletedProcess:
    proc = subprocess.run([sys.executable, "-c", blocked_child_source([str(a) for a in argv])],
                          cwd=REPO, capture_output=True, text=True, timeout=timeout)
    printed = proc.stdout + proc.stderr
    assert "blocker ineffective" not in printed, printed
    return proc


def vr_run_fixture(tmp_path: Path) -> tuple[str, str, str, str]:
    """(model root, vr model name, audio path, out dir) for a real run -- or a failure.

    Fail rather than skip, in this fork's established style
    (`tests/golden/settings.py` raises `FixtureMissing` for exactly this): a skipped
    test here would be the headline claim quietly stopping being tested. The message
    says what to provide, not that the machine is wrong.
    """
    root = uvr_paths.MODEL_ROOT
    cat = catalog.Catalog(root, writable_dir=tmp_path / "writable")
    names = cat.names("vr") if root.exists() else []
    model = "1_HP-UVR" if "1_HP-UVR" in names else (names[0] if names else None)
    missing = []
    if model is None:
        missing.append(f"no VR weight under {root} -- set UVR_MODEL_ROOT to a weights "
                       f"dir holding VR_Models/*.pth")
    if not uvr_paths.INPUT_FILE.is_file():
        missing.append(f"golden input {uvr_paths.INPUT_FILE} is absent -- "
                       f"set UVR_INPUT_FILE")
    if missing:
        pytest.fail("test 4's real separation is BLOCKED, not simulated: "
                    + "; ".join(missing))
    return str(root), model, str(uvr_paths.INPUT_FILE), str(tmp_path / "blocked-run-out")


def test_dry_run_exits_zero_with_tkinter_unimportable(tmp_path):
    """The cheap half of the claim: resolve + plan + print, blocker armed, exit 0."""
    root, model, audio, out = vr_run_fixture(tmp_path)
    proc = run_under_blocker(["--audio", audio, "--out", out, "--model-root", root,
                              "--writable-dir", tmp_path / "writable",
                              "--arch", "vr", "--model", model,
                              "--device", "cpu", "--dry-run"])
    assert proc.returncode == 0, (
        f"--dry-run failed under the tkinter blocker:\n{proc.stdout}\n{proc.stderr}")
    assert "plan: RunPlan with 92 fields" in proc.stdout, proc.stdout
    assert not wav_files(Path(out)), "--dry-run wrote stems; --dry-run must run nothing"


def test_a_real_vr_separation_completes_with_tkinter_unimportable(tmp_path):
    """The fork's headline claim as one assertion: audio out, GUI unimportable.

    Nothing else in the suite proves this. The purity guard proves the modules
    *import* clean; a widget reached only inside `seperate()` -- a `root.update()`
    in an inference loop, a `tkinterdnd2` import behind a progress callback --
    passes that guard and breaks every headless run. This child is a real ROCm
    separation of the golden crop, so it inherits `CHILD_TIMEOUT_SECONDS` (600 s):
    ROCm cold start plus MIOpen's first-run kernel search is genuinely slow, and
    that bound exists to turn a hang into a failure, not to be polite.
    """
    root, model, audio, out = vr_run_fixture(tmp_path)
    proc = run_under_blocker(["--audio", audio, "--out", out, "--model-root", root,
                              "--writable-dir", tmp_path / "writable",
                              "--arch", "vr", "--model", model, "--device", "rocm"])
    assert proc.returncode == 0, (
        "a real separation did not complete with tkinter blocked -- this is the "
        "claim the whole fork makes:\n"
        f"stdout(tail):\n{proc.stdout[-4000:]}\nstderr(tail):\n{proc.stderr[-4000:]}")
    written = wav_files(Path(out))
    stems = {p.stem for p in written}
    assert len(written) == 2, f"expected two stems, got {sorted(str(p) for p in written)}"
    base = os.path.splitext(os.path.basename(audio))[0]
    for stem in ("Vocals", "Instrumental"):
        expected = f"{base}_({stem})"
        assert expected in stems, f"{expected}.wav missing from {sorted(stems)}"
        path = Path(out) / f"{expected}.wav"
        assert path.stat().st_size > 10_000, f"{path} is {path.stat().st_size} bytes"
    assert "[cli] wrote" in proc.stdout, proc.stdout[-2000:]
    assert "GUI package" not in proc.stdout + proc.stderr
    # Evidence for the report, visible under `-s`: the two WAVs this child wrote with
    # tkinter unimportable, by name and size.
    print(f"\n[blocker] rc={proc.returncode} model={model} root={root} audio={audio}")
    for path in written:
        print(f"[blocker] wrote {path.name} {path.stat().st_size} bytes")


# --------------------------------------------------------------------------- #
# 5. device honesty
# --------------------------------------------------------------------------- #

def _main_capturing(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = separate_cli.main([str(a) for a in argv], out=out, err=err)
    return code, out.getvalue(), err.getvalue()


def test_the_cli_forwards_the_backend_backends_resolved(tmp_path):
    """No device logic is re-derived here: what prints is what `resolve` returned."""
    import backends

    root, model, audio, out = vr_run_fixture(tmp_path)
    code, printed, err_printed = _main_capturing(
        "--audio", audio, "--out", out, "--model-root", root,
        "--writable-dir", tmp_path / "writable",
        "--arch", "vr", "--model", model, "--device", "cpu", "--dry-run")
    backend = backends.resolve(force="cpu")
    text = printed + err_printed
    assert f"name={backend.name}" in text, text
    assert f"torch_device={backend.torch_device}" in text, text
    assert f"ort_providers={backend.ort_providers}" in text, text
    assert "(requested 'cpu')" in text, text
    assert code == 0, text
    # CPU asked for and CPU honoured: there is no degradation to report.
    assert "WARNING" not in text, text


def test_a_gpu_request_the_environment_cannot_honour_is_said_out_loud(tmp_path, monkeypatch):
    """The silent-CPU-fallback failure mode, made visible by the CLI's own report."""
    import backends

    fake = types.SimpleNamespace(
        __version__="0.0+fake",
        version=types.SimpleNamespace(hip=None),
        cuda=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert backends.resolve(force="rocm").torch_device == "cuda"
    text = "\n".join(separate_cli.describe_device(backends.resolve(force="rocm"), "rocm"))
    assert "cuda_available=False" in text, text
    assert "WARNING" in text and "compute on CPU" in text, (
        f"a rocm request that cannot open the device printed only good news:\n{text}")


# --------------------------------------------------------------------------- #
# 6. no network
# --------------------------------------------------------------------------- #

NETWORK_ROOTS = ("urllib", "socket", "requests", "http", "ftplib", "urllib3", "ssl")


def imported_module_names(path: Path, *, top_level_only: bool = False) -> set[str]:
    """Module names this file imports, whole-tree or at module scope only.

    `top_level_only` deliberately does *not* descend: an `ast.walk` from a
    `FunctionDef` reaches the imports inside its body, which would make the
    lazy-import assertion below fail on the code it exists to protect.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()

    def add(node) -> None:
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)

    if top_level_only:
        for child in tree.body:
            add(child)
    else:
        for node in ast.walk(tree):
            add(node)
    return found


def test_the_cli_imports_no_network_module_at_all():
    names = imported_module_names(CLI)
    offending = sorted(n for n in names if n.split(".")[0] in NETWORK_ROOTS)
    assert not offending, (
        f"separate_cli.py imports {offending}; this fork has no app-initiated network "
        "(Task 12) and the CLI is what a headless run trusts")
    # `model_refresh` is the one network-capable module the CLI touches, and it
    # arrives lazily inside the command that asks for it.
    assert "model_refresh" in names, "the refresh path must delegate, not re-implement"
    eager = imported_module_names(CLI, top_level_only=True)
    assert "model_refresh" not in eager, (
        "model_refresh is imported at module scope, so every --list-models drags the "
        "network-capable module into the import graph")
    assert "separate" not in eager, (
        "separate imports torch; importing it eagerly would make --list-models and "
        "--dry-run require the ROCm venv for no reason")


def test_refresh_without_confirmation_refuses_before_any_fetch(tmp_path):
    """Non-zero, no fetch, no write -- and the refusal is `model_refresh`'s own text."""
    writable = tmp_path / "writable"
    proc = run_cli("--refresh-model-tables", "--model-root", tmp_path / "empty-root",
                   "--writable-dir", writable)
    printed = proc.stdout + proc.stderr
    assert proc.returncode != 0, printed
    assert proc.returncode == separate_cli.EXIT_REFRESH_REFUSED, printed
    assert "refusing to touch the network" in printed, (
        f"the CLI paraphrased model_refresh's refusal instead of printing it:\n{printed}")
    assert not list(writable.rglob("*.json")), "a refused refresh wrote table files"
