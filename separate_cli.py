"""Task 13: the GUI-free separation CLI.

Composition layer only. Everything that decides *what a run means* already lives
elsewhere and is tested there: `settings.Settings` (the 50 UI fields),
`catalog.Catalog` (four resolution tiers, the blocked-arch refusals, the tier-3
message that names a hash and a file), `plan.build()` (the 92-field `RunPlan`),
`backends.resolve()` (the device and its degradation notices),
`model_refresh.refresh()` (the only network-capable module in the fork). This file
pairs them up and prints what they answered; it re-implements none of them.

Why a composition layer is where the pairing is legal (ruling R-T10-writable):
`Catalog` must not reach into `settings` for its writable directory, because the
design's one-way dependency (`app -> settings`, `app -> engine`) is the diagram
everything rests on. The CLI is the `app`, so the CLI is where
`settings.config_dir()/model_data` and `uvr_paths.MODEL_ROOT` are allowed to meet.
`model_refresh.build_catalog()` performs the identical pairing for the refresh
path; both are composition roots, neither is a catalog internal.

Two properties the rest of the fork depends on and which this module owns:

* **No GUI, and no import of one.** Nothing here imports `tkinter`,
  `gui_data.*` or `PIL.ImageTk`, and `separate.py` -- which pulls torch, and so is
  by far the slowest import in the fork -- is imported *lazily*, only on the path
  that actually separates. `--list-models` and `--dry-run` therefore answer without
  touching torch, which keeps them usable on a machine whose venv is not the ROCm
  one. `tests/test_engine_purity.py` imports this module under the armed tkinter
  blocker, so the claim is executed and not asserted; `tests/test_separate_cli.py`
  then runs a real separation in a child where `tkinter` raises `ImportError`.
* **No network, and no import of a network module.** The AST of this file contains
  no `urllib`, `socket` or `requests` (pinned by test). `model_refresh` is imported
  lazily, inside the one command that asks for it, and it is called through its own
  confirmation rule rather than around it.

Refusals, not simulations (fork policy): an architecture with no weights on this
machine exits non-zero naming the catalog's own reason, and writes nothing. The
tier-3 case -- a weight whose model params are known to nobody -- exits non-zero
with the catalog's message, which carries the hash and the exact `<hash>.json` path
the user can write themselves. A CLI cannot open the params dialog, so that message
is the only self-service path there is.
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from pathlib import Path
from typing import Callable, TextIO

import backends
import catalog
import plan
import settings as settings_module
import uvr_paths

__all__ = ["ARCH_CLASS", "PROCESS_DATA_KEYS", "build_process_data", "main",
           "make_catalog", "run", "describe_compute_device", "describe_device",
           "writable_model_data_dir"]

# --------------------------------------------------------------------------- #
# exit codes -- a script may branch on these, so they are named, not ad-hoc
# --------------------------------------------------------------------------- #

EXIT_OK = 0
#: anything the engine raised during a real run
EXIT_ENGINE = 1
#: argparse's own code (a malformed command line)
EXIT_USAGE = 2
#: tier 3: the weight resolved, but nobody knows its params
EXIT_TIER3 = 3
#: blocked architecture: mdx_c / onnx / roformer (fork policy, see `catalog.BLOCKED_ARCHS`)
EXIT_BLOCKED_ARCH = 4
#: a declared model whose weight file is not on disk in any root
EXIT_MISSING_WEIGHT = 5
#: a model name no root's tables or files declare
EXIT_UNKNOWN_NAME = 6
#: `plan.build()` refused (Apollo-without-tier-1, a tier-3 spec, a missing MDX-C yaml…)
EXIT_PLAN = 7
#: `--refresh-model-tables` without `--confirm`, or a refresh that failed
EXIT_REFRESH_REFUSED = 8
#: the input audio is missing or not a file
EXIT_INPUT = 9
#: the writable catalog dir sits inside a read-only or git-tracked root
EXIT_WRITABLE_LOCATION = 10

#: Exception class -> exit code. Keyed on the classes the composition layer can
#: *receive*, so a new catalog failure class is a deliberate edit here and not a
#: silent drop into `EXIT_ENGINE`.
ERROR_CODES: dict[type[BaseException], int] = {
    catalog.UnknownModelParamsError: EXIT_TIER3,
    catalog.BlockedArchError: EXIT_BLOCKED_ARCH,
    catalog.MissingWeightError: EXIT_MISSING_WEIGHT,
    catalog.UnknownModelNameError: EXIT_UNKNOWN_NAME,
    catalog.WritableLocationError: EXIT_WRITABLE_LOCATION,
    plan.PlanError: EXIT_PLAN,
}


def code_for(exc: BaseException) -> int:
    """The exit code for `exc`, walking its own MRO.

    Walking the MRO rather than looking up `type(exc)` means a future catalog
    subclass of `MissingWeightError`, say, still gets the code that names its
    condition instead of falling into `EXIT_ENGINE`.
    """
    for klass in type(exc).__mro__:
        if klass in ERROR_CODES:
            return ERROR_CODES[klass]
    return EXIT_ENGINE

# --------------------------------------------------------------------------- #
# engine classes, resolved lazily
# --------------------------------------------------------------------------- #

#: arch -> class name inside `separate.py`. The *names* are held here because
#: importing `separate` at module scope would make `--list-models` import torch;
#: `engine_classes()` does the import on the only path that needs it.
#:
#: `mdx_c` maps to a real class but is unreachable from the command line: the
#: catalog blocks that arch, so `--arch mdx_c` refuses before this table is read
#: (that refusal is what `tests/test_separate_cli.py` test 2 proves by absence of
#: written output). `apollo` maps to nothing on purpose -- its `process_method` is
#: None, so `plan.build()` raises before a class is needed.
ARCH_CLASS: dict[str, str] = {
    "vr": "SeperateVR",
    "mdx": "SeperateMDX",
    "mdx_c": "SeperateMDXC",
    "demucs": "SeperateDemucs",
}


def engine_classes() -> dict[str, type]:
    """Resolve `ARCH_CLASS` against the real engine module (imports torch)."""
    import separate
    return {arch: getattr(separate, cls_name) for arch, cls_name in ARCH_CLASS.items()}


# --------------------------------------------------------------------------- #
# process_data -- the engine's second positional argument
# --------------------------------------------------------------------------- #

#: The 11 keys `SeperateAttributes.__init__` indexes unconditionally
#: (`separate.py:138-152`). Pinned twice in `tests/test_separate_cli.py`: against
#: what this function returns and against an AST re-derivation of the engine's own
#: subscripts. There is no `progress` or `audio_path` parameter on the engine and
#: no `job.start()` -- progress and paths travel here, and the entry point is
#: `job.seperate()`.
PROCESS_DATA_KEYS = frozenset({
    "audio_file", "audio_file_base", "cached_model_source_holder",
    "cached_source_callback", "export_path", "is_4_stem_ensemble",
    "is_ensemble_master", "list_all_models", "process_iteration",
    "set_progress_bar", "write_to_console",
})


def build_process_data(audio_file: Path | str, export_path: Path | str, *,
                       printer: Callable[[str], None] = print,
                       console_log: list[str] | None = None) -> dict:
    """The per-run inputs the engine reads out of `process_data`.

    Plain callables, not mocks, and the shape upstream's GUI passes
    (`UVR.py:7226-7229`): `set_progress_bar(step, inference_iterations=0)` and
    `write_to_console(progress_text, base_text="")` keep those signatures. Here the
    progress bar is a text line and the console is stdout, so a run reports without
    a widget. The two source-cache callbacks report "nothing cached", which is the
    honest answer for a one-shot process -- no branch may short-circuit on a stale
    tensor from an earlier arch.
    """
    export = Path(export_path)
    export.mkdir(parents=True, exist_ok=True)
    audio = Path(audio_file)
    log = console_log if console_log is not None else []
    shown: set[object] = set()

    def set_progress_bar(step, inference_iterations=0):
        # The engine calls this per block, and the iteration counter moves on every
        # one of them, so the *step* is what an announcement is keyed on. Without
        # this a 30-second crop printed 160 identical-looking lines and buried the
        # console text the engine writes through `write_to_console`.
        if step not in shown:
            shown.add(step)
            printer(f"[cli] progress {step} (inference_iterations={inference_iterations})")
        return None

    def write_to_console(progress_text, base_text=""):
        line = f"{base_text}{progress_text}".strip()
        if line:
            log.append(line)
            printer(f"[cli] {line}")
        return None

    def cached_source_callback(process_method, model_name=None):
        return None, None

    def cached_model_source_holder(process_method, sources, model_name=None):
        return None

    def process_iteration():
        return None

    return {
        "audio_file": str(audio),
        # `<index>_<stem>` upstream (`UVR.py:7229`); a single-file CLI run has no
        # index. The engine appends `_(<Stem>).wav` itself.
        "audio_file_base": os.path.splitext(os.path.basename(str(audio)))[0],
        "export_path": str(export),
        "set_progress_bar": set_progress_bar,
        "write_to_console": write_to_console,
        "cached_source_callback": cached_source_callback,
        "cached_model_source_holder": cached_model_source_holder,
        "is_ensemble_master": False,
        "is_4_stem_ensemble": False,
        "list_all_models": [],
        "process_iteration": process_iteration,
    }


# --------------------------------------------------------------------------- #
# composition root: the two settings-derived paths the catalog may not derive
# --------------------------------------------------------------------------- #

def writable_model_data_dir() -> Path:
    """The tier-1/tier-2 write target: `%APPDATA%\\UVR-fork\\model_data`.

    The same location `model_refresh.build_catalog()` uses, for the same reason:
    the weights root is read-only input (`F:/…/models` and the git-tracked repo
    `models/` tree both), so new per-model JSON goes here and nowhere else.
    """
    return settings_module.config_dir() / "model_data"


def make_catalog(model_root: Path | str | None = None,
                 writable_dir: Path | str | None = None) -> catalog.Catalog:
    """One `Catalog` over the active weights root and the writable catalog dir."""
    return catalog.Catalog(
        Path(model_root) if model_root is not None else uvr_paths.MODEL_ROOT,
        Path(writable_dir) if writable_dir is not None else writable_model_data_dir(),
    )


# --------------------------------------------------------------------------- #
# settings / device adaptation
# --------------------------------------------------------------------------- #

def effective_settings(st: settings_module.Settings, spec: catalog.ModelSpec,
                       device: str | None) -> settings_module.Settings:
    """The `Settings` a CLI run actually uses, from the saved one plus the flags.

    Two overrides, both of them the job the GUI used to do at run time:

    * `chosen_process_method` follows the *resolved* model, because `plan._build`
      branches its Demucs stem-only handling on it (`plan.py:447`) and a CLI run of
      `--arch demucs` whose saved settings still say VR would take the wrong stem
      flags. It is not a user-facing knob: the model decides it.
    * `gpu_conversion` follows `--device` when one was given (`UVR.py:384`'s input,
      read by `plan.py:329`), so `--device cpu` cannot be silently overridden by a
      settings file that says otherwise. Device honesty is the point: the flag a
      user typed is the flag the engine sees.
    """
    updates: dict[str, object] = {}
    if spec.process_method and st.chosen_process_method != spec.process_method:
        updates["chosen_process_method"] = spec.process_method
    if device is not None:
        updates["gpu_conversion"] = device != "cpu"
    return dataclasses.replace(st, **updates) if updates else st


def describe_device(backend: backends.Backend, requested: str | None) -> list[str]:
    """The device report a CLI run prints, GPU fallback included.

    `backends.resolve()` is the reporter (Task 6): it returns the `Backend` the
    engine will be handed. This function does not re-derive any of that -- it prints
    the object and then checks the one thing a forced name cannot tell you: whether
    the requested CUDA-family device is actually reachable *in this interpreter*.
    A ROCm wheel that cannot open the device computes on CPU, and saying nothing
    about it is the silent-fallback failure this fork has been burned by, so the
    line below is emitted whenever the requested family and the live device disagree.
    """
    lines = [f"[cli] backend: name={backend.name} torch_device={backend.torch_device} "
             f"ort_providers={backend.ort_providers}"]
    if requested is not None:
        lines[0] += f" (requested {requested!r})"
    if backend.torch_device != "cpu":
        try:
            import torch
        except ImportError as exc:                       # torch missing is not silent
            lines.append(f"[cli] WARNING: torch unavailable ({exc}); "
                         f"the run cannot use {backend.torch_device} and will fail "
                         f"or compute on CPU.")
            return lines
        available = bool(torch.cuda.is_available())
        hip = getattr(torch.version, "hip", None)
        lines.append(f"[cli] torch {torch.__version__} hip={hip} "
                     f"cuda_available={available}"
                     + (f" device={torch.cuda.get_device_name(0)}" if available else ""))
        if not available:
            lines.append(f"[cli] WARNING: --device {requested} was asked for, "
                         f"torch.cuda.is_available() is False and this build's hip "
                         f"version is {hip!r}, so the engine will compute on CPU. "
                         f"Nothing about that is hidden; the output is still written, "
                         f"just not by the GPU.")
    return lines


#: `gui_data.constants.DEFAULT`, inlined because this module may not import
#: `gui_data` (see the module docstring's no-GUI claim, executed by
#: `tests/test_engine_purity.py`). It is the one `device_set` value that means "no
#: index suffix", exactly as `separate.py:272` treats it.
DEVICE_SET_DEFAULT = "Default"


def describe_compute_device(is_gpu_conversion: int, backend: backends.Backend,
                            requested: str | None = None,
                            device_set: str | None = None) -> list[str]:
    """What the engine will *actually* compute on, derived from the plan.

    Why this line exists next to the `backend:` line rather than replacing it: they
    answer two different questions, and a run has to satisfy both. `backends.resolve()`
    answers "which device does this machine offer?", and `separate.py:233-241` answers
    "which device will this run use?" -- by *ignoring* the authority whenever
    `plan.is_gpu_conversion < 0` (`separate.py:253` never enters the GPU block, so
    `self.device` stays CPU and `self.run_type` stays `['CPUExecutionProvider']`).
    `is_gpu_conversion` comes from `Settings.gpu_conversion`, which is the user's
    persisted config, so an un-flagged CLI run on a box whose saved setting is "GPU
    off" prints `backend: name=rocm` and computes on CPU. That is the defect this
    function kills: it is the second line, and it states the engine's own decision.

    The rule for the pair is that they may never disagree about what runs, so the
    branch conditions are lifted verbatim from `separate.py`:
    `mps` when torch reports one (`:254`), `<torch_device>[:device_set]` when the
    backend asks for a GPU device *and* torch can open one (`:270-272`), and CPU
    otherwise -- including "the plan says GPU but torch has no device" and "the
    saved setting says GPU off". Nothing here overrides `Settings`; the only way to
    turn the GPU on from the command line is `--device`, which
    `effective_settings()` already maps onto `gpu_conversion`.
    """
    import torch  # lazy: the no-run commands never reach this line

    hip = getattr(torch.version, "hip", None)

    if is_gpu_conversion < 0:
        why = (f"--device {requested!r} asked for it" if requested == "cpu"
               else "pass --device rocm to use the GPU")
        return [f"[cli] compute device: cpu (is_gpu_conversion={is_gpu_conversion} "
                f"from Settings; {why})"]

    is_macos = sys.platform == "darwin"
    mps = bool(torch.backends.mps.is_available()) if is_macos else False
    if mps:
        return [f"[cli] compute device: mps (is_gpu_conversion={is_gpu_conversion}; "
                "separate.py takes the Apple path and sets is_other_gpu)"]

    if backend.torch_device == "cpu":
        return [f"[cli] compute device: cpu (is_gpu_conversion={is_gpu_conversion} "
                f"asks for the GPU, but backends.resolve() returned "
                f"torch_device='cpu' (hip={hip!r}), so separate.py keeps its CPU "
                f"device)"]

    if not torch.cuda.is_available():
        return [f"[cli] compute device: cpu (is_gpu_conversion={is_gpu_conversion} "
                f"asks for the GPU and the backend is {backend.name!r}, but "
                f"torch.cuda.is_available() is False (hip={hip!r}), so separate.py "
                f"keeps its CPU device -- see the WARNING above]"]

    device = ("cuda" if device_set in (None, "", DEVICE_SET_DEFAULT)
              else f"cuda:{device_set}")
    return [f"[cli] compute device: {device} (torch.version.hip={hip})"]


# --------------------------------------------------------------------------- #
# the commands
# --------------------------------------------------------------------------- #

def print_models(cat: catalog.Catalog, arch: str, printer: Callable[[str], None]) -> int:
    """The three views the catalog exists to provide, for one architecture."""
    installed = cat.names(arch)
    known = cat.known_names(arch)
    uninstalled = cat.uninstalled_names(arch)
    printer(f"[cli] arch {arch} -- model root {cat.model_root}, "
            f"writable catalog dir {cat.writable_dir}")
    printer(f"[cli]   names() ({len(installed)}): installed right now, in some root")
    for name in installed:
        printer(f"[cli]     {name}")
    printer(f"[cli]   known_names() ({len(known)}): declared by the baked tables")
    for name in known:
        printer(f"[cli]     {name}")
    printer(f"[cli]   uninstalled_names() ({len(uninstalled)}): declared but no weight here")
    for name in uninstalled:
        printer(f"[cli]     {name}")
    return EXIT_OK


def print_plan(rp: plan.RunPlan, printer: Callable[[str], None]) -> None:
    """Every field of the `RunPlan` the engine would be handed, one per line."""
    printer(f"[cli] plan: {type(rp).__name__} with "
            f"{len(dataclasses.fields(rp))} fields (nothing was run)")
    for field in dataclasses.fields(rp):
        printer(f"[cli]   {field.name} = {getattr(rp, field.name)!r}")


def do_refresh(args: argparse.Namespace, *, cat: catalog.Catalog,
               printer: Callable[[str], None]) -> int:
    """Delegate `--refresh-model-tables` to `model_refresh`, confirmation and all.

    The confirmation rule, the four URLs and every target path belong to
    `model_refresh`/`Catalog.write_table`; re-deriving any of them here would be a
    second copy of a policy that only has to be stated once. Importing the module
    lazily keeps this file's own import graph network-free (test 6).
    """
    import model_refresh
    try:
        written = model_refresh.refresh(cat, confirm=args.confirm, printer=printer)
    except model_refresh.RefreshError as exc:
        printer(f"[cli] refresh refused: {exc}")
        return EXIT_REFRESH_REFUSED
    except (catalog.CatalogError, OSError) as exc:
        printer(f"[cli] refresh failed: {type(exc).__name__}: {exc}")
        return EXIT_REFRESH_REFUSED
    for path in written:
        printer(f"[cli] refreshed {path}")
    return EXIT_OK


def run(args: argparse.Namespace, *, out: TextIO = sys.stdout,
        err: TextIO = sys.stderr) -> int:
    """One CLI invocation -> the build order the brief pins, and its exit code."""
    def printer(message: str) -> None:
        print(message, file=out)

    def failure(message: str, code: int) -> int:
        print(message, file=err)
        return code

    try:
        cat = make_catalog(args.model_root, args.writable_dir)
    except catalog.CatalogError as exc:
        return failure(f"[cli] {exc}", code_for(exc))

    if args.list_models is not None:
        try:
            return print_models(cat, args.list_models, printer)
        except catalog.CatalogError as exc:
            return failure(f"[cli] {exc}", code_for(exc))

    if args.refresh_model_tables:
        return do_refresh(args, cat=cat, printer=printer)

    if not args.model:
        return failure("[cli] a run needs a model: --model NAME (or a positional one), "
                       "or use --list-models / --refresh-model-tables.", EXIT_USAGE)

    st = settings_module.load(Path(args.settings) if args.settings else None)

    from app import session                                  # the one assembly path

    try:
        sess = session.assemble(st=st, cat=cat, model=args.model, arch=args.arch,
                                force_device=args.device, printer=printer)
    except catalog.UnknownModelParamsError as exc:
        # Print the catalog's own message text verbatim: the hash and the
        # `<hash>.json` path *are* the payload, and paraphrasing loses them. The
        # extra lines only say what a CLI user can *do*, which the catalog cannot
        # know (it has no opinion about who is reading).
        return failure(
            f"[cli] {exc}\n"
            f"[cli] A GUI-free run cannot open the model-params dialog, so this is a "
            f"hard error. Write that JSON yourself -- "
            f"Catalog.write_params({exc.spec.arch!r}, {exc.spec.hash!r}, params) puts "
            f"it under {cat.writable_dir} and the catalog reads it back before any "
            f"baked table.", EXIT_TIER3)
    except catalog.CatalogError as exc:
        return failure(f"[cli] {exc}", code_for(exc))
    except ValueError as exc:
        return failure(f"[cli] {exc}", EXIT_USAGE)
    except plan.PlanError as exc:
        return failure(f"[cli] plan refused: {exc}", EXIT_PLAN)

    for line in sess.compute_lines:
        printer(line)

    if args.dry_run:
        print_plan(sess.rp, printer)
        printer(f"[cli] --dry-run: model {sess.spec.name} (arch {sess.spec.arch}, "
                f"tier {sess.spec.provenance.label if sess.spec.provenance else '?'}) resolved; "
                f"nothing was run and nothing was written.")
        return EXIT_OK

    if not args.audio:
        return failure("[cli] --audio PATH is required for a run "
                       "(or use --dry-run to stop after the plan).", EXIT_USAGE)
    audio = Path(args.audio)
    if not audio.is_file():
        return failure(f"[cli] input audio not found: {audio}", EXIT_INPUT)

    export_path = Path(args.out) if args.out else (audio.parent / "UVR_out")
    try:
        written = session.run_job(sess, audio=audio, export_path=export_path, printer=printer)
    except session.NoEngineClass as exc:
        # The branch that used to sit here printed this text at EXIT_PLAN with no
        # "plan refused:" prefix. Ruling R-T2-noclass: keep those bytes, pin them with a
        # test -- the old code had none.
        return failure(f"[cli] {exc}", EXIT_PLAN)
    except Exception as exc:                              # engine failures are reported
        return failure(f"[cli] engine failed ({type(exc).__name__}): {exc}", EXIT_ENGINE)
    if not written:
        return failure(f"[cli] the engine returned without writing any stem to "
                       f"{export_path}; that is an engine failure, not a success.",
                       EXIT_ENGINE)
    for path in written:
        printer(f"[cli] wrote {path} ({path.stat().st_size} bytes)")
    printer(f"[cli] done: {len(written)} file(s) in {export_path}")
    return EXIT_OK


# --------------------------------------------------------------------------- #
# argparse front end
# --------------------------------------------------------------------------- #

#: `--arch` accepts the blocked names too. Refusing them with an argparse `choices`
#: list would print argparse's own usage text and lose the catalog's pinned reason,
#: which is the payload (`catalog.BLOCKED_ARCHS`). Let the catalog answer.
ARCH_CHOICES = tuple(sorted(catalog.ARCHS)) + tuple(sorted(catalog.BLOCKED_ARCHS))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="separate_cli",
        description="Separate audio with UVR's engine and no GUI at all: no tkinter "
                    "import, no window, no auto-update, no network. Model metadata "
                    "comes from the catalog (per-model JSON -> baked tables -> a hard "
                    "error naming the JSON it wanted).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python separate_cli.py --list-models vr\n"
            "  python separate_cli.py --audio song.flac --arch vr --model UVR-DeEcho-DeReverb\n"
            "  python separate_cli.py --audio song.flac --model UVR-MDX-NET-Inst_HQ_3 --device rocm\n"
            "  python separate_cli.py --audio song.flac --model 9s1 -n --out out       (dry run)\n"
            "  python separate_cli.py --refresh-model-tables --confirm  (the only\n"
            "      command in this fork that contacts the network; without --confirm\n"
            "      it refuses and exits non-zero)\n"
            "\n"
            "Scope: this CLI runs the single-model path only -- plan.build() is called\n"
            "    with selections={}, so a secondary model, a pre-proc model and the\n"
            "    vocal splitter are unreachable from here (they need the legacy GUI's\n"
            "    model-selection widgets; Plan 2's app is where they return).\n"
            "    --device picks the compute device; without it the saved\n"
            "    Settings.gpu_conversion decides, and the 'compute device:' line prints\n"
            "    whichever device the engine will actually use.\n"
            "\n"
            "Exit codes: 0 ok, 1 engine failure, 2 usage, 3 unknown model params "
            "(tier 3), 4 blocked arch, 5 missing weight, 6 unknown name, 7 plan "
            "refused, 8 refresh refused, 9 input audio missing, 10 writable catalog "
            "location refused.\n"),
    )
    parser.add_argument("model_positional", nargs="?", default=None, metavar="MODEL",
                        help="model name, as an alternative to --model")
    parser.add_argument("-a", "--audio", default=None, metavar="PATH",
                        help="input audio file (required for a run)")
    parser.add_argument("-o", "--out", default=None, metavar="DIR",
                        help="output directory; the engine writes "
                             "<base>_(<Stem>).wav there (default: UVR_out beside the input)")
    parser.add_argument("-A", "--arch", default=None, choices=ARCH_CHOICES,
                        help="restrict resolution to one architecture (default: whichever "
                             "root declares the name)")
    parser.add_argument("-m", "--model", default=None, metavar="NAME",
                        help="model name or weight file name to resolve through the catalog")
    parser.add_argument("-d", "--device", default=None, choices=("cpu", "rocm", "cuda"),
                        help="device to ask for, through backends.resolve(); a GPU that "
                             "cannot be opened is reported, never silently downgraded")
    parser.add_argument("-s", "--settings", default=None, metavar="PATH",
                        help="settings JSON to load (default: the fork's config dir, "
                             "then built-in defaults)")
    parser.add_argument("--model-root", default=None, metavar="DIR",
                        help="weights root for reads (default: UVR_MODEL_ROOT)")
    parser.add_argument("--writable-dir", default=None, metavar="DIR",
                        help="catalog dir for tier-1/tier-2 writes "
                             "(default: %%APPDATA%%/UVR-fork/model_data)")
    parser.add_argument("-n", "--dry-run", action="store_true",
                        help="resolve the model, build the plan, print all its fields, "
                             "run nothing")
    parser.add_argument("-L", "--list-models", nargs="?", const="vr", default=None,
                        metavar="ARCH", choices=ARCH_CHOICES,
                        help="print names() / known_names() / uninstalled_names() for one "
                             "arch (default vr) and exit")
    parser.add_argument("-R", "--refresh-model-tables", action="store_true",
                        help="refresh the four baked model-metadata tables by delegating "
                             "to model_refresh; needs --confirm")
    parser.add_argument("-C", "--confirm", "--confirm-refresh", dest="confirm",
                        action="store_true",
                        help="the explicit confirmation model_refresh demands")
    return parser


def main(argv: list[str] | None = None, *, out: TextIO = sys.stdout,
         err: TextIO = sys.stderr) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.model is None and args.model_positional is not None:
        args.model = args.model_positional
    return run(args, out=out, err=err)


if __name__ == "__main__":
    sys.exit(main())
