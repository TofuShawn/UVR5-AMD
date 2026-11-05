"""The one place a model name becomes a runnable job.

Both front ends go through here. The CLI used to own this sequence inline
(`separate_cli.py:435-497`), and the reason that matters is not tidiness: the
branch-end review of Plan 1 found a Critical that reached a green 459-test suite
because every test passed an explicit `--device cpu` and nothing exercised the
un-flagged default. Two front ends resolving a device differently is the same bug
with a wider blast radius, so there is exactly one implementation of "settings ->
catalog -> backend -> plan" and it lives here.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Callable

import backends
import catalog
import plan
import separate_cli
import settings as settings_module


class Cancelled(RuntimeError):
    """Raised out of the engine's per-batch progress hook when the user asks to stop.

    The engine has no stop flag of its own (`is_process_stopped` appears zero times
    in `separate.py`; upstream's STOP was `thread.terminate()`, `UVR.py:6297`). The
    fine-grained hook that does exist is `set_progress_bar`, called once per batch
    inside the VR loop (`separate.py:1383`, loop head `:1370`), and `separate.py`
    swallows nothing there (its only `except Exception` blocks are at `:1564`/`:1570`).
    """


@dataclasses.dataclass(frozen=True)
class NoEngineClass(RuntimeError):
    """The catalog knows an arch that `separate.py` has no class for.

    `ARCH_CLASS` and the catalog's arch table are two hand-maintained lists that can
    drift, and the refusal must name both. Raised by `run_job()` on the run path only
    -- never on `--dry-run`, which is what keeps the plan inspectable without pulling
    the engine's import graph in (see the module docstring of
    `separate_cli.py:22-27` and `tests/test_session.py::test_dry_run_never_imports_the_engine`).
    """

    arch: str
    known: tuple[str, ...]

    def __str__(self) -> str:
        return (f"arch {self.arch!r} has no engine class in separate.py; "
                f"known: {', '.join(sorted(self.known))}")


@dataclasses.dataclass(frozen=True)
class Session:
    spec: catalog.ModelSpec
    backend: backends.Backend
    rp: plan.RunPlan
    compute_lines: tuple[str, ...]
    engine_class: type | None = None       # override for tests; else run_job resolves it


def assemble(*, st: settings_module.Settings, cat: catalog.Catalog, model: str,
             arch: str | None = None, force_device: str | None = None,
             selections: dict | None = None,
             printer: Callable[[str], None] = print) -> Session:
    """settings -> catalog -> backend -> plan, printing the device line where the CLI prints it.

    Raises `catalog.UnknownModelParamsError` (tier 3 -- a GUI opens the params dialog,
    a CLI prints and exits 3), `catalog.CatalogError`, `ValueError` (bad device
    request) and `plan.PlanError` without catching them: the caller's error surface is
    the whole point of keeping these typed.
    """
    spec = cat.resolve(model, arch, required=True)
    backend = backends.resolve(force=force_device)
    for line in separate_cli.describe_device(backend, force_device):
        printer(line)
    rp = plan.build(settings=separate_cli.effective_settings(st, spec, force_device),
                    catalog=cat, backend=backend, spec=spec,
                    selections=dict(selections or {}))
    compute_lines = tuple(separate_cli.describe_compute_device(
        rp.is_gpu_conversion, backend, force_device, rp.device_set))
    # No engine-class lookup here: `engine_classes()` imports `separate`, which imports
    # torch -- the slowest import in the fork. `--dry-run` must stay answerable without
    # it (separate_cli.py:22-27, ruling R-T2-lazy), so `run_job()` resolves it instead.
    return Session(spec=spec, backend=backend, rp=rp, compute_lines=compute_lines)


def model_count(rp: plan.RunPlan) -> int:
    """Upstream's `true_model_count` (`UVR.py:6601-6603`), for the models a plan carries.

    Faithful in the terms `RunPlan` can answer, and explicit about the two it cannot:

    * secondary: `2 if m.is_secondary_model_activated else 1` (`:6603`) -- ensemble mode
      is out of Plan 2's scope, so the sum is over the single primary entry;
    * 4-stem: upstream adds `m.demucs_4_stem_added_count` (`:6601`), a *ModelData* field
      that excludes self-naming slots (`plan.py:649-650` explains why they stay in the
      list). `RunPlan` has no such field, so this counts `secondary_model_4_stem`;
    * pre-proc: upstream adds **2** per activated pre-proc model (`:6602`), because a
      pre-proc pass is itself a modelled stage -- an earlier draft of this file had 1,
      which under-counted;
    * vocal splitter: upstream's `determine_voc_split` (`:6562-6570`) adds 1 only when a
      stems-only selection is active *and* the splitter appears in the stem list. Those
      are UI-state gates `RunPlan` does not carry, so this counts presence.

    Net effect: on a demucs "All Stems" run with a self-naming 4-stem slot the
    denominator can be 1-2 higher than upstream's, which stretches the progress bar's
    tail. It cannot make a run wrong, and it is not presented as an exact port.
    """
    base = 2 if rp.is_secondary_model_activated else 1
    four_stem = len(rp.secondary_model_4_stem or ())
    pre_proc = 2 if rp.pre_proc_model else 0
    splitter = 1 if rp.vocal_split_model else 0
    return base + four_stem + pre_proc + splitter


def progress_percent(*, models_started: int, model_count: int, step: float,
                     inference_iterations: float = 0.0, total_files: int = 1) -> float:
    """Upstream's percentage, ported term by term -- not re-derived.

    `UVR.py:6271-6274`: `base = 100 / (true_model_count * total_files)`,
    `progress = base*iteration - base + base*step`. The `step` it receives is already
    the sum: `UVR.py:6641` wraps the engine callback as
    `lambda step, inference_iterations=0: process_update_progress(total_files=total_files,
    step=(step + inference_iterations))`, so the engine's second positional argument is
    an *offset to step*, not the model counter.

    `models_started` is the port's name for upstream's `self.iteration` (`UVR.py:1308`,
    `:6273`), renamed from the old `completed_models` because it is the number of models
    that have *started*, not finished -- which is exactly what upstream counts.
    `UVR.py:6632-6633` advances it at the head of each model's loop
    (`for current_model_num, current_model in enumerate(model, start=1): self.iteration += 1`),
    so during model *k* it equals *k*: a single-model plan sits at `1` for the whole run,
    and `base*1 - base + base*step = base*step` fills 0 -> 100 across that one model. For
    two models the primary is `base*1 - base + base*step = base*step` (0 -> 50) and the
    secondary `base*2 - base + base*step` (50 -> 100) -- upstream's shape, not a smoothing
    trick. `run_job` seeds the counter at 1 for that reason and `process_iteration` is the
    bump for every model after the primary: `separate.py:1455-1456` and `:1481-1482` call
    it at the head of `process_secondary_model`/`process_chain_model` (guarded by
    `if not is_pre_proc_model`), and `:1082`/`:1085` in a Demucs pre-proc pass -- never for
    the primary VR pass. With the counter at 0 the primary would report `base*0 - base`,
    negative and clamped to 0 for the whole run, which is what the clamp is here to catch,
    not to present as progress.

    `UVR.py:6617` also adds `self.true_model_count` when a file fails verification
    (`self.iteration += self.true_model_count`); that is multi-file accounting and is out
    of scope while `total_files == 1`.
    """
    base = 100.0 / max(1, model_count * total_files)
    progress = base * models_started - base + base * (step + inference_iterations)
    return max(0.0, min(100.0, progress))


def run_job(session: Session, *, audio: Path | str, export_path: Path | str,
            printer: Callable[[str], None] = print,
            console_log: list[str] | None = None,
            progress: Callable[[float], None] | None = None,
            cancel: Callable[[], bool] | None = None) -> list[Path]:
    """Run the engine for one file and return the stems it wrote.

    `progress` receives a 0-100 percentage; `cancel` is polled at every engine
    progress hook and a true answer aborts with `Cancelled`. The written-file check is
    a pre-run snapshot of (path -> mtime) rather than a post-run existence test,
    because re-running the same model overwrites the same two filenames and an engine
    that produced nothing must not exit 0 (`separate_cli.py:521-541`).
    """
    audio, export_path = Path(audio), Path(export_path)
    engine_class = session.engine_class
    if engine_class is None:
        # Resolved here and not in assemble(): this is the only path that separates, so
        # this is the only path that may import the engine (ruling R-T2-lazy). Pinned by
        # tests/test_session.py::test_dry_run_never_imports_the_engine.
        classes = separate_cli.engine_classes()
        engine_class = classes.get(session.spec.arch)
        if engine_class is None:
            raise NoEngineClass(arch=session.spec.arch, known=tuple(classes))
    process_data = separate_cli.build_process_data(audio, export_path,
                                                   printer=printer, console_log=console_log)
    if progress is not None or cancel is not None:
        raw_progress = process_data["set_progress_bar"]
        raw_iteration = process_data["process_iteration"]
        count = model_count(session.rp)
        # Seed at 1, not 0: `run_job` runs one file and `assemble` builds one primary
        # model, so the primary model has *started* the instant the engine begins -- which
        # is exactly what upstream's `self.iteration` holds at `UVR.py:6633` before the
        # first batch. The secondary/chain bump that `separate.py` fires below then walks
        # 2..N. An earlier draft seeded 0 and waited for a `process_iteration` hook the
        # primary pass never fires, so the whole single-model run clamped to 0%.
        state = {"iteration": 1}

        def set_progress_bar(step, inference_iterations=0):
            if cancel is not None and cancel():
                raise Cancelled("stopped by the user")
            raw_progress(step, inference_iterations=inference_iterations)
            if progress is not None:
                progress(progress_percent(models_started=state["iteration"],
                                          model_count=count, step=step,
                                          inference_iterations=inference_iterations))

        def process_iteration():
            if cancel is not None and cancel():
                raise Cancelled("stopped by the user")
            state["iteration"] += 1
            raw_iteration()

        process_data["set_progress_bar"] = set_progress_bar
        process_data["process_iteration"] = process_iteration

    export_path.mkdir(parents=True, exist_ok=True)
    before = {p: p.stat().st_mtime_ns for p in export_path.rglob("*.wav") if "_(" in p.stem}
    job = engine_class(session.rp, process_data)
    job.seperate()
    return sorted(p for p in export_path.rglob("*.wav")
                  if "_(" in p.stem and before.get(p) != p.stat().st_mtime_ns)
