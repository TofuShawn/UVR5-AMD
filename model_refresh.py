"""Task 12 — the model-table refresh, kept on purpose and moved out of the dead GUI.

What upstream did, and what this fork does instead
--------------------------------------------------
The official build fetched four model-metadata JSON tables on **every start-up**
(`UVR.py`'s old `download_model_settings`, reached from the start-up gate inside
`online_data_refresh`) and wrote them straight into `MODELS_DIR`. Under this fork's
default root that is the git-tracked repo `models/` tree, so a fresh clone both phoned
home unattended and dirtied its own worktree. That path is also the only thing that
kept the shipped tables current — measured, the repo copies are strictly staler than
installed ones (MDX params 61 vs 86 entries, name mapper 20 vs 46, the missing 26 being
exactly the Roformer/MDX23C class this fork targets). So killing the *application's*
self-update must not strand the catalog: per ruling R-T10-refresh the table fetch
survives as an **explicit, user-invoked command**, writing outside any version-controlled
path.

That is this module. It is imported only when someone asks for it (Task 13 wires the CLI
flag; `python model_refresh.py --confirm` works on its own), it never runs on import, and
it refuses to touch the network without an explicit confirmation.

The four URLs live here now — they were *moved*, not copied, and
`gui_data/constants.py` no longer defines them (that absence is what
`tests/test_no_self_update.py` asserts, alongside "no `UVR.py` loads them").

Writes go through `Catalog.write_table(arch, kind, table)` and nowhere else: that method
owns the path convention (`writable_dir / <the arch's declared table rel path>`) and the
table-cache invalidation, so this module never re-derives
`<writable_dir>/VR_Models/model_data/model_data.json` by hand. `Catalog.__post_init__`
already refuses a `writable_dir` that sits inside `model_root`, `fallback_root` or the
repo `models/` tree (ruling R-T10-writable), which is the structural reason a refresh
here cannot dirty a worktree.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import catalog
import settings
import uvr_paths

__all__ = ["TABLES", "TableSource", "RefreshError", "fetch_json", "refresh",
           "build_catalog", "target_paths", "announce", "main"]

@dataclass(frozen=True)
class TableSource:
    """One baked tier-2 table: where it comes from and where `Catalog` puts it.

    `arch`/`kind` are the coordinates `Catalog.write_table` and `Catalog.baked_table`
    already speak, so the mapping is checked against the arch's own `ArchSpec` rather
    than against a path string repeated here. `moved_from` records the
    `gui_data/constants.py` name the URL was moved out of; note that
    `MDX23_CONFIG_CHECKS`, which points at the same `application_data` repo, deliberately
    stayed there — the config fetch it builds is a user-requested model download, not
    telemetry.
    """

    url: str
    arch: str
    kind: str
    moved_from: str

    def label(self) -> str:
        return f"{self.arch}/{self.kind}"


#: The four tables, in the order upstream fetched them. The mapping is the brief's:
#: `VR_MODEL_DATA_LINK -> ("vr", "params")`, `MDX_MODEL_DATA_LINK -> ("mdx", "params")`,
#: `MDX_MODEL_NAME_DATA_LINK -> ("mdx", "names")`,
#: `DEMUCS_MODEL_NAME_DATA_LINK -> ("demucs", "names")`. The URLs are spelled out in full
#: so a grep for one of them finds exactly this file.
TABLES: tuple[TableSource, ...] = (
    TableSource("https://raw.githubusercontent.com/TRvlvr/application_data/main/vr_model_data/model_data_new.json",
                "vr", "params", "VR_MODEL_DATA_LINK"),
    TableSource("https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/model_data_new.json",
                "mdx", "params", "MDX_MODEL_DATA_LINK"),
    TableSource("https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/model_name_mapper.json",
                "mdx", "names", "MDX_MODEL_NAME_DATA_LINK"),
    TableSource("https://raw.githubusercontent.com/TRvlvr/application_data/main/demucs_model_data/model_name_mapper.json",
                "demucs", "names", "DEMUCS_MODEL_NAME_DATA_LINK"),
)


class RefreshError(RuntimeError):
    """A refresh that will not happen, printed rather than traced by the CLI."""


def _default_opener(url: str):
    """The only place in this module that calls `urlopen` — and it is reached only from
    an explicit refresh, never at import.

    `timeout=30` because this is a command a person waits on: without it a stalled
    connection hangs the CLI forever with nothing on screen to explain why.
    """
    return urllib.request.urlopen(url, timeout=30)


def fetch_json(url: str, *, opener: Callable[[str], object] | None = None) -> dict:
    """GET `url` and parse it as a JSON object.

    `utf-8-sig` for the same reason `catalog._load_json` uses it: the shipped tables
    carry a BOM, and a decode that ignores it would write one into the refreshed copy.
    """
    opener = opener or _default_opener
    with opener(url) as response:  # type: ignore[attr-defined]
        raw = response.read()
    try:
        table = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RefreshError(f"{url} is not valid JSON: {exc}") from exc
    if not isinstance(table, dict):
        raise RefreshError(
            f"{url} returned {type(table).__name__}, but a model table is a JSON object "
            f"keyed by model hash or name.")
    return table


def build_catalog(model_root: Path | str | None = None,
                  writable_dir: Path | str | None = None) -> catalog.Catalog:
    """The catalog a refresh writes into, with the fork's usual composition.

    `model_root` is `uvr_paths.MODEL_ROOT` (honours `UVR_MODEL_ROOT`) and the writable
    catalog dir is `settings.config_dir()/model_data` — `%APPDATA%\\UVR-fork\\model_data`
    on Windows. Passing them here, rather than letting `Catalog` reach for
    `settings.config_dir()` itself, is what keeps the one-way dependency the design
    diagram shows (ruling R-T10-writable): `catalog` stays settings-free and testable.
    """
    return catalog.Catalog(
        Path(model_root) if model_root is not None else uvr_paths.MODEL_ROOT,
        Path(writable_dir) if writable_dir is not None else settings.config_dir() / "model_data",
    )


def target_paths(cat: catalog.Catalog) -> list[Path]:
    """Where each table would land — computed by `Catalog`, not re-derived here."""
    out = []
    for source in TABLES:
        rel = cat._table_rel(cat._arch(source.arch), source.kind)
        if rel is None:
            raise RefreshError(f"catalog declares no {source.kind} table for {source.arch}")
        out.append(cat.writable_dir / rel)
    return out


def announce(cat: catalog.Catalog, *, printer: Callable[[str], None] = print) -> None:
    """Say plainly what a refresh is about to do *before* doing it."""
    printer("[model_refresh] this contacts the network — four HTTPS GETs:")
    for source, target in zip(TABLES, target_paths(cat)):
        printer(f"[model_refresh]   GET {source.url}")
        printer(f"[model_refresh]     -> {source.label()} ({source.moved_from}) -> {target}")
    printer(f"[model_refresh] weights root {cat.model_root} is read-only input and is "
            f"never written; the repo models/ tree is not a target either.")


def refresh(cat: catalog.Catalog, *,
            confirm: bool = False,
            fetch: Callable[[str], dict] | None = None,
            opener: Callable[[str], object] | None = None,
            printer: Callable[[str], None] = print) -> list[Path]:
    """Fetch all four tables and write each through `Catalog.write_table`.

    Returns the four target paths. Refuses — without fetching anything — unless
    `confirm` is true: nothing in this fork refreshes by default any more.
    """
    if not confirm:
        raise RefreshError(
            "refusing to touch the network: a model-table refresh has to be asked for "
            "explicitly (pass confirm=True / --confirm).")

    if fetch is None:
        fetch = (lambda url: fetch_json(url, opener=opener)) if opener is not None else fetch_json

    written: list[Path] = []
    for source in TABLES:
        printer(f"[model_refresh] fetching {source.url}")
        table = fetch(source.url)
        target = cat.write_table(source.arch, source.kind, table)
        printer(f"[model_refresh] wrote {source.label()} — {len(table)} entries -> {target}")
        written.append(target)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="model_refresh",
        description="Refresh the four baked model-metadata tables from the UVR project's "
                    "application_data repo. There is no automatic update in this fork: "
                    "this runs only when you invoke it, and it writes to the "
                    "user-writable catalog directory, never to the weights root.")
    parser.add_argument("--confirm", action="store_true",
                        help="actually perform the four network fetches and writes; "
                             "without this flag nothing is requested and nothing is written")
    parser.add_argument("--model-root", default=None,
                        help="weights root used for read fallback (default: UVR_MODEL_ROOT)")
    parser.add_argument("--writable-dir", default=None,
                        help="catalog directory to write into (default: %APPDATA%/UVR-fork/model_data)")
    args = parser.parse_args(argv)

    cat = build_catalog(args.model_root, args.writable_dir)

    if not args.confirm:
        announce(cat, printer=print)
        print("[model_refresh] DRY RUN — nothing was fetched or written. "
              "Re-run with --confirm to touch the network.")
        return 1

    announce(cat, printer=print)
    try:
        refresh(cat, confirm=True)
    except (RefreshError, catalog.CatalogError, OSError) as exc:
        print(f"[model_refresh] {exc}", file=sys.stderr)
        return 2
    print("[model_refresh] done — Catalog.baked_table/known_names see the new content "
          "in this process immediately.")
    return 0


if __name__ == "__main__":  # pragma: no cover - the CLI entry point
    sys.exit(main())
