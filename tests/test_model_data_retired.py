"""`UVR.py` is retired: the model table, then the GUI, and a guard for both.

`UVR.py` was 7,342 lines of legacy tkinter GUI until Plan 2 Task 8 deleted all but its
389-line header, so this file deliberately checks *shapes in the AST* rather than
importing the module -- it cannot be imported on this machine anyway (`import pyglet`
still fails here, and the refusal at the top only fires when the file is *run*, not when
it is read). Task 12 parses it the same way.

Three properties, one per clause of the 11d ruling that started this file:

1. `ModelData` is gone -- not defined, and not *loaded* by name anywhere. The AST
   matters because the file still speaks the word in prose (the comment above the
   refusal), and a substring check would fail on text that is doing useful work.
   `ApolloModelData` shared that suffix, which was a second reason to compare
   `Name.id` rather than grep -- and which is why clause 2 below reads the way it does
   now.
2. `ApolloModelData` used to survive, and Task 8 deleted it. The property the 11d
   ruling wanted -- "Apollo still resolves" -- is now asserted where the resolution
   actually lives, in `catalog`, and `test_apollo_model_data_is_gone_and_catalog_owns_
   the_resolution` says so.
3. The legacy entry point refuses as the *first statement* of the `__main__` block.
   Not a `sys.exit()` at the bottom, not a warning: nothing above it may run, so
   that no window, no DPI call and no half-read config table precedes the refusal.

Task 8 then retired the rest of the tkinter layer -- `Ensembler`, `AudioTools`,
`ApolloModelData`, `ToolTip`, the batch/combo/console widgets, `MainWindow`, the
drag-and-drop callback, the `pop_up_*` dialogs and `legacy_main` -- and with them the
two properties above that were asserted as *presences inside that layer*
(`assemble_model_data`'s body, and `ApolloModelData` being the only Apollo resolution
path). Both are asserted as absences now, and the behaviour each was standing in for is
tested where it actually lives: Apollo resolves through `catalog.Catalog.resolve`
(`tests/test_catalog.py::test_apollo_resolves_via_tier1_only`), and the GUI-free
assembly path is `app/session.py` plus `plan.build`.

What keeps that from becoming a one-sided "it is gone" suite is
`test_tkinter_is_gone_and_the_engine_still_runs`: absence plus presence, in one
function, because a guard that only checks absence is satisfied by deleting the fork's
whole feature. The fork's feature is the app.

The residue gets the same treatment as the deletion. `UVR.py`'s header comment used to
justify its import block by claiming the set-up below the refusal reads it, which the
file refutes -- so the dead half is now measured and pinned here (`DEAD_IMPORT_BINDINGS`,
`READ_IMPORT_BINDINGS`, `CALLER_LESS_FUNCTIONS` and their two tests) instead of asserted
by prose nobody can check. Pinning it in both directions is the point: deleting an unread
import is welcome and only costs editing the list, and adding one to a file nothing runs
is refused.
"""
import ast
import dataclasses
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]
UVR_SRC = (REPO / "UVR.py").read_text(encoding="utf-8")
UVR_TREE = ast.parse(UVR_SRC)
REFUSAL = ("legacy tkinter GUI removed; use separate_cli.py "
           "(PySide6 app arrives in Plan 2)")

#: Every GUI name Task 8 deleted. Checked both as AST definitions (so prose may still
#: *say* the word, as the header comment does) and as source text for the two names
#: whose appearance is itself the problem -- see `test_tkinter_is_gone_and_the_engine_still_runs`.
GONE_CLASSES = ("Ensembler", "AudioTools", "ApolloModelData", "ToolTip",
                "ListboxBatchFrame", "ComboBoxEditableMenu", "ComboBoxMenu",
                "ThreadSafeConsole", "MainWindow")
GONE_FUNCTIONS = ("legacy_main", "drop", "read_bulliten_text_mac", "open_link",
                  "auto_hyperlink", "vip_downloads", "extract_stems",
                  "assemble_model_data")
#: The popup bodies `tests/test_dialogs.py::POPUP_EVIDENCE` was read from. The table is
#: the last copy of that evidence; these assertions are what tell a reader so.
GONE_POPUPS = ("pop_up_vr_param", "pop_up_mdx_model", "pop_up_mdx_c_param",
               "pop_up_apollo_param")

# --- The inert residue, measured instead of claimed -------------------------- #
# `UVR.py`'s header comment used to justify its import block by saying the set-up below
# the refusal reads it. Measured against the file's own AST that is false for 40 of 61
# names, and the named reason -- `pyglet`, which is what makes the module unimportable --
# is itself one of the unread. The fix for a comment that lies is a true comment plus a
# number nobody has to trust, so the split lives here as data with a guard.
#
# Scope of the measurement: 62 explicit import statements bind the 61 unique names below
# (`from collections import Counter` appears twice), and the two `from x import *` lines
# are excluded because an AST cannot enumerate what they bind. The comment that binds the
# file to these lists is in `UVR.py` above the import block; change one, change both.

#: Import names `UVR.py` binds and never references. Deleting one is an improvement and
#: this test will say so; adding one is the residue growing and this test refuses it.
DEAD_IMPORT_BINDINGS = frozenset({
    "Counter", "Fernet", "List", "ModelParameters", "PBKDF2HMAC", "Path",
    "SeperateDemucs", "SeperateMDX", "SeperateMDXC", "SeperateVR", "VERSION",
    "apollo_inference", "audioread", "base64", "clear_gpu_cache", "datetime",
    "error_dialouge", "error_text", "hashes", "hashlib", "librosa", "match", "math",
    "natsort", "onnx", "playsound", "pyglet_font", "pyperclip", "re", "save_format",
    "sf", "shutil", "spec_utils", "torch", "traceback", "urllib", "webbrowser", "wget",
    "windll", "wintypes",
})

#: The 21 it does reference -- `os`, `sys`, `subprocess` and the rest of the path/OS
#: set-up, the three `gui_data.old_data_check` helpers it calls at import time, the two
#: device probes and the three patch-id names from `__version__`. Seven of these
#: twenty-one (`json`, `time`, `queue`, `psutil`, `yaml`, `ConfigDict`, `KThread`) are
#: referenced only from inside the
#: caller-less functions below, so what the set-up itself earns is fourteen names.
READ_IMPORT_BINDINGS = frozenset({
    "ConfigDict", "KThread", "PATCH", "PATCH_LINUX", "PATCH_MAC", "cuda_available",
    "expanduser", "file_check", "json", "mps_available", "os", "pickle", "psutil",
    "queue", "remove_temps", "remove_unneeded_yamls", "ssl", "subprocess", "sys",
    "time", "yaml",
})

#: Header functions with no caller in any tracked file -- five of the seven that survive.
#: The cross-file half was measured with
#: `git grep -n "get_execution_time\|close_process\|load_model_hash_data\|font_checker\|get_yaml_data" -- "*.py"`
#: on 2026-11-06: outside `UVR.py` the only hits are this list and the prose naming it.
#: What is asserted below is the stronger in-file half -- not one `ast.Name` load, so not
#: even a sibling calling them. Task 8's report counted one caller-less function; five is
#: the measurement, and `UVR.py`'s header comment now says so.
CALLER_LESS_FUNCTIONS = ("get_execution_time", "close_process", "load_model_hash_data",
                         "font_checker", "get_yaml_data")

#: The two that ARE reached: `data = load_data()` is `UVR.py`'s last statement, and
#: `save_data` is called from `load_data`'s except branch.
REACHED_FUNCTIONS = ("save_data", "load_data")


def _import_split(tree):
    """(referenced, unreferenced) over the names the file's explicit imports bind."""
    bound = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bound |= {a.asname or a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            bound |= {a.asname or a.name for a in node.names if a.name != "*"}
    loads = {n.id for n in ast.walk(tree)
             if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    return bound & loads, bound - loads


def _top_level(name):
    return [n for n in UVR_TREE.body
            if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name]


def _any_defs(name):
    """Definitions at ANY depth -- the retired GUI methods lived inside `MainWindow`."""
    return [n for n in ast.walk(UVR_TREE)
            if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name == name]


def _name_loads(name):
    return [n for n in ast.walk(UVR_TREE)
            if isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Load)]


def test_model_data_is_no_longer_defined():
    assert not _top_level("ModelData"), (
        "ModelData is back as a top-level class. Task 11 replaced it with "
        "catalog.Catalog.resolve + plan.build; a second source of model metadata is "
        "the exact duplication the ratchet in tests/test_engine_purity.py exists to "
        "keep out.")


def test_no_name_load_of_model_data_survives_in_uvr_py():
    """AST, not substring -- prose may name it, code may not read it."""
    loads = _name_loads("ModelData")
    assert not loads, (
        f"UVR.py loads the name ModelData {len(loads)} time(s), at line(s) "
        f"{sorted(n.lineno for n in loads)}. Each one is a NameError on a path that "
        "can no longer build the object; delete the reference or point it at "
        "plan.RunPlan. Mentions inside docstrings and comments are fine -- this "
        "assertion is about ast.Name nodes.")


def test_apollo_model_data_is_gone_and_catalog_owns_the_resolution():
    """Task 11d kept this class because it sat outside the range *that* ruling deleted
    and it was then the only Apollo resolution path left in the GUI. Task 8 deleted the
    whole tkinter layer, so the property inverts: nothing may resolve a model from
    `UVR.py` any more.

    Not a weakening -- the behaviour moved where the assertion can still reach it.
    Apollo params resolve through `catalog.Catalog.resolve`, tier 1 only, and that is
    tested live at `tests/test_catalog.py::test_apollo_resolves_via_tier1_only` and
    `::test_apollo_has_no_baked_table_and_that_is_not_an_empty_dict`.
    """
    assert not _any_defs("ApolloModelData"), (
        f"ApolloModelData is defined again at line(s) "
        f"{[n.lineno for n in _any_defs('ApolloModelData')]}. Plan 1 put model metadata "
        f"in catalog.Catalog.resolve and Task 8 deleted the last GUI copy of that "
        f"resolution; a second source is what the ratchet in tests/test_engine_purity.py "
        f"exists to keep out.")


def test_assemble_model_data_is_gone_with_its_callers():
    """This was 11 `ModelData(...)` comprehensions, then a bare `raise NotImplementedError`
    with the assertion above pinned that shape because 11 call sites still needed the
    method to exist. Every one of those call sites was a `MainWindow` button, so with the
    GUI gone the method has no caller and the refusal has nothing to refuse.

    The live assembly path is `app/session.assemble()` -> `plan.build()`; a GUI-free
    caller is `separate_cli.run()`. Neither reaches this file.
    """
    assert not _any_defs("assemble_model_data"), (
        f"assemble_model_data is back at line(s) "
        f"{[n.lineno for n in _any_defs('assemble_model_data')]}; if it has a caller "
        f"again, the answer is plan.RunPlan -- see tests/test_session.py -- not a "
        f"legacy half-builder.")


def test_the_popups_the_table_was_read_from_are_gone():
    """`tests/test_dialogs.py::POPUP_EVIDENCE` is a frozen table, and a frozen table is
    only honest if the reader can tell it is frozen. This is that half: the four popup
    bodies the table was transcribed from are not in the file, so nothing here is
    quietly re-reading them.
    """
    for popup in GONE_POPUPS:
        assert not _any_defs(popup), (
            f"{popup} is defined at line(s) {[n.lineno for n in _any_defs(popup)]}. "
            f"POPUP_EVIDENCE in tests/test_dialogs.py is the frozen copy of what that "
            f"body collected, read from UVR.py at commit 720ca46; if the body is back, "
            f"re-derive the row from it in the same commit instead of leaving the two "
            f"to disagree.")


def test_no_class_and_no_popup_prefix_survives_at_all():
    """The sweep behind the two lists above, because a list is something an editor can
    trim. The deleted layer was nine classes and 16 `pop_up_*` bodies; what is left of
    this module defines *no* class at all and no `pop_up_*` function of any name, so
    tkinter coming back has to fail here before it can fail by deleting a name from
    `GONE_CLASSES`. (The seven functions that survive are header helpers, not GUI
    classes: `get_execution_time`, `close_process`, `save_data`, `load_data`,
    `load_model_hash_data`, `font_checker`, `get_yaml_data` -- and five of those seven
    have no caller anywhere, which is `CALLER_LESS_FUNCTIONS`'s business, not this
    test's. "Not a GUI class" is not the same claim as "used".)
    """
    classes = sorted({n.name for n in ast.walk(UVR_TREE) if isinstance(n, ast.ClassDef)})
    assert not classes, (
        f"UVR.py defines classes again: {classes}. The front end is `app/`; a class in "
        f"this file means a GUI layer is being rebuilt here instead of there.")
    popups = sorted({n.name for n in ast.walk(UVR_TREE)
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and n.name.startswith("pop_up_")})
    assert not popups, (
        f"`pop_up_*` dialogs are back: {popups} -- 16 of them went in Plan 2 Task 8, and "
        f"`tests/test_dialogs.py::POPUP_EVIDENCE` is the frozen record of what four of "
        f"them collected; a new popup needs the same treatment, not a second live read.")


def test_the_inert_header_import_block_is_still_the_measured_one():
    """`UVR.py` cannot be imported here and is imported by nothing, so its import block
    is a claim about dead text -- and the claim used to be false. Pinned as two sets
    rather than as a sentence: the header comment above that block quotes 40 unread of
    61, and this is the measurement behind the number.

    Asserted as EQUALITY, not as a ceiling, because both directions have to be noticed.
    A dead name disappearing means someone deleted an unread import, which is the right
    thing to do and only costs moving it out of `DEAD_IMPORT_BINDINGS` and re-saying the
    count in `UVR.py`'s comment. A dead name appearing means an unread import was added
    to a file nothing runs, and that deserves the argument.
    """
    read, dead = _import_split(UVR_TREE)
    assert dead == DEAD_IMPORT_BINDINGS, (
        f"`UVR.py`'s unreferenced import bindings no longer match the pinned inventory: "
        f"live={sorted(dead)} pinned={sorted(DEAD_IMPORT_BINDINGS)}. "
        f"Newly dead: {sorted(dead - DEAD_IMPORT_BINDINGS)}; cleaned up: "
        f"{sorted(DEAD_IMPORT_BINDINGS - dead)}. Either way the header comment above the "
        f"import block quotes these counts, so change the comment and this list in the "
        f"same commit -- the finding this guard answers was a comment that had stopped "
        f"matching the file.")
    assert read == READ_IMPORT_BINDINGS, (
        f"the imports `UVR.py` actually references have changed: "
        f"live={sorted(read)} pinned={sorted(READ_IMPORT_BINDINGS)}; "
        f"differences {sorted(read ^ READ_IMPORT_BINDINGS)}. Seven of the pinned set are "
        f"read only inside caller-less functions, and `UVR.py`'s comment says the "
        f"set-up itself reads fourteen -- re-derive that sentence with this list.")


def test_the_caller_less_header_functions_are_still_caller_less():
    """The other half of the same autopsy: five of the seven surviving functions have no
    caller at all. This asserts the in-file half, which is the half that can rot without
    anyone editing a comment -- a call added inside `UVR.py` itself.

    Deliberately not a machine check of "no other file calls them": `UVR.py` is not
    importable, so any cross-file scan here would be a text grep over trees this suite
    does not own. That half is measured, recorded and dated in the comment above
    `CALLER_LESS_FUNCTIONS`, and re-run with the command quoted there.
    """
    for name in CALLER_LESS_FUNCTIONS:
        assert _any_defs(name), (f"{name} is gone from UVR.py; it was recorded as inert "
                                 f"header code, not as missing -- if the residue is being "
                                 f"deleted, say so in UVR.py's header comment in the same "
                                 f"commit.")
        loads = _name_loads(name)
        assert not loads, (
            f"{name} is now called from inside UVR.py at line(s) "
            f"{sorted(n.lineno for n in loads)}. It leaves CALLER_LESS_FUNCTIONS, and "
            f"`UVR.py`'s header comment ('five have no caller') has to follow.")
    for name in REACHED_FUNCTIONS:
        assert _name_loads(name), (
            f"{name} has lost its only caller, so the caller-less count is six, not "
            f"five: move it from REACHED_FUNCTIONS into CALLER_LESS_FUNCTIONS and update "
            f"the paragraph at the foot of UVR.py's header comment with it.")


def test_tkinter_is_gone_and_the_engine_still_runs():
    """Paired assertions on purpose: a guard that only checks absence is satisfied by
    deleting the feature, and this fork's feature is the app. Both halves must hold."""
    source = (REPO / "UVR.py").read_text(encoding="utf-8")
    for gone in ("class MainWindow", "class Ensembler", "class AudioTools",
                 "class ThreadSafeConsole", "def legacy_main", "tkinterdnd2"):
        assert gone not in source, gone

    # The same list as definitions rather than substrings, so a prose mention in the
    # module header cannot satisfy -- or accidentally trip -- this half.
    for name in GONE_CLASSES + GONE_FUNCTIONS:
        assert not _any_defs(name), (
            f"{name} is defined in UVR.py again, at line(s) "
            f"{[n.lineno for n in _any_defs(name)]}")

    # The block set in tests/test_engine_purity.py names these three as GUI imports; if
    # the file reached for one, the purity claim for the engine would be a claim about a
    # front end that no longer exists.
    imported = []
    for node in ast.walk(UVR_TREE):
        if isinstance(node, ast.Import):
            imported += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    gui_reaches = sorted({name for name in imported
                          if name in ("tkinter", "PIL.ImageTk")
                          or name.startswith("tkinter.")
                          or name.endswith((".sv_ttk", ".tkinterdnd2", ".ImageTk"))})
    assert not gui_reaches, (
        f"UVR.py now imports GUI modules again: {gui_reaches}. The whole module list is "
        f"{sorted(set(imported))}; tests/test_engine_purity.py blocks these names for "
        f"the engine surface, and after Task 8 nothing in this file may need them.")

    import separate, backends, plan, catalog, settings, selections   # noqa: F401
    assert len(dataclasses.fields(settings.Settings)) == 50
    assert len(dataclasses.fields(selections.Selections)) == 33


def test_the_three_process_determine_methods_are_gone():
    """catalog.py owns those decisions (secondary_for/demucs_pre_proc_model_for/
    vocal_split_for); a tkinter-dependent copy in UVR.py keeps the `root.` dependency
    alive, which is what Task 8's ratchet counts."""
    for method in ("process_determine_secondary_model",
                   "process_determine_demucs_pre_proc_model",
                   "process_determine_vocal_split_model"):
        hits = [n for n in ast.walk(UVR_TREE)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == method]
        assert not hits, f"{method} is still defined at line(s) {[h.lineno for h in hits]}"


def test_main_block_refuses_first():
    def is_main(node):
        return (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__"
                and len(node.test.comparators) == 1
                and isinstance(node.test.comparators[0], ast.Constant)
                and node.test.comparators[0].value == "__main__")

    mains = [n for n in UVR_TREE.body if is_main(n)]
    assert len(mains) == 1, f"found {len(mains)} __main__ blocks, want exactly 1"
    first = mains[0].body[0]
    assert isinstance(first, ast.Raise), (
        f"the __main__ block opens with a {type(first).__name__} at line {first.lineno}, "
        "not a refusal -- everything above a raise still runs.")
    exc = first.exc
    assert isinstance(exc, ast.Call) and ast.unparse(exc.func) == "SystemExit", (
        f"__main__ raises {ast.unparse(exc) if exc else first.cause!r}; the ruling asks "
        "for SystemExit so `python UVR.py` exits non-zero without a traceback flood.")
    args = [ast.literal_eval(a) for a in exc.args]
    assert args == [REFUSAL], (
        f"the refusal says {args!r} but tests quote it verbatim, and users are told "
        "where to go instead: update the string here and in UVR.py together.")


def test_separate_py_has_no_model_data_left_and_still_parses():
    src = (REPO / "separate.py").read_text(encoding="utf-8")
    assert "ModelData" not in src, (
        "separate.py still names ModelData. Its TYPE_CHECKING import of a class that "
        "no longer exists would be a lie to every reader of the file.")
    ast.parse(src)
