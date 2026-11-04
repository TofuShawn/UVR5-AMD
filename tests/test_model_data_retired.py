"""Task 11d: `ModelData` is retired, and the file says so in three places.

`UVR.py` is 7,400 lines of legacy tkinter GUI that Plan 2 replaces, so this test
deliberately checks *shapes in the AST* rather than importing the module -- it
cannot be imported on this machine anyway (it dies at `import pyglet`, several
hundred lines before any of this). Task 12 parses it the same way.

Three properties, one per clause of the 11d ruling:

1. `ModelData` is gone -- not defined, and not *loaded* by name anywhere. The AST
   matters because the file still speaks the word in prose (the `assemble_model_data`
   docstring, the comment above the refusal), and a substring check would fail on
   text that is doing useful work. `ApolloModelData` shares the suffix, which is a
   second reason to compare `Name.id` rather than grep.
2. `ApolloModelData` survives. It sits outside the deleted range (`:1059` when
   `ModelData` started at `:365`) and it is the only Apollo resolution path, so
   deleting it would have silently blocked a working arch.
3. The legacy entry point refuses as the *first statement* of the `__main__` block.
   Not a `sys.exit()` at the bottom, not a warning: nothing above it may run, so
   that no window, no DPI call and no half-read config table precedes the refusal.
"""
import ast
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]
UVR_SRC = (REPO / "UVR.py").read_text(encoding="utf-8")
UVR_TREE = ast.parse(UVR_SRC)
REFUSAL = ("legacy tkinter GUI removed; use separate_cli.py "
           "(PySide6 app arrives in Plan 2)")


def _top_level(name):
    return [n for n in UVR_TREE.body
            if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name]


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


def test_apollo_model_data_still_defined():
    """It lived outside 365-804 and it is the only Apollo resolution path."""
    defs = _top_level("ApolloModelData")
    assert len(defs) == 1, f"ApolloModelData defined {len(defs)} times, want 1"
    assert ast.unparse(defs[0]).count("def "), "ApolloModelData lost its methods"


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


def test_assemble_model_data_refuses_instead_of_half_running():
    """Its old body was seven `ModelData(...)` comprehensions."""
    holder = next((n for n in ast.walk(UVR_TREE)
                   if isinstance(n, ast.FunctionDef) and n.name == "assemble_model_data"),
                  None)
    assert holder is not None, "assemble_model_data vanished; its 11 callers need it to exist"
    body = [s for s in holder.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
    assert len(body) == 1 and isinstance(body[0], ast.Raise), (
        "assemble_model_data should raise and do nothing else -- anything more is a "
        "legacy path that half-works.")


def test_separate_py_has_no_model_data_left_and_still_parses():
    src = (REPO / "separate.py").read_text(encoding="utf-8")
    assert "ModelData" not in src, (
        "separate.py still names ModelData. Its TYPE_CHECKING import of a class that "
        "no longer exists would be a lie to every reader of the file.")
    ast.parse(src)
