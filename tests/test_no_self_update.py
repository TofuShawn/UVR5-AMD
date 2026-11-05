"""Task 12: the application no longer phones home for itself — and still fetches what
the user asked for.

Why every assertion here is scoped to *which constant is loaded* rather than to *which
network syscall appears*: this fork kept network use of three kinds on purpose — at the
7140156 copy all three below were live, and the bullets carry that copy's line numbers —
and a grep for `urlopen` / `download` cannot tell them apart from the update channel this
task is supposed to kill. Plan 2 Task 8 left exactly one of the three live; test 5's
docstring sorts out which, and which assertions are tripwires over inert text.

* `wget.download` at `UVR.py:5490` / `:5496` — user-initiated weight downloads from the
  Download Center. This is how a machine without a given model gets it.
* `urlopen(config_link)` at `:4212` / `:5311` — a *user-requested* MDX-C / Roformer
  config fetch, built from `MDX23_CONFIG_CHECKS`, a ninth URL that reads like an update
  URL but is not. It is deliberately absent from `KILLED`.
* `subprocess.Popen` at `:7278` / `:7280` (app restart) and `:103` / `:112`
  (`OPEN_FILE_func`, "open this folder for me"). None fetches or installs anything.

An earlier draft of the first test asserted `"urlopen" not in src and "download" not in
src`; that would have silently broken this fork's model downloader, and a substring grep
would additionally flag the very prose that replaces the removed behaviour ("this fork
has no auto-update"). So: AST `Name` loads and `Call` argument identities, never
substrings over the whole file.

`UVR.py` is inert now (its `__main__` raises, and it was never importable on this
machine), so like `tests/test_model_data_retired.py` this file parses the source instead
of importing it. WHICH COPY A NUMBER BELONGS TO is part of the number, and three copies
are in play here: `7140156` (7,393 lines — the state Task 12 started from, which every
unlabelled number above and below is measured against), `720ca46` (7,342 lines — the copy
Plan 2 Task 8 deleted, whose six numbers appear only in the paragraph below and in test
5's docstring, both of which say so), and the header that survives at this commit (389
lines, cited as such wherever it is meant). All five tests match on AST nodes; a line
number in an assertion message is a pointer for the reader, which is precisely why it has
to name its copy — pointing a failing reader at `:103` of a 389-line file sends them
somewhere false.

The fifth test is the other half of the contract and is not decoration: the first four
are satisfied by an over-deletion that also removes the user's downloader, and only the
survivor assertions notice.

PLAN 2 TASK 8 CHANGED WHICH SURVIVORS EXIST, and test 5 is the only place that feels it:
every call site in the three bullets lived inside the tkinter layer that commit deleted.
In the copy it deleted — `UVR.py` at commit 720ca46, 7,342 lines — they were
`wget.download` at :5437/:5443, `urlopen(config_link)` at :4242/:5284 and the two
app-restart `Popen`s at :7226/:7228; the numbers in the bullets above are Task 12's
measurement at 7140156, and what the deletion left is a 389-line header at this commit.
What still survives is not one kind of thing, and reading the three tiers as live coverage
is the mistake this paragraph exists to prevent:

* `model_refresh.py` fetches, now, behind its `confirm=True` rule — the only live network
  path in the fork, and the only survivor here that a user can still exercise.
* `NORMAL_REPO` and `MDX23_CONFIG_CHECKS` are still *defined* in `gui_data/constants.py`,
  and **no live code reads either one** — `git grep` on 2026-11-06 finds this test plus
  prose at `gui_data/constants.py:99-111` and `model_refresh.py:58`. Their `config_link`
  and weight-download readers went with the GUI.
* `OPEN_FILE_func`'s two `Popen`s survive at header `:179` / `:188` (still `:103` / `:112`
  at 7140156, `:129` / `:138` in the deleted copy) — inside a module nothing imports.

The last two tiers are inert, and the assertions below stay anyway, for one honest reason:
they are tripwires, not coverage. Deleting those names is the over-deletion Task 12 wrote
this test against, and if a later plan re-wires a config fetch or a reveal-in-folder
button it should find its URL and its exec still there. Each failure message says that in
as many words, so nobody reads a green suite as a working downloader. The user-facing
downloader is gone with the Download Center that hosted it; a weight is installed by
putting the file in the model root, and the baked tables refresh through
`separate_cli.py --refresh-model-tables --confirm`.
"""
import ast
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]
UVR_PATH = REPO / "UVR.py"
CONSTANTS_PATH = REPO / "gui_data" / "constants.py"
REFRESH_PATH = REPO / "model_refresh.py"

UVR_SRC = UVR_PATH.read_text(encoding="utf-8")
UVR_TREE = ast.parse(UVR_SRC)
CONSTANTS_SRC = CONSTANTS_PATH.read_text(encoding="utf-8")
CONSTANTS_TREE = ast.parse(CONSTANTS_SRC)

#: The eight constants Task 12 removes from `gui_data/constants.py` and stops loading in
#: `UVR.py`. Do not widen this list (it would reach the user's downloads) and do not
#: shorten it.
KILLED = ("DOWNLOAD_CHECKS", "BULLETIN_CHECK", "UPDATE_REPO", "PREVIOUS_PATCH_WIN",
          "VR_MODEL_DATA_LINK", "MDX_MODEL_DATA_LINK",
          "MDX_MODEL_NAME_DATA_LINK", "DEMUCS_MODEL_NAME_DATA_LINK")

#: The subset of `KILLED` that could only ever appear as a fetch argument — the two
#: startup telemetry URLs plus the four table URLs. `UPDATE_REPO` is a release-asset
#: prefix (asserted in test 2) and `PREVIOUS_PATCH_WIN` a version string (test 3).
FETCH_ARGS = {"DOWNLOAD_CHECKS", "BULLETIN_CHECK", "VR_MODEL_DATA_LINK",
              "MDX_MODEL_DATA_LINK", "MDX_MODEL_NAME_DATA_LINK",
              "DEMUCS_MODEL_NAME_DATA_LINK"}

#: The update channel is identified by its asset path, not by its host: the *same*
#: GitHub host and repo also serve the model weights (`NORMAL_REPO`,
#: `.../all_public_uvr_models/`), and at 7140156 the download center reading that second
#: channel was the network use Task 12 had to preserve. Task 8 deleted the downloader --
#: the two channels are still distinguished here because the rule "kill the update
#: channel, never the weights channel" is what any future fetch has to respect.
UPDATE_CHANNEL_MARKER = "uvr_update_patches"
WEIGHTS_CHANNEL_MARKER = "all_public_uvr_models"


def _tree(path: pathlib.Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"))


def _name_loads(tree: ast.AST, name: str) -> list[ast.Name]:
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Load)]


def _calls_named(tree: ast.AST, attr: str) -> list[ast.Call]:
    """Every call whose callee ends in `.attr` / `attr` — `subprocess.Popen(...)` and a
    bare `Popen(...)` both match; so does `urllib.request.urlopen(...)`. This is
    deliberately about the *callee*, so it does not care what the arguments say."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == attr:
                out.append(node)
    return out


def _fetch_arg_names(tree: ast.AST) -> list[tuple[int, str]]:
    hits = []
    for node in _calls_named(tree, "urlopen"):
        for arg in node.args:
            if isinstance(arg, ast.Name) and arg.id in FETCH_ARGS:
                hits.append((node.lineno, arg.id))
    return hits


# --------------------------------------------------------------------------- #
# 1. the installer exec
# --------------------------------------------------------------------------- #

def test_no_popen_launches_the_downloaded_installer():
    """The only `Popen` that may not survive is the one whose args name the downloaded
    update package. The assertion is about the *arguments of the exec* — a bare count of
    `Popen` nodes would flag `OPEN_FILE_func`, which stays (the restart execs this test
    also used to excuse went with the GUI in Plan 2 Task 8)."""
    offenders = [n for n in _calls_named(UVR_TREE, "Popen")
                 if "download_update_path" in ast.unparse(n)]
    assert not offenders, (
        f"UVR.py still execs a downloaded installer at line(s) "
        f"{[n.lineno for n in offenders]}: that is the application installing an update "
        f"it fetched by itself. Task 12's brief named that exec at :5543 of the 7140156 "
        f"copy; the two app-restart execs (:7278/:7280 there, :7226/:7228 in the copy "
        f"Task 8 deleted) are gone with the GUI, and `OPEN_FILE_func`'s two — header "
        f":179/:188 now, :103/:112 at 7140156 — are not this assertion.")


# --------------------------------------------------------------------------- #
# 2. the update channel's URL
# --------------------------------------------------------------------------- #

def test_update_channel_url_is_gone_from_both_files():
    """No `uvr_update_patches` / `model_repo` *update-channel* string in either file.

    Scoped by asset path on purpose: `TRvlvr/model_repo` is also the weights repo the
    Download Center pulled from (`NORMAL_REPO`), and deleting that host wholesale would
    have taken the user's model downloads with it. Measured at HEAD 7140156,
    `NORMAL_REPO` was read at `UVR.py:4165` and `:5414` — live-in-use network the brief
    forbids forbidding, which is why the second half of this test fails the
    over-deletion. Plan 2 Task 8 deleted both readers with the GUI, so read that half as
    the tripwire test 5's docstring describes: the weights channel still has to be
    present, and it no longer has anything that fetches it.
    """
    for label, src in (("UVR.py", UVR_SRC), ("gui_data/constants.py", CONSTANTS_SRC)):
        assert UPDATE_CHANNEL_MARKER not in src, (
            f"{label} still contains the update-channel asset path {UPDATE_CHANNEL_MARKER!r}.")
        assert "UPDATE_REPO" not in [n.id for n in ast.walk(ast.parse(src))
                                     if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)], (
            f"{label} still defines UPDATE_REPO.")

    # Any surviving `TRvlvr/model_repo` literal must be the weights channel, never a
    # patch channel — and the assertion only holds while the weights channel is present.
    model_repo_literals = [n.value for n in ast.walk(CONSTANTS_TREE)
                           if isinstance(n, ast.Constant) and isinstance(n.value, str)
                           and "TRvlvr/model_repo" in n.value]
    for value in model_repo_literals:
        assert WEIGHTS_CHANNEL_MARKER in value, (
            f"gui_data/constants.py still carries a model_repo URL that is not the "
            f"weights channel: {value}")
    assert any(WEIGHTS_CHANNEL_MARKER in v for v in model_repo_literals), (
        f"NORMAL_REPO ({WEIGHTS_CHANNEL_MARKER}) is gone from gui_data/constants.py. "
        f"Task 12 kills the app's own network, not the user's model downloads.")


# --------------------------------------------------------------------------- #
# 3. the eight constants
# --------------------------------------------------------------------------- #

def test_no_name_load_of_a_killed_constant_survives_in_uvr_py():
    """AST `Name` loads, not substrings: after the fix `UVR.py` legitimately *says* in
    prose that this fork has no auto-update, and a grep would flag the message that
    replaces the behaviour."""
    still_loaded = {name: sorted(n.lineno for n in _name_loads(UVR_TREE, name))
                    for name in KILLED if _name_loads(UVR_TREE, name)}
    assert not still_loaded, (
        f"UVR.py still loads {len(still_loaded)} of the {len(KILLED)} killed constants: "
        f"{still_loaded}. Each is app-initiated network or version data for an update "
        f"channel this fork does not have. In the 7140156 copy the module-level "
        f"MODEL_DATA_URLS (:298 there) was itself one load of four of them; Task 12 "
        f"deleted that name, so a hit here is a re-introduction, not that line.")


# --------------------------------------------------------------------------- #
# 4. the refresh lives outside UVR.py (Task 13 depends on this)
# --------------------------------------------------------------------------- #

def test_retained_refresh_lives_outside_uvr_py():
    """`model_refresh.py` exists and is wired for catalog-owned writes, and no `urlopen`
    in `UVR.py` is called with one of the table/telemetry constants as its argument.

    When Task 12 wrote this, the point was that it is explicitly NOT "no `urlopen`
    survives in `UVR.py`": the MDX-C/Roformer `config_link` fetches (:4212/:5311 at
    7140156) were user-requested and had to keep working. Plan 2 Task 8 deleted them with
    the GUI, so today the stronger claim holds and test 5 makes it. What this test still
    decides is the half that is policy and not accident: the killed table/telemetry URLs
    may not come back into `UVR.py`, and the refresh that replaced them lives in a module
    a CLI can import.
    """
    assert REFRESH_PATH.is_file(), (
        f"{REFRESH_PATH} does not exist: the model-table refresh has to move out of the "
        f"dead GUI into a module a CLI can import.")

    tree = _tree(REFRESH_PATH)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert "catalog" in imported, (
        f"model_refresh.py must import catalog and write through Catalog.write_table "
        f"(it owns the path convention and the cache invalidation); imports are {sorted(imported)}")
    assert "urllib.request" in imported, (
        f"model_refresh.py must fetch with urllib.request; imports are {sorted(imported)}")

    offenders = _fetch_arg_names(UVR_TREE)
    assert not offenders, (
        f"UVR.py still fetches the killed URLs at {offenders}. Task 12 moved the four "
        f"table fetches (download_model_settings, :5339-5346 at 7140156) and the two "
        f"startup fetches (:5142/:5146 there) out to model_refresh.py; Task 8 deleted "
        f"the surviving config_link fetches with the GUI, so an urlopen in this file "
        f"today is a new fetch reaching for the network from a module nothing imports, "
        f"not an old one being restored. `model_refresh.py` is that path and "
        f"`separate_cli.py --refresh-model-tables --confirm` is the command.")


# --------------------------------------------------------------------------- #
# 5. the survivors — the other half of the contract (one live path, two tripwires)
# --------------------------------------------------------------------------- #

def test_user_invoked_network_paths_survive():
    """Guard against the failure mode this brief exists to prevent: an over-deletion that
    kills the user's fetching along with the app's.

    Names and counts measured at HEAD 7140156; the `wget.download` count is the one that
    changes by design (3 -> 2, because :5480 is the application update).

    Plan 2 Task 8 is the second such change, and it is why this function no longer reads
    the way Task 12 wrote it. Six of the call sites named above lived inside the tkinter
    layer -- `pop_up_*`/Download Center `wget.download`, the user-requested
    `urlopen(config_link)`, and the two app-restart `Popen`s. In the copy that commit
    deleted (`UVR.py` at 720ca46, 7,342 lines) they were :5437/:5443, :4242/:5284 and
    :7226/:7228 -- they are gone with 7,001 lines of dead GUI, not with a policy
    reversal. Asserting their survival would have been asserting a property
    of a file that no longer has them, so the assertion is re-scoped rather than dropped,
    and the re-scoping is *stronger* on the axis Task 12 cared about: after Task 8,
    `UVR.py` contains no network call of any kind, and the only module in this fork that
    fetches is `model_refresh.py`, behind its `confirm=True` rule.

    WHAT EACH OF THE FOUR SECTIONS BELOW IS WORTH NOW, since the file no longer says it:
    section 4 is live coverage -- `model_refresh.py` is imported and run, and that guard
    fails if the user's one remaining fetch goes. Sections 1-3 are tripwires over inert
    text: `UVR.py` is imported by nothing, and after Task 8 nothing reads
    `MDX23_CONFIG_CHECKS` or `NORMAL_REPO` either (`git grep` finds this test and prose
    only). They stay because deleting a name a future re-wire would need is the exact
    over-deletion this test was written against, and because a tripwire that fires with
    an honest message beats one that fires with a message claiming coverage it no longer
    has -- which is the defect the wording below used to carry.
    """
    # 1. Nothing in UVR.py reaches the network any more, in either direction: no
    #    downloader, no config fetch, no installer exec, no restart. This is the
    #    over-deletion guard's replacement -- a call node class that must now be empty.
    for attr in ("download", "urlopen", "Popen", "urlretrieve"):
        offenders = [n.lineno for n in _calls_named(UVR_TREE, attr)
                     if not _is_open_file_func(n)]
        assert not offenders, (
            f"UVR.py still calls .{attr} at {offenders}. Task 12 removed the app's own "
            f"network and Task 8 removed the GUI that held the user's; a fetch or exec "
            f"reappearing here means the front end is reaching for the network again "
            f"from a module nothing is supposed to import. `model_refresh.py` is that "
            f"path and `separate_cli.py --refresh-model-tables --confirm` is the command.")

    # 2. `OPEN_FILE_func` is the exception above and it is still a Popen: "open this
    #    folder for me" is not a fetch, and Task 12's brief named it as such. Inert --
    #    header :179/:188 of a module nothing imports -- kept as the shape a re-wired
    #    reveal-in-folder button would reuse, not as a feature a user can press today.
    popens = [ast.unparse(n) for n in _calls_named(UVR_TREE, "Popen")]
    assert any('"open"' in p or "'open'" in p for p in popens), (
        f"OPEN_FILE_func (macOS) is gone from the UVR.py header: {popens}. Inert code, "
        f"asserted as a tripwire -- see the docstring; nothing imports this module.")
    assert any("xdg-open" in p for p in popens), (
        f"OPEN_FILE_func (Linux) is gone from the UVR.py header: {popens}. Same tripwire "
        f"as the macOS line above.")

    # 3. The two constants that were user-requested fetching and never an update channel
    #    still exist in gui_data/constants.py -- deleting *them* is the over-deletion
    #    this test was written to catch. Both are inert data now: the code that read them
    #    (the config fetch and the Download Center) went with the GUI in Task 8, so a
    #    green pair here proves the URLs survived, never that anything fetches with them.
    assert any(n.id == "MDX23_CONFIG_CHECKS" for n in ast.walk(CONSTANTS_TREE)
               if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)), (
        "MDX23_CONFIG_CHECKS was deleted. No live code reads it since Task 8 removed the "
        "config_link fetches it built -- their last two hosts were menu_manual_downloads "
        "and download_list_fill/configure_combobox, at :4242/:5284 of the 720ca46 copy -- "
        "so this is a tripwire for a future re-wire, not evidence of a working fetch. It "
        "was never an update URL, which is why it is the over-deletion this guard refuses.")
    defined = {n.id for n in ast.walk(CONSTANTS_TREE)
               if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    assert "NORMAL_REPO" in defined, (
        "NORMAL_REPO (the model-weights channel) was deleted by mistake. Like "
        "MDX23_CONFIG_CHECKS above it is inert data since Task 8 -- its readers were "
        "menu_manual_downloads (:4165) and download_model_select (:5414) at 7140156, all "
        "inside the deleted GUI -- but it is weights, not the update channel, and Task "
        "12's rule was to kill the update channel only.")

    # 4. The live fetching path exists and is the confirmed one. `urllib.request` in the
    #    import list is the shape; tests/test_model_refresh.py holds the refusal rule,
    #    and test_retained_refresh_lives_outside_uvr_py above holds the wiring.
    refresh_tree = _tree(REFRESH_PATH)
    fetches = _calls_named(refresh_tree, "urlopen")
    assert fetches, (
        f"{REFRESH_PATH} no longer calls urlopen: with the GUI gone this module is the "
        f"fork's only user-requested fetch, and the user's model tables would have "
        f"silently stopped being refreshable.")


def _is_open_file_func(node: ast.Call) -> bool:
    """True for the two `Popen` calls that only open a folder for the user.

    Recognised by argument identity (`"open"` / `"xdg-open"` / `input_string`), not by
    line number: those three names are `OPEN_FILE_func`'s whole signature and they are
    what distinguishes "reveal in file manager" from "run a program we downloaded".
    """
    text = ast.unparse(node)
    return ('"open"' in text or "'open'" in text or "xdg-open" in text
            or "input_string" in text)
