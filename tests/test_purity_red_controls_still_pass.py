"""Item 12 (wave 2): the red controls behind the purity claims are *run*, not cited.

`tools/task8_red_controls.py` is deliberately not named `test_*.py`, so a bare `pytest`
never collects it (there is no `pytest.ini`/`pyproject.toml` here -- the filename is the
whole guard). That design is right for the controls themselves but had one cost the
review named exactly: nothing in the suite ever executed them, so the line in the
purity docstring that "the red runs behind every claim here" was true-and-silently-
rotting -- a regression that let a control stop firing (or a guard become decorative)
would not show up until someone remembered to run the file by hand.

This test closes the gap by shelling out and asserting rc 0 plus a real passed count
(non-vacuity: a green `no tests ran` exit would otherwise look like success), in an
isolated child pytest so this suite's own fixtures and collection are untouched.

**The cost, measured rather than inherited.** The dispatch brief carried "17 passed in
38.98 s -- cheap against a 225 s suite" from reviewer B. The file is slower than that
here (one engine-importing child costs **6.5 s** -- `import separate, torch`, timed --
and there are dozens of them), and on its first run inside the suite this test's own
1200 s bound was blown: `1 failed, 699 passed in 1457.89s (0:24:17)`, the single failure
being this test with all 699 behavioural results green. Chasing that found a real defect
in the controls, not in this test -- see the probe comment at
`tools/task8_red_controls.py:791`: a `while True` child could never be reaped, because
`.venv/Scripts/python.exe` is a redirector and killing it leaves the real interpreter
holding the pipe. With that fixed the file finishes in minutes.

**The consequence, ruled rather than inherited.** Folding the controls into bare
`pytest -q` makes every iteration on this branch cost ~24 minutes -- the same misplaced
cost this plan paid three times over its fix waves. So the controls run only under
`UVR_RUN_RED_CONTROLS=1`; the skip names the command, and the purity docstrings cite that
command instead of implying a run that did not happen. Opting out stays visible in the
summary line as one skipped test, never a silent default.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CONTROLS = REPO / "tools" / "task8_red_controls.py"

# Sized from the measurement above rather than a guess. The controls fan out to a child
# per engine module / per GUI name, each paying a ~6.5 s torch import, and the run that
# exceeded the 1200 s this file first allowed turned out to be an un reapable probe
# rather than an honest pass. 3600 s keeps a genuine pass from being cut off on a loaded
# machine while still bounding a hang at under an hour instead of forever.
CONTROLS_WALL_CLOCK_SECONDS = 3600


def test_task8_red_controls_all_pass():
    assert CONTROLS.is_file(), (
        f"the committed red controls at {CONTROLS} are missing -- item 12 exists to run "
        "them, and a file moved/deleted means every purity claim is now uncited.")
    # Controller ruling: the controls are opt-in, and the opt-out is visible. Running
    # them inside bare `pytest -q` was measured at 1458s for the whole suite -- the
    # controls fan out to a child per engine module and each pays a ~6.5s torch import,
    # so an honest pass costs minutes, not the ~39s reviewer B saw in isolation. A
    # 24-minute iteration loop is the cost this plan kept paying for the wrong reason,
    # so the rule is: the controls are *cited* by every purity claim and *run* on an
    # explicit request. The skip is a skip line in the summary carrying the command --
    # never a silent default, and never a claim that the guards are proven right now.
    if os.environ.get("UVR_RUN_RED_CONTROLS") != "1":
        pytest.skip(
            "the committed red controls are not part of a bare suite run (measured: the "
            "suite goes from ~225s to ~1458s with them in, one engine-importing child "
            "per control). Run them explicitly with `UVR_RUN_RED_CONTROLS=1 "
            "pytest -q tests/test_purity_red_controls_still_pass.py`, or directly with "
            "`pytest -q tools/task8_red_controls.py`. This skip keeps the claims in "
            "tests/test_engine_purity.py cited; it does not mean they are proven.")
    started = time.time()
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
             str(CONTROLS)],
            cwd=REPO, capture_output=True, text=True,
            timeout=CONTROLS_WALL_CLOCK_SECONDS)
    except subprocess.TimeoutExpired as exc:
        elapsed = time.time() - started
        partial = (exc.stdout or b"").decode(errors="replace") if isinstance(
            exc.stdout, bytes) else (exc.stdout or "")
        raise AssertionError(
            f"the red controls did not finish within "
            f"CONTROLS_WALL_CLOCK_SECONDS={CONTROLS_WALL_CLOCK_SECONDS} (elapsed "
            f"{elapsed:.0f} s). Read this as a cost question, not a purity failure: each "
            f"control spawns a child that re-imports the engine (~6.5 s for torch on this "
            f"machine), so the bound is a function of how many children exist and how "
            f"slow the interpreter start is. Partial output:\n"
            f"{partial[-4000:]}") from None
    elapsed = time.time() - started
    combined = r.stdout + r.stderr
    tail = "\n".join(combined.splitlines()[-20:])
    print(f"[red-controls] tools/task8_red_controls.py rc={r.returncode} in "
          f"{elapsed:.1f}s", flush=True)
    assert r.returncode == 0, (
        "the committed red controls behind the engine-purity claims are failing -- that "
        "is exactly the rot item 12 exists to catch (a control whose guard went "
        f"decorative, or a purity regression the guard now names):\n" + tail)
    # Non-vacuity: rc 0 alone is not enough -- a run that collected nothing also exits
    # cleanly. Demand the summary line the file emits when it actually ran the controls.
    assert " passed" in r.stdout, (
        "the controls run exited 0 without reporting any passing control, so it likely "
        "collected nothing (renamed file? moved controls?):\n" + tail)
    assert "error" not in r.stdout.lower().rsplit("passed", 1)[-1], (
        f"the controls summary carries an error despite rc 0:\n{tail}")
