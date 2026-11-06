"""Item 12 (wave 2): the red controls behind the purity claims are *run*, not cited.

`tools/task8_red_controls.py` is deliberately not named `test_*.py`, so a bare `pytest`
never collects it (there is no `pytest.ini`/`pyproject.toml` here -- the filename is the
whole guard). That design is right for the controls themselves but had one cost the
review named exactly: nothing in the suite ever executed them, so the line in the
purity docstring that "the red runs behind every claim here" was true-and-silently-
rotting -- a regression that let a control stop firing (or a guard become decorative)
would not show up until someone remembered to run the file by hand. The reviewer ran it
explicitly and it was 17 passed in ~39s, cheap against the ~225s suite.

This test closes the gap by shelling out to run the committed controls and asserting rc
0 plus a real passed count (non-vacuity: a green `no tests ran` exit would otherwise
look like success). It runs the controls in an isolated child pytest so this suite's own
fixtures/collection are untouched; the controls spawn a torch-importing child each, so
the bound is a generous wall-clock ceiling rather than a guess at the child timeout.
"""
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONTROLS = REPO / "tools" / "task8_red_controls.py"

# The controls fan out to a child per engine module / per GUI name plus the plant
# probes; the committed per-child bound is 600s and a full pass was measured at ~39s,
# so 1200s is headroom that still bounds a hang without failing a slow-but-honest run.
CONTROLS_WALL_CLOCK_SECONDS = 1200


def test_task8_red_controls_all_pass():
    assert CONTROLS.is_file(), (
        f"the committed red controls at {CONTROLS} are missing -- item 12 exists to run "
        "them, and a file moved/deleted means every purity claim is now uncited.")
    started = time.time()
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(CONTROLS)],
        cwd=REPO, capture_output=True, text=True, timeout=CONTROLS_WALL_CLOCK_SECONDS)
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
