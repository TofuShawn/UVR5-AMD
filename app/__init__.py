"""The Fluent-Qt front end. No engine module imports it -- at import time.

That qualifier is the whole claim, and stating it sloppier cost this fork two fix
rounds in Plan 2 Task 1, so: `tests/test_engine_purity.py` blocks `app` alongside
`PySide6` and `fluentqt` in the child process that imports each module of the engine
surface, which proves *an engine module reaching `app` while it is being imported*
fails a guard. It does not prove no engine file ever names `app`, and one edge is
deliberate: `separate_cli.run()` imports `app.session` from inside the function body
(ruling R-T2-blockset), because `session.assemble` is the fork's single assembly path
and the CLI is supposed to go through it rather than around it.

What the block set still proves about that edge is Qt-freedom: `app` is the only name
`RUN_CHILD_BLOCKED` drops, `PySide6` and `fluentqt` stay in it, and the CLI runs in that
child. `tests/test_session.py::test_importing_session_pulls_in_no_qt` asserts the same
property straight at `app.session` -- importing it reaches no Qt module, which is why a
GUI-free run never needs a display, and why the engine surface can keep importing
`separate_cli` itself.
"""
