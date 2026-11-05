"""The Fluent-Qt front end. Nothing in here may be imported by the engine.

`tests/test_engine_purity.py` blocks `app` (alongside `PySide6` and `fluentqt`) in
the child process that imports the engine surface, so the direction in the design
diagram is enforced rather than promised.
"""
