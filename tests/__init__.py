"""Marks ``tests`` as a regular package.

Required, not cosmetic: the ``spafe`` wheel (a Task 2 dependency of the Roformer
stack) installs a top-level ``tests`` package into site-packages. Without an
``__init__.py`` here, ``tests`` resolves as a *namespace* package, and a regular
package later on ``sys.path`` beats a namespace portion — so
``import tests.golden`` picked up site-packages/tests and failed. With this file
the repository root (first on ``sys.path`` via ``tests/conftest.py``) wins.
"""
