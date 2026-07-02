# AGENTS.md

## Cursor Cloud specific instructions

`addivortes` is a single Python library (no long-running services, servers, or
databases). It ships a compiled C++20 pybind11 extension (`addivortes._core`),
so a C++20 compiler and the Python dev headers are required at install time
(these are provided by the VM snapshot: `g++`, `python3.12-dev`,
`python3.12-venv`).

### Environment
- Dependencies live in a virtualenv at `.venv` (git-ignored). The startup update
  script creates it and runs an editable install with the `dev` extra.
- Activate it before running any command: `source .venv/bin/activate` (or prefix
  commands with `.venv/bin/`).
- The install is editable, but the `_core` C++ extension is compiled at install
  time. If you change `python_src/addivortes_python.cpp`, re-run
  `.venv/bin/python -m pip install -e ".[dev]"` to recompile; Python-only edits
  under `addivortes/` are picked up without reinstalling.

### Common commands (run inside the venv)
- Tests: `python -m pytest` (config in `pyproject.toml`, `testpaths=["tests"]`).
- Docs build: `python -m mkdocs build --strict`; live preview: `python -m mkdocs serve`.
- Package: `python -m build --sdist` then `python -m twine check dist/*`.
- There is no configured linter/formatter (no ruff/mypy/flake8/black config and
  no lint step in CI), so there is no lint command to run.

### Notes
- `python` is not on PATH system-wide (only `python3`), but the venv provides
  `python` once activated.
