# PyGrad

A lightweight automatic differentiation library implementing dynamic computation
graph construction and reverse-mode autodiff, with NumPy and CuPy backends for
CPU and GPU execution.

## Development setup

PyGrad keeps runtime dependencies in `pyproject.toml`. Development tools are
declared in the `dev` optional dependency group instead of being mixed into
the package users install.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Install the optional GPU backend when the matching CUDA runtime is available:

```bash
python -m pip install -e ".[gpu]"
```

For reproducible development, use a lockfile tool such as
[uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev
uv lock
```

Commit `uv.lock` for an application or a development environment. For a
reusable library, keep version ranges in `pyproject.toml` so pip can resolve
compatible dependencies for each user's platform; use a lockfile for CI and
local development.

## Building an installable package

Build source and wheel distributions from the repository root:

```bash
python -m pip install --upgrade build
python -m build
```

The generated files are written to `dist/`. Test the wheel in a clean virtual
environment before publishing:

```bash
python -m venv /tmp/pygrad-wheel-test
/tmp/pygrad-wheel-test/bin/python -m pip install dist/pygrad-*.whl
/tmp/pygrad-wheel-test/bin/python -c "import pygrad; print(pygrad.__version__)"
```

After registering a package name on PyPI, publish with a trusted publishing
workflow from GitHub Actions or, for a one-off release:

```bash
python -m pip install twine
python -m twine upload dist/*
```

Choose a new version in `pyproject.toml` for each release. Do not commit API
tokens; configure them through PyPI trusted publishing or environment-backed
credentials.
