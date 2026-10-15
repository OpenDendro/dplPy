# dplPy Developer Instructions

Welcome to the dplPy developer manual. We welcome all code contributions, bug
reports, bug fixes, documentation improvements, and suggestions.

## Index
- [Environment setup](#environment-setup)
- [Repository layout](#repository-layout)
- [Running the tests](#running-the-tests)
- [Building the documentation](#building-the-documentation)
- [Coding conventions](#coding-conventions)
- [Making changes and submitting a pull request](#making-changes-and-submitting-a-pull-request)
- [Releasing](#releasing)
- [Versioned documentation with mike (option, pre-1.0)](#versioned-documentation-with-mike-option-pre-10)

The user-facing API reference is generated from the docstrings and published at
<https://opendendro.github.io/dplPy/reference/io/>; it is not duplicated here.

## Environment setup

### 1. GitHub setup

#### 1.1 Create a dplPy fork on GitHub

You will need your own copy of dplPy to work on the code. Go to the
[dplPy GitHub page](https://github.com/OpenDendro/dplPy) and click the fork
button. Make sure the option to copy only the main branch is unchecked.

#### 1.2 Create a local repository

In your local terminal, clone the fork to your computer using the commands shown
below. Replace `{your-user}` with your GitHub username.

```
$ git clone https://github.com/{your-user}/dplPy.git dplpy-{your-user}
$ cd dplpy-{your-user}
$ git remote add upstream https://github.com/OpenDendro/dplPy.git
$ git fetch upstream
```

This creates a local repository in `dplpy-{your-user}` connected to your fork
(`origin`) and to the main dplPy repository (`upstream`).

#### 1.3 Create a feature branch

First, ensure that the main branch of your fork is up to date with the main
dplPy repository:

```
git checkout main
git pull upstream main --ff-only
```

Then create a new branch for your changes (replace `{feature_name}` with the
name of your feature or preferred branch name):

```
git checkout -b {feature_name}
```

### 2. Python environment

dplPy uses a `src/` layout (the package lives in `src/dplpy/`), so the working
copy must be installed in *editable* mode, or `src` put on the path, for
`import dplpy` to pick up your changes. Either of the following works.

#### 2.1 Option A: conda / mamba

The packages required to develop dplPy are specified in `environment.yml`,
which installs the scientific dependencies from conda-forge and then
`pip install -e .` (editable install of the working copy) plus `pytest`,
`pytest-cov`, `ruff` and the Jupyter tooling for the notebooks. Create the
environment with [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
or [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html):

```
$ conda env create -f environment.yml      # or: mamba env create -f environment.yml
$ conda activate dplpy
```

If prompted for permission to install required packages, select `y`. You will
need to have the environment activated whenever you run or test code from the
package; `(dplpy)` appears at the left of the prompt when it is active.

#### 2.2 Option B: pip and a virtual environment

```
$ python -m venv .venv
$ source .venv/bin/activate                 # Windows: .venv\Scripts\activate
$ pip install -e ".[lipd]" pytest pytest-cov ruff
```

The `[lipd]` extra installs `pylipd` so the LiPD tests run instead of skipping;
it is optional.

#### 2.3 Check the installation

```
$ python -c "import dplpy; print(dplpy.__version__, dplpy.__file__)"
$ python -m dplpy --version
```

The printed path should be inside your working copy (`.../src/dplpy/__init__.py`).

### 3. IDE setup

We recommend using VSCode for development. The following shows how to set up
VSCode to recognize the environment and debug tests.

#### 3.1 Open the dplPy folder in VSCode

In VSCode, open the folder containing your local dplPy repository (named
`dplpy-{your-github-username}` if you followed the instructions above). The
package source is under `src/dplpy/`, one module per public function
(e.g. `src/dplpy/detrend.py`); `src/dplpy/__init__.py` imports every public name.

#### 3.2 Select the environment's Python interpreter

In the bottom corner of the VSCode window, select the language interpreter.
Choose the interpreter `Python 3.x ('dplpy')`, with a path that ends in
`/envs/dplpy/python` (conda) or `.venv/bin/python` (venv). You can then run any
Python file in the folder with the run button instead of through the terminal.

Note: if the terminal is opened after the interpreter has been set, the
environment is activated automatically in that terminal.

#### 3.3 Set up the test explorer

Go to the testing tab (on the left side of the VSCode window). If the tests are
not discovered automatically, open `.vscode/settings.json` (create it if it does
not exist) and add:

```
{
    "python.testing.pytestArgs": [
        "./tests"
    ],
    "python.testing.unittestEnabled": false,
    "python.testing.pytestEnabled": true
}
```

Go back to the testing tab and verify that the dplPy tests are listed. Run them
with the play button to the right of `tests`.

#### 3.4 Keep your fork up to date

Frequently update your local `main` with any changes to the upstream dplPy
`main`:

```
git checkout main
git pull upstream main --ff-only
```

## Repository layout

```
src/dplpy/            the package: one module per public function
src/dplpy/_validate.py shared input validation / warning / option helpers
src/dplpy/cli.py      the `dplpy` console command and dpl.help()/dpl.readme()
tests/unit/           unit tests, one file per module (test_<module>.py)
tests/integs/         integration tests across modules (test_integ_*.py)
tests/data/           test datasets (see tests/data/README.md for provenance)
tests/conftest.py     shared fixtures: ca533_rwl, ca533_rwi, co021_rwl, DATA
docs/                 the mkdocs site (guides and the generated API reference)
notebooks/            example notebooks (copied into docs/examples at build)
CHANGELOG.md          keep-a-changelog format; add an entry with every PR
```

## Running the tests

`pytest` is configured in `pyproject.toml` (`[tool.pytest.ini_options]`): `src`
is put on `sys.path`, so the suite runs from a plain checkout even without an
editable install, and a bare `pytest` finds the `tests/` tree. Run everything
from the repository root:

```
pytest                              # all tests (about 3-5 minutes)
pytest tests/unit --cov=dplpy       # unit tests with coverage
pytest tests/integs -rA             # integration tests, with a summary of every test
pytest tests/unit/test_detrend.py -q  # one module
```

`--cov` needs `pytest-cov` (installed by `environment.yml`); `-rA` prints a
summary of all tests at the end, which is useful for quickly spotting issues.

Shared fixtures live in `tests/conftest.py`: `ca533_rwl`, `ca533_rwi` (spline
detrended) and `co021_rwl` each return a fresh copy of a dataset read once per
session, and `DATA` is the `tests/data` path, so tests do not depend on the
current working directory. Prefer these over re-reading files in each test.

Tests should run headless: `conftest.py` sets the `Agg` matplotlib backend, and
tests that draw figures pass `plot=False`/`show=False` or close their figures.

### Linting

CI runs `ruff check` with the rules in `[tool.ruff]` in `pyproject.toml`
(pyflakes + pycodestyle errors/warnings; `E402` and `E501` are ignored because of
the license header before imports and the wide docstrings). Findings are
currently reported but do not fail the build (`continue-on-error`). Run it
locally before pushing:

```
ruff check .
```

## Building the documentation

The documentation site is built with [MkDocs](https://www.mkdocs.org/) and the
Material theme; the API reference pages (`docs/reference/*.md`) are generated
from the numpy-style docstrings by mkdocstrings, so **the docstring is the
documentation**. Install the docs dependencies and build strictly (the deploy
workflow uses `--strict`, so a broken link or a reference to a missing function
must fail locally too):

```
pip install -r docs/requirements.txt
mkdocs build --strict
mkdocs serve                       # live preview at http://127.0.0.1:8000
```

`mkdocs_hooks.py` copies the example notebooks from `notebooks/` into
`docs/examples/` (gitignored) at build time. When you add a public function:
add it to `src/dplpy/__init__.py` (import and `__all__`), add a `## name` /
`::: dplpy.module.name` entry to the right `docs/reference/*.md` page, mention
it in the relevant guide, and add a CHANGELOG entry.

The site deploys from `main` on every push (`publish-docs.yml`), so it can be
ahead of the released PyPI version; the index page says so in a banner.

## Coding conventions

- Every module starts with the GPL `__copyright__`/`__license__` header, then
  its imports (hence `E402` is ignored in ruff).
- One public function per module, imported in `src/dplpy/__init__.py` and
  listed in `__all__`. Numpy-style docstrings with `Parameters`, `Returns`,
  `Examples`, and a `References` section citing the method (dplR, ARSTAN,
  COFECHA, or the paper). Refer to other functions as `` `name()` `` in plain
  backticks.
- Functions never modify their input DataFrame: copy first, return new
  objects (`tests/unit/test_no_mutation.py` checks this for every transform).
- Use the shared helpers in `_validate.py` for argument checking: `_warn()` for
  warnings that should point at the caller, `_normalize_choice()` for
  case-insensitive option strings, and the frame coercion helpers.
- Keyword names and defaults follow the package-wide vocabulary: `plot=False`
  for figures, `verbose=` for console output, `window_length=` for moving
  windows, `spline_period=` for spline stiffness, `ar_max=` for the AR order
  cap, `lag=` for a crossdating lag search, `fit=`/`method=` for the curve and
  the ratio/difference arithmetic. When a name changes, keep the old one as a
  deprecated keyword alias (with a `DeprecationWarning`) for one release.
- Plotting goes through `_plot_style.py` helpers and never changes global
  matplotlib state; plotting functions return their `Figure`/`Axes`.
- Add or extend a unit test in `tests/unit/test_<module>.py` with every change,
  and a line under `[Unreleased]` in `CHANGELOG.md`.

## Making changes and submitting a pull request

### 1. Commit and push changes to your feature branch

#### 1.1 View and stage changes

Run `git status` in the terminal to view all your uncommitted changes. Stage
changes for commit with `git add`:

```
# to add all files in current directory
git add .

# to add specific folder/files from current directory
git add {folder_name}
```

#### 1.2 Commit and push changes

Note: make sure to test your changes (`pytest`, `ruff check .`, and
`mkdocs build --strict` if you touched docstrings or docs) before committing.

When you're ready, commit your staged changes. You must add a commit message.

```
git commit -m "Your commit message goes here"
```

Push your commits to your fork's remote branch:

```
git push origin {branch_name}
```

Your code is now on GitHub, but for it to be part of the main dplPy repository
you need to create a pull request.

Note: before creating a pull request, wait for your changes to be tested by the
workflow that runs the unit and integration tests on all supported Python
versions (3.11-3.15). You can check the status of the workflows on the GitHub
page of your branch by clicking the status indicator to the right of the most
recent commit message: a check mark if all workflows ran successfully, an X if
any failed, and an orange dot if workflows are still running.

### 2. Creating and submitting a pull request

#### 2.1 Create the pull request

Navigate to your forked dplPy repository on GitHub, and in the pull requests
tab create a new pull request against `OpenDendro/dplPy:main`.

Note: if you recently pushed changes to a branch other than main, GitHub will
sometimes offer to compare and make a pull request immediately. This works too.

#### 2.2 Review changes and submit the PR

Pull requests allow you to view a side-by-side diff of all changed files.
Review all changes to ensure correctness.

If you are satisfied with your changes, give the PR a descriptive title, and
specify in the description what changes were made and what (if any) issues were
addressed. Then submit the pull request. It will be reviewed by the repository
maintainers.

## Releasing

Releases are made by the maintainers from `main` by creating a `vX.Y.Z` branch;
the workflow builds, publishes to PyPI, and creates the GitHub release and
Zenodo archive. The step-by-step procedure, including the CHANGELOG and
`CITATION.cff` updates, is in `release-instructions.md`.

## Versioned documentation with mike (option, pre-1.0)

Today `publish-docs.yml` runs `mkdocs gh-deploy` on every push to `main`, so the
live site always documents the development branch; the index page carries a
banner saying so. Before v1.0.0 it is worth considering publishing versioned
docs with [`mike`](https://github.com/jimporter/mike), the version manager
designed for mkdocs-material, so that <https://opendendro.github.io/dplPy/>
shows the released API by default and the development docs under a separate
label. This section describes how that would work; it is **not** implemented.

**How mike works.** `mike deploy <version> [alias]` builds the site and commits
it into a *subdirectory* of the `gh-pages` branch (`/0.9.0/`, `/dev/`, ...),
leaving the other versions in place, and maintains a `versions.json` at the
root that mkdocs-material's version selector reads. `mike set-default latest`
writes a root `index.html` that redirects to the `latest` alias. The `gh-pages`
branch therefore holds every published version side by side, and a deploy only
rewrites one directory.

**Configuration.** In `mkdocs.yml`:

```yaml
extra:
  version:
    provider: mike
    default: latest
```

and add `mike` to `docs/requirements.txt`. The theme then shows a version
drop-down in the header populated from `versions.json`.

**Two deploy paths in `publish-docs.yml`:**

- On every push to `main` (today's trigger): build the development docs under
  the `dev` label:

  ```
  mike deploy --push --update-aliases dev
  ```

  (`mkdocs build --strict` first, so a broken reference still fails the job.)

- On a release (the `v*` tag created by `pypi_release.yml`, or
  `on: release: types: [published]`): deploy the tagged commit's docs under the
  version number and move the `latest` alias to it:

  ```
  mike deploy --push --update-aliases 0.9.0 latest
  mike set-default --push latest
  ```

  Only the latest release needs the `latest` alias; older versions remain
  browsable under their numbers. Patch releases can reuse a `0.9` label
  (`mike deploy 0.9 latest`) so the list does not grow with every patch.

**Operational notes.** `mike` needs the full `gh-pages` history
(`actions/checkout@v4` with `fetch-depth: 0`) and a git identity in the job; the
existing `permissions: contents: write` is sufficient. The first adoption step
is a one-off `mike deploy 0.8.0 latest` from the `v0.8.0` tag so the default
view matches PyPI, after which the `main` banner can be reduced to the version
selector plus a short "development version" note that mkdocs-material shows
automatically for non-default versions (`version.default` + the
`outdated`/`development` banner in Material >= 9.5). The docs index banner
should then read the version from the selector rather than from `pyproject.toml`.
