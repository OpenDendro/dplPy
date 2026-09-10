# Instructions for publishing dplPy to PyPI

The following instructions are for key contributors only. They describe how to
create a new release of dplPy, publish it to PyPI and a new GitHub tag, deploy
the documentation, and archive the release on Zenodo with a DOI.

## 0. Before you branch: verify the working tree

Releases are gated on the tests passing (the version branch will not publish if
`run_tests` fails), so confirm the release is green *first*:

- **Run the test suite** and make sure it passes:
  `python -m pytest tests/`
- **Build the docs strictly** so a broken link or reference does not ship:
  `mkdocs build --strict` (deps are pinned in `docs/requirements.txt`).
- Make sure everything intended for the release is committed to `main`.

## 1. Update version information in main

In the `main` branch, update the version number in the following places. The
version string must be identical everywhere.

**Version files (bump the number):**

- `src/dplpy/__init__.py` — update `__version__`.
- `pyproject.toml` — update `version` in the `[project]` section to match.
- `README.md` — in the "Current Version" section, update the stated version
  (e.g. `v0.5.0` → `v0.6.0`).
- `CITATION.cff` — update `version:` and set `date-released:` to the release date
  (YYYY-MM-DD). This is what the "Cite this repository" panel and Zenodo show.

**`.github/workflows/pypi_release.yml`** (protected; edit via the GitHub web
editor or locally — the file bridge cannot write it). Make **4 changes**, all to
`v` + the new version number:

1. `on.workflow_run.branches` — the branch the workflow runs on.
2. `steps` → "Checkout source" → `with.ref` — check out that same branch.
3. "Create Release" → `tag_name` **and** `release_name`.
4. "Create Release" → `body` — the changelog/release notes. `draft` and
   `prerelease` there control whether it is a full release or a prerelease.

**`.zenodo.json`** — no change needed per release: Zenodo takes the version from
the GitHub release tag. Update it only when the author list, affiliations, or
description change.

## 2. Branch to the new version branch

On GitHub, create a branch from `main` named `v` + the new version number (e.g.
`v0.6.0`). Creating the branch kicks off the workflows: `run_tests` → (on
success) build → publish to PyPI → create the GitHub (pre)release. The release is
contingent on all unit and integration tests passing.

## 3. Documentation deploys automatically

The docs site is built and deployed by `publish-docs.yml` to the `gh-pages`
branch. GitHub Pages **Source must be set to `gh-pages` / root** (Settings →
Pages) — not `main`/`docs`, which would serve raw Markdown through Jekyll.
`publish-docs.yml` is a protected workflow file, so any change to it is applied by
a maintainer, not through the file bridge.

## 4. Zenodo archiving and the DOI

The GitHub–Zenodo hook is **enabled** for `OpenDendro/dplPy`, so creating the
GitHub release (step 2) automatically deposits the tagged release on Zenodo and
mints a **version DOI**; the first archived release also creates a **concept
DOI** that always resolves to the latest version. Zenodo reads the deposit
metadata from `.zenodo.json`.

Note: Zenodo only archives releases created **after** the hook was enabled;
earlier releases are not retroactively archived.

After the release is published:

1. On Zenodo, copy the **concept DOI** (the "all versions" DOI).
2. Add it to `CITATION.cff` — fill in the `doi:` and `identifiers:` block (the
   stub is already in the file).
3. Add the concept-DOI badge to `README.md`.
4. Commit these to `main` (they will be picked up by the next release).

## Quick checklist

- [ ] Tests pass (`pytest`) and docs build (`mkdocs build --strict`).
- [ ] Version bumped in `__init__.py`, `pyproject.toml`, `README.md`,
      `CITATION.cff` (+ `date-released`).
- [ ] `pypi_release.yml` updated (branch, ref, tag_name, release_name, body;
      prerelease flag as intended).
- [ ] Everything committed to `main`.
- [ ] Create the `vX.Y.Z` branch → PyPI + GitHub release fire.
- [ ] Docs deployed to `gh-pages` (Pages source = `gh-pages`).
- [ ] Zenodo archived the release; add the concept DOI to `CITATION.cff` +
      README badge.
