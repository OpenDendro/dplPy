# Instructions for publishing dplPy to PyPI

The following instructions are for key contributors only. They describe how to
create a new release of dplPy, publish it to PyPI and a new GitHub tag/release,
deploy the documentation, and archive the release on Zenodo with a DOI.

The release workflow (`.github/workflows/pypi_release.yml`) derives the tag,
release name and release notes from the release **branch name** and
`CHANGELOG.md`, so **no workflow file is edited per release**.

## 0. Before you branch: verify the working tree

Releases are gated on the tests passing (the version branch will not publish if
`run_tests` fails), so confirm the release is green *first*:

- **Run the test suite** and make sure it passes:
  `python -m pytest tests/`
- **Build the docs strictly** so a broken link or reference does not ship:
  `mkdocs build --strict` (deps are pinned in `docs/requirements.txt`).
- **Build the package** once locally (`python -m build`) so a packaging error
  does not surface only in the release workflow.
- Make sure everything intended for the release is committed to `main`.

## 1. Update version information in main

In the `main` branch, update the version number in the following places. The
version string must be identical everywhere (the release workflow refuses to
publish if the branch name does not equal `v` + the `pyproject.toml` version).

**Version files (bump the number):**

- `src/dplpy/__init__.py` — update `__version__`.
- `pyproject.toml` — update `version` in the `[project]` section to match.
- `README.md` — in the "Current Version" section, update the stated version
  (e.g. `v0.8.0` → `v0.9.0`).
- `CITATION.cff` — update `version:` and set `date-released:` to the release date
  (YYYY-MM-DD). This is what the "Cite this repository" panel and Zenodo show.
  (The version DOI is added *after* the release; see step 4.)

**`CHANGELOG.md`** — turn the `[Unreleased]` section into the release:

1. Rename the heading `## [Unreleased]` to `## [X.Y.Z] - YYYY-MM-DD`.
2. At the bottom of the file, add the compare link for the new version
   (`[X.Y.Z]: https://github.com/OpenDendro/dplPy/compare/vPREV...vX.Y.Z`) and
   point the `[Unreleased]` link at `compare/vX.Y.Z...HEAD`.
3. Recreate an empty `## [Unreleased]` section above the new release heading
   (the standard `### Added` / `### Changed` / `### Fixed` / `### Removed`
   sub-headings can be added as items arrive).

The release workflow copies the `## [X.Y.Z]` section of `CHANGELOG.md` into the
GitHub release notes, so this section *is* the release announcement.

**`.github/workflows/pypi_release.yml`** — no change needed per release. It runs
on any branch matching `v[0-9]*` after `run_tests` succeeds, checks out the
tested commit, builds, publishes to PyPI, and creates the GitHub release and tag
named after the branch. A branch name with a suffix (`v1.0.0-rc1`) is published
as a GitHub *pre-release*.

**`.zenodo.json`** — no change needed per release: Zenodo takes the version from
the GitHub release tag. Update it only when the author list, affiliations, or
description change.

## 2. Branch to the new version branch

On GitHub, create a branch from `main` named `v` + the new version number (e.g.
`v0.9.0`). Creating the branch kicks off the workflows: `run_tests` → (on
success) build → publish to PyPI → create the GitHub (pre)release and tag. The
release is contingent on all unit and integration tests passing.

Check the "Publish to PyPI" run under the Actions tab. If the version check step
fails ("branch does not match pyproject version"), fix the version on `main`,
delete the branch, and recreate it.

### One-time setup: PyPI trusted publishing

The workflow publishes with [PyPI trusted publishing](https://docs.pypi.org/trusted-publishers/)
(OpenID Connect) instead of an API token, so there is no `DPLPY_PYPI` secret to
rotate. This must be configured **once** by a project owner on PyPI:
pypi.org → project `dplpy` → *Publishing* → *Add a new publisher* (GitHub):
owner `OpenDendro`, repository `dplPy`, workflow name `pypi_release.yml`,
environment name `pypi`. The workflow's `permissions: id-token: write` and
`environment: pypi` entries must stay in place for this to work. Optionally,
protect the `pypi` environment in the repository settings (Settings →
Environments) with required reviewers so a release needs an approval click.

## 3. Documentation deploys automatically

The docs site is built and deployed by `publish-docs.yml` to the `gh-pages`
branch on every push to `main`. GitHub Pages **Source must be set to
`gh-pages` / root** (Settings → Pages) — not `main`/`docs`, which would serve raw
Markdown through Jekyll. `publish-docs.yml` is a protected workflow file, so any
change to it is applied by a maintainer, not through the file bridge.

Because the site tracks `main`, the docs may describe functions that are not in
the released PyPI version; the docs index page carries a banner saying so. See
dev-instructions.md ("Versioned documentation with mike") for the option of
publishing `latest` and `dev` versions of the site side by side.

## 4. Zenodo archiving and the DOI

The GitHub–Zenodo hook is **enabled** for `OpenDendro/dplPy`, so creating the
GitHub release (step 2) automatically deposits the tagged release on Zenodo and
mints a **version DOI**; the first archived release also created the **concept
DOI** (`10.5281/zenodo.22695105`), which always resolves to the latest version.
Zenodo reads the deposit metadata from `.zenodo.json`.

Note: Zenodo only archives releases created **after** the hook was enabled;
earlier releases are not retroactively archived.

**If no Zenodo record appears within about an hour**, the webhook delivery
probably timed out (Zenodo's receiver downloads and processes the repo tarball
synchronously and sometimes does not answer inside GitHub's delivery window, and
this hook gives up after one attempt). Fix it without cutting a new release:
GitHub -> repo *Settings -> Webhooks ->* the `zenodo.org` hook *-> Recent
Deliveries*, open the failed `release` delivery whose payload `"action"` is
`"published"` (ignore the `release.edited` ones, which only update an existing
deposit), and click **Redeliver**. Confirm Zenodo is reachable first; redeliver
when it is. This happened for v0.9.0 (record 23422091).

After Zenodo has minted the version DOI (a few minutes after the release):

1. On Zenodo, copy the **version DOI** for the new release.
2. Add it to `CITATION.cff` under `identifiers:` as a new entry
   (`type: doi`, `value: 10.5281/zenodo.NNNNNNNN`,
   `description: "Version DOI (vX.Y.Z)"`), keeping the concept DOI as the
   primary `doi:`. Keep the earlier version DOIs listed.
3. Commit this to `main` (it will be picked up by the next release). The
   README DOI badge points at the concept DOI and does not change.

## Quick checklist

- [ ] Tests pass (`pytest`), docs build (`mkdocs build --strict`), package
      builds (`python -m build`).
- [ ] Version bumped in `__init__.py`, `pyproject.toml`, `README.md`,
      `CITATION.cff` (+ `date-released`).
- [ ] `CHANGELOG.md`: `[Unreleased]` renamed to `[X.Y.Z] - date`, compare link
      added, empty `[Unreleased]` recreated.
- [ ] Everything committed to `main`.
- [ ] Create the `vX.Y.Z` branch → tests → PyPI + GitHub release fire.
- [ ] Docs deployed to `gh-pages` (Pages source = `gh-pages`).
- [ ] Zenodo archived the release; add the **version DOI** to `CITATION.cff`.
