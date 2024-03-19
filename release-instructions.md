# Instructions for publishing dplpy to PyPI

The following instructions are for key contributors only, and they describe the process of creating a new release for dplpy and publishing it to pypi and to a newly created github tag.

## 1. Updating version information in main.
In the main branch of the dplpy github repository, update the following files in the indicated fields:

- In the `__init__.py` file in the `dplpy` folder, update `__version__` to the new version number.
- In the `pyproject.toml` file, update the version number in the `[project]` section to match the new version number set in `__init__.py`.
- In the `.github/workflows` folder, find `pypi_release.yml`, where you'll make **4 total changes**.  First, update the branch name for the workflow to run in to v + the new version number (i.e., v0.1.1 if the version number is 0.1.1).
- Still in `pypi_release.yml`, ensure that the first step of the workflow uses code from the branch name you just set the workflow to run in (which should be named v + the version number).
- Still in `pypi_release.yml`, update the name of the tag being created in the create release step to v + the new version number, and change the release name to the new version number. Finally, update the description of the git release using the `body` field located under the tag name and release name. You can also set the release as a draft or prerelease using the remaining fields.

## 2. Branch to new version branch
On github, create a branch from main and call it v + the new version number. This should kickstart all the necessary workflows to package and release the current state of dplpy to git and PyPI. Note that the releases are contingent on the success of all the unit and integration tests.
