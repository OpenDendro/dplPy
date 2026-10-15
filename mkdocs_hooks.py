# MkDocs build hooks.
#
# 1. Copies the trusted example notebooks from notebooks/ into docs/examples/ at
#    build time, so the canonical copies stay in notebooks/ and the docs render
#    them (via mkdocs-jupyter) without a second tracked copy.
# 2. Substitutes the placeholder ``{{ dplpy_version }}`` in any page with the
#    version in pyproject.toml, so the "these docs track main" banner on the
#    index page names the release without being edited by hand.
import pathlib
import shutil
import tomllib

_NOTEBOOKS = [
    "readers_demonstration",
    "crossdating_demonstration",
    "rcs_demonstration",
]

_VERSION_PLACEHOLDER = "{{ dplpy_version }}"


def _project_root(config):
    return pathlib.Path(config["config_file_path"]).parent


def _package_version(root):
    try:
        with open(root / "pyproject.toml", "rb") as fh:
            return tomllib.load(fh)["project"]["version"]
    except Exception:                                   # pragma: no cover
        return "see PyPI"


def on_config(config):
    root = _project_root(config)
    src = root / "notebooks"
    dst = root / "docs" / "examples"
    dst.mkdir(parents=True, exist_ok=True)
    for name in _NOTEBOOKS:
        nb = src / (name + ".ipynb")
        if nb.exists():
            shutil.copy2(nb, dst / (name + ".ipynb"))
    config["extra"]["dplpy_version"] = _package_version(root)
    return config


def on_page_markdown(markdown, page, config, files):
    if _VERSION_PLACEHOLDER in markdown:
        markdown = markdown.replace(_VERSION_PLACEHOLDER,
                                    config["extra"]["dplpy_version"])
    return markdown
