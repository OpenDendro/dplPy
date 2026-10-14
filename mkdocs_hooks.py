# MkDocs build hook. Copies the trusted example notebooks from notebooks/ into
# docs/examples/ at build time, so the canonical copies stay in notebooks/ and
# the docs render them (via mkdocs-jupyter) without a second tracked copy.
import shutil
import pathlib

_NOTEBOOKS = [
    "readers_demonstration",
    "crossdating_demonstration",
    "rcs_demonstration",
]


def on_config(config):
    root = pathlib.Path(config["config_file_path"]).parent
    src = root / "notebooks"
    dst = root / "docs" / "examples"
    dst.mkdir(parents=True, exist_ok=True)
    for name in _NOTEBOOKS:
        nb = src / (name + ".ipynb")
        if nb.exists():
            shutil.copy2(nb, dst / (name + ".ipynb"))
    return config
