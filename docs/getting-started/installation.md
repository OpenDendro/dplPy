# Installation

## Requirements

dplPy requires **Python 3.10 or newer**. Its core scientific dependencies
(NumPy, pandas, SciPy, matplotlib, statsmodels, csaps) are installed
automatically.

## Install from PyPI

```bash
pip install dplpy
```

Then confirm the install:

```python
import dplpy as dpl
print(dpl.__version__)
```

## Optional extras

LiPD import/export is built on [`pylipd`](https://pypi.org/project/pylipd/),
which most users do not need, so it is an optional extra:

```bash
pip install "dplpy[lipd]"
```

## Install with conda / mamba

If you prefer a conda environment, create one with Python 3.10+ and install
dplPy into it with pip:

```bash
conda create -n dplpy python=3.11
conda activate dplpy
pip install dplpy
```

## Install from source

To work from the latest `main` (unstable) or to contribute:

```bash
git clone https://github.com/OpenDendro/dplPy.git
cd dplPy
pip install -e .
```

The project develops on `main`, with tagged pre/releases branched from it and
published to PyPI. See the
[repository](https://github.com/opendendro/dplpy) for the current version and
release notes.

## Next steps

Head to the [Quickstart](quickstart.md) for a complete worked example, or dive
into the [User Guide](../guide/reading-writing-data.md).
