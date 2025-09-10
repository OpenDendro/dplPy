# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

dplPy is the Dendrochronology Program Library for Python, a scientific computing package for tree ring width time series analyses. It provides functionality for reading dendrochronological data, statistical analysis, detrending, chronology building, and cross-dating.

## Environment Setup

The project uses conda for environment management:

```bash
# Create environment from YAML
conda env create -f environment.yml
# OR with mamba (faster)
mamba env create -f environment.yml

# Activate environment
conda activate dplpy

# Install kernel for Jupyter
python -m ipykernel install --user --name dplpy --display-name "Python (dplpy)"
```

## Key Dependencies

- **Core scientific**: numpy, pandas, scipy, statsmodels, matplotlib
- **Specialized**: csaps (cubic smoothing splines)
- **Development**: jupyter, jupyterlab, notebook
- **Data parsing**: bs4, wget

## Architecture

### Package Structure
- `src/` - Main source code directory containing all modules
- `src/dplpy.py` - Main entry point with help system and imports
- `src/__init__.py` - Package initialization
- Individual modules for specific functionality:
  - `readers.py` - File I/O for RWL and CSV formats
  - `writers.py` - Data export functionality
  - `detrend.py` - Detrending methods (spline, ModNegEx, Hugershoff, linear, horizontal)
  - `chron.py` - Chronology building with biweight mean
  - `autoreg.py` - Autoregressive modeling
  - `xdate.py` - Cross-dating functionality
  - `plot.py` - Visualization (line, spaghetti, segment plots)
  - `stats.py`, `summary.py`, `report.py` - Statistical analysis
  - `series_corr.py` - Series correlation analysis

### Data Format Support
- **RWL files**: Tucson format ring-width files (standard dendrochronology format)
- **CSV files**: Comma-delimited versions of tree ring data
- Test data available in `tests/data/` with examples from ITRDB

### Core Workflow Pattern
1. Load data: `dpl.readers("/path/to/file.rwl")`
2. Analyze: `dpl.summary()`, `dpl.stats()`, `dpl.report()`
3. Detrend: `dpl.detrend(data, fit="spline", method="residual")`
4. Build chronology: `dpl.chron(detrended_data, biweight=True)`
5. Visualize: `dpl.plot(data, type="line"/"spag"/"seg")`

## Testing

Testing is done with individual Python test files in `src/`:
- `test_chron.py` - Tests chronology functionality
- `test_xdate.py` - Tests cross-dating
- `testtbrm.py` - Tests TBRM functionality
- Run tests individually: `python src/test_chron.py`

## Installation and Building

Install in development mode:
```bash
pip install -e .
```

The package uses setuptools with configuration in:
- `setup.py` - Main setup script
- `setup.cfg` - Setup configuration
- `pyproject.toml` - Build system requirements

## Import Pattern

Standard usage pattern:
```python
import dplpy as dpl
# Access functions as dpl.readers(), dpl.detrend(), etc.
```

## Important Notes

- Project is conda-first (not pip-first) - use environment.yml for setup
- No formal test framework (pytest/unittest) - uses custom test files
- Scientific computing focused - heavy use of numpy/pandas/matplotlib
- Supports both programmatic and interactive (Jupyter) usage
- Test data includes real dendrochronology datasets from ITRDB