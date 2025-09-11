# Installation Guide

This guide provides step-by-step instructions for installing dplPy and setting up your development environment for dendrochronological analysis.

## Prerequisites

Before installing dplPy, ensure you have the following software installed:

- **Python 3.6 or higher** 
- **Conda** ([Anaconda](https://docs.anaconda.com/anaconda/install/index.html) or [Miniconda](https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/index.html))
- **Git** (for cloning the repository)

### Recommended Tools

- [Mamba](https://mamba.readthedocs.io/en/latest/installation.html) - A faster conda alternative
- [VSCode](https://code.visualstudio.com/) - Recommended IDE with excellent Python support

## Installation Methods

### Method 1: Conda Environment (Recommended)

This is the recommended installation method as it ensures all dependencies are properly managed.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/opendendro/dplpy.git
   cd dplpy
   ```

2. **Create the conda environment:**
   ```bash
   conda env create -f environment.yml
   ```

3. **Activate the environment:**
   ```bash
   conda activate dplpy
   ```

4. **Verify installation:**
   ```python
   import dplpy as dpl
   print(dpl.__version__)
   dpl.help()
   ```

### Method 2: Manual Installation

If you prefer to manage dependencies manually:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/opendendro/dplpy.git
   cd dplpy
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install in development mode:**
   ```bash
   pip install -e .
   ```

## Environment Setup

### Using Mamba (Faster Alternative)

If you have Mamba installed, you can use it instead of conda for faster environment creation:

```bash
mamba env create -f environment.yml
mamba activate dplpy
```

### Environment File Contents

The `environment.yml` file includes all necessary dependencies for dplPy:

- **Core packages**: pandas, numpy, matplotlib, scipy
- **Statistical packages**: statsmodels, scikit-learn  
- **Dendrochronology-specific**: csaps (for smoothing splines)
- **Development tools**: jupyter, pytest (for testing)

## Verification

### Test Basic Functionality

After installation, verify that dplPy is working correctly:

```python
import dplpy as dpl
import pandas as pd

# Test data reading
data = dpl.readers('tests/data/csv/ca533.csv')
print(f"Data loaded successfully: {data.shape}")

# Test basic statistics
stats = dpl.stats(data)
print("Statistics calculated successfully")

# Test plotting
dpl.plot(data.iloc[:, :5], plot=True)  # Plot first 5 series
```

### Run Unit Tests

To ensure all functionality is working:

```bash
# From the dplpy root directory
python -m pytest tests/
```

## Troubleshooting

### Common Issues

**1. Environment Creation Fails**
```bash
# Clear conda cache and try again
conda clean --all
conda env create -f environment.yml
```

**2. ImportError for specific packages**
```bash
# Install missing packages manually
conda activate dplpy
conda install <package-name>
```

**3. CSAPS Installation Issues**
```bash
# Install csaps separately
pip install csaps
```

**4. Jupyter Kernel Issues**
```bash
# Add dplpy environment to Jupyter
conda activate dplpy
python -m ipykernel install --user --name dplpy --display-name "Python (dplpy)"
```

### System-Specific Notes

**Windows Users:**
- Use Anaconda Prompt or PowerShell
- Ensure Git is properly installed and in PATH
- Consider using WSL2 for Linux-like environment

**macOS Users:**
- Install Xcode Command Line Tools: `xcode-select --install`
- Use Homebrew for Git if not already installed

**Linux Users:**
- Most distributions should work out of the box
- Ensure build tools are installed: `sudo apt-get install build-essential` (Ubuntu/Debian)

## Development Installation

For developers who want to contribute to dplPy:

1. **Fork the repository** on GitHub

2. **Clone your fork:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/dplpy.git
   cd dplpy
   ```

3. **Create development environment:**
   ```bash
   conda env create -f environment.yml
   conda activate dplpy
   ```

4. **Install in development mode:**
   ```bash
   pip install -e .
   ```

5. **Install additional development tools:**
   ```bash
   conda install pytest pytest-cov black flake8 mypy
   ```

## Updating dplPy

To update your dplPy installation:

```bash
# Navigate to dplpy directory
cd path/to/dplpy

# Pull latest changes
git pull origin main

# Update environment
conda env update -f environment.yml

# Reactivate environment
conda activate dplpy
```

## Uninstalling

To completely remove dplPy:

```bash
# Remove conda environment
conda env remove -n dplpy

# Remove cloned repository
rm -rf path/to/dplpy
```

## Getting Help

If you encounter installation issues:

1. Check the [GitHub Issues](https://github.com/opendendro/dplpy/issues) page
2. Create a new issue with:
   - Your operating system and version
   - Python and conda versions
   - Complete error messages
   - Steps you've already tried

## Next Steps

After successful installation:

1. Read the [Quick Start Guide](quickstart.md)
2. Try the [Basic Workflow Tutorial](tutorials/basic_workflow.md)
3. Explore the example Jupyter notebook: `runnable_example.ipynb`
4. Check out the [API Reference](api/index.md) for detailed function documentation