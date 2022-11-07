# dplPy
The Dendrochronology Program Library for Python

---

## Index

- [Issues](#issues)
- [Requirements](#requirements)
- [Building Environment](#building-environment)
- [Using jupyter](#using-jupyter)
    - [Linux, macOS](#linux-macos)
    - [Windows](#windows)
- [Functionality and usage](#functionalities-and-usage)
    - [Loading data](#loading-data)
    - [Data Summary](#data-summary)
    - [Data Stastics](#data-stastics)
    - [Data Report](#data-report)
    - [Plotting](#plotting)
    - [Autoregressive (AR) modeling](#autoregressive-ar-modeling)
    - [Detrending](#detrending)
    - [Chronology](#chronology)

---

## Issues

We're using [ZenHub](https://app.zenhub.com/workspaces/opendendro-60ec698d8790d700171ceee8/board?repos=385244315) to manage our [GitHub Issues](https://github.com/opendendro/dplpy/issues)

---

## Requirements

**Note**: DplPy has been successfully tested on Ubuntu 20, Ubuntu 22, macOS (Intel, M2).

- Conda ([Anaconda](https://docs.anaconda.com/anaconda/install/index.html) or [Miniconda](https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/index.html))
- (Suggested) [Mamba](https://mamba.readthedocs.io/en/latest/installation.html)
- (Suggested) [VSCode](https://code.visualstudio.com/)

## Building Environment

> :warning: **it is recommended to _NOT_ use GitHub Codespaces (as of Mar 2022)**

0\. Clone and change directory to this repository

```
$ git clone https://github.com/OpenDendro/dplPy.git
$ cd dplPy
```

1\. Create a conda environment through the `environment.yml` file. This will ensure all packages required are installed.

```
$ conda env create -f environment.yml     

# if you have mamba installed you could instead do

$ mamba env create -f environment.yml
```

When prompted for permission to install required packages (with `y/n`), select `y`.

2\. Activate your environment:

```
$ conda activate dplpy
```

Your environment should be successfully built.

3\. Your python environment should be able to import `numpy`, `pandas`, `matplotlib`, `statsmodels` and `csaps`:

![env_3](docs/assets/env_3.png)

---

## Using Jupyter

The Conda enviroment is essential as it provides will all necessary packages. To execute the code, use Jupyter Notebook.

**Note**: if using Jupyter from the terminal, you need to ensure that the kernel is findable by doing the following command once the environment is active:

```
python -m ipykernel install --user --name dplpy --display-name "Python (dplpy)"
```

### Linux, MacOS

1\. In your VSCode terminal, activate the conda environment with `conda activate dplpy3`.
2\. Open a Jupyer Notebook (`<file>.ipynb`) and select the `dplpy3` Kernel when prompted (or from the top right of your screen).
This will automatically load the environment we created.

### Windows

In VSCode:

1\. In your VSCode terminal window, activate the conda environment with `conda activate dplpy3`.
2\. In the same terminal window, start a Jupyter Notebook with `jupyter notebook`. Jupyter will then return URLs that you can copy; *Copy* one of these URLs. 

![ipynb_env1](docs/assets/ipynb_env1.jpg)

3\. Open a Jupyter Notebook (`<file>.ipynb`) and from the **bottom right** of the VSCode screen, click **Jupyter Server**;

![ipynb_env2](docs/assets/ipynb_env2.jpg) 

A dropdown menu will open from the top of the screen: select Existing and *paste* the URL you copied.

![ipynb_env3](docs/assets/ipynb_env3.jpg)

4\. Jupyter Notebook will now be able to access the environment created.

---

## Functionalities and Usage

Import the DplPy tool with

```
import dplpy as dpl
```

This will load the necessary functions.

### Loading data

- Description: reads data from supported file types (`csv` and `rwl`) and stores them in a dataframe.
- Options: 
    - `header`: input files often have a header present; Default is `False`, use `True` if input has a header.
- Usage example:
    ```
    >>> data = dpl.readers("/path/to/file.rwl", header=True)
    ```

### Data Summary

- Description: generates a summary of each series recorded in `rwl`  and `csv` format files
- Usage Example:
    ```
    >>> dpl.summary("/path/to/file.rwl")
    # or
    >>> dpl.summary(data)
    ```

### Data Stastics

- Description: generates summary statistics for `rwl`  and `csv` format files
- Usage Example:
    ```
    >>> dpl.stats("/path/to/file.rwl")
    # or
    >>> dpl.stats(data)
    ```

### Data Report

- Description: generates a report about absent rings in the data set
- Usage Example:
    ```
    >>> dpl.report("/path/to/file.rwl")
    # or
    >>> dpl.report(data)
    ```

### Plotting

- Description: generates plots of tree ring with data from dataframes. Currently capable of generating `line` (default), `spag` (spaghetti) and `seg` (segment) plots.
- Options:
    - `type="line"`: creates a line plot (default)
    - `type="spag"`: creates a spaghetti plot
    - `type="seg"`: creates a segment plot
- Usage Example:
    ```
    >>> dpl.report("/path/to/file.rwl")
    # or 
    >>> dpl.plot(data)

    # User is able to select specific series of interests.
    # In the example below, the user selects SERIES_1, SERIES_2, SERIES_3 
    # from the "data" dataset and generates a spaghetti plot
    >>> dpl.plot(data[[SERIES_1, SERIES_2, SERIES_3]], type="spag")
    ```

### Autoregressive (AR) modeling 

- Description: ontains methods that fit series to autoregressive models and perform functions related to AR modeling.
- Functions:
    - `autoreg(data['Name of series'], max_lag)`: returns parameters of best fit AR model with maxlag of 5 (default) or other specified number
    - `ar_func(data['Name of series'], max_lag)`: returns residuals plus mean of best fit from AR models with max lag of either 5 (default) or specified number
- Options:
    - `max_lag`: default 5, can be specified to user's needs.
- Usage Example:
    ```
    >>> dpl.autoreg(data[SERIES_1])
    # or
    >>> dpl.ar_func(data[SERIES_2], max_lag=7)
    ```

### Detrending

- Description: Detrends a given series or data frame, first by fitting data to curve(s), and then by calculating residuals or differences compared to the original data.
- Options:
    - `fit="spline"`: default detrending method.
    - `fit="ModNegEx"`: detrending using negative exponent method.
    - `fit="Hugershoff"`: detrending using the Hugenshoff method.
    - `fit="linear"`: detrending using the linear method.
    - `fit="horizontal"`: detrending using the horizontal method.
    - `method="residual"`: calculates residuals vs original data (default).
    - `method="difference"`: calculates differences vs original data.
    - `Plot`: plotting results default is `True`, accepts `False`.
- Usage Example:
    ```
    >>> dpl.detrend(data)
    # or
    >>> dpl.detrend(data, fit="Hugershoff", method="difference")

    # User is able to select specific series of interests.
    >>> dpl.detrend(data[[SERIES_1, SERIES_2, SERIES_3]], fit="Hugershoff", method="difference")
    ```

### Chronology

- Description: creates a mean value chronology for a dataset, typically the ring width indices of a detrended series. **Note: input data has to be detrended first.**
- Options:
    - `biweight`: find means using Tukey's biweight robust mean; default `True`.
    - `prewhiten`: prewhitens data by fitting to an AR model; default `False`.
    - `plot`: plots results; default `True`.
- Usage Example:
    ```
    # Detrend data first!
    >>> rwi_data = dpl.detrend(data)

    # Perform chronology
    >>> dpl.chron(rwi_data, biweight=False, plot=False)
    ```