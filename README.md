 <p align="center">
 <img src="https://github.com/OpenDendro/dplPy/blob/main/docs/assets/dplpy.png?raw=true" width="175"> 

# dplPy -the Dendrochronology Program Library in Python
The Dendrochronology Program Library (DPL) in Python has its roots in both the [original FORTRAN program](https://www.ltrr.arizona.edu/software.html) created by the [legendary Richard Holmes](https://arizona.aws.openrepository.com/handle/10150/262569?show=full) and the subsequent R Project package by Andy Bunn, [dplR](https://github.com/OpenDendro/dplR).  Our aim is to provide researchers working with tree-ring data the necessary tools in open-source environments, promoting open science practices, enhancing rigor and transparency in dendrochronology, and eventually allowing reproducible research entirely in a single programming language.

 The development of dplPy is supported by a grant from the Paleoclimate program of the US National Science Foundation (AGS-2054516) to Andy Bunn, Kevin Anchukaitis, Ed Cook, and Tyson Swetnam.
<br>


---


## Index

- [dplPy - the Dendrochronology Program Library in Python](#dplpy---the-dendrochronology-program-library-in-python)
  - [Index](#index)
  - [Requirements](#requirements)
  - [Current Version and Changelog](#current-version-and-changelog)
  - [Installation](#installation)
  - [Building directly from Github](#building-directly-from-github)
  - [Using VSCode in your operating system](#using-vscode-in-your-operating-system)
    - [Linux or MacOS](#linux-or-macos)
    - [Windows](#windows)
  - [Functionalities and Usage](#functionalities-and-usage)
    - [Loading data using  `readers`](#loading-data-using--readers)
    - [Loading data from online sources using `readers_url`](#loading-data-from-online-sources-using-readers_url)
    - [Data Summary from `summary`](#data-summary-from-summary)
    - [Data Stastics from `stats`](#data-stastics-from-stats)
    - [Data Report from `report`](#data-report-from-report)
    - [Plotting raw data with `plot`](#plotting-raw-data-with-plot)
    - [Detrending using `detrend`](#detrending-using-detrend)
    - [Autoregressive (AR) modeling](#autoregressive-ar-modeling)
    - [Build a chronology with `chron`](#build-a-chronology-with-chron)
    - [Build a variance stabilized chronology with `chron_stabilized`](#build-a-variance-stabilized-chronology-with-chron_stabilized)
    - [Crossdate with `xdate`](#crossdate-with-xdate)
    - [Output data to files using `writers`](#output-data-to-files-using-writers)

---

## Requirements

- Python (>=3.10)
- Conda ([Anaconda](https://docs.anaconda.com/anaconda/install/index.html) or [Miniconda](https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/index.html)), or [Pip](https://pip.pypa.io/en/stable/installation/)
- (Suggested) [Mamba](https://mamba.readthedocs.io/en/latest/installation.html)
- (Suggested) [VSCode](https://code.visualstudio.com/)

Under the hood, dplPy uses `numpy`, `pandas`, `matplotlib`, `statsmodels`, `scipy`, and `csaps`.

:warning: dplPy has been successfully tested thus far on Ubuntu 20, Ubuntu 22, macOS (Intel and M2). Other operating systems may experience unexpected errors or conflicts.  Please let the developers know. 

## Current Version and Changelog

dplPy is currently at version `v0.1.5` - The project has changing to a new development structure where all development will be on `main` and releases and updates to [Pypi](https://pypi.org/project/dplpy/) will be branched to a version number and deployed.

## Installation

dplPy is now available to [install via pip](https://pypi.org/project/dplpy/):

```
pip install dplpy
```

To ensure you have the latest version of dplPy installed, you can run:

```
pip install dplpy --upgrade
```


You can install a conda virtual environment using the [environment.yml for the project](https://github.com/OpenDendro/dplPy/blob/main/environment.yml):

```
$ conda env create -f environment.yml     
```

---


## Building directly from Github

You can still still install dplPy firectly from Github if you wish:

1\. Clone and change directory to this repository


```
$ git clone https://github.com/OpenDendro/dplPy.git
$ cd dplPy
```

2\. Create a conda environment through the `environment.yml` file. This will ensure all packages required are installed.

```
$ conda env create -f environment.yml     

# if you have mamba installed you could instead do

$ mamba env create -f environment.yml
```

When prompted for permission to install required packages (with `y/n`), select `y`.

3\. Activate your environment:

```
$ conda activate dplpy
```

Your environment should be successfully built.

4\. Your python environment should be able to import `numpy`, `pandas`, `matplotlib`, `statsmodels` and `csaps`.

---

## Using VSCode in your operating system

### Linux or MacOS

Note: The instructions in this section assume the conda environment where you have dplpy and its dependencies installed is named `dplpy`

1\. In your VSCode terminal, activate the conda environment with `conda activate dplpy`.

2\. Open a Jupyer Notebook (`<file>.ipynb`) and select the `dplpy` Kernel when prompted (or from the top right of your screen). This will automatically load the environment we created.

### Windows

In VSCode:

1\. In your VSCode terminal window, activate the conda environment with `conda activate dplpy`.

2\. In the same terminal window, start a Jupyter Notebook with `jupyter notebook`. Jupyter will then return URLs that you can copy; *Copy* one of these URLs. 

3\. Open a Jupyter Notebook (`<file>.ipynb`) and from the **bottom right** of the VSCode screen, click **Jupyter Server**;

![ipynb_env2](docs/assets/ipynb_env2.jpg) 

A dropdown menu will open from the top of the screen: select Existing and *paste* the URL you copied.

![ipynb_env3](docs/assets/ipynb_env3.jpg)

4\. Jupyter Notebook will now be able to access the environment created.

---

## Functionalities and Usage

Import the dplPy tool with
```
import dplpy 
```
or to import with an alias (we will use `dpl`):

```
import dplpy as dpl
```
  
This will load the package and its functions, allowing them to be accessed with the package name or alias given.


### Loading data using  `readers`

- Description: reads data from supported file types (`csv` and `rwl`) and stores them in a dataframe.
- Options: 
    - `header`: rwl input files often have a header present; Default is `False`, use `True` if input has a header.
- Usage examples:
    ```
    >>> data = dpl.readers("/path/to/file.csv")
    # or
    >>> data = dpl.readers("/path/to/file.rwl", header=True)
    ```

### Loading data from online sources using  `readers_url`
**Note: This function is still in development and has only been tested so far with `rwl` raw data files from the [NCEI website](https://www.ncei.noaa.gov/pub/data/paleo/treering/measurements/)**

- Description: reads `rwl` formatted data directly from online sources.
- Options: 
    - `header`: rwl input files often have a header present; Default is `False`, use `True` if input has a header.
- Usage examples:
    ```
    >>> data = dpl.readers_url("http://link/to/file.rwl")
    >>> data = dpl.readers_url("http://link/to/file.rwl", header=True)
    ```

### Data Summary from `summary`

- Description: generates a summary of each series recorded in `rwl`  and `csv` format files
- Usage examples:
    ```
    >>> dpl.summary("/path/to/file.rwl")
    # or
    >>> dpl.summary(data)
    ```

### Data Stastics from `stats`

- Description: generates summary statistics for `rwl`  and `csv` format files
- Usage Example:
    ```
    >>> dpl.stats("/path/to/file.rwl")
    # or
    >>> dpl.stats(data)
    ```

### Data Report from `report`

- Description: generates a report about ring measurements and absent rings in the data set
- Usage Example:
    ```
    >>> dpl.report("/path/to/file.rwl")
    # or
    >>> dpl.report(data)
    ```

### Plotting raw data with `plot`

- Description: generates plots of tree ring with data from dataframes. Currently capable of generating `line`, `spag` (spaghetti) and `seg` (segment, default) plots.
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

### Detrending using `detrend`
 
- Description: Detrends a given series or data frame, first by fitting data to curve(s), and then by calculating residuals or differences compared to the original data.
- Options:
    - `fit="spline"`: default detrending method.
    - `fit="ModNegEx"`: detrending using negative exponent method.
    - `fit="Hugershoff"`: detrending using the Hugenshoff method.
    - `fit="linear"`: detrending using the linear method.
    - `fit="horizontal"`: detrending using the horizontal method.
    - `method="residual"`: calculates residuals vs original data (default).
    - `method="difference"`: calculates differences vs original data.
    - `plot=True|False`: whether or not to plot results, default is `True`.
- Usage Example:
    ```
    # detrend with default options
    >>> dpl.detrend(data)
    
    # specify fit to hugershoff curve and detrend with difference
    >>> dpl.detrend(data, fit="Hugershoff", method="difference")

    # detrend only SERIES_1, SERIES_2 and SERIES_3
    >>> dpl.detrend(data[[SERIES_1, SERIES_2, SERIES_3]], fit="Hugershoff", method="difference")
    ```


### Autoregressive (AR) modeling 

- Description: Contains methods that fit series to autoregressive models and perform functions related to AR modeling.
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

### Build a chronology with `chron`

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

### Build a variance stabilized chronology with `chron_stabilized`

- Description: Builds a variance stabilized mean-value chronology for a dataset of **detrended** ring width indices, by multiplying the chronology with the square root of the effective independent sample size, $ Neff $.

    Note: where n(t) is the number of series at time t, and rbar is the running interseries correlation, 

    $$ Neff = { n(t) \over 1+(n(t)-1)rbar(t) } $$

- Options:
    - `win_length`: an integer for specifying the window lengths where interseries correlations will be calculated (default `50`). Should not be greater than the number of years in the dataset, recommended to be between 30% and 50% of the number of years.
    - `min_seg_ratio`: the minimum ratio of non-NA values to the window length for a series to be considered in an Neff calculation (default `0.33`).
    - `biweight`: boolean indicating whether or not to use Tukey's bi-weight robust mean when calculating the mean-value chronology; default `True`.
    - `running_rbar`: boolean indicating whether or not to return the running interseries correlations as part of chronology output; default `False`.
- Usage Example:
    ```
    # Detrend data first!
    >>> rwi_data = dpl.detrend(data)

    # Perform chronology with default args
    >>> dpl.chron_stabilized(rwi_data)

    # Specify win_length, min_seg_ratio and running_rbar
    >>> dpl.chron_stabilized(rwi_data, win_length=60, min_seg_ratio=0.5, running_rbar=True)
    ```

### Crossdate with `xdate`
- Description: This function calculates correlation serially between each tree-ring series and a master chronology built from all the other series in the dataset (leave-one-out principle).
- Options:
    - `prewhiten`: default `True`, determines whether or not to prewhiten series using AR modeling
    - `corr`: default `'Spearman'`, the type of correlation to use. Can be `'Pearson'` or `'Spearman'`.
    - `slide_period`: default `50`, the number of years to compare to the master chronology at a time.
    - `bin_floor`: default `100`, determines the minimum bin year. The minimum bin year is calculated as $ \lceil (min\_yr/bin\_floor)\rceil*bin.floor $ where `min_yr` is the first year in the dataset.
    - `p_val`: default `0.05`, determines the critical value below which interseries correlations are flagged.
    - `show_flags`: default `True`, determines whether to show flags in the function output to the console.
- Usage examples:
    ```
    >>> ca533_rwi = dpl.detrend(ca533, plot=False)

    # Crossdating of detrended data with default args
    >>> dpl.xdate(ca533_rwi)

    # Crossdating with Pearson correlation and show flags 
    # (other options set to defaults when not specified).
    >>> dpl.xdate(ca533_rwi, corr="Pearson" show_flags=True)
    ```

### Output data to files using  `writers`

- Description: writes data from dataframe to supported file types (`csv`, `rwl`, `crn`, `txt`).
- Required parameters: 
    - `data`: dataframe with ring widths (presumably one read from `readers` or `readers_url`)
    - `label`: name (can include file path) to give to the created file. **should not include file extension**
    - `format`: extension for file to be created. Can be `'csv'`, `'rwl'`, `'crn'` or `'txt'`.

- Usage examples:
    ```
    # Write data to file_name.csv in current working directory.
    >>> dpl.writers(data, "file_name", "csv")

    # Write data to file_name.csv in ./path/to/ directory.
    >>> dpl.writers(data, "./path/to/file_name", "csv")
    ```