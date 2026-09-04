 <p align="center">
 <img src="https://github.com/OpenDendro/dplPy/blob/main/docs/assets/dplpy.png?raw=true" width="175"> 

# dplPy -the Dendrochronology Program Library in Python
The Dendrochronology Program Library (DPL) in Python has its roots in both the [original FORTRAN program](https://www.ltrr.arizona.edu/software.html) created by the [legendary Richard Holmes](https://repository.arizona.edu/items/e7703eeb-adca-43c8-926e-daf18f86b654) and the subsequent R Project package by Andy Bunn, [dplR](https://github.com/OpenDendro/dplR).  Our aim is to provide researchers working with tree-ring data the necessary tools in open-source environments, promoting open science practices, enhancing rigor and transparency in dendrochronology, and eventually allowing reproducible research entirely in a single programming language.

 The development of dplPy is supported by a grant from the Paleoclimate program of the US National Science Foundation (AGS-2054516) to Andy Bunn, Kevin Anchukaitis, Ed Cook, and Tyson Swetnam.
<br>


---


## Index

- [dplPy - the Dendrochronology Program Library in Python](#dplpy---the-dendrochronology-program-library-in-python)
  - [Index](#index)
  - [Requirements](#requirements)
  - [Current Version](#current-version-and-changelog)
  - [Installation](#installation)
  - [Building directly from Github](#building-directly-from-github)
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
    - [Build an AR-based chronology with `chron_ars`](#build-an-ar-based-chronology-with-chron_ars)
    - [Crossdate with `xdate`](#crossdate-with-xdate)
    - [Date a floating series with `xdate_floater`](#date-a-floating-series-with-xdate_floater)
    - [Regional Curve Standardization with `rcs`](#regional-curve-standardization-with-rcs)
    - [Read chronology files with `read_crn`](#read-chronology-files-with-read_crn)
    - [Agreement statistics: `glk` and `sgc`](#agreement-statistics-glk-and-sgc)
    - [Mean sensitivity: `sens1` and `sens2`](#mean-sensitivity-sens1-and-sens2)
    - [Average cores to trees with `tree_mean`](#average-cores-to-trees-with-tree_mean)
    - [Basal area increment with `bai_out` and `bai_in`](#basal-area-increment-with-bai_out-and-bai_in)
    - [Combine datasets with `combine_rwl`](#combine-datasets-with-combine_rwl)
    - [Export and import LiPD with `to_lipd` and `from_lipd`](#export-and-import-lipd-with-to_lipd-and-from_lipd)
    - [Other functions](#other-functions)
    - [Output data to files using `writers`](#output-data-to-files-using-writers)

---

## Requirements

- Python (>=3.10)
- Conda ([Anaconda](https://docs.anaconda.com/anaconda/install/index.html) or [Miniconda](https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/index.html)), or [Pip](https://pip.pypa.io/en/stable/installation/)

Under the hood, dplPy uses `numpy`, `pandas`, `matplotlib`, `statsmodels`, `scipy`, and `csaps`. LiPD import/export (`to_lipd`/`from_lipd`) additionally requires the optional [`pylipd`](https://pypi.org/project/pylipd/) package (`pip install pylipd`); it is not needed for any other functionality.

dplPy has been successfully tested thus far on Ubuntu 20, Ubuntu 22, macOS (Intel and M2). Other operating systems may experience unexpected errors or conflicts.  Please let the developers know. 

## Current Version

dplPy is currently at version `v0.4.0` - The project uses a unusual development structure where all development will be on `main` (and therefore main is unstable) and then pre/releases and updates to [Pypi](https://pypi.org/project/dplpy/) are first branched to a version number branch and then deployed (an action that is triggered by the creation of the branch).

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

- Description: reads data from supported file types (`csv` and `rwl`) and stores them in a year-indexed dataframe (one column per series).
- Options:
    - `header`: whether the `rwl` file has header lines. Default is `None`, which **auto-detects** the number of header lines; pass `header=True`/`False` to force it, or `skip_lines=N` to skip a known number of lines.
    - `on_error`: `"raise"` (default) stops on a malformed record; `"warn"` salvages what it can and reports problems.  Users should be very careful using salvage mode. 
- Usage examples:
    ```
    >>> data = dpl.readers("/path/to/file.csv")
    # or (header lines auto-detected)
    >>> data = dpl.readers("/path/to/file.rwl")
    # force header handling, or salvage a messy file
    >>> data = dpl.readers("/path/to/file.rwl", header=True)
    >>> data = dpl.readers("/path/to/file.rwl", on_error="warn")
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

- Description: generates plots of tree-ring data from dataframes. Capable of `seg` (segment, the default), `spag` (spaghetti) and `line` plots.
- Options:
    - `type="seg"`: segment/coverage plot, one bar per series (default)
    - `type="spag"`: spaghetti plot; `color=` takes a colormap name (e.g. `"viridis"`, `"turbo"`) to shade series by first year, or any single colour (default `"black"`)
    - `type="line"`: overplot every series against year
    - `ax=`: draw into an existing matplotlib Axes; `show=False`: return the figure without displaying it (so you can save it). `plot()` returns `(fig, ax)`.
- Usage Example:
    ```
    >>> dpl.plot(data)                     # segment plot (default)
    >>> dpl.plot(data, type="spag")        # spaghetti (black)
    >>> dpl.plot(data, type="spag", color="viridis")   # shade by first year

    # Select specific series of interest (SERIES_1, SERIES_2, SERIES_3):
    >>> dpl.plot(data[["SERIES_1", "SERIES_2", "SERIES_3"]], type="spag")

    # Keep the figure to save it:
    >>> fig, ax = dpl.plot(data, type="seg", show=False)
    >>> fig.savefig("segments.png", dpi=300)
    ```

### Detrending using `detrend`

- Description: Detrends a given series or dataframe, first by fitting a growth curve (`fit`), then by forming the ring-width index as a ratio or difference of the data to the curve (`method`).
- **`fit` chooses the CURVE, `method` chooses the ARITHMETIC** (this differs from dplR, where `method` selects the curve). Names are case-insensitive; the canonical spellings are dplR's.
- Options:
    - `fit=` (the growth curve), one of:
        - `"Spline"` — smoothing spline (default)
        - `"AgeDepSpline"` — age-dependent smoothing spline
        - `"ModNegExp"` — modified negative exponential (with a linear→mean fallback)
        - `"ModHugershoff"` — Hugershoff curve (nonlinear least squares, dplR-style)
        - `"Hugershoff"` — Hugershoff curve (Cook/ARSTAN log-linearised closed form)
        - `"Linear"` — best-fit straight line
        - `"Mean"` — horizontal line at the series mean
        - `fit` may also be a **list** of curves (e.g. `["Spline", "ModNegExp"]`) to compare them.
    - `method="ratio"`: ring-width index = data ÷ curve (default; `"division"` is a synonym)
    - `method="difference"`: ring-width index = data − curve
    - `plot=True|False`: whether to plot results, default `True`.
- Usage Example:
    ```
    # detrend with default options (Spline fit, ratio index)
    >>> rwi = dpl.detrend(data)

    # fit a Hugershoff curve and form the index by difference
    >>> dpl.detrend(data, fit="ModHugershoff", method="difference")

    # detrend only SERIES_1, SERIES_2 and SERIES_3
    >>> dpl.detrend(data[["SERIES_1", "SERIES_2", "SERIES_3"]], fit="ModNegExp")
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
    - `make_plot`: default `False`; when `True`, also draws the dplR-style crossdating overview (`corr.rwl.seg` / `plot.crs`).
- Usage examples:
    ```
    >>> ca533_rwi = dpl.detrend(ca533, plot=False)

    # Crossdating of detrended data with default args
    >>> dpl.xdate(ca533_rwi)

    # Crossdating with Pearson correlation, plus the overview plot
    # (other options set to defaults when not specified).
    >>> dpl.xdate(ca533_rwi, corr="Pearson", make_plot=True)
    ```
- Related: `dpl.xdate_plot(rwi)` draws the crossdating overview on its own; `dpl.series_corr(rwi, "SERIES_1")` gives the per-series moving-correlation and cross-correlation diagnostics; `dpl.interseries_corr(rwi)` returns the mean interseries correlation.

### Build an AR-based chronology with `chron_ars`

- Description: builds ARSTAN-style chronologies from **detrended** ring-width indices — the standard mean chronology, the residual (AR-prewhitened) chronology, and the rescaled "ARSTAN" chronology that adds the pooled autoregression back onto the residual chronology.
- Usage Example:
    ```
    >>> rwi = dpl.detrend(data, plot=False)
    >>> dpl.chron_ars(rwi)                 # standard + residual + ARSTAN chronologies
    ```

### Date a floating series with `xdate_floater`

- Description: finds the best calendar placement for an **undated (floating)** ring-width series against a dated master collection, reporting a t-value, autocorrelation-adjusted degrees of freedom, Bonferroni-adjusted p-value and isolation factor for each candidate position (after Wilson 2026).
- Key arguments: `data` (the dated master collection), `series` (the undated series), `min_overlap` (default `50`), `make_plot=True` for the Wilson-style dating figure.
- Usage Example:
    ```
    # 'master' is a dated rwl/rwi collection; 'floater' is an undated series
    >>> result = dpl.xdate_floater(master, floater, series_name="UNK01", make_plot=True)
    >>> result["best"]["max_year"], result["best"]["t"]   # best end year and its t-value
    ```

### Regional Curve Standardization with `rcs`

- Description: detrends by the Regional Curve method — aligning all series by cambial age, fitting one common growth curve, and dividing each series by it (preserves low-frequency/long-timescale variance that per-series detrending removes).
- Options: `po` (a pith-offset table aligning series to cambial age), `nyrs`/`f` (curve stiffness), `make_plot` (default `True`).
- Usage Example:
    ```
    >>> rwi = dpl.rcs(data, po=pith_offsets)     # po: DataFrame of series -> years-to-pith
    ```

### Read chronology files with `read_crn`

- Description: reads Tucson `.crn` chronology files into a dataframe (the read side of `writers(..., format="crn")`).
- Usage Example:
    ```
    >>> crn = dpl.read_crn("/path/to/file.crn")
    ```

### Agreement statistics: `glk` and `sgc`

- Description: `glk` computes Gleichläufigkeit (the sign-agreement / parallel-run statistic) between series; `sgc` computes synchronous growth changes. Both return the pairwise matrix (and, by default, its significance).
- Usage Example:
    ```
    >>> dpl.glk(data)      # Gleichläufigkeit
    >>> dpl.sgc(data)      # synchronous growth changes
    ```

### Mean sensitivity: `sens1` and `sens2`

- Description: `sens1` is the classic mean sensitivity of each series; `sens2` is the alternative (Bunn/dplR) formulation.
- Usage Example:
    ```
    >>> dpl.sens1(data)
    >>> dpl.sens2(data)
    ```

### Average cores to trees with `tree_mean`

- Description: averages multiple cores from the same tree into per-tree series (dplR `treeMean`), given an ID mapping (see `read_ids`).
- Usage Example:
    ```
    >>> ids = dpl.read_ids(data)
    >>> trees = dpl.tree_mean(data, ids)
    ```

### Basal area increment with `bai_out` and `bai_in`

- Description: converts ring widths to basal area increment, the annual cross-sectional area of wood added. `bai_out` works from the outside (bark) in; `bai_in` works from the pith out.
- Options: `diam` (per-series stem diameters) for `bai_out`; `d2pith` (distance-to-pith offsets) for `bai_in`.
- Usage Example:
    ```
    >>> bai = dpl.bai_out(data)                    # radius = summed ring widths
    >>> bai = dpl.bai_out(data, diam=diam_table)   # measured stem diameters
    >>> bai = dpl.bai_in(data, d2pith=d2pith_table)
    ```

### Combine datasets with `combine_rwl`

- Description: merges several ring-width datasets onto the union of their years (a thin wrapper over `pandas`, matching dplR's `combine.rwl`).
- Usage Example:
    ```
    >>> both = dpl.combine_rwl([site_a, site_b])   # or dpl.combine_rwl(site_a, site_b)
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
    
### Export and import LiPD with `to_lipd` and `from_lipd`

- Description: writes a completed chronology (and, optionally, the underlying ring widths, running statistics, site metadata and publication info) to a LiPD file, and reads dplPy/ITRDB LiPD files back in. Requires the optional `pylipd` dependency (`pip install pylipd`).
- Usage Example:
    ```
    >>> rwl = dpl.readers("co021.rwl")
    >>> rwi = dpl.detrend(rwl, plot=False)
    >>> crn = dpl.chron(rwi, plot=False)
    >>> dpl.to_lipd(crn, "co021", rwl=rwl)         # writes co021.lpd
    >>> back = dpl.from_lipd("co021.lpd")          # dict: chronology, rwl, metadata, ...
    ```

### Other functions

These are also available; see each function's docstring (`help(dpl.<name>)`) for details.

| Function | Purpose |
|---|---|
| `series_corr` | per-series crossdating diagnostics (moving correlation + cross-correlation) |
| `interseries_corr` | mean interseries correlation (rbar) for a dataset |
| `rwi_stats` / `rwi_stats_running` | chronology signal statistics (rbar, EPS, SNR), overall or in a running window |
| `sss` | subsample signal strength |
| `samp_stats` | sample-depth statistics through time |
| `common_interval` | find (and optionally plot) the common overlap interval of a dataset |
| `ssf` | simple signal-free standardization |
| `powt` | power transformation of ring widths (Cook & Peters) |
| `ads` | standalone age-dependent smoothing spline |
| `fill_internal` | fill internal missing (NA) values within series |
| `read_ids` | parse tree/core IDs from series names |
| `po_to_wc` / `wc_to_po` | convert between pith offsets and years-to-pith |
| `SiteMetadata` | container for site metadata (used by the LiPD and `.crn` writers) |

