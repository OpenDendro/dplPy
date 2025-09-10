from __future__ import print_function

__copyright__ = """
   dplPy for tree ring width time series analyses
   Copyright (C) 2025  OpenDendro

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
__license__ = "GNU GPLv3"

#!/usr/bin/python
# -*- coding: utf-8 -*-

# Date: 11/1/2022
# Author: Ifeoluwa Ale
# Title: chron.py
# Project: OpenDendro dplPy
# Description: Creates a mean value chronology for a dataset, typically the ring width indices of a detrended series.
#              Takes three optional arguments 'biweight', 'prewhiten', and 'plot'. They determine whether to find means using tukey's
#              biweight robust mean (default True), whether to prewhiten data by fitting to an AR model (default False), and
#              whether to plot the results of the chronology (default True).
# example usage from Python Console: 
# >>> import dplpy as dpl 
# >>> data = dpl.readers("../tests/data/csv/file.csv")
# >>> rwi_data = dpl.detrend(data)
# >>> dpl.chron(rwi_data)
# >>> dpl.chron(rwi_data, prewhiten=True)
# >>> chron_data = dpl.chron(rwi_data, biweight=False, plot=False)

from typing import Union, Optional
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ..utils.tbrm import tbrm
from .autoreg import ar_func

def chron(rwi_data: pd.DataFrame, biweight: bool = True, prewhiten: bool = False, plot: bool = True) -> pd.DataFrame:
    """
    Create a mean value chronology from ring width index data.
    
    This function builds a master chronology by calculating mean ring width indices
    across multiple tree ring series for each year. The chronology represents the
    common environmental signal shared across trees at a site, with individual
    tree-specific variations averaged out.
    
    Parameters
    ----------
    rwi_data : dict of pandas.Series
        Dictionary containing ring width index data from detrended series.
        Keys are series identifiers, values are pandas Series with year indices
        and ring width index values (typically around 1.0).
    biweight : bool, optional
        Whether to use Tukey's biweight robust mean for calculating chronology values,
        by default True. If False, uses arithmetic mean. Biweight mean reduces
        the influence of outliers and is recommended for dendrochronological applications.
    prewhiten : bool, optional
        Whether to prewhiten data by removing autocorrelation using autoregressive
        modeling before creating chronology, by default False. Prewhitening can
        improve the representation of high-frequency climate signals.
    plot : bool, optional
        Whether to display diagnostic plots showing the chronology and sample depth
        through time, by default True.
    
    Returns
    -------
    pandas.DataFrame
        Chronology DataFrame with year index and the following columns:
        - 'Mean RWI': Mean ring width indices for each year
        - 'Mean Res': Prewhitened mean residuals (only if prewhiten=True)
        - 'Sample depth': Number of series contributing to each year's mean
    
    Notes
    -----
    Chronology development process:
    
    1. **Data aggregation**: Ring width indices from all series are grouped by year
    2. **Mean calculation**: For each year, calculate mean across available series
    3. **Robust statistics**: Tukey's biweight mean reduces outlier influence
    4. **Prewhitening** (optional): Remove autocorrelation using AR modeling
    5. **Sample depth tracking**: Record number of series for each year
    
    The resulting chronology represents the common growth signal and is suitable for:
    - Climate reconstruction studies
    - Comparison with instrumental climate records  
    - Regional climate pattern analysis
    - Dating archaeological or historical samples
    
    Sample depth considerations:
    - Higher sample depths (more series) generally provide more reliable chronology values
    - Early and late portions of chronologies often have lower sample depths
    - Minimum sample depth thresholds (e.g., n≥5) are often applied in studies
    
    References
    ----------
    Cook, E. R., & Kairiukstis, L. A. (Eds.). (1990). Methods of dendrochronology: 
    applications in the environmental sciences. Springer Science & Business Media.
    
    Mosteller, F., & Tukey, J. W. (1977). Data analysis and regression: 
    a second course in statistics. Addison-Wesley.
    
    Examples
    --------
    >>> import dplpy as dpl
    >>> # Load and detrend data
    >>> data = dpl.readers('../tests/data/rwl/ca533.rwl')
    >>> rwi_data = dpl.detrend(data, plot=False)
    >>> 
    >>> # Create basic chronology with biweight mean
    >>> chronology = dpl.chron(rwi_data)
    >>> print(chronology.head())
    >>> 
    >>> # Create prewhitened chronology (removes autocorrelation)
    >>> chron_prewhit = dpl.chron(rwi_data, prewhiten=True)
    >>> 
    >>> # Create chronology with arithmetic mean, no plots
    >>> chron_simple = dpl.chron(rwi_data, biweight=False, plot=False)
    >>> 
    >>> # Check sample depth statistics
    >>> print(f"Sample depth range: {chronology['Sample depth'].min()}-{chronology['Sample depth'].max()}")
    
    See Also
    --------
    detrend : Detrend raw ring width data to create ring width indices
    tbrm : Tukey's biweight robust mean calculation
    ar_func : Autoregressive modeling for prewhitening
    plot_chron : Plot chronology with sample depth
    """
    chron_data = {}
    for series in rwi_data:
        series_data = rwi_data[series].dropna()
        for year in series_data.keys().tolist():
            if year not in chron_data:
                chron_data[year] = [1, series_data[year]]
            else:
                chron_data[year][0] += 1
                chron_data[year].append(series_data[year])
    
    years, means, depths = get_chron_info(chron_data, biweight)
    chron_res = pd.DataFrame(data={"Year":years})
    chron_res = pd.concat([chron_res, pd.Series(data=means, name="Mean RWI")], axis=1)
    
    if prewhiten:
        whitened_means = get_whitened_chron_info(rwi_data, chron_data, biweight)
        chron_res = pd.concat([chron_res, pd.Series(data=whitened_means, name="Mean Res")], axis=1)
    else:
        whitened_means = None
    
    chron_res = pd.concat([chron_res, pd.Series(data=depths, name="Sample depth")], axis=1)
    chron_res.set_index('Year', inplace = True, drop = True)

    if plot:
        plot_chron(years, depths, means, whitened_means)
    
    return chron_res

def get_chron_info(chron_data, biweight):
    """
    Calculate chronology statistics from organized ring width index data.
    
    This internal function processes the organized chronology data structure
    to compute mean values and sample depths for each year using either
    arithmetic or biweight robust means.
    
    Parameters
    ----------
    chron_data : dict
        Dictionary with year keys and values as lists where first element
        is sample count and remaining elements are RWI values for that year.
    biweight : bool
        Whether to use Tukey's biweight robust mean (True) or arithmetic mean (False).
    
    Returns
    -------
    tuple of (list, list, list)
        - years: Sorted list of years
        - means: Mean RWI values for each year
        - depths: Sample depth (number of series) for each year
    
    Notes
    -----
    The biweight mean is more robust to outliers than arithmetic mean
    and is preferred in dendrochronological applications where individual
    trees may show unusual growth patterns due to local disturbances.
    """
    years = []
    means = []
    depths = []
    
    for year in sorted(chron_data):
        years.append(year)
        if biweight:
            means.append(tbrm(chron_data[year][1:]))
        else:
            means.append(sum(chron_data[year][1:])/chron_data[year][0])
        depths.append(chron_data[year][0])

    return years, means, depths

def get_whitened_chron_info(rwi_data, chron_data, biweight):
    """
    Create prewhitened chronology by removing autocorrelation from individual series.
    
    This function applies autoregressive (AR) modeling to each ring width index
    series to remove autocorrelation, then calculates mean residuals to create
    a prewhitened chronology that emphasizes high-frequency climate signals.
    
    Parameters
    ----------
    rwi_data : dict of pandas.Series
        Dictionary of ring width index series to be prewhitened.
    chron_data : dict
        Organized chronology data structure with years and sample information.
    biweight : bool
        Whether to use Tukey's biweight robust mean for calculating means.
    
    Returns
    -------
    list
        List of prewhitened mean values aligned with chronology years.
        Contains NaN for years where no prewhitened data is available.
    
    Notes
    -----
    Prewhitening process:
    
    1. Fit AR model to each individual RWI series
    2. Extract residuals (detrended for autocorrelation)
    3. Calculate mean residuals for each year
    4. Align with original chronology time span
    
    Prewhitened chronologies are particularly useful for:
    - Climate reconstruction where high-frequency signals are important
    - Removing biological persistence in tree growth
    - Improving correlation with climate variables
    - Reducing the influence of previous year's growth on current year
    
    The AR modeling may result in fewer data points due to the need for
    lagged values, which is why the function handles alignment carefully.
    """
    whitened_data = {}
    ar_fit_data = {}

    for series in rwi_data:
        series_data = rwi_data[series].dropna()
        series_years = series_data.keys().tolist()
        ar_fit_data[series] = ar_func(series_data)

        offset = len(series_years) - len(ar_fit_data[series])
        i = 0
        for year in series_years[offset:]:
            if year not in whitened_data:
                whitened_data[year] = [1, ar_fit_data[series][i]]
            else:
                whitened_data[year][0] += 1
                whitened_data[year].append(ar_fit_data[series][i])
            i += 1

    whitened_means = []
    for year in sorted(chron_data):
        if year not in whitened_data:
            whitened_means.append(np.nan)
        elif biweight:
            whitened_means.append(tbrm(whitened_data[year][1:]))
        else:
            whitened_means.append(sum(whitened_data[year][1:])/whitened_data[year][0])
    return whitened_means

def plot_chron(years, depths, means, whitened_means):
    """
    Create diagnostic plots for chronology data.
    
    This function generates a dual-axis plot showing both the chronology values
    (either standard or prewhitened) and the sample depth through time.
    
    Parameters
    ----------
    years : list
        List of years for the chronology.
    depths : list
        Sample depth (number of contributing series) for each year.
    means : list
        Standard mean RWI values for each year.
    whitened_means : list or None
        Prewhitened mean residuals for each year, or None if not calculated.
    
    Returns
    -------
    None
        Displays the plot but does not return data.
    
    Notes
    -----
    The plot includes:
    - **Primary y-axis**: Chronology values (RWI or prewhitened residuals)
    - **Secondary y-axis**: Sample depth shown as filled area
    - **Time series**: Complete chronology span with year labels
    
    Plot interpretation:
    - Higher chronology values indicate favorable growing conditions
    - Lower chronology values indicate unfavorable conditions  
    - Sample depth shows reliability of chronology values
    - Periods with low sample depth should be interpreted cautiously
    
    The function automatically selects prewhitened data for plotting if available,
    otherwise displays standard chronology values.
    """
    # create figure and axis objects with subplots()
    fig,ax = plt.subplots()

    if whitened_means is not None:
        y_val = whitened_means
        y_label = "Mean Res"
    else:
        y_val = means
        y_label = "Mean RWI"

    # make plot of RWI means
    ax.plot(years, y_val, "k-")
    ax.set_xlabel("Year", fontsize = 14)
    ax.set_ylabel(y_label, fontsize=14)

    # twin object for two different y-axis on the sample plot
    ax2=ax.twinx()
    # make plot of sample depths
    ax2.fill_between(years, depths, color=((0.2, 0.6, 0.9, 0.3)))
    ax2.set_ylabel("Sample depth",fontsize=14)
    fig.set_size_inches(14, 8)
    plt.show()