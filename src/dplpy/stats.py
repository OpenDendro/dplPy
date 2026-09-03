__copyright__ = """
   dplPy for tree ring width time series analyses
   Copyright (C) 2024  OpenDendro

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

# Date: 5/27/2022
# Author: Ifeoluwa Ale
# Title: stats.py
# Description: Generates summary statistics for Tucson format and CSV format files
# example usage:
# >>> import dplpy as dpl 
# >>> data = dpl.readers("../tests/data/csv/file.csv")
# >>> dpl.stats(data)
#
# >>> dpl.stats("../tests/data/csv/file.csv")
# >>> Note: for file pathname inputs, only CSV and RWL file formats are accepted

# Create Summaries for Tucson (*rwl) files

# Note on ar1: dplR's rwl.stats reports the lag-1 autocorrelation (the acf
# coefficient at lag 1), not an OLS AR(1) slope. We compute it directly (see
# get_ar1 below), which matches dplR to machine precision.

import pandas as pd
from ._validate import _coerce_to_frame
import numpy as np

def stats(inp: pd.DataFrame | str):
    """Generates summary statistics
    
    Extended Summary
    ----------------
    Generates summary statistics for .RWL and .CSV format files. 
    It outputs a dataframe with 'first', 'last', 'year', 'mean', 'median', 'stdev',
    'skew', 'kurtosis', 'gini', 'ar1' for each series in data file.
    
    Parameters
    ----------
    data : str
        a data file (.CSV or .RWL) or a pandas dataframe imported from dpl.readers().
    
    Returns
    -------
    data : pandas dataframe
    
    Examples
    --------
    >>> dpl.stats(<data>)
    
    References
    ----------
    .. [1] https:/opendendro.org/dplpy-man/#stats
    
    
    """
    series_data = _coerce_to_frame(inp)

        
    stats = {"series":[], "first":[], "last":[], "year": [], "mean": [], "median":[], "stdev":[], "skew":[], "kurtosis":[], "gini":[], "ar1":[]}

    for series_name, data in series_data.items():
        stats["series"].append(series_name)
        stats["first"].append(data.first_valid_index())
        stats["last"].append(data.last_valid_index())
        stats["year"].append(stats["last"][-1] - stats["first"][-1] + 1)
        stats["mean"].append(round(data.mean(), 3))
        stats["median"].append(round(data.median(), 3))
        stats["stdev"].append(round(data.std(), 3))
        stats["skew"].append(round(get_skew(data), 3))
        stats["kurtosis"].append(round(get_kurtosis(data), 3))
        stats["gini"].append(round(get_gini(data.dropna().to_numpy()), 3))
        stats["ar1"].append(round(get_ar1(data), 3))


    statistics = pd.DataFrame(stats)
    statistics.index += 1
    return statistics

def get_gini(data_array):
    # Sort-based O(n log n) formula, equivalent to the mean-absolute-difference
    # definition but avoids materializing an O(n^2) pairwise difference matrix.
    sorted_data = np.sort(np.asarray(data_array, dtype=float))
    n = len(sorted_data)
    ranks = np.arange(1, n + 1)
    total = np.sum(sorted_data)
    g = (2 * np.sum(ranks * sorted_data) - (n + 1) * total) / (n * total)
    return g

# gets skew values for each series
def get_skew(data_series):
    return (((data_series - data_series.mean()) / data_series.std()) ** 3).mean()

# gets (excess) kurtosis for each series -- matches dplR rwl.stats' kurt():
#   n * sum(y2^4) / (sum(y2^2)^2) * (1 - 1/n)^2 - 3,  y2 = y - mean(y)
def get_kurtosis(data_series):
    y = np.asarray(data_series.dropna(), dtype=float)
    n = len(y)
    y2 = y - y.mean()
    return n * np.sum(y2 ** 4) / (np.sum(y2 ** 2) ** 2) * (1 - 1 / n) ** 2 - 3

# gets lag-1 autocorrelation (acf coefficient at lag 1) for each series --
# matches dplR rwl.stats' ar1. This is the (biased, mean-centred) autocorrelation
#   r1 = sum_t(y2_t * y2_{t+1}) / sum_t(y2_t^2),   y2 = y - mean(y)
# not an OLS AR(1) regression slope. The 1/n biased-estimator divisor cancels
# between numerator and denominator, so this ratio *is* the acf at lag 1
# (equivalently statsmodels.tsa.stattools.acf(y, nlags=1, adjusted=False)[1]).
def get_ar1(data_series):
    y = np.asarray(data_series.dropna(), dtype=float)
    y2 = y - y.mean()
    return np.sum(y2[1:] * y2[:-1]) / np.sum(y2 ** 2)