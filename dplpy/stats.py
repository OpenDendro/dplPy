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

# Ignore the following comments:
#Code to calculate ar1 when statsmodels can be imported
#from statsmodels.tsa import stattools
# x = 1-D array
# Yield normalized autocorrelation function of number lags
#autocorr = stattools.acf( x )

# Get autocorrelation coefficient at lag = 1
#autocorr_coeff = autocorr[1]

import pandas as pd
import numpy as np
from readers import readers
from statsmodels.tsa.ar_model import AutoReg

def stats(inp: pd.DataFrame | str):
    """Generates summary statistics
    
    Extended Summary
    ----------------
    Generates summary statistics for .RWL and .CSV format files. 
    It outputs a table with 'first', 'last', 'year', 'mean', 'median', 'stdev', 
    'skew', 'gini', 'ar1' for each series in data file.
    
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
    if isinstance(inp, pd.DataFrame):
        series_data = inp
    elif isinstance(inp, str):
        series_data = readers(inp)

        
    stats = {"series":[], "first":[], "last":[], "year": [], "mean": [], "median":[], "stdev":[], "skew":[], "gini":[], "ar1":[]}

    for series_name, data in series_data.items():
        stats["series"].append(series_name)
        stats["first"].append(data.first_valid_index())
        stats["last"].append(data.last_valid_index())
        stats["year"].append(stats["last"][-1] - stats["first"][-1] + 1)
        stats["mean"].append(round(data.mean(), 3))
        stats["median"].append(round(data.median(), 2))
        stats["stdev"].append(round(data.std(), 3))
        stats["skew"].append(round(get_skew(data), 3))
        stats["gini"].append(round(get_gini(data.dropna().to_numpy()), 3))
        stats["ar1"].append(round(AutoReg(data.dropna().to_numpy(), 1, old_names=False).fit().params[1], 3))


    statistics = pd.DataFrame(stats)
    statistics.index += 1
    return statistics

def get_gini(data_array):
    # might need to work on more efficient solution
    # Mean absolute difference
    mad = np.abs(np.subtract.outer(data_array, data_array)).mean()
    # Relative mean absolute difference
    rmad = mad/np.mean(data_array)
    # Gini coefficient
    g = 0.5 * rmad
    return g

# gets skew values for each series
def get_skew(data_series):
    return (((data_series - data_series.mean()) / data_series.std()) ** 3).mean()