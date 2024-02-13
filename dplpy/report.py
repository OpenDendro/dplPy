from __future__ import print_function

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

# Date: 9/11/2022
# Author: Ifeoluwa Ale
# Title: report.py
# Description: Generates a report about absent rings in the data set


import pandas as pd
from readers import readers
from stats import stats
import numpy as np
from statsmodels.tsa.ar_model import AutoReg

def report(inp: pd.DataFrame | str):
    """Generates a report
    
    Extended Summary
    ----------------
    Generates a text report about the input dataset that includes:
        Number of dated series
        Number of measurements
        Avg series length (years)
        Range (total years)
        Span (start-end year)
        Mean (Standard Deviation) series intercorrelation
        Mean (Standard Deviation) AR1
        Years with absent rings listed by series
    
    Parameters
    ----------
    data : str
           a data file (.CSV or .RWL) or a pandas dataframe imported from dpl.readers().
              
    Returns
    -------
    data : pandas dataframes
    
    Examples
    --------
    >>> import dplpy as dpl
    >>> data = dpl.readers("../tests/data/csv/file.csv")
    >>> dpl.report(data) 
    
    References
    ----------
    .. [1] https:/opendendro.org/dplpy-man/#report

    """
    if isinstance(inp, pd.DataFrame):
        series_data = inp
    elif isinstance(inp, str):
        series_data = readers(inp)
    else:
        return
    
    statistics = stats(series_data)
    no_of_series = series_data.shape[1]
    no_of_measurements = series_data.count().sum()
    first_year = statistics["first"].min()
    last_year = statistics["last"].max()
    missing_rings, avg_ar = get_report_stats(series_data)

    print("Number of dated series:", no_of_series)
    print("Number of measurements:", no_of_measurements)
    print("Avg series length:", round(no_of_measurements/no_of_series, 4))
    print("Range:", (last_year - first_year + 1))
    print("Span:", first_year, "-", last_year)
    print("Mean (Std dev) series intercorrelation:")
    print("Mean (Std dev) AR1:", round(avg_ar, 4))
    print("-------------")
    print("Years with absent rings listed by series\n")
    print_missing_ring_data(missing_rings)
    print("-------------")
    print("Years with internal NA values listed by series")

# Analyze the dataframe to generate report on missing data (and eventually internal NAs)
def get_report_stats(series_data):
    ar1s = []
    missing_rings = {}
    nans = {}
    for series_name, data in series_data.items():
        missing_rings[series_name] = list(map(str, data[data==0].index.tolist()))
        nans[series_name] = list(map(str, data[data==np.nan].index.tolist()))
        ar1s.append(round(AutoReg(data.dropna().to_numpy(), 1, old_names=False).fit().params[1], 3))
    avg_ar = sum(ar1s)/len(ar1s)

    #print(nans)
    internal_nans = {}
    for series_name, data in nans.items():
        if len(data) == 0:
            continue
        i = 1
        j = len(data) - 2
        
        while j > i:
            if data[i] != (data[i-1] + 1) and data[j+1] != (data[j] + 1):
                internal_nans[series_data] = data[i:j]
                break
            if data[i] == data[i-1] + 1:
                i += 1
            if data[j+1] == data[j] + 1:
                j += 1

    return missing_rings, avg_ar

# Print data about missing rings
def print_missing_ring_data(missing_rings):
    for series, missing in missing_rings.items():
        if len(missing) != 0:
            print("     ", series, "--", " ".join(missing))