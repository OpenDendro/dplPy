from __future__ import print_function
from sys import intern

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

# Date: 9/11/2022
# Author: Ifeoluwa Ale
# Title: report.py
# Description: Generates a report about absent rings in the data set
# example usage:
# >>> import dplpy as dpl 
# >>> data = dpl.readers("../tests/data/csv/file.csv")
# >>> dpl.report(data)
#
# >>> dpl.report("../tests/data/csv/file.csv")
# Note: for file pathname inputs, only CSV and RWL file formats are accepted

import pandas as pd
from ..io.readers import readers
from .stats import stats
import numpy as np
from statsmodels.tsa.ar_model import AutoReg

def report(inp):
    """
    Generate comprehensive data quality report for tree ring width datasets.
    
    This function creates a detailed analytical report focusing on data quality,
    temporal coverage, missing values, and statistical properties. It provides
    essential information for assessing dataset suitability for dendrochronological
    analyses and identifying potential data issues.
    
    Parameters
    ----------
    inp : pandas.DataFrame or str
        Input tree ring data. Can be either:
        - pandas.DataFrame: Tree ring data with year index and series columns
        - str: File path to CSV or RWL file that will be read automatically
    
    Returns
    -------
    None
        Prints comprehensive report to console. Does not return data objects.
    
    Notes
    -----
    The report includes the following sections:
    
    **Dataset Overview**:
    - Number of dated series in the dataset
    - Total number of ring width measurements
    - Average series length (measurements per series)
    - Temporal range (total years spanned)
    - Temporal span (first year to last year)
    
    **Statistical Summary**:
    - Mean series intercorrelation (placeholder - not currently implemented)
    - Mean and standard deviation of first-order autocorrelation (AR1)
    
    **Data Quality Assessment**:
    - **Absent rings analysis**: Years with zero ring width values by series
      - Zero values may indicate actual absent rings or measurement artifacts
      - Important for assessing data quality and crossdating accuracy
    
    - **Internal NA values**: Missing values within the temporal span of series
      - Helps identify incomplete measurements or data processing issues
    
    **Dendrochronological Relevance**:
    - **AR1 statistics**: First-order autocorrelation indicates growth persistence
      - High AR1 (~0.6-0.8): Strong year-to-year growth correlation
      - Moderate AR1 (~0.3-0.6): Typical for most temperate tree species
      - Low AR1 (<0.3): Weak persistence, potentially good climate signal
    
    - **Absent rings**: Critical for crossdating and data quality
      - May indicate actual biological phenomena or measurement issues
      - Patterns can reveal site-specific stress events
    
    Examples
    --------
    >>> import dplpy as dpl
    >>> # Generate report from DataFrame
    >>> data = dpl.readers('../tests/data/rwl/ca533.rwl')
    >>> dpl.report(data)
    
    >>> # Generate report directly from file
    >>> dpl.report('../tests/data/rwl/ca533.rwl')
    
    **Example output**:
    ```
    Number of dated series: 15
    Number of measurements: 2847
    Avg series length: 189.8
    Range: 500
    Span: 1530 - 2029
    Mean (Std dev) series intercorrelation:
    Mean (Std dev) AR1: 0.642
    -------------
    Years with absent rings listed by series
    
         CAM011 -- 1580 1623
         CAM021 -- 1640
    -------------
    Years with internal NA values listed by series
    ```
    
    See Also
    --------
    stats : Detailed statistical analysis of individual series
    summary : Basic statistical summary
    get_report_stats : Internal function for missing data analysis
    print_missing_ring_data : Internal function for formatting absent ring output
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

def get_report_stats(series_data):
    """
    Analyze tree ring dataset for missing data patterns and statistical properties.
    
    This internal function examines the dataset to identify absent rings (zero values),
    internal missing values (NaN), and calculates AR1 statistics for the report.
    
    Parameters
    ----------
    series_data : pandas.DataFrame
        Tree ring width data with year index and series columns.
    
    Returns
    -------
    tuple of (dict, float)
        - missing_rings: Dictionary with series names as keys and lists of years 
          with absent rings (zero values) as string values
        - avg_ar: Average first-order autocorrelation coefficient across all series
    
    Notes
    -----
    **Missing data analysis**:
    - **Absent rings**: Identified as years with exactly zero ring width
      - May represent true biological absent rings or measurement artifacts
      - Critical for crossdating validation
    
    - **Internal NAs**: Missing values within series temporal span
      - Currently detected but not fully implemented in output
      - Important for data completeness assessment
    
    **AR1 calculation**:
    - Computed using AutoReg model with lag=1
    - Calculated for each series individually, then averaged
    - Missing values automatically excluded from AR model fitting
    
    **Known issues**:
    - Internal NA detection logic has potential bugs in indexing
    - Some edge cases in missing data patterns may not be handled correctly
    """
    ar1s = []
    missing_rings = {}
    nans = {}
    for series_name, data in series_data.items():
        missing_rings[series_name] = list(map(str, data[data==0].index.tolist()))
        nans[series_name] = list(map(str, data[pd.isna(data)].index.tolist()))
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

def print_missing_ring_data(missing_rings):
    """
    Format and print information about absent rings in tree ring series.
    
    This internal function creates formatted output showing which years
    have absent rings (zero values) for each series in the dataset.
    
    Parameters
    ----------
    missing_rings : dict
        Dictionary with series names as keys and lists of years (as strings)
        with absent rings as values.
    
    Returns
    -------
    None
        Prints formatted absent ring information to console.
    
    Notes
    -----
    Output format for each series with absent rings:
    ```
          SERIES_NAME -- YEAR1 YEAR2 YEAR3
    ```
    
    Series with no absent rings are not displayed, making the output
    concise and focused on data quality issues that need attention.
    
    This information is crucial for:
    - Crossdating validation
    - Data quality assessment  
    - Identifying systematic measurement issues
    - Understanding site-specific stress events
    """
    for series, missing in missing_rings.items():
        if len(missing) != 0:
            print("     ", series, "--", " ".join(missing))