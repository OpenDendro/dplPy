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

# Date: 5/27/2022
# Author: Ifeoluwa Ale
# Title: summary.py
# Description: Generates a summary of each series recorded in Tucson format and CSV format files
# example usage:
# >>> import dplpy as dpl 
# >>> data = dpl.readers("../tests/data/csv/file.csv")
# >>> dpl.summary(data)
#
# >>> dpl.summary("../tests/data/csv/file.csv")
# >>> Note: for file pathname inputs, only CSV and RWL file formats are accepted

# Create Summaries for Tucson (*rwl) files
import pandas as pd
import numpy as np

from ..io.readers import readers
def summary(inp):
    """
    Generate basic statistical summary for tree ring width data.
    
    This function provides a quick statistical overview using pandas' built-in
    describe() method, offering standard descriptive statistics for all series
    in the dataset. It serves as a simplified alternative to the more comprehensive
    stats() function.
    
    Parameters
    ----------
    inp : pandas.DataFrame or str
        Input tree ring data. Can be either:
        - pandas.DataFrame: Tree ring data with year index and series columns
        - str: File path to CSV or RWL file that will be read automatically
    
    Returns
    -------
    pandas.DataFrame or None
        Statistical summary DataFrame with the following statistics for each series:
        - count: Number of non-NaN observations
        - mean: Arithmetic mean
        - std: Standard deviation
        - min: Minimum value
        - 25%: 25th percentile (first quartile)
        - 50%: 50th percentile (median)
        - 75%: 75th percentile (third quartile)
        - max: Maximum value
        
        Returns None if invalid input is provided, with error messages printed.
    
    Notes
    -----
    This function is a wrapper around pandas.DataFrame.describe() and provides:
    - Quick statistical overview without detailed dendrochronological metrics
    - Standard descriptive statistics familiar to general data analysis
    - Quartile information useful for understanding data distribution
    - Count information to assess data completeness
    
    **Comparison with stats() function**:
    - summary(): Basic statistics, fast computation, general-purpose
    - stats(): Dendrochronology-specific statistics, includes AR1, Gini, skewness
    
    The summary is particularly useful for:
    - Initial data exploration and quality checks
    - Quick assessment of data ranges and distributions
    - Identifying potential outliers or data entry errors
    - General statistical reporting
    
    Examples
    --------
    >>> import dplpy as dpl
    >>> # Generate summary from DataFrame
    >>> data = dpl.readers('../tests/data/rwl/ca533.rwl')
    >>> basic_stats = dpl.summary(data)
    >>> print(basic_stats)
    >>> 
    >>> # Generate summary directly from file
    >>> file_summary = dpl.summary('../tests/data/rwl/ca533.rwl')
    >>> 
    >>> # Check data completeness
    >>> print(f"Data completeness: {basic_stats.loc['count'].min()}/{len(data)} years")
    >>> 
    >>> # Identify potential outliers
    >>> for col in basic_stats.columns:
    ...     q75 = basic_stats.loc['75%', col]
    ...     q25 = basic_stats.loc['25%', col]
    ...     iqr = q75 - q25
    ...     upper_bound = q75 + 1.5 * iqr
    ...     print(f"{col} outlier threshold: {upper_bound}")
    
    See Also
    --------
    stats : Comprehensive dendrochronological statistics
    report : Detailed analysis including missing rings
    pandas.DataFrame.describe : Underlying pandas method
    """
    if isinstance(inp, pd.DataFrame):
        series_data = inp
    elif isinstance(inp, str):
        series_data = readers(inp)
    else:
        print("\nUnable to generate summary report. Invalid input")
        print("Note: for file pathname inputs, only CSV and RWL file formats are accepted\n")
        print("example usages:")
        print(">>> import dplpy as dpl")
        print(">>> data = dpl.readers('../tests/data/csv/file.csv')")
        print(">>> dpl.summary(data)\n")
        print(">>> dpl.summary('../tests/data/csv/file.csv')\n")
        return None

    summary = series_data.describe()
    return summary