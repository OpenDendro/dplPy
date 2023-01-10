__copyright__ = """
   dplPy for tree ring width time series analyses
   Copyright (C) 2022  OpenDendro

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

# Date: 01/07/2023
# Author: Ifeoluwa Ale
# Title: xdate.py
# Project: OpenDendro dplPy
# Description: Crossdating function for dplPy datasets.
#
# example usage from Python Console: 
# >>> import dplpy as dpl 
# >>> data = dpl.readers("../tests/data/csv/file.csv")
# >>> dpl.xdate(data)
# >>> dpl.xdate(data, prewhiten=False, corr="Pearson")
# >>> dpl.xdate(data, slide_period=50, bin_floor=10)

from detrend import detrend
from autoreg import ar_func_series
from chron import chron
import pandas as pd
import numpy as np
import scipy

# Main crossdating function, returns a dictionary that maps each series to its correlation against
# a chronology of the other series.
def xdate(data, prewhiten=True, corr="Spearman", ar_lag=5, slide_period=50, bin_floor=100):
    # Identify first and last valid indexes, for separating into bins.
    bins = get_bins(data.first_valid_index(), data.last_valid_index(), bin_floor, slide_period)

    rwi_data = detrend(data, fit="horizontal", plot=False)

    # if detrending returns error, raise to output
    if isinstance(rwi_data, ValueError) or isinstance(rwi_data, TypeError):
        raise rwi_data

    # drop nans, prewhiten series if necessary
    ready_series = {}
    for series in rwi_data:
        nullremoved_data = rwi_data[series].dropna()
        if prewhiten is True:
            res = ar_func_series(nullremoved_data, ar_lag)
            offset = len(nullremoved_data) - len(res)
            ready_series[series] = pd.Series(data=res, name=series, index=nullremoved_data.index.to_numpy()[offset:])
        else:
            ready_series[series] = nullremoved_data

    ready_series_copy = ready_series.copy()
    
    series_corr = {}

    for series in ready_series:
        removed = ready_series_copy.pop(series)

        # create chronology with current series removed
        new_chron = chron(ready_series_copy, plot=False)["Mean RWI"]

        # correlate current series against chronology of remaining series
        inp = pd.concat([removed, new_chron], axis=1, join='inner')
        series_corr[series] = correlate(inp, corr)
        
        # evaluation of current series vs chronology of others by segments of years (the bins created earlier)
        for range in bins:
            start = int(range.split("-")[0])
            end = int(range.split("-")[1])
            if start >= removed.first_valid_index() and end <= removed.last_valid_index():
                segment = removed.loc[start:end]
                compare_segment(segment, new_chron, slide_period, corr, slide=False)

        ready_series_copy[series] = removed
    return series_corr

# Generates the bins given the first and last years of recorded data, the bin floor and
# the desired number of years in a period (bin).
def get_bins(first_year, last_year, bin_floor, slide_period):
    overlap = int(slide_period/2)
    if bin_floor == 0 or first_year % bin_floor == 0:
        start = first_year
    else:
        start = (int(first_year/bin_floor) * bin_floor) + bin_floor
    
    i = start

    bins = []
    while i+slide_period-1 < last_year:
        bins.append(str(i) + "-" + str(i+slide_period-1))
        i += overlap
    return bins

# Returns the correlation value of the given data. Can find Spearman or Pearson's
# correlations
def correlate(data, type):
    if type == "Spearman":
        return scipy.stats.spearmanr(data, axis=0).correlation
    elif type == "Pearson":
        return np.corrcoef(data, rowvar=False)[0, 1]
    
# Compares segments 
def compare_segment(segment, new_chron, slide_period, correlation_type, slide=True, left_most=-10, right_most=10):
    if segment.size < slide_period:
        return
    series_name = segment.name
    data = pd.concat([segment, new_chron], axis=1, join='inner')
    original = correlate(data, correlation_type)

    # dplR flags correlation values < 0.25
    # Update warning message
    if original < 0.2:
        print(segment.name, "has low (< 0.2) correlation with rest of series' chronology, from", segment.first_valid_index(), "to", segment.last_valid_index())

    if not slide:
        return 0, original
    
    best_lag = 0
    best_coeff = original

    for shift in range(left_most, right_most+1):
        shifted = data[series_name].copy(deep=False)
        shifted.index += shift
        overlapping_df = pd.concat([shifted, new_chron], axis=1, join='inner').dropna()
        if overlapping_df.size == slide_period * 2:
            new_coeff = correlate(overlapping_df, correlation_type)
            if new_coeff > best_coeff:
                # for debugging print(segment.name, segment.first_valid_index(), shift)
                best_lag = shift
                best_coeff = new_coeff
    
    if (best_lag != 0 and abs(best_coeff - original) >= 0.1):
        print("\nUnexpected results for", segment.name, "in the range of", segment.first_valid_index(), "to", segment.last_valid_index())
        print("Original correlation was", original)
        print("Best correlation was at lag", best_lag, "and correlation was", best_coeff, "\n")
    
    return best_lag, best_coeff

