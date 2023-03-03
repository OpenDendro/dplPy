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
def xdate(data, prewhiten=True, corr="Spearman", slide_period=50, bin_floor=100, p_val=0.05, show_flags=True):

    # Identify first and last valid indexes, for separating into bins.
    bins, bin_data = get_bins(data.first_valid_index(), data.last_valid_index(), bin_floor, slide_period)

    rwi_data = detrend(data, fit="horizontal", plot=False)

    # if detrending returns error, raise to output
    if isinstance(rwi_data, ValueError) or isinstance(rwi_data, TypeError):
        raise rwi_data


    # drop nans, prewhiten series if necessary
    ready_series = {}
    for series in rwi_data:
        nullremoved_data = rwi_data[series].dropna()
        if prewhiten is True:
            res = ar_func_series(nullremoved_data, get_ar_lag(nullremoved_data))
            offset = len(nullremoved_data) - len(res)
            ready_series[series] = pd.Series(data=res, name=series, index=nullremoved_data.index.to_numpy()[offset:])
        else:
            ready_series[series] = nullremoved_data

    ready_series_copy = ready_series.copy()
    
    series_names = []
    series_corr = []

    for series in sorted(ready_series):
        removed = ready_series_copy.pop(series)

        # create chronology with current series removed
        new_chron = chron(ready_series_copy, plot=False)["Mean RWI"]

        # correlate current series against chronology of remaining series
        inp = pd.concat([removed, new_chron], axis=1, join='inner')
        series_names.append(series)
        series_corr.append(correlate(inp, corr))

        flags = {"A":[], "B":[]}
        
        # evaluation of current series vs chronology of others by segments of years (the bins created earlier)
        for range in bins:
            start = int(range.split("-")[0])
            end = int(range.split("-")[1])
            if start >= removed.first_valid_index() and end <= removed.last_valid_index():
                segment = removed.loc[start:end]

                seg_corr, flag, flag_data = compare_segment(segment, new_chron, slide_period, corr, p_val, slide=show_flags)

                if flag is not None:
                    flags[flag].append(flag_data)
                bin_data[range].append(seg_corr)
            else:
                bin_data[range].append(np.nan)
        ready_series_copy[series] = removed
        
        if show_flags is True:
            print_flags(series, flags)
    print()

    #bin_res = pd.DataFrame.from_dict(bin_data)
    #bin_res.set_index(pd.Index(series_names), inplace=True)
    #print(bin_res)
    return pd.DataFrame.from_dict({"Series":series_names, "Correlation":series_corr})

# Noticed this is how R determines the max lag to use for its AR function, brough values closer to dplR.
def get_ar_lag(data):
    n = len(data)
    return min(int(n-1), int(10 * np.log(n)))

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
    bin_data = {}
    while i+slide_period-1 < last_year:
        period = str(i) + "-" + str(i+slide_period-1)
        bins.append(period)
        bin_data[period] = []
        i += overlap
    return bins, bin_data

# Returns the correlation value of the given data. Can find Spearman or Pearson's
# correlations
def correlate(data, type):
    if type == "Spearman":
        return scipy.stats.spearmanr(data, axis=0).correlation
    elif type == "Pearson":
        return np.corrcoef(data, rowvar=False)[0, 1]

def get_crit(alpha=0.01, n=50, type="one-tailed"):
    if type == "two-tailed":
        tcrit = scipy.stats.t.ppf(1-(alpha/2),n-2)
    else:
        tcrit = scipy.stats.t.ppf(1-(alpha),n-2)
    
    cc = pow((tcrit/np.sqrt(n-2)), 2)
    rcrit = np.sqrt(cc/(1+cc))
    return rcrit
  
# Compares segments 
def compare_segment(segment, new_chron, slide_period, correlation_type, p_val, slide=True, left_most=-10, right_most=10):
    flag = None

    if segment.size < slide_period:
        return
    series_name = segment.name
    data = pd.concat([segment, new_chron], axis=1, join='inner')
    original = correlate(data, correlation_type)

    # Will set threshold to 99% confidence p value (dk what that means yet) that is based on segment length.
    # Update warning message
    if original < get_crit(p_val):
        flag = "A"

    if not slide:
        return original, flag, None
    
    best_lag = 0
    best_coeff = original
    segment_data = []

    for shift in range(left_most, right_most+1):
        shifted = data[series_name].copy(deep=False)
        shifted.index += shift
        overlapping_df = pd.concat([shifted, new_chron], axis=1, join='inner').dropna()
        if overlapping_df.size == slide_period * 2:
            new_coeff = correlate(overlapping_df, correlation_type)
            segment_data.append(('{0:.2f}'.format(new_coeff)).rjust(5))
            if new_coeff > best_coeff:
                # for debugging print(segment.name, segment.first_valid_index(), shift)
                best_lag = shift
                best_coeff = new_coeff
        else:
            segment_data.append("     ")
    
    if (best_lag != 0 and abs(best_coeff - original) >= 0.08):
        flag = "B"

    if flag is not None:    
    # return original, flag ("A", "B" or None), with segment data, (None if flag is None and an array if otherwise.)
        segment_data = [(str(segment.first_valid_index()) + "-" + str(segment.last_valid_index())).rjust(12), str(best_lag).rjust(4)] + segment_data
        return original, flag, segment_data
    else:
        return original, flag, None

def print_flags(series_name, flags):
    if flags["A"] == [] and flags["B"] == []:
        return

    print("Flags for", series_name)
    if flags["A"] != []:
        print("[A] Segment  High   -10    -9    -8    -7    -6    -5    -4    -3    -2    -1     0    +1    +2    +3    +4    +5    +6    +7    +8    +9   +10")
        for flag in flags["A"]:
            print(" ".join(flag))
    if flags["B"] != []:
        print("[B] Segment  High   -10    -9    -8    -7    -6    -5    -4    -3    -2    -1     0    +1    +2    +3    +4    +5    +6    +7    +8    +9   +10")
        for flag in flags["B"]:
            print(" ".join(flag))
    
    print()
