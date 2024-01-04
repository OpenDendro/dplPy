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

# Date: 5/12/2023
# Author: Ifeoluwa Ale
# Title: xdate.py
# Project: OpenDendro dplPy
# Description: Crossdating function for dplPy datasets.
#
# example usage from Python Console: 
# >>> import dplpy as dpl 
# >>> data = dpl.readers("../tests/data/csv/file.csv")
# >>> dpl.xdate(data)
# >>> dpl.xdate(data, prewhiten=False, corr="Pearson", show_flags=False)
# >>> dpl.xdate(data, slide_period=50, bin_floor=10, p_val=0.02)

from detrend import detrend
from autoreg import ar_func_series
from chron import chron
from stats import stats
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy
import re


# Main crossdating function, returns a dataframe of each series' segment correlations compared to the same
# segments in the master chronology.
def xdate(data: pd.DataFrame, prewhiten=True, corr="Spearman", slide_period=50, bin_floor=100, p_val=0.05, show_flags=True):
    """Crossdating function
    
    Extended Summary
    ----------------

    Parameters
    ----------
    data : pandas dataframe
           a data file (.CSV or .RWL), or an array imported from dpl.readers()
    prewhiten : boolean, default True
                prewhiten series using AR modeling
    corr : str, default Spearman
           correlation type, options: 'Pearson' or 'Spearman'
    slide_period : int, default 50
                   period window (years)
    bin_floor : int, default 100
                bin size
    p_val : float, default 0.05
            p-value, options: '0.05', '0.01', '0.001'
    show_flags : boolean, default True
                 show flags in the output
                   
    Returns
    -------
    data : pandas dataframe
    plot : matplotlib.pyplot figure
    
    Examples
    --------
    >>> ca533_rwi = dpl.detrend(ca533, fit="spline", method="residual", plot=False)
    # Crossdating of detrended data
    >>> dpl.xdate(ca533_rwi, prewhiten=True, corr="Spearman", slide_period=50, bin_floor=100, p_val=0.05, show_flags=True)

    References
    ----------
    .. [1] https:/opendendro.org/dplpy-man/#xdate
            
    """
    # Check types of inputs
    if not isinstance(data, pd.DataFrame):
        errorMsg = "Expected dataframe input, got " + str(type(data)) + " instead."
        raise TypeError(errorMsg)
    
    # Identify first and last valid indexes, for separating into bins.
    bins, bin_data = get_bins(data.first_valid_index(), data.last_valid_index(), bin_floor, slide_period)

    rwi_data = detrend(data, fit="horizontal", plot=False)

    # if detrending returns error, raise to output
    if isinstance(rwi_data, ValueError) or isinstance(rwi_data, TypeError):
        raise rwi_data

    # drop nans, prewhiten series if necessary
    df_start = pd.DataFrame(index=pd.Index(data.index))
    to_concat = [df_start]

    for series in rwi_data:
        nullremoved_data = rwi_data[series].dropna()
        if prewhiten is True:
            try:
                res = ar_func_series(nullremoved_data, get_ar_lag(nullremoved_data))
                offset = len(nullremoved_data) - len(res)
                to_concat.append(pd.Series(data=res, name=series, index=nullremoved_data.index.to_numpy()[offset:]))
            except ZeroDivisionError:
                print("Zero division error for series:", series, ". Dropping series.")
        else:
            to_concat.append(nullremoved_data)

    ready_series = pd.concat(to_concat, axis=1)

    ready_series_copy = ready_series.copy()
    ready_series = ready_series.rename_axis(data.index.name)

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
            # print(range) # useful for debugging but not necessary once operational
            start = int(re.split("(?<=\\d)-", range)[0])
            end = int(re.split("(?<=\\d)-", range)[1])
            if start >= removed.first_valid_index() and end <= removed.last_valid_index():
                segment = removed.loc[start:end]

                seg_corr, flag, flag_data = compare_segment(segment, new_chron, slide_period, corr, p_val, slide=show_flags)

                if flag is not None:
                    flags[flag].append(flag_data)
                bin_data[range].append(seg_corr)
            else:
                bin_data[range].append(np.nan)
        #ready_series_copy[series] = removed
        ready_series_copy = pd.concat([ready_series_copy, removed], axis=1)
        
        if show_flags is True:
            print_flags(series, flags)
    print()

    bin_res = pd.DataFrame.from_dict(bin_data)
    bin_res.set_index(pd.Index(series_names), inplace=True)

    return bin_res.transpose()

# Variation of xdate function that plots a graph that color codes the segment correlations. 
# Will be merged into original xdate function when completed, so that users can choose to
# show the graph by passing an optional argument.
def xdate_plot(data: pd.DataFrame, slide_period=50, bin_floor=10):
    plot_data = xdate(data, slide_period=slide_period, bin_floor=bin_floor, show_flags=False)
    bins = plot_data.index.to_numpy()

    # obtain a list of series names sorted by the start date
    data_stats = stats(data)
    series_by_start_date = data_stats.sort_values(by='first')['series']

    # Change the style of plot
    plt.style.use('seaborn-v0_8-darkgrid')

    years = data.index.to_numpy()

    # set width and height of the window based on the data
    dimensions = (max((years[-1] - years[0])//80, 8), max(len(data.columns)//2, 8))
    lin_wid = 7.5

    plt.figure(figsize=(dimensions))
 
    # separate plots for each series using the offset
    offset = (data.mean().mean() * 2)

    y_divisions = [] # needed to put series names on the y-axis
    num=0
    for column_name in series_by_start_date:
        num+=1
        first = np.nan
        second = np.nan
        penult = np.nan
        last = np.nan
        
        for i in range(0, len(bins)-1, 2):
            bin_range1 = list(map(int, bins[i].split("-")))
            bin_range2 = list(map(int, bins[i+1].split("-")))

            # plot for bin range 1 if valid
            if (bin_range1[0] >= data[column_name].first_valid_index() and bin_range1[1] <= data[column_name].last_valid_index()):
                range_1_color = get_graph_color(plot_data.loc[bins[i]][column_name])
                plt.plot(bin_range1, np.zeros((2,), dtype=int) + (offset * (num-1) + (offset/2)), marker='_', linewidth=lin_wid, alpha=0.9, color=range_1_color)
                first = np.nanmin([first, bin_range1[0]])
                second = np.nanmin([second, bin_range1[1]])

            # plot for bin range 2 if valid
            if (bin_range2[0] >= data[column_name].first_valid_index() and bin_range2[1] <= data[column_name].last_valid_index()):
                range_2_color = get_graph_color(plot_data.loc[bins[i+1]][column_name])
                plt.plot(bin_range2, np.zeros((2,), dtype=int) + (offset * (num-1)), marker='_', linewidth=lin_wid, alpha=0.9, color=range_2_color)
                penult = np.nanmax([penult, bin_range2[0]])
                last = np.nanmax([last, bin_range2[1]])
        
        # pad before and after first 2 and last 2 bins with greens
        pad_start_and_end_of_series_graph(data[column_name], first, second, penult, last, num, offset, lin_wid)
        y_divisions.append(offset*(num-1))

    # set y-axis to display series names at equal intervals, and x-axis to display years
    plt.yticks(y_divisions, series_by_start_date)
    plt.xlabel("Year")

    # Show the graph
    plt.show()

def pad_start_and_end_of_series_graph(series, first, second, penult, last, num, offset, lin_wid):
    if not np.isnan(first):
        first_range = [series.first_valid_index(), first]
        plt.plot(first_range, np.zeros((2,), dtype=int) + (offset * (num-1) + (offset/2)), marker='_', linewidth=lin_wid, alpha=0.9, color='#00ff00')

                       
    if not np.isnan(second):
        second_range = [series.first_valid_index(), second]
        plt.plot(second_range, np.zeros((2,), dtype=int) + (offset * (num-1)), marker='_', linewidth=lin_wid, alpha=0.9, color='#00ff00')
    
    if not np.isnan(penult):
        penult_range = [penult, series.last_valid_index()]
        plt.plot(penult_range, np.zeros((2,), dtype=int) + (offset * (num-1) + (offset/2)), marker='_', linewidth=lin_wid, alpha=0.9, color='#00ff00')

    if not np.isnan(last):
        last_range = [last, series.last_valid_index()]
        plt.plot(last_range, np.zeros((2,), dtype=int) + (offset * (num-1) + (offset/2)), marker='_', linewidth=lin_wid, alpha=0.9, color='#00ff00')


# Helper function that determines the color of a segment of the graph depending on the correlation value.
def get_graph_color(corr_val):
    if corr_val == np.nan:
        return '#00ff00'
    elif corr_val < 0.1:
        return '#ff0d1a'
    elif corr_val < 0.3:
        return '#add8ff'
    elif corr_val < 0.4:
        return '#9cc7ff'
    elif corr_val < 0.5:
        return '#33b6ff'
    elif corr_val < 0.6:
        return '#0033ff'
    elif corr_val < 0.7:
        return '#0000ff'
    elif corr_val < 0.8:
        return '#0000dd'
    elif corr_val < 0.9:
        return '#0000b3'
    elif corr_val < 1:
        return '#000099'

# Determines the max lag to use for AR modeling function.
def get_ar_lag(data):
    n = len(data)
    res = min(int(n-1), int(10 * np.log10(n)))
    return res

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
  
# Compares segments to the mean values of the master chronology excluding the current series.
# This is where flags in dating are detected.
def compare_segment(segment, new_chron, slide_period, correlation_type, p_val, slide=True, left_most=-10, right_most=10):
    flag = None

    if segment.size < slide_period:
        return
    series_name = segment.name
    data = pd.concat([segment, new_chron], axis=1, join='inner')
    original = correlate(data, correlation_type)

    # Will set threshold to 99% confidence p value that is based on segment length.
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

# Prints the flags that had been detected for a series.
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
