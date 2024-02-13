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
# Title: series_corr.py
# Project: OpenDendro dplPy
# Description: Crossdating function that focuses on the comparison of one series to the
# master chronology.
#
# example usage from Python Console: 
# >>> import dplpy as dpl 
# >>> data = dpl.readers("../tests/data/csv/file.csv")
# >>> dpl.series_corr(data, "series_name")
# >>> dpl.series_corr(data, "series_name", prewhiten=False, corr="Pearson", bin_floor=10)

from detrend import detrend
from autoreg import ar_func_series
from chron import chron
from xdate import get_ar_lag, correlate, compare_segment, get_crit

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Analyzes the crossdating of one series compared to the master chronology
def series_corr(data: pd.DataFrame, series_name: str, prewhiten=True, corr="Spearman", seg_length=50, bin_floor=100, p_val=0.05):
    """Crossdating correlation function
    
    Extended Summary
    ----------------
    Crossdating correlation function that focuses on the comparison of one series to the master chronology.
    
    Parameters
    ----------
    data : pandas dataframe
           a data file (.CSV or .RWL), or an array imported from dpl.readers()
    series : str    
             a series name from the dataframe
    prewhiten : boolean, default False
                run pre-whitening on the time series, options: 'True' or 'False'.
    corr : str, default 'Spearman'
           select correlation type if 'prewhiten=True', options: 'Pearson' or 'Spearman'.
    seg_length :  int, default 50
                  segment length (years).
    bin_floor : int, default 100
                select bin size.
    p_val : double, default 0.05
            select a p-value, e.g., '0.05', '0.01', '0.001'.
    plot :  boolean, default True
            plot the output.
    
    Returns
    -------
    data : pandas dataframe
    
    Examples
    --------
    >>> dpl.series_corr(ca533, "CAM191", prewhiten=False, corr="Pearson", bin_floor=10)    
    
    References
    ----------
    .. [1] https:/opendendro.org/dplpy-man/#series_corr
    
    """
    # Check types of inputs
    if not isinstance(data, pd.DataFrame):
        errorMsg = "Expected dataframe input, got " + str(type(data)) + " instead."
        raise TypeError(errorMsg)
    
    if not isinstance(series_name, str):
        errorMsg = "Expected string input as series name, got " + str(type(series_name)) + " instead."
        raise TypeError(errorMsg)

    if series_name not in data.columns:
        errorMsg = "Series named " + series_name + " not found in provided dataframe."
        raise ValueError(errorMsg)

    rwi_data = detrend(data, fit="horizontal", plot=False)
 
    # drop nans, prewhiten series if necessary
    df_start = pd.DataFrame(index=pd.Index(data.index))
    to_concat = [df_start]
    for series in rwi_data:
        nullremoved_data = rwi_data[series].dropna()
        if prewhiten is True:
            res = ar_func_series(nullremoved_data, get_ar_lag(nullremoved_data))
            offset = len(nullremoved_data) - len(res)
            to_concat.append(pd.Series(data=res, name=series, index=nullremoved_data.index.to_numpy()[offset:]))
        else:
            to_concat.append(nullremoved_data)
    ready_series = pd.concat(to_concat, axis=1)
    ready_series = ready_series.rename_axis(data.index.name)

    removed = ready_series.pop(series_name)
    new_chron = chron(ready_series, plot=False)["Mean RWI"]

    inp = pd.concat([removed, new_chron], axis=1, join='inner')
    correlate(inp, corr)

    data_first = data.first_valid_index()
    data_last = data.last_valid_index()
    ser_first = removed.first_valid_index()
    ser_last = removed.last_valid_index()

    start, end = get_rel_range(data_first, data_last, ser_first, ser_last, bin_floor, seg_length)
    
    plt.style.use('seaborn-v0_8-darkgrid')
    wid = max((end - start)//30, 1)
    hei = 10
    base_corr = get_crit(p_val)
    
    dimensions = (wid, hei)
    plt.figure(num=1, figsize=(dimensions))
    plt.grid(True)

    years = []
    corrs = []

    second_plot = []
    # Find correlations at segment whose year is at center. So segments should be from i-25 to i+25, where i starts from bin start
    for i in range(start, end):
        segment = removed.loc[i-(seg_length//2):i+(seg_length//2)-1]

        if segment.size != seg_length:
            continue
        seg_corr, flag, flag_data = compare_segment(segment, new_chron, seg_length, corr, p_val, slide=False)

        years.append(i)
        corrs.append(seg_corr)

        if (((i-start-(seg_length//2)) % seg_length == 0)  or (((i-start-(seg_length//2)) % seg_length) == seg_length//2)):
            seg_range = [i-seg_length//2, i+seg_length//2]
            seg_corr_y = [seg_corr, seg_corr]
            plt.plot(seg_range, seg_corr_y, color="k")
            second_plot.append(analyze_segment(segment, new_chron, seg_length, corr))

            
    plt.plot(years, corrs, color="k")
    plt.plot([years[0]-seg_length, years[-1]+seg_length], [base_corr, base_corr], linestyle="dashed", color="k")
    plt.xlim([years[0]-seg_length, years[-1]+seg_length])
    plt.xlabel("Year")
    plt.ylabel("Correlation")

    plt.figure(num=2)
    plt.style.use('_mpl-gallery')
    cols = 5
    rows = (len(second_plot) // 5) + 1
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(14,14))

    j = 0
    for lags in second_plot:
        row = j // 5
        col = j % 5
        x_vals = np.arange(-5, 6, 1)
        y_vals = lags

        axes[row][col].stem(x_vals, y_vals)
        axes[row][col].plot([-5, 5], [base_corr, base_corr], linestyle="dashed")
        axes[row][col].plot([-5, 5], [-base_corr, -base_corr], linestyle="dashed")
        axes[row][col].set_xlabel('Lag')
        axes[row][col].set_ylabel('Correlation')
        axes[row][col].set_title(str(start + ((seg_length//2)*j)) + '.' + str(start + ((seg_length//2)*j) + seg_length - 1))
        axes[row][col].set_xlim([-6, 6])
        axes[row][col].set_ylim([-0.5, 1])
        j += 1
    
    while j < (rows*cols):
        row = j // 5
        col = j % 5
        axes[row][col].set_axis_off()
        j += 1

    fig.tight_layout()
    plt.show()

# Gets correlation data for segments lagged by 5 years forwards and backwards.
def analyze_segment(segment, new_chron, slide_period, correlation_type):
    if segment.size < slide_period:
        return
    series_name = segment.name
    data = pd.concat([segment, new_chron], axis=1, join='inner')
    
    segment_data = []

    for shift in range(-5, 6):
        shifted = data[series_name].copy(deep=False)
        shifted.index += shift
        overlapping_df = pd.concat([shifted, new_chron], axis=1, join='inner').dropna()
        if overlapping_df.size == slide_period * 2:
            new_coeff = correlate(overlapping_df, correlation_type)
            segment_data.append(new_coeff)
        else:
            segment_data.append(0)
    return segment_data

# Finds the effective first and last year of crossdating for the series depending on the
# values of bin floor and segment length.
def get_rel_range(data_first, data_last, series_first, series_last, bin_floor, seg_len):
    overlap = seg_len/2

    if bin_floor == 0 or data_first % bin_floor == 0:
        start = data_first
    else:
        start = (int(data_first/bin_floor) * bin_floor) + bin_floor

    rel_start = 9999
    rel_stop = 0
    i = start
    while i+seg_len-1 < data_last:
        if i >= series_first:
            rel_start = min(rel_start, i)
            if i <= series_last:
                rel_stop = max(rel_stop, i+seg_len-1)
            else:
                break
        i += overlap

    return int(rel_start), int(rel_stop)
