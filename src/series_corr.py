# Works, but still in development to be made more efficient.
# Analyzes the crossdating of one series compared to the master chronology

from detrend import detrend
from autoreg import ar_func_series
from chron import chron
from xdate import get_ar_lag, correlate, compare_segment
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt

def series_corr(data, series_name, prewhiten=True, corr="Spearman", seg_length=50, bin_floor=100, p_val=0.05, plot=True):
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

    removed = ready_series.pop(series_name)
    new_chron = chron(ready_series, plot=False)["Mean RWI"]

    inp = pd.concat([removed, new_chron], axis=1, join='inner')
    correlate(inp, corr)
    start = removed.first_valid_index()
    end = removed.last_valid_index()

    plt.style.use('seaborn-darkgrid')
    wid = max((end - start)//30, 1)
    hei = 10
    
    dimensions = (wid, hei)
    plt.figure(figsize=(dimensions))
    plt.grid(True)

    years = []
    corrs = []

    for i in range(start, end):
        segment = removed.loc[i:i+seg_length-1]

        if segment.size != seg_length:
            break
        seg_corr, flag, flag_data = compare_segment(segment, new_chron, seg_length, corr, p_val, slide=False)

        if (i - start) % seg_length == 0:
            seg_range = [i-seg_length//2, i+seg_length//2]
            seg_corr_y = [seg_corr, seg_corr]
            plt.plot(seg_range, seg_corr_y, color="k")
        else:
            years.append(i)
            corrs.append(seg_corr)
    plt.plot(years, corrs, color="k")

    plt.xlabel("Year")
    plt.ylabel("Correlation")
    plt.show()
