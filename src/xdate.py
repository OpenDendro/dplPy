from detrend import detrend
from autoreg import ar_func_series
from chron import chron
import pandas as pd
import numpy as np
import scipy

def xdate(data, prewhiten=True, corr="Spearman", ar_lag=5, slide_period=50, bin_floor=100):
    # Identify first and last valid indexes, for separating into bins.

    print("Bins: ")
    bins = get_bins(data.first_valid_index(), data.last_valid_index(), bin_floor, slide_period)
    print(bins)

    rwi_data = detrend(data, fit="horizontal", plot=False)

    # if detrending returns error, raise to output
    if isinstance(rwi_data, ValueError) or isinstance(rwi_data, TypeError):
        raise rwi_data

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
        new_chron = chron(ready_series_copy, plot=False)["Mean RWI"]

        # edit removed and new to make sure they are the same size
        inp = pd.concat([removed, new_chron], axis=1, join='inner')

        series_corr[series] = correlate(inp, corr)
        
        for range in bins:
            start = int(range.split("-")[0])
            end = int(range.split("-")[1])
            if start >= removed.first_valid_index() and end <= removed.last_valid_index():
                segment = removed.loc[start:end]
                compare_segment(segment, new_chron, slide_period, corr)

        ready_series_copy[series] = removed
    return series_corr

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

def correlate(data, type):
    if type == "Spearman":
        return scipy.stats.spearmanr(data, axis=0).correlation
    elif type == "Pearson":
        return np.corrcoef(data, rowvar=False)[0, 1]
    

def compare_segment(segment, new_chron, slide_period, correlation_type, left_most=-10, right_most=10):
    if segment.size < slide_period:
        return
    series_name = segment.name
    data = pd.concat([segment, new_chron], axis=1, join='inner')
    original = correlate(data, correlation_type)

    if original < 0.2:
        print("Dating issue with", segment.name, "in the range of", segment.first_valid_index(), "to", segment.last_valid_index())

    best_lag = 0
    best_coeff = original

    for shift in range(left_most, right_most+1):
        shifted = data[series_name]
        shifted.index += shift
        overlapping_df = pd.concat([shifted, new_chron], axis=1, join='inner').dropna()
        if overlapping_df.size == slide_period * 2:
            new_coeff = np.corrcoef(overlapping_df, rowvar=False)[0, 1]
            if new_coeff > best_coeff:
                best_lag = shift
                best_coeff = new_coeff
    warning = True
    """
    if (best_lag != 0 and abs(best_coeff - original) >= 0.1):
        print("\nUnexpected results for", segment.name, "in the range of", segment.first_valid_index(), "to", segment.last_valid_index())
        print("Original correlation was", original)
        print("Best correlation was at lag", best_lag, "and correlation was", best_coeff, "\n")
        warning = True
    else:
        warning = False
    """
    return (best_lag, best_coeff, warning)