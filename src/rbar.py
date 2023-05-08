import pandas as pd
import numpy as np
from detrend import detrend
from chron import chron
from xdate import correlate

def common_interval(data):
    # split out the variables for clarity - probably don't need to do this
    year = data.index.to_numpy() # this is the year vector
    crn = data.iloc[:,:] # these are the chronologies

    # get the size of some things - 23 series covering up to 286 years
    num_years, num_series = crn.shape

    # across-column sum of non-NaN values to get the sample size = sample size
    sample_depth = np.sum(~np.isnan(crn), axis=1)
    # allocate
    N = np.full((num_years, num_years), np.nan) # square matrix with dimensions the length of the series (which reflects both starting year and possible block length)

    # loop over - this is a straight port from my MATLAB, possibly inefficient
    for i in range(num_years):  # effectively, looping over from 1 to the maximum length of the series in the data as potential lengths of a common interval block
        # define a block size, using smaller and smaller blocks as you get toward the last year of the series ... this loop therefore gets shorter as block size i gets larger ...
        for j in range(num_years - i):
            # for a starting year j and block length i, the smallest number of chronologies in that particular block
            N[j, i] = np.min(sample_depth[j:j+i+1])

    # pointwise multiplication of two square matrices - this essentially convolves sample size and block length to get number of pairwise comparisons possible
    N0 = N * np.tile(np.arange(num_years) + 1, (num_years, 1))

    # row (startyear) and column (block length) position of maximum value - is this an OK way to do this? tried other things that didn't work
    startYear, windowWidth = np.where(N0 == np.nanmax(N0))
    # this gives the same answer as MATLAB - 1828 to 1982 common interval
    return year[int(startYear)], year[int(startYear+windowWidth-1)]

# can use osborn, frank or 67spline methods
def rbar(data, start, end, method="osborn", seg_length=50, seg_overlap=0.5, corr_type="Spearman"):
    # how we deal with nans will depend on method chosen for finding rbar. 
    # drop all series with nans for osborn, but drop only if they are not up to fraction of seg_length for frank
    rel_series = data.loc[start:end].dropna(axis=1)
    series_names = rel_series.columns

    # osborn assumes all series are overlapping along the entire period
    if method == "osborn":
        count = 0
        total_corr = 0
        for i in range(len(series_names)):
            for j in range(i+1, len(series_names)):
                if j < len(series_names):
                    total_corr += correlate(rel_series[[series_names[i], series_names[j]]], corr_type)
                    count += 1
        return [total_corr/count] * (end-start+1)
    elif method == "frank":
        results = []
        for i in range(start, end, seg_length):
            rel_segment = rel_series.loc[i:i+seg_length-1]
            rel_series_names = rel_series.columns
            if (rel_segment.shape[0] < seg_length):
                results += [1] * min(seg_length, end-i+1)
                continue
            count = 0
            total_corr = 0
            for j in range(len(rel_series_names)):
                for k in range(j, len(rel_series_names)):
                    if k < len(rel_series_names):
                        total_corr += correlate(rel_segment[[rel_series_names[j], rel_series_names[k]]], corr_type)
                        count += 1
            results += [total_corr/count] * seg_length
        return results
    elif method == "67spline":
        signs = rel_series.where(rel_series < 0, 1)
        signs = signs.where(signs >= 0, -1)
        rel_series = rel_series.abs()
        rel_series_rwi = detrend(rel_series, fit="spline")
        res_frame = rel_series_rwi * signs
        return chron(res_frame, plot=False)['Mean RWI'].tolist()

    return None