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
# Title: rbar.py
# Description: Contains functions for finding best interval of overlapping series over a long
#              period of years, and calculating rbar constant for a dataset over this best
#              period of overlap

import numpy as np
from detrend import detrend
from chron import chron

# common_interval finds a range of years in the provided dataframe where there is maximum overlap between series over the longest period of years.
def common_interval(data):
    year = data.index.to_numpy() # this is the year vector
    crn = data.iloc[:,:] # these are the chronologies

    num_years = crn.shape[0]

    
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

    start_year, window_width = np.where(N0 == np.nanmax(N0))
    # this gives the same answer as MATLAB - 1828 to 1982 common interval
    return year[int(start_year)], year[int(start_year+window_width-1)]

  
# rbar returns a list of constants to multiply with each mean value generated for a range of years from a mean value chronology.
# Can use osborn, frank and 67spline methods to generate rbar values.
# Will be updated in the future to prioritize number of series, number of years or both. Currently attempts to do both.
def get_running_rbar(data, min_seg_ratio, method="osborn", corr_type="pearson"):
    # how we deal with nans will depend on method chosen for finding rbar. 
    # drop all series with nans for osborn, but drop only if they are not up to fraction of seg_length for frank

    # Osborn assumes all series are overlapping along the entire period. Drops none
    if method == "osborn":
        r_bar = mean_series_intercorrelation(data, corr_type, min_seg_ratio)
        return r_bar
    
    elif method == "frank":
        rel_data = data.copy()
        drop_columns = []

        # Identify columns that need to be dropped and drop them
        
        for column in rel_data:
            num_valid_elems = rel_data[column].size
            if num_valid_elems/data.shape[0] < min_seg_ratio:
                drop_columns.append(column)
        rel_data = rel_data.drop(columns=drop_columns)

        r_bar = mean_series_intercorrelation(rel_data, corr_type, min_seg_ratio)

        return r_bar
    
    elif method == "67spline":
        # probably need to update this
        rel_data = data.copy()
        signs = rel_data.where(rel_data < 0, 1)

        signs = signs.where(signs >= 0, -1)
        rel_series = rel_series.abs()
        rel_series_rwi = detrend(rel_series, fit="spline")
        res_frame = rel_series_rwi * signs
        return chron(res_frame, plot=False)['Mean RWI'].tolist()

    return None

def mean_series_intercorrelation(data_set, corr_type, min_seg_ratio):
    presence_df = data_set.notnull().astype('int')
    trans_presence_df = presence_df.transpose()

    corr_mat = data_set.corr(corr_type)
    np.fill_diagonal(corr_mat.values, np.nan)

    overlap_mat = trans_presence_df @ presence_df

    min_overlap = data_set.shape[0] * min_seg_ratio

    corr_mat.where(overlap_mat > min_overlap, inplace=True)
    
    return corr_mat.mean().mean()
