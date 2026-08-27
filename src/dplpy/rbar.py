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
import pandas as pd

# NOTE: the earlier rectangle-maximising common_interval() that lived here has
# been superseded by dplpy.common_interval (see common_interval.py), a faithful
# port of dplR's common.interval() offering the 'series', 'years' and 'both'
# selection strategies and validated exactly against dplR.


# rbar: the mean inter-series correlation over a window, used by
# chron_stabilized to adjust chronology variance. Osborn's definition -- all
# series assumed to overlap the period, none dropped -- is the one dplR/dplPy's
# chron.stabilized uses.
def get_running_rbar(data, min_seg_ratio, corr_type="pearson"):
    return mean_series_intercorrelation(data, corr_type, min_seg_ratio)

def mean_series_intercorrelation(data_set, corr_type, min_seg_ratio, apply_mask=True):
    # corr_mat.values / .to_numpy() can come back as a read-only view under
    # numpy 2 / pandas' copy-on-write, so np.fill_diagonal needs an explicit,
    # independent writable copy rather than the view pandas hands back.
    corr_mat = data_set.corr(corr_type)
    corr_arr = corr_mat.to_numpy(copy=True)
    np.fill_diagonal(corr_arr, np.nan)
    corr_mat = pd.DataFrame(corr_arr, index=corr_mat.index, columns=corr_mat.columns)

    if apply_mask:
        # Only used for the moving-window rbar (dplR's rbarWinLength): a series
        # pair needs at least min_seg_ratio of the window's years overlapping
        # to count. The overall rbar constant (apply_mask=False) is not
        # filtered this way in dplR -- it's a plain pairwise-complete mean.
        presence_df = data_set.notnull().astype('int')
        trans_presence_df = presence_df.transpose()
        overlap_mat = trans_presence_df @ presence_df
        min_overlap = data_set.shape[0] * min_seg_ratio
        corr_mat = corr_mat.where(overlap_mat >= min_overlap)

    return corr_mat.mean().mean()
