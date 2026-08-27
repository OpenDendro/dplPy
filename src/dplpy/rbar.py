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

def pairwise_corr_mean(data, method="pearson", min_overlap=None, strict=False):
    """Mean of the off-diagonal pairwise correlations between the columns of
    ``data`` (self-correlations excluded by setting the diagonal to NaN).

    When ``min_overlap`` is given, a series pair is counted only if its number of
    overlapping (both-present) years passes the threshold -- strictly greater
    (``strict=True``, ARSTAN's Briffa n>20) or at least (``strict=False``, the
    moving-window rbar). This is the single home for the correlation-matrix ->
    overlap-mask -> mean-of-means step shared by the rbar variants.
    """
    # corr.to_numpy() can be a read-only view under numpy 2 / copy-on-write, so
    # fill_diagonal needs an explicit writable copy, not the view pandas hands back.
    corr = data.corr(method)
    arr = corr.to_numpy(copy=True)
    np.fill_diagonal(arr, np.nan)
    corr = pd.DataFrame(arr, index=corr.index, columns=corr.columns)
    if min_overlap is not None:
        presence = data.notnull().astype("int")
        overlap = presence.transpose() @ presence
        corr = corr.where(overlap > min_overlap if strict else overlap >= min_overlap)
    return corr.mean().mean()


def mean_series_intercorrelation(data_set, corr_type, min_seg_ratio, apply_mask=True):
    # apply_mask=True is the moving-window rbar (dplR's rbarWinLength): a series
    # pair needs at least min_seg_ratio of the window's years overlapping to
    # count. apply_mask=False is the overall rbar constant, which dplR does not
    # filter this way -- a plain pairwise-complete mean.
    min_overlap = data_set.shape[0] * min_seg_ratio if apply_mask else None
    return pairwise_corr_mean(data_set, corr_type, min_overlap=min_overlap, strict=False)
