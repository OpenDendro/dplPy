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
# selection strategies.


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
    overlap-mask -> mean step shared by the rbar variants.

    The average is a FLAT mean over every valid off-diagonal entry of the
    (symmetric) matrix -- i.e. every retained series pair weighted equally --
    matching dplR's ``mean(corMat, na.rm = TRUE)`` in chron.stabilized(). It must
    NOT be a mean-of-column-means: the two coincide only when every series has the
    same number of retained partners, but once the overlap mask drops pairs
    unevenly across series (interior windows), a mean-of-column-means weights each
    series equally instead of each pair and drifts from dplR on those windows.
    The flat mean matches dplR's ``mean(corMat, na.rm = TRUE)``.
    """
    # corr.to_numpy() can be a read-only view under numpy 2 / copy-on-write, so
    # fill_diagonal needs an explicit writable copy, not the view pandas hands back.
    corr = data.corr(method)
    arr = corr.to_numpy(copy=True)
    np.fill_diagonal(arr, np.nan)
    if min_overlap is not None:
        presence = data.notnull().astype("int")
        overlap = (presence.transpose() @ presence).to_numpy()
        keep = overlap > min_overlap if strict else overlap >= min_overlap
        arr = np.where(keep, arr, np.nan)
    finite = arr[~np.isnan(arr)]
    return float(finite.mean()) if finite.size else np.nan


def mean_series_intercorrelation(data_set, corr_type, min_seg_ratio, apply_mask=True):
    # apply_mask=True is the moving-window rbar (dplR's rbarWinLength): a series
    # pair needs at least min_seg_ratio of the window's years overlapping to
    # count. apply_mask=False is the overall rbar constant, which dplR does not
    # filter this way -- a plain pairwise-complete mean.
    min_overlap = data_set.shape[0] * min_seg_ratio if apply_mask else None
    return pairwise_corr_mean(data_set, corr_type, min_overlap=min_overlap, strict=False)


def running_rbar_vector(data, win_length, min_seg_ratio, n_samps=None):
    """Moving-window rbar vector (Frank et al. 2006 "RUNNINGr"), padded at the
    ends, matching dplR's chron.stabilized().

    ``data`` is a years x series DataFrame (typically zero-mean). Returns a numpy
    array of length ``nrows``: the window rbar centred on each year, with the
    leading/trailing ``win_length/2`` positions padded with the nearest computed
    value, and NaN where the total sample depth is 0.

    This is the shared engine used by both chron_stabilized() and
    stabilize_chron() so the running rbar is computed identically in each.
    """
    num_years = data.shape[0]
    rbar_array = np.full(num_years, np.nan)
    target = win_length / 2 if win_length % 2 == 0 else (win_length - 1) / 2
    for i in range(num_years - win_length + 1):
        seg = data.iloc[i:i + win_length]
        if seg.shape[0] < win_length:
            continue
        rbar_array[int(i + target)] = get_running_rbar(seg, min_seg_ratio)

    if n_samps is None:
        n_samps = data.notnull().sum(axis=1).to_numpy()
    else:
        n_samps = np.asarray(n_samps)

    # pad the leading/trailing not-yet-computed runs with the nearest real value
    # (dplR pads with the first/last non-NA entry of movingRbarVec); a genuine
    # rbar of 0.0 is a valid value, so we detect "not computed" via NaN, not 0.
    valid = np.flatnonzero(~np.isnan(rbar_array))
    if valid.size > 0:
        rbar_array[:valid[0]] = rbar_array[valid[0]]
        rbar_array[valid[-1]:] = rbar_array[valid[-1]]

    # a year with zero total sample depth has no meaningful rbar, padded or not
    rbar_array[n_samps == 0] = np.nan
    return rbar_array
