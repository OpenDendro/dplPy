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

# Title: sss.py
# Project: OpenDendro dplPy
# Description: Subsample signal strength (SSS) of Wigley et al. (1984): how well
#              the subsample of series available in each year represents the
#              full N-series chronology. A port of dplR's sss(). Unlike the
#              running EPS from rwi_stats_running(), SSS holds the mean
#              interseries correlation (rbar) and the full sample size N fixed at
#              their whole-record values and varies only the per-year sample
#              depth -- exactly as dplR does.
#
#              As Buras (2017) emphasizes, SSS (not EPS) is the statistic Wigley
#              et al. (1984) intended for judging the loss of a reconstruction's
#              explanatory power as sample depth declines back in time; the
#              often-cited 0.85 example was an SSS illustration. No threshold is
#              applied here -- interpretation is left to the analyst.
#
# example usage from Python Console:
# >>> import dplpy as dpl
# >>> data = dpl.readers("../tests/data/csv/file.csv")
# >>> rwi = dpl.detrend(data, plot=False)
# >>> dpl.sss(rwi)                                  # one core per tree
# >>> dpl.sss(rwi, ids=dpl.read_ids(data, stc=(3, 2, 1)))

import numpy as np
import pandas as pd

from .rwi_stats import rwi_stats, _resolve_tree_mapping
from .common_interval import apply_common_interval


def sss(rwi_data: pd.DataFrame, ids=None, corr="Spearman", zero_is_missing=True,
        common_interval=None):
    """Subsample signal strength (SSS) as a per-year series.

    Extended Summary
    ----------------
    For each year, computes how well the subsample of series present in that
    year represents the full N-series chronology, following Wigley et al.
    (1984):

        SSS(t) = n(t) * (1 + (N - 1) * rbar) / (N * (1 + (n(t) - 1) * rbar))

    where N is the total number of trees in the record, rbar is the effective
    mean interseries correlation (both taken, fixed, from a whole-record
    dpl.rwi_stats() call), and n(t) is the number of trees (or cores, if no
    tree IDs are given) present in year t. SSS rises toward 1 as n(t)
    approaches N.

    This mirrors dplR's sss(): the correlation and full sample size are held
    constant and only the per-year sample depth varies. It is therefore
    distinct from the running EPS produced by dpl.rwi_stats_running(), which
    recomputes rbar within each moving window.

    No acceptance threshold is applied. As Buras (2017) clarifies, SSS is the
    statistic Wigley et al. (1984) actually intended for assessing declining
    reconstruction skill back in time (the frequently cited 0.85 value was an
    illustrative SSS example, not a rule); whether a given SSS is adequate is a
    judgement for the analyst.

    Parameters
    ----------
    rwi_data : pandas dataframe
        detrended ring-width indices (e.g. from dpl.detrend()), years as index
        and series as columns.
    ids : dict, pandas dataframe, or None, default None
        tree/core structure (see dpl.rwi_stats() / dpl.read_ids()). When given,
        both N and the per-year sample depth are counted in trees; when None,
        every series is its own tree and they are counted in cores.
    corr : str, default "Spearman"
        correlation type used for the underlying rbar ("Spearman" or "Pearson").
    zero_is_missing : boolean, default True
        treat exact zeros as missing (applied consistently to the rbar
        calculation and the per-year sample-depth count).
    common_interval : str, (int, int), or None, default None
        restrict to a common interval before computing SSS: 'series', 'years'
        or 'both' selects one via dpl.common_interval(), and a
        (start_year, end_year) pair restricts to that span. rbar, N, n(t) and
        the returned years are then all taken from that trimmed block --
        equivalent to dplR's sss(common.interval(rwi)). None (default)
        reproduces dplR's plain sss() over the full record.

    Returns
    -------
    result : pandas Series named "sss", indexed by year, with the per-year
        subsample signal strength (over the common interval if one was chosen).

    Examples
    --------
    >>> import dplpy as dpl
    >>> rwi = dpl.detrend(dpl.readers("../tests/data/csv/file.csv"), plot=False)
    >>> dpl.sss(rwi)

    References
    ----------
    .. [1] https://rdrr.io/cran/dplR/man/sss.html
    .. [2] Wigley, Briffa & Jones (1984), J. Climate Appl. Meteor., 23, 201-213.
    .. [3] Buras (2017), Dendrochronologia, 44, 130-132.

    """
    if not isinstance(rwi_data, pd.DataFrame):
        raise TypeError("Expected dataframe input, got " + str(type(rwi_data)) + " instead.")

    # Optionally restrict to a common interval first. This matches dplR's own
    # recipe for a common-interval SSS -- sss(common.interval(rwi)) -- where the
    # correlation, the reference sample size N, the per-year depth n(t), and the
    # returned years all come from the same trimmed block (so SSS stays <= 1).
    # None (default) reproduces dplR's plain sss() on the full record.
    rwi_data = apply_common_interval(rwi_data, common_interval)

    # Whole-record rbar and N, from the same engine (and same defaults) as dplR.
    stats = rwi_stats(rwi_data, ids=ids, corr=corr,
                      zero_is_missing=zero_is_missing, round_decimals=None)
    rbar = float(stats.iloc[0]["rbar_eff"])
    big_n = float(stats.iloc[0]["n_trees"])

    # Per-year sample depth, with the same zero handling used for rbar.
    data = rwi_data.copy()
    if zero_is_missing:
        data = data.where(data != 0)

    if ids is None:
        # one core per tree: depth = number of cores present each year
        n_t = data.notna().sum(axis=1).to_numpy(dtype=float)
    else:
        tree_of = _resolve_tree_mapping(ids, list(data.columns))
        present = data.notna()
        tree_labels = pd.Series({c: tree_of[c] for c in data.columns})
        # (trees x years) True if any of a tree's cores is present that year
        tree_present = present.T.groupby(tree_labels).any()
        n_t = tree_present.sum(axis=0).to_numpy(dtype=float)

    sss_vals = (n_t * (1 + (big_n - 1) * rbar)) / (big_n * (1 + (n_t - 1) * rbar))

    return pd.Series(sss_vals, index=rwi_data.index, name="sss")
