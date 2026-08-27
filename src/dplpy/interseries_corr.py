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

# Title: interseries_corr.py
# Project: OpenDendro dplPy
# Description: Calculates the mean interseries correlation -- the correlation
#              between each series in a dataset and a master chronology built
#              from every other series (leave-one-out principle). This is the
#              statistic commonly reported by COFECHA and by dplR's
#              interseries.cor(), and is a distinct quantity from rbar (the
#              mean pairwise inter-series correlation reported by dpl.rwi_stats()):
#              rbar is the mean of pairwise correlations between
#              every series and every OTHER series individually, while the
#              interseries correlation here is the mean of the correlation
#              between each series and the composite chronology of the rest.
#
# example usage from Python Console:
# >>> import dplpy as dpl
# >>> data = dpl.readers("../tests/data/csv/file.csv")
# >>> dpl.interseries_corr(data)
# >>> dpl.interseries_corr(data, prewhiten=False, corr="Pearson")

from .xdate import normalize_for_crossdating, _row_mean
from .tbrm import tbrm_rows

import numpy as np
import pandas as pd
from ._validate import _require_dataframe, _normalize_corr
import scipy.stats


def interseries_corr(data: pd.DataFrame, prewhiten=True, biweight=True, corr="Spearman"):
    """Mean interseries correlation

    Extended Summary
    ----------------
    For every series in the dataset, calculates the correlation between that
    series and a master chronology built from every *other* series in the
    dataset (leave-one-out principle). This is the same quantity commonly
    reported by COFECHA as the mean interseries correlation, and computed by
    dplR's interseries.cor().

    This is a fundamentally different statistic from rbar (reported by
    dpl.rwi_stats() and used by dpl.chron_stabilized()): rbar is the mean of the
    pairwise correlations
    between every series and every OTHER series individually -- a stricter
    test to pass -- while the interseries correlation computed here is the
    mean of the correlation between each series and the composite chronology
    built from the rest. dplR's own documentation for interseries.cor()
    draws this same distinction explicitly.

    Each series is first normalized by dividing by its own mean (dplPy's
    "horizontal" detrend -- equivalent to dplR's normalize.xdate() without
    its optional Hanning-filter alternative, which dplPy does not currently
    implement), then optionally prewhitened with an autoregressive model
    (matching dplR's default). Note also that, unlike dplR's
    rwi.stats()/rwi.stats.running(), this function does not exclude series
    with very few (four or fewer) valid observations from contributing to
    other series' composite chronologies; for typical dendrochronological
    datasets this is not expected to matter, but a pathologically short
    series could be weighted differently here than in dplR.

    Parameters
    ----------
    data : pandas dataframe
        a dataframe of raw ring widths, as produced by dpl.readers(). Unlike
        chron() or chron_stabilized(), this function expects raw
        measurements rather than already-detrended ring-width indices -- it
        performs its own normalization internally, matching dplR's
        interseries.cor().
    prewhiten : boolean, default True
        whether to prewhiten each series with an autoregressive model before
        computing correlations.
    biweight : boolean, default True
        whether to use Tukey's biweight robust mean (rather than the
        arithmetic mean) when building each series' leave-one-out composite
        chronology.
    corr : str, default "Spearman"
        correlation type to use: "Spearman" or "Pearson".

    Returns
    -------
    result : pandas dataframe with one row per series (indexed by series
        name), containing the interseries correlation and its (one-sided,
        "greater") p-value.

    Examples
    --------
    >>> import dplpy as dpl
    >>> data = dpl.readers("../tests/data/csv/file.csv")
    >>> dpl.interseries_corr(data)
    >>> dpl.interseries_corr(data, prewhiten=False, corr="Pearson")

    References
    ----------
    .. [1] https://rdrr.io/cran/dplR/man/interseries.cor.html

    """
    _require_dataframe(data)

    method = _normalize_corr(corr, allowed=("spearman", "pearson"))

    ready_series = normalize_for_crossdating(data, prewhiten)

    # Work on a dense (years x series) matrix so each series' leave-one-out master
    # is a single vectorized robust mean rather than a full chron() rebuild
    # (previously O(n_series^2 x years)). tbrm_rows reproduces chron(biweight=True)
    # ["std"] exactly; _row_mean reproduces the arithmetic (biweight=False) mean.
    names = list(ready_series.columns)
    M = ready_series.to_numpy(dtype=float)
    aggregate = tbrm_rows if biweight else _row_mean

    series_names = []
    interseries_corrs = []
    p_values = []

    for series_name in sorted(names):
        i = names.index(series_name)
        keep = np.ones(M.shape[1], dtype=bool)
        keep[i] = False                                     # leave this series out
        master = aggregate(M[:, keep])                      # master of all the others
        series = M[:, i]
        ok = ~np.isnan(series) & ~np.isnan(master)          # overlap (both present)

        # alternative="greater": matches dplR's cor.test(..., alternative="greater")
        # -- a one-sided test, since a series is expected to correlate
        # positively with a chronology built largely from its own signal.
        if method == "spearman":
            test_result = scipy.stats.spearmanr(series[ok], master[ok], alternative="greater")
        else:
            test_result = scipy.stats.pearsonr(series[ok], master[ok], alternative="greater")

        series_names.append(series_name)
        interseries_corrs.append(round(test_result.statistic, 3))
        p_values.append(test_result.pvalue)

    result_df = pd.DataFrame(
        data={"interseries_corr": interseries_corrs, "p_val": p_values},
        index=pd.Index(series_names, name="series"),
    )

    return result_df
