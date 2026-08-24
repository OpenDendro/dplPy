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

# Title: interseries_cor.py
# Project: OpenDendro dplPy
# Description: Calculates the mean interseries correlation -- the correlation
#              between each series in a dataset and a master chronology built
#              from every other series (leave-one-out principle). This is the
#              statistic commonly reported by COFECHA and by dplR's
#              interseries.cor(), and is a distinct quantity from rbar (see
#              rbar.py): rbar is the mean of pairwise correlations between
#              every series and every OTHER series individually, while the
#              interseries correlation here is the mean of the correlation
#              between each series and the composite chronology of the rest.
#
# example usage from Python Console:
# >>> import dplpy as dpl
# >>> data = dpl.readers("../tests/data/csv/file.csv")
# >>> dpl.interseries_cor(data)
# >>> dpl.interseries_cor(data, prewhiten=False, corr="Pearson")

from .chron import chron
from .xdate import normalize_for_crossdating

import pandas as pd
import scipy.stats


def interseries_cor(data: pd.DataFrame, prewhiten=True, biweight=True, corr="Spearman"):
    """Mean interseries correlation

    Extended Summary
    ----------------
    For every series in the dataset, calculates the correlation between that
    series and a master chronology built from every *other* series in the
    dataset (leave-one-out principle). This is the same quantity commonly
    reported by COFECHA as the mean interseries correlation, and computed by
    dplR's interseries.cor().

    This is a fundamentally different statistic from rbar (see rbar.py and
    chron_stabilized.py): rbar is the mean of the pairwise correlations
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
    >>> dpl.interseries_cor(data)
    >>> dpl.interseries_cor(data, prewhiten=False, corr="Pearson")

    References
    ----------
    .. [1] https://rdrr.io/cran/dplR/man/interseries.cor.html

    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Expected dataframe input, got " + str(type(data)) + " instead.")

    if corr not in ("Spearman", "Pearson"):
        raise ValueError("corr must be either 'Spearman' or 'Pearson', got '" + str(corr) + "'.")

    ready_series = normalize_for_crossdating(data, prewhiten)

    series_names = []
    interseries_corrs = []
    p_values = []

    for series_name in sorted(ready_series.columns):
        removed = ready_series[series_name].dropna()
        # Non-destructive: unlike series_corr()'s single-series pop/restore,
        # every series needs its own turn being left out here, so the full
        # set is never mutated.
        others = ready_series.drop(columns=[series_name])

        master_chron = chron(others, biweight=biweight, plot=False)["Mean RWI"]

        inp = pd.concat([removed, master_chron], axis=1, join="inner").dropna()

        # alternative="greater": matches dplR's cor.test(..., alternative="greater")
        # -- a one-sided test, since a series is expected to correlate
        # positively with a chronology built largely from its own signal.
        if corr == "Spearman":
            test_result = scipy.stats.spearmanr(inp.iloc[:, 0], inp.iloc[:, 1], alternative="greater")
        else:
            test_result = scipy.stats.pearsonr(inp.iloc[:, 0], inp.iloc[:, 1], alternative="greater")

        series_names.append(series_name)
        interseries_corrs.append(round(test_result.statistic, 3))
        p_values.append(test_result.pvalue)

    result_df = pd.DataFrame(
        data={"interseries_cor": interseries_corrs, "p_val": p_values},
        index=pd.Index(series_names, name="series"),
    )

    return result_df
