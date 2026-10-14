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

import warnings

import numpy as np
import pandas as pd
from ._validate import _require_dataframe, _normalize_corr
import scipy.stats


def interseries_corr(data: pd.DataFrame, prewhiten=True, biweight=True, corr="Spearman"):
    """Mean interseries correlation (COFECHA's headline statistic).

    For each series, correlate it against a master chronology built from every other
    series (leave-one-out), then average. This is dplR's ``interseries.cor()``, and
    is distinct from rbar (``rwi_stats``), which averages the pairwise correlations
    between individual series.

    Expects raw ring widths, not indices: each series is normalized by its own mean
    and, by default, prewhitened before the correlations are computed.

    Parameters
    ----------
    data : pandas.DataFrame
        raw ring widths (as from ``dpl.readers``), years as the index, series as
        columns.
    prewhiten : boolean, default True
        prewhiten each series with an autoregressive model before correlating.
    biweight : boolean, default True
        build each leave-one-out master with a Tukey biweight robust mean rather than
        the arithmetic mean.
    corr : {"Spearman", "Pearson"}, default "Spearman"
        correlation type.

    Returns
    -------
    mean_corr : float
        collection-wide mean interseries correlation (3 decimals).
    result : pandas.DataFrame
        per-series interseries correlation and one-sided p-value.

    Examples
    --------
    >>> import dplpy as dpl
    >>> mean_corr, per_series = dpl.interseries_corr(dpl.readers("series.rwl"))

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
    too_short = []

    for series_name in sorted(names):
        i = names.index(series_name)
        keep = np.ones(M.shape[1], dtype=bool)
        keep[i] = False                                     # leave this series out
        master = aggregate(M[:, keep])                      # master of all the others
        series = M[:, i]
        ok = ~np.isnan(series) & ~np.isnan(master)          # overlap (both present)
        n_overlap = int(ok.sum())

        # Too few overlapping years to correlate: an empty series, or one sharing
        # fewer than 3 years with its master (which can happen after prewhitening
        # drops a series' first few values, or at the ends of a record). scipy would
        # either raise (pearsonr, n<2) or return a spurious +/-1 (spearman, n=2), so
        # record NaN and name the series instead -- matching series_corr and dplR.
        if n_overlap < 3:
            series_names.append(series_name)
            interseries_corrs.append(np.nan)
            p_values.append(np.nan)
            too_short.append("%s (%d yr%s overlap)"
                             % (series_name, n_overlap, "" if n_overlap == 1 else "s"))
            continue

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

    if too_short:
        warnings.warn(
            "interseries_corr: %d series had fewer than 3 years overlapping the "
            "master and were given an NaN correlation: %s"
            % (len(too_short), ", ".join(too_short)))

    result_df = pd.DataFrame(
        data={"interseries_corr": interseries_corrs, "p_val": p_values},
        index=pd.Index(series_names, name="series"),
    )

    # The number users most often want is the collection-wide mean interseries
    # correlation (COFECHA's headline statistic), so return it first, with the
    # per-series table second. Both correlations are reported to 3 decimals.
    mean_corr = round(float(result_df["interseries_corr"].mean()), 3)
    return mean_corr, result_df
