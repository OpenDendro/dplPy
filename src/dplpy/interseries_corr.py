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
# >>> dpl.interseries_corr(data)["mean"]
# >>> dpl.interseries_corr(data, prewhiten=False, corr="pearson")["table"]

from .xdate import normalize_for_crossdating, _row_mean, _corr_pval
from .tbrm import tbrm_rows

import numpy as np
import pandas as pd
from ._validate import _require_dataframe, _normalize_corr, _require_year_index, _warn


def interseries_corr(data: pd.DataFrame, prewhiten=True, biweight=True,
                     corr="spearman", ar_max=10, first_aic_min=True):
    """Mean interseries correlation (COFECHA's headline statistic).

    For each series, correlate it against a master chronology built from every other
    series (leave-one-out), then average. This is dplR's ``interseries.cor()``, and
    is distinct from rbar (``rwi_stats``), which averages the pairwise correlations
    between individual series.

    Expects raw ring widths, not indices: each series is normalized by its own mean
    and, by default, prewhitened before the correlations are computed (the same
    transform as ``dpl.xdate``, so the per-series values equal
    ``xdate()["overall"]["rho"]``).

    Parameters
    ----------
    data : pandas.DataFrame
        raw ring widths (as from ``dpl.readers``), years as the index, series as
        columns.
    prewhiten : boolean, default True
        prewhiten each series with an autoregressive model before correlating
        (``autoreg._prewhiten``, Yule-Walker, first local AIC minimum up to ``ar_max``).
    biweight : boolean, default True
        build each leave-one-out master with a Tukey biweight robust mean rather than
        the arithmetic mean.
    corr : {"spearman", "pearson", "kendall"}, default "spearman"
        correlation type (case-insensitive).
    ar_max : int or None, default 10
        AR-order ceiling for prewhitening; ``None`` uses dplR's ``floor(10*log10(n))``.
    first_aic_min : bool, default True
        AR order at the first local AIC minimum (ARSTAN rule) rather than the global
        minimum; ``ar_max=None, first_aic_min=False`` is dplR's ``interseries.cor``.

    Returns
    -------
    dict
        ``mean`` -- the collection-wide mean interseries correlation (3 decimals,
        NaN-skipping); ``table`` -- a DataFrame indexed by series (in the input's
        column order) with columns ``rho`` (3 decimals) and ``p_val`` (one-sided);
        ``n_series`` -- the number of series with a correlation; ``too_short`` --
        the names of series that overlapped their master by fewer than 3 years and
        were given NaN.

    Examples
    --------
    >>> import dplpy as dpl
    >>> res = dpl.interseries_corr(dpl.readers("series.rwl"))
    >>> res["mean"], res["table"]

    References
    ----------
    .. [1] https://rdrr.io/cran/dplR/man/interseries.cor.html
    .. [2] https://opendendro.github.io/dplPy/reference/crossdating/#interseries_corr
    """
    _require_dataframe(data)
    data = _require_year_index(data, "data")
    method = _normalize_corr(corr)

    ready_series = normalize_for_crossdating(data, prewhiten, ar_max=ar_max,
                                             first_aic_min=first_aic_min)

    # Work on a dense (years x series) matrix so each series' leave-one-out master
    # is a single vectorized robust mean rather than a full chron() rebuild
    # (previously O(n_series^2 x years)). tbrm_rows reproduces chron(biweight=True)
    # ["std"] exactly; _row_mean reproduces the arithmetic (biweight=False) mean.
    names = list(ready_series.columns)
    M = ready_series.to_numpy(dtype=float)
    aggregate = tbrm_rows if biweight else _row_mean

    rhos, p_values, too_short = [], [], []
    for series_name in data.columns:
        if series_name not in names:                        # all-NaN column, dropped upstream
            rhos.append(np.nan)
            p_values.append(np.nan)
            too_short.append("%s (0 yrs overlap)" % series_name)
            continue
        i = names.index(series_name)
        keep = np.ones(M.shape[1], dtype=bool)
        keep[i] = False                                     # leave this series out
        master = aggregate(M[:, keep])                      # master of all the others
        series = M[:, i]
        n_overlap = int((~np.isnan(series) & ~np.isnan(master)).sum())

        # Too few overlapping years to correlate: an empty series, or one sharing
        # fewer than 3 years with its master (which can happen after prewhitening
        # drops a series' first few values, or at the ends of a record). scipy would
        # either raise (pearsonr, n<2) or return a spurious +/-1 (spearman, n=2), so
        # record NaN and name the series instead -- matching series_corr and dplR.
        if n_overlap < 3:
            rhos.append(np.nan)
            p_values.append(np.nan)
            too_short.append("%s (%d yr%s overlap)"
                             % (series_name, n_overlap, "" if n_overlap == 1 else "s"))
            continue

        # one-sided (alternative="greater"), matching dplR's cor.test(...): a
        # series is expected to correlate positively with a chronology built
        # largely from its own signal.
        rho, pv = _corr_pval(series, master, method)
        rhos.append(round(rho, 3) if np.isfinite(rho) else np.nan)
        p_values.append(pv)

    if too_short:
        _warn("interseries_corr: %d series had fewer than 3 years overlapping the "
              "master and were given an NaN correlation: %s"
              % (len(too_short), ", ".join(too_short)))

    table = pd.DataFrame(data={"rho": rhos, "p_val": p_values},
                         index=pd.Index(list(data.columns), name="series"))
    mean_corr = round(float(table["rho"].mean()), 3) if table["rho"].notna().any() else np.nan
    return {"mean": mean_corr, "table": table,
            "n_series": int(table["rho"].notna().sum()),
            "too_short": [t.split(" (")[0] for t in too_short]}
