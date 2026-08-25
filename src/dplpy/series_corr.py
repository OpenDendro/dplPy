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

# Date: 5/12/2023 (rewritten 2026)
# Author: Ifeoluwa Ale (original), OpenDendro
# Title: series_corr.py
# Description: Crossdating focused on ONE series vs a leave-one-out master,
#   mirroring dplR's corr.series.seg() (a moving correlation and per-segment
#   correlations) together with ccf.series.rwl() (the per-segment lag / COFECHA
#   table drawn as stem plots).
#
# >>> import dplpy as dpl
# >>> data = dpl.readers("../tests/data/csv/file.csv")
# >>> dpl.series_corr(data, "CAM011")

from .xdate import (normalize_for_crossdating, _row_biweight, _row_mean, _bin_bounds,
                    _corr_pval, get_bins, get_crit, _CORR_ALIASES)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def series_corr(data: pd.DataFrame, series_name: str, prewhiten=True,
                corr="spearman", seg_length=50, bin_floor=100, p_val=0.05,
                biweight=True, lag=5, make_plot=True):
    """Crossdate one series against the master built from all the others.

    Produces (and, by default, plots) a moving correlation of the series against
    the leave-one-out master, the per-segment correlations, and a per-segment
    lag table (COFECHA-style) showing how the correlation changes when the
    segment is shifted +/- ``lag`` years -- large off-zero peaks suggest a
    dating error in that segment.

    Parameters
    ----------
    data : pandas.DataFrame
        ring-width series (typically detrended RWI).
    series_name : str
        the series to examine.
    prewhiten, corr, seg_length, bin_floor, p_val, biweight, lag
        as in dpl.xdate() (seg_length is the segment length).
    make_plot : bool, default True
        draw the moving-correlation and lag-stem figures.

    Returns
    -------
    dict with keys ``moving_corr`` (Series), ``seg_corr`` (Series over bins),
    ``overall`` ((rho, p_val)), ``lag_table`` (DataFrame lags x bins) and
    ``bins``.

    References
    ----------
    .. [1] https:/opendendro.org/dplpy-man/#series_corr
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Expected dataframe input, got " + str(type(data)) + " instead.")
    if not isinstance(series_name, str):
        raise TypeError("Expected string input as series name, got " + str(type(series_name)) + " instead.")
    if series_name not in data.columns:
        raise ValueError("Series named " + series_name + " not found in provided dataframe.")
    method = _CORR_ALIASES.get(str(corr).strip().lower())
    if method is None:
        raise ValueError("corr must be 'spearman', 'pearson' or 'kendall'")

    ready = normalize_for_crossdating(data, prewhiten)
    first_year = int(ready.first_valid_index())
    last_year = int(ready.last_valid_index())
    years = np.arange(first_year, last_year + 1)
    ready = ready.reindex(years)

    names = list(ready.columns)
    idx = names.index(series_name)
    M = ready.to_numpy(dtype=float)
    good = np.array([np.sum(~np.isnan(M[:, i])) > 3 for i in range(M.shape[1])])
    keep = good.copy()
    keep[idx] = False
    master = (_row_biweight if biweight else _row_mean)(M[:, keep])
    series = M[:, idx]

    overall = _corr_pval(series, master, method)

    bins, _ = get_bins(first_year, last_year, bin_floor, seg_length)
    bin_bounds = [_bin_bounds(b) for b in bins]

    # per-segment correlation and lag table
    seg_corr = pd.Series(index=bins, dtype=float)
    lag_vec = list(range(-lag, lag + 1))
    lag_table = pd.DataFrame(index=["lag." + str(k) for k in lag_vec], columns=bins, dtype=float)
    for (lo, hi), blabel in zip(bin_bounds, bins):
        mask0 = (years >= lo) & (years <= hi)
        if mask0.sum() != seg_length or np.isnan(series[mask0]).any() or np.isnan(master[mask0]).any():
            continue
        seg_corr[blabel] = _corr_pval(series[mask0], master[mask0], method)[0]
        for k in lag_vec:
            m = (years >= lo + k) & (years <= hi + k)
            if m.sum() != seg_length or np.isnan(series[m]).any() or np.isnan(master[mask0]).any():
                continue
            lag_table.loc["lag." + str(k), blabel] = _corr_pval(series[m], master[mask0], method)[0]

    # moving correlation (window seg_length, step 1, stored at window centre)
    seg_lag = seg_length // 2
    mov_years, mov_corr = [], []
    for t in range(0, len(years) - seg_length + 1):
        s = series[t:t + seg_length]
        m = master[t:t + seg_length]
        if np.isnan(s).any() or np.isnan(m).any():
            continue
        mov_years.append(years[t] + seg_lag)
        mov_corr.append(_corr_pval(s, m, method)[0])
    moving_corr = pd.Series(data=mov_corr, index=mov_years, dtype=float)

    if make_plot:
        _plot_series_corr(series_name, moving_corr, seg_corr, bin_bounds,
                          lag_table, lag_vec, seg_length, p_val)

    return {"moving_corr": moving_corr, "seg_corr": seg_corr, "overall": overall,
            "lag_table": lag_table, "bins": bins}


def _plot_series_corr(name, moving_corr, seg_corr, bin_bounds, lag_table,
                      lag_vec, seg_length, p_val):
    sig = scipy_norm_ppf(1 - p_val / 2) / np.sqrt(seg_length)   # dplR's significance line

    plt.style.use("seaborn-v0_8-darkgrid")
    plt.figure(num=1, figsize=(max(len(moving_corr) // 30, 8), 5))
    if len(moving_corr):
        plt.plot(moving_corr.index.to_numpy(), moving_corr.to_numpy(), color="k", lw=1.2,
                 label="moving correlation")
    for (lo, hi), blabel in zip(bin_bounds, seg_corr.index):
        v = seg_corr.get(blabel, np.nan)
        if not np.isnan(v):
            plt.plot([lo, hi], [v, v], color="k", lw=2.5)
    plt.axhline(sig, ls="--", color="r", label="p=%.3g" % p_val)
    plt.title("Crossdating of " + name)
    plt.xlabel("Year")
    plt.ylabel("Correlation")
    plt.legend()

    # lag stems, one panel per segment
    segs = [b for b in seg_corr.index if not lag_table[b].isna().all()]
    if segs:
        cols = 5
        rows = (len(segs) + cols - 1) // cols
        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(14, 3 * rows), squeeze=False)
        for j, blabel in enumerate(segs):
            ax = axes[j // cols][j % cols]
            ax.stem(lag_vec, lag_table[blabel].to_numpy())
            ax.axhline(sig, ls="--", color="r")
            ax.set_title(blabel, fontsize=8)
            ax.set_xlabel("Lag")
            ax.set_ylabel("r")
            ax.set_ylim([-0.5, 1])
        for j in range(len(segs), rows * cols):
            axes[j // cols][j % cols].set_axis_off()
        fig.tight_layout()
    plt.show()


def scipy_norm_ppf(q):
    import scipy.stats
    return scipy.stats.norm.ppf(q)
