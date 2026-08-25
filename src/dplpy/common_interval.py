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

# Title: common_interval.py
# Project: OpenDendro dplPy
# Description: Select a "common interval" -- a complete (no-gap) rectangle of
#              series x years in which every retained series has data for every
#              retained year. This is the interval used for chronology-level
#              statistics (rbar, EPS, SSS) on unevenly-distributed data. Three
#              selection strategies are offered, ported from and validated
#              against dplR's common.interval() (Bunn 2008): maximise the number
#              of series, the number of years, or the number of data cells.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_TYPES = ("series", "years", "both")


def _rm_short(notNA, row_idx, flag):
    """dplR's rm.short: greedily drop the shortest series to extend the run of
    fully-overlapped years, tracking the best (cells-maximising) configuration.

    Returns (n_years, n_series, keep_row_mask over row_idx, keep_col_mask over
    all columns). With flag=True it stops at the first improvement, which keeps
    the highest sample depth (used for type='series')."""
    sub = notNA[row_idx, :]
    m, ncol_full = sub.shape
    which_good = np.where(sub.any(axis=0))[0]
    ncol_orig = len(which_good)
    span_len = np.empty(ncol_orig)
    for k, c in enumerate(which_good):
        pos = np.where(sub[:, c])[0]
        span_len[k] = pos[-1] - pos[0]
    span_order = which_good[np.argsort(span_len, kind="stable")]   # shortest first

    keep_col = np.zeros(ncol_full, dtype=bool)
    keep_col[which_good] = True
    keep_col_out = keep_col.copy()
    dontkeep_row = np.ones(m, dtype=bool)
    keep_row_out = np.zeros(m, dtype=bool)
    nrow = 0
    nrow_out = 0
    ncol_out = ncol_orig
    ncol = ncol_orig
    best = 0

    for i in range(0, max(0, ncol_orig - 2) + 1):
        if i > 0:
            keep_col[span_order[i - 1]] = False
            ncol -= 1
            if ncol * m < best:              # cannot beat the best area any more
                break
        cand = np.where(dontkeep_row)[0]
        if len(cand):
            tmp = sub[np.ix_(cand, np.where(keep_col)[0])].all(axis=1)
        else:
            tmp = np.zeros(0, dtype=bool)
        dontkeep_row[cand] = ~tmp
        nrow += int(tmp.sum())
        area = ncol * nrow
        if area > best:
            best = area
            keep_col_out = keep_col.copy()
            keep_row_out = ~dontkeep_row.copy()
            ncol_out = ncol
            nrow_out = nrow
            if flag:
                break
    return nrow_out, ncol_out, keep_row_out, keep_col_out


def common_interval(rwl: pd.DataFrame, type="both", make_plot=False):
    """Select a complete common interval from a set of ring-width series.

    Extended Summary
    ----------------
    Finds the largest block of years and series in which every retained series
    has a value for every retained year (a complete, gap-free rectangle). This
    is the interval over which chronology-level statistics such as rbar, EPS and
    SSS can be computed without missing values, and is the recommended way to
    handle datasets whose series are unevenly distributed in time. Ported from,
    and validated to reproduce exactly, dplR's ``common.interval()``.

    Three strategies trade off the number of series against the number of years:

    - ``"series"`` maximises the number of series (cores), at the highest sample
      depth -- typically a short, deep interval.
    - ``"years"`` maximises the number of years, dropping short series to extend
      the span -- typically a long, shallow interval.
    - ``"both"`` (default) maximises the number of data cells (series x years),
      a balance of the two. (Following dplR, this is a specific heuristic rather
      than the global cell maximum, which ``"years"`` can occasionally exceed.)

    To use an interval of your own choosing instead, simply slice the frame
    (e.g. ``rwi.loc[1800:1980]``) before passing it on.

    Parameters
    ----------
    rwl : pandas.DataFrame
        ring-width series (raw or detrended), years as the index, series as
        columns.
    type : {"series", "years", "both"}, default "both"
        the selection strategy described above.
    make_plot : bool, default False
        if True, draw a coverage diagram (each series as a grey span, with the
        retained common interval overlaid in black). dplR draws this by default;
        here it is off by default so the function is quiet unless asked.

    Returns
    -------
    pandas.DataFrame
        the trimmed frame containing only the retained series and years (a
        complete rectangle). The full frame is returned unchanged if no trimming
        is possible/needed.

    Examples
    --------
    >>> import dplpy as dpl
    >>> rwi = dpl.detrend(dpl.readers("../tests/data/csv/co021.csv"), plot=False)
    >>> ci = dpl.common_interval(rwi, type="both")
    >>> dpl.rwi_stats(ci)                       # rbar/EPS/SNR over the interval

    References
    ----------
    .. [1] https://rdrr.io/cran/dplR/man/common.interval.html
    .. [2] Bunn (2008), Dendrochronologia, 26, 115-124.
    """
    if not isinstance(rwl, pd.DataFrame):
        # NB: 'type' is a parameter here, so the builtin is shadowed -- use
        # rwl.__class__ rather than type(rwl).
        raise TypeError("Expected dataframe input, got " + str(rwl.__class__) + " instead.")
    if type not in _TYPES:
        raise ValueError("type must be 'series', 'years' or 'both', got '" + str(type) + "'.")

    rwl = rwl.sort_index()
    notNA = ~rwl.isna().to_numpy()
    nrow_rwl, ncol_rwl = notNA.shape
    if ncol_rwl == 0:
        return rwl
    samp_depth = notNA.sum(axis=1)
    if samp_depth.max() < 2:
        return rwl.iloc[0:0, 0:0]

    best = 0
    keep_row_out = np.array([], dtype=int)     # row positions
    keep_col_out = np.zeros(ncol_rwl, dtype=bool)
    ncol_out = 0
    nrow_out = 0
    is_series = type == "series"
    is_years = type == "years"

    for i in range(int(samp_depth.max()), 1, -1):
        rows = np.where(samp_depth >= i)[0]
        if len(rows) == 0:
            continue
        row_idx = np.arange(rows[0], rows[-1] + 1)     # contiguous span
        nrow = len(row_idx)
        if i * nrow < best:                            # cannot improve
            break
        if is_series:
            nrow_out, ncol_out, krm, kcm = _rm_short(notNA, row_idx, flag=True)
            keep_row_out = row_idx[krm]
            keep_col_out = kcm
            break
        elif is_years:
            nrow, ncol, keep_row, keep_col = _rm_short(notNA, row_idx, flag=False)
        else:                                          # "both"
            keep_col = notNA[row_idx, :].all(axis=0)
            ncol = int(keep_col.sum())
        area = nrow * ncol
        if area > best:
            best = area
            nrow_out = nrow
            ncol_out = ncol
            keep_row_out = row_idx[keep_row] if is_years else row_idx
            keep_col_out = keep_col

    if make_plot:
        _plot_common(rwl, notNA, keep_row_out, keep_col_out, type,
                     ncol_out, nrow_out)

    if nrow_out < nrow_rwl or ncol_out < ncol_rwl:
        return rwl.iloc[keep_row_out, keep_col_out]
    return rwl


def apply_common_interval(rwl, spec):
    """Trim ``rwl`` per a common-interval spec, or return it unchanged.

    ``spec`` may be None (no trimming), one of 'series'/'years'/'both' (a
    dpl.common_interval strategy), or a (start_year, end_year) pair (an inclusive
    span of the caller's choosing). Shared by rwi_stats() and sss() so both treat
    the argument identically.
    """
    if spec is None:
        return rwl
    if isinstance(spec, str):
        return common_interval(rwl, type=spec)
    if isinstance(spec, (tuple, list)) and len(spec) == 2:
        return rwl.loc[int(spec[0]):int(spec[1])]
    raise ValueError("common_interval must be 'series', 'years', 'both', "
                     "or a (start_year, end_year) pair.")


def _plot_common(rwl, notNA, keep_row_out, keep_col_out, type, ncol_out, nrow_out):
    """Coverage diagram: every series as a grey span; the retained common
    interval overlaid in black (mirrors dplR's common.interval plot)."""
    yrs = rwl.index.to_numpy()
    names = list(rwl.columns)
    first = np.array([yrs[np.where(notNA[:, k])[0][0]] if notNA[:, k].any() else np.nan
                      for k in range(len(names))], dtype=float)
    last = np.array([yrs[np.where(notNA[:, k])[0][-1]] if notNA[:, k].any() else np.nan
                     for k in range(len(names))], dtype=float)
    order = np.argsort(np.where(np.isnan(first), np.inf, first), kind="stable")

    keep_rows = set(int(r) for r in keep_row_out)
    cyrs = [yrs[r] for r in sorted(keep_rows)]
    cfirst, clast = (min(cyrs), max(cyrs)) if cyrs else (None, None)

    fig, ax = plt.subplots(figsize=(max((yrs.max() - yrs.min()) / 90, 8),
                                    max(len(names) * 0.28, 4)))
    ax.set_facecolor("white")
    for rank, k in enumerate(order, start=1):
        if np.isnan(first[k]):
            continue
        ax.plot([first[k], last[k]], [rank, rank], color="grey", lw=2, zorder=1)
        if keep_col_out[k] and cfirst is not None:
            ax.plot([cfirst, clast], [rank, rank], color="black", lw=2.5, zorder=2)
    if cfirst is not None:
        for xv in (cfirst, clast + 1):
            ax.axvline(xv, ls="--", color="black", lw=1)
    ax.set_yticks(range(1, len(names) + 1))
    ax.set_yticklabels([names[k] for k in order], fontsize=8)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_title("Common interval (type='%s'): %d series x %d years = %d"
                 % (type, ncol_out, nrow_out, ncol_out * nrow_out), fontsize=12)
    for spine in ax.spines.values():
        spine.set_color("black")
    fig.tight_layout()
    plt.show()
