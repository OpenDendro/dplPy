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

# Title: rwi_stats.py
# Project: OpenDendro dplPy
# Description: Chronology signal statistics -- mean interseries correlation
#              (rbar), the expressed population signal (EPS), and the
#              signal-to-noise ratio (SNR), computed either over the whole
#              record (rwi_stats) or as a running series over moving windows
#              (rwi_stats_running). This is a port of dplR's rwi.stats() and
#              rwi.stats.running() (Wigley et al. 1984), following that
#              implementation's outcomes while taking advantage of pandas'
#              native pairwise-correlation handling.
#
#              NOTE ON TERMINOLOGY: the rbar reported here is the mean pairwise
#              correlation between series (optionally split into within-tree and
#              between-tree components when tree IDs are supplied). This is a
#              different statistic from the "interseries correlation" computed
#              by dpl.interseries_cor(), which correlates each series against a
#              composite chronology of the others.
#
# example usage from Python Console:
# >>> import dplpy as dpl
# >>> data = dpl.readers("../tests/data/csv/file.csv")
# >>> rwi = dpl.detrend(data, fit="spline", plot=False)
# >>> dpl.rwi_stats(rwi)                        # whole-record statistics
# >>> dpl.rwi_stats_running(rwi)                # running (moving-window) series
# >>> ids = dpl.read_ids(data)                  # optional tree/core structure
# >>> dpl.rwi_stats_running(rwi, ids=ids)

from collections import Counter
import warnings

import numpy as np
import pandas as pd

from autoreg import ar_func


def rwi_stats(rwi_data: pd.DataFrame, ids=None, period="max", corr="Spearman",
              prewhiten=False, min_corr_overlap=None, zero_is_missing=True,
              round_decimals=3):
    """Whole-record chronology signal statistics (rbar, EPS, SNR).

    Extended Summary
    ----------------
    Computes the mean interseries correlation (rbar), expressed population
    signal (EPS), and signal-to-noise ratio (SNR) over the entire record, as
    a single-row summary. This is the non-running counterpart of
    rwi_stats_running(): it simply calls that function with
    running_window=False, exactly as dplR's rwi.stats() delegates to
    rwi.stats.running().

    See rwi_stats_running() for a full description of the parameters and the
    statistics computed.

    Parameters
    ----------
    rwi_data : pandas dataframe
        detrended ring-width indices (e.g. from dpl.detrend()).
    ids : dict, pandas dataframe, or None, default None
        tree/core structure; see rwi_stats_running().
    period : str, default "max"
        "max" (pairwise-complete overlaps) or "common" (complete rows only).
    corr : str, default "Spearman"
        correlation type, "Spearman" or "Pearson".
    prewhiten : boolean, default False
        prewhiten each series with an AR model before computing correlations.
    min_corr_overlap : int or None, default None
        minimum number of overlapping years for a series pair to contribute.
    zero_is_missing : boolean, default True
        treat exact zeros as missing values.
    round_decimals : int or None, default 3
        decimal places to round the statistic columns to.

    Returns
    -------
    result : a one-row pandas dataframe of chronology signal statistics.

    Examples
    --------
    >>> import dplpy as dpl
    >>> rwi = dpl.detrend(dpl.readers("../tests/data/csv/file.csv"), plot=False)
    >>> dpl.rwi_stats(rwi)

    References
    ----------
    .. [1] https://rdrr.io/cran/dplR/man/rwi.stats.running.html
    .. [2] Wigley, Briffa & Jones (1984), J. Climate Appl. Meteor., 23, 201-213.

    """
    return rwi_stats_running(rwi_data, ids=ids, period=period, corr=corr,
                             prewhiten=prewhiten, running_window=False,
                             min_corr_overlap=min_corr_overlap,
                             zero_is_missing=zero_is_missing,
                             round_decimals=round_decimals)


def rwi_stats_running(rwi_data: pd.DataFrame, ids=None, period="max",
                      corr="Spearman", prewhiten=False, running_window=True,
                      window_length=None, window_overlap=None,
                      min_corr_overlap=None, zero_is_missing=True,
                      round_decimals=3):
    """Running (moving-window) chronology signal statistics (rbar, EPS, SNR).

    Extended Summary
    ----------------
    Computes, for each moving window along the chronology, the mean interseries
    correlation (rbar) and, from it, the expressed population signal (EPS) and
    signal-to-noise ratio (SNR) of Wigley et al. (1984). This is a port of
    dplR's rwi.stats.running(); rwi_stats() is the whole-record special case
    (running_window=False).

    When tree/core IDs are supplied (see `ids`), correlations are split into
    within-tree (rbar_wt) and between-tree (rbar_bt) components, and the
    effective signal rbar_eff accounts for the averaging of multiple cores per
    tree (via an effective cores-per-tree factor c_eff). Without IDs, every
    series is treated as its own tree and rbar_eff == rbar_bt == rbar_tot.

    EPS = n * rbar_eff / ((n - 1) * rbar_eff + 1), where n is the number of
    trees contributing to the between-tree correlations, measures how well the
    finite sample represents a (hypothetically infinite) population chronology.
    SNR = n * rbar_eff / (1 - rbar_eff) is the corresponding signal-to-noise
    ratio (Cook & Pederson 2011).

    NOTE: no acceptance threshold (e.g. the frequently cited EPS > 0.85) is
    applied or implied here. As Wigley et al. (1984) state and Buras (2017)
    reemphasizes, that threshold was an illustrative example for SSS (see
    dpl.sss()), not a general criterion for EPS; whether a value is adequate is
    a judgement for the analyst, not the software.

    Implementation notes
    --------------------
    The within-window pairwise correlations are computed with a single
    pandas .corr(min_periods=...) call, which natively performs
    pairwise-complete correlation and drops pairs with fewer than
    min_corr_overlap overlapping years -- replacing dplR's hand-written
    per-pair correlation loops. Per-series mean normalization (which dplR
    applies internally) is deliberately skipped because it cannot change any
    correlation; AR prewhitening, which does change correlations, is applied
    when requested. The effective cores-per-tree factor is computed by counting
    cores-with-data per tree directly, rather than back-calculating it from the
    number of valid within-tree correlation pairs as dplR does; the two agree
    whenever within-tree overlap is adequate.

    Parameters
    ----------
    rwi_data : pandas dataframe
        detrended ring-width indices (e.g. from dpl.detrend()), with years as
        the index and series as columns.
    ids : dict, pandas dataframe, or None, default None
        tree/core structure used to split within- vs between-tree correlations.
        Either a dict mapping {series_name: tree_id}, a dataframe with a 'tree'
        column indexed by series name (such as produced by dpl.read_ids()), or
        None to treat every series as its own tree (one core per tree).
    period : str, default "max"
        "max" uses each pair's own overlapping years (pairwise-complete);
        "common" restricts each window to years present in every series.
    corr : str, default "Spearman"
        correlation type, "Spearman" or "Pearson".
    prewhiten : boolean, default False
        prewhiten each series with an AR model before computing correlations.
    running_window : boolean, default True
        if False, a single window spanning the whole record is used (this is
        what rwi_stats() invokes).
    window_length : int or None, default None
        length of each moving window in years; defaults to min(50, n_years).
    window_overlap : int or None, default None
        number of years successive windows overlap; defaults to
        window_length // 2.
    min_corr_overlap : int or None, default None
        minimum number of overlapping years required for a series pair to
        contribute a correlation; defaults to min(30, window_length).
    zero_is_missing : boolean, default True
        treat exact zeros as missing values (a zero RWI is treated as absent).
    round_decimals : int or None, default 3
        decimal places to round the real-valued statistic columns to; None
        leaves them unrounded.

    Returns
    -------
    result : pandas dataframe with one row per window (or a single row if
        running_window=False). Columns: start_year, mid_year, end_year (running
        only), n_cores, n_trees, n, n_tot, n_wt, n_bt, rbar_tot, rbar_wt,
        rbar_bt, c_eff, rbar_eff, eps, snr.

    Examples
    --------
    >>> import dplpy as dpl
    >>> rwi = dpl.detrend(dpl.readers("../tests/data/csv/file.csv"), plot=False)
    >>> dpl.rwi_stats_running(rwi, window_length=60, window_overlap=30)

    References
    ----------
    .. [1] https://rdrr.io/cran/dplR/man/rwi.stats.running.html
    .. [2] Wigley, Briffa & Jones (1984), J. Climate Appl. Meteor., 23, 201-213.
    .. [3] Buras (2017), Dendrochronologia, 44, 130-132.

    """
    if not isinstance(rwi_data, pd.DataFrame):
        raise TypeError("Expected dataframe input, got " + str(type(rwi_data)) + " instead.")

    corr_method = corr.lower()
    if corr_method not in ("spearman", "pearson"):
        raise ValueError("corr must be either 'Spearman' or 'Pearson', got '" + str(corr) + "'.")
    if period not in ("max", "common"):
        raise ValueError("period must be either 'max' or 'common', got '" + str(period) + "'.")

    n_years = rwi_data.shape[0]

    if window_length is None:
        window_length = min(50, n_years)
    if window_overlap is None:
        window_overlap = window_length // 2
    if min_corr_overlap is None:
        min_corr_overlap = min(30, window_length)

    data = rwi_data.copy()

    if zero_is_missing:
        # dplR recodes exact zeros to NA (a zero RWI is treated as missing).
        data = data.where(data != 0)
    elif bool((data == 0).to_numpy().any()):
        warnings.warn("There are zeros in the data. Consider zero_is_missing=True.\n")

    # Guard analogous to dplR's non-positive grand mean check. Unlike dplR we do
    # not divide by the mean (correlations are invariant to a positive rescale),
    # so a small/zero mean cannot break the computation -- but a non-positive
    # grand mean signals the input is probably difference-detrended data, for
    # which rbar is not meaningful. dplR stops here; we warn and proceed.
    grand_mean = np.nanmean(data.to_numpy(dtype=float))
    if not np.isnan(grand_mean) and grand_mean <= 0:
        warnings.warn("'rwi_data' has a non-positive grand mean; rbar/EPS/SNR may not "
                      "be meaningful. This can happen with difference-detrended data.\n")

    if prewhiten:
        # AR prewhitening DOES change correlations, so (unlike the mean-division
        # dplR also performs) it must actually be applied. Reuses dplPy's AR
        # machinery.
        data = ar_func(data)

    columns = list(data.columns)
    tree_of = _resolve_tree_mapping(ids, columns)

    if len(set(tree_of.values())) < 2:
        raise ValueError("at least 2 trees are needed")

    years = data.index.to_numpy()

    if running_window:
        if window_length < 3:
            raise ValueError("minimum window_length is 3")
        window_advance = window_length - window_overlap
        if window_advance < 1:
            raise ValueError("window_overlap is too large; it must be less than window_length")
        if window_length < min_corr_overlap:
            raise ValueError("window_length is smaller than min_corr_overlap")
        if window_length > n_years:
            raise ValueError("window_length is larger than the number of years in rwi_data")
        starts = list(range(0, n_years - window_length + 1, window_advance))
        win_len = window_length
    else:
        starts = [0]
        win_len = n_years

    rows = [
        _window_stats(data.iloc[s:s + win_len], tree_of, corr_method,
                      min_corr_overlap, period, running_window, years, s, win_len)
        for s in starts
    ]

    result = pd.DataFrame(rows)

    # dplR-style heads-up: if nothing was computable (e.g. period="common" on a
    # record with no fully-overlapping interval), rbar is NA everywhere.
    if result["rbar_tot"].isna().all():
        warnings.warn("Correlations are all NA -- no series pairs met the overlap "
                      "requirement. With period='common', there may be no interval "
                      "common to all series.\n")

    if round_decimals is not None and round_decimals >= 0:
        stat_cols = ["rbar_tot", "rbar_wt", "rbar_bt", "c_eff", "rbar_eff", "eps", "snr"]
        result[stat_cols] = result[stat_cols].round(round_decimals)

    return result


def _resolve_tree_mapping(ids, columns):
    """Return a {series_name: tree_id} dict covering every column.

    ids may be None (one core per tree), a dict, or a dataframe with a 'tree'
    column indexed by series name.
    """
    if ids is None:
        return {c: c for c in columns}

    if isinstance(ids, pd.DataFrame):
        if "tree" not in ids.columns:
            raise ValueError("ids DataFrame must have a 'tree' column.")
        mapping = ids["tree"].to_dict()
    elif isinstance(ids, dict):
        mapping = ids
    else:
        raise TypeError("ids must be None, a dict of {series: tree}, or a DataFrame "
                        "with a 'tree' column, got " + str(type(ids)) + ".")

    missing = [c for c in columns if c not in mapping]
    if missing:
        raise ValueError("ids is missing tree assignments for series: " + ", ".join(map(str, missing)))

    return {c: mapping[c] for c in columns}


def _window_stats(window, tree_of, corr_method, min_corr_overlap, period,
                  running_window, years, s_idx, win_len):
    """Compute the signal statistics for a single window, as a dict row."""
    if period == "common":
        # restrict to years present in every (present) series in the window
        window = window.dropna(axis=1, how="all").dropna(axis=0, how="any")

    series = list(window.columns)

    # sample depth: cores (and their trees) with any data in the window
    if len(series) > 0:
        present = window.notna().to_numpy().any(axis=0)
    else:
        present = np.array([], dtype=bool)
    present_series = [series[k] for k in range(len(series)) if present[k]]
    n_cores = len(present_series)
    n_trees = len(set(tree_of[c] for c in present_series))

    out = {}
    if running_window:
        start_year = int(years[s_idx])
        end_year = int(years[s_idx + win_len - 1])
        out["start_year"] = start_year
        out["mid_year"] = int(np.floor((start_year + end_year) / 2))
        out["end_year"] = end_year

    # degenerate-window defaults (overwritten below when computable)
    out.update({
        "n_cores": n_cores, "n_trees": n_trees, "n": 0,
        "n_tot": 0, "n_wt": 0, "n_bt": 0,
        "rbar_tot": np.nan, "rbar_wt": np.nan, "rbar_bt": np.nan,
        "c_eff": np.nan, "rbar_eff": np.nan, "eps": np.nan, "snr": np.nan,
    })

    if n_cores < 2:
        return out

    # Pairwise-complete correlation with a minimum-overlap requirement, in one
    # native pandas call (replaces dplR's cor.with.limit / cor.with.limit.upper
    # per-pair loops). .to_numpy(copy=True) avoids the read-only-view issue when
    # writing the NaN diagonal under numpy 2 / pandas copy-on-write.
    corr_mat = window.corr(method=corr_method, min_periods=min_corr_overlap)
    corr_arr = corr_mat.to_numpy(copy=True)
    order = list(corr_mat.columns)
    trees = np.array([tree_of[c] for c in order], dtype=object)

    n = corr_arr.shape[0]
    iu = np.triu_indices(n, k=1)
    pair_r = corr_arr[iu]
    valid = ~np.isnan(pair_r)
    if not valid.any():
        return out

    tree_i = trees[iu[0]]
    tree_j = trees[iu[1]]
    same_tree = tree_i == tree_j
    bt = valid & ~same_tree
    wt = valid & same_tree

    n_bt = int(bt.sum())
    n_wt = int(wt.sum())
    n_tot = n_bt + n_wt
    rsum_bt = float(pair_r[bt].sum())
    rsum_wt = float(pair_r[wt].sum())

    rbar_tot = (rsum_bt + rsum_wt) / n_tot if n_tot > 0 else np.nan
    rbar_bt = rsum_bt / n_bt if n_bt > 0 else np.nan
    rbar_wt = rsum_wt / n_wt if n_wt > 0 else np.nan

    # trees participating in at least one valid between-tree correlation
    good_trees = set(tree_i[bt]) | set(tree_j[bt])
    n_good = len(good_trees)

    # effective cores-per-tree correction (direct core count; see module notes)
    if n_wt == 0:
        c_eff = 1.0 if n_bt > 0 else 0.0
        rbar_eff = rbar_bt
    else:
        cores_per_tree = Counter(tree_of[c] for c in present_series)
        nc = np.array([max(cores_per_tree[t], 1) for t in good_trees], dtype=float)
        rproc = float(np.mean(1.0 / nc))
        c_eff = 1.0 / rproc
        rbar_eff = rbar_bt / (rbar_wt + (1.0 - rbar_wt) * rproc)

    # EPS and SNR use n = number of good (between-tree-correlated) trees
    if n_good >= 1 and rbar_eff is not None and not np.isnan(rbar_eff):
        eps_denom = (n_good - 1) * rbar_eff + 1
        eps = n_good * rbar_eff / eps_denom if eps_denom != 0 else np.nan
        snr = n_good * rbar_eff / (1 - rbar_eff) if (1 - rbar_eff) != 0 else np.nan
    else:
        eps = np.nan
        snr = np.nan

    out.update({
        "n": n_good, "n_tot": n_tot, "n_wt": n_wt, "n_bt": n_bt,
        "rbar_tot": rbar_tot, "rbar_wt": rbar_wt, "rbar_bt": rbar_bt,
        "c_eff": c_eff, "rbar_eff": rbar_eff, "eps": eps, "snr": snr,
    })
    return out
