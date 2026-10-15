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
#              (rwi_stats_running), after dplR's rwi.stats() and
#              rwi.stats.running() (Wigley et al. 1984). Whole-record statistics
#              reproduce dplR. In running windows with tree IDs, the effective
#              cores-per-tree count (c_eff) is ARSTAN-style -- cores present in
#              the window, less any core whose within-tree correlations all
#              fail min_corr_overlap -- which differs from dplR's
#              back-calculation from the number of valid within-tree
#              correlations (0.5 + sqrt(0.25 + 2 n_wt)); the two agree when
#              every within-tree pair is valid and trees have at most two cores.
#              Window placement starts at the first year and leaves any trailing
#              remainder uncovered (dplR picks the start maximising coverage),
#              and period="common" restricts each window to years present in all
#              of the window's series (dplR's prewhiten/common handling differs).
#
#              NOTE ON TERMINOLOGY: the rbar reported here is the mean pairwise
#              correlation between series (optionally split into within-tree and
#              between-tree components when tree IDs are supplied). This is a
#              different statistic from the "interseries correlation" computed
#              by dpl.interseries_corr(), which correlates each series against a
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
import numpy as np
import pandas as pd
from ._validate import _require_dataframe, _normalize_corr, _normalize_choice, _warn

from .autoreg import _prewhiten
from .rbar import pairwise_corr_matrix
from .common_interval import apply_common_interval


def rwi_stats(rwi_data: pd.DataFrame, ids=None, period="max", corr="spearman",
              prewhiten=False, min_corr_overlap=None, zero_is_missing=True,
              round_decimals=3, common_interval=None, ar_max=10, ar_method="yw",
              first_aic_min=True):
    """Whole-record chronology signal statistics (rbar, EPS, SNR).

    Computes the mean interseries correlation (rbar), expressed population signal
    (EPS), and signal-to-noise ratio (SNR) over the entire record as a single-row
    summary. The non-running counterpart of rwi_stats_running(), which it calls with
    running_window=False; see that function for full parameter and statistic
    descriptions.

    Parameters
    ----------
    rwi_data : pandas dataframe
        detrended ring-width indices (e.g. from dpl.detrend()).
    ids : dict, pandas dataframe, or None, default None
        tree/core structure; see rwi_stats_running().
    period : str, default "max"
        "max" (pairwise-complete overlaps) or "common" (complete rows only);
        case-insensitive.
    corr : str, default "spearman"
        correlation type, "spearman" or "pearson" (case-insensitive).
    prewhiten : boolean, default False
        prewhiten each series with an AR model before computing correlations
        (see rwi_stats_running()).
    min_corr_overlap : int or None, default None
        minimum number of overlapping years for a series pair to contribute.
    zero_is_missing : boolean, default True
        treat exact zeros as missing values.
    round_decimals : int or None, default 3
        decimal places to round the statistic columns to.
    common_interval : str, (int, int), or None, default None
        restrict to a common interval first (see rwi_stats_running()).
    ar_max, ar_method, first_aic_min :
        prewhitening controls (see rwi_stats_running()).

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
                             round_decimals=round_decimals,
                             common_interval=common_interval, ar_max=ar_max,
                             ar_method=ar_method, first_aic_min=first_aic_min)


def rwi_stats_running(rwi_data: pd.DataFrame, ids=None, period="max",
                      corr="spearman", prewhiten=False, running_window=True,
                      window_length=50, window_overlap=None,
                      min_corr_overlap=None, zero_is_missing=True,
                      round_decimals=3, common_interval=None, ar_max=10,
                      ar_method="yw", first_aic_min=True):
    """Running (moving-window) chronology signal statistics (rbar, EPS, SNR).

    For each moving window along the chronology, computes the mean interseries
    correlation (rbar) and, from it, the expressed population signal (EPS) and
    signal-to-noise ratio (SNR) of Wigley et al. (1984), after dplR's
    rwi.stats.running(); rwi_stats() is the whole-record special case
    (running_window=False).

    When tree/core IDs are supplied (see ``ids``), correlations are split into
    within-tree (rbar_wt) and between-tree (rbar_bt) components, and the effective
    signal rbar_eff accounts for averaging multiple cores per tree via an
    effective cores-per-tree factor c_eff. c_eff is ARSTAN-style running
    effective replication: the cores of each tree present in the window, not
    counting a core whose within-tree correlations all fail ``min_corr_overlap``.
    This differs from dplR's back-calculation of the core count from the number
    of valid within-tree correlations (``0.5 + sqrt(0.25 + 2 n_wt)``); the two
    agree for the whole record when every within-tree pair is valid and trees
    have at most two cores, and can differ inside running windows. Without IDs,
    every series is its own tree and rbar_eff == rbar_bt == rbar_tot.

    Windows start at the first year and advance by ``window_length -
    window_overlap``; a trailing remainder shorter than a window is not covered
    (dplR instead chooses the start that maximises coverage).

    EPS = n * rbar_eff / ((n - 1) * rbar_eff + 1) and
    SNR = n * rbar_eff / (1 - rbar_eff), where n is the number of trees contributing
    to the between-tree correlations.

    NOTE: no acceptance threshold (e.g. the frequently cited EPS > 0.85) is applied
    or implied here; that threshold was an illustrative example for SSS (see
    dpl.sss()), not a general criterion for EPS.

    Parameters
    ----------
    rwi_data : pandas dataframe
        detrended ring-width indices (e.g. from dpl.detrend()), with years as the
        index and series as columns.
    ids : dict, pandas dataframe, or None, default None
        tree/core structure used to split within- vs between-tree correlations.
        Either a dict mapping {series_name: tree_id}, a dataframe with a 'tree'
        column indexed by series name (such as produced by dpl.read_ids()), or None
        to treat every series as its own tree (one core per tree).
    period : str, default "max"
        "max" uses each pair's own overlapping years (pairwise-complete); "common"
        restricts each window to years present in every series that has data in
        the window (case-insensitive).
    corr : str, default "spearman"
        correlation type, "spearman" or "pearson" (case-insensitive).
    prewhiten : boolean, default False
        prewhiten each series with an AR model before computing correlations,
        using the package-standard `autoreg._prewhiten()` (Yule-Walker, order by
        the first local AIC minimum up to ``ar_max``, ARSTAN backcast initials so
        no years are lost). NOTE: R's ar() / dplR take the global AIC minimum up
        to floor(10*log10(n)); pass ``first_aic_min=False`` for that rule.
    running_window : boolean, default True
        if False, a single window spanning the whole record is used (this is what
        rwi_stats() invokes).
    window_length : int or None, default 50
        length of each moving window in years (None: min(50, n_years)).
    window_overlap : int or None, default None
        number of years successive windows overlap; defaults to window_length // 2.
    min_corr_overlap : int or None, default None
        minimum number of overlapping years required for a series pair to contribute
        a correlation; defaults to min(30, window_length).
    zero_is_missing : boolean, default True
        treat exact zeros as missing values (a zero RWI is treated as absent).
    round_decimals : int or None, default 3
        decimal places to round the real-valued statistic columns to; None leaves
        them unrounded.
    common_interval : str, (int, int), or None, default None
        restrict the data to a common interval before computing statistics. 'series',
        'years' or 'both' selects a complete overlap-only rectangle via
        dpl.common_interval() (maximising cores, years, or cells); a
        (start_year, end_year) pair restricts to that inclusive span. None (default)
        uses the full record. Recommended for unevenly-distributed data, where
        period='common' on the full record can be empty.
    ar_max : int, default 10
        maximum AR order for ``prewhiten=True``.
    ar_method : {"yw", "burg", "ols"}, default "yw"
        AR estimator for ``prewhiten=True`` (case-insensitive).
    first_aic_min : boolean, default True
        AR order rule for ``prewhiten=True``: first local AIC minimum (True) or
        global minimum (False, R's ar()).

    Returns
    -------
    result : pandas dataframe with one row per window (or a single row if
        running_window=False). Columns: start_year, mid_year, end_year (running
        only), n_cores, n_trees, n, n_tot, n_wt, n_bt, rbar_tot, rbar_wt, rbar_bt,
        c_eff, rbar_eff, eps, snr.

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
    .. [4] https://opendendro.github.io/dplPy/reference/statistics/#rwi_stats_running
    """
    _require_dataframe(rwi_data)

    # Optionally restrict to a common interval before computing statistics: a
    # 'series'/'years'/'both' strategy or a (start_year, end_year) pair. The
    # result is a complete, overlap-only block.
    rwi_data = apply_common_interval(rwi_data, common_interval)

    corr_method = _normalize_corr(corr, allowed=("spearman", "pearson"))
    period = _normalize_choice(period, ("max", "common"), "period")
    ar_method = _normalize_choice(ar_method, ("yw", "burg", "ols"), "ar_method")

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
        _warn("There are zeros in the data. Consider zero_is_missing=True.\n")

    # Guard analogous to dplR's non-positive grand mean check. Unlike dplR we do
    # not divide by the mean (correlations are invariant to a positive rescale),
    # so a small/zero mean cannot break the computation -- but a non-positive
    # grand mean signals the input is probably difference-detrended data, for
    # which rbar is not meaningful. dplR stops here; we warn and proceed.
    grand_mean = np.nanmean(data.to_numpy(dtype=float))
    if not np.isnan(grand_mean) and grand_mean <= 0:
        _warn("'rwi_data' has a non-positive grand mean; rbar/EPS/SNR may not "
              "be meaningful. This can happen with difference-detrended data.\n")

    if prewhiten:
        # AR prewhitening DOES change correlations, so (unlike the mean-division
        # dplR also performs) it must actually be applied -- with the one
        # package-standard prewhitening (audit D1).
        data = pd.DataFrame(
            {c: _prewhiten(data[c].to_numpy(dtype=float), estimator=ar_method,
                           ar_max=ar_max, first_aic_min=first_aic_min)
             for c in data.columns}, index=data.index)

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
        _warn("Correlations are all NA -- no series pairs met the overlap "
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

    # Pairwise-complete correlation with a minimum-overlap requirement (the
    # shared rbar.pairwise_corr_matrix; replaces dplR's cor.with.limit /
    # cor.with.limit.upper per-pair loops).
    corr_arr = pairwise_corr_matrix(window, corr_method, min_overlap=min_corr_overlap,
                                    strict=False)
    order = list(window.columns)
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

    # effective cores-per-tree correction: ARSTAN-style running effective
    # replication -- the cores of each good tree present in the window, not
    # counting a core whose within-tree correlations all fail min_corr_overlap
    # (such a core adds nothing to the within-tree signal; dplR drops it too).
    # Differs from dplR's back-calculation 0.5 + sqrt(0.25 + 2 n_wt) for trees
    # with three or more cores (see the module header).
    if n_wt == 0:
        c_eff = 1.0 if n_bt > 0 else 0.0
        rbar_eff = rbar_bt
    else:
        idx_i, idx_j = iu[0][wt], iu[1][wt]
        has_valid_wt = set(order[k] for k in idx_i) | set(order[k] for k in idx_j)
        cores_per_tree = Counter(tree_of[c] for c in present_series if c in has_valid_wt)
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
