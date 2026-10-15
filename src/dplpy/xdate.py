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

# Date: 5/12/2023 (rewritten 2026 for efficiency and dplR fidelity)
# Author: Ifeoluwa Ale (original), OpenDendro
# Title: xdate.py
# Project: OpenDendro dplPy
# Description: Crossdating for dplPy datasets, mirroring dplR's corr.rwl.seg():
#   normalize each series (divide by its mean), optionally AR-prewhiten (the
#   package-standard autoreg._prewhiten), build a leave-one-out biweight master,
#   and correlate each series against it over overlapping segments -- reporting
#   per-segment correlation, its one-tailed p-value, an overall correlation, and
#   flags for (A) non-significant segments and (B) segments that correlate better
#   at a lag (the COFECHA-style lag table).
#
#   Lag convention, everywhere in dplPy: the dated segment stays put and the
#   MASTER slides past it, so a NEGATIVE best lag means a MISSING ring in the
#   series (the series is one ring short, as in COFECHA and dplR's documentation).
#
# example usage:
# >>> import dplpy as dpl
# >>> data = dpl.readers("../tests/data/csv/file.csv")
# >>> rwi  = dpl.detrend(data, fit="spline", plot=False)
# >>> res  = dpl.xdate(rwi)                 # dict of results; prints flags
# >>> res["seg_corr"]                        # segment correlations (series x bins)

from .detrend import detrend
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
from ._validate import (_require_dataframe, _normalize_corr, _plot_kwarg, _warn,
                        _alias_kwarg, _reject_kwargs, _normalize_choice,
                        _require_year_index, _carry_attrs)
from .autoreg import _prewhiten
from .tbrm import tbrm_rows
from .sensitivity import sens1
import numpy as np
import scipy
import scipy.stats
import warnings
import re

# COFECHA fits no AR model to a series shorter than this (MEMPR's N >= 8 rule).
_COFECHA_AR_NMIN = 8

# the scipy.stats warnings a degenerate (constant / tiny) correlation input raises;
# the helpers below return NaN for those cases, so the warnings are noise
_SCIPY_CORR_WARNINGS = tuple({RuntimeWarning,
                              getattr(scipy.stats, "ConstantInputWarning", RuntimeWarning),
                              getattr(scipy.stats, "NearConstantInputWarning", RuntimeWarning)})


def _prewhiten_for_xdate(vals, estimator="yw", ar_max=10, first_aic_min=True,
                         backcast=False, return_params=False):
    """`autoreg._prewhiten()` with the crossdating fallback: when the AIC never
    turns up within ``ar_max`` the first-local-minimum rule has no answer and
    ``_prewhiten`` raises (as dplR's chron.ars does); a per-series crossdating
    run cannot stop there, so the order is then the AIC minimum within the
    ceiling (ARSTAN's behaviour when the AIC keeps falling: the highest order)."""
    try:
        return _prewhiten(vals, estimator=estimator, ar_max=ar_max,
                          first_aic_min=first_aic_min, backcast=backcast,
                          return_params=return_params)
    except ValueError:
        if not first_aic_min:
            raise
        return _prewhiten(vals, estimator=estimator, ar_max=ar_max,
                          first_aic_min=False, backcast=backcast,
                          return_params=return_params)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

# The leave-one-out crossdating master is the per-row biweight robust mean
# (dplR's apply(subset, 1, tbrm, C=9)); the single implementation lives in
# tbrm.tbrm_rows. Kept under this name so series_corr / rcs imports are stable.
_row_biweight = tbrm_rows


def _row_mean(mat):
    """Per-row arithmetic mean, NaN-aware (the non-biweight master)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(mat, axis=1)


def dense_year_grid(df):
    """Reindex a year-indexed frame onto the full consecutive-year span of its
    INDEX. Returns (reindexed_frame, years, first_year, last_year) -- shared by
    xdate() and series_corr(), which both need a gap-free year grid for the
    segment logic.

    The span comes from the index range, NOT first_valid_index()/last_valid_index().
    ``normalize_for_crossdating`` seeds the frame with the whole data year index,
    so the range is the data's own span; AR prewhitening leaves the leading
    ``order`` years NaN, and trimming to the first *valid* year would shift the
    bin-floor later (e.g. a series starting 1698 whose prewhitened first value is
    1706 would floor bins to 1800 instead of 1700). dplR keeps the full year range
    and pads NaN, so the segment bins are not shifted by AR-order loss -- this
    matches that behaviour."""
    first_year = int(df.index.min())
    last_year = int(df.index.max())
    years = np.arange(first_year, last_year + 1)
    return df.reindex(years), years, first_year, last_year


def _fast_corr(a, b, method, b_ranked=False):
    """Correlation only (no p-value), over pairwise-complete elements -- the one
    correlation helper for the many lag-table / sliding correlations where the
    p-value is not needed (xdate, series_corr, check, xdate_floater, xdate_undated).
    For spearman `b` may be passed pre-ranked (b_ranked=True) to avoid re-ranking
    the master. Returns NaN with fewer than 3 complete pairs or a constant input."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = ~np.isnan(a) & ~np.isnan(b)
    if ok.sum() < 3:
        return np.nan
    a2, b2 = a[ok], b[ok]
    if method == "kendall":
        with warnings.catch_warnings():
            for cat in _SCIPY_CORR_WARNINGS:
                warnings.simplefilter("ignore", cat)
            return float(scipy.stats.kendalltau(a2, b2)[0])
    if method == "spearman":
        a2 = scipy.stats.rankdata(a2)
        b2 = scipy.stats.rankdata(b2) if not b_ranked else b2
    da = a2 - a2.mean()
    db = b2 - b2.mean()
    denom = np.sqrt(np.dot(da, da) * np.dot(db, db))
    if denom == 0:
        return np.nan
    return float(np.dot(da, db) / denom)


def _corr_pval(a, b, method):
    """One-tailed (alternative='greater') correlation and p-value over the
    pairwise-complete elements of a and b, matching dplR's cor.test(...)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = ~np.isnan(a) & ~np.isnan(b)
    if ok.sum() < 3:
        return np.nan, np.nan
    a, b = a[ok], b[ok]
    if np.all(a == a[0]) or np.all(b == b[0]):
        return np.nan, np.nan
    with warnings.catch_warnings():
        for cat in _SCIPY_CORR_WARNINGS:
            warnings.simplefilter("ignore", cat)
        if method == "pearson":
            r = scipy.stats.pearsonr(a, b, alternative="greater")
        elif method == "kendall":
            r = scipy.stats.kendalltau(a, b, alternative="greater")
        else:
            r = scipy.stats.spearmanr(a, b, alternative="greater")
    stat = r.statistic if hasattr(r, "statistic") else r[0]
    pval = r.pvalue if hasattr(r, "pvalue") else r[1]
    return float(stat), float(pval)


def _thin(vals, target=10):
    """Thin a list of tick positions to at most ``target`` evenly spaced ones
    (shared by the xdate and series_corr segment plots)."""
    vals = list(vals)
    if len(vals) <= target:
        return vals
    return vals[:: int(np.ceil(len(vals) / target))]


# ---------------------------------------------------------------------------
# Bins
# ---------------------------------------------------------------------------

def _bin_bounds(label):
    """(start, end) integers from a bin label like '700-749' or '-390--341'
    (handles negative/BC years, where a naive split on '-' fails)."""
    m = re.match(r"(-?\d+)-(-?\d+)$", label)
    return int(m.group(1)), int(m.group(2))


def get_bins(first_year, last_year, bin_floor, window_length):
    """Overlapping segment bins matching dplR's corr.rwl.seg / ccf.series.rwl.

    The first bin is floored to a multiple of ``bin_floor`` (0/None: the first
    year itself), segments are ``window_length`` years long and overlap by half,
    and the last bin ends no later than ``last_year``. Returns ``(bins, bin_data)``
    -- the list of "start-end" labels and a dict of empty lists keyed by label.
    Warns when no bin fits (audit B12), so an empty crossdating result is never
    silent."""
    window_length = int(window_length)
    seg_lag = window_length // 2
    if bin_floor is None or bin_floor == 0:
        min_bin = int(first_year)
    else:
        min_bin = int(np.ceil(first_year / bin_floor)) * int(bin_floor)
    max_bin = int(last_year) - window_length + 1
    bins = []
    bin_data = {}
    if min_bin > max_bin:
        _warn("get_bins: no complete %d-year segment fits between %d and %d with "
              "bin_floor=%s; nothing can be tested. Use a shorter window_length "
              "or a smaller bin_floor." % (window_length, first_year, last_year,
                                          bin_floor), stacklevel=4)
        return bins, bin_data
    i = min_bin
    while i <= max_bin:
        period = str(i) + "-" + str(i + window_length - 1)
        bins.append(period)
        bin_data[period] = []
        i += seg_lag
    return bins, bin_data


# ---------------------------------------------------------------------------
# Normalisation / preparation (shared)
# ---------------------------------------------------------------------------

def normalize_for_crossdating(data: pd.DataFrame, prewhiten=True, ar_max=10,
                              first_aic_min=True, backcast=False, estimator="yw",
                              spline_period=None, zscore=False, **kwargs) -> pd.DataFrame:
    """Divide each series by its own mean (dplR's normalize1 with n=NULL, i.e.
    dplPy's 'horizontal' detrend) and, optionally, prewhiten it keeping the series
    length. The one series transform behind xdate(), series_corr(),
    interseries_corr(), xdate_floater() and xdate_undated(). Returns a
    year-indexed dataframe on the data's own year index (values stay at their
    years; interior gaps are never compressed).

    Prewhitening is the package-standard `autoreg._prewhiten()`: ``estimator``
    ("yw" Yule-Walker, "burg", "ols"), order chosen by AIC up to ``ar_max`` (an
    int; ``None`` uses dplR's ``floor(10*log10(n))`` ceiling), at the first local
    AIC minimum when ``first_aic_min`` (the ARSTAN/COFECHA rule) else the global
    minimum (R's ``ar()``). With ``backcast=False`` (default, as dplR) the first
    ``order`` years of each series become NaN; ``backcast=True`` keeps full length
    with ARSTAN's bckcst initials. ``estimator="burg"`` follows COFECHA: no AR
    below 8 rings and the pre-model years kept as the data (RES(I)=F(I)).

    ``spline_period`` (COFECHA's spline variance stabilization, applied before AR)
    and ``zscore`` (each series to mean 0 / SD 1 before the master is built) are
    the extra steps the COFECHA preset switches on. ``method=`` and
    ``stabilize_period=`` are accepted as deprecated aliases of ``estimator`` and
    ``spline_period``."""
    estimator = _alias_kwarg(kwargs, "method", "estimator", estimator,
                             "normalize_for_crossdating")
    spline_period = _alias_kwarg(kwargs, "stabilize_period", "spline_period",
                                 spline_period, "normalize_for_crossdating")
    _reject_kwargs(kwargs)
    estimator = _normalize_choice(estimator, ("yw", "burg", "ols"), "estimator")
    data = _require_year_index(data, "data")

    rwi_data = detrend(data, fit="horizontal", plot=False)
    if isinstance(rwi_data, (ValueError, TypeError)):
        raise rwi_data

    to_concat = [pd.DataFrame(index=pd.Index(data.index))]
    for series in rwi_data:
        col = rwi_data[series].dropna()
        vals = col.to_numpy(dtype=float)
        if spline_period and len(vals) >= 8:
            from .smoothingspline import variance_stabilize_spline
            vals = variance_stabilize_spline(vals, period=spline_period)
        if prewhiten and len(vals) > 3:
            n = len(vals)
            ceiling = int(ar_max) if ar_max is not None else int(np.floor(10 * np.log10(n)))
            if not (estimator == "burg" and n < _COFECHA_AR_NMIN):
                w = _prewhiten_for_xdate(vals, estimator=estimator, ar_max=ceiling,
                                         first_aic_min=first_aic_min, backcast=backcast)
                if estimator == "burg" and not backcast:
                    # COFECHA's convention (RES(I)=F(I)): the pre-model years keep
                    # the data itself rather than NaN, so the series stays full length.
                    lead = np.isnan(w) & ~np.isnan(vals)
                    w[lead] = vals[lead]
                vals = w
        if zscore:
            sd = np.nanstd(vals)
            vals = (vals - np.nanmean(vals)) / sd if sd > 0 else vals - np.nanmean(vals)
        to_concat.append(pd.Series(data=vals, name=series, index=col.index))
    ready = pd.concat(to_concat, axis=1)
    ready = ready.rename_axis(data.index.name)
    return _carry_attrs(data, ready, kind="rwi")


# ---------------------------------------------------------------------------
# Main crossdating
# ---------------------------------------------------------------------------

def _dated_rings_checked(data, seg_corr):
    """Count ring measurements that actually entered a tested segment.

    A segment is "tested" for a series when it produced a correlation (a
    non-NaN cell in ``seg_corr``). A ring is counted if it is a real
    measurement (non-NaN in ``data``) whose year falls inside at least one of
    that series' tested segments. This is dplPy's analogue of COFECHA's
    "Total dated rings checked": total rings minus those in never-tested spans
    (short end pieces, or a series' portion with no overlapping master).
    """
    idx = data.index.to_numpy()
    total = 0
    for name in seg_corr.index:
        if name not in data.columns:
            continue
        row = seg_corr.loc[name]
        tested = [c for c in seg_corr.columns if not pd.isna(row[c])]
        if not tested:
            continue
        covered = np.zeros(idx.shape[0], dtype=bool)
        for c in tested:
            lo, hi = _bin_bounds(c)
            covered |= (idx >= lo) & (idx <= hi)
        colv = data[name].to_numpy(dtype=float)
        total += int(np.sum(covered & ~np.isnan(colv)))
    return total


def _summary_values(data, overall, seg_corr, first_year, last_year, n_problems,
                    weighted=False):
    """The COFECHA header-box statistics, computed the running method's way.

    ``intercorrelation`` is the mean of each series' correlation with its
    leave-one-out master (``overall['rho']`` -- the same quantity as
    `dpl.interseries_corr()`); ``mean_sensitivity`` is `dpl.sens1()` averaged
    over series. Both are read off whatever ``data`` xdate received, so the
    default (dplR) path and ``preset="COFECHA"`` each report their own numbers.

    ``weighted`` selects how those two means combine series: the dplR-native
    default is a plain mean, while COFECHA length-weights every summary statistic
    (``ZSEN = sum(SEN*N)/sum(N)``), so ``preset="COFECHA"`` passes
    ``weighted=True`` to reproduce COFECHA's box exactly.
    """
    nseries = data.shape[1]
    total_rings = int(data.count().sum())
    span = int(last_year) - int(first_year) + 1
    counts = data.count()                              # rings per series (weights)

    def _combine(vals_series):
        v = np.asarray(vals_series, dtype=float)
        w = counts.reindex(vals_series.index).to_numpy(dtype=float)
        ok = ~np.isnan(v) & ~np.isnan(w)
        if not ok.any():
            return float("nan")
        if weighted:
            return float(np.sum(v[ok] * w[ok]) / np.sum(w[ok]))
        return float(np.mean(v[ok]))

    intercorr = _combine(overall["rho"]) if nseries else float("nan")
    try:
        ms = _combine(sens1(data))
    except (ValueError, TypeError, ZeroDivisionError, FloatingPointError) as e:
        _warn("xdate: mean sensitivity could not be computed (%s); the summary "
              "reports NaN." % e, stacklevel=4)
        ms = float("nan")
    return {"n_series": nseries, "first_year": int(first_year),
            "last_year": int(last_year), "span": span,
            "total_rings": total_rings,
            "dated_rings_checked": _dated_rings_checked(data, seg_corr),
            "intercorrelation": intercorr, "mean_sensitivity": ms,
            "n_problems": int(n_problems),
            "mean_series_length": (total_rings / nseries) if nseries else float("nan")}


def _format_summary_box(s):
    """Render `_summary_values()` as a clean labeled block (no border),
    matching the header style used by `dpl.xdate_report()`."""
    def z(v, d):
        return ("%.*f" % (d, v)) if v == v else "NA"   # v == v is False for NaN
    L = []

    def _field(label, val):
        L.append("      %-28s: %s" % (label, val))
    _field("Number of dated series", s["n_series"])
    _field("Master series", "%d-%d, %d yrs"
           % (s["first_year"], s["last_year"], s["span"]))
    _field("Total rings in all series", s["total_rings"])
    _field("Total dated rings checked", s["dated_rings_checked"])
    _field("Series intercorrelation", z(s["intercorrelation"], 3))
    _field("Avg mean sensitivity", z(s["mean_sensitivity"], 3))
    _field("Segments, possible problems", s["n_problems"])
    _field("Mean length of series", z(s["mean_series_length"], 1))
    return "\n".join(L)


def xdate(data: pd.DataFrame, prewhiten=True, corr="spearman", window_length=50,
          bin_floor=100, p_val=0.05, biweight=True, lag=10, show_flags=True,
          plot=False, preset=None, seg_lag=None, ar_max=10, absent=None,
          spline_period=32, first_aic_min=True, **kwargs):
    """Crossdate a set of ring-width series against a leave-one-out master.

    Each series is normalized, optionally AR-prewhitened, and correlated against
    a biweight master built from all the other series, over segments of
    ``window_length`` years overlapping by half. Mirrors dplR's ``corr.rwl.seg``.

    Flagged segments are labelled **A** or **B** (mutually exclusive):

    * **B** -- the segment correlates better at a non-dated lag (best lag != 0): a
      possible dating shift. (dplR flags on significance only; dplPy adds this
      COFECHA-style lag flag, also used by ``preset="COFECHA"``.)
    * **A** -- lag 0 is the best match but the correlation is not significant
      (one-tailed p >= ``p_val``): weak but correctly placed.

    Lag sign: the dated segment stays fixed and the master slides past it, so a
    **negative lag means a missing ring in the series** (as in COFECHA). The same
    convention holds in ``series_corr`` and ``check``.

    Parameters
    ----------
    data : pandas.DataFrame
        detrended RWI (e.g. from ``dpl.detrend``). With ``preset="COFECHA"`` pass the
        RAW ring-width frame instead; the preset detrends internally, so do not
        pre-detrend.
    prewhiten : bool, default True
        AR-prewhiten each series (``autoreg._prewhiten``: Yule-Walker, order by the
        first local AIC minimum up to ``ar_max``; the first ``order`` years of a
        series become NaN as in dplR).
    corr : {"spearman", "pearson", "kendall"}, default "spearman"
        correlation method (case-insensitive).
    window_length : int, default 50
        segment length in years (``slide_period=`` is a deprecated alias).
    bin_floor : int, default 100
        the first segment is floored to a multiple of this.
    p_val : float, default 0.05
        significance level for the A flag.
    biweight : bool, default True
        build the master with a Tukey biweight robust mean (else arithmetic).
    lag : int, default 10
        maximum +/- lag examined for the B flag.
    show_flags : bool, default True
        print the summary box, the flag summary and lag tables.
    plot : bool, default False
        draw the segment-correlation plot.
    preset : str or None, default None
        ``"COFECHA"`` (case-insensitive) emulates the COFECHA program instead of
        dplR. Pass the RAW frame (the preset detrends and variance-stabilizes it
        itself): it uses Pearson correlation, a z-scored arithmetic leave-one-out
        master, 50-year segments on a 25-year grid anchored to each series' span, a
        99% one-tailed-t critical value, Burg AR prewhitening, and a lag search that
        slides the master past the dated segment. Adds the ``segments`` key;
        ``corr``, ``biweight``, ``bin_floor`` and ``p_val`` are ignored.
    seg_lag : int or None, default None
        segment step in years (overlap). None uses ``window_length // 2``. Preset only.
    ar_max : int or None, default 10
        AR-order ceiling for prewhitening (ARSTAN/COFECHA's 10). ``None`` uses
        dplR's ``floor(10*log10(n))``.
    first_aic_min : bool, default True
        pick the AR order at the first local AIC minimum (the ARSTAN/COFECHA rule,
        dplPy's package default) rather than the global minimum. ``ar_max=None,
        first_aic_min=False`` reproduces dplR's ``corr.rwl.seg`` prewhitening
        exactly. Ignored by the COFECHA preset (always first-minimum, Burg).
    absent : pandas.DataFrame or None, default None
        COFECHA preset only -- marks locally absent rings to omit from a series'
        correlations. A boolean (years x series) frame, or a raw ring-width frame whose
        zeros mark them (e.g. ``absent=raw_rwl == 0``). None disables the omission.
    spline_period : int, default 32
        COFECHA preset only -- spline wavelength (years) for the internal detrend and
        variance stabilization (``stabilize_period=`` is a deprecated alias).

    Returns
    -------
    dict
        with keys:

        - ``seg_corr`` -- DataFrame (series x bins) of segment correlations
        - ``p_val`` -- DataFrame (series x bins) of one-tailed p-values
        - ``overall`` -- DataFrame (series x ['rho', 'p_val']), the correlation with
          the leave-one-out master
        - ``avg_seg_corr`` -- Series (bins), mean correlation across series
        - ``flags`` -- dict {series: {'A': [...], 'B': [...]}}, each entry a dict
          ``{segment, best_lag, best_corr, lags}`` where ``lags`` is the list of
          correlations at lags ``-lag..+lag`` (floats, NaN where untestable)
        - ``bins`` -- list of "start-end" bin labels
        - ``rwi`` -- DataFrame of the normalized/prewhitened series used
          (``attrs["kind"] == "rwi"``)
        - ``n_problems`` -- int, the number of flagged segments (A + B)
        - ``summary`` -- dict, the header-box statistics (n_series, first_year,
          last_year, span, total_rings, dated_rings_checked, intercorrelation,
          mean_sensitivity, n_problems, mean_series_length)

        With ``preset="COFECHA"`` also:

        - ``segments`` -- dict {series: [{lo, hi, r0, best_lag, best_corr, n, crit, flag, lags}]}
        - ``preset`` -- "COFECHA"

    Examples
    --------
    >>> rwi = dpl.detrend(ca533, fit="spline", plot=False)
    >>> res = dpl.xdate(rwi)
    >>> res = dpl.xdate(rwl, preset="COFECHA")   # pass the RAW frame
    >>> res["n_problems"]

    References
    ----------
    .. [1] https://opendendro.github.io/dplPy/reference/crossdating/#xdate
    """
    window_length = _alias_kwarg(kwargs, "slide_period", "window_length",
                                 window_length, "xdate")
    spline_period = _alias_kwarg(kwargs, "stabilize_period", "spline_period",
                                 spline_period, "xdate")
    plot = _plot_kwarg(plot, kwargs)
    _require_dataframe(data)
    data = _require_year_index(data, "data")
    preset = _normalize_choice(preset, ("COFECHA",), "preset", none_ok=True)
    window_length = int(window_length)
    lag = int(lag)

    # COFECHA emulation: fixed 50-yr segments on a 25-yr grid, anchored to each
    # series' own first/last year, correlated (Pearson) against an arithmetic
    # leave-one-out master, and flagged against COFECHA's derived critical value.
    if preset == "COFECHA":
        return _xdate_cofecha(data, prewhiten=prewhiten, window_length=window_length,
                              seg_lag=seg_lag, lag=lag, ar_max=ar_max, absent=absent,
                              spline_period=spline_period,
                              show_flags=show_flags, plot=plot)

    method = _normalize_corr(corr)

    # normalize + prewhiten, then work on a dense (years x series) matrix on a
    # consecutive-year grid (like dplR), so the leave-one-out master is a single
    # vectorized robust mean rather than a per-series chronology rebuild.
    ready = normalize_for_crossdating(data, prewhiten, ar_max=ar_max,
                                      first_aic_min=first_aic_min)
    ready, years, first_year, last_year = dense_year_grid(ready)
    ready = _carry_attrs(data, ready, kind="rwi")
    series_names = list(ready.columns)
    M = ready.to_numpy(dtype=float)                 # (nyears, nseries), NaN gaps
    nyears, nseries = M.shape
    good = np.array([np.sum(~np.isnan(M[:, i])) > 3 for i in range(nseries)])

    bins, _ = get_bins(first_year, last_year, bin_floor, window_length)
    bin_bounds = [_bin_bounds(b) for b in bins]
    row_master = _row_biweight if biweight else _row_mean

    seg_corr = pd.DataFrame(index=series_names, columns=bins, dtype=float)
    seg_pval = pd.DataFrame(index=series_names, columns=bins, dtype=float)
    overall = pd.DataFrame(index=series_names, columns=["rho", "p_val"], dtype=float)
    flags = {}

    for i, name in enumerate(series_names):
        keep = good.copy()
        keep[i] = False
        master = row_master(M[:, keep]) if keep.any() else np.full(nyears, np.nan)
        series = M[:, i]

        overall.loc[name, "rho"], overall.loc[name, "p_val"] = _corr_pval(series, master, method)

        a_flags, b_flags = [], []
        for (lo, hi), blabel in zip(bin_bounds, bins):
            mask = (years >= lo) & (years <= hi)
            seg = series[mask]
            mas = master[mask]
            if mask.sum() != window_length or np.isnan(seg).any() or np.isnan(mas).any():
                continue                              # require complete overlap (dplR)
            rho, pv = _corr_pval(seg, mas, method)
            seg_corr.loc[name, blabel] = rho
            seg_pval.loc[name, blabel] = pv
            # The per-lag table is computed for every segment (cheap) so that both
            # flag kinds can report it (users want to see the lag profile behind an
            # A flag too, not only a B flag).
            lag_row, best_lag, best_coeff = _lag_table(series, master, years, lo, hi,
                                                       window_length, method, lag)
            entry = {"segment": blabel, "best_lag": best_lag,
                     "best_corr": best_coeff, "lags": [float(v) for v in lag_row]}
            # Classify each segment as exactly ONE of A or B, COFECHA-style:
            #   B -- the segment correlates better with the master at a NON-dated
            #        lag (best_lag != 0): a possible dating shift. This is the
            #        COFECHA-derived screen (SLSG: IF MXCOR != dated, no margin);
            #        dplR's corr.rwl.seg has no B flag of its own.
            #   A -- the dated (lag-0) position IS the best, but the correlation is
            #        still not significant (one-tailed p >= p_val): a weak segment
            #        that is nonetheless best where dated ('best' is 0, by
            #        definition).
            # A and B are mutually exclusive: a not-significant segment whose best
            # match is at another lag is a B (the dating shift is the real issue),
            # not an A. dplR flags every non-significant segment (no lag concept);
            # here those split into A (lag 0 is best) and B (a better lag exists),
            # so A never carries a non-zero best lag.
            if best_lag != 0:
                b_flags.append(dict(entry))
            elif not np.isnan(pv) and pv >= p_val:
                a_flags.append(dict(entry))
        if a_flags or b_flags:
            flags[name] = {"A": a_flags, "B": b_flags}

    avg_seg = seg_corr.mean(axis=0, skipna=True)
    overall["rho"] = overall["rho"].round(3)          # report the master correlation to 3 dp
    n_problems = sum(len(f["A"]) + len(f["B"]) for f in flags.values())
    summary = _summary_values(data, overall, seg_corr, first_year, last_year,
                              n_problems)

    if show_flags:
        print(_format_summary_box(summary))
        print()
        _print_flags(flags, lag)
    if plot:
        _plot_crs(seg_corr, seg_pval, ready, bins, bin_bounds, p_val,
                  window_length, window_length // 2)

    return {"seg_corr": seg_corr, "p_val": seg_pval, "overall": overall,
            "avg_seg_corr": avg_seg, "flags": flags, "bins": bins,
            "rwi": ready, "n_problems": n_problems, "summary": summary}


# ---------------------------------------------------------------------------
# COFECHA emulation (preset="COFECHA")
# ---------------------------------------------------------------------------

def _cofecha_segments(y0, y1, seg_len=50, seg_lag=25):
    """COFECHA's segment layout for one series spanning [y0, y1].

    Interior segments are ``seg_len``-year windows beginning on every multiple
    of ``seg_lag`` that fits wholly inside the span (a 50% overlap for the usual
    50/25). To that COFECHA adds a segment *anchored to the series' first year*
    (``[y0, y0+seg_len-1]``) and one *anchored to its last year*
    (``[y1-seg_len+1, y1]``) whenever those endpoints don't already fall on the
    grid -- so the very start and end of every series are always tested at full
    window length. A series shorter than one window is tested as a single
    short segment ``[y0, y1]``, against the stiffer critical value
    `_cofecha_crit()` gives for its length (audit B13).
    Returns a sorted, de-duplicated list of ``(lo, hi)`` inclusive year pairs."""
    grid = []
    s = ((y0 + seg_lag - 1) // seg_lag) * seg_lag      # first multiple >= y0
    while s + seg_len - 1 <= y1:
        grid.append((s, s + seg_len - 1))
        s += seg_lag
    segs = []
    if (not grid or grid[0][0] != y0) and y0 + seg_len - 1 <= y1:
        segs.append((y0, y0 + seg_len - 1))            # start anchor
    segs.extend(grid)
    if (not grid or grid[-1][1] != y1) and y1 - seg_len + 1 >= y0:
        segs.append((y1 - seg_len + 1, y1))            # end anchor
    if not segs:
        segs = [(y0, y1)]                              # series shorter than window
    out, seen = [], set()
    for t in sorted(segs):
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _cofecha_crit(n, conf=0.99):
    """COFECHA's per-segment critical correlation: the one-tailed Student-t
    value at ``conf`` confidence on ``n-2`` degrees of freedom, converted to r.
    Reproduces COFECHA's table exactly (n=50 -> 0.3281, n=25 -> 0.4622). ``n`` is
    the number of rings actually correlated, so a segment shortened by omitted
    absent rings, or a series shorter than one window tested as a single segment,
    faces a stiffer threshold, as in COFECHA."""
    df = n - 2
    if df < 1:
        return np.inf
    t = scipy.stats.t.ppf(conf, df)
    return float(t / np.sqrt(t * t + df))


def _xdate_cofecha(data, prewhiten=True, window_length=50, seg_lag=None, lag=10,
                   ar_max=10, spline_period=32, absent=None, show_flags=True,
                   plot=False):
    """COFECHA-style crossdating (see `xdate()` with ``preset="COFECHA"``).

    Takes the RAW ring-width frame and detrends it internally with a rigid
    ``spline_period``-year cubic spline (ratios), the way COFECHA does, so the
    caller passes raw measurements -- not a pre-detrended index.

    Differs from the dplR-faithful default in exactly the ways COFECHA does:
    Pearson correlation, an *arithmetic* leave-one-out master of z-scored series,
    COFECHA's segment anchoring (`_cofecha_segments()`), its derived critical
    value (`_cofecha_crit()`) rather than a p-value threshold, COFECHA's
    spline variance stabilization before AR, and Ed Cook / Paul Krusic's
    Burg AR prewhitening (ceiling 10, first-local-AIC-minimum order, N>=8) rather
    than dplR's Yule-Walker global-minimum search. A segment is flagged **A** when
    the dated (lag-0) position is the highest correlation over -lag..+lag but still
    falls below the critical value, and **B** when some non-dated lag correlates
    higher (a possible dating error) -- the two counts that make up COFECHA's
    "Segments, possible problems". The lag search slides the master against the
    fixed dated segment, matching COFECHA's SLSG."""
    seg_len = int(window_length)
    if seg_lag is None:
        seg_lag = seg_len // 2
    if ar_max is None:
        ar_max = 10                                     # Cook/Krusic ARSTAN ceiling

    # COFECHA works from RAW ring widths: it removes the growth trend itself with a
    # rigid cubic spline (ratios), then variance-stabilizes, prewhitens, and
    # z-scores. So detrend here -- the preset takes the raw measurement frame
    # directly, and callers must NOT pre-detrend (that would detrend twice). The
    # spline wavelength is ``spline_period`` (COFECHA's 32 years), the same
    # stiffness used for the variance-stabilization step inside
    # normalize_for_crossdating below.
    rwi = detrend(data, fit="Spline", spline_period=spline_period, plot=False)
    if isinstance(rwi, (ValueError, TypeError)):
        raise rwi
    ready = normalize_for_crossdating(rwi, prewhiten, ar_max=ar_max,
                                      first_aic_min=True, estimator="burg",
                                      spline_period=spline_period, zscore=True)
    ready, years, first_year, last_year = dense_year_grid(ready)
    ready = _carry_attrs(data, ready, kind="rwi")
    series_names = list(ready.columns)
    M = ready.to_numpy(dtype=float)
    nyears, nseries = M.shape
    good = np.array([np.sum(~np.isnan(M[:, i])) > 3 for i in range(nseries)])
    yr_pos = {int(y): k for k, y in enumerate(years)}

    # COFECHA's checkable span: a segment can only be tested where TWO OR MORE
    # series are present, so that after the current series is removed the
    # leave-one-out master still has at least one series to correlate against.
    # COFECHA clips every series to the first/last year with >=2 series before it
    # lays out segments (cofecha_erc.f: JYM2/LYM2 and the "[*] ... cannot be
    # checked -- not matched by another series" trim); this is what stops a series
    # that extends before the multi-series portion from being scored against an
    # essentially empty master (e.g. ca533's CAM211 before 695, where it is the
    # only series). We reproduce that clip below. (COFECHA also intersects with the
    # continuous portion of the master; that refinement is rarely the binding
    # constraint and is not ported.)
    depth = np.sum(~np.isnan(M), axis=1)
    ge2 = np.where(depth >= 2)[0]
    check_lo = int(years[ge2[0]]) if ge2.size else None
    check_hi = int(years[ge2[-1]]) if ge2.size else None

    # Absent-ring ("omit absent rings") mask, COFECHA's QAC=Y option: years where
    # a series' raw ring width is zero are dropped from that series' segment
    # correlations. Ring widths become non-zero after detrending, so the mask must
    # come from a pre-detrend source -- ``absent`` (a boolean DataFrame, or a raw
    # ring-width DataFrame whose zeros mark absent rings). None disables it.
    absent_M = None
    if absent is not None:
        amask = absent if absent.to_numpy().dtype == bool else (absent == 0)
        amask = amask.reindex(index=years, columns=series_names, fill_value=False)
        absent_M = amask.to_numpy(dtype=bool)

    # display grid: one column per 25-yr start across the whole data set
    grid_starts = list(range(int(first_year // seg_lag * seg_lag),
                             int(last_year) + 1, seg_lag))
    grid_labels = {g: "%d-%d" % (g, g + seg_len - 1) for g in grid_starts}
    bins = [grid_labels[g] for g in grid_starts]

    seg_corr = pd.DataFrame(index=series_names, columns=bins, dtype=float)
    seg_pval = pd.DataFrame(index=series_names, columns=bins, dtype=float)
    overall = pd.DataFrame(index=series_names, columns=["rho", "p_val"], dtype=float)
    flags, segments = {}, {}

    for i, name in enumerate(series_names):
        col = M[:, i]
        fin = np.where(~np.isnan(col))[0]
        if fin.size == 0:
            continue
        y0, y1 = int(years[fin[0]]), int(years[fin[-1]])
        # Clip the segmented range to the >=2-series span (see above), so segments
        # are never anchored to a year the master cannot cover.
        if check_lo is not None:
            y0, y1 = max(y0, check_lo), min(y1, check_hi)
        keep = good.copy()
        keep[i] = False
        master = _row_mean(M[:, keep]) if keep.any() else np.full(nyears, np.nan)

        overall.loc[name, "rho"], overall.loc[name, "p_val"] = \
            _corr_pval(col, master, "pearson")

        ab_col = absent_M[:, i] if absent_M is not None else None
        a_flags, b_flags, seg_list, used_cols = [], [], [], set()
        # No testable span (this series never overlaps the >=2-series portion):
        # leave it with no segments/flags, as COFECHA drops such a series.
        seg_ranges = _cofecha_segments(y0, y1, seg_len, seg_lag) if y0 <= y1 else []
        for (lo, hi) in seg_ranges:
            lag_row, best_lag, best_coeff, r0, n0 = _cofecha_lag_table(
                col, master, yr_pos, first_year, last_year, lo, hi, lag,
                absent=ab_col)
            if np.isnan(r0):
                continue
            _, pv = _corr_pval(col[yr_pos[lo]:yr_pos[hi] + 1],
                               master[yr_pos[lo]:yr_pos[hi] + 1], "pearson")
            crit = _cofecha_crit(n0 if n0 else seg_len)
            flag = ""
            entry = {"segment": "%d-%d" % (lo, hi), "best_lag": best_lag,
                     "best_corr": best_coeff, "lags": [float(v) for v in lag_row]}
            if best_lag == 0 and r0 < crit:
                flag = "A"
                a_flags.append(entry)
            elif best_lag != 0:
                flag = "B"
                b_flags.append(entry)
            seg_list.append({"lo": lo, "hi": hi, "r0": r0, "best_lag": best_lag,
                             "best_corr": best_coeff, "n": n0, "crit": crit,
                             "flag": flag, "lags": entry["lags"]})
            # place on the display grid (bump on collision, e.g. end anchor)
            gcol = lo // seg_lag * seg_lag
            while gcol in used_cols and gcol + seg_lag <= grid_starts[-1]:
                gcol += seg_lag
            if gcol in grid_labels:
                used_cols.add(gcol)
                seg_corr.loc[name, grid_labels[gcol]] = r0
                seg_pval.loc[name, grid_labels[gcol]] = pv

        segments[name] = seg_list
        if a_flags or b_flags:
            flags[name] = {"A": a_flags, "B": b_flags}

    avg_seg = seg_corr.mean(axis=0, skipna=True)
    overall["rho"] = overall["rho"].round(3)          # report the master correlation to 3 dp
    n_problems = sum(len(f["A"]) + len(f["B"]) for f in flags.values())
    summary = _summary_values(data, overall, seg_corr, first_year, last_year,
                              n_problems, weighted=True)

    if show_flags:
        print(_format_summary_box(summary))
        print()
        _print_flags(flags, lag)
    if plot:
        bin_bounds = [_bin_bounds(b) for b in bins]
        _plot_crs(seg_corr, seg_pval, ready, bins, bin_bounds,
                  _cofecha_crit(seg_len), seg_len, seg_lag)

    return {"seg_corr": seg_corr, "p_val": seg_pval, "overall": overall,
            "avg_seg_corr": avg_seg, "flags": flags, "bins": bins,
            "rwi": ready, "segments": segments, "n_problems": n_problems,
            "summary": summary, "preset": "COFECHA"}


def _cofecha_lag_table(series, master, yr_pos, first_year, last_year,
                       lo, hi, lag, absent=None):
    """Correlate the fixed dated series segment ``[lo, hi]`` against the master
    over lags -lag..+lag (Pearson, via `_fast_corr()`), sliding the MASTER window
    while the series segment stays put -- exactly COFECHA's SLSG (``ZSERM`` fixed,
    ``YMSMA(IA+..)`` shifted). Returns (corrs, best_lag, best_corr, r0, n_at_lag0)
    with ``corrs`` an array over the lags (NaN where untestable). Only full master
    windows lying inside the data span count. A NEGATIVE best lag means a missing
    ring in the series. The segment may be shorter than the standard window (a
    series shorter than one window is tested whole, audit B13).

    ``absent`` is an optional boolean array over the whole year grid (COFECHA's
    QAC=Y "omit absent rings"): years where the tested series has an absent ring
    are dropped from every correlation. The mask is aligned to the fixed segment
    (it follows the series, not the sliding master), as in COFECHA's CORRP0."""
    n_lags = 2 * lag + 1
    empty = np.full(n_lags, np.nan)
    if lo not in yr_pos or hi not in yr_pos:
        return empty, 0, np.nan, np.nan, 0
    seg = series[yr_pos[lo]:yr_pos[hi] + 1]              # fixed dated segment
    n_seg = hi - lo + 1
    if seg.shape[0] != n_seg or np.isnan(seg).all():
        return empty, 0, np.nan, np.nan, 0
    if absent is not None:
        seg = np.where(absent[yr_pos[lo]:yr_pos[hi] + 1], np.nan, seg)

    row = np.full(n_lags, np.nan)
    r0, n0 = np.nan, 0
    for k, shift in enumerate(range(-lag, lag + 1)):
        mlo, mhi = lo + shift, hi + shift               # slide the master window
        if mlo < first_year or mhi > last_year:
            continue
        mas = master[yr_pos[mlo]:yr_pos[mhi] + 1]
        if mas.shape[0] != n_seg:
            continue
        row[k] = _fast_corr(seg, mas, "pearson")
        if shift == 0:
            r0 = row[k]
            n0 = int((~np.isnan(seg) & ~np.isnan(mas)).sum())
    best_lag, best_coeff = _best_lag(row, lag)
    return row, best_lag, best_coeff, r0, n0


def _best_lag(corrs, lag):
    """(best_lag, best_corr) of a lag row over -lag..+lag: the highest finite
    correlation, the first (most negative lag) on ties; (0, NaN) when none."""
    corrs = np.asarray(corrs, dtype=float)
    if not np.isfinite(corrs).any():
        return 0, np.nan
    k = int(np.nanargmax(corrs))
    return k - int(lag), float(corrs[k])


def _window(arr, years, lo, hi):
    """``arr`` over calendar years ``lo..hi`` (inclusive) on the dense, sorted
    year grid ``years``; NaN where the window runs off the grid."""
    y0 = int(years[0])
    n = arr.shape[0]
    out = np.full(int(hi) - int(lo) + 1, np.nan)
    a, b = int(lo) - y0, int(hi) - y0 + 1
    ca, cb = max(a, 0), min(b, n)
    if cb > ca:
        out[ca - a:cb - a] = arr[ca:cb]
    return out


def _lag_table(series, master, years, lo, hi, window_length, method, lag,
               min_overlap=None):
    """The one lag scan behind xdate, series_corr and check.

    Correlate the fixed dated series segment ``[lo, hi]`` against the master at
    lags -lag..+lag, sliding the MASTER window while the segment stays put --
    COFECHA's SLSG convention, so a NEGATIVE best lag means a MISSING ring in the
    series. Returns ``(corrs, best_lag, best_corr)``: an array of correlations
    over the lags (NaN where untestable), the lag of the highest one, and that
    value (NaN when nothing could be tested). ``years`` is the dense, sorted
    year grid ``series`` and ``master`` sit on.

    With ``min_overlap=None`` (xdate, series_corr) only COMPLETE windows count,
    as in dplR: the segment must span exactly ``window_length`` years with no
    NaN, and a shifted master window with any NaN is skipped. With an integer
    ``min_overlap`` (check) the correlation is taken over the pairwise-complete
    pairs of each shifted window, requiring at least that many. Correlation uses
    ``method`` (the caller's choice), so the lag table is consistent with the
    segment correlations around it."""
    n_lags = 2 * lag + 1
    shifts = np.arange(-lag, lag + 1)
    seg = _window(series, years, lo, hi)
    corrs = np.full(n_lags, np.nan)
    if min_overlap is None:
        if window_length is not None and seg.shape[0] != int(window_length):
            return corrs, 0, np.nan
        if np.isnan(seg).any():
            return corrs, 0, np.nan
    elif int((~np.isnan(seg)).sum()) < int(min_overlap):
        return corrs, 0, np.nan

    # Stack the shifted MASTER windows into one matrix. Sliding the master rather
    # than the series keeps the tested segment fixed (COFECHA/SLSG), so the "best
    # at another lag" question is asked exactly as COFECHA asks it.
    Wm = np.vstack([_window(master, years, lo + int(s), hi + int(s)) for s in shifts])

    if min_overlap is not None:
        for k in range(n_lags):
            ok = ~np.isnan(seg) & ~np.isnan(Wm[k])
            if ok.sum() >= int(min_overlap):
                corrs[k] = _fast_corr(seg, Wm[k], method)
        return (corrs,) + _best_lag(corrs, lag)

    valid = ~np.isnan(Wm).any(axis=1)
    if valid.any():
        if method == "kendall":
            for k in np.where(valid)[0]:
                corrs[k] = _fast_corr(seg, Wm[k], "kendall")
        else:
            # rank (spearman) or centre (pearson) once, then one matrix product
            if method == "spearman":
                sv = scipy.stats.rankdata(seg)
                Mv = scipy.stats.rankdata(Wm[valid], axis=1)
            else:  # pearson
                sv, Mv = seg, Wm[valid]
            a = sv - sv.mean()
            B = Mv - Mv.mean(axis=1, keepdims=True)
            den = np.sqrt(np.dot(a, a) * (B * B).sum(axis=1))
            with np.errstate(invalid="ignore", divide="ignore"):
                corrs[valid] = np.where(den > 0, (B @ a) / den, np.nan)
    return (corrs,) + _best_lag(corrs, lag)


def _format_lag_cells(lags, width=6):
    """Fixed-width text cells for a numeric lag row (blank where NaN)."""
    return "".join((("%.2f" % v) if np.isfinite(v) else "").rjust(width) for v in lags)


def _print_flags(flags, lag):
    """Print the per-segment lag tables for the A (not significant as dated) and
    B (correlates better at another lag) flags. The lag-offset column headers are
    width-matched to the correlation cells below them so the numbers line up (the
    same fixed-column style as xdate_report). Negative lag = missing ring."""
    if not flags:
        print()
        return
    lags = list(range(-lag, lag + 1))
    CW = 6                                        # width of each lag/correlation column
    axis = "".join((("%+d" % k) if k != 0 else "0").rjust(CW) for k in lags)
    # Header and data rows share the same-width prefix so columns align exactly.
    head = "      {0:<11} {1:>4}  {2}".format("Segment", "best", axis)

    def line(e):
        return "      {0:<11} {1:>+4d}  {2}".format(e["segment"], e["best_lag"],
                                                   _format_lag_cells(e["lags"], CW))

    for name, fl in flags.items():
        print("Flags for", name)
        if fl["A"]:
            print("  [A] not significant as dated ('best' = lag of highest correlation):")
            print(head)
            for e in fl["A"]:
                print(line(e))
        if fl["B"]:
            print("  [B] correlates better at another lag ('best' = that lag; "
                  "negative = missing ring):")
            print(head)
            for e in fl["B"]:
                print(line(e))
        print()


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

# RColorBrewer "Set1" -- the same three colors dplR's plot.crs uses.
_CRS_EXTENT = "#4DAF4A"   # green: the series exists but no full segment was analyzed
_CRS_DATED  = "#377EB8"   # blue:  segment analyzed and significantly correlated
_CRS_FLAG   = "#E41A1C"   # red:   segment flagged (p >= pcrit) -- possible dating error


def _plot_crs(seg_corr, seg_pval, rwi, bins, bin_bounds, pcrit,
              seg_length, seg_lag):
    """dplR-style crossdating overview (mirrors corr.rwl.seg / plot.crs).

    One row per series, sorted by first year (earliest at the bottom). The
    50%-overlapping segments are split into two offset half-rows -- even-indexed
    segments on the bottom half, odd-indexed on the top half, so consecutive
    (overlapping) segments never paint over each other. In each half a green bar
    marks the series extent, blue marks segments that date well, and red marks
    flagged segments (segment p-value >= ``pcrit``). Because neighbouring
    segments alternate halves, you can read off exactly which segment start
    first drops to non-significant. Left-side labels only.
    """
    names = list(seg_corr.index)
    # Extent per series from the analyzed rwi (matches dplR's use of x$rwi),
    # i.e. the prewhitened series' first/last finite year.
    first, last = {}, {}
    for name in names:
        col = rwi[name]
        vi, vl = col.first_valid_index(), col.last_valid_index()
        first[name] = np.nan if vi is None else float(vi)
        last[name] = np.nan if vl is None else float(vl)
    order = sorted(names, key=lambda n: (np.isnan(first[n]), first[n]))

    valid_first = [first[n] for n in names if not np.isnan(first[n])]
    valid_last = [last[n] for n in names if not np.isnan(last[n])]
    if not valid_first:
        raise ValueError("no datable series to plot")
    minyr, maxyr = min(valid_first), max(valid_last)
    span = max(maxyr - minyr, 1)
    n = len(order)
    nbins = len(bins)

    fig, ax = plt.subplots(figsize=(max(span / 90, 8), max(n * 0.34, 5)))
    ax.set_facecolor("white")
    qh = 0.30   # half-height of each offset sub-row

    # faint grey stripes on alternating rows (dplR's grey90)
    for k in range(0, n, 2):
        ax.add_patch(Rectangle((minyr - span, k + 0.5), 3 * span, 1.0,
                               facecolor="#eeeeee", edgecolor="none", zorder=0))
    # dotted grey guides at each bin boundary
    for b in sorted({lo for lo, hi in bin_bounds} | {hi + 1 for lo, hi in bin_bounds}):
        ax.axvline(b, color="grey", lw=0.4, ls=":", zorder=1)

    # even-indexed segments -> bottom half, odd-indexed -> top half (dplR stagger)
    halves = ((range(0, nbins, 2), -qh, 0.0), (range(1, nbins, 2), 0.0, qh))
    for k, name in enumerate(order):
        y = k + 1
        if np.isnan(first[name]):
            continue
        ext_w = last[name] + 1 - first[name]
        for idxs, dyb, dyt in halves:
            yb, yt = y + dyb, y + dyt
            # green extent (base layer for this half)
            ax.add_patch(Rectangle((first[name], yb), ext_w, yt - yb,
                                   facecolor=_CRS_EXTENT, edgecolor="none",
                                   zorder=2))
            # blue for analyzed segments, red (on top) for flagged ones
            for j in idxs:
                lo, hi = bin_bounds[j]
                pv = seg_pval.loc[name, bins[j]]
                if pd.isna(pv):
                    continue
                flagged = pv >= pcrit
                ax.add_patch(Rectangle((lo, yb), hi + 1 - lo, yt - yb,
                                       facecolor=_CRS_FLAG if flagged else _CRS_DATED,
                                       edgecolor="none", zorder=4 if flagged else 3))
        # white centre line separating the two offset halves (like dplR)
        ax.hlines(y, first[name], last[name] + 1, color="white", lw=0.6, zorder=5)

    ax.set_xlim(minyr - span * 0.02, maxyr + span * 0.02)
    ax.set_ylim(0.3, n + 0.7)

    # Series labels alternate left / right (dplR's axis 2 / axis 4) so each has
    # room to breathe -- rows 1,3,5.. on the left, rows 2,4,6.. on the right.
    positions = list(range(1, n + 1))
    ax.set_yticks(positions[0::2])
    ax.set_yticklabels(order[0::2], fontsize=13)
    axr = ax.secondary_yaxis("right")
    axr.set_yticks(positions[1::2])
    axr.set_yticklabels(order[1::2], fontsize=13)
    axr.tick_params(length=3, color="black")

    # Offset segment-boundary years: the bottom axis carries the lower half-row
    # (even-indexed) segment boundaries, the top axis the upper half-row
    # (odd-indexed) ones -- offset by seg_lag, exactly like dplR's axis 1 / 3.
    def _seg_bounds(idxs):
        if not idxs:
            return []
        return [bin_bounds[j][0] for j in idxs] + [bin_bounds[idxs[-1]][1] + 1]

    bot_ticks = _thin(_seg_bounds(list(range(0, nbins, 2))))
    top_ticks = _thin(_seg_bounds(list(range(1, nbins, 2))))
    ax.set_xticks(bot_ticks)
    ax.tick_params(axis="x", labelsize=15, length=4, color="black")
    ax.tick_params(axis="y", length=3, color="black")
    axt = ax.secondary_xaxis("top")
    axt.set_xticks(top_ticks)
    axt.tick_params(axis="x", labelsize=15, length=4, color="black")

    ax.set_xlabel("Year", fontsize=16)
    fig.text(0.5, 0.005,
             "Segments: length=%d, lag=%d" % (seg_length, seg_lag),
             ha="center", fontsize=13)
    for spine in ax.spines.values():
        spine.set_color("black")

    handles = [Rectangle((0, 0), 1, 1, facecolor=c) for c in
               (_CRS_EXTENT, _CRS_DATED, _CRS_FLAG)]
    labels = ["series extent", "dated (p<%g)" % pcrit, "flagged (p≥%g)" % pcrit]
    ax.legend(handles, labels, loc="upper left", fontsize=11,
              framealpha=0.95, edgecolor="black")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    from ._plot_style import finalize_font
    finalize_font(fig)
    plt.show()
    return ax


def xdate_plot(data: pd.DataFrame, prewhiten=True, corr="spearman",
               window_length=50, bin_floor=100, p_val=0.05, biweight=True, **kwargs):
    """dplR-style crossdating overview for a set of series (see corr.rwl.seg).

    A thin wrapper: crossdates ``data`` with `xdate()` (same parameters;
    ``slide_period=`` is a deprecated alias of ``window_length``) and draws the
    green/blue/red segment plot -- green = series extent, blue = a segment that
    correlates significantly with the master, red = a flagged segment
    (p >= ``p_val``). Returns the matplotlib Axes.

    Note: this plot shows only the **significance** screen (dplR's corr.rwl.seg,
    i.e. the **A** flag) -- red marks a segment whose dated (lag-0) correlation is
    not significant. It does NOT show the **B** flag (a segment that correlates
    better at a non-dated lag -- a possible dating shift), which dplR's plot has no
    concept of. A B-flagged segment that still dates significantly is drawn blue
    here, so it looks fine on the plot even though `xdate()` reports it as a
    problem. Always read the text ``xdate()`` flag report alongside this plot to
    catch B (lag) flags.

    References
    ----------
    .. [1] https://opendendro.github.io/dplPy/reference/crossdating/#xdate_plot
    """
    window_length = _alias_kwarg(kwargs, "slide_period", "window_length",
                                 window_length, "xdate_plot")
    _reject_kwargs(kwargs)
    window_length = int(window_length)
    res = xdate(data, prewhiten=prewhiten, corr=corr, window_length=window_length,
                bin_floor=bin_floor, p_val=p_val, biweight=biweight,
                show_flags=False, plot=False)
    bin_bounds = [_bin_bounds(b) for b in res["bins"]]
    return _plot_crs(res["seg_corr"], res["p_val"], res["rwi"], res["bins"],
                     bin_bounds, p_val, window_length, window_length // 2)
