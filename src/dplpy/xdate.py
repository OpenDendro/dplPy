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
#   normalize each series (divide by its mean), optionally Yule-Walker prewhiten
#   (matching dplR's ar()), build a leave-one-out biweight master, and correlate
#   each series against it over overlapping segments -- reporting per-segment
#   correlation, its one-tailed p-value, an overall correlation, and flags for
#   (A) non-significant segments and (B) segments that correlate better at a lag
#   (the COFECHA-style lag table).
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
from ._validate import _require_dataframe, _normalize_corr
from .tbrm import tbrm_rows
import numpy as np
import scipy
import warnings
import re


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _ar_yw_prewhiten(x, ar_max=None, first_aic_min=False, backcast=False):
    """Prewhiten a 1-D series with a Yule-Walker AR model, matching dplR's ar():
    AIC order selection up to floor(10*log10(n)), residuals + series mean, and
    the series length preserved (the first `order` values become NaN). Validated
    to reproduce R's ar() to ~1e-15.  `x` must be NaN-free.

    ``ar_max`` optionally overrides the order ceiling (else floor(10*log10(n))).
    ``first_aic_min`` switches order selection from the GLOBAL AIC minimum (R's
    ar(), the dplPy/dplR default) to the FIRST LOCAL AIC minimum -- Ed Cook /
    Paul Krusic's ARSTAN rule (:func:`autoreg._first_aic_min`, the same rule
    ``chron_ars`` uses). The COFECHA preset uses ``ar_max=10, first_aic_min=True``
    because dplR's global-min-to-floor(10*log10(n)) can pick very high orders
    (e.g. 21) that COFECHA/ARSTAN never would -- inflating the problem-segment
    count, depressing the inter-series correlation, and discarding many early
    years of a series.

    ``backcast`` (ARSTAN's ``bckcst``) fills the ``order`` pre-sample values by a
    zero-innovation reverse-AR recursion (``x_bc[t] = sum_j phi_j * x[t+j]``) so
    residuals can be formed for *every* year -- the whitened series keeps its
    full length instead of losing the first ``order`` values to NaN. dplR/dplPy
    leave this False (NaN-padded, matching R's ar()); the COFECHA preset sets it
    True, as ARSTAN does, so a series' first segment is anchored at its true
    first year."""
    from .autoreg import _first_aic_min
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return x.astype(float).copy()
    ceiling = int(ar_max) if ar_max is not None else int(np.floor(10 * np.log10(n)))
    order_max = min(n - 1, ceiling)
    if order_max < 1:
        return x.astype(float).copy()
    xbar = x.mean()
    xc = x - xbar
    acov = np.array([np.dot(xc[:n - k], xc[k:]) / n for k in range(order_max + 1)])
    if acov[0] == 0:
        return x.astype(float).copy()
    # Levinson-Durbin: coefficients and prediction variance for every order
    v = acov[0]
    a = np.zeros(order_max + 1)
    var_pred = [v]
    coeffs_by_order = [np.array([])]
    for k in range(1, order_max + 1):
        acc = acov[k] - (np.dot(a[1:k], acov[1:k][::-1]) if k > 1 else 0.0)
        refl = acc / v
        new_a = a.copy()
        new_a[k] = refl
        for j in range(1, k):
            new_a[j] = a[j] - refl * a[k - j]
        a = new_a
        v = v * (1 - refl ** 2)
        var_pred.append(v)
        coeffs_by_order.append(a[1:k + 1].copy())
    var_pred = np.array(var_pred)
    aic = n * np.log(var_pred) + 2 * np.arange(order_max + 1)
    order = _first_aic_min(list(aic)) if first_aic_min else int(np.argmin(aic))
    phi = coeffs_by_order[order]
    out = np.full(n, np.nan)
    if order == 0:
        out[:] = xc + xbar
        return out
    if backcast:
        # ARSTAN bckcst: prepend `order` zero-innovation backcasts, then form
        # residuals for all n years (no leading NaN). ext = [x_bc | xc].
        ext = np.empty(order + n)
        ext[order:] = xc
        for idx in range(order - 1, -1, -1):
            ext[idx] = np.dot(phi, ext[idx + 1:idx + order + 1])
        out = np.empty(n)
        for t in range(n):
            out[t] = ext[order + t] - np.dot(phi, ext[t:order + t][::-1])
        return out + xbar
    for t in range(order, n):
        out[t] = xc[t] - np.dot(phi, xc[t - order:t][::-1])
    return out + xbar


def _ar_burg_prewhiten(x, ar_max=10, first_aic_min=True, nmin=8):
    """Prewhiten a series the way COFECHA's MEMPR does: Burg (maximum-entropy) AR
    coefficients (from :func:`autoreg._burg_params_aic`, ceiling ``ar_max``, order
    by first-local-AIC-minimum), residuals formed for years after the model order,
    and -- COFECHA's convention -- the first ``order`` (pre-model) values kept as
    the centred filtered data (``RES(I)=F(I)``) rather than NaN or a backcast, so
    the whitened series keeps full length. Series shorter than ``nmin`` (COFECHA's
    N>=8 rule) are returned unmodelled. `x` must be NaN-free."""
    from .autoreg import _burg_params_aic
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < nmin:
        return x.astype(float).copy()               # COFECHA: no AR below N=8
    params = _burg_params_aic(x, min(int(ar_max), n - 1), aic=True,
                              first_aic_min=first_aic_min)
    phi = params[1:]
    order = len(phi)
    xbar = x.mean()
    xc = x - xbar
    out = xc.copy()                                 # leading `order` = centred data
    for t in range(order, n):
        out[t] = xc[t] - np.dot(phi, xc[t - order:t][::-1])
    return out + xbar


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
    """Correlation only (no p-value), over pairwise-complete elements -- used for
    the many lag-table correlations where the p-value is not needed. For spearman
    `b` may be passed pre-ranked (b_ranked=True) to avoid re-ranking the master."""
    ok = ~np.isnan(a) & ~np.isnan(b)
    if ok.sum() < 3:
        return np.nan
    a2, b2 = a[ok], b[ok]
    if method == "kendall":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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
        warnings.simplefilter("ignore")
        if method == "pearson":
            r = scipy.stats.pearsonr(a, b, alternative="greater")
        elif method == "kendall":
            r = scipy.stats.kendalltau(a, b, alternative="greater")
        else:
            r = scipy.stats.spearmanr(a, b, alternative="greater")
    stat = r.statistic if hasattr(r, "statistic") else r[0]
    pval = r.pvalue if hasattr(r, "pvalue") else r[1]
    return float(stat), float(pval)


# ---------------------------------------------------------------------------
# Bins
# ---------------------------------------------------------------------------

def _bin_bounds(label):
    """(start, end) integers from a bin label like '700-749' or '-390--341'
    (handles negative/BC years, where a naive split on '-' fails)."""
    m = re.match(r"(-?\d+)-(-?\d+)$", label)
    return int(m.group(1)), int(m.group(2))


def get_bins(first_year, last_year, bin_floor, slide_period, floor_plus1=False):
    """Overlapping segment bins matching dplR's corr.rwl.seg: first bin floored to
    bin_floor, segments of `slide_period` years overlapping by half, last bin
    ending no later than last_year."""
    seg_lag = slide_period // 2
    if bin_floor is None or bin_floor == 0:
        min_bin = first_year
    elif floor_plus1:
        min_bin = int(np.ceil((first_year - 1) / bin_floor)) * bin_floor + 1
    else:
        min_bin = int(np.ceil(first_year / bin_floor)) * bin_floor
    max_bin = last_year - slide_period + 1
    bins = []
    bin_data = {}
    i = min_bin
    while i <= max_bin:
        period = str(i) + "-" + str(i + slide_period - 1)
        bins.append(period)
        bin_data[period] = []
        i += seg_lag
    return bins, bin_data


# ---------------------------------------------------------------------------
# Normalisation / preparation (shared)
# ---------------------------------------------------------------------------

def normalize_for_crossdating(data: pd.DataFrame, prewhiten=True, ar_max=None,
                              first_aic_min=False, backcast=False, method="yw",
                              stabilize_period=None, zscore=False) -> pd.DataFrame:
    """Divide each series by its own mean (dplR's normalize1 with n=NULL, i.e.
    dplPy's 'horizontal' detrend) and, optionally, prewhiten it keeping the series
    length. Returns a year-indexed dataframe. Shared by series_corr() and
    interseries_corr().

    The dplR-faithful defaults (``method="yw"``, no variance stabilization, no
    z-score) reproduce dplR's ``corr.rwl.seg``. The COFECHA preset instead passes
    ``method="burg"`` (Burg AR via :func:`_ar_burg_prewhiten`),
    ``stabilize_period`` (COFECHA's spline variance stabilization, applied before
    AR) and ``zscore=True`` (each series to mean 0 / SD 1 before the master is
    built, COFECHA's normalize step). ``ar_max`` overrides the AR order ceiling,
    ``first_aic_min`` selects the ARSTAN first-local-minimum rule, and ``backcast``
    (YW path only) keeps full series length via ARSTAN's bckcst (see
    :func:`_ar_yw_prewhiten`)."""
    rwi_data = detrend(data, fit="horizontal", plot=False)
    if isinstance(rwi_data, (ValueError, TypeError)):
        raise rwi_data

    to_concat = [pd.DataFrame(index=pd.Index(data.index))]
    for series in rwi_data:
        col = rwi_data[series].dropna()
        vals = col.to_numpy()
        if stabilize_period and len(vals) >= 8:
            from .smoothingspline import variance_stabilize_spline
            vals = variance_stabilize_spline(vals, period=stabilize_period)
        if prewhiten and len(col) > 3:
            if method == "burg":
                vals = _ar_burg_prewhiten(vals, ar_max=(ar_max or 10),
                                          first_aic_min=first_aic_min)
            else:
                vals = _ar_yw_prewhiten(vals, ar_max=ar_max,
                                        first_aic_min=first_aic_min, backcast=backcast)
        if zscore:
            sd = np.nanstd(vals)
            vals = (vals - np.nanmean(vals)) / sd if sd > 0 else vals - np.nanmean(vals)
        to_concat.append(pd.Series(data=vals, name=series, index=col.index))
    ready = pd.concat(to_concat, axis=1)
    ready = ready.rename_axis(data.index.name)
    return ready


# ---------------------------------------------------------------------------
# Main crossdating
# ---------------------------------------------------------------------------

def xdate(data: pd.DataFrame, prewhiten=True, corr="spearman", slide_period=50,
          bin_floor=100, p_val=0.05, biweight=True, lag=10, show_flags=True,
          make_plot=False, preset=None, seg_lag=None, ar_max=None, absent=None):
    """Crossdate a set of ring-width series against a leave-one-out master.

    The segment correlations mirror dplR's corr.rwl.seg(): each series is
    normalized (divided by its mean), optionally Yule-Walker prewhitened, and
    correlated against a biweight master built from all the *other* series, over
    segments of ``slide_period`` years that overlap by half; the **A** flag
    reproduces dplR exactly -- a segment is flagged A when it is not significant
    (one-tailed p >= p_val).

    The **B** flag is *not* from dplR (corr.rwl.seg has no lag flag): it is a
    COFECHA-derived dating-shift screen, flagging a segment when it correlates
    better with the master at a non-dated lag (best_lag != 0, no margin), using
    COFECHA's SLSG convention of sliding the master past the fixed segment. The
    same B rule is used in ``preset="COFECHA"``. The per-segment lag table is
    printed for flagged segments.

    Parameters
    ----------
    data : pandas.DataFrame
        ring-width series (typically detrended RWI from dpl.detrend()).
    prewhiten : bool, default True
        AR-prewhiten each series (Yule-Walker, matching dplR).
    corr : {'spearman','pearson','kendall'}, default 'spearman'
        correlation method (case-insensitive).
    slide_period : int, default 50
        segment length in years.
    bin_floor : int, default 100
        the first segment is floored to a multiple of this.
    p_val : float, default 0.05
        significance level for the segment flag.
    biweight : bool, default True
        build the master with a Tukey biweight robust mean (else arithmetic).
    lag : int, default 10
        maximum +/- lag examined for the lag (B) flag / COFECHA table.
    show_flags : bool, default True
        print the flag summary and lag tables.
    make_plot : bool, default False
        draw the segment-correlation plot.
    preset : str or None, default None
        set to ``"COFECHA"`` to emulate the COFECHA program instead of dplR's
        ``corr.rwl.seg``. This overrides the transform, correlation, master and
        segmentation machinery to match COFECHA's FORTRAN: Pearson correlation, an
        arithmetic leave-one-out master of z-scored series, 50-yr segments on a
        25-yr grid anchored to each series' first and last year, a critical value
        derived from the 99%% one-tailed t rather than ``p_val``, COFECHA's spline
        variance stabilization before AR, Cook/Krusic Burg AR prewhitening
        (ceiling 10, first-local-AIC-minimum, N>=8), and a lag search that slides
        the master against the fixed dated segment (COFECHA's SLSG). Returns the
        extra keys ``segments`` and ``n_problems``. ``corr``, ``biweight``,
        ``bin_floor`` and ``p_val`` are ignored in this mode.
    seg_lag : int or None, default None
        segment step in years (segment overlap). ``None`` uses
        ``slide_period // 2`` (COFECHA's 50%% overlap). Only used by the preset.
    absent : pandas.DataFrame or None, default None
        COFECHA preset only -- its "omit absent rings" option (QAC=Y). A boolean
        DataFrame (years x series) marking absent rings, or a raw ring-width
        DataFrame whose zeros mark them. Absent years are dropped from that
        series' segment correlations. Ring widths are non-zero after detrending,
        so pass a *pre-detrend* source, e.g. ``absent=raw_rwl == 0`` (or just the
        raw frame). ``None`` disables the omission.
    ar_max : int or None, default None
        AR-order ceiling for prewhitening. ``None`` is dplR's floor(10*log10(n));
        the COFECHA preset defaults it to 10 (Cook/Krusic's ARSTAN ceiling) and
        selects the order by the first-local-AIC-minimum rule.

    Returns
    -------
    dict with keys:
      ``seg_corr``     DataFrame (series x bins) of segment correlations
      ``p_val``        DataFrame (series x bins) of one-tailed p-values
      ``overall``      DataFrame (series x ['rho','p_val'])
      ``avg_seg_corr`` Series (bins) mean correlation across series
      ``flags``        dict {series: {'A': [...], 'B': [...]}}
      ``bins``         list of "start-end" bin labels
      ``rwi``          DataFrame of the normalized/prewhitened series used
    and, for ``preset="COFECHA"`` only:
      ``segments``     dict {series: [{lo,hi,r0,best_lag,best_corr,n,crit,flag,lags}]}
      ``n_problems``   int, COFECHA's "Segments, possible problems" count

    Examples
    --------
    >>> rwi = dpl.detrend(ca533, fit="spline", plot=False)
    >>> res = dpl.xdate(rwi, corr="spearman", slide_period=50, bin_floor=100)
    >>> # COFECHA emulation (32-yr spline detrend, then the preset):
    >>> rwi = dpl.detrend(rwl, fit="Spline", period=32, plot=False)
    >>> res = dpl.xdate(rwi, preset="COFECHA")
    >>> res["n_problems"]                      # COFECHA "possible problems" count

    References
    ----------
    .. [1] https:/opendendro.org/dplpy-man/#xdate
    """
    _require_dataframe(data)

    # COFECHA emulation: fixed 50-yr segments on a 25-yr grid, anchored to each
    # series' own first/last year, correlated (Pearson) against an arithmetic
    # leave-one-out master, and flagged against COFECHA's derived critical value.
    if preset is not None and str(preset).strip().lower() == "cofecha":
        return _xdate_cofecha(data, prewhiten=prewhiten, slide_period=slide_period,
                              seg_lag=seg_lag, lag=lag, ar_max=ar_max, absent=absent,
                              show_flags=show_flags, make_plot=make_plot)

    method = _normalize_corr(corr)

    # normalize + prewhiten, then work on a dense (years x series) matrix on a
    # consecutive-year grid (like dplR), so the leave-one-out master is a single
    # vectorized robust mean rather than a per-series chronology rebuild.
    ready = normalize_for_crossdating(data, prewhiten)
    ready, years, first_year, last_year = dense_year_grid(ready)
    series_names = list(ready.columns)
    M = ready.to_numpy(dtype=float)                 # (nyears, nseries), NaN gaps
    nyears, nseries = M.shape
    good = np.array([np.sum(~np.isnan(M[:, i])) > 3 for i in range(nseries)])

    bins, _ = get_bins(first_year, last_year, bin_floor, slide_period)
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
            if mask.sum() != slide_period or np.isnan(seg).any() or np.isnan(mas).any():
                continue                              # require complete overlap (dplR)
            rho, pv = _corr_pval(seg, mas, method)
            seg_corr.loc[name, blabel] = rho
            seg_pval.loc[name, blabel] = pv
            # (A) significance flag -- independent of (B)
            if not np.isnan(pv) and pv >= p_val:
                a_flags.append(blabel)
            # (B) COFECHA-style lag flag: the segment correlates better with the
            # master at a non-dated position (best_lag != 0), no margin -- exactly
            # COFECHA's rule (SLSG: IF MXCOR != dated). This is a COFECHA-derived
            # dating-shift screen; dplR's corr.rwl.seg has no B flag of its own.
            lag_row, best_lag, best_coeff = _lag_table(series, master, years, lo, hi,
                                                       slide_period, method, lag)
            if best_lag != 0:
                b_flags.append({"segment": blabel, "best_lag": best_lag,
                                "best_corr": best_coeff, "lags": lag_row})
        if a_flags or b_flags:
            flags[name] = {"A": a_flags, "B": b_flags}

    avg_seg = seg_corr.mean(axis=0, skipna=True)

    if show_flags:
        _print_flags(flags, lag)
    if make_plot:
        _plot_crs(seg_corr, seg_pval, ready, bins, bin_bounds, p_val,
                  slide_period, slide_period // 2)

    return {"seg_corr": seg_corr, "p_val": seg_pval, "overall": overall,
            "avg_seg_corr": avg_seg, "flags": flags, "bins": bins,
            "rwi": ready}


# ---------------------------------------------------------------------------
# COFECHA emulation (preset="COFECHA")
# ---------------------------------------------------------------------------

def _cofecha_segments(y0, y1, seg_len=50, seg_lag=25):
    """COFECHA's segment layout for one series spanning [y0, y1].

    Interior segments are ``seg_len``-year windows beginning on every multiple
    of ``seg_lag`` that fits wholly inside the span (a 50%% overlap for the usual
    50/25). To that COFECHA adds a segment *anchored to the series' first year*
    (``[y0, y0+seg_len-1]``) and one *anchored to its last year*
    (``[y1-seg_len+1, y1]``) whenever those endpoints don't already fall on the
    grid -- so the very start and end of every series are always tested at full
    window length. A series shorter than one window is a single segment.
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
    Reproduces COFECHA's table exactly (n=50 -> 0.3281, n=25 -> 0.4622); shorter
    end segments therefore face a stiffer threshold, as in COFECHA."""
    df = n - 2
    if df < 1:
        return np.inf
    t = scipy.stats.t.ppf(conf, df)
    return float(t / np.sqrt(t * t + df))


def _xdate_cofecha(data, prewhiten=True, slide_period=50, seg_lag=None, lag=10,
                   ar_max=None, stabilize_period=32, absent=None, show_flags=True,
                   make_plot=False):
    """COFECHA-style crossdating (see :func:`xdate` with ``preset="COFECHA"``).

    Differs from the dplR-faithful default in exactly the ways COFECHA does:
    Pearson correlation, an *arithmetic* leave-one-out master of z-scored series,
    COFECHA's segment anchoring (:func:`_cofecha_segments`), its derived critical
    value (:func:`_cofecha_crit`) rather than a p-value threshold, COFECHA's
    spline variance stabilization before AR, and Ed Cook / Paul Krusic's
    Burg AR prewhitening (ceiling 10, first-local-AIC-minimum order, N>=8) rather
    than dplR's Yule-Walker global-minimum search. A segment is flagged **A** when
    the dated (lag-0) position is the highest correlation over -lag..+lag but still
    falls below the critical value, and **B** when some non-dated lag correlates
    higher (a possible dating error) -- the two counts that make up COFECHA's
    "Segments, possible problems". The lag search slides the master against the
    fixed dated segment, matching COFECHA's SLSG."""
    seg_len = int(slide_period)
    if seg_lag is None:
        seg_lag = seg_len // 2
    if ar_max is None:
        ar_max = 10                                     # Cook/Krusic ARSTAN ceiling

    ready = normalize_for_crossdating(data, prewhiten, ar_max=ar_max,
                                      first_aic_min=True, method="burg",
                                      stabilize_period=stabilize_period, zscore=True)
    ready, years, first_year, last_year = dense_year_grid(ready)
    series_names = list(ready.columns)
    M = ready.to_numpy(dtype=float)
    nyears, nseries = M.shape
    good = np.array([np.sum(~np.isnan(M[:, i])) > 3 for i in range(nseries)])
    yr_pos = {int(y): k for k, y in enumerate(years)}

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
        keep = good.copy()
        keep[i] = False
        master = _row_mean(M[:, keep]) if keep.any() else np.full(nyears, np.nan)

        overall.loc[name, "rho"], overall.loc[name, "p_val"] = \
            _corr_pval(col, master, "pearson")

        ab_col = absent_M[:, i] if absent_M is not None else None
        a_flags, b_flags, seg_list, used_cols = [], [], [], set()
        for (lo, hi) in _cofecha_segments(y0, y1, seg_len, seg_lag):
            lag_row, best_lag, best_coeff, r0, n0 = _cofecha_lag_table(
                col, master, yr_pos, first_year, last_year, lo, hi, seg_len, lag,
                absent=ab_col)
            if np.isnan(r0):
                continue
            _, pv = _corr_pval(col[yr_pos[lo]:yr_pos[hi] + 1],
                               master[yr_pos[lo]:yr_pos[hi] + 1], "pearson")
            crit = _cofecha_crit(n0 if n0 else seg_len)
            flag = ""
            if best_lag == 0 and r0 < crit:
                flag = "A"
                a_flags.append("%d-%d" % (lo, hi))
            elif best_lag != 0:
                flag = "B"
                b_flags.append({"segment": "%d-%d" % (lo, hi), "best_lag": best_lag,
                                "best_corr": best_coeff, "lags": lag_row})
            seg_list.append({"lo": lo, "hi": hi, "r0": r0, "best_lag": best_lag,
                             "best_corr": best_coeff, "n": n0, "crit": crit,
                             "flag": flag, "lags": lag_row})
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
    n_problems = sum(len(f["A"]) + len(f["B"]) for f in flags.values())

    if show_flags:
        _print_flags(flags, lag)
        print("Segments, possible problems: %d" % n_problems)
    if make_plot:
        bin_bounds = [_bin_bounds(b) for b in bins]
        _plot_crs(seg_corr, seg_pval, ready, bins, bin_bounds,
                  _cofecha_crit(seg_len), seg_len, seg_lag)

    return {"seg_corr": seg_corr, "p_val": seg_pval, "overall": overall,
            "avg_seg_corr": avg_seg, "flags": flags, "bins": bins,
            "rwi": ready, "segments": segments, "n_problems": n_problems,
            "preset": "COFECHA"}


def _cofecha_lag_table(series, master, yr_pos, first_year, last_year,
                       lo, hi, seg_len, lag_max, absent=None):
    """Correlate the fixed dated series segment ``[lo, hi]`` against the master
    over lags -lag_max..+lag_max (Pearson), sliding the MASTER window while the
    series segment stays put -- exactly COFECHA's SLSG (``ZSERM`` fixed,
    ``YMSMA(IA+..)`` shifted). Returns (row_strings, best_lag, best_corr, r0,
    n_at_lag0). Only full ``seg_len`` master windows lying inside the data span
    count. A positive lag means the segment matches the master shifted later.

    ``absent`` is an optional boolean array over the whole year grid (COFECHA's
    QAC=Y "omit absent rings"): years where the tested series has an absent ring
    are dropped from every correlation. The mask is aligned to the fixed segment
    (it follows the series, not the sliding master), as in COFECHA's CORRP0."""
    n_lags = 2 * lag_max + 1
    if lo not in yr_pos or hi not in yr_pos:
        return ["     "] * n_lags, 0, np.nan, np.nan, 0
    seg = series[yr_pos[lo]:yr_pos[hi] + 1]              # fixed dated segment
    if seg.shape[0] != seg_len or np.isnan(seg).all():
        return ["     "] * n_lags, 0, np.nan, np.nan, 0
    seg_absent = (absent[yr_pos[lo]:yr_pos[hi] + 1]
                  if absent is not None else np.zeros(seg_len, dtype=bool))

    row, best_lag, best_coeff, r0, n0 = [], 0, np.nan, np.nan, 0
    for shift in range(-lag_max, lag_max + 1):
        mlo, mhi = lo + shift, hi + shift               # slide the master window
        r = np.nan
        if mlo >= first_year and mhi <= last_year:
            mas = master[yr_pos[mlo]:yr_pos[mhi] + 1]
            if mas.shape[0] == seg_len:
                ok = ~np.isnan(seg) & ~np.isnan(mas) & ~seg_absent
                if ok.sum() >= 3:
                    a, b = seg[ok], mas[ok]
                    if not (np.all(a == a[0]) or np.all(b == b[0])):
                        da, db = a - a.mean(), b - b.mean()
                        den = np.sqrt(np.dot(da, da) * np.dot(db, db))
                        if den > 0:
                            r = float(np.dot(da, db) / den)
                if shift == 0:
                    r0, n0 = r, int(ok.sum())
        row.append(("{0:.2f}".format(r)).rjust(5) if not np.isnan(r) else "     ")
        if not np.isnan(r) and (np.isnan(best_coeff) or r > best_coeff):
            best_coeff, best_lag = r, int(shift)
    return row, best_lag, (best_coeff if not np.isnan(best_coeff) else np.nan), r0, n0


def _lag_table(series, master, years, lo, hi, slide_period, method, lag_max):
    """Correlate the fixed dated series segment ``[lo, hi]`` against the master at
    lags -lag_max..+lag_max, sliding the MASTER window while the segment stays put
    -- COFECHA's SLSG convention (the same one the COFECHA preset uses). Returns
    (row_strings, best_lag, best_corr). The B flag fires when best_lag != 0, i.e.
    the segment matches the master better at a non-dated position, with no margin,
    exactly as COFECHA does. Correlation uses ``method`` (the caller's choice), so
    the lag table is consistent with the segment correlations around it."""
    n_lags = 2 * lag_max + 1
    shifts = np.arange(-lag_max, lag_max + 1)
    mask0 = (years >= lo) & (years <= hi)
    seg = series[mask0]                                   # fixed dated segment
    if seg.shape[0] != slide_period or np.isnan(seg).any():
        return ["     "] * n_lags, 0, -np.inf

    # Stack the (valid, complete) shifted MASTER windows into one matrix and
    # rank/correlate them in a single vectorized pass. Sliding the master rather
    # than the series keeps the tested segment fixed (COFECHA/SLSG), so the "best
    # at another lag" question is asked exactly as COFECHA asks it.
    Wm = np.full((n_lags, slide_period), np.nan)
    valid = np.zeros(n_lags, dtype=bool)
    for k, shift in enumerate(shifts):
        m = (years >= lo + shift) & (years <= hi + shift)
        if m.sum() == slide_period:
            mw = master[m]
            if not np.isnan(mw).any():
                Wm[k] = mw
                valid[k] = True

    corrs = np.full(n_lags, np.nan)
    if valid.any():
        if method == "kendall":
            for k in np.where(valid)[0]:
                corrs[k] = _fast_corr(seg, Wm[k], "kendall")
        else:
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

    row = []
    best_lag, best_coeff = 0, np.nan
    for k, shift in enumerate(shifts):
        r = corrs[k]
        row.append(("{0:.2f}".format(r)).rjust(5) if not np.isnan(r) else "     ")
        if not np.isnan(r) and (np.isnan(best_coeff) or r > best_coeff):
            best_coeff, best_lag = r, int(shift)
    return row, best_lag, (best_coeff if not np.isnan(best_coeff) else -np.inf)


def _print_flags(flags, lag_max):
    if not flags:
        print()
        return
    header = " ".join(["{0:>+4d}".format(k) if k != 0 else "   0"
                       for k in range(-lag_max, lag_max + 1)])
    for name, fl in flags.items():
        print("Flags for", name)
        if fl["A"]:
            print("  [A] not significant:", ", ".join(fl["A"]))
        if fl["B"]:
            print("  [B] better at a lag:")
            print("      Segment       High " + header)
            for b in fl["B"]:
                lead = (b["segment"]).rjust(12) + " " + "{0:>+4d}".format(b["best_lag"])
                print("     ", lead, " ".join(b["lags"]))
        print()


# ---------------------------------------------------------------------------
# Critical correlation (kept for plotting / callers)
# ---------------------------------------------------------------------------

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

    def _thin(vals, target=10):
        if len(vals) <= target:
            return vals
        return vals[:: int(np.ceil(len(vals) / target))]

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
               slide_period=50, bin_floor=100, p_val=0.05, biweight=True):
    """dplR-style crossdating overview for a set of series (see corr.rwl.seg).

    A thin wrapper: crossdates ``data`` with :func:`xdate` (same parameters) and
    draws the green/blue/red segment plot -- green = series extent, blue = a
    segment that correlates significantly with the master, red = a flagged
    segment (p >= ``p_val``). Returns the matplotlib Axes.
    """
    res = xdate(data, prewhiten=prewhiten, corr=corr, slide_period=slide_period,
                bin_floor=bin_floor, p_val=p_val, biweight=biweight,
                show_flags=False, make_plot=False)
    bin_bounds = [_bin_bounds(b) for b in res["bins"]]
    return _plot_crs(res["seg_corr"], res["p_val"], res["rwi"], res["bins"],
                     bin_bounds, p_val, slide_period, slide_period // 2)
