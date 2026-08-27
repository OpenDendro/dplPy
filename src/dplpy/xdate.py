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

def _ar_yw_prewhiten(x):
    """Prewhiten a 1-D series with a Yule-Walker AR model, matching dplR's ar():
    AIC order selection up to floor(10*log10(n)), residuals + series mean, and
    the series length preserved (the first `order` values become NaN). Validated
    to reproduce R's ar() to ~1e-15.  `x` must be NaN-free."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return x.astype(float).copy()
    order_max = min(n - 1, int(np.floor(10 * np.log10(n))))
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
    order = int(np.argmin(aic))
    phi = coeffs_by_order[order]
    out = np.full(n, np.nan)
    if order == 0:
        out[:] = xc + xbar
        return out
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

def normalize_for_crossdating(data: pd.DataFrame, prewhiten=True) -> pd.DataFrame:
    """Divide each series by its own mean (dplR's normalize1 with n=NULL, i.e.
    dplPy's 'horizontal' detrend) and, optionally, Yule-Walker prewhiten it
    keeping the series length. Returns a year-indexed dataframe. Shared by
    series_corr() and interseries_cor()."""
    rwi_data = detrend(data, fit="horizontal", plot=False)
    if isinstance(rwi_data, (ValueError, TypeError)):
        raise rwi_data

    to_concat = [pd.DataFrame(index=pd.Index(data.index))]
    for series in rwi_data:
        col = rwi_data[series].dropna()
        if prewhiten and len(col) > 3:
            pw = _ar_yw_prewhiten(col.to_numpy())
            to_concat.append(pd.Series(data=pw, name=series, index=col.index))
        else:
            to_concat.append(col)
    ready = pd.concat(to_concat, axis=1)
    ready = ready.rename_axis(data.index.name)
    return ready


# ---------------------------------------------------------------------------
# Main crossdating
# ---------------------------------------------------------------------------

def xdate(data: pd.DataFrame, prewhiten=True, corr="spearman", slide_period=50,
          bin_floor=100, p_val=0.05, biweight=True, lag=10, show_flags=True,
          make_plot=False):
    """Crossdate a set of ring-width series against a leave-one-out master.

    Mirrors dplR's corr.rwl.seg(): each series is normalized (divided by its
    mean), optionally Yule-Walker prewhitened, and correlated against a biweight
    master built from all the *other* series, over segments of ``slide_period``
    years that overlap by half. For every segment it reports the correlation and
    its one-tailed p-value; a segment is flagged **A** if it is not significant
    (p >= p_val) and **B** if it correlates markedly better at a non-zero lag
    (a possible dating error). The per-segment lag table (COFECHA-style) is
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

    Examples
    --------
    >>> rwi = dpl.detrend(ca533, fit="spline", plot=False)
    >>> res = dpl.xdate(rwi, corr="spearman", slide_period=50, bin_floor=100)

    References
    ----------
    .. [1] https:/opendendro.org/dplpy-man/#xdate
    """
    _require_dataframe(data)
    method = _normalize_corr(corr)

    # normalize + prewhiten, then work on a dense (years x series) matrix on a
    # consecutive-year grid (like dplR), so the leave-one-out master is a single
    # vectorized robust mean rather than a per-series chronology rebuild.
    ready = normalize_for_crossdating(data, prewhiten)
    first_year = int(ready.first_valid_index())
    last_year = int(ready.last_valid_index())
    years = np.arange(first_year, last_year + 1)
    ready = ready.reindex(years)
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
            # (B) lag flag + COFECHA lag table
            lag_row, best_lag, best_coeff = _lag_table(series, master, years, lo, hi,
                                                       slide_period, method, lag)
            if best_lag != 0 and (best_coeff - rho) >= 0.08:
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


def _lag_table(series, master, years, lo, hi, slide_period, method, lag_max):
    """Correlation of a segment against the master at lags -lag_max..+lag_max
    (the COFECHA-style table). Returns (row_strings, best_lag, best_corr)."""
    n_lags = 2 * lag_max + 1
    shifts = np.arange(-lag_max, lag_max + 1)
    mask0 = (years >= lo) & (years <= hi)
    mas = master[mask0]
    if mas.shape[0] != slide_period or np.isnan(mas).any():
        return ["     "] * n_lags, 0, -np.inf

    # Stack the (valid, complete) lag windows into one matrix and rank/correlate
    # them in a single vectorized pass rather than 2*lag_max+1 separate calls.
    W = np.full((n_lags, slide_period), np.nan)
    valid = np.zeros(n_lags, dtype=bool)
    for k, shift in enumerate(shifts):
        m = (years >= lo + shift) & (years <= hi + shift)
        if m.sum() == slide_period:
            seg = series[m]
            if not np.isnan(seg).any():
                W[k] = seg
                valid[k] = True

    corrs = np.full(n_lags, np.nan)
    if valid.any():
        if method == "kendall":
            for k in np.where(valid)[0]:
                corrs[k] = _fast_corr(W[k], mas, "kendall")
        else:
            if method == "spearman":
                Wv = scipy.stats.rankdata(W[valid], axis=1)
                bv = scipy.stats.rankdata(mas)
            else:  # pearson
                Wv, bv = W[valid], mas
            b = bv - bv.mean()
            A = Wv - Wv.mean(axis=1, keepdims=True)
            den = np.sqrt((A * A).sum(axis=1) * np.dot(b, b))
            with np.errstate(invalid="ignore", divide="ignore"):
                corrs[valid] = np.where(den > 0, (A @ b) / den, np.nan)

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
