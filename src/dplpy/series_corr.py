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
#   table drawn as stem plots). Lag convention, as everywhere in dplPy and in
#   COFECHA: a NEGATIVE lag means a MISSING ring in the series.
#
# >>> import dplpy as dpl
# >>> data = dpl.readers("../tests/data/csv/file.csv")
# >>> dpl.series_corr(data, "CAM011")

from .xdate import (normalize_for_crossdating, _row_biweight, _row_mean, _bin_bounds,
                    _corr_pval, _fast_corr, _lag_table, get_bins, dense_year_grid,
                    _thin, _carry_attrs)

import pandas as pd
from ._validate import (_require_dataframe, _normalize_corr, _plot_kwarg, _alias_kwarg,
                        _normalize_choice, _require_year_index)
import numpy as np
import matplotlib.pyplot as plt
from ._plot_style import style_axes, finalize_font, ACCENT, ACCENT_WARM

# Sign-coded stem colours (house palette): positive r warm, negative r blue.
# The red/blue distinction is semantic (sign of correlation), kept from dplR's
# ccf plot but recoloured to match the rest of dplPy's figures.
_CCF_POS = (ACCENT_WARM, "#e8b0a6")       # (stem/edge, dot fill) -- positive r
_CCF_NEG = (ACCENT, "#adc3db")            #                          negative r


def _dplR_ccf(x, y, lag):
    """R's ``ccf(x, y)``: standardised cross-covariance, fixed means, /n.

    Returns the correlation at lags -lag..+lag, where the value at lag k
    estimates cor(x[t+k], y[t]). Follows R's stats::ccf definition.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x - x.mean()
    y = y - y.mean()
    n = len(x)
    denom = np.sqrt(np.mean(x ** 2) * np.mean(y ** 2))
    out = []
    for k in range(-lag, lag + 1):
        c = (np.sum(x[k:] * y[:n - k]) if k >= 0 else np.sum(x[:n + k] * y[-k:])) / n
        out.append(c / denom)
    return np.array(out)


def _ccf_bins(series_years, bin_floor, window_length):
    """dplR ccf.series.rwl bin layout: bins floored from the SERIES' own first
    year, the last ending no later than its last year. The same layout rule as
    `xdate.get_bins()` applied to the series' span; returns ``(lo, hi)`` pairs."""
    if len(series_years) == 0:
        return []
    bins, _ = get_bins(int(np.min(series_years)), int(np.max(series_years)),
                       bin_floor, window_length)
    return [_bin_bounds(b) for b in bins]


def series_corr(data: pd.DataFrame, series_name: str, prewhiten=True,
                corr="spearman", window_length=50, bin_floor=100, p_val=0.05,
                biweight=True, lag=10, plot=True, series_x=False,
                which="both", ar_max=10, first_aic_min=True, **kwargs):
    """Crossdate one series against the master built from all the others.

    Produces (and, by default, plots) a moving correlation of the series against
    the leave-one-out master, the per-segment correlations, and a per-segment
    lag table (COFECHA-style) showing how the correlation changes when the
    master is shifted +/- ``lag`` years past the dated segment -- an off-zero
    peak suggests a dating error in that segment.

    Lag convention (``lag_table`` and, by default, ``ccf``): the dated segment
    stays fixed and the MASTER slides, so a **negative lag means a missing ring
    in the series**, as in COFECHA, ``dpl.xdate`` and ``dpl.check``. The
    ``lag_table`` is `xdate._lag_table()`, the same scan behind xdate's B flags.

    Parameters
    ----------
    data : pandas.DataFrame
        ring-width series (typically detrended RWI).
    series_name : str
        the series to examine.
    prewhiten, corr, window_length, bin_floor, p_val, biweight, lag, ar_max, first_aic_min
        as in `dpl.xdate()` (``window_length`` is the segment length;
        ``seg_length=`` is a deprecated alias).
    plot : bool, default True
        master switch for drawing any figure.
    series_x : bool, default False
        lag convention for the dplR-style ``ccf`` only (``lag_table`` always uses
        the COFECHA convention). False (dplR's stock ``series.x=FALSE``) calls
        ``ccf(x=master, y=series)`` so a *negative* lag marks a missing ring in the
        series, consistent with ``lag_table``. True calls ``ccf(x=series,
        y=master)``, flipping the sign so a *positive* lag marks a missing ring.
    which : {'both', 'moving', 'ccf'}, default 'both'
        when ``plot`` is True, which figure(s) to draw: both, only the
        moving-correlation plot (``corr.series.seg``), or only the per-segment
        ccf panels (``ccf.series.rwl``). Case-insensitive.

    Returns
    -------
    dict with keys ``moving_corr`` (Series), ``seg_corr`` (Series over bins),
    ``overall`` (DataFrame, one row for the series, columns ``rho`` and
    ``p_val`` -- the same shape as ``xdate()["overall"]``), ``lag_table``
    (correlations at shifted master windows, lags x bins, using ``corr``),
    ``ccf`` (dplR-style Pearson cross-correlation per segment, lags x bins),
    ``ccf_bins``, ``bins`` and ``rwi`` (the transformed series used).

    References
    ----------
    .. [1] https://opendendro.github.io/dplPy/reference/crossdating/#series_corr
    """
    window_length = _alias_kwarg(kwargs, "seg_length", "window_length",
                                 window_length, "series_corr")
    plot = _plot_kwarg(plot, kwargs)
    _require_dataframe(data)
    if not isinstance(series_name, str):
        raise TypeError("Expected string input as series name, got " + str(type(series_name)) + " instead.")
    if series_name not in data.columns:
        raise ValueError("Series named " + series_name + " not found in provided dataframe.")
    data = _require_year_index(data, "data")
    method = _normalize_corr(corr)
    which = _normalize_choice(which, ("both", "moving", "ccf"), "which")
    window_length = int(window_length)
    lag = int(lag)

    ready = normalize_for_crossdating(data, prewhiten, ar_max=ar_max,
                                      first_aic_min=first_aic_min)
    ready, years, first_year, last_year = dense_year_grid(ready)
    ready = _carry_attrs(data, ready, kind="rwi")

    names = list(ready.columns)
    idx = names.index(series_name)
    M = ready.to_numpy(dtype=float)
    good = np.array([np.sum(~np.isnan(M[:, i])) > 3 for i in range(M.shape[1])])
    keep = good.copy()
    keep[idx] = False
    master = (_row_biweight if biweight else _row_mean)(M[:, keep])
    series = M[:, idx]

    rho, pv = _corr_pval(series, master, method)
    overall = pd.DataFrame({"rho": [rho], "p_val": [pv]}, index=pd.Index([series_name]))

    bins, _ = get_bins(first_year, last_year, bin_floor, window_length)
    bin_bounds = [_bin_bounds(b) for b in bins]

    # per-segment correlation and the COFECHA-convention lag table (one
    # implementation, shared with xdate: the master slides, the segment is fixed)
    seg_corr = pd.Series(index=bins, dtype=float)
    lag_vec = list(range(-lag, lag + 1))
    lag_index = ["lag." + str(k) for k in lag_vec]
    lag_table = pd.DataFrame(index=lag_index, columns=bins, dtype=float)
    for (lo, hi), blabel in zip(bin_bounds, bins):
        mask0 = (years >= lo) & (years <= hi)
        if mask0.sum() != window_length or np.isnan(series[mask0]).any() or np.isnan(master[mask0]).any():
            continue
        seg_corr[blabel] = _fast_corr(series[mask0], master[mask0], method)
        corrs, _, _ = _lag_table(series, master, years, lo, hi, window_length, method, lag)
        lag_table[blabel] = corrs

    # moving correlation (window window_length, step 1, stored at window centre)
    seg_lag = window_length // 2
    mov_years, mov_corr = [], []
    for t in range(0, len(years) - window_length + 1):
        s = series[t:t + window_length]
        m = master[t:t + window_length]
        if np.isnan(s).any() or np.isnan(m).any():
            continue
        mov_years.append(years[t] + seg_lag)
        mov_corr.append(_fast_corr(s, m, method))
    moving_corr = pd.Series(data=mov_corr, index=mov_years, dtype=float)

    # dplR-style cross-correlation table (ccf.series.rwl): Pearson ccf per
    # segment. ``series_x`` picks the lag convention -- False (default, dplR's
    # stock series.x=FALSE and the COFECHA convention) calls ccf(x=master,
    # y=series) so a *negative* lag marks a missing ring in the series; True calls
    # ccf(x=series, y=master), where a *positive* lag marks a missing ring.
    valid = ~np.isnan(series) & ~np.isnan(master)
    ccf_table = pd.DataFrame(index=lag_index, dtype=float)
    ccf_bin_labels = []
    if valid.any():
        for lo, hi in _ccf_bins(years[valid], bin_floor, window_length):
            blabel = str(lo) + "-" + str(hi)
            ccf_bin_labels.append(blabel)
            m = (years >= lo) & (years <= hi)
            if m.sum() != window_length or np.isnan(series[m]).any() or np.isnan(master[m]).any():
                ccf_table[blabel] = np.nan
                continue
            xx, yy = (series[m], master[m]) if series_x else (master[m], series[m])
            ccf_table[blabel] = _dplR_ccf(xx, yy, lag)
    note = ("NB: with series_x=True, POSITIVE lags indicate missing rings in the series "
            "(the opposite of the lag_table / COFECHA convention)"
            if series_x else
            "NB: negative lags indicate missing rings in the series (as in COFECHA)")

    if plot:
        if which in ("both", "moving"):
            _plot_moving(series_name, moving_corr, seg_corr, bin_bounds, window_length, p_val)
        if which in ("both", "ccf"):
            print(note)                       # the lag convention applies to the ccf plot
            _plot_ccf(series_name, ccf_table, lag_vec, ccf_bin_labels, window_length, p_val, note)
        plt.show()

    return {"moving_corr": moving_corr, "seg_corr": seg_corr, "overall": overall,
            "lag_table": lag_table, "ccf": ccf_table, "ccf_bins": ccf_bin_labels,
            "bins": bins, "rwi": ready}


def _plot_moving(name, moving_corr, seg_corr, bin_bounds, window_length, p_val):
    """Section 5 -- moving correlation + per-segment bars (dplR corr.series.seg).

    Base-R look: black moving line, black per-segment bars, a dashed
    significance line, and offset bottom/top year axes at the bin boundaries.
    """
    sig = scipy_norm_ppf(1 - p_val / 2) / np.sqrt(window_length)
    # Restrict to the segments the series actually spans (dplR limits the plot
    # to the series' own range rather than the whole collection).
    active = [(lo, hi) for (lo, hi), b in zip(bin_bounds, seg_corr.index)
              if not np.isnan(seg_corr.get(b, np.nan))]
    if active:
        xlo, xhi = active[0][0], active[-1][1] + 1
    elif len(moving_corr):
        xlo, xhi = float(moving_corr.index.min()), float(moving_corr.index.max())
    else:
        xlo, xhi = 0, 1

    bot_ticks = _thin([lo for lo, hi in active[0::2]] +
                      ([active[-1][1] + 1] if active else []), target=11)
    top_ticks = _thin([lo for lo, hi in active[1::2]], target=11)

    fig, ax = plt.subplots(figsize=(max((xhi - xlo) / 45, 9), 5))
    for lo, hi in active:
        ax.axvline(lo, color="0.85", lw=0.4, ls=":", zorder=1)
    if len(moving_corr):
        ax.plot(moving_corr.index.to_numpy(), moving_corr.to_numpy(),
                color=ACCENT, lw=1.3, zorder=3)
    for (lo, hi), blabel in zip(bin_bounds, seg_corr.index):
        v = seg_corr.get(blabel, np.nan)
        if not np.isnan(v):
            ax.plot([lo, hi], [v, v], color="0.25", lw=3, zorder=4)
    ax.axhline(sig, ls="--", color="0.55", lw=1.1, zorder=2)

    ax.set_xlim(xlo - (xhi - xlo) * 0.02, xhi + (xhi - xlo) * 0.02)
    ax.set_xticks(bot_ticks)
    ax.tick_params(axis="both", length=4)
    axt = ax.secondary_xaxis("top")
    axt.set_xticks(top_ticks)
    axt.tick_params(axis="x", length=4)
    ax.set_xlabel("Year")
    ax.set_ylabel("Correlation")
    ax.set_title(name, fontsize=11)
    fig.text(0.5, 0.005, "Segments: length=%d, lag=%d" % (window_length, window_length // 2),
             ha="center", fontsize=9, color="0.4")
    # keep the top spine (the secondary year axis lives there); trim the right
    style_axes(ax, xgrid=False, hide_spines=("right",))
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    finalize_font(fig)


def _plot_ccf(name, ccf_table, lag_vec, bin_labels, window_length, p_val, note):
    """Section 6 -- per-segment cross-correlation stems (dplR ccf.series.rwl).

    One panel per segment; stems and dots coloured dark red (positive r) or
    dark blue (negative r); grey gridlines, bold zero axes, and both +/- sig
    lines, mirroring dplR's lattice ccf plot.
    """
    segs = [b for b in bin_labels if not ccf_table[b].isna().all()]
    if not segs:
        return
    sig = scipy_norm_ppf(1 - p_val / 2) / np.sqrt(window_length)
    lo = min(-0.5, float(np.nanmin(ccf_table[segs].to_numpy())) * 1.1, -sig * 1.1)
    hi = max(1.0, float(np.nanmax(ccf_table[segs].to_numpy())) * 1.1, sig * 1.1)

    cols = 5
    rows = (len(segs) + cols - 1) // cols
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 2.7 * rows),
                             squeeze=False)
    for j, blabel in enumerate(segs):
        ax = axes[j // cols][j % cols]
        vals = ccf_table[blabel].to_numpy()
        for gy in np.arange(-1, 1.0001, 0.1):
            ax.axhline(gy, color="0.9", lw=0.5, zorder=0)
        for lg in lag_vec:
            ax.axvline(lg, color="0.9", lw=0.5, zorder=0)
        ax.axhline(0, color="0.3", lw=1.1, zorder=2)
        ax.axvline(0, color="0.3", lw=1.1, zorder=2)
        ax.axhline(sig, ls="--", color="0.55", lw=1.0, zorder=2)
        ax.axhline(-sig, ls="--", color="0.55", lw=1.0, zorder=2)
        for lg, v in zip(lag_vec, vals):
            if np.isnan(v):
                continue
            stem_c, dot_c = _CCF_POS if v > 0 else _CCF_NEG
            ax.plot([lg, lg], [0, v], color=stem_c, lw=2, zorder=3)
            ax.plot([lg], [v], marker="o", ms=6, markerfacecolor=dot_c,
                    markeredgecolor=stem_c, zorder=4)
        ax.set_title(blabel, fontsize=9)
        ax.set_xlim(min(lag_vec) - 0.5, max(lag_vec) + 0.5)
        ax.set_ylim(lo, hi)
        ax.set_xticks(lag_vec[::2])
        ax.tick_params(labelsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if j % cols == 0:
            ax.set_ylabel("Correlation", fontsize=10)
        if j // cols == rows - 1:
            ax.set_xlabel("Lag", fontsize=10)
    for j in range(len(segs), rows * cols):
        axes[j // cols][j % cols].set_axis_off()
    fig.suptitle(name, fontsize=12, y=1.0)
    fig.text(0.5, 0.002, note, ha="center", fontsize=9, color="0.4")
    fig.tight_layout(rect=(0, 0.02, 1, 0.99))
    finalize_font(fig)


def scipy_norm_ppf(q):
    import scipy.stats
    return scipy.stats.norm.ppf(q)
