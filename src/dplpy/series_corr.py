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
                    _corr_pval, get_bins, _CORR_ALIASES)

from math import ceil
import pandas as pd
from ._validate import _require_dataframe
import numpy as np
import matplotlib.pyplot as plt

# RColorBrewer-ish stem colours dplR's ccf plot uses: positive vs negative r.
_CCF_POS = ("darkred", "lightsalmon")     # (stem/edge, dot fill)
_CCF_NEG = ("darkblue", "lightblue")


def _dplR_ccf(x, y, lag_max):
    """R's ``ccf(x, y)``: standardised cross-covariance, fixed means, /n.

    Returns the correlation at lags -lag_max..+lag_max, where the value at lag
    k estimates cor(x[t+k], y[t]). Reproduces stats::ccf to ~1e-10.
    """
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    x = x - x.mean(); y = y - y.mean()
    n = len(x)
    denom = np.sqrt(np.mean(x ** 2) * np.mean(y ** 2))
    out = []
    for k in range(-lag_max, lag_max + 1):
        c = (np.sum(x[k:] * y[:n - k]) if k >= 0 else np.sum(x[:n + k] * y[-k:])) / n
        out.append(c / denom)
    return np.array(out)


def _ccf_bins(series_years, bin_floor, seg_length, floor_plus1=False):
    """dplR ccf.series.rwl bin layout: min.bin from the series' own first year,
    last bin ends at the series' last year (mirrors its min.bin/`to` formula)."""
    seg_lag = seg_length // 2
    smin = int(np.min(series_years)); smax = int(np.max(series_years))
    if not bin_floor:
        min_bin = smin
    elif floor_plus1:
        min_bin = ceil((smin - 1) / bin_floor) * bin_floor + 1
    else:
        min_bin = ceil(smin / bin_floor) * bin_floor
    to = smax - seg_length - seg_lag + 1
    if min_bin > to:
        return []
    return [(s, s + seg_length - 1) for s in range(min_bin, to + seg_lag + 1, seg_lag)]


def series_corr(data: pd.DataFrame, series_name: str, prewhiten=True,
                corr="spearman", seg_length=50, bin_floor=100, p_val=0.05,
                biweight=True, lag=5, make_plot=True, series_x=True,
                which="both"):
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
        master switch for drawing any figure.
    which : {'both', 'moving', 'ccf'}, default 'both'
        when ``make_plot`` is True, which figure(s) to draw: both, only the
        moving-correlation plot (``corr.series.seg``), or only the per-segment
        ccf panels (``ccf.series.rwl``).
    series_x : bool, default True
        lag convention for the dplR-style ``ccf``. True calls
        ``ccf(x=series, y=master)`` so a *positive* lag marks a missing ring in
        the series (Bunn's intuitive convention); False matches dplR's stock
        ``series.x=FALSE`` (negative lag = missing ring).

    Returns
    -------
    dict with keys ``moving_corr`` (Series), ``seg_corr`` (Series over bins),
    ``overall`` ((rho, p_val)), ``lag_table`` (Spearman rank correlations at
    shifted windows, lags x bins), ``ccf`` (dplR-style Pearson cross-correlation
    per segment, lags x bins), ``ccf_bins`` and ``bins``.

    References
    ----------
    .. [1] https:/opendendro.org/dplpy-man/#series_corr
    """
    _require_dataframe(data)
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

    # dplR-style cross-correlation table (ccf.series.rwl): Pearson ccf per
    # segment. ``series_x`` picks the lag convention -- True (default) calls
    # ccf(x=series, y=master) so a *positive* lag marks a missing ring in the
    # series (the intuitive convention Bunn adopts); False matches dplR's stock
    # series.x=FALSE default (negative lag = missing ring).
    valid = ~np.isnan(series) & ~np.isnan(master)
    ccf_table = pd.DataFrame(index=["lag." + str(k) for k in lag_vec], dtype=float)
    ccf_bin_labels = []
    if valid.any():
        for lo, hi in _ccf_bins(years[valid], bin_floor, seg_length):
            blabel = str(lo) + "-" + str(hi)
            ccf_bin_labels.append(blabel)
            m = (years >= lo) & (years <= hi)
            if m.sum() != seg_length or np.isnan(series[m]).any() or np.isnan(master[m]).any():
                ccf_table[blabel] = np.nan
                continue
            xx, yy = (series[m], master[m]) if series_x else (master[m], series[m])
            ccf_table[blabel] = _dplR_ccf(xx, yy, lag)
    note = ("NB: with series_x=True, positive lags indicate missing rings in the series"
            if series_x else
            "NB: with series_x=False, negative lags indicate missing rings in the series")

    if make_plot:
        if which not in ("both", "moving", "ccf"):
            raise ValueError("which must be 'both', 'moving' or 'ccf'")
        if which in ("both", "moving"):
            _plot_moving(series_name, moving_corr, seg_corr, bin_bounds, seg_length, p_val)
        if which in ("both", "ccf"):
            print(note)                       # the lag convention applies to the ccf plot
            _plot_ccf(series_name, ccf_table, lag_vec, ccf_bin_labels, seg_length, p_val, note)
        plt.show()

    return {"moving_corr": moving_corr, "seg_corr": seg_corr, "overall": overall,
            "lag_table": lag_table, "ccf": ccf_table, "ccf_bins": ccf_bin_labels,
            "bins": bins}


def _plot_moving(name, moving_corr, seg_corr, bin_bounds, seg_length, p_val):
    """Section 5 -- moving correlation + per-segment bars (dplR corr.series.seg).

    Base-R look: black moving line, black per-segment bars, a dashed
    significance line, and offset bottom/top year axes at the bin boundaries.
    """
    sig = scipy_norm_ppf(1 - p_val / 2) / np.sqrt(seg_length)
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

    def _thin(vals, target=11):
        return vals if len(vals) <= target else vals[:: int(np.ceil(len(vals) / target))]

    bot_ticks = _thin([lo for lo, hi in active[0::2]] +
                      ([active[-1][1] + 1] if active else []))
    top_ticks = _thin([lo for lo, hi in active[1::2]])

    fig, ax = plt.subplots(figsize=(max((xhi - xlo) / 45, 9), 5))
    ax.set_facecolor("white")
    for lo, hi in active:
        ax.axvline(lo, color="grey", lw=0.4, ls=":", zorder=1)
    if len(moving_corr):
        ax.plot(moving_corr.index.to_numpy(), moving_corr.to_numpy(),
                color="black", lw=1.5, zorder=3)
    for (lo, hi), blabel in zip(bin_bounds, seg_corr.index):
        v = seg_corr.get(blabel, np.nan)
        if not np.isnan(v):
            ax.plot([lo, hi], [v, v], color="black", lw=3, zorder=4)
    ax.axhline(sig, ls="--", color="black", lw=1.2, zorder=2)

    ax.set_xlim(xlo - (xhi - xlo) * 0.02, xhi + (xhi - xlo) * 0.02)
    ax.set_xticks(bot_ticks)
    ax.tick_params(axis="both", labelsize=13, length=4, color="black")
    axt = ax.secondary_xaxis("top")
    axt.set_xticks(top_ticks)
    axt.tick_params(axis="x", labelsize=13, length=4, color="black")
    ax.set_xlabel("Year", fontsize=15)
    ax.set_ylabel("Correlation", fontsize=15)
    ax.set_title(name, fontsize=15)
    fig.text(0.5, 0.005, "Segments: length=%d, lag=%d" % (seg_length, seg_length // 2),
             ha="center", fontsize=12)
    for spine in ax.spines.values():
        spine.set_color("black")
    fig.tight_layout(rect=(0, 0.02, 1, 1))


def _plot_ccf(name, ccf_table, lag_vec, bin_labels, seg_length, p_val, note):
    """Section 6 -- per-segment cross-correlation stems (dplR ccf.series.rwl).

    One panel per segment; stems and dots coloured dark red (positive r) or
    dark blue (negative r); grey gridlines, bold zero axes, and both +/- sig
    lines, mirroring dplR's lattice ccf plot.
    """
    segs = [b for b in bin_labels if not ccf_table[b].isna().all()]
    if not segs:
        return
    sig = scipy_norm_ppf(1 - p_val / 2) / np.sqrt(seg_length)
    lo = min(-0.5, float(np.nanmin(ccf_table[segs].to_numpy())) * 1.1, -sig * 1.1)
    hi = max(1.0, float(np.nanmax(ccf_table[segs].to_numpy())) * 1.1, sig * 1.1)

    cols = 5
    rows = (len(segs) + cols - 1) // cols
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 2.7 * rows),
                             squeeze=False)
    for j, blabel in enumerate(segs):
        ax = axes[j // cols][j % cols]
        ax.set_facecolor("white")
        vals = ccf_table[blabel].to_numpy()
        for gy in np.arange(-1, 1.0001, 0.1):
            ax.axhline(gy, color="lightgrey", lw=0.5, zorder=0)
        for lg in lag_vec:
            ax.axvline(lg, color="lightgrey", lw=0.5, zorder=0)
        ax.axhline(0, color="black", lw=1.3, zorder=2)
        ax.axvline(0, color="black", lw=1.3, zorder=2)
        ax.axhline(sig, ls="--", color="black", lw=1.1, zorder=2)
        ax.axhline(-sig, ls="--", color="black", lw=1.1, zorder=2)
        for lg, v in zip(lag_vec, vals):
            if np.isnan(v):
                continue
            stem_c, dot_c = _CCF_POS if v > 0 else _CCF_NEG
            ax.plot([lg, lg], [0, v], color=stem_c, lw=2, zorder=3)
            ax.plot([lg], [v], marker="o", ms=6, markerfacecolor=dot_c,
                    markeredgecolor=stem_c, zorder=4)
        ax.set_title(blabel, fontsize=11)
        ax.set_xlim(min(lag_vec) - 0.5, max(lag_vec) + 0.5)
        ax.set_ylim(lo, hi)
        ax.set_xticks(lag_vec[::2])
        ax.tick_params(labelsize=10)
        if j % cols == 0:
            ax.set_ylabel("Correlation", fontsize=12)
        if j // cols == rows - 1:
            ax.set_xlabel("Lag", fontsize=12)
    for j in range(len(segs), rows * cols):
        axes[j // cols][j % cols].set_axis_off()
    fig.suptitle(name, fontsize=15, y=1.0)
    fig.text(0.5, 0.002, note, ha="center", fontsize=11)
    fig.tight_layout(rect=(0, 0.02, 1, 0.99))


def scipy_norm_ppf(q):
    import scipy.stats
    return scipy.stats.norm.ppf(q)
