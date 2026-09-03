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

# Title: xdate_floater.py
# Description: Crossdate a *floating* (undated) ring-width series against a dated
#   reference collection, to estimate the calendar placement of the undated wood
#   (e.g. a beam from a historic building or a remnant/sub-fossil log). A port of
#   dplR's xdate.floater(), with the significance statistics dendro dating
#   practice relies on.
#
#   Method: build a master chronology from the dated `rwl` (divide each series by
#   its mean, apply a high-pass transform, then a Tukey-biweight robust row mean);
#   transform the floating series the same way; slide it across the master over
#   every offset with at least `min_overlap` overlapping rings; and at each offset
#   record the correlation r and the crossdating statistics below. The offset with
#   the highest t-value is the estimated dating.
#
#   At each offset we report:
#     * r         -- the sliding correlation (Spearman by default);
#     * t         -- the Baillie & Pilcher (1973) t statistic, t = r*sqrt(df-2) /
#                    sqrt(1-r^2), the standard crossdating measure (a value >= ~3.5
#                    is the classic acceptance threshold; see Fowler & Bridge 2017);
#     * eff_df    -- the effective degrees of freedom, reduced from the overlap n
#                    for the lag-1 autocorrelation of the two series (autocorrelated
#                    series carry less independent information; Wigley et al. 1987;
#                    Wilson 2026);
#     * p_bonf    -- the one-tailed p-value of t, Bonferroni-corrected for the many
#                    offsets tested (conservative, but honest about the multiple
#                    comparisons a sliding search makes);
#   and for the best match, an isolation factor (IF) -- how many times larger the
#   runner-up's p-value is than the best offset's; a large IF means the date stands
#   out cleanly from the alternatives (Wigley et al. 1987; Wilson 2026).
#
# References:
#   Baillie, M.G.L., Pilcher, J.R., 1973. A simple cross-dating program for
#     tree-ring research. Tree-Ring Bulletin 33, 7-14.
#   Fowler, A.M., Bridge, M.C., 2017. Empirically-determined statistical
#     significance of the Baillie and Pilcher (1973) t statistic for British Isles
#     oak. Dendrochronologia 42, 51-55.
#   Wigley, T.M.L., Jones, P.D., Briffa, K.R., 1987. Cross-dating methods in
#     dendrochronology. J. Archaeol. Sci. 14 (1), 51-64.
#   Wilson, R., 2026. Multi-parameter crossdating for sub-fossil and historical
#     samples. Dendrochronologia 96, 126485.
#
# example usage:
#   >>> import dplpy as dpl
#   >>> ref = dpl.readers("reference.rwl")        # a dated reference collection
#   >>> res = dpl.xdate_floater(ref, floating_series, series_name="beam1")
#   >>> res["best"]     # {min_year, max_year, r, t, eff_df, p_bonf, isolation_factor, n}
#   >>> res["floater_cor_stats"]                  # every offset, best (highest t) first

import numpy as np
import pandas as pd
import scipy.stats

from ._validate import _require_dataframe, _normalize_corr
from .stats import get_ar1
from .xdate import (normalize_for_crossdating, dense_year_grid,
                    _row_biweight, _row_mean, _ar_yw_prewhiten, _corr_pval)

_TRANSFORMS = ("pw", "fd", "none")


def _as_series_values(series):
    """Coerce the floating series to a 1-D float array of its (non-NaN) values."""
    if isinstance(series, pd.DataFrame):
        if series.shape[1] != 1:
            raise ValueError("`series` as a DataFrame must have exactly one "
                             "column (the floating series).")
        series = series.iloc[:, 0]
    vals = np.asarray(pd.Series(series).to_numpy(), dtype=float)
    return vals[~np.isnan(vals)]


def _build_master(data, transform, biweight):
    """Master chronology from the dated collection: mean-normalize each series,
    apply the high-pass transform, then a robust row mean. Returns (values, years)
    with NaN rows dropped."""
    ready = normalize_for_crossdating(data, prewhiten=(transform == "pw"))
    ready, years, _, _ = dense_year_grid(ready)
    M = ready.to_numpy(dtype=float)
    if transform == "fd":                       # first difference; label by later year
        M = np.diff(M, axis=0)
        years = years[1:]
    good = np.array([np.sum(~np.isnan(M[:, i])) > 3 for i in range(M.shape[1])])
    if not good.any():
        raise ValueError("no reference series had enough data to build a master.")
    row_master = _row_biweight if biweight else _row_mean
    master = row_master(M[:, good])
    ok = ~np.isnan(master)
    return master[ok], years[ok].astype(int)


def _transform_series(y_raw, transform):
    """Mean-normalize then transform the floating series; return its values."""
    y = y_raw / np.nanmean(y_raw)
    if transform == "pw" and len(y) > 3:
        y = _ar_yw_prewhiten(y)
    elif transform == "fd":
        y = np.diff(y)
    return y[~np.isnan(y)]


def _abs_ac1(values):
    """|lag-1 autocorrelation| of a 1-D array (0 if undefined)."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 3 or np.all(v == v[0]):
        return 0.0
    return abs(float(get_ar1(pd.Series(v))))


def xdate_floater(data: pd.DataFrame, series, series_name="Unknown",
                  min_overlap=50, transform="pw", prewhiten=None, biweight=True,
                  corr="spearman", make_plot=False, return_rwl=False,
                  verbose=True):
    """Estimate the calendar dating of a floating (undated) ring-width series.

    Parameters
    ----------
    data : pandas.DataFrame
        A **dated** reference collection (year-indexed ring widths), from which
        the master chronology is built.
    series : sequence or pandas.Series/DataFrame
        The floating (undated) ring-width series -- just the ring values, oldest
        to youngest; any calendar index is ignored.
    series_name : str, default "Unknown"
        A label for the floating series (used in outputs).
    min_overlap : int, default 50
        Minimum number of overlapping rings required to score an offset.
    transform : {"pw", "fd", "none"}, default "pw"
        High-pass transform applied (after mean-normalization) before correlating:
        ``"pw"`` Yule-Walker AR prewhitening (residual series, the default),
        ``"fd"`` first differencing, or ``"none"``. First differencing can sharpen
        the common high-frequency signal but sharply reduces the effective degrees
        of freedom (Wilson 2026).
    prewhiten : bool, optional
        Deprecated back-compatible alias: ``prewhiten=True`` -> ``transform="pw"``,
        ``prewhiten=False`` -> ``transform="none"``. If given, it overrides
        ``transform``.
    biweight : bool, default True
        Build the master with a Tukey biweight robust mean (else arithmetic mean).
    corr : {"spearman", "pearson", "kendall"}, default "spearman"
        Correlation used to score each offset (one-sided, alternative "greater").
    make_plot : bool, default False
        Plot the sliding t-value against the series' end year, marking the best fit.
    return_rwl : bool, default False
        Also return the floating series placed at its best-fit calendar years
        (``placed``) and combined with the reference (``combined``).
    verbose : bool, default True
        Print a short summary of the best-fit dating.

    Returns
    -------
    dict
        ``series_name``; ``floater_cor_stats`` (a DataFrame with one row per offset,
        highest t first, columns ``min_year, max_year, r, t, eff_df, p_bonf, n``);
        ``best`` (the top-scoring placement, incl. ``isolation_factor`` and
        ``one_over_p``); and, when ``return_rwl=True``, ``placed`` and ``combined``.

    Notes
    -----
    Ports dplR's ``xdate.floater`` (default ``n=NULL`` mean normalization) and adds
    the crossdating statistics of Baillie & Pilcher (1973), Wigley et al. (1987)
    and Wilson (2026): the t-value, autocorrelation-adjusted effective degrees of
    freedom, a Bonferroni-corrected p-value, and an isolation factor. The
    correlation scoring reuses dplPy's crossdating internals, so results are
    consistent with ``dpl.xdate``.
    """
    _require_dataframe(data)
    method = _normalize_corr(corr)
    if prewhiten is not None:                   # back-compat alias
        transform = "pw" if prewhiten else "none"
    if transform not in _TRANSFORMS:
        raise ValueError("transform must be one of %s, got '%s'."
                         % (str(_TRANSFORMS), transform))

    y_raw = _as_series_values(series)
    n_series = len(y_raw)                        # original (full timber) ring count
    if n_series == 0:
        raise ValueError("`series` contains no finite ring-width values.")

    x, yrs = _build_master(data, transform, biweight)
    nx = len(x)
    y = _transform_series(y_raw, transform)
    ny = len(y)

    if min_overlap > ny:
        raise ValueError("min_overlap (%d) must be <= the series length after "
                         "transform (%d)." % (min_overlap, ny))
    if min_overlap > nx:
        raise ValueError("min_overlap (%d) exceeds the reference master length "
                         "(%d)." % (min_overlap, nx))

    # lag-1 autocorrelation of each (transformed) series -> effective-df reduction
    ac1_ref, ac1_und = _abs_ac1(x), _abs_ac1(y)
    ac_factor = (1.0 - ac1_ref * ac1_und) / (1.0 + ac1_ref * ac1_und)

    if verbose:
        print("Reference master years: %d to %d (%d)"
              % (int(yrs.min()), int(yrs.max()), nx))
        print("Floating series length: %d (transform '%s' -> %d)"
              % (n_series, transform, ny))
        print("Minimum overlap for search:", min_overlap)

    # Slide the floating series across the master over every offset with at least
    # `min_overlap` overlap. Transcribes dplR's three-regime crawl; `i` is dplR's
    # 1-based offset, converted to 0-based slices here.
    rows = []
    edge = 0
    for i in range(nx + (ny - min_overlap), min_overlap - 1, -1):
        if i > nx:                              # series overhangs the young end
            xseg, yseg, xyrs = x[i - ny:nx], y[0:ny - (i - nx)], yrs[i - ny:nx]
            max_year = int(xyrs.max()) + ny - min_overlap + edge
            edge -= 1
        elif i >= ny:                           # series fully inside the master
            xseg, yseg, xyrs = x[i - ny:i], y, yrs[i - ny:i]
            max_year = int(xyrs.max())
        else:                                   # series overhangs the old end
            xseg, yseg, xyrs = x[0:i], y[ny - i:ny], yrs[0:i]
            max_year = int(xyrs.max())
        r, _ = _corr_pval(xseg, yseg, method)
        rows.append({"min_year": max_year - n_series + 1, "max_year": max_year,
                     "r": r, "n": len(xseg)})

    stats = pd.DataFrame(rows, columns=["min_year", "max_year", "r", "n"])

    # crossdating statistics (Baillie & Pilcher t; effective df adjusted for
    # autocorrelation; one-tailed p, Bonferroni-corrected for the offsets tested)
    r = stats["r"].to_numpy(dtype=float)
    eff_df = stats["n"].to_numpy(dtype=float) * ac_factor
    with np.errstate(invalid="ignore", divide="ignore"):
        t = r * np.sqrt(eff_df - 2.0) / np.sqrt(1.0 - r ** 2)
        p = scipy.stats.t.sf(t, df=eff_df)      # one-tailed p-value of t
    n_offsets = int(np.isfinite(t).sum())
    p_bonf = np.minimum(p * n_offsets, 1.0)
    stats["t"] = t
    stats["eff_df"] = eff_df
    stats["p_bonf"] = p_bonf
    stats = stats[["min_year", "max_year", "r", "t", "eff_df", "p_bonf", "n"]]
    stats = stats.sort_values("t", ascending=False,
                              na_position="last").reset_index(drop=True)

    # isolation factor: how many times larger the runner-up p is than the best p
    # (invariant to the Bonferroni constant, so computed from the raw t-p values)
    finite_p = np.sort(p[np.isfinite(p)])
    if finite_p.size >= 2 and finite_p[0] > 0:
        distinct = finite_p[finite_p > finite_p[0]]
        isolation = float(round(distinct[0] / finite_p[0])) if distinct.size else np.inf
    else:
        isolation = np.nan

    top = stats.iloc[0]
    best = {"min_year": int(top["min_year"]), "max_year": int(top["max_year"]),
            "r": float(top["r"]), "t": float(top["t"]),
            "eff_df": float(top["eff_df"]), "p_bonf": float(top["p_bonf"]),
            "one_over_p": (float("inf") if top["p_bonf"] == 0
                           else 1.0 / float(top["p_bonf"])),
            "isolation_factor": isolation, "n": int(top["n"])}

    if verbose:
        if_disp = (">1000" if np.isfinite(isolation) and isolation > 1000
                   else str(isolation))
        oop = best["one_over_p"]
        oop_disp = ">1 million" if oop > 1e6 else "%.3g" % oop
        print("\nBest match for '%s': %d to %d  (t = %.2f, r = %.3f, "
              "eff_df = %.0f, p_bonf = %.2e, 1/p = %s, IF = %s, overlap n = %d)"
              % (series_name, best["min_year"], best["max_year"], best["t"],
                 best["r"], best["eff_df"], best["p_bonf"], oop_disp,
                 if_disp, best["n"]))

    result = {"series_name": series_name, "floater_cor_stats": stats, "best": best}

    if return_rwl or make_plot:
        placed = pd.DataFrame(
            {series_name: y_raw},
            index=pd.Index(range(best["min_year"], best["min_year"] + n_series),
                           name=data.index.name or "Year"))
        result["placed"] = placed
        result["combined"] = data.join(placed, how="outer")

    if make_plot:
        _plot_floater(result)

    return result


def _plot_floater(result, show=True):
    """A four-panel dating figure after Wilson (2026): the sliding t-value and the
    sliding Bonferroni-adjusted p-value as stacked line plots against calendar
    year, and the distribution of all t-values (with the best marked) to the right.
    Returns the matplotlib Figure."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator

    stats = result["floater_cor_stats"]
    best = result["best"]
    name = result["series_name"]
    order = stats.sort_values("max_year")
    yr = order["max_year"].to_numpy(dtype=float)
    tval = order["t"].to_numpy(dtype=float)
    pval = order["p_bonf"].to_numpy(dtype=float)
    all_t = stats["t"].to_numpy(dtype=float)
    all_t = all_t[np.isfinite(all_t)]
    thr = 4.0 * np.nanstd(all_t)

    teal, blue, red, grey = "#458B74", "#1f4fd8", "#d62728", "#999999"
    # display caps, matching Wilson's convention
    IF = best["isolation_factor"]
    if_disp = ">1000" if (np.isfinite(IF) and IF > 1000) else (
        "%d" % IF if np.isfinite(IF) else "NA")
    oop = best["one_over_p"]
    oop_disp = ">1 million" if oop > 1e6 else "%.3g" % oop
    p_disp = "<0.0001" if best["p_bonf"] < 1e-4 else "%.4f" % best["p_bonf"]

    def _titles(ax, main, sub):
        ax.text(0.0, 1.16, main, transform=ax.transAxes, ha="left", va="bottom",
                fontweight="bold", fontsize=11)
        ax.text(0.0, 1.02, sub, transform=ax.transAxes, ha="left", va="bottom",
                color=blue, fontweight="bold", fontsize=9.5)

    fig = plt.figure(figsize=(15, 6.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.4, 1], height_ratios=[1, 1],
                          hspace=0.62, wspace=0.22, top=0.9, bottom=0.09,
                          left=0.06, right=0.98)
    ax_t = fig.add_subplot(gs[0, 0])
    ax_p = fig.add_subplot(gs[1, 0], sharex=ax_t)
    ax_h = fig.add_subplot(gs[:, 1])

    # --- (1) sliding t-values ---
    ax_t.axhline(0, ls="--", lw=0.5, color="black")
    ax_t.axhline(thr, ls="--", lw=0.5, color=grey)
    ax_t.text(yr.min(), thr, " 4 STDEVs", va="bottom", ha="left",
              fontsize=8, color=grey, fontweight="bold")
    ax_t.plot(yr, tval, lw=0.4, color=teal)
    ax_t.plot([best["max_year"]], [best["t"]], "o", color=blue, ms=7, zorder=5)
    ax_t.set_ylabel("T-Value")
    _titles(ax_t, "Sliding T values",
            "Strongest Candidate Outer Year Date = %d CE    T = %.2f"
            % (best["max_year"], best["t"]))

    # --- (2) sliding Bonferroni-adjusted p-values (reversed log) ---
    pv = np.clip(pval, 1e-300, 1.0)
    ax_p.plot(yr, pv, lw=0.4, color="black")
    ax_p.set_yscale("log")
    ax_p.set_ylim(1.0, max(pv.min(), 1e-300) * 0.2)   # inverted: small p at top
    ax_p.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
    for ref, lab in ((0.05, "p = 0.05"), (1e-4, "p = 0.0001")):
        ax_p.axhline(ref, ls="--", lw=0.5, color=red)
        ax_p.text(yr.min(), ref, " " + lab, va="bottom", ha="left",
                  fontsize=7, color=red)
    ax_p.set_ylabel("adjusted P values")
    ax_p.set_xlabel("Calendar Years CE")
    _titles(ax_p, "Sliding Bonferroni adjusted p values",
            "p = %s    1/p = %s    IF = %s" % (p_disp, oop_disp, if_disp))

    # --- (3) distribution of all t-values, best marked ---
    ax_h.hist(all_t, bins=40, density=True, color=teal, edgecolor="black",
              linewidth=0.3)
    ax_h.axvline(best["t"], ls="--", lw=1.2, color=blue)
    ax_h.annotate("", xy=(best["t"], ax_h.get_ylim()[1] * 0.55),
                  xytext=(best["t"] - 0.18 * (all_t.max() - all_t.min()),
                          ax_h.get_ylim()[1] * 0.55),
                  arrowprops=dict(arrowstyle="->", color=blue, lw=1.2))
    ax_h.set_xlabel("T-value")
    ax_h.set_ylabel("Density")
    _titles(ax_h, "T-value density distribution",
            "%d CE    T = %.2f" % (best["max_year"], best["t"]))

    fig.suptitle(name, color=red, fontweight="bold", fontsize=18, y=1.0)
    if show:
        plt.show()
    return fig
