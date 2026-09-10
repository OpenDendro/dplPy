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

# Title: powt.py
# Project: OpenDendro dplPy
# Description: Adaptive power transformation of ring-width series to stabilise
#              variance (Cook & Peters 1997), a port of dplR's powt(). Each
#              series' spread is regressed on its level (from adjacent rings);
#              the fitted slope b gives a power p = |1 - b| and the series is
#              transformed x -> x**p. This is ARSTAN's data-transformation
#              option 4 ("adaptive power transform"). Two methods are offered:
#              'cook' (default) fits p independently for each series; 'universal'
#              fits a single p across all series with a linear mixed-effects
#              model (dplR's default; a later addition, not in ARSTAN).

import warnings

import numpy as np
import pandas as pd


def _skew(x):
    """Population (g1) skewness of the non-NaN values, matching ARSTAN's moment
    and scipy.stats.skew (bias=True). NaN if fewer than 3 finite values."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size < 3:
        return np.nan
    m = x.mean()
    s = x.std(ddof=0)
    if s == 0 or not np.isfinite(s):
        return np.nan
    return float(np.mean(((x - m) / s) ** 3))


def _spread_level_corr(x, prec):
    """Pearson correlation of log(spread) on log(level) from adjacent rings --
    the strength of the level-spread dependence the power transform removes. Uses
    the same mean-based level/spread as the power fit (ARSTAN's diagnostic uses
    the trailing ring for the level; dplPy stays consistent with its own power)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size < 3:
        return np.nan
    M, S = _level_spread(x, prec)
    lm, ls = np.log(M), np.log(S)
    ok = np.isfinite(lm) & np.isfinite(ls)
    if ok.sum() < 3 or np.std(lm[ok]) == 0 or np.std(ls[ok]) == 0:
        return np.nan
    return float(np.corrcoef(lm[ok], ls[ok])[0, 1])


def _getprec(values):
    """Measurement resolution used to floor zero level/spread values before the
    log regression. Mirrors dplR's getprec()."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v) & (v != 0)]
    if v.size == 0:
        return np.nan
    if np.any(v != np.round(v)):          # any fractional part -> decimal branch
        def ndec(x):
            s = np.format_float_positional(x, trim="-")
            return len(s.split(".")[1]) if "." in s else 0
        maxdig = max(ndec(x) for x in v)
    else:                                 # integer data -> trailing-zeros branch
        def ntz(x):
            s = str(int(round(x)))
            return len(s) - len(s.rstrip("0"))
        maxdig = -min(ntz(x) for x in v)
    return 10.0 ** (-maxdig)


def _level_spread(x, prec):
    """Adjacent-ring level M = mean and spread S = |difference| for Cook's power
    transform, with any exact zero floored to ``prec``. Returns (M, S)."""
    M = (x[1:] + x[:-1]) / 2.0
    S = np.abs(x[1:] - x[:-1])
    return np.where(M == 0, prec, M), np.where(S == 0, prec, S)


def _fit_slope(series, prec):
    """Return 1 - b, where b is the slope of log(spread) on log(level) built
    from adjacent rings of ``series`` (which must be NaN-free)."""
    runn_M, runn_S = _level_spread(series, prec)
    # OLS of log(S) on [1, log(M)]; slope is the second coefficient.
    b = np.polyfit(np.log(runn_M), np.log(runn_S), 1)[0]
    return 1.0 - b


def _cook_power(col, prec):
    """Cook's per-series power p = |1 - b| from the NaN-omitted series."""
    x = np.asarray(col, dtype=float)
    x = x[~np.isnan(x)]
    return abs(_fit_slope(x, prec))


def _apply_power(col, p):
    """Transform a column x -> x**p at its non-NaN positions (log fallback when
    p <= 0, matching dplR's universal method)."""
    x = np.asarray(col, dtype=float)
    out = x.copy()
    mask = ~np.isnan(x)
    if p <= 0:
        out[mask] = np.log(x[mask])
    else:
        out[mask] = x[mask] ** p
    return out


def _rescale(transformed, original):
    """z-score the transformed series then restore the original mean and sd
    (dplR's rescale=TRUE), NaN-aware."""
    t = np.asarray(transformed, dtype=float)
    o = np.asarray(original, dtype=float)
    tm, ts = np.nanmean(t), np.nanstd(t, ddof=1)
    if ts == 0 or np.isnan(ts):
        return t
    return (t - tm) / ts * np.nanstd(o, ddof=1) + np.nanmean(o)


def _powt_diagnostic_plot(stats):
    """ARSTAN-style 'data transform statistics' figure: per-series skew and
    spread-vs-level correlation, before (top) and after (bottom) the transform,
    as bars with median and quartile-hinge reference lines. ``stats`` is the
    per-series DataFrame built by powt (columns skew_before/after, r_before/after).
    A dplPy addition emulating ARSTAN's power1 (dplR's powt has no such plot)."""
    import matplotlib.pyplot as plt
    from ._plot_style import style_axes, finalize_font, ACCENT, ACCENT_WARM

    idx = np.arange(1, len(stats) + 1)

    def panel(ax, vals, title, ylab):
        vals = np.asarray(vals, dtype=float)
        ax.bar(idx, vals, width=0.7, color=ACCENT, edgecolor="none")
        ax.axhline(0, color="0.4", lw=0.8)
        finite = vals[np.isfinite(vals)]
        if finite.size:
            med = float(np.median(finite))
            lo, hi = np.percentile(finite, [25, 75])
            ax.axhline(med, color=ACCENT_WARM, lw=1.1, label="median %.2f" % med)
            ax.axhline(lo, color=ACCENT_WARM, lw=0.7, ls="--")
            ax.axhline(hi, color=ACCENT_WARM, lw=0.7, ls="--")
            ax.legend(loc="upper right", fontsize=7, frameon=False)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("series")
        ax.set_ylabel(ylab)
        style_axes(ax)

    fig, axs = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle("Data transform statistics (adaptive power, Cook & Peters 1997)",
                 fontsize=12)
    panel(axs[0, 0], stats["skew_before"], "Before: skew", "skew")
    panel(axs[0, 1], stats["r_before"], "Before: spread-vs-level correlation", "r")
    panel(axs[1, 0], stats["skew_after"], "After: skew", "skew")
    panel(axs[1, 1], stats["r_after"], "After: spread-vs-level correlation", "r")
    # share the y-axis within each column so before/after are directly comparable
    for col in (0, 1):
        lo = min(axs[0, col].get_ylim()[0], axs[1, col].get_ylim()[0])
        hi = max(axs[0, col].get_ylim()[1], axs[1, col].get_ylim()[1])
        axs[0, col].set_ylim(lo, hi)
        axs[1, col].set_ylim(lo, hi)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    finalize_font(fig)
    plt.show()
    return fig


def powt(rwl, method="cook", rescale=False, return_power=False, plot=False,
         return_stats=False):
    """Adaptive power transformation of ring-width series (variance stabilising).

    Extended Summary
    ----------------
    Stabilises the variance of ring-width series by raising each value to a data
    driven power p. The power comes from the empirical relationship between local
    level and local spread: for adjacent rings, level M = (x[t] + x[t-1]) / 2 and
    spread S = |x[t] - x[t-1]|; regressing log(S) on log(M) gives a slope b, and
    p = |1 - b| removes the level-spread dependence. This is ARSTAN's fourth data
    transformation option (Cook & Peters 1997) and a port of dplR's powt().

    Two methods are offered:

    - ``"cook"`` (default) estimates p independently for each series.
    - ``"universal"`` estimates a single p for the whole data set with a linear
      mixed-effects model (log(S) ~ log(M) with a random intercept per year),
      matching dplR's default. Requires statsmodels; if the estimate is p <= 0 a
      log transform is used instead. Not part of ARSTAN.

    Parameters
    ----------
    rwl : pandas.DataFrame or pandas.Series
        ring-width series (raw, non-negative), years as the index. A Series is
        allowed only with method='cook'.
    method : {"cook", "universal"}, default "cook"
        per-series ('cook') or single mixed-model power ('universal').
    rescale : bool, default False
        if True, rescale each transformed series back to its original mean and
        standard deviation.
    return_power : bool, default False
        if True, return a (transformed, power) tuple instead of just the
        transformed data; ``power`` is a Series over series for 'cook' or a
        single float for 'universal'.
    plot : bool, default False
        if True, draw the "data transform statistics" diagnostic -- a 2x2 figure
        of per-series skew and spread-vs-level correlation, before (top row) and
        after (bottom row) the transform, with median and quartile reference
        lines. A successful transform drives the bottom-row bars toward zero
        (series made symmetric and variance decoupled from level). This emulates
        ARSTAN's power-transform diagnostic (dplR's powt has no such plot).
    return_stats : bool, default False
        if True, also return a per-series diagnostics DataFrame with columns
        ``skew_before``, ``skew_after``, ``r_before``, ``r_after`` (the
        spread-vs-level correlations) and ``power``. The spread-vs-level
        correlation uses dplPy's mean-based level/spread (the same used to fit the
        power), which differs slightly from ARSTAN's trailing-ring level.

    Returns
    -------
    transformed data of the same type as ``rwl``. With ``return_power`` and/or
    ``return_stats`` the return becomes a tuple: ``(data, power)``,
    ``(data, stats)``, or ``(data, power, stats)`` (power first, stats last).

    Notes
    -----
    **Difference from ARSTAN's power transform.** The power comes from regressing
    log(spread) on log(level) across adjacent rings. dplPy (following dplR) defines
    the *level* as the pair mean ``(x[t] + x[t-1]) / 2``; ARSTAN's power path
    (``trnfrm``/``mndf`` with ``iopt=2``) instead uses the trailing ring ``x[t]``.
    ARSTAN also clamps the fitted power to ``[0, 1]`` (with ``p = 0`` becoming a
    log transform), whereas dplPy applies ``p = |1 - b|`` directly (log only when
    ``p <= 0``). So dplPy reproduces dplR's power exactly, but a given series'
    power can differ slightly from ARSTAN's. The ``plot``/``return_stats``
    diagnostic uses the same mean-based level as the fit, so it stays consistent
    with the power dplPy actually applies.

    Examples
    --------
    >>> import dplpy as dpl
    >>> rwl = dpl.readers("../tests/data/csv/ca533.csv")
    >>> pt = dpl.powt(rwl)                          # per-series (cook)
    >>> pt, p = dpl.powt(rwl, return_power=True)
    >>> rwi = dpl.detrend(pt, plot=False)           # transform, then detrend

    References
    ----------
    .. [1] https://rdrr.io/cran/dplR/man/powt.html
    .. [2] Cook & Peters (1997), The Holocene, 7, 361-370.
    """
    is_series = isinstance(rwl, pd.Series)
    if not (isinstance(rwl, pd.DataFrame) or is_series):
        raise TypeError("Expected a pandas DataFrame or Series, got "
                        + str(rwl.__class__) + " instead.")
    if method not in ("cook", "universal"):
        raise ValueError("method must be 'cook' or 'universal', got '" + str(method) + "'.")
    vals = rwl.to_numpy(dtype=float)
    if np.nanmin(vals) < 0:
        raise ValueError("'rwl' values cannot be negative.")
    if is_series and method != "cook":
        raise ValueError("If rwl is a Series, method must be 'cook'.")

    if method == "cook":
        return _powt_cook(rwl, is_series, rescale, return_power, plot, return_stats)
    return _powt_universal(rwl, rescale, return_power, plot, return_stats)


def _finish(result, power, stats, return_power, return_stats, plot):
    """Assemble powt's return value and draw the diagnostic if requested."""
    if plot and stats is not None:
        _powt_diagnostic_plot(stats)
    if return_power and return_stats:
        return result, power, stats
    if return_stats:
        return result, stats
    if return_power:
        return result, power
    return result


def _series_stats(name, raw, transformed, prec, power):
    """One row of the diagnostics table for a single series."""
    return {
        "series": name,
        "skew_before": _skew(raw),
        "skew_after": _skew(transformed),
        "r_before": _spread_level_corr(raw, prec),
        "r_after": _spread_level_corr(transformed, prec),
        "power": power,
    }


def _powt_cook(rwl, is_series, rescale, return_power, plot=False, return_stats=False):
    prec = _getprec(rwl.to_numpy())
    need_stats = plot or return_stats
    if is_series:
        raw = rwl.to_numpy()
        p = _cook_power(raw, prec)
        out = _apply_power(raw, p)
        if rescale:
            out = _rescale(out, raw)
        result = pd.Series(out, index=rwl.index, name=rwl.name)
        stats = None
        if need_stats:
            nm = rwl.name if rwl.name is not None else "series"
            stats = pd.DataFrame([_series_stats(nm, raw, out, prec, p)]
                                 ).set_index("series")
        return _finish(result, p, stats, return_power, return_stats, plot)

    powers = {}
    cols = {}
    rows = []
    for name in rwl.columns:
        col = rwl[name].to_numpy()
        p = _cook_power(col, prec)
        out = _apply_power(col, p)
        if rescale:
            out = _rescale(out, col)
        powers[name] = p
        cols[name] = out
        if need_stats:
            rows.append(_series_stats(name, col, out, prec, p))
    result = pd.DataFrame(cols, index=rwl.index)
    power = pd.Series(powers, name="power")
    stats = pd.DataFrame(rows).set_index("series") if need_stats else None
    return _finish(result, power, stats, return_power, return_stats, plot)


def _powt_universal(rwl, rescale, return_power, plot=False, return_stats=False):
    try:
        import statsmodels.formula.api as smf
    except ImportError as exc:
        raise ImportError("method='universal' requires statsmodels; install it "
                          "or use method='cook'.") from exc

    prec = _getprec(rwl.to_numpy())
    years = rwl.index.to_numpy()
    run_M, run_S, year, ID = [], [], [], []
    for name in rwl.columns:
        x = rwl[name].to_numpy(dtype=float)
        M, S = _level_spread(x, prec)
        run_M.append(np.log(M))
        run_S.append(np.log(S))
        year.append(years[1:])
        ID.append(np.repeat(name, len(M)))
    df = pd.DataFrame({
        "run_M": np.concatenate(run_M),
        "run_S": np.concatenate(run_S),
        "year": np.concatenate(year).astype(str),
        "ID": np.concatenate(ID),
    }).dropna(subset=["run_M", "run_S"])

    # log(S) ~ log(M) with a random intercept per year (dplR's (1|year), ML fit).
    # NB: statsmodels' default 'lbfgs' collapses the year variance to the
    # boundary here (giving the wrong slope); 'powell'/'cg'/'bfgs' recover the
    # proper estimate (consistent with lme4's), so try those in order.
    model = smf.mixedlm("run_S ~ run_M", df, groups=df["year"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = model.fit(reml=False, method=["powell", "cg", "bfgs"])
    b = float(fit.fe_params["run_M"])
    p = 1.0 - b

    need_stats = plot or return_stats
    cols = {}
    rows = []
    for name in rwl.columns:
        col = rwl[name].to_numpy()
        out = _apply_power(col, p)
        if rescale:
            out = _rescale(out, col)
        cols[name] = out
        if need_stats:
            rows.append(_series_stats(name, col, out, prec, p))
    result = pd.DataFrame(cols, index=rwl.index)
    stats = pd.DataFrame(rows).set_index("series") if need_stats else None
    return _finish(result, p, stats, return_power, return_stats, plot)
