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

# Title: fill_internal.py
# Project: OpenDendro dplPy
# Description: Fill interior gaps (true missing measurements, i.e. NaN between a
#              series' first and last ring -- NOT a real 0, which is a locally
#              absent ring). The simple methods ("Mean", "Spline", "Linear", or a
#              constant) are a port of dplR's fill.internal.NA(). The "ARSTAN"
#              method is a re-expression of Ed Cook's ARSTAN `fillin` subroutine,
#              which imputes each missing ring from the population's common signal
#              scaled to the individual series (see _fill_arstan).

import numpy as np
import pandas as pd

_SIMPLE_METHODS = ("Mean", "Spline", "Linear")


def _fmm_spline(x, y, xout):
    """Interpolating cubic spline with Forsythe-Malcolm-Moler end conditions,
    matching R's spline(..., method="fmm") -- the method dplR's fill.internal.NA
    uses. Translated from the classic FMM SPLINE/SEVAL routine; the end knots take
    the third derivative of the exact cubic through the four nearest points.
    Returns the spline evaluated at xout."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    if n < 2:
        raise ValueError("need at least two points")
    if n == 2:                                   # a straight line
        slope = (y[1] - y[0]) / (x[1] - x[0])
        return y[0] + slope * (np.asarray(xout, dtype=float) - x[0])

    b = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)

    # tridiagonal set-up
    d[0] = x[1] - x[0]
    c[1] = (y[1] - y[0]) / d[0]
    for i in range(1, n - 1):
        d[i] = x[i + 1] - x[i]
        b[i] = 2.0 * (d[i - 1] + d[i])
        c[i + 1] = (y[i + 1] - y[i]) / d[i]
        c[i] = c[i + 1] - c[i]

    # FMM end conditions (third derivative from the end cubic)
    b[0] = -d[0]
    b[n - 1] = -d[n - 2]
    c[0] = 0.0
    c[n - 1] = 0.0
    if n > 3:
        c[0] = c[2] / (x[3] - x[1]) - c[1] / (x[2] - x[0])
        c[n - 1] = c[n - 2] / (x[n - 1] - x[n - 3]) - c[n - 3] / (x[n - 2] - x[n - 4])
        c[0] = c[0] * d[0] ** 2 / (x[3] - x[0])
        c[n - 1] = -c[n - 1] * d[n - 2] ** 2 / (x[n - 1] - x[n - 4])

    # forward elimination
    for i in range(1, n):
        t = d[i - 1] / b[i - 1]
        b[i] -= t * d[i - 1]
        c[i] -= t * c[i - 1]

    # back substitution (c becomes the second-derivative terms)
    c[n - 1] /= b[n - 1]
    for i in range(n - 2, -1, -1):
        c[i] = (c[i] - d[i] * c[i + 1]) / b[i]

    # polynomial coefficients b, c, d
    b[n - 1] = (y[n - 1] - y[n - 2]) / d[n - 2] + d[n - 2] * (c[n - 2] + 2.0 * c[n - 1])
    for i in range(n - 1):
        b[i] = (y[i + 1] - y[i]) / d[i] - d[i] * (c[i + 1] + 2.0 * c[i])
        d[i] = (c[i + 1] - c[i]) / d[i]
        c[i] *= 3.0
    c[n - 1] *= 3.0
    d[n - 1] = d[n - 2]

    # evaluate: S(u) = y[i] + b[i]*h + c[i]*h^2 + d[i]*h^3, h = u - x[i]
    xout = np.asarray(xout, dtype=float)
    idx = np.searchsorted(x, xout) - 1
    idx = np.clip(idx, 0, n - 2)               # extrapolation uses the end piece
    h = xout - x[idx]
    return y[idx] + h * (b[idx] + h * (c[idx] + h * d[idx]))


def fill_internal(data: pd.DataFrame, fill="Mean", **kwargs):
    """Fill interior gaps (interior NaN) in a ring-width dataset.

    Only *interior* NaN -- those between a series' first and last real ring --
    are filled; leading and trailing NaN are left untouched. A real 0 (a locally
    absent ring) is data and is never treated as a gap. Series with one or no
    real value are returned unchanged.

    Parameters
    ----------
    data : pandas dataframe
        ring widths (or indices), years as the index and series as columns.
    fill : {"Mean", "Spline", "Linear", "ARSTAN"} or a number, default "Mean"
        how to fill each interior gap:

        - "Mean"   : the series' mean (flat fill). [dplR fill.internal.NA]
        - "Linear" : straight-line interpolation between the flanking values.
        - "Spline" : cubic-spline interpolation through the present values.
        - a number : that constant value.
        - "ARSTAN" : Ed Cook's ARSTAN `fillin` -- impute each missing ring from
          the common (mean) chronology signal, moment-matched to the individual
          series and modulated by its growth curve (see Notes and _fill_arstan).

        The four simple options are a port of dplR's fill.internal.NA(); "Mean",
        "Linear" and "Spline" interpolate within each series only.
    **kwargs
        for fill="ARSTAN", forwarded to _fill_arstan (e.g. growth_nyrs,
        long_gap, flank).

    Returns
    -------
    out : pandas dataframe, same shape/index/columns as `data`, with interior
        gaps filled.

    Notes
    -----
    The simple methods know nothing about the other series -- they just bridge
    each hole -- so use them when you need an NaN-free series for a downstream
    step, not as if the filled values were measured. "ARSTAN" instead borrows the
    shared year-to-year signal from all series, which is the point of Cook's
    method.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Expected input data to be pandas dataframe, not "
                        + str(type(data)))

    # Resolve the fill mode: a number (constant), "ARSTAN", or a simple method.
    if isinstance(fill, str):
        if fill == "ARSTAN":
            return _fill_arstan(data, **kwargs)
        if fill not in _SIMPLE_METHODS:
            raise ValueError('fill must be one of "Mean", "Spline", "Linear", '
                             '"ARSTAN", or a number; got ' + repr(fill))
        mode = fill
        const = None
    elif isinstance(fill, (int, float, np.integer, np.floating)) and not isinstance(fill, bool):
        mode = "Constant"
        const = float(fill)
    else:
        raise ValueError('fill must be one of "Mean", "Spline", "Linear", '
                         '"ARSTAN", or a number; got ' + repr(fill))

    out = data.copy()
    for col in out.columns:
        out[col] = _fill_series_simple(out[col].to_numpy(dtype=float), mode, const)
    return out


def _fill_series_simple(x, mode, const):
    """Fill interior NaN of one series (a 1-D float array) by `mode`. Positions
    are unit-spaced indices within the interior span, matching dplR's use of
    which(!is.na) as the interpolation abscissa."""
    x = x.copy()
    ok = np.flatnonzero(~np.isnan(x))
    if ok.size <= 1:
        return x
    first, last = ok[0], ok[-1]
    span = np.arange(first, last + 1)
    seg = x[span]
    na = np.isnan(seg)
    if not na.any():                       # no interior gaps
        return x

    good = np.flatnonzero(~na)             # positions within the span
    bad = np.flatnonzero(na)
    if mode == "Mean":
        seg[bad] = seg[good].mean()
    elif mode == "Constant":
        seg[bad] = const
    elif mode == "Linear":
        seg[bad] = np.interp(bad, good, seg[good])
    elif mode == "Spline":
        # R's spline() default method "fmm" (matches dplR fill.internal.NA).
        seg[bad] = _fmm_spline(good, seg[good], bad)
    x[span] = seg
    return x


def _runs_of_nan(col, first, last):
    """List of (start, end) index pairs (inclusive) for each run of NaN strictly
    inside [first, last] of a 1-D array."""
    runs = []
    t = first
    while t <= last:
        if np.isnan(col[t]):
            gs = t
            while t <= last and np.isnan(col[t]):
                t += 1
            runs.append((gs, t - 1))
        else:
            t += 1
    return runs


def _fill_arstan(data, growth_nyrs=20, long_gap=20, flank=10, biweight=True,
                 stabilize=True, stabilize_nyrs=50):
    """Fill interior gaps by Ed Cook's ARSTAN `fillin` method (re-expressed).

    Each missing ring is imputed from the population's *common signal* scaled to
    the individual series, rather than merely interpolated. The stages mirror the
    ARSTAN subroutine:

    1. For every series, fit a growth curve (a `growth_nyrs`-year smoothing
       spline) through its present rings. A gap longer than `long_gap` years first
       gets a low-frequency linear seed from up to `flank` present years on each
       side (ARSTAN's pearsn1 regression), so the growth spline does not sag into
       long gaps.
    2. Build the common chronology: the (biweight) mean of the raw widths across
       series each year, detrended by an n/3-year spline, then (optionally)
       variance-stabilized with ARSTAN's spline `stabit` step.
    3. For each missing ring, take the common signal at that year, standardize it
       against the common signal's mean/SD over the series' span, rescale it to
       the series' own index mean/SD (moment matching), clamp negatives to zero,
       and multiply back through the series' growth curve to recover a width.

    Present rings are returned unchanged. This is a faithful re-expression of the
    FORTRAN algorithm's intent (there is no released reference to validate it to
    machine precision), so treat the filled values as principled estimates, not
    measurements.

    Parameters
    ----------
    data : pandas dataframe
        ring widths, years x series, interior gaps as NaN.
    growth_nyrs : int, default 20
        stiffness (years) of the per-series growth spline.
    long_gap : int, default 20
        gaps longer than this get the flanking-years linear seed.
    flank : int, default 10
        number of present years each side used for the long-gap linear seed.
    biweight : bool, default True
        aggregate series with Tukey's biweight mean (else arithmetic).
    stabilize : bool, default True
        variance-stabilize the common chronology (ARSTAN's stabit).
    stabilize_nyrs : int, default 50
        stiffness (years) for the stabit spline when stabilize=True.
    """
    from csaps import csaps
    from .smoothingspline import get_param
    from .chron_stabilized import _spline_stabilize
    from .tbrm import tbrm

    df = data.copy()
    if not all(np.issubdtype(df[c].dtype, np.number) for c in df.columns):
        raise TypeError("ARSTAN fill requires all-numeric columns.")
    years = np.asarray(df.index, dtype=float)
    X = df.to_numpy(dtype=float)
    n_years, n_series = X.shape
    depth = (~np.isnan(X)).sum(axis=1)

    # ---- 1. per-series growth curves (+ long-gap linear seed) ----
    growth = np.full((n_years, n_series), np.nan)
    gaps_by_series = []
    for s in range(n_series):
        col = X[:, s]
        present = ~np.isnan(col)
        if present.sum() < 2:
            gaps_by_series.append([])
            continue
        idx = np.flatnonzero(present)
        first, last = idx[0], idx[-1]
        gaps = _runs_of_nan(col, first, last)
        gaps_by_series.append(gaps)

        seed = col.copy()
        anchor = present.copy()
        for (gs, ge) in gaps:
            if (ge - gs + 1) > long_gap:
                pre = idx[idx < gs][-flank:]
                post = idx[idx > ge][:flank]
                fx = np.concatenate([pre, post])
                if fx.size >= 2:
                    slope, intercept = np.polyfit(years[fx], col[fx], 1)
                    gy = np.arange(gs, ge + 1)
                    seed[gy] = intercept + slope * years[gy]
                    anchor[gy] = True

        span = np.arange(first, last + 1)
        pos = np.arange(span.size, dtype=float)          # unit-spaced positions
        a_local = np.flatnonzero(anchor[first:last + 1])
        if a_local.size >= 4:
            p = get_param(0.5, max(int(growth_nyrs), 2))
            g_span = np.asarray(
                csaps(pos[a_local], seed[span][a_local], pos, smooth=p), dtype=float)
        else:                                             # too few anchors: flat
            g_span = np.full(span.size, np.nanmean(col[present]))
        g_span = np.where(g_span <= 0, np.nan, g_span)    # growth must be positive
        growth[span, s] = g_span

    # ---- 2. common chronology: biweight mean of raw, n/3-spline detrend, stabit ----
    m_raw = np.full(n_years, np.nan)
    for t in range(n_years):
        row = X[t][~np.isnan(X[t])]
        if row.size > 0:
            m_raw[t] = tbrm(row, c=9) if biweight else row.mean()

    present_year = depth > 0
    yi = np.flatnonzero(present_year)
    C = np.full(n_years, np.nan)
    if yi.size >= 4:
        fp, lp = yi[0], yi[-1]
        mspan = np.arange(fp, lp + 1)
        maxn = mspan.size
        mp = np.arange(maxn, dtype=float)
        seg = m_raw[mspan].copy()
        # bridge any interior all-gap years so the spline has no NaN to fit
        nanmask = np.isnan(seg)
        if nanmask.any():
            seg[nanmask] = np.interp(mp[nanmask], mp[~nanmask], seg[~nanmask])
        p2 = get_param(0.5, max(maxn // 3, 2))
        trend = np.asarray(csaps(mp, seg, mp, smooth=p2), dtype=float)
        m_index = seg / np.where(trend <= 0, np.nan, trend)
        if stabilize:
            C[mspan] = _spline_stabilize(m_index, depth[mspan],
                                         spline_nyrs=stabilize_nyrs)
        else:
            C[mspan] = m_index

    # ---- 3. moment-matched infill of each series' gaps ----
    out = X.copy()
    for s in range(n_series):
        gaps = gaps_by_series[s]
        if not gaps:
            continue
        col = X[:, s]
        present = ~np.isnan(col)
        idx = np.flatnonzero(present)
        first, last = idx[0], idx[-1]
        span = np.arange(first, last + 1)
        g = growth[:, s]

        with np.errstate(invalid="ignore", divide="ignore"):
            i_s = col[span] / g[span]                     # series index over span
        ave2 = np.nanmean(i_s)
        sdev2 = np.nanstd(i_s, ddof=1)
        cspan = C[span]
        cok = ~np.isnan(cspan)
        if cok.sum() < 2 or not np.isfinite(sdev2) or sdev2 == 0:
            continue                                       # cannot moment-match
        ave3 = cspan[cok].mean()
        sdev3 = cspan[cok].std(ddof=1)
        if sdev3 == 0:
            continue
        for (gs, ge) in gaps:
            for t in range(gs, ge + 1):
                if np.isnan(C[t]) or np.isnan(g[t]):
                    continue
                r = (C[t] - ave3) / sdev3 * sdev2 + ave2   # match series index
                if r < 0:
                    r = 0.0
                out[t, s] = r * g[t]                        # back to width units

    return pd.DataFrame(out, index=df.index, columns=df.columns)
