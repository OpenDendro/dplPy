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

# Date: 11/1/2022
# Author: Ifeoluwa Ale
# Title: curvefit.py
# Description: This file contains helper functions which fit data
#              from a series to curves. The curves included are hugershoff,
#              modified negative exponential, linear and horizontal.

import warnings

import numpy as np
from scipy.optimize import curve_fit


def _mean_curve(y):
    """The detrend-by-the-series-mean fallback shared by every fitter: a flat
    curve at the series mean, with its model_info. Returns (curve, meta)."""
    m = float(np.mean(y))
    return np.full(len(y), m), {"method": "Mean", "mean": m}


def _line_info(x, yl):
    """model_info for a straight-line detrend fit, from its endpoints (the
    intercept/slope block shared by the ModNegExp and ModHugershoff fallbacks)."""
    slope = (yl[-1] - yl[0]) / (x[-1] - x[0]) if x[-1] != x[0] else 0.0
    return {"method": "Line",
            "coefs": {"intercept": float(yl[0] - slope * x[0]), "slope": float(slope)}}


# Modified hugershoff function
def hugershoff_function(x, a, b, c, d):
    return a*(x**b)*np.exp(c*x) + d

# Hugershoff growth curve, Ed Cook's ARSTAN method (subroutine hughdi): fit
# y = a * t^m * exp(-k*t) by LINEARIZING -- ln(y) = ln(a) + m*ln(t) - k*t is an
# ordinary 3-parameter linear regression of ln(y) on [1, ln(t), t], solved in
# closed form (no iteration, so it never fails to converge or falls back). The
# amplitude is then rescaled by cc = sum(y)/sum(fit) so the fitted curve preserves
# the data's total (ARSTAN's log-transform bias correction). Only positive rings
# enter the regression. Unlike dplR's ModHugershoff nls, this is deterministic and
# always returns a positive curve, but it optimises log-space error rather than
# data-space error, so it gives a different (not merely better/worse-converged)
# curve. Cook deliberately does not reject m<=0 (that check is commented out in
# hughdi), so on monotonically declining data it returns a power-law decline.
def hugershoff_arstan(x, y, name="", info=False):
    def out(curve, meta):
        return (curve, meta) if info else curve
    y = np.asarray(y, dtype=float)
    n = len(y)
    t = np.arange(1.0, n + 1)
    ok = (y > 0) & (t > 0)                       # ln needs positive rings
    tk, yk = t[ok], y[ok]
    xl, yl = np.log(tk), np.log(yk)
    nred = len(tk)
    sumt = tk.sum();  sumt2 = (tk * tk).sum()
    sumx = xl.sum();  sumx2 = (xl * xl).sum();  sumxt = (xl * tk).sum()
    sumy = yl.sum();  sumxy = (xl * yl).sum();  sumyt = (yl * tk).sum()
    # 3x3 normal equations for [ln a, m, -k] via Cramer's rule (as in hughdi)
    rn1 = sumx2 * sumt2 - sumxt ** 2
    rn2 = sumt * sumxt - sumx * sumt2
    rn3 = sumx * sumxt - sumt * sumx2
    rn = nred * rn1 + sumx * rn2 + sumt * rn3
    if not np.isfinite(rn) or abs(rn) < 1e-300 or nred < 3:
        warnings.warn("Hugershoff (ARSTAN) regression is singular for "
                      + str(name) + "; detrending by the series mean instead.\n")
        return out(*_mean_curve(y))
    a_ = (sumy * rn1 + sumxy * rn2 + sumyt * rn3) / rn                 # ln(a)
    rz2 = sumx * sumt - nred * sumxt
    cm = (sumy * rn2 + sumxy * (nred * sumt2 - sumt ** 2) + sumyt * rz2) / rn   # m
    rz1 = sumt * sumx2 - sumx * sumxt
    rz2b = nred * sumxt - sumx * sumt
    ck = (sumy * rz1 + sumxy * rz2b + sumyt * (sumx ** 2 - nred * sumx2)) / rn
    dcoef = (np.exp(a_), cm, -ck)                # (a, m, coefficient on t = -k)
    fit = dcoef[0] * t ** dcoef[1] * np.exp(dcoef[2] * t)
    cc = y.sum() / fit.sum()                     # cc = se/sf + 1 = sum(y)/sum(fit)
    fit = fit * cc
    return out(fit, {"method": "Hugershoff",
                     "coefs": {"a": float(dcoef[0] * cc), "m": float(cm),
                               "k": float(ck)}})


# Modified Hugershoff detrending curve with dplR's fallback chain. Mirrors dplR
# detrend.series.R (method="ModHugershoff"): fit y = a*t^b*exp(-g*t)+d (here the
# exponent parameter c plays the role of dplR's -g); reject the fit if a<=0,
# b<=0, or the curve ends non-positive (dplR rejects on a<=0 || b<=0 and on a
# non-positive tail). On rejection, fall back to a straight line -- accepted only
# when its slope is <= 0 (or pos_slope) AND all its values are positive -- and
# finally to the series mean, exactly as ModNegExp does.
def mod_hugershoff(x, y, pos_slope=False, name="", info=False):
    def out(curve, meta):
        return (curve, meta) if info else curve
    nY = len(y)
    t = np.arange(1, nY + 1)
    # dplR default constrain.nls="never": unconstrained nls with start values
    # a = d = mean of the last 10%, b = 1, g = 0.1 (so c = -g = -0.1). Reject on
    # a<=0, b<=0, or a non-positive tail, as dplR's hug.func does.
    tail = y[int(np.floor(nY * 0.9)) - 1:]
    a0 = float(np.mean(tail)) if len(tail) else float(np.mean(y))
    try:
        pars, pcov = curve_fit(hugershoff_function, t, y, p0=[a0, 1.0, -0.1, a0],
                               maxfev=200000)
        a, b, c, d = pars
        fit = hugershoff_function(t, a, b, c, d)
        if a > 0 and b > 0 and fit[-1] > 0 and np.all(np.isfinite(fit)):
            # The 4-parameter Hugershoff is often poorly identifiable. scipy's
            # solver is more robust than dplR's nls and will fit series where
            # dplR gives up and falls back; when the fit's covariance is very
            # ill-conditioned we warn, since another implementation (dplR) may
            # reject this fit -- so a divergence is never silent.
            cond = np.linalg.cond(pcov) if np.all(np.isfinite(pcov)) else np.inf
            if cond > 1e12:
                warnings.warn(
                    "ModHugershoff fit for " + str(name) + " is poorly "
                    "constrained (covariance condition number "
                    + ("%.1e" % cond) + "); the 4-parameter Hugershoff can be "
                    "unstable and other implementations (e.g. dplR) may reject it "
                    "and fall back to a line/mean. Inspect the curve, or prefer "
                    "fit='Spline' or fit='ModNegExp'.\n")
            return out(fit, {"method": "Hugershoff",
                             "coefs": {"a": float(a), "b": float(b),
                                       "c": float(c), "d": float(d)},
                             "cond": float(cond)})
    except (RuntimeError, ValueError):
        pass
    # straight-line fallback
    yl = linear(x, y)
    if (yl[-1] - yl[0] <= 0 or pos_slope) and np.all(yl > 0):
        warnings.warn("ModHugershoff could not fit " + str(name)
                      + "; using a linear fit instead.\n")
        return out(yl, _line_info(x, yl))
    # final fallback: the series mean
    warnings.warn("ModHugershoff and the linear fallback are unsuitable for "
                  + str(name) + "; detrending by the series mean instead.\n")
    return out(*_mean_curve(y))


# Modified negative exponential function
def negex_function(x, a, b, k):
    return a * np.exp(b * x) + k

# Modified negative exponential detrending curve with dplR's fallback chain.
# Mirrors dplR detrend.series.R (method="ModNegExp"): fit y = a*exp(b*t)+k with
# a>0, b<0; reject the fit if a<=0, b>=0, or the curve ends non-positive. If the
# neg-exp is rejected, fall back to a straight line, accepted only when its slope
# is <= 0 (or pos_slope is True) AND all its fitted values are positive; otherwise
# fall back to the series mean (dplR's "dirty dog" case). This keeps a whole
# collection from aborting on one series the neg-exp can't fit.
def mod_neg_exp(x, y, pos_slope=False, name="", info=False):
    # With info=True return (curve, model_info); otherwise just the curve, so
    # existing callers are unaffected. model_info mirrors dplR's method labels:
    # "NegativeExponential" (the nls fit), "Line" (linear fallback), or "Mean".
    def out(curve, meta):
        return (curve, meta) if info else curve
    nY = len(y)
    t = np.arange(1, nY + 1)
    # 1. unconstrained negative-exponential fit with dplR's start values
    #    (dplR default constrain.nls="never"): a = mean of the first 10%,
    #    b = -0.01, k = mean of the last 10%. Reject on a<=0, b>=0, or a
    #    non-positive tail, exactly as dplR's nec.func does.
    a0 = float(np.mean(y[:max(1, int(np.floor(nY * 0.1)))]))
    k0 = float(np.mean(y[int(np.floor(nY * 0.9)) - 1:]))
    try:
        pars, _ = curve_fit(negex_function, t, y, p0=[a0, -0.01, k0],
                            maxfev=100000)
        a, b, k = pars
        fit = negex_function(t, a, b, k)
        if a > 0 and b < 0 and fit[-1] > 0 and np.all(np.isfinite(fit)):
            return out(fit, {"method": "NegativeExponential",
                             "coefs": {"a": float(a), "b": float(b), "k": float(k)}})
    except (RuntimeError, ValueError):
        pass
    # 2. straight-line fallback
    yl = linear(x, y)
    if (yl[-1] - yl[0] <= 0 or pos_slope) and np.all(yl > 0):
        warnings.warn("ModNegExp could not fit " + str(name)
                      + "; using a linear fit instead.\n")
        return out(yl, _line_info(x, yl))
    # 3. final fallback: the series mean
    warnings.warn("ModNegExp and the linear fallback are unsuitable for "
                  + str(name) + "; detrending by the series mean instead.\n")
    return out(*_mean_curve(y))

# --- ARSTAN deterministic negative exponential (subroutine `curve`) ----------
# Ed Cook / R.L. Holmes' deterministic neg-exp fit: model f(i) = ah*exp(-b*i) + eh
# (i = 1..n). Unlike dplR's ModNegExp (a nonlinear least-squares fit), the decay
# rate b is found by a 1-D adaptive search seeded from the first-10/last-10 ring
# means, and the amplitude ah and asymptote eh are solved in closed form at each
# candidate b. It minimises the SAME data-space RSS as ModNegExp for the SAME
# model, so when the nls converges cleanly the two curves coincide (to numerical
# precision) -- unlike the Hugershoff pair, whose two members optimise different
# error spaces (log vs data). NegExp's value is therefore robustness, not a
# different shape: the deterministic search never fails to converge, so it still
# returns a fit on series where the nls diverges and ModNegExp falls back to a
# line or the mean. It is the neg-exp counterpart of hugershoff_arstan only in
# spirit (an ARSTAN deterministic port beside a dplR nls fit).
def _neg_exp_profile(y, b):
    """Closed-form (ah, eh, rss) for a fixed decay rate b (ARSTAN curve inner
    solve): least-squares fit of y = eh + ah*exp(-b*i). Returns None if degenerate."""
    n = len(y)
    idx = np.arange(1.0, n + 1)
    if np.any(np.abs(b) * idx > 112.8):        # exp overflow guard (as in curve)
        return None
    f = np.exp(-b * idx)
    a = float(n); bsum = f.sum(); d = y.sum()
    e = float((f * f).sum()); g = float((y * f).sum())
    if not (1e-15 < bsum < 1e15) or not (1e-15 < e < 1e15):
        return None
    denom = a * e - bsum * bsum
    if not np.isfinite(denom) or denom == 0:
        return None
    eh = (d * e - bsum * g) / denom
    ah = (d - a * eh) / bsum
    rss = float(np.sum((y - eh - ah * f) ** 2))
    return ah, eh, rss


def _neg_exp_curve(y):
    """ARSTAN `curve`: deterministic search for the neg-exp decay rate b, with
    (ah, eh) profiled out. Returns (ah, b, eh) or None on failure (n<20, overflow,
    or a rejected fit: b<=0 or eh<0), mirroring the routine's flag-and-fall-back."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 20:
        return None
    yf = float(y[:10].mean())
    yl = float(y[-10:].mean())
    if yf <= 0 or yl <= 0:
        return None
    b = (np.log(yf) - np.log(yl)) / (n - 10.0)
    if b <= 0:
        b = 1e-8
    dlt = 0.0005
    dinc = 5e-7
    mid = _neg_exp_profile(y, b)
    if mid is None:
        return None
    ah, eh, rss_mid = mid
    # Two-phase 1-D search (ARSTAN `curve`): compare the two neighbours b +/- dlt,
    # greedily stepping the centre onto whichever lowers the RSS. `mo` latches the
    # moment an iteration finds NO improvement; while it is unset (still improving)
    # the step DOUBLES to accelerate toward the optimum, and once latched the step
    # HALVES each iteration to refine, until dlt <= dinc (convergence).
    mo = False
    for _ in range(34):
        improved = False
        for bk in (b - dlt, b + dlt):
            r = _neg_exp_profile(y, bk)         # arg-overflow -> ARSTAN's goto 99
            if r is None:
                return None
            if r[2] < rss_mid:                  # sequential/greedy, as in curve
                b, (ah, eh, rss_mid) = bk, r
                improved = True
        if not improved:
            mo = True                           # latch: leave the accelerate phase
        if not mo:
            dlt *= 2.0                          # accelerate: widen the step
        else:
            dlt *= 0.5                          # refine: shrink the step
            if dlt <= dinc:
                break
    # ARSTAN reject conditions (line 11903): asymptote too large, non-positive
    # amplitude, or a negative decay rate -> signal the caller to fall back.
    if eh > 9999.0 or ah <= 0.0 or b < 0.0:
        return None
    return ah, b, eh


def neg_exp_arstan(x, y, pos_slope=False, name="", info=False):
    """Deterministic negative-exponential detrend (ARSTAN `curve`), with the same
    line->mean fallback chain as ModNegExp. The ARSTAN counterpart of dplR's
    ModNegExp: same model f = a*exp(-b*t)+k (a>0, b>0, k>=0), fit deterministically
    rather than by nonlinear least squares."""
    def out(curve, meta):
        return (curve, meta) if info else curve
    y = np.asarray(y, dtype=float)
    n = len(y)
    sol = _neg_exp_curve(y)
    if sol is not None:
        ah, b, eh = sol
        t = np.arange(1.0, n + 1)
        fit = ah * np.exp(-b * t) + eh
        if np.all(fit > 0) and np.all(np.isfinite(fit)):
            return out(fit, {"method": "NegativeExponential",
                             "coefs": {"a": float(ah), "b": float(-b), "k": float(eh)}})
    # straight-line fallback (slope <= 0 unless pos_slope), then the mean
    yl = linear(x, y)
    if (yl[-1] - yl[0] <= 0 or pos_slope) and np.all(yl > 0):
        warnings.warn("NegExp (deterministic) could not fit " + str(name)
                      + "; using a linear fit instead.\n")
        return out(yl, _line_info(x, yl))
    warnings.warn("NegExp (deterministic) and the linear fallback are unsuitable for "
                  + str(name) + "; detrending by the series mean instead.\n")
    return out(*_mean_curve(y))


# --- ARSTAN general exponential (subroutine `gexp`, menu option 8) ------------
# The Hugershoff family with the power exponent fixed at 1: f(t) = a * t * exp(b*t)
# (a>0, b<0). Fit deterministically by linearising -- ln(y/t) = ln(a) + b*t is an
# ordinary least-squares line of ln(y/t) on t over the positive rings -- so it is
# closed-form and never fails to converge. A gentler rise-then-decline curve than
# the full 4-parameter Hugershoff; ARSTAN's gexp carries no intercept term.
def general_exp(x, y, name="", info=False):
    def out(curve, meta):
        return (curve, meta) if info else curve
    y = np.asarray(y, dtype=float)
    n = len(y)
    t = np.arange(1.0, n + 1)
    ok = (y > 0) & (t > 0)
    tk = t[ok]
    if tk.size < 3:
        warnings.warn("GeneralExp needs >=3 positive rings for " + str(name)
                      + "; detrending by the series mean instead.\n")
        return out(*_mean_curve(y))
    z = np.log(y[ok]) - np.log(tk)             # ln(y/t)
    sxx = float(np.sum((tk - tk.mean()) ** 2))
    if sxx == 0:
        return out(*_mean_curve(y))
    b = float(np.sum((tk - tk.mean()) * (z - z.mean())) / sxx)   # OLS slope
    a = float(np.exp(z.mean() - b * tk.mean()))                  # ln-intercept -> a
    fit = a * t * np.exp(b * t)
    if not (np.all(fit > 0) and np.all(np.isfinite(fit))):
        warnings.warn("GeneralExp fit is not all positive for " + str(name)
                      + "; detrending by the series mean instead.\n")
        return out(*_mean_curve(y))
    return out(fit, {"method": "GeneralExponential",
                     "coefs": {"a": a, "b": b}})


# Fit a horizontal line to the series
def horizontal(x, y):
    yi = np.asarray([np.mean(y)] * len(x))
    return yi
    
# Equation of a straight line
def line_function(x, m, c):
    return (m * x) + c

# Fit a line to the series
def linear(x, y, bounds=False):
    if bounds is False:
        pars, unk = curve_fit(line_function, x, y)
    else:
        # Currently unused: every caller invokes linear() with the default
        # bounds=False. This branch constrains the slope to <= 0 (an upper bound
        # of 0) and is kept only for a possible future constrained-fit option.
        pars, unk = curve_fit(line_function, x, y, bounds=([-np.inf, -np.inf], [0, np.inf]))
    m, c = pars
    yi = line_function(x, m, c)
    return yi


# ARSTAN opt 5 / LinearNegative: a best-fit straight line accepted only when its
# slope is <= 0 (a non-increasing line) AND all its values are positive; otherwise
# detrend by the series mean. (opt 4 / LinearAny is the plain any-slope linear().)
def linear_neg(x, y, name="", info=False):
    def out(curve, meta):
        return (curve, meta) if info else curve
    yl = linear(x, y)
    if (yl[-1] - yl[0]) <= 0 and np.all(yl > 0):
        return out(yl, _line_info(x, yl))
    warnings.warn("LinearNegative: the best-fit slope is positive (or the line is "
                  "non-positive) for " + str(name)
                  + "; detrending by the series mean instead.\n")
    return out(*_mean_curve(y))