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

# Modified hugershoff function
def hugershoff_function(x, a, b, c, d):
    return a*(x**b)*np.exp(c*x) + d

# Attempt to fit a hugershoff curve to the series
def hugershoff(x, y):
    xi = np.arange(1, len(y)+1)
    pars, unk= curve_fit(hugershoff_function, xi, y, bounds=([0, -2, -np.inf, min(y)], [np.inf, 2, 0, max(y)]), 
                            p0=[max(y)-min(y), 0, 0, y[0]])
    a, b, c, d = pars

    yi = hugershoff_function(xi, a, b, c, d)
    return yi

# Modified Hugershoff detrending curve with dplR's fallback chain. Mirrors dplR
# detrend.series.R (method="ModHugershoff"): fit y = a*t^b*exp(-g*t)+d (here the
# exponent parameter c plays the role of dplR's -g); reject the fit if a<=0,
# b<=0, or the curve ends non-positive (dplR rejects on a<=0 || b<=0 and on a
# non-positive tail). On rejection, fall back to a straight line -- accepted only
# when its slope is <= 0 (or pos_slope) AND all its values are positive -- and
# finally to the series mean, exactly as ModNegExp does.
def mod_hugershoff(x, y, pos_slope=False, name=""):
    t = np.arange(1, len(y) + 1)
    nY = len(y)
    tail = y[int(np.floor(nY * 0.9)) - 1:]          # dplR seeds a, d from last ~10%
    a0 = float(np.mean(tail)) if len(tail) else float(np.mean(y))
    try:
        pars, _ = curve_fit(hugershoff_function, t, y, p0=[a0, 1.0, -0.1, a0],
                            bounds=([0, 0, -np.inf, 0], [np.inf, np.inf, 0, np.inf]),
                            maxfev=10000)
        a, b, c, d = pars
        fit = hugershoff_function(t, a, b, c, d)
        if a > 0 and b > 0 and fit[-1] > 0 and np.all(np.isfinite(fit)):
            return fit
    except (RuntimeError, ValueError):
        pass
    # straight-line fallback
    yl = linear(x, y)
    if (yl[-1] - yl[0] <= 0 or pos_slope) and np.all(yl > 0):
        warnings.warn("ModHugershoff could not fit " + str(name)
                      + "; using a linear fit instead.\n")
        return yl
    # final fallback: the series mean
    warnings.warn("ModHugershoff and the linear fallback are unsuitable for "
                  + str(name) + "; detrending by the series mean instead.\n")
    return np.full_like(y, np.mean(y))


# Modified negative exponential function
def negex_function(x, a, b, k):
    return a * np.exp(b * x) + k

# Attempt to fit a negative exponential curve to the series
def negex(x, y):
    xi = np.arange(1, len(y)+1)
    pars, unk= curve_fit(negex_function, xi, y, bounds=([0, -np.inf, 0], [np.inf, 0, np.inf]))
    a, b, k = pars

    yi = negex_function(xi, a, b, k)
    return yi


# Modified negative exponential detrending curve with dplR's fallback chain.
# Mirrors dplR detrend.series.R (method="ModNegExp"): fit y = a*exp(b*t)+k with
# a>0, b<0; reject the fit if a<=0, b>=0, or the curve ends non-positive. If the
# neg-exp is rejected, fall back to a straight line, accepted only when its slope
# is <= 0 (or pos_slope is True) AND all its fitted values are positive; otherwise
# fall back to the series mean (dplR's "dirty dog" case). This keeps a whole
# collection from aborting on one series the neg-exp can't fit.
def mod_neg_exp(x, y, pos_slope=False, name=""):
    t = np.arange(1, len(y) + 1)
    # 1. constrained negative-exponential fit
    try:
        pars, _ = curve_fit(negex_function, t, y,
                             bounds=([0, -np.inf, 0], [np.inf, 0, np.inf]))
        a, b, k = pars
        fit = negex_function(t, a, b, k)
        if a > 0 and b < 0 and fit[-1] > 0 and np.all(np.isfinite(fit)):
            return fit
    except (RuntimeError, ValueError):
        pass
    # 2. straight-line fallback
    yl = linear(x, y)
    slope_sign = yl[-1] - yl[0]            # x is increasing, so this signs the slope
    if (slope_sign <= 0 or pos_slope) and np.all(yl > 0):
        warnings.warn("ModNegExp could not fit " + str(name)
                      + "; using a linear fit instead.\n")
        return yl
    # 3. final fallback: the series mean
    warnings.warn("ModNegExp and the linear fallback are unsuitable for "
                  + str(name) + "; detrending by the series mean instead.\n")
    return np.full_like(y, np.mean(y))

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
        pars, unk = curve_fit(line_function, x, y, bounds=([-np.inf, -np.inf], [0, np.inf]))
    m, c = pars
    yi = line_function(x, m, c)
    return yi