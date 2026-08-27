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
# Title: smoothingspline.py
# Description: This contains the spline method which fits a series to
#              a spline curve.

from math import cos
from math import pi, floor
from csaps import csaps

# Returns the spline parameter, given amplitude of the series and the period
def get_param(amp, period):
    freq = 1/period
    spline_param = 1/(((cos(2 * pi * freq) + 2) * (1 - amp)/(12 * amp * (cos(2 * pi * freq) - 1) ** 2))+ 1)
    return spline_param

def get_period(period, n):
    # The default "n-year spline" wavelength is floor(0.67 * n), matching dplR's
    # detrend.series (nyrs = floor(nY2 * 0.67)). Using an unfloored 0.67 * n left
    # dplPy's spline ~3e-5 off dplR's -- the sole source of the crossdating gap.
    if period is None:
        return floor(n * 0.67)
    elif period < 0:
        return n * abs(period)/100
    elif period <= 1:
        return n * period
    else:
        return period

# The one csaps smoothing-spline call, shared by spline() and rcs's caps(): fit
# at x with an f-amplitude cutoff at the already-resolved wavelength `nyrs`, and
# evaluate back at x. Callers resolve `nyrs` themselves -- spline() via
# get_period, rcs's caps via integer truncation -- so this only centralizes the
# csaps/get_param wiring, leaving each caller's wavelength convention intact.
def _smooth_csaps(x, y, nyrs, f):
    return csaps(x, y, x, smooth=get_param(f, nyrs))

# Fits a curve to the series given as input and returns the y-values of the curve.
# `f` is the spline's frequency-response amplitude at the `period` wavelength
# (dplR's `f`, default 0.5: a 50% amplitude cutoff at that wavelength).
def spline(x, y, period=None, f=0.5):
    return _smooth_csaps(x, y, get_period(period, len(x)), f)