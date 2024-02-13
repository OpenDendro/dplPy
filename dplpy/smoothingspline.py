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
from math import pi
from csaps import csaps

# Returns the spline parameter, given amplitude of the series and the period
def get_param(amp, period):
    freq = 1/period
    spline_param = 1/(((cos(2 * pi * freq) + 2) * (1 - amp)/(12 * amp * (cos(2 * pi * freq) - 1) ** 2))+ 1)
    return spline_param

def get_period(period, n):
    if period is None:
        return n * 0.67
    elif period < 0:
        return n * abs(period)/100  
    elif period <= 1:
        return n * period
    else:
        return period

# Fits a curve to the series given as input and returns the y-values of the curve
def spline(x, y, period=None):
    p = get_param(0.5, get_period(period, len(x)))
    yi = csaps(x, y, x, smooth=p)
    return yi