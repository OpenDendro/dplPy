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