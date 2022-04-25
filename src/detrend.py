__copyright__ = """
   dplPy for tree ring width time series analyses
   Copyright (C) 2022  OpenDendro

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

# Date: 4/11/2022
# Author: Ifeoluwa Ale
# Title: detrend.py
# Description: Detrends a the every series in a given dataset, first by fitting
#              data to a spline curve and then by finding residuals and differences
# example usage:
# >>> import dplpy as dpl 
# >>> data = dpl.readers("../tests/data/csv/file.csv")
# >>> dpl.detrend(data)

from tkinter import Y
import pandas as pd
import matplotlib.pyplot as plt
from smoothingspline import spline
import curvefit

# In the future, detrend will probably only take a series as input
# Currently takes a dataframe as input and detrends all the series in the
# dataframe
# probably will add another argument to this function that specifies what type
# of curve fit, and detrending method
def detrend(series_data):
    for series_name, data in series_data.items():
        nullremoved_data = data.dropna()
        # For testing curvefit.py
        try:
            print("Cubic smoothing spline")
            yi = spline(nullremoved_data)
            print("Hugershoff curve fit")
            yi = curvefit.hugershoff(nullremoved_data)
            print("NegEx curve fit")
            yi = curvefit.negex(nullremoved_data)
            print("Linear curve fit")
            yi = curvefit.linear(nullremoved_data)
            print("Horizontal curve fit")
            yi = curvefit.horizontal(nullremoved_data)
        except RuntimeError:
            print("Error")
    
# Detrends by finding ratio of original series data to curve data
def residual(series, yi):
    x = series.index.to_numpy()
    y = series.to_numpy()
    res = y/yi

    plt.plot(x, res, "-")
    plt.show()
    return res

# Detrends by finding difference between original series data and curve fit data
def difference(series, yi):
    x = series.index.to_numpy()
    y = series.to_numpy()
    res = y - yi

    plt.plot(x, res, "-")
    plt.show()
    return res