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

# Date: 5/27/2022
# Author: Ifeoluwa Ale
# Title: detrend.py
# Description: Detrends a given series, first by fitting data 
#              to a spline curve and then by calculating residuals
# example usage:
# >>> import dplpy as dpl 
# >>> data = dpl.readers("../tests/data/csv/file.csv")
# >>> dpl.detrend(data['Name of series'])

from tkinter import Y
import pandas as pd
import matplotlib.pyplot as plt
from smoothingspline import spline
import emdecomp as emd
from autoreg import ar_func
import curvefit

# Takes a series as input and fits it to a spline, then detrends it
# by calculating residuals
# In the future, will  add functionality that allows for different
# types of curve fits, and choice of detrending by residual or
# differences
def detrend(data, fit="spline", method="residual"):
    nullremoved_data = data.dropna()
    if fit=="spline":
        yi = spline(nullremoved_data)
        residual(nullremoved_data, yi)

    #Empirical Mode Decomposition (EMD)
    elif fit=="emd":
        imfs = emd.emd(nullremoved_data)
        imfs_p = emd.phase_spectrum(imfs)
        mis = emd.phase_mi(imfs_p)
        stochastic_component, yi = emd.divide_signal(nullremoved_data, imfs, mis, cutoff=0.5)
        residual(nullremoved_data, yi)

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