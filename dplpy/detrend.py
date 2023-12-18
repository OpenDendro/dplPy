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

# Date: 11/1/2022
# Author: Ifeoluwa Ale
# Title: detrend.py
# Description: Detrends a given series or data frame, first by fitting data to curve(s), 
#              with spline(s) as the default, and then by calculating residuals or differences 
#              compared to the original data (residuals by default).
# example usage:
# >>> import dplpy as dpl 
# >>> data = dpl.readers("../tests/data/csv/file.csv")
# >>> dpl.detrend(data)
# >>> dpl.detrend(data['Name of series'])

from tkinter import Y
import pandas as pd
import matplotlib.pyplot as plt
from smoothingspline import spline
from autoreg import ar_func
import curvefit

def detrend(data: pd.DataFrame | pd.Series, fit="spline", method="residual", plot=True, period=None):
    if isinstance(data, pd.DataFrame):
        res = pd.DataFrame(index=pd.Index(data.index))
        to_add = [res]
        for column in data.columns:
            to_add.append(detrend_series(data[column], column, fit, method, plot, period))
        output_df = pd.concat(to_add, axis=1)
        return output_df.rename_axis(data.index.name)
    
    elif isinstance(data, pd.Series):
        return detrend_series(data, data.name, fit, method, plot, period)
    else:
        raise TypeError("argument should be either pandas dataframe or pandas series.")

# Takes a series as input and by default fits it to a spline, then 
# detrends it by calculating residuals
# Can specify what type of alternate curve-fits, or if the user
# would like to detrend by using differences
# Need to add series names to the top of the plots, and display the plots side by side
def detrend_series(data, series_name, fit, method, plot, period=None):
    nullremoved_data = data.dropna()
    x = nullremoved_data.index.to_numpy()
    y = nullremoved_data.to_numpy()

    if fit == "spline":
        yi = spline(x, y, period)
    elif fit == "ModNegEx":
        yi = curvefit.negex(x, y)
    elif fit == "Hugershoff":
        yi = curvefit.hugershoff(x, y)
    elif fit == "linear":
        yi = curvefit.linear(x, y)
    elif fit == "horizontal":
        yi = curvefit.horizontal(x, y)
    else:
        # give error message for unsupported curve fit
        raise ValueError("unsupported keyword for curve-fit type. See documentation for more info.")
    
    if method == "residual":
        detrended_data = residual(y, yi)
    elif method == "difference":
        detrended_data = difference(y, yi)
    else:
        # give error message for unsupported detrending method
        raise ValueError("unsupported keyword for detrending method. See documentation for more info.")
    
    if plot:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(7,3))
    
        axes[0].plot(x, y, "k-", x, yi, "r-", linewidth=2)
        axes[0].set_xlabel('Year')
        axes[0].set_ylabel('Length')
        axes[0].set_title(series_name + " curve fit to " + fit)

        axes[1].plot(x, detrended_data, 'k-')
        axes[1].set_xlabel('Year')
        axes[1].set_ylabel('Index')
        axes[1].set_title(series_name + " detrended by " + method)

        fig.tight_layout()
    
        plt.show()

    return pd.Series(detrended_data, index=x, name=series_name)

# Detrends by finding ratio of original series data to curve data
def residual(y, yi):
    return y/yi


# Detrends by finding difference between original series data and curve fit data
def difference(y, yi):
    return y - yi