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
# Title: detrend.py
# Description: Detrends a given series or data frame, first by fitting data to curve(s),
#              with spline(s) as the default, and then by forming the ring-width index
#              as a ratio (division) or a difference (subtraction) of the data to the
#              fitted curve (ratio by default). Note: "residual" is deliberately NOT used
#              here for the division result -- in dplPy "residual" refers only to the
#              residual (AR-prewhitened) chronology produced by chron()/chron_ars().

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .smoothingspline import spline
from .agedepspline import ads
from . import curvefit

def detrend(data: pd.DataFrame | pd.Series, fit="spline", method="ratio",
            plot=True, period=None, nyrs0=50, pos_slope=False):
    """Detrends a given series or dataframe
    
    Extended Summary
    ----------------
    Detrends a given series or dataframe, first by fitting data to curve(s),
    with 'spline' as the default, and then by forming the ring-width index as a
    ratio ('ratio', i.e. division -- the default) or a difference ('difference',
    i.e. subtraction) of the data to the fitted curve.
    Other supported curve fitting methods are 'ModNegex' (modified negative exponential),
    'Hugershoff', 'linear', 'horizontal'.
                  
    Parameters
    ----------
    data: pandas dataframe or series
        a data frame loaded using dpl.readers(), or a series extracted from such a datafame.
    fit: str, default spline
        fitting method of curve. can be 'horizontal', 'Hugershoff', 'linear', 'ModNegex' (modified negative exponential), 'spline', and 'AgeDepSpline' (age-dependent spline).
    method : str, default 'ratio'
        how the ring-width index is formed from the data and the fitted curve.
        'ratio' (equivalently 'division') divides the data by the curve;
        'difference' subtracts the curve from the data. This mirrors dplR's
        detrend(), where division is the default and 'difference=TRUE' subtracts.
        The former spelling 'residual' for the division result is deprecated --
        in dplPy 'residual' now refers only to the residual (AR-prewhitened)
        chronology -- and is accepted for now as an alias of 'ratio' with a
        warning.
    plot : boolean, default True
        flag indicating whether or not to plot the results.
    nyrs0 : int, default 50
        initial spline stiffness for fit='AgeDepSpline' (see dpl.ads()).
    pos_slope : boolean, default False
        for fit='AgeDepSpline', whether to allow a positive slope at the end of
        the series (dplR's detrend uses pos.slope=FALSE).
    
    Returns
    -------
    data: pandas dataframe or series of detrended data.
    
    Examples
    --------
    >>> import dplpy as dpl
    >>> data = dpl.readers("../tests/data/csv/file.csv")
    >>> dpl.detrend(data) # Detrends all series in a dataframe
    >>> dpl.detrend(data["SeriesA"]) # Detrends only SeriesA
    >>> dpl.detrend(data["SeriesA"], fit="ModNegex", method="difference", plot=True)
    
    References
    ----------
    .. [1] https:/opendendro.org/dplpy-man/#detrend
         
    """
    method = _normalize_method(method)

    if isinstance(data, pd.DataFrame):
        res = pd.DataFrame(index=pd.Index(data.index))
        to_add = [res]
        for column in data.columns:
            to_add.append(detrend_series(data[column], fit, method, plot, period,
                                         nyrs0, pos_slope))
        output_df = pd.concat(to_add, axis=1)
        return output_df.rename_axis(data.index.name)

    elif isinstance(data, pd.Series):
        return detrend_series(data, fit, method, plot, period, nyrs0, pos_slope)
    else:
        raise TypeError("argument should be either pandas dataframe or pandas series.")

# Takes a series as input and by default fits it to a spline, then 
# detrends it by calculating residuals
# Can specify what type of alternate curve-fits, or if the user
# would like to detrend by using differences
# Need to add series names to the top of the plots, and display the plots side by side
def detrend_series(data: pd.Series, fit, method, plot, period=None,
                   nyrs0=50, pos_slope=False):
    series_name = data.name
    method = _normalize_method(method)      # idempotent; canonical passes through
    nullremoved_data = data.dropna()
    x = nullremoved_data.index.to_numpy()
    y = nullremoved_data.to_numpy(dtype=float).copy()

    # dplR's detrend.series recodes zero ring-widths to 0.001 before fitting the
    # curve and dividing (see detrend.series.R: "y2[y2 == 0] <- 0.001"). A zero
    # is a locally-absent ring -- a real, dated near-zero-growth year -- so this
    # keeps it as a small positive index instead of collapsing it to an exact 0
    # (0 / curve = 0), which also matches dplR's RWI to machine precision.
    y[y == 0] = 0.001

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
    elif fit == "AgeDepSpline":
        yi = ads(y, nyrs0=nyrs0, pos_slope=pos_slope)
        # dplR: if the age-dependent spline is not all-positive (very rare),
        # fall back to detrending by the series mean.
        if np.any(yi <= 0):
            warnings.warn("Fits from fit='AgeDepSpline' are not all positive for "
                          + str(series_name) + "; detrending by the mean instead.\n")
            yi = np.full_like(y, np.mean(y))
    else:
        # give error message for unsupported curve fit
        raise ValueError("unsupported keyword for curve-fit type. See documentation for more info.")
    
    if method == "ratio":
        detrended_data = ratio(y, yi)
    else:  # method == "difference"
        detrended_data = difference(y, yi)
    
    if plot:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(7,3))
    
        axes[0].plot(x, y, "k-", x, yi, "r-", linewidth=2)
        axes[0].set_xlabel('Year')
        axes[0].set_ylabel('Ring Width')
        axes[0].set_title(series_name + " curve fit to " + fit)

        axes[1].plot(x, detrended_data, 'k-')
        axes[1].set_xlabel('Year')
        axes[1].set_ylabel('Index')
        axes[1].set_title(series_name + " detrended by " + method)

        fig.tight_layout()
    
        plt.show()

    return pd.Series(detrended_data, index=pd.Index(data=x, name="Year"), name=series_name).combine(data, pick_first)

def pick_first(a, b):
    return a


# Canonical detrending-method names. Division = 'ratio' (aka 'division');
# subtraction = 'difference'. 'residual' is a deprecated alias for 'ratio' -- in
# dplPy 'residual' otherwise refers only to the residual (AR-prewhitened)
# chronology, so it is being retired from the detrending vocabulary.
def _normalize_method(method):
    m = str(method).strip().lower()
    if m in ("ratio", "division"):
        return "ratio"
    if m == "difference":
        return "difference"
    if m == "residual":
        warnings.warn(
            "detrend(method='residual') is deprecated and will be removed in a "
            "future release. 'residual' previously meant detrending by DIVISION; "
            "use method='ratio' (or 'division') for division, or "
            "method='difference' for subtraction. In dplPy 'residual' now refers "
            "only to the residual (AR-prewhitened) chronology. Treating "
            "method='residual' as 'ratio' for now.",
            FutureWarning, stacklevel=3)
        return "ratio"
    raise ValueError(
        "unsupported detrending method '" + str(method) + "'. Use 'ratio' "
        "(division, the default) or 'difference' (subtraction).")


# Detrends by dividing the original series by the fitted curve (ratio / division).
def ratio(y, yi):
    return y / yi


# Detrends by subtracting the fitted curve from the original series (difference).
def difference(y, yi):
    return y - yi
