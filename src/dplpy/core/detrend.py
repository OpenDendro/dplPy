__copyright__ = """
   dplPy for tree ring width time series analyses
   Copyright (C) 2025  OpenDendro

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

# Note: tkinter import Y appears to be unused and can be removed
from typing import Union, Optional
import pandas as pd
import matplotlib.pyplot as plt
from ..utils.smoothingspline import spline
from ..analysis.autoreg import ar_func
from ..utils import curvefit

def detrend(data: pd.DataFrame, fit: str = "spline", method: str = "residual", plot: bool = True, period: Optional[int] = None) -> pd.DataFrame:
    """
    Detrend tree ring width data by fitting curves and calculating residuals or differences.
    
    This function removes long-term trends from tree ring width series to create
    dimensionless ring width indices. It supports multiple curve fitting methods
    and detrending approaches, commonly used in dendrochronology for standardizing
    growth series and reducing the influence of tree age and environmental trends.
    
    Parameters
    ----------
    data : pandas.DataFrame or pandas.Series
        Input tree ring width data. If DataFrame, each column represents a 
        different tree ring series with years as index. If Series, represents
        a single tree ring series.
    fit : str, optional
        Method for curve fitting, by default "spline". Supported options:
        - "spline": Smoothing spline fit (default, flexible)
        - "ModNegEx": Modified negative exponential curve
        - "Hugershoff": Hugershoff curve (biological growth model)
        - "linear": Linear regression trend
        - "horizontal": Horizontal line (mean value)
    method : str, optional
        Detrending method, by default "residual". Supported options:
        - "residual": Calculate ratios (original/fitted), preserves variance structure
        - "difference": Calculate differences (original-fitted), preserves absolute values
    plot : bool, optional
        Whether to display diagnostic plots showing original data with fitted curves
        and resulting detrended series, by default True.
    period : int or None, optional
        Period parameter for spline fitting. Only used when fit="spline".
        Controls the smoothing of the spline curve. If None, uses default spline parameters.
    
    Returns
    -------
    dict or pandas.Series
        If input is DataFrame, returns dict with series names as keys and 
        detrended Series as values. If input is Series, returns single 
        detrended Series with original index preserved.
    
    Raises
    ------
    ValueError
        If unsupported fit method or detrending method is specified.
    TypeError
        If input data is neither pandas DataFrame nor pandas Series.
    
    Notes
    -----
    Detrending is a fundamental step in dendrochronological analysis that:
    
    1. Removes biological age-related trends in tree growth
    2. Reduces the influence of stand dynamics and competition
    3. Emphasizes climate-related variation in ring widths
    4. Standardizes series for chronology development
    
    The choice of curve fitting method depends on the growth characteristics:
    - Splines: Most flexible, good for various growth patterns
    - ModNegEx: Suitable for series with exponential decay trends
    - Hugershoff: Appropriate for series showing biological growth curves
    - Linear: For series with linear trends
    - Horizontal: When no trend removal is desired (mean normalization only)
    
    References
    ----------
    Cook, E. R., & Peters, K. (1981). The smoothing spline: a new approach to 
    standardizing forest interior tree-ring width series for dendroclimatic studies. 
    Tree-ring bulletin, 41, 45-53.
    
    Examples
    --------
    >>> import dplpy as dpl
    >>> # Load sample data
    >>> data = dpl.readers('../tests/data/rwl/ca533.rwl')
    >>> 
    >>> # Detrend all series with default spline fit
    >>> detrended = dpl.detrend(data)
    >>> 
    >>> # Detrend single series with linear fit, no plots
    >>> single_detrended = dpl.detrend(data['CAM011'], fit='linear', plot=False)
    >>> 
    >>> # Use difference method instead of ratios
    >>> diff_detrended = dpl.detrend(data, method='difference')
    >>> 
    >>> # Custom spline with specific period
    >>> custom_detrended = dpl.detrend(data, fit='spline', period=50)
    
    See Also
    --------
    detrend_series : Internal function for detrending individual series
    residual : Function for calculating ratios (residual method)
    difference : Function for calculating differences
    """
    if isinstance(data, pd.DataFrame):
        res = {}
        for column in data.columns:
            res[column] = detrend_series(data[column], column, fit, method, plot, period=None)
        return res
    elif isinstance(data, pd.Series):
        return detrend_series(data, data.name, fit, method, plot)
    else:
        return TypeError("argument should be either pandas dataframe or pandas series.")

def detrend_series(data, series_name, fit, method, plot, period=None):
    """
    Detrend a single tree ring width series.
    
    Internal function that processes individual tree ring series by fitting
    a specified curve type and calculating detrended values using either
    residuals (ratios) or differences.
    
    Parameters
    ----------
    data : pandas.Series
        Tree ring width series with year index and numeric values.
    series_name : str
        Name of the series for plot titles and identification.
    fit : str
        Curve fitting method ('spline', 'ModNegEx', 'Hugershoff', 'linear', 'horizontal').
    method : str
        Detrending method ('residual' or 'difference').
    plot : bool
        Whether to generate diagnostic plots.
    period : int or None, optional
        Period parameter for spline fitting, by default None.
    
    Returns
    -------
    pandas.Series
        Detrended series with original year index and detrended values.
    
    Notes
    -----
    This function:
    1. Removes NaN values before processing
    2. Applies the specified curve fitting method
    3. Calculates detrended values using the specified method
    4. Optionally generates diagnostic plots showing original vs fitted data
       and the resulting detrended series
    
    The diagnostic plots help assess the appropriateness of the chosen
    fitting method and detrending approach.
    """
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
        print()
        return ValueError("unsupported keyword for curve-fit type. See documentation for more info.")
    
    if method == "residual":
        detrended_data = residual(y, yi)
    elif method == "difference":
        detrended_data = difference(y, yi)
    else:
        # give error message for unsupported detrending method
        print()
        return ValueError("unsupported keyword for detrending method. See documentation for more info.")
    
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

    return pd.Series(detrended_data, index=x)

def residual(y, yi):
    """
    Calculate ring width indices using the residual (ratio) method.
    
    This function computes dimensionless ring width indices by dividing
    original ring width values by fitted curve values. This method preserves
    the relative variance structure of the original series.
    
    Parameters
    ----------
    y : array-like
        Original ring width measurements.
    yi : array-like
        Fitted curve values corresponding to the original measurements.
    
    Returns
    -------
    numpy.ndarray
        Ring width indices calculated as y/yi. Values around 1.0 indicate
        growth close to the expected trend, values >1.0 indicate above-trend
        growth, and values <1.0 indicate below-trend growth.
    
    Notes
    -----
    The residual method is preferred in dendrochronology because:
    - It produces dimensionless indices suitable for averaging across series
    - It preserves the variance structure relative to the mean
    - It handles heteroscedastic variance patterns common in tree growth
    - Results are multiplicative rather than additive
    
    Examples
    --------
    >>> y_orig = np.array([100, 120, 80, 110])
    >>> y_fitted = np.array([110, 115, 90, 105])
    >>> indices = residual(y_orig, y_fitted)
    >>> print(indices)  # [0.91, 1.04, 0.89, 1.05]
    """
    return y/yi


def difference(y, yi):
    """
    Calculate detrended values using the difference method.
    
    This function computes detrended values by subtracting fitted curve values
    from original ring width measurements. This method preserves the absolute
    scale of variations.
    
    Parameters
    ----------
    y : array-like
        Original ring width measurements.
    yi : array-like
        Fitted curve values corresponding to the original measurements.
    
    Returns
    -------
    numpy.ndarray
        Detrended values calculated as y - yi. Positive values indicate
        above-trend growth, negative values indicate below-trend growth,
        and values near zero indicate growth close to the expected trend.
    
    Notes
    -----
    The difference method:
    - Preserves absolute measurement units (e.g., millimeters)
    - May be preferred for studies focusing on absolute growth variations
    - Results are additive rather than multiplicative
    - Less commonly used in traditional dendrochronology compared to ratios
    
    Caution
    -------
    When using the difference method, be aware that:
    - Series with different mean values may not combine well in chronologies
    - Variance may not be stabilized across the length of the series
    - May not be suitable for standardized chronology development
    
    Examples
    --------
    >>> y_orig = np.array([100, 120, 80, 110])
    >>> y_fitted = np.array([110, 115, 90, 105])
    >>> differences = difference(y_orig, y_fitted)
    >>> print(differences)  # [-10, 5, -10, 5]
    """
    return y - yi