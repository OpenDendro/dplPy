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
from .smoothingspline import spline, get_period
from .agedepspline import ads
from . import curvefit

def detrend(data: pd.DataFrame | pd.Series, fit="Spline", method="ratio",
            plot=True, period=None, nyrs0=50, pos_slope=False, f=0.5,
            verbose=False, return_info=False):
    """Detrends a given series or dataframe

    Extended Summary
    ----------------
    Detrends a given series or dataframe, first by fitting data to a growth
    curve (a smoothing spline by default), and then by forming the ring-width
    index as a ratio ('ratio', i.e. division -- the default) or a difference
    ('difference', i.e. subtraction) of the data to the fitted curve.

    IMPORTANT -- ``fit`` chooses the CURVE, ``method`` chooses the ARITHMETIC.
    This differs from dplR, where ``method`` selects the curve and ``difference``
    selects the arithmetic. In dplPy the curve type is ``fit=`` and the
    ratio/difference choice is ``method=``; passing a curve name to ``method=``
    raises a helpful error.

    Parameters
    ----------
    data: pandas dataframe or series
        a data frame loaded using dpl.readers(), or a series extracted from such a datafame.
    fit: str, default 'Spline'
        the growth curve to fit. Accepted values (case-insensitive; dplR's names
        are used as the canonical spelling):
        'Spline' (smoothing spline), 'AgeDepSpline' (age-dependent spline),
        'ModNegExp' (modified negative exponential, with a linear->mean fallback),
        'ModHugershoff' (Hugershoff growth curve), 'Mean' (horizontal line at the
        series mean), and 'Linear' (best-fit straight line; a dplPy addition).
        Legacy dplPy spellings ('ModNegEx', 'Hugershoff', 'horizontal') are still
        accepted as case-insensitive aliases.
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
        for fit='ModNegExp' and fit='AgeDepSpline', whether to allow a positive
        slope (in the ModNegExp linear fallback, and at the end of the age-
        dependent spline). dplR's detrend uses pos.slope=FALSE.
    f : float, default 0.5
        frequency-response amplitude of the smoothing spline at the `period`
        wavelength (dplR's `f`): the fraction of amplitude retained there. Only
        used for fit='Spline'.
    verbose : boolean, default False
        print, per series, the curve fitted and the detrending arithmetic used.
        (Fallbacks in fit='ModNegExp' are also reported via warnings.)
    return_info : boolean, default False
        if True, return a dict (mirroring dplR's detrend return.info) instead of
        just the detrended data::

            {"rwi":        detrended series (same shape as the input),
             "curves":     the fitted detrending curves,
             "model_info": per-series fit description -- the method actually used
                           (e.g. 'NegativeExponential', 'Line', 'Mean', 'Spline')
                           and its parameters/coefficients,
             "data_info":  per-series {n_zeros, zero_years} of the input,
             "dirty_dog":  True if any series' curve went non-positive or fell
                           back to the series mean (dplR's "dirty dog" flag)}
    
    Returns
    -------
    data: pandas dataframe or series of detrended data.
    
    Examples
    --------
    >>> import dplpy as dpl
    >>> data = dpl.readers("../tests/data/csv/file.csv")
    >>> dpl.detrend(data) # Detrends all series in a dataframe
    >>> dpl.detrend(data["SeriesA"]) # Detrends only SeriesA
    >>> dpl.detrend(data["SeriesA"], fit="ModNegExp", method="difference", plot=True)
    
    References
    ----------
    .. [1] https:/opendendro.org/dplpy-man/#detrend
         
    """
    method = _normalize_method(method)
    fit = _normalize_fit(fit)

    if isinstance(data, pd.DataFrame):
        if return_info:
            rwis, curves, model_info, data_info = [], [], {}, {}
            dirty_dog = False
            for column in data.columns:
                r, c, mi, di, dd = detrend_series(
                    data[column], fit, method, plot, period, nyrs0, pos_slope,
                    f, verbose, return_info=True)
                rwis.append(r)
                curves.append(c)
                model_info[column] = mi
                data_info[column] = di
                dirty_dog = dirty_dog or dd
            base = pd.DataFrame(index=pd.Index(data.index))
            rwi_df = pd.concat([base] + rwis, axis=1).rename_axis(data.index.name)
            curve_df = pd.concat([base] + curves, axis=1).rename_axis(data.index.name)
            return {"rwi": rwi_df, "curves": curve_df, "model_info": model_info,
                    "data_info": data_info, "dirty_dog": dirty_dog}
        res = pd.DataFrame(index=pd.Index(data.index))
        to_add = [res]
        for column in data.columns:
            to_add.append(detrend_series(data[column], fit, method, plot, period,
                                         nyrs0, pos_slope, f, verbose))
        output_df = pd.concat(to_add, axis=1)
        return output_df.rename_axis(data.index.name)

    elif isinstance(data, pd.Series):
        if return_info:
            r, c, mi, di, dd = detrend_series(
                data, fit, method, plot, period, nyrs0, pos_slope, f, verbose,
                return_info=True)
            return {"rwi": r, "curves": c, "model_info": mi,
                    "data_info": di, "dirty_dog": dd}
        return detrend_series(data, fit, method, plot, period, nyrs0, pos_slope,
                              f, verbose)
    else:
        raise TypeError("argument should be either pandas dataframe or pandas series.")

# Takes a series as input and by default fits it to a spline, then 
# detrends it by calculating residuals
# Can specify what type of alternate curve-fits, or if the user
# would like to detrend by using differences
# Need to add series names to the top of the plots, and display the plots side by side
def detrend_series(data: pd.Series, fit, method, plot, period=None,
                   nyrs0=50, pos_slope=False, f=0.5, verbose=False,
                   return_info=False):
    series_name = data.name
    method = _normalize_method(method)      # idempotent; canonical passes through
    fit = _normalize_fit(fit)               # idempotent; canonical passes through
    nullremoved_data = data.dropna()
    x = nullremoved_data.index.to_numpy()
    y = nullremoved_data.to_numpy(dtype=float).copy()

    # zero ring-widths, recorded before the recode below (dplR reports the zero
    # years of the *input* series in return.info$data.info).
    zero_mask = (y == 0)
    data_info = {"n_zeros": int(zero_mask.sum()),
                 "zero_years": [int(v) for v in x[zero_mask]]}

    # dplR's detrend.series recodes zero ring-widths to 0.001 before fitting the
    # curve and dividing (see detrend.series.R: "y2[y2 == 0] <- 0.001"). A zero
    # is a locally-absent ring -- a real, dated near-zero-growth year -- so this
    # keeps it as a small positive index instead of collapsing it to an exact 0
    # (0 / curve = 0), which also matches dplR's RWI to machine precision.
    y[y == 0] = 0.001

    if verbose:
        detail = ""
        if fit == "Spline":
            detail = " (nyrs=%d, f=%g)" % (get_period(period, len(x)), f)
        elif fit == "AgeDepSpline":
            detail = " (nyrs0=%d, pos_slope=%s)" % (nyrs0, pos_slope)
        print("detrend: %s  fit=%s  method=%s  n=%d%s"
              % (series_name, fit, method, len(y), detail))

    # fit is already canonical (see _normalize_fit): Spline, AgeDepSpline,
    # ModNegExp, ModHugershoff, Mean, Linear. model_info records what was actually
    # fit (mirroring dplR's return.info$model.info), collected only when needed.
    model_info = None
    if fit == "Spline":
        yi = spline(x, y, period, f)
        if return_info:
            model_info = {"method": "Spline", "nyrs": get_period(period, len(x)),
                          "f": f}
    elif fit == "ModNegExp":
        if return_info:
            yi, model_info = curvefit.mod_neg_exp(x, y, pos_slope, series_name,
                                                  info=True)
        else:
            yi = curvefit.mod_neg_exp(x, y, pos_slope, series_name)
    elif fit == "ModHugershoff":
        if return_info:
            yi, model_info = curvefit.mod_hugershoff(x, y, pos_slope, series_name,
                                                     info=True)
        else:
            yi = curvefit.mod_hugershoff(x, y, pos_slope, series_name)
    elif fit == "Linear":
        yi = curvefit.linear(x, y)
        if return_info:
            slope = (yi[-1] - yi[0]) / (x[-1] - x[0]) if x[-1] != x[0] else 0.0
            model_info = {"method": "Line",
                          "coefs": {"intercept": float(yi[0] - slope * x[0]),
                                    "slope": float(slope)}}
    elif fit == "Mean":
        yi = curvefit.horizontal(x, y)
        if return_info:
            model_info = {"method": "Mean", "mean": float(np.mean(y))}
    elif fit == "AgeDepSpline":
        yi = ads(y, nyrs0=nyrs0, pos_slope=pos_slope)
        # dplR: if the age-dependent spline is not all-positive (very rare),
        # fall back to detrending by the series mean.
        if np.any(yi <= 0):
            warnings.warn("Fits from fit='AgeDepSpline' are not all positive for "
                          + str(series_name) + "; detrending by the mean instead.\n")
            yi = np.full_like(y, np.mean(y))
            if return_info:
                model_info = {"method": "Mean", "mean": float(np.mean(y))}
        elif return_info:
            model_info = {"method": "Age-Dep Spline", "nyrs0": nyrs0,
                          "pos_slope": pos_slope}

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

    rwi = pd.Series(detrended_data, index=pd.Index(data=x, name="Year"),
                    name=series_name).combine(data, pick_first)
    if not return_info:
        return rwi
    curve = pd.Series(yi, index=pd.Index(data=x, name="Year"),
                      name=series_name).combine(data, pick_first)
    # dplR's "dirty dog": a curve that went non-positive, or a fit that had to
    # fall back to the series mean.
    dirty = bool(np.any(yi <= 0)) or (
        model_info is not None and model_info.get("method") == "Mean"
        and fit != "Mean")
    return rwi, curve, model_info, data_info, dirty

def pick_first(a, b):
    return a


# Canonical curve-fit names use dplR's spelling. The map takes any accepted
# spelling (case-insensitive) to the canonical name; dplPy's older names and
# dplR's names both resolve here, so 'spline', 'Spline', 'ModNegExp', 'ModNegEx',
# 'modnegexp', 'Hugershoff', 'ModHugershoff', 'horizontal', 'Mean' all work.
_FIT_CANON = {
    "spline": "Spline",
    "agedepspline": "AgeDepSpline",
    "modnegexp": "ModNegExp", "modnegex": "ModNegExp",
    "modhugershoff": "ModHugershoff", "hugershoff": "ModHugershoff",
    "mean": "Mean", "horizontal": "Mean",
    "linear": "Linear",
}
_FIT_OPTIONS = "Spline, AgeDepSpline, ModNegExp, ModHugershoff, Mean, Linear"


def _normalize_fit(fit):
    """Resolve a curve-fit name to its canonical (dplR) spelling,
    case-insensitively. Raises with the full option list on an unknown name."""
    key = str(fit).strip().lower()
    if key in _FIT_CANON:
        return _FIT_CANON[key]
    raise ValueError(
        "unsupported curve-fit type '" + str(fit) + "'. Choose one of: "
        + _FIT_OPTIONS + " (case-insensitive; dplR's names are accepted).")


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
    if m in _FIT_CANON:
        # A curve name was passed to method= -- the classic dplR/dplPy mix-up
        # (dplR's `method` chooses the curve; dplPy's chooses the arithmetic).
        raise ValueError(
            "method='" + str(method) + "' looks like a curve type. In dplPy the "
            "curve is chosen with fit= (e.g. fit='" + _FIT_CANON[m] + "'); "
            "method= only selects 'ratio' (division) or 'difference' "
            "(subtraction).")
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
