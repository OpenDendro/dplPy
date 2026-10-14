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
from ._plot_style import style_axes, finalize_font, ACCENT, ACCENT_WARM, NEUTRAL
from .smoothingspline import spline, get_period
from .agedepspline import ads
from . import curvefit

def detrend(data: pd.DataFrame | pd.Series, fit="Spline", method="ratio",
            plot=True, period=None, nyrs0=50, pos_slope=False, f=0.5,
            verbose=False, return_info=False):
    """Detrend a series or dataframe to ring-width indices.

    Fit a growth curve to each series (a smoothing spline by default) and form the
    ring-width index as the ratio (``method="ratio"``, division, the default) or the
    difference (``method="difference"``, subtraction) of the data to the curve.

    ``fit`` chooses the CURVE and ``method`` chooses the ARITHMETIC. This differs
    from dplR, where ``method`` selects the curve; passing a curve name to
    ``method`` raises a helpful error.

    Parameters
    ----------
    data : pandas dataframe or series
        ring widths from ``dpl.readers``, or a single series taken from one.
    fit : str or list, default "Spline"
        the growth curve to fit (case-insensitive; dplR's names are canonical):
        ``"Spline"`` (smoothing spline), ``"AgeDepSpline"`` (age-dependent spline),
        ``"ModNegExp"`` and ``"ModHugershoff"`` (negative-exponential and Hugershoff
        curves fit by nonlinear least squares, with a linear->mean fallback),
        ``"NegExp"`` and ``"Hugershoff"`` (the same curves by ARSTAN's deterministic
        fit), ``"GeneralExp"``, ``"LinearAny"`` (any-slope line), ``"LinearNegative"``
        (non-positive-slope line, else the mean), and ``"Mean"`` (horizontal line at
        the series mean). A list of curves detrends by each and returns a comparison:
        a Series gives a DataFrame with one column per curve, a DataFrame gives a dict
        of them (not combinable with ``return_info``).
    method : {"ratio", "difference"}, default "ratio"
        how the index is formed: ``"ratio"`` divides the data by the curve,
        ``"difference"`` subtracts it.
    plot : boolean, default True
        draw the fitted curve and the detrended index.
    period : int or float, default None
        smoothing wavelength for ``fit="Spline"`` (``get_period`` rule): a fraction of
        length in ``(0, 1]``, a percent-of-length spline if negative, or a fixed
        wavelength in years if > 1. None uses the dplR default (two thirds of length).
    nyrs0 : int, default 50
        initial spline stiffness for ``fit="AgeDepSpline"`` (see ``dpl.ads``).
    pos_slope : boolean, default False
        allow a positive slope at the old-age end (the ``ModNegExp`` linear fallback
        and ``AgeDepSpline``). dplR uses ``pos.slope=FALSE``.
    f : float, default 0.5
        spline frequency-response amplitude at ``period`` (``fit="Spline"`` only).
    verbose : boolean, default False
        print, per series, the curve fitted and the arithmetic used.
    return_info : boolean, default False
        if True, return a dict (like dplR's ``return.info``) instead of just the data::

            {"rwi":        the detrended series (same shape as the input),
             "curves":     the fitted detrending curves,
             "model_info": per-series description of the fit actually used,
             "data_info":  per-series {n_zeros, zero_years} of the input,
             "dirty_dog":  True if any curve went non-positive or fell back to the mean}

    Returns
    -------
    data : pandas dataframe or series
        the detrended data.

    Examples
    --------
    >>> import dplpy as dpl
    >>> data = dpl.readers("../tests/data/csv/file.csv")
    >>> dpl.detrend(data)                     # all series in a dataframe
    >>> dpl.detrend(data["SeriesA"], fit="ModNegExp", method="difference")

    Notes
    -----
    Interior ``NaN`` are dropped before fitting. The position-based fits (every curve
    except ``Spline`` and ``Mean``) reconstruct cambial age by counting the surviving
    rings, so an interior gap compresses the age axis after it -- usually a minor
    distortion, but for large gaps prefer ``fit="Spline"`` or infill first with
    ``dpl.fill_internal``. ``detrend`` warns when a position-based fit meets a series
    with interior gaps. ``Spline`` (fit over calendar years) and ``Mean``
    (order-invariant) are unaffected.

    References
    ----------
    .. [1] https:/opendendro.org/dplpy-man/#detrend
    """
    method = _normalize_method(method)

    # D9: fit may be a LIST of curve types -> return one detrend per method
    # (mirrors dplR's method vector). For a Series this is a DataFrame with one
    # column per method; for a DataFrame it is a dict {series: DataFrame}. A
    # single-element list collapses to the ordinary scalar-fit behavior.
    if isinstance(fit, (list, tuple)):
        methods = list(dict.fromkeys(_normalize_fit(m) for m in fit))  # dedup, ordered
        if len(methods) == 1:
            fit = methods[0]
        else:
            if return_info:
                raise ValueError("return_info=True is not supported together with "
                                 "a list of fit methods; request one method at a time.")
            if isinstance(data, pd.Series):
                cols = {m: detrend_series(data, m, method, plot, period, nyrs0,
                                          pos_slope, f, verbose) for m in methods}
                return pd.DataFrame(cols)
            elif isinstance(data, pd.DataFrame):
                return {col: detrend(data[col], methods, method, plot, period,
                                     nyrs0, pos_slope, f, verbose)
                        for col in data.columns}
            else:
                raise TypeError("argument should be either pandas dataframe or pandas series.")

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
def _nonpositive_spline_fallback(yi, y, method, fit, series_name, return_info):
    """A flexible spline fit can dip to <= 0. Division ('ratio') detrending cannot
    divide by a non-positive curve, so fall back to the series mean -- but ONLY in
    ratio mode. 'difference' detrending subtracts the curve, which handles a
    non-positive fit perfectly well (e.g. log-transformed or isotope series), so the
    spline is kept. Either way the fallback, when it happens, is announced.
    Returns (yi, model_info_or_None, fell_back)."""
    if np.any(yi <= 0) and method == "ratio":
        warnings.warn("Fits from fit='" + str(fit) + "' are not all positive for "
                      + str(series_name) + "; detrending by the mean instead "
                      "(ratio/division cannot use a non-positive curve; use "
                      "method='difference' to keep the fit).\n")
        yi = np.full_like(y, np.mean(y))
        info = {"method": "Mean", "mean": float(np.mean(y))} if return_info else None
        return yi, info, True
    return yi, None, False


def detrend_series(data: pd.Series, fit, method, plot, period=None,
                   nyrs0=50, pos_slope=False, f=0.5, verbose=False,
                   return_info=False):
    series_name = data.name
    method = _normalize_method(method)      # idempotent; canonical passes through
    fit = _normalize_fit(fit)               # idempotent; canonical passes through

    # Interior gaps (NaN between the first and last measured ring) are dropped
    # before fitting, so they can never corrupt a fit. But every fit except
    # those in _GAP_SAFE_FITS is fitted against reconstructed ring position
    # 1..n over the surviving rings, so an interior gap compresses the cambial-
    # age axis after it -- benign for a smooth curve but not exact. Warn and let
    # the user decide (switch fit, or infill); dplPy never fills silently.
    # See docs/guide/internal-gaps.md.
    if fit not in _GAP_SAFE_FITS:
        _fvi, _lvi = data.first_valid_index(), data.last_valid_index()
        _n_gaps = int(data.loc[_fvi:_lvi].isna().sum()) if _fvi is not None else 0
        if _n_gaps > 0:
            warnings.warn(
                "detrend: series '%s' has %d interior missing year(s) and "
                "fit='%s' is fitted against reconstructed ring position, so the "
                "gap compresses the cambial-age axis after it (benign for a "
                "smooth curve but not exact). Use fit='Spline' (spans the gap in "
                "calendar time) or infill first with dpl.fill_internal "
                "(e.g. fill='ARSTAN'); see the 'Internal gaps' guide."
                % (series_name, _n_gaps, fit))

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
    # (0 / curve = 0), as dplR's detrend.series does.
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
    # ModNegExp, NegExp, ModHugershoff, Hugershoff, GeneralExp, Mean, LinearAny,
    # LinearNegative. model_info records what was actually fit (mirroring dplR's
    # return.info$model.info), collected only when needed.
    model_info = None
    if fit == "Spline":
        yi = spline(x, y, period, f)
        yi, mi_fb, fell_back = _nonpositive_spline_fallback(
            yi, y, method, "Spline", series_name, return_info)
        if return_info:
            model_info = mi_fb if fell_back else {
                "method": "Spline", "nyrs": get_period(period, len(x)), "f": f}
    elif fit == "ModNegExp":
        if return_info:
            yi, model_info = curvefit.mod_neg_exp(x, y, pos_slope, series_name,
                                                  info=True)
        else:
            yi = curvefit.mod_neg_exp(x, y, pos_slope, series_name)
    elif fit == "NegExp":
        if return_info:
            yi, model_info = curvefit.neg_exp_arstan(x, y, pos_slope, series_name,
                                                     info=True)
        else:
            yi = curvefit.neg_exp_arstan(x, y, pos_slope, series_name)
    elif fit == "ModHugershoff":
        if return_info:
            yi, model_info = curvefit.mod_hugershoff(x, y, pos_slope, series_name,
                                                     info=True)
        else:
            yi = curvefit.mod_hugershoff(x, y, pos_slope, series_name)
    elif fit == "Hugershoff":
        if return_info:
            yi, model_info = curvefit.hugershoff_arstan(x, y, series_name, info=True)
        else:
            yi = curvefit.hugershoff_arstan(x, y, series_name)
    elif fit == "GeneralExp":
        if return_info:
            yi, model_info = curvefit.general_exp(x, y, series_name, info=True)
        else:
            yi = curvefit.general_exp(x, y, series_name)
    elif fit == "LinearAny":
        yi = curvefit.linear(x, y)
        if return_info:
            model_info = curvefit._line_info(x, yi)
    elif fit == "LinearNegative":
        if return_info:
            yi, model_info = curvefit.linear_neg(x, y, series_name, info=True)
        else:
            yi = curvefit.linear_neg(x, y, series_name)
    elif fit == "Mean":
        yi = curvefit.horizontal(x, y)
        if return_info:
            model_info = {"method": "Mean", "mean": float(np.mean(y))}
    elif fit == "AgeDepSpline":
        yi = ads(y, nyrs0=nyrs0, pos_slope=pos_slope)
        # An age-dependent spline can dip non-positive (rare); fall back to the
        # series mean only for ratio detrending (difference mode keeps the fit).
        yi, mi_fb, fell_back = _nonpositive_spline_fallback(
            yi, y, method, "AgeDepSpline", series_name, return_info)
        if return_info:
            model_info = mi_fb if fell_back else {
                "method": "Age-Dep Spline", "nyrs0": nyrs0, "pos_slope": pos_slope}

    if method == "ratio":
        detrended_data = ratio(y, yi)
    else:  # method == "difference"
        detrended_data = difference(y, yi)
    
    if plot:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 3.2))

        # left: raw ring width (grey) with the fitted growth curve (warm)
        axes[0].plot(x, y, color=NEUTRAL, linewidth=0.9, label="ring width")
        axes[0].plot(x, yi, color=ACCENT_WARM, linewidth=2.0,
                     label=fit + " fit")
        axes[0].set_xlabel("Year")
        axes[0].set_ylabel("Ring width")
        axes[0].set_title(str(series_name) + " — curve fit", fontsize=10)
        axes[0].legend(frameon=False, fontsize=8)

        # right: the detrended index; a ratio index oscillates about 1, a
        # difference index about 0 -- mark the relevant reference line
        axes[1].axhline(1.0 if method == "ratio" else 0.0,
                        color="0.8", linewidth=0.8)
        axes[1].plot(x, detrended_data, color=ACCENT, linewidth=0.9)
        axes[1].set_xlabel("Year")
        axes[1].set_ylabel("Index")
        axes[1].set_title(str(series_name) + " — detrended by " + method,
                          fontsize=10)

        for a in axes:
            style_axes(a, xgrid=True, ygrid=True)
        fig.tight_layout()
        finalize_font(fig)

        plt.show()

    # reindex the detrended values (on their measured years) back onto the full
    # series index, so leading, trailing, and interior gap years stay NaN
    rwi = pd.Series(detrended_data, index=pd.Index(data=x, name="Year"),
                    name=series_name).reindex(data.index)
    if not return_info:
        return rwi
    curve = pd.Series(yi, index=pd.Index(data=x, name="Year"),
                      name=series_name).reindex(data.index)
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
# dplR's names both resolve here. Note there are TWO Hugershoff curves:
# 'Hugershoff' = Ed Cook's ARSTAN log-linearised closed-form fit, and
# 'ModHugershoff' = dplR's modified nonlinear-least-squares (nls) fit.
_FIT_CANON = {
    "spline": "Spline",
    "agedepspline": "AgeDepSpline",
    "modnegexp": "ModNegExp", "modnegex": "ModNegExp",
    "negexp": "NegExp", "negex": "NegExp",
    "modhugershoff": "ModHugershoff",
    "hugershoff": "Hugershoff",
    "generalexp": "GeneralExp", "genexp": "GeneralExp",
    "mean": "Mean", "horizontal": "Mean",
    # 'Linear' kept as an alias of the any-slope line for back-compatibility;
    # ARSTAN's two linear options are 'LinearAny' (opt 4) and 'LinearNegative'
    # (opt 5, slope constrained <= 0).
    "linear": "LinearAny", "linearany": "LinearAny",
    "linearnegative": "LinearNegative", "linearneg": "LinearNegative",
}
# Fits that are NOT distorted by an interior gap: 'Spline' is a smoothing
# spline over the real calendar years (spans a gap correctly in time) and
# 'Mean' is order-invariant. Every other fit is built on reconstructed ring
# position 1..n over the surviving rings, so a gap compresses the age axis.
_GAP_SAFE_FITS = frozenset({"Spline", "Mean"})

_FIT_OPTIONS = ("Spline, AgeDepSpline, ModNegExp, NegExp, ModHugershoff, "
                "Hugershoff, GeneralExp, Mean, LinearAny, LinearNegative")


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
