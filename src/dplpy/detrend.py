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

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._plot_style import style_axes, finalize_font, ACCENT, ACCENT_WARM, NEUTRAL
from ._validate import (_warn, _alias_kwarg, _reject_kwargs, _normalize_choice,
                        _drop_allnan_columns, _carry_attrs)
from .smoothingspline import spline, get_period
from .agedepspline import ads
from . import curvefit


def _finish_figure(fig, mode=True):
    """Show a finished figure and release it.

    ``mode=True`` shows it and, under a non-interactive backend (where
    ``plt.show()`` is a no-op and figures would otherwise pile up in memory),
    closes it. ``mode="sequential"`` shows it without blocking, waits for the
    user to press Enter, then closes it so the next series' figure can be drawn.
    Shared by every plotting function in the detrending family (audit F2).
    """
    if mode == "sequential":
        plt.show(block=False)
        try:
            input("  [Enter] for the next figure ...")
        except (EOFError, OSError):
            pass
        plt.close(fig)
        return
    plt.show()
    if not matplotlib.is_interactive():
        plt.close(fig)


def _normalize_plot(plot):
    """detrend's plot switch: False, True (one figure per series, all left open
    under an interactive backend) or "sequential" (one at a time)."""
    if isinstance(plot, (bool, np.bool_, int, np.integer)):
        return bool(plot)
    if plot is None:
        return False
    return _normalize_choice(plot, ("sequential",), "plot")


def detrend(data: pd.DataFrame | pd.Series, fit="Spline", method="ratio",
            plot=False, spline_period=None, nyrs0=50, pos_slope=False, f=0.5,
            verbose=False, return_info=False, **kwargs):
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
        ring widths from ``dpl.readers``, or a single series taken from one. Series
        with no measured values are dropped with a warning (a lone empty Series
        returns all-NaN).
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
    plot : bool or "sequential", default False
        ``True`` draws one figure per series (the fitted curve and the detrended
        index) and leaves them all open under an interactive backend;
        ``"sequential"`` draws them one at a time, waiting for Enter between
        series; ``False`` (the default, as dplR's ``make.plot=FALSE`` for data
        frames) draws nothing. Figures are closed after display under a
        non-interactive backend.
    spline_period : int or float, default None
        smoothing wavelength for ``fit="Spline"`` (``get_period`` rule): a fraction of
        length in ``(0, 1]``, a percent-of-length spline if negative, or a fixed
        wavelength in years if > 1. None uses the dplR default (two thirds of length).
        The former name ``period=`` is accepted with a DeprecationWarning.
    nyrs0 : int, default 50
        initial spline stiffness for ``fit="AgeDepSpline"`` (see ``dpl.ads``).
    pos_slope : boolean, default False
        allow a positive slope at the old-age end: for ``AgeDepSpline`` the spline
        may rise at the end, and for ``ModNegExp``, ``NegExp`` and ``ModHugershoff``
        the straight-line fallback may slope upward (otherwise a rising line is
        rejected and the series mean is used). dplR uses ``pos.slope=FALSE``.
    f : float, default 0.5
        spline frequency-response amplitude at ``spline_period`` (``fit="Spline"``
        only).
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
        the detrended data, tagged ``attrs["kind"] = "rwi"`` and carrying the
        input's provenance attrs.

    Examples
    --------
    >>> import dplpy as dpl
    >>> data = dpl.readers("../tests/data/csv/file.csv")
    >>> dpl.detrend(data)                     # all series in a dataframe
    >>> dpl.detrend(data["SeriesA"], fit="ModNegExp", method="difference")

    Notes
    -----
    Interior ``NaN`` are unmeasured rings. They are left out of the fit and stay
    NaN in the index, but they still count as rings: the position-based fits
    (every curve except ``Spline`` and ``Mean``) are fitted on the true ring
    position, so a gap advances cambial age rather than compressing the axis
    after it (``AgeDepSpline`` carries the previous ring's value across the gap
    for its fit). ``Spline`` is fit over calendar years and ``Mean`` is
    order-invariant. To fill gaps instead, use ``dpl.fill_internal`` first.

    A series too short for the requested curve (fewer than 2 to 4 rings,
    depending on the fit) is detrended by its mean with a warning -- dplR's
    "dirty dog".

    References
    ----------
    .. [1] https://opendendro.github.io/dplPy/reference/detrending/#detrend
    """
    spline_period = _alias_kwarg(kwargs, "period", "spline_period", spline_period,
                                 "detrend")
    _reject_kwargs(kwargs)
    method = _normalize_method(method)
    plot = _normalize_plot(plot)

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
                cols = {m: detrend_series(data, m, method, plot, spline_period,
                                          nyrs0, pos_slope, f, verbose)
                        for m in methods}
                return pd.DataFrame(cols)
            elif isinstance(data, pd.DataFrame):
                data = _drop_allnan_columns(data, "data")
                return {col: detrend(data[col], methods, method, plot,
                                     spline_period, nyrs0, pos_slope, f, verbose)
                        for col in data.columns}
            else:
                raise TypeError("argument should be either pandas dataframe or pandas series.")

    fit = _normalize_fit(fit)

    if isinstance(data, pd.DataFrame):
        src = data
        data = _drop_allnan_columns(data, "data")
        base = pd.DataFrame(index=pd.Index(data.index))
        if return_info:
            rwis, curves, model_info, data_info = [], [], {}, {}
            dirty_dog = False
            for column in data.columns:
                r, c, mi, di, dd = detrend_series(
                    data[column], fit, method, plot, spline_period, nyrs0,
                    pos_slope, f, verbose, return_info=True)
                rwis.append(r)
                curves.append(c)
                model_info[column] = mi
                data_info[column] = di
                dirty_dog = dirty_dog or dd
            rwi_df = pd.concat([base] + rwis, axis=1).rename_axis(data.index.name)
            curve_df = pd.concat([base] + curves, axis=1).rename_axis(data.index.name)
            _carry_attrs(src, rwi_df, kind="rwi")
            return {"rwi": rwi_df, "curves": curve_df, "model_info": model_info,
                    "data_info": data_info, "dirty_dog": dirty_dog}
        to_add = [base]
        for column in data.columns:
            to_add.append(detrend_series(data[column], fit, method, plot,
                                         spline_period, nyrs0, pos_slope, f, verbose))
        output_df = pd.concat(to_add, axis=1).rename_axis(data.index.name)
        return _carry_attrs(src, output_df, kind="rwi")

    elif isinstance(data, pd.Series):
        if return_info:
            r, c, mi, di, dd = detrend_series(
                data, fit, method, plot, spline_period, nyrs0, pos_slope, f,
                verbose, return_info=True)
            return {"rwi": r, "curves": c, "model_info": mi,
                    "data_info": di, "dirty_dog": dd}
        return detrend_series(data, fit, method, plot, spline_period, nyrs0,
                              pos_slope, f, verbose)
    else:
        raise TypeError("argument should be either pandas dataframe or pandas series.")


def _nonpositive_spline_fallback(yi, y, method, fit, series_name, return_info):
    """A flexible spline fit can dip to <= 0. Division ('ratio') detrending cannot
    divide by a non-positive curve, so fall back to the series mean -- but ONLY in
    ratio mode. 'difference' detrending subtracts the curve, which handles a
    non-positive fit perfectly well (e.g. log-transformed or isotope series), so the
    spline is kept. Either way the fallback, when it happens, is announced.
    Returns (yi, model_info_or_None, fell_back)."""
    if np.any(yi <= 0) and method == "ratio":
        _warn("Fits from fit='" + str(fit) + "' are not all positive for "
              + str(series_name) + "; detrending by the mean instead "
              "(ratio/division cannot use a non-positive curve; use "
              "method='difference' to keep the fit).", stacklevel=5)
        yi = np.full_like(y, np.mean(y))
        info = {"method": "Mean", "mean": float(np.mean(y))} if return_info else None
        return yi, info, True
    return yi, None, False


def _ads_on_true_axis(y, t, nyrs0, pos_slope):
    """Age-dependent spline of ``y`` measured at ring positions ``t`` (1-based,
    strictly increasing). With interior gaps the span is infilled by
    carry-forward, the spline fit over the full span, and the curve read back at
    the measured positions -- so the stiffness grows with TRUE cambial age."""
    t = np.asarray(t, dtype=int)
    if len(t) == int(t[-1]):                     # contiguous: nothing to infill
        return ads(y, nyrs0=nyrs0, pos_slope=pos_slope)
    full = np.full(int(t[-1]), np.nan)
    full[t - 1] = y
    for m in range(1, len(full)):
        if np.isnan(full[m]):
            full[m] = full[m - 1]
    return np.asarray(ads(full, nyrs0=nyrs0, pos_slope=pos_slope))[t - 1]


def _empty_rwi(data, series_name, return_info):
    """detrend_series on a series with no measured values: all-NaN out, with a
    warning (audit B1)."""
    _warn("detrend: series '" + str(series_name) + "' has no measured values; "
          "returning NaN.", stacklevel=5)
    rwi = pd.Series(np.nan, index=data.index, name=series_name, dtype=float)
    if not return_info:
        return rwi
    return rwi, rwi.copy(), None, {"n_zeros": 0, "zero_years": []}, False


def detrend_series(data: pd.Series, fit, method, plot=False, spline_period=None,
                   nyrs0=50, pos_slope=False, f=0.5, verbose=False,
                   return_info=False, **kwargs):
    spline_period = _alias_kwarg(kwargs, "period", "spline_period", spline_period,
                                 "detrend_series")
    _reject_kwargs(kwargs)
    series_name = data.name
    method = _normalize_method(method)      # idempotent; canonical passes through
    fit = _normalize_fit(fit)               # idempotent; canonical passes through
    plot = _normalize_plot(plot)

    # Interior gaps (NaN between the first and last measured ring) are dropped
    # before fitting, so they can never corrupt a fit -- but they are unmeasured
    # RINGS, so the position-based fits are given the true ring position
    # t = year - first_year + 1 (a gap advances age; audit A1). The spline is fit
    # over calendar years and the mean is order-invariant.
    nullremoved_data = data.dropna()
    x = nullremoved_data.index.to_numpy()
    y = nullremoved_data.to_numpy(dtype=float).copy()
    n = len(y)
    if n == 0:
        return _empty_rwi(data, series_name, return_info)
    t = np.asarray(x - x[0] + 1, dtype=float)

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

    # B9: too few rings for the requested curve -> the series mean, announced
    # (dplR's "dirty dog").
    requested_fit = fit
    if n < _MIN_N[fit]:
        _warn("detrend: series '" + str(series_name) + "' has " + str(n)
              + " ring(s), too few for fit='" + fit + "' (needs "
              + str(_MIN_N[fit]) + "); detrending by the mean instead.",
              stacklevel=4)
        fit = "Mean"

    if verbose:
        detail = ""
        if fit == "Spline":
            detail = " (nyrs=%d, f=%g)" % (get_period(spline_period, n), f)
        elif fit == "AgeDepSpline":
            detail = " (nyrs0=%d, pos_slope=%s)" % (nyrs0, pos_slope)
        print("detrend: %s  fit=%s  method=%s  n=%d%s"
              % (series_name, fit, method, n, detail))

    # fit is canonical (see _normalize_fit). Every fitter is called with
    # info=True and model_info discarded unless return_info (audit E22).
    if fit == "Spline":
        yi = spline(x, y, spline_period, f)
        yi, mi_fb, fell_back = _nonpositive_spline_fallback(
            yi, y, method, "Spline", series_name, return_info)
        model_info = mi_fb if fell_back else {
            "method": "Spline", "nyrs": get_period(spline_period, n), "f": f}
    elif fit == "AgeDepSpline":
        yi = _ads_on_true_axis(y, t, nyrs0, pos_slope)
        # An age-dependent spline can dip non-positive (rare); fall back to the
        # series mean only for ratio detrending (difference mode keeps the fit).
        yi, mi_fb, fell_back = _nonpositive_spline_fallback(
            yi, y, method, "AgeDepSpline", series_name, return_info)
        model_info = mi_fb if fell_back else {
            "method": "Age-Dep Spline", "nyrs0": nyrs0, "pos_slope": pos_slope}
    elif fit == "ModNegExp":
        yi, model_info = curvefit.mod_neg_exp(x, y, pos_slope, series_name,
                                              info=True, t=t)
    elif fit == "NegExp":
        yi, model_info = curvefit.neg_exp_arstan(x, y, pos_slope, series_name,
                                                 info=True, t=t)
    elif fit == "ModHugershoff":
        yi, model_info = curvefit.mod_hugershoff(x, y, pos_slope, series_name,
                                                 info=True, t=t)
    elif fit == "Hugershoff":
        yi, model_info = curvefit.hugershoff_arstan(x, y, series_name, info=True, t=t)
    elif fit == "GeneralExp":
        yi, model_info = curvefit.general_exp(x, y, series_name, info=True, t=t)
    elif fit == "LinearAny":
        yi = curvefit.linear(x, y)
        model_info = curvefit._line_info(x, yi)
    elif fit == "LinearNegative":
        yi, model_info = curvefit.linear_neg(x, y, series_name, info=True)
    else:  # "Mean"
        yi = curvefit.horizontal(x, y)
        model_info = {"method": "Mean", "mean": float(np.mean(y))}

    if method == "ratio":
        detrended_data = ratio(y, yi)
    else:  # method == "difference"
        detrended_data = difference(y, yi)

    if plot:
        _plot_detrend(x, y, yi, detrended_data, fit, method, series_name, plot)

    # reindex the detrended values (on their measured years) back onto the full
    # series index, so leading, trailing, and interior gap years stay NaN
    rwi = pd.Series(detrended_data, index=pd.Index(data=x, name="Year"),
                    name=series_name).reindex(data.index)
    _carry_attrs(data, rwi, kind="rwi")
    if not return_info:
        return rwi
    curve = pd.Series(yi, index=pd.Index(data=x, name="Year"),
                      name=series_name).reindex(data.index)
    # dplR's "dirty dog": a curve that went non-positive, or a fit that had to
    # fall back to the series mean.
    dirty = bool(np.any(yi <= 0)) or (
        model_info is not None and model_info.get("method") == "Mean"
        and requested_fit != "Mean")
    return rwi, curve, model_info, data_info, dirty


def _plot_detrend(x, y, yi, detrended_data, fit, method, series_name, mode):
    """The two-panel detrend figure: raw widths + fitted curve, and the index."""
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
    _finish_figure(fig, mode)
    return fig


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

# Minimum number of rings each curve can be fit to without a degenerate or
# failing solve (audit B9): the spline needs 3 for a >= 2-year wavelength, the
# age-dependent spline 4 (its banded system), the nls curves their parameter
# count, a line 2. Shorter series fall back to the mean with a warning.
_MIN_N = {"Spline": 3, "AgeDepSpline": 4, "ModNegExp": 3, "NegExp": 2,
          "ModHugershoff": 4, "Hugershoff": 3, "GeneralExp": 3,
          "LinearAny": 2, "LinearNegative": 2, "Mean": 1}

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
        _warn(
            "detrend(method='residual') is deprecated and will be removed in a "
            "future release. 'residual' previously meant detrending by DIVISION; "
            "use method='ratio' (or 'division') for division, or "
            "method='difference' for subtraction. In dplPy 'residual' now refers "
            "only to the residual (AR-prewhitened) chronology. Treating "
            "method='residual' as 'ratio' for now.", FutureWarning, stacklevel=4)
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
