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
# Title: autoreg.py
# Description: Contains methods that fit series to autoregressive models and perform functions
#              related to AR modeling.
#              NOTE: This function only accepts pandas series and dataframes as parameters.

from statsmodels.tsa.ar_model import ar_select_order, AutoReg
import pandas as pd
import numpy as np
import warnings
from ._validate import (_alias_kwarg, _reject_kwargs, _normalize_choice,
                        _drop_allnan_columns, _carry_attrs)

def ar_func(data: pd.DataFrame | pd.Series, ar_max=10, aic=True, method="yw", first_aic_min=True, **kwargs) -> (pd.DataFrame | pd.Series):
    """Prewhiten series with a best-fit AR model.

    Fits each series to the best-fit autoregressive model and returns the residuals
    plus the series mean (the prewhitened series).

    Parameters
    ----------
    data : pd.DataFrame or pd.Series
        ring widths from dpl.readers(), or a single series taken from one.
    ar_max : int, default 10
        maximum AR order considered when selecting the model. (``max_lag=`` is
        accepted as a deprecated alias.)
    aic : bool, default True
        select the AR order by AIC up to ar_max; if False, fit a fixed AR of order
        ar_max.
    method : {"yw", "burg", "ols"}, default "yw"
        AR estimator. "yw" (Yule-Walker) is dplPy's package default and matches
        R's ar() default and dplR; "burg" is the maximum-entropy estimator ARSTAN
        and COFECHA use; "ols" matches R's ar(method="ols").
    first_aic_min : bool, default True
        when aic is True, True takes the first local AIC minimum (ARSTAN's rule,
        dplPy's package default), False the global minimum. NOTE this default
        differs from R's ar() / dplR, which take the global minimum.

    Returns
    -------
    res : pandas dataframe or series
        the prewhitened data (residuals plus mean), matching the input type. The
        first ``p`` values of each series (the AR warm-up) are dropped, so a
        Series comes back ``p`` shorter and a DataFrame has NaN there. Columns
        with no measured values are dropped with a warning.

    Examples
    --------
    >>> import dplpy as dpl
    >>> data = dpl.readers("../tests/data/csv/file.csv")
    >>> dpl.ar_func(data["series name"])

    References
    ----------
    .. [1] https://opendendro.github.io/dplPy/reference/chronology/#ar_func
    """
    ar_max = _alias_kwarg(kwargs, "max_lag", "ar_max", ar_max, "ar_func")
    _reject_kwargs(kwargs)
    method = _normalize_choice(method, ("yw", "burg", "ols"), "method")
    if isinstance(data, pd.DataFrame):
        data = _drop_allnan_columns(data, "data")
        start_df = pd.DataFrame(index=pd.Index(data.index))
        to_concat = [start_df]
        for column in data.columns:
            to_concat.append(ar_func_series(data[column], ar_max, aic, method=method, first_aic_min=first_aic_min))
        res = pd.concat(to_concat, axis=1)
        return _carry_attrs(data, res, kind="rwi")
    elif isinstance(data, pd.Series):
        res = ar_func_series(data, ar_max, aic, method=method, first_aic_min=first_aic_min)
        return res
    else:
        raise TypeError("Data argument should be either pandas dataframe or pandas series.")

# This function returns residuals plus mean of the best fit AR
# model of the data.
def ar_func_series(data: pd.Series, ar_max, aic=True, method="yw", first_aic_min=True) -> pd.Series:
    nullremoved_data = data.dropna()
    pars = autoreg(nullremoved_data, ar_max, aic, method=method, first_aic_min=first_aic_min)

    y = nullremoved_data

    yi = fitted_values(y, pars)

    res = y[len(pars)-1:] - yi

    mean = np.mean(y)

    # Add mean to the residuals
    res = res + mean

    return res


def autoreg(data: pd.Series, ar_max=10, aic=True, method="yw", first_aic_min=True, **kwargs):
    """Fit the best-fit AR model to a series and return its parameters.

    Selects the AR order up to max_lag by AIC and returns the fitted model
    parameters.

    Parameters
    ----------
    data : pd.Series
        an individual series representing tree rings/widths.
    ar_max : int, default 10
        maximum AR order considered when selecting the model. (``max_lag=`` is
        accepted as a deprecated alias.)
    aic : bool, default True
        select the AR order by AIC up to ar_max; if False, fit a fixed AR of order
        ar_max.
    method : {"yw", "burg", "ols"}, default "yw"
        AR estimator. "yw" (Yule-Walker) is dplPy's package default and matches
        R's ar() default and dplR; "burg" is the estimator ARSTAN and COFECHA use;
        "ols" matches R's ar(method="ols").
    first_aic_min : bool, default True
        when aic is True, True takes the first local AIC minimum (ARSTAN's rule,
        dplPy's package default), False the global minimum. NOTE this default
        differs from R's ar() / dplR, which take the global minimum.

    Returns
    -------
    params : array
        parameters of the best-fit AR model, as [intercept, phi_1, ..., phi_p].

    Examples
    --------
    >>> import dplpy as dpl
    >>> data = dpl.readers("../tests/data/csv/file.csv")
    >>> dpl.autoreg(data["series name"])

    References
    ----------
    .. [1] https://opendendro.github.io/dplPy/reference/chronology/#autoreg
    """
    ar_max = _alias_kwarg(kwargs, "max_lag", "ar_max", ar_max, "autoreg")
    _reject_kwargs(kwargs)
    if not isinstance(data, pd.Series):
        raise TypeError("Data argument should be pandas series. Received " + str(type(data)) + " instead.")
    method = _normalize_choice(method, ("yw", "burg", "ols"), "method")

    max_allowable_lag = len(data.dropna())//2 - 1
    max_lag_used = ar_max if ar_max <= max_allowable_lag else max_allowable_lag
    # B11: too few observations for any AR model -> order 0 (mean only) on every
    # estimator path, instead of statsmodels raising on the OLS path.
    if max_lag_used < 1:
        x = data.dropna().to_numpy(dtype=float)
        return np.array([float(np.mean(x)) if x.size else 0.0])

    # Yule-Walker path -- matches R's ar(method="yule-walker") (dplR's default,
    # and what chron() uses so its residual chronology matches dplR). Returned in
    # the same [intercept, phi_1..phi_p] layout as the OLS path so fitted_values()
    # and the residual+mean convention downstream are unchanged.
    if method == "yw":
        return _yw_params_aic(data.dropna().to_numpy(dtype=float), max_lag_used, aic,
                              first_aic_min=first_aic_min)

    # Burg (maximum-entropy) path -- Ed Cook / Paul Krusic's estimator, and the
    # one COFECHA's MEMPR uses. Same [intercept, phi_1..phi_p] layout as the
    # others. Burg minimises the summed forward+backward prediction error, which
    # is better conditioned on short series than Yule-Walker; it is the estimator
    # used for the COFECHA preset in xdate.
    if method == "burg":
        return _burg_params_aic(data.dropna().to_numpy(dtype=float), max_lag_used, aic,
                                first_aic_min=first_aic_min)

    # OLS path -- matches R's ar(method="ols"). statsmodels AutoReg is conditional
    # least squares; ar_select_order picks the order by AIC up to max_lag.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        if aic:
            # select the AR order by AIC, up to max_lag (dplR ar(aic=TRUE))
            ar_data = ar_select_order(data.dropna(), max_lag_used, ic='aic')
            results = ar_data.model.fit()
        else:
            # fit a fixed AR of order max_lag (dplR ar(aic=FALSE, order.max=...))
            results = AutoReg(data.dropna(), lags=max(max_lag_used, 1)).fit()
    return results.params


def _yw_params_aic(x, max_lag, aic=True, first_aic_min=False):
    """Yule-Walker AR fit, matching R's ar(method="yule-walker").

    Selects the order by AIC up to ``max_lag`` when ``aic`` is True, else a fixed
    order ``max_lag``. Returns ``[intercept, phi_1, ..., phi_p]`` with intercept =
    mean*(1 - sum(phi)), so the array plugs straight into fitted_values(). When
    ``aic`` is True, ``first_aic_min`` selects the first local AIC minimum (more
    parsimonious) rather than the global minimum.
    """
    x = x[~np.isnan(x)]
    n = len(x)
    m = float(np.mean(x)) if n else 0.0
    xd = x - m
    ml = int(min(max_lag, n - 1)) if n > 1 else 0
    if ml < 1 or xd @ xd == 0:
        return np.array([m])                       # order 0: intercept only
    # biased autocovariances r[0..ml]
    r = np.array([np.dot(xd[:n - k], xd[k:]) / n for k in range(ml + 1)])
    # Levinson-Durbin, then AIC at each order
    phis, vs = _levinson(r)
    aics = [n * np.log(vs[k]) + 2 * k for k in range(ml + 1)]
    if not aic:
        p = ml
    elif first_aic_min:
        p = _first_aic_min(aics)                    # ARSTAN rule (raises if monotone)
    else:
        p = int(np.argmin(aics))                    # global AIC min (R's ar())
    phi_p = phis[p]
    return np.concatenate(([m * (1.0 - float(np.sum(phi_p)))], phi_p))


def _levinson(r):
    """Levinson-Durbin recursion on the autocovariances (or autocorrelations)
    ``r[0..m]``. Returns ``(phis, vs)``: ``phis[k]`` the AR(k) coefficients in
    the positive convention (``x[t] = sum_j phi[j] x[t-j] + e``) and ``vs[k]``
    the innovations variance at order k (in the units of ``r[0]``), for
    k = 0..m. The one Yule-Walker/Levinson implementation (audit E7), used by
    the per-series fits here and by chron_ars's pooled AR."""
    r = np.asarray(r, dtype=float)
    m = r.size - 1
    v = r[0]
    phi = np.zeros(0)
    phis = [phi]
    vs = [v]
    for k in range(1, m + 1):
        acc = r[k] - (np.dot(phi, r[k - 1:0:-1]) if k > 1 else 0.0)
        refl = acc / v
        phin = np.empty(k)
        if k > 1:
            phin[:k - 1] = phi - refl * phi[::-1]
        phin[k - 1] = refl
        v = v * (1.0 - refl * refl)
        phi = phin
        phis.append(phi.copy())
        vs.append(v)
    return phis, vs


def _burg_params_aic(x, max_lag, aic=True, first_aic_min=False):
    """Burg (maximum-entropy) AR fit.

    Selects the order by AIC up to ``max_lag`` (``first_aic_min`` takes the first
    local AIC minimum). Returns ``[intercept, phi_1, ..., phi_p]`` with intercept =
    mean*(1 - sum(phi)), matching `_yw_params_aic()`.
    """
    x = x[~np.isnan(x)]
    n = len(x)
    m = float(np.mean(x)) if n else 0.0
    xc = x - m
    ml = int(min(max_lag, n - 1)) if n > 1 else 0
    if ml < 1 or xc @ xc == 0:
        return np.array([m])                        # order 0: intercept only
    f = xc.copy()
    b = xc.copy()
    P = float(xc @ xc) / n
    aics = [n * np.log(P) + 2.0]
    phis = [np.zeros(0)]
    a = np.zeros(0)
    for mm in range(ml):
        fp = f[mm + 1:n]
        bp = b[mm:n - 1]
        den = float(fp @ fp + bp @ bp)
        if den == 0.0:
            break
        k = 2.0 * float(fp @ bp) / den
        # update forward/backward errors (store b at [mm+1:n], aligned with f)
        f[mm + 1:n] = fp - k * bp
        b[mm + 1:n] = bp - k * fp
        anew = np.zeros(mm + 1)
        anew[mm] = k
        if mm > 0:
            anew[:mm] = a - k * a[::-1]
        a = anew
        P = P * (1.0 - k * k)
        if P <= 0.0:
            aics.append(np.inf)
            phis.append(a.copy())
            break
        aics.append(n * np.log(P) + 2.0 * (mm + 2))
        phis.append(a.copy())
    if not aic:
        p = len(phis) - 1
    elif first_aic_min:
        p = _first_aic_min(aics)
    else:
        p = int(np.argmin(aics))
    phi_p = phis[p]
    return np.concatenate(([m * (1.0 - float(np.sum(phi_p)))], phi_p))


def _first_aic_min(aics, what="The AIC"):
    """Order of the first local AIC minimum: the first order after which AIC turns
    back up (or order 0 if AIC already rises from 0). The single home for
    ARSTAN's / dplR chron.ars's firstAICmin rule (audit E8/G5). If AIC never
    turns up within the maximum order (monotone decreasing) there is no local
    minimum and a ValueError is raised, as dplR's chron.ars does -- raise
    ``ar_max`` or use the global minimum (``first_aic_min=False``)."""
    for k in range(len(aics) - 1):
        if aics[k] < aics[k + 1]:
            return k
    raise ValueError(
        what + " did not reach a minimum within the maximum AR order ("
        + str(len(aics) - 1) + "); the order cannot be determined by the "
        "first-local-minimum rule. Increase ar_max or pass first_aic_min=False "
        "(global minimum). Highly persistent or non-stationary series can "
        "prevent the AIC from reaching a minimum (see Cook 1985).")

# This function calculates the in-sample predicted values of a series,
# given an array containing the original data and the parameters for
# the AR model
def fitted_values(data_series, params):
    data_arr = np.asarray(data_series, dtype=float)
    par = np.asarray(params, dtype=float)

    p = len(par) - 1
    n = len(data_arr)

    if n <= p:
        return np.asarray([])

    # pred[i] = par[0] + sum_{j=1..p} par[j] * data_arr[i-j], for i in [p, n-1].
    # Vectorized over i via slicing; the inner sum over lags j is kept as a
    # (typically short) Python loop so floating-point accumulation order,
    # and therefore the result, matches the original nested-loop version.
    results = np.full(n - p, par[0])
    for j in range(1, len(par)):
        results = results + par[j] * data_arr[p - j: n - j]
    return results


# --------------------------------------------------------------------------- #
# Canonical prewhitening (audit D1 / A12). The single home for "whiten a series
# with an AR model" and "put the persistence back": every function that
# prewhitens (chron, chron_ars, xdate family, rwi_stats, check, ...) routes
# through _prewhiten so that ``prewhiten=True`` means ONE thing package-wide.
# Package default: Yule-Walker, ar_max=10, first local AIC minimum (ARSTAN's
# rule). Burg is the estimator for explicit ARSTAN/COFECHA presets.
# --------------------------------------------------------------------------- #
def _backcast(xd, phi, nbc=None):
    """ARSTAN's ``bckcst``: prepend ``nbc`` (default ``len(phi)``) pre-sample
    values to the DEMEANED series ``xd``. Each is the AR prediction from the
    ``p`` values that FOLLOW it, filled backward from the start of the series:
    ``x[i] = sum_j phi[j] * x[i+1+j]``. (dplR's postAR, and dplPy's old
    ``_post_ar``, took initials from the END of the record instead -- audit A12.)
    Returns the extended array (length ``nbc + len(xd)``)."""
    phi = np.asarray(phi, dtype=float)
    p = phi.size
    xd = np.asarray(xd, dtype=float)
    if p == 0:
        return xd.copy()
    nbc = p if nbc is None else int(nbc)
    out = np.concatenate([np.zeros(nbc), xd])
    for i in range(nbc - 1, -1, -1):
        acc = 0.0
        for j in range(p):
            acc += phi[j] * out[i + 1 + j]
        out[i] = acc
    return out


def _whiten(x, phi, backcast=True):
    """ARSTAN's ``albino``: residuals ``e[t] = x[t] - sum_j phi[j] x[t-j]`` on
    a complete (no-NaN) array. With ``backcast=True`` the first ``p`` residuals
    use backcast initials so no years are lost; with ``backcast=False`` they are
    NaN (R's ar() / dplR behaviour). The series mean is removed before and
    restored after, so the result is residuals + mean. Returns an array the
    same length as ``x``."""
    x = np.asarray(x, dtype=float)
    phi = np.asarray(phi, dtype=float)
    p = phi.size
    m = float(np.mean(x)) if x.size else 0.0
    xd = x - m
    if p == 0:
        return x.copy()
    n = xd.size
    if backcast:
        w = _backcast(xd, phi, p)                    # length n + p, data at [p:]
        e = w[p:].copy()
        for j in range(1, p + 1):
            e -= phi[j - 1] * w[p - j:p - j + n]
    else:
        e = np.full(n, np.nan)
        for t in range(p, n):
            acc = xd[t]
            for j in range(1, p + 1):
                acc -= phi[j - 1] * xd[t - j]
            e[t] = acc
    return e + m


def _redden(e, phi, backcast=True):
    """ARSTAN's ``commie``: put the AR persistence back into a white series,
    ``y[t] = e[t] + sum_j phi[j] y[t-j]``, with ``bckcst`` initials (or zeros
    when ``backcast=False``). Mean removed before and restored after. Returns
    an array the same length as ``e``."""
    e = np.asarray(e, dtype=float)
    phi = np.asarray(phi, dtype=float)
    p = phi.size
    m = float(np.mean(e)) if e.size else 0.0
    ed = e - m
    if p == 0:
        return e.copy()
    n = ed.size
    w = _backcast(ed, phi, p) if backcast else np.concatenate([np.zeros(p), ed])
    for t in range(p, p + n):
        acc = w[t]
        for j in range(1, p + 1):
            acc += phi[j - 1] * w[t - j]
        w[t] = acc
    return w[p:] + m


def _prewhiten(x, estimator="yw", ar_max=10, first_aic_min=True, order=None,
               backcast=True, return_params=False):
    """Package-standard prewhitening of one series (NaN-aware).

    Fits an AR model to the non-NaN values (treating them as contiguous, as
    dplR's ar.func does), selects the order by AIC up to ``ar_max`` (first local
    minimum when ``first_aic_min``, the ARSTAN rule) or uses a fixed ``order``,
    whitens with ARSTAN-style backcast initials (no years lost) unless
    ``backcast=False``, and returns residuals + mean at the original positions
    (NaN where the input is NaN).

    Parameters
    ----------
    x : 1-D array-like
    estimator : {"yw", "burg", "ols"}, default "yw"
    ar_max : int, default 10
    first_aic_min : bool, default True
    order : int or None
        a fixed AR order (``None`` selects by AIC).
    backcast : bool, default True
    return_params : bool
        also return ``[intercept, phi_1..phi_p]``.
    """
    x = np.asarray(x, dtype=float)
    mask = ~np.isnan(x)
    out = np.full(x.shape, np.nan)
    xv = x[mask]
    if xv.size < 3:
        out[mask] = xv
        return (out, np.array([float(np.mean(xv)) if xv.size else 0.0])) if return_params else out
    estimator = _normalize_choice(estimator, ("yw", "burg", "ols"), "estimator")
    if order is not None:
        # a fixed order needs only n > p (R's ar(aic=FALSE)); AIC selection is
        # capped at n/2 - 1 so the candidate orders stay well determined.
        aic, ml = False, max(0, min(int(order), xv.size - 1))
    else:
        aic, ml = True, max(0, min(int(ar_max), xv.size // 2 - 1))
    if ml < 1:
        params = np.array([float(np.mean(xv))])
    elif estimator == "yw":
        params = _yw_params_aic(xv, ml, aic, first_aic_min=first_aic_min)
    elif estimator == "burg":
        params = _burg_params_aic(xv, ml, aic, first_aic_min=first_aic_min)
    else:
        params = autoreg(pd.Series(xv), ml, aic, method="ols", first_aic_min=first_aic_min)
        params = np.asarray(params, dtype=float)
    phi = params[1:]
    out[mask] = _whiten(xv, phi, backcast=backcast)
    return (out, params) if return_params else out
