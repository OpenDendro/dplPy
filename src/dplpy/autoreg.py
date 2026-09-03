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

def ar_func(data: pd.DataFrame | pd.Series, max_lag=5, aic=True, method="ols", first_aic_min=False) -> (pd.DataFrame | pd.Series):
    """Auto Regressive (AR) functions 
      
    Extended Summary
    ---------------
    Fits a given data to an the best-fit autoregressive model, returns the residuals
    of AR fit relative to the original data + the mean of the original data.

    Parameters
    ----------
    data : pd.DataFrame | pd.Series
        a pandas dataframe imported from dpl.readers() or a series extracted
        from such a dataframe.
    lag: int, default 5
        max lag to consider when selecting the AR model.
   
    Returns
    -------
    res :  pandas dataframe or series of AR-modeled data, depending on which was given as input.
    
    Examples
    --------
    >>> import dplpy as dpl 
    >>> data = dpl.readers("../tests/data/csv/file.csv")
    >>> dpl.ar_func(data['series name']) -> returns residuals plus mean of best fit 
                                            from AR models with max lag of either 5 
                                            (default) or specified number
    
    References
    ----------
    .. [1] https:/opendendro.org/dplpy-man/#ar_func 
    
    """
    if isinstance(data, pd.DataFrame):
        start_df = pd.DataFrame(index=pd.Index(data.index))
        to_concat = [start_df]
        for column in data.columns:
            to_concat.append(ar_func_series(data[column], max_lag, aic, method=method, first_aic_min=first_aic_min))
        res = pd.concat(to_concat, axis=1)
        return res
    elif isinstance(data, pd.Series):
        res = ar_func_series(data, max_lag, aic, method=method, first_aic_min=first_aic_min)
        return res
    else:
        raise TypeError("Data argument should be either pandas dataframe or pandas series.")

# This function returns residuals plus mean of the best fit AR
# model of the data.
def ar_func_series(data: pd.Series, max_lag, aic=True, method="ols", first_aic_min=False) -> pd.Series:
    nullremoved_data = data.dropna()
    pars = autoreg(nullremoved_data, max_lag, aic, method=method, first_aic_min=first_aic_min)
    
    y = nullremoved_data
    
    yi = fitted_values(y, pars)

    res = y[len(pars)-1:] - yi

    mean = np.mean(y)

    # Add mean to the residuals
    res = res + mean

    return res


def autoreg(data: pd.Series, max_lag=5, aic=True, method="ols", first_aic_min=False):
    """ Auto Regressive (AR) functions
    
    Extended Summary
    ----------------
    Selects the best AR model with a specified maximum order for the given data,
    and returns the parameters for the model. The best model is selected based 
    on AIC value.

    Parameters
    ----------
    series : pd.Series
        an individual (Pandas) series representing tree rings/widths.
    lag : int, default 5
        max lag to consider when selecting the AR model.
            
    Returns
    -------
    params: array containing the parameters of best-fit AR model in order.
        
    Examples
    --------
    >>> import dplpy as dpl 
    >>> data = dpl.readers("../tests/data/csv/file.csv")
    >>> dpl.autoreg(data['series name']) -> returns parameters of best fit AR model
                                            with maxlag of 5 (default) or other 
                                            specified number
    References
    ----------
    .. [1] https:/opendendro.org/dplpy-man/#autoreg
    
    """
    # validate data?
    if not isinstance(data, pd.Series):
        raise TypeError("Data argument should be pandas series. Received " + str(type(data)) + " instead.")

    if method not in ("ols", "yw"):
        raise ValueError("method must be 'ols' or 'yw', got '" + str(method) + "'")

    max_allowable_lag = len(data.dropna())//2 - 1
    max_lag_used = max_lag if max_lag <= max_allowable_lag else max_allowable_lag

    # Yule-Walker path -- matches R's ar(method="yule-walker") (dplR's default,
    # and what chron() uses so its residual chronology matches dplR). Returned in
    # the same [intercept, phi_1..phi_p] layout as the OLS path so fitted_values()
    # and the residual+mean convention downstream are unchanged.
    if method == "yw":
        return _yw_params_aic(data.dropna().to_numpy(dtype=float), max_lag_used, aic,
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
    """Yule-Walker AR fit matching R's ar(method="yule-walker").

    Uses the Levinson-Durbin recursion on the biased (divisor-n) autocovariances,
    selecting the order by AIC (n*log(var_p) + 2p, R's criterion) up to ``max_lag``
    when ``aic`` is True, else a fixed order ``max_lag``. Verified against R to
    ~1e-15 in both the selected order and the coefficients. Returns parameters as
    ``[intercept, phi_1, ..., phi_p]`` where intercept = mean*(1 - sum(phi)), so
    the array plugs straight into fitted_values() like the OLS coefficients.

    When ``aic`` is True, ``first_aic_min`` chooses between two order-selection
    rules: False (default) takes the GLOBAL AIC minimum -- R's ar() behaviour, so
    chron() reproduces dplR's chron(); True takes the FIRST LOCAL AIC minimum (the
    first order after which AIC turns back up), which is ARSTAN's rule (and what
    chron_ars uses via firstAICmin). The first-local rule is more parsimonious --
    on real collections it rarely selects an order above ~7 -- at the cost of a
    little AIC. The two agree whenever the AIC minimum is not a late, shallow one.
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
    # Levinson-Durbin, tracking AIC and coefficients at each order
    aics = [n * np.log(r[0])]
    phis = [np.zeros(0)]
    v = r[0]
    phi = np.zeros(0)
    for k in range(1, ml + 1):
        acc = r[k] - (np.dot(phi, r[k - 1:0:-1]) if k > 1 else 0.0)
        refl = acc / v
        phin = np.empty(k)
        if k > 1:
            phin[:k - 1] = phi - refl * phi[::-1]
        phin[k - 1] = refl
        v = v * (1.0 - refl * refl)
        phi = phin
        aics.append(n * np.log(v) + 2 * k)
        phis.append(phi.copy())
    if not aic:
        p = ml
    elif first_aic_min:
        p = _first_aic_min(aics)                    # ARSTAN rule (as chron_ars)
    else:
        p = int(np.argmin(aics))                    # global AIC min (R's ar())
    phi_p = phis[p]
    return np.concatenate(([m * (1.0 - float(np.sum(phi_p)))], phi_p))


def _first_aic_min(aics):
    """Order of the first local AIC minimum: the first order after which AIC turns
    back up (or order 0 if AIC already rises from 0). Mirrors chron_ars/dplR's
    firstAICmin. If AIC never turns up within max_lag (monotone decreasing), fall
    back to the last (highest) order -- there is no earlier local minimum."""
    for k in range(len(aics) - 1):
        if aics[k] < aics[k + 1]:
            return k
    return len(aics) - 1

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
