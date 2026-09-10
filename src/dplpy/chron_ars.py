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

# Title: chron_ars.py
# Project: OpenDendro dplPy
# Description: ARSTAN-style chronologies (Cook 1985): the standard (std),
#              residual (res), and re-reddened ARSTAN (ars) chronologies. This
#              is a port of dplR's chron.ars(), which Andy Bunn and Kevin
#              Anchukaitis themselves ported from Ed Cook's original FORTRAN and
#              validated against it. The dplR authors flagged the pooled-AR loop
#              as a performance "time killer" needing vectorization; this port
#              vectorizes it with numpy while reproducing dplR's output.
#
#              NOTES: the pooled ACF, AR-coefficient matrix, AIC, and selected
#              order follow dplR's algorithm; the std/res/ars chronologies match
#              dplR for the ar.yw method (both biweight=FALSE and, after the tbrm
#              epsilon fix, biweight=TRUE). The arima.CSS-ML method differs from
#              R's arima optimizer (statsmodels' state-space MLE vs R's), mainly
#              in low-replication years.
#
# example usage from Python Console:
# >>> import dplpy as dpl
# >>> data = dpl.readers("../tests/data/csv/file.csv")
# >>> rwi = dpl.detrend(data, fit="spline", plot=False)
# >>> dpl.chron_ars(rwi)
# >>> dpl.chron_ars(rwi, prewhiten_method="arima.CSS-ML", verbose=False)

import warnings

import numpy as np
import pandas as pd
from ._validate import _require_dataframe
from scipy.signal import lfilter

from .tbrm import tbrm_rows

_KNOWN_PREWHITEN_METHODS = ("ar.yw", "arima.CSS-ML")
_KNOWN_ARS_METHODS = ("arstan", "dplr")


def chron_ars(rwi_data: pd.DataFrame, biweight=True, max_lag=10,
              first_aic_min=True, verbose=True, prewhiten_method="ar.yw",
              ars_method="arstan", stabilize=None, stabilize_kwargs=None):
    """Build ARSTAN standard, residual, and re-reddened chronologies.

    Extended Summary
    ----------------
    Produces the three ARSTAN chronologies of Cook (1985) from a set of
    detrended ring-width indices:

    - **std**: the standard chronology -- the (robust) mean of the RWI series,
      with no autoregressive modeling.
    - **res**: the residual chronology -- each series is prewhitened with a
      pooled AR(p) model, the prewhitened series are averaged, and that mean is
      prewhitened once more to order p, yielding a near-white chronology.
    - **ars**: the ARSTAN chronology -- the common red-noise persistence
      (the *pooled* AR model) is reintroduced by "re-reddening" (postAR). How
      this is done is controlled by ``ars_method`` (see Notes): the default
      ``"arstan"`` re-reddens the mean of the prewhitened series (average then
      re-redden, as in Ed Cook's ARSTAN); ``"dplr"`` re-reddens each series and
      then averages (matching dplR's chron.ars).

    The AR order p is chosen from a pooled autocovariance accumulated across all
    series and lags (Cook's pooled-AR approach), converted to AR coefficients
    (Durbin-Levinson) and selected by AIC.

    This is a port of dplR's chron.ars(). The pooled-AR accumulation -- the
    step the dplR authors flagged as a "time killer" -- is vectorized with numpy
    here (a per-series-pair loop of vectorized lag dot-products, replacing
    dplR's triple loop with per-iteration cbind/rowSums), while following
    dplR/FORTRAN's behavior, including its practice of compressing each
    series pair to its common overlap and then lagging by position. The
    re-reddening (postAR) forward filter is applied with scipy.signal.lfilter.

    Parameters
    ----------
    rwi_data : pandas dataframe
        detrended ring-width indices (e.g. from dpl.detrend()), with years as
        the index and series as columns.
    biweight : boolean, default True
        if True, aggregate series with Tukey's biweight robust mean (tbrm);
        if False, use the arithmetic mean.
    max_lag : int, default 10
        maximum AR lag considered when selecting the pooled AR order.
    first_aic_min : boolean, default True
        if True, select the order at the first local AIC minimum (dplR's
        default); if False, select the global AIC minimum.
    verbose : boolean, default True
        if True, print the pooled ACF, AR coefficients, AIC, and selected order.
    prewhiten_method : str, default "ar.yw"
        "ar.yw" (Yule-Walker, the default and the numerically exact match to
        dplR) or "arima.CSS-ML" (matches dplR within tolerance).
    ars_method : str, default "arstan"
        how the ARSTAN (re-reddened) chronology is constructed (see Notes):
        "arstan" re-reddens the mean of the prewhitened series (average then
        re-redden -- Ed Cook's ARSTAN order); "dplr" re-reddens each series and
        then averages (dplR's chron.ars order). The two agree only when every
        series spans the full period; under staggered sample depth they differ.
    stabilize : {None, "rbar", "spline", "both"}, default None
        If not None, also variance-stabilize each chronology and return the
        stabilized versions in extra ``*_vsc`` columns (see Notes). This mirrors
        ARSTAN, where variance stabilization (menu option ``isb``) is applied to
        the finished chronologies. The method families are as in
        :func:`stabilize_chron`: ``"rbar"`` (Osborn/Frank N_eff scaling),
        ``"spline"`` (ARSTAN stabit spline), or ``"both"`` (rbar then a 67%
        spline; ARSTAN ``isb=2``).
    stabilize_kwargs : dict, optional
        Extra keyword arguments forwarded to :func:`stabilize_chron` when
        ``stabilize`` is set (e.g. ``{"rbar_mode": "constant"}``). The
        chronology, column, source matrix, and sample depth are supplied
        automatically and may not be overridden here.

    Returns
    -------
    out : pandas dataframe indexed by year with columns 'std', 'res', 'ars',
        and 'samp_depth'. When ``stabilize`` is set, a stabilized ``std_vsc``,
        ``res_vsc``, and ``ars_vsc`` column is added next to each base
        chronology.

    Notes
    -----
    **ARSTAN chronology construction (``ars_method``).** The re-reddening and
    averaging steps do not commute when sample depth changes through time, so
    there are two distinct ways to build the ARSTAN chronology:

    - ``"arstan"`` (default): ``ars = postAR(mean(prewhitened series))`` --
      average the prewhitened series first, then re-redden that single mean
      series once. This reproduces Ed Cook's ARSTAN FORTRAN, which re-reddens
      the residual chronology directly (``commie1`` applied to the residual
      chronology). dplPy defaults to this to be faithful to ARSTAN, from which
      chron_ars takes its DNA.
    - ``"dplr"``: ``ars = mean(postAR(prewhitened series_i))`` -- re-redden each
      prewhitened series and then average. This matches dplR's ``chron.ars`` and
      was dplPy's behavior before this option existed.

    The ``std`` and ``res`` chronologies, the pooled AR order, and the
    common-pooled-order prewhitening are identical for both settings; only the
    ``ars`` column changes. Use ``ars_method="dplr"`` to reproduce dplR exactly.

    **Variance stabilization (``stabilize``).** Following ARSTAN's per-chronology
    rbar convention, ``std_vsc`` uses the rbar of the standard (detrended) RWI
    matrix, and both ``res_vsc`` and the ARSTAN chronology use the rbar of the
    residual (prewhitened) matrix. The ARSTAN column is stabilized in the ARSTAN
    order: with ``ars_method="arstan"`` the residual-style mean of the prewhitened
    series is stabilized **before** re-reddening (ARSTAN stabilizes the residual
    chronology, then ``commie1``), so ``ars_vsc`` stays consistent with the
    unstabilized ``ars``. With ``ars_method="dplr"`` the ARSTAN chronology is
    already built by re-reddening each series, so ``ars_vsc`` is stabilized
    post-hoc on the reddened chronology (not the ARSTAN order). All columns
    restandardize to their own full period (see :func:`stabilize_chron`).

    Examples
    --------
    >>> import dplpy as dpl
    >>> rwi = dpl.detrend(dpl.readers("../tests/data/csv/file.csv"), plot=False)
    >>> dpl.chron_ars(rwi)

    References
    ----------
    .. [1] https://rdrr.io/cran/dplR/man/chron.ars.html
    .. [2] Cook, E.R. (1985) A time series analysis approach to tree-ring
           standardization. PhD dissertation, University of Arizona.

    """
    _require_dataframe(rwi_data)
    if prewhiten_method not in _KNOWN_PREWHITEN_METHODS:
        raise ValueError("prewhiten_method must be one of " + str(_KNOWN_PREWHITEN_METHODS)
                         + ", got '" + str(prewhiten_method) + "'.")
    if ars_method not in _KNOWN_ARS_METHODS:
        raise ValueError("ars_method must be one of " + str(_KNOWN_ARS_METHODS)
                         + ", got '" + str(ars_method) + "'.")
    from .stabilize import stabilize_chron, _prepare_stabilize_kwargs
    skw = _prepare_stabilize_kwargs(stabilize, stabilize_kwargs)

    x = rwi_data.to_numpy(dtype=float)
    n_series = x.shape[1]
    samp_depth = (~np.isnan(x)).sum(axis=1)

    # standard chronology
    std_crn = _aggregate(x, biweight)

    # pooled AR: ACF, AR coefs, AIC, selected order
    out_ar = _pooled_ar(x, max_lag, first_aic_min)
    p = out_ar["order"]

    if verbose:
        _print_summary(out_ar, max_lag)

    prewhiten = _prewhiten_ar_yw if prewhiten_method == "ar.yw" else _prewhiten_arima

    # prewhiten each series individually to the pooled order
    rwi_clean = np.column_stack([prewhiten(x[:, s], p) for s in range(n_series)])

    # residual chronology: mean of prewhitened series, prewhitened again to p
    res_crn = prewhiten(_aggregate(rwi_clean, biweight), p)

    # re-reddened (ARSTAN) chronology. postAR is called with the pooled AR
    # coefficients. Two constructions are offered (see the docstring):
    #   "arstan" (default): re-redden the already-built residual-style mean of
    #      the prewhitened series -- ars = postAR(mean(prewhitened series)) --
    #      i.e. average THEN re-redden. This is what Ed Cook's ARSTAN does
    #      (commie1 applied to the residual chronology).
    #   "dplr": re-redden each prewhitened series, THEN average --
    #      ars = mean(postAR(prewhitened series_i)) -- matching dplR's chron.ars
    #      and dplPy's original behavior. These do not commute under staggered
    #      sample depth.
    if p > 0:
        # out_ar["arcoefs"] holds dplR's negated coefs (ARcoefs); dplR calls
        # postAR with -phi, i.e. the positive AR coefficients.
        phi = -out_ar["arcoefs"][p - 1, :p]
    else:
        phi = np.empty(0)
    if ars_method == "arstan":
        # average then re-redden: re-redden the mean of the prewhitened series.
        # NOTE: unlike res_crn, this mean is NOT prewhitened again before
        # re-reddening -- prewhiten-then-postAR to the same order would undo
        # each other. ARSTAN re-reddens the residual chronology directly.
        ars_crn = _post_ar(_aggregate(rwi_clean, biweight), phi)
    else:  # "dplr": re-redden each series then average
        rwi_ar = np.column_stack(
            [_post_ar(rwi_clean[:, s], phi) for s in range(n_series)])
        ars_crn = _aggregate(rwi_ar, biweight)

    out = pd.DataFrame(
        {"std": std_crn, "res": res_crn, "ars": ars_crn, "samp_depth": samp_depth},
        index=rwi_data.index,
    )

    if skw is not None:
        # residual (prewhitened) RWI matrix, for the residual/ARSTAN rbar
        # (ARSTAN's per-chronology rbar convention).
        resid_matrix = pd.DataFrame(rwi_clean, index=rwi_data.index,
                                    columns=rwi_data.columns)

        # std_vsc: standard chronology, standard matrix rbar
        out["std_vsc"] = stabilize_chron(
            pd.Series(std_crn, index=rwi_data.index), stabilize,
            rwi=rwi_data, **skw)["vsc"]
        # res_vsc: residual chronology, residual matrix rbar
        out["res_vsc"] = stabilize_chron(
            pd.Series(res_crn, index=rwi_data.index), stabilize,
            rwi=resid_matrix, **skw)["vsc"]

        if ars_method == "arstan":
            # ARSTAN order: stabilize the residual-style mean of the prewhitened
            # series (residual rbar), THEN re-redden -- the same average-then-
            # redden pipeline as the unstabilized ars, matching ARSTAN.
            resid_mean = _aggregate(rwi_clean, biweight)
            resid_mean_vsc = stabilize_chron(
                pd.Series(resid_mean, index=rwi_data.index), stabilize,
                rwi=resid_matrix, **skw)["vsc"].to_numpy()
            out["ars_vsc"] = _post_ar(resid_mean_vsc, phi)
        else:  # "dplr": ars is already reddened -- stabilize it post-hoc
            out["ars_vsc"] = stabilize_chron(
                pd.Series(ars_crn, index=rwi_data.index), stabilize,
                rwi=resid_matrix, **skw)["vsc"]

        # keep each stabilized column beside its base chronology
        out = out[["std", "std_vsc", "res", "res_vsc", "ars", "ars_vsc",
                   "samp_depth"]]

    return out


def _aggregate(mat, biweight):
    """Per-year aggregation across series: biweight robust mean or arithmetic
    mean, ignoring NaN. An all-NaN year yields NaN."""
    if biweight:
        return tbrm_rows(mat)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows -> NaN
        return np.nanmean(mat, axis=1)


def _pooled_ar(x, max_lag, first_aic_min):
    """Pooled autoregression across all series (Cook's pooled AR).

    Vectorized port of dplR's pooledAR: accumulates a pooled product-sum across
    every ordered series pair and lag, converts to a pooled ACF, then to AR
    coefficients (Durbin-Levinson), and selects the order by AIC. The per-pair
    common-overlap compression + position lag follows dplR/FORTRAN.
    """
    n_years, n_series = x.shape

    # center each column by its na.rm mean (matches R scale(center=TRUE)),
    # preserving NaN positions
    xc = x - np.nanmean(x, axis=0, keepdims=True)
    present = ~np.isnan(xc)

    product_sum = np.zeros(max_lag + 1)
    for j in range(n_series):
        pj = present[:, j]
        for i in range(n_series):
            common = pj & present[:, i]
            lc = int(common.sum())
            if lc == 0:
                continue
            # compress both series to their common overlap, in time order
            s1 = xc[common, j]
            s2 = xc[common, i]
            for k in range(min(max_lag, lc - 1) + 1):
                # lag by POSITION within the compressed overlap (dplR/FORTRAN):
                # sum_{m=k}^{lc-1} s1[m] * s2[m-k]
                product_sum[k] += np.dot(s1[k:lc], s2[0:lc - k])

    r0 = product_sum[0]
    acf = product_sum / r0
    arcoefs_pos = _acf2ar(acf, max_lag)   # Durbin-Levinson, positive convention
    arcoefs = -arcoefs_pos                  # dplR stores negated coefs (ARcoefs)

    # prediction error variance per order, then AIC (matches dplR's vp_shortcut)
    vp = np.empty(max_lag + 1)
    vp[0] = r0
    vp[1:] = r0 + arcoefs @ product_sum[1:]
    aic = n_years * np.log(vp) + 2.0 * np.arange(1, max_lag + 2)

    if first_aic_min:
        first_min = _get_first_min(aic)
        if first_min is None:
            raise ValueError(
                "The pooled AIC did not reach a minimum within max_lag ("
                + str(max_lag) + ") lags; the AR order cannot be determined. "
                "Consider increasing max_lag. Highly persistent or non-stationary "
                "series can prevent the AIC from reaching a minimum (see Cook 1985)."
            )
        order = first_min - 1
    else:
        order = int(np.argmin(aic))

    return {"acf": acf, "arcoefs": arcoefs, "aic": aic, "order": order}


def _acf2ar(acf, max_lag):
    """Durbin-Levinson recursion (equivalent to R's acf2AR): returns a
    max_lag x max_lag matrix whose row p-1 holds the AR(p) coefficients."""
    A = np.zeros((max_lag, max_lag))
    phi = np.zeros(max_lag + 1)
    phi_prev = np.zeros(max_lag + 1)
    v = 1.0
    for k in range(1, max_lag + 1):
        if k == 1:
            kappa = acf[1]
        else:
            kappa = (acf[k] - np.sum(phi_prev[1:k] * acf[k - 1:0:-1])) / v
        phi[k] = kappa
        for j in range(1, k):
            phi[j] = phi_prev[j] - kappa * phi_prev[k - j]
        v = v * (1 - kappa * kappa)
        A[k - 1, :k] = phi[1:k + 1]
        phi_prev[:k + 1] = phi[:k + 1]
    return A


def _get_first_min(y):
    """First local minimum of the AIC (mirrors dplR's getFirstMin; 1-based
    like R, so the caller subtracts 1 for the lag-0 offset). Returns None if
    the AIC never turns back up (dplR's Inf)."""
    if y[0] < y[1]:
        return 1
    increasing = np.flatnonzero(np.diff(y) > 0)
    if increasing.size == 0:
        return None
    return int(increasing[0]) + 1


def _prewhiten_ar_yw(series, p):
    """Prewhiten one series with a fixed-order Yule-Walker AR(p) model.
    Matches dplR's ar(..., method="yule-walker") residuals. The first p values
    become NaN, as in R's ar() residuals."""
    from statsmodels.regression.linear_model import yule_walker

    if p == 0:
        return series.copy()
    mask = np.isnan(series)
    x2 = series[mask == False]  # non-NaN values, in order
    n = len(x2)
    m = x2.mean()
    xd = x2 - m
    # method="mle" uses the biased autocovariance, matching R's acf default
    rho, _sigma = yule_walker(xd, order=p, method="mle", demean=False)

    # residuals: resid[t] = xd[t] - sum_j rho[j] xd[t-j]; first p are NaN.
    # vectorized over years, looping only over the (small) lag order.
    pred = np.zeros(n)
    for j in range(1, p + 1):
        pred[j:] += rho[j - 1] * xd[:n - j]
    resid = np.full(n, np.nan)
    resid[p:] = xd[p:] - pred[p:]

    out = series.copy()
    out[~mask] = resid + m
    return out


def _prewhiten_arima(series, p):
    """Prewhiten one series with an ARIMA(p,0,0) CSS-ML fit. Matches dplR's
    arima(..., method="CSS-ML") within tolerance (different optimizer)."""
    if p == 0:
        return series.copy()
    from statsmodels.tsa.arima.model import ARIMA

    mask = np.isnan(series)
    x2 = series[mask == False]
    m = x2.mean()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ARIMA(x2, order=(p, 0, 0), trend="c").fit(method="statespace")
    out = series.copy()
    out[~mask] = fit.resid + m
    return out


def _post_ar(series, phi):
    """Re-redden a prewhitened series by reintroducing pooled AR persistence.

    Port of dplR's postAR: the whitened series is reversed, run through the AR
    filter (scipy.signal.lfilter, zero initial conditions), then a short
    backcast improves the initial values, and the AR filter is applied once
    more with those improved initials.
    """
    mask = np.isnan(series)
    x0 = series.copy()
    phi = np.asarray(phi, dtype=float)
    n_phi = phi.size
    if n_phi == 0:
        return x0

    xv = series[mask == False].astype(float)
    x_mean = xv.mean()
    xv = xv - x_mean
    nx = xv.size
    x_rev = xv[::-1]

    # steps 2-3: forward AR filter with zero initial conditions.
    # y[t] = x[t] + sum_j phi[j] y[t-j]  == IIR filter b=[1], a=[1, -phi...]
    a = np.concatenate([[1.0], -phi])
    x_rev_init = lfilter([1.0], a, x_rev)

    # step 5: backcast better initial values (short loop over the AR order)
    for i in range(1, n_phi + 1):
        cntr = n_phi - i
        acc = 0.0
        for j in range(1, n_phi + 1):
            acc += phi[j - 1] * x_rev_init[cntr + j]
        x_rev_init[cntr] = acc

    # steps 6-7: reapply the AR filter with the improved initial values.
    # (kept as an explicit recurrence because the initials are non-zero)
    x_ar = np.concatenate([x_rev_init[:n_phi], xv])
    for i in range(x_rev_init.size):
        acc = 0.0
        for j in range(1, n_phi + 1):
            acc += phi[j - 1] * x_ar[i + n_phi - j]
        x_ar[i + n_phi] += acc

    x0[~mask] = x_ar[n_phi:] + x_mean
    return x0


def _print_summary(out_ar, max_lag):
    labels = ["ar(" + str(k) + ")" for k in range(max_lag + 1)]
    print("Pooled AR Summary")
    print("ACF")
    print(pd.Series(out_ar["acf"], index=labels).to_string())
    print("Selected Order")
    print(out_ar["order"])
