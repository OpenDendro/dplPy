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
#              residual (res), and re-reddened ARSTAN (ars) chronologies. A
#              port of dplR's chron.ars(), which Andy Bunn and Kevin
#              Anchukaitis themselves ported from Ed Cook's original FORTRAN and
#              validated against it. The dplR authors flagged the pooled-AR loop
#              as a performance "time killer" needing vectorization; this port
#              vectorizes it with numpy while reproducing dplR's output.
#
#              NOTES: the pooled ACF, AR-coefficient matrix, AIC, and selected
#              order follow dplR's algorithm; the std/res chronologies match
#              dplR for the ar.yw method. The ars chronology follows ARSTAN
#              (and the current dplR development version) by default -- see
#              ars_method. The arima.CSS-ML method differs from R's arima
#              optimizer (statsmodels' state-space MLE vs R's), mainly in
#              low-replication years. All whitening and re-reddening routes
#              through the canonical helpers in autoreg (audit D1 / A12).
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
from ._validate import (_require_dataframe, _alias_kwarg, _reject_kwargs,
                        _normalize_choice, _drop_allnan_columns, _carry_attrs,
                        _warn)

from .tbrm import tbrm_rows
from .autoreg import _prewhiten, _redden, _levinson, _first_aic_min

_KNOWN_PREWHITEN_METHODS = ("ar.yw", "burg", "arima.CSS-ML")
_KNOWN_ARS_METHODS = ("arstan", "dplr")


def chron_ars(rwi_data: pd.DataFrame, biweight=True, ar_max=10,
              first_aic_min=True, verbose=True, prewhiten_method="ar.yw",
              ars_method="arstan", backcast=True, stabilize=None,
              stabilize_kwargs=None, **kwargs):
    """Build ARSTAN standard, residual, and re-reddened chronologies.

    Produces the three ARSTAN chronologies of Cook (1985) from a set of detrended
    ring-width indices:

    - **std**: the standard chronology, the (robust) mean of the RWI series with no
      autoregressive modeling.
    - **res**: the residual chronology, each series prewhitened by an AR(p) model
      of the pooled order, averaged, then prewhitened once more to order p
      (near-white).
    - **ars**: the ARSTAN chronology, the pooled red-noise persistence reintroduced
      by re-reddening; ``ars_method`` sets how (see Notes).

    The pooled AR order p is selected by AIC from an autocovariance accumulated
    across all series. Analog of dplR's ``chron.ars``.

    Parameters
    ----------
    rwi_data : pandas dataframe
        detrended ring-width indices (e.g. from ``dpl.detrend``), years as the
        index and series as columns. Columns with no measured values are dropped
        with a warning.
    biweight : boolean, default True
        aggregate series with Tukey's biweight robust mean if True, else the
        arithmetic mean.
    ar_max : int, default 10
        maximum AR lag considered when selecting the pooled AR order.
        (``max_lag=`` is accepted as a deprecated alias.)
    first_aic_min : boolean, default True
        select the order at the first local AIC minimum (ARSTAN's and dplR's rule)
        if True, else the global AIC minimum. With True, a pooled AIC that never
        turns back up within ``ar_max`` raises ValueError.
    verbose : boolean, default True
        print the pooled ACF, AR coefficients, AIC, and selected order.
    prewhiten_method : str, default "ar.yw"
        estimator for the per-series (and residual-chronology) AR(p) whitening,
        case-insensitive: ``"ar.yw"`` (Yule-Walker; numerically exact match to
        dplR), ``"burg"`` (Burg maximum entropy, ARSTAN's estimator) or
        ``"arima.CSS-ML"`` (matches dplR within tolerance).
    ars_method : str, default "arstan"
        how the re-reddened ARSTAN chronology is built (see Notes): ``"arstan"``
        re-reddens the residual chronology, reproducing Ed Cook's ARSTAN and the
        current dplR development version; ``"dplr"`` re-reddens each prewhitened
        series then averages, reproducing released dplR <= 1.8.0.
    backcast : boolean, default True
        backcast the initial values of each AR whitening and re-reddening step
        (ARSTAN's ``bckcst``) so no leading years are lost. False sets the first
        ``p`` residuals of every series to NaN, as dplR's ``chron.ars`` does, and
        re-reddens from zero initials. Applies to the ``"ar.yw"`` and ``"burg"``
        paths; ``"arima.CSS-ML"`` already returns full-length residuals.
    stabilize : {None, "rbar", "spline", "both"}, default None
        if set, also variance-stabilize each chronology and return the stabilized
        versions in extra ``*_vsc`` columns (see Notes). Method families are as in
        `stabilize_chron()`.
    stabilize_kwargs : dict, optional
        extra keyword arguments forwarded to `stabilize_chron()` when
        ``stabilize`` is set. The chronology, column, source matrix, and sample
        depth are supplied automatically and may not be overridden here.

    Returns
    -------
    out : pandas dataframe indexed by year with columns 'std', 'res', 'ars', and
        'samp_depth'. When ``stabilize`` is set, a stabilized ``std_vsc``,
        ``res_vsc``, and ``ars_vsc`` column is added next to each base chronology.

    Notes
    -----
    **ARSTAN chronology (``ars_method``).** ARSTAN whitens every series to the
    pooled order p, averages them, whitens that mean once more (the residual
    chronology, ``albino1``), and then puts the pooled persistence back into the
    residual chronology (``commie1``): ``ars = redden(res, phi_pooled)``. That is
    ``ars_method="arstan"`` (default), and also what the dplR development version
    computes. Released dplR <= 1.8.0 instead re-reddened each prewhitened series
    and averaged those; ``ars_method="dplr"`` reproduces it. Re-reddening and
    averaging do not commute when sample depth changes through time, so the two
    differ; the ``std`` and ``res`` columns, the pooled AR order, and the
    prewhitening are identical either way.

    **Backcasting (``backcast``).** An AR(p) filter has no data for its first ``p``
    lagged terms. With ``backcast=True`` (default) those pre-sample values are
    estimated by running the model in reverse from the start of each series
    (ARSTAN's ``bckcst``), so the residual and ARSTAN chronologies are full length
    (no leading NaN); the first ``p`` years of each series are model-based, so
    treat them as slightly less certain. With ``backcast=False`` the first ``p``
    residuals of every series are NaN, as in dplR's ``chron.ars``. Because series
    start in different years, that changes the chronology ``res`` (and ``ars``)
    in every year where some series is within its first ``p`` years -- not only
    the first ``p`` years of the chronology. dplR's ``postAR`` took its
    re-reddening initials from the end of the record; dplPy follows ARSTAN here
    too, so ``ars`` differs from dplR in roughly the first 10 years of each
    series' contribution.

    **Variance stabilization (``stabilize``).** ``std_vsc`` uses the rbar of the
    standard RWI matrix; ``res_vsc`` the rbar of the residual matrix; ``ars_vsc``
    is the stabilized residual chronology re-reddened (``"arstan"``) or the
    reddened chronology stabilized with the residual rbar (``"dplr"``). Each
    column restandardizes to its own full period (see `stabilize_chron()`).

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
    .. [3] https://opendendro.github.io/dplPy/reference/chronology/#chron_ars
    """
    _require_dataframe(rwi_data)
    ar_max = _alias_kwarg(kwargs, "max_lag", "ar_max", ar_max, "chron_ars")
    _reject_kwargs(kwargs)
    prewhiten_method = _normalize_choice(prewhiten_method, _KNOWN_PREWHITEN_METHODS,
                                         "prewhiten_method")
    ars_method = _normalize_choice(ars_method, _KNOWN_ARS_METHODS, "ars_method")
    from .stabilize import stabilize_chron, _prepare_stabilize_kwargs
    stabilize, skw = _prepare_stabilize_kwargs(stabilize, stabilize_kwargs)

    rwi_data = _drop_allnan_columns(rwi_data, "rwi_data")
    x = rwi_data.to_numpy(dtype=float)
    n_series = x.shape[1]
    samp_depth = rwi_data.notna().sum(axis=1).to_numpy()

    # standard chronology
    std_crn = _aggregate(x, biweight)

    # pooled AR: ACF, AR coefs, AIC, selected order
    out_ar = _pooled_ar(x, ar_max, first_aic_min)
    p = out_ar["order"]
    if p > 0:
        # out_ar["arcoefs"] holds dplR's negated coefs (ARcoefs); re-reddening
        # takes the positive AR coefficients.
        phi = -out_ar["arcoefs"][p - 1, :p]
    else:
        phi = np.empty(0)

    if verbose:
        _print_summary(out_ar, ar_max)

    # prewhiten each series individually to the pooled order (ARSTAN's robar /
    # albino per series; backcast initials when backcast=True so no leading
    # years are lost). A series with n <= p cannot be whitened and becomes NaN.
    rwi_clean = np.column_stack(
        [_whiten_series(x[:, s], p, prewhiten_method, backcast, rwi_data.columns[s])
         for s in range(n_series)])

    # residual chronology: mean of prewhitened series, prewhitened again to p
    # (again backcasting the mean, as ARSTAN's albino1 does).
    res_crn = _whiten_series(_aggregate(rwi_clean, biweight), p, prewhiten_method,
                             backcast)

    # re-reddened (ARSTAN) chronology, with the pooled AR coefficients
    if ars_method == "arstan":
        # ARSTAN commie1 (and dev dplR): put the persistence back into the
        # residual chronology itself.
        ars_crn = _post_ar(res_crn, phi, backcast)
    else:  # "dplr": released dplR <= 1.8.0 re-reddens each series, then averages
        rwi_ar = np.column_stack(
            [_post_ar(rwi_clean[:, s], phi, backcast) for s in range(n_series)])
        ars_crn = _aggregate(rwi_ar, biweight)

    out = pd.DataFrame(
        {"std": std_crn, "res": res_crn, "ars": ars_crn, "samp_depth": samp_depth},
        index=rwi_data.index,
    )

    if skw is not None:
        # residual (prewhitened) RWI matrix, for the residual/ARSTAN rbar
        # (ARSTAN's per-chronology rbar convention). Each matrix's running rbar
        # is computed once and reused (audit F5).
        resid_matrix = pd.DataFrame(rwi_clean, index=rwi_data.index,
                                    columns=rwi_data.columns)
        stab_std = _matrix_stabilizer(rwi_data, stabilize, skw, stabilize_chron)
        stab_res = _matrix_stabilizer(resid_matrix, stabilize, skw, stabilize_chron)

        out["std_vsc"] = stab_std(std_crn, rwi_data.index)
        out["res_vsc"] = stab_res(res_crn, rwi_data.index)
        if ars_method == "arstan":
            # ARSTAN order: stabilize the residual chronology, THEN re-redden --
            # the same pipeline as the unstabilized ars.
            out["ars_vsc"] = _post_ar(out["res_vsc"].to_numpy(), phi, backcast)
        else:  # "dplr": ars is already reddened -- stabilize it post-hoc
            out["ars_vsc"] = stab_res(ars_crn, rwi_data.index)

        # keep each stabilized column beside its base chronology
        out = out[["std", "std_vsc", "res", "res_vsc", "ars", "ars_vsc",
                   "samp_depth"]]

    return _carry_attrs(rwi_data, out, kind="crn")


def _matrix_stabilizer(matrix, stabilize, skw, stabilize_chron):
    """Return ``f(values, index) -> vsc array`` that stabilizes chronologies with
    the rbar of ``matrix``, computing that (running) rbar once on the first call
    and reusing it afterwards."""
    cache = {}

    def run(values, index):
        series = pd.Series(np.asarray(values, dtype=float), index=index)
        if stabilize == "spline":
            return stabilize_chron(series, stabilize, **skw)["vsc"].to_numpy()
        if not cache:
            info = stabilize_chron(series, stabilize, rwi=matrix, return_info=True, **skw)
            cache["rbar"] = np.asarray(info["rbar"], dtype=float)
            cache["depth"] = np.asarray(info["samp_depth"], dtype=float)
            return info["vsc"]["vsc"].to_numpy()
        kw = dict(skw)
        mode = _normalize_choice(kw.get("rbar_mode", "running"),
                                 ("running", "constant"), "rbar_mode")
        if mode == "running":
            kw["running_rbar"] = cache["rbar"]
        else:
            kw["rbar"] = float(cache["rbar"][0])
        return stabilize_chron(series, stabilize, samp_depth=cache["depth"],
                               **kw)["vsc"].to_numpy()

    return run


def _aggregate(mat, biweight):
    """Per-year aggregation across series: biweight robust mean or arithmetic
    mean, ignoring NaN. An all-NaN year yields NaN. The one row aggregator the
    chronology functions share (audit E16)."""
    mat = np.asarray(mat, dtype=float)
    if biweight:
        return tbrm_rows(mat)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows -> NaN
        return np.nanmean(mat, axis=1)


def _pooled_ar(x, max_lag, first_aic_min):
    """Pooled autoregression across all series (Cook's pooled AR).

    Vectorized port of dplR's pooledAR: accumulates a pooled product-sum across
    every ordered series pair and lag, converts to a pooled ACF, then to AR
    coefficients (Levinson-Durbin, shared with autoreg), and selects the order by
    AIC. The per-pair common-overlap compression + position lag follows
    dplR/FORTRAN.
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
    # Levinson-Durbin on the pooled ACF: row p-1 holds the AR(p) coefficients
    # (positive convention); dplR stores the negated coefs (ARcoefs).
    phis, _v = _levinson(acf)
    arcoefs_pos = np.zeros((max_lag, max_lag))
    for k in range(1, max_lag + 1):
        arcoefs_pos[k - 1, :k] = phis[k]
    arcoefs = -arcoefs_pos

    # prediction error variance per order, then AIC (matches dplR's vp_shortcut)
    vp = np.empty(max_lag + 1)
    vp[0] = r0
    vp[1:] = r0 + arcoefs @ product_sum[1:]
    aic = n_years * np.log(vp) + 2.0 * np.arange(1, max_lag + 2)

    if first_aic_min:
        order = _first_aic_min(list(aic), what="The pooled AIC")
    else:
        order = int(np.argmin(aic))

    return {"acf": acf, "arcoefs": arcoefs, "aic": aic, "order": order}


def _whiten_series(series, p, method="ar.yw", backcast=True, name=None):
    """Prewhiten one (NaN-aware) series with a fixed-order AR(p) model fitted to
    that series: Yule-Walker (``"ar.yw"``, dplR's ar(aic=FALSE, order.max=p)) or
    Burg (``"burg"``, ARSTAN) through `autoreg._prewhiten()`, or an
    ARIMA(p,0,0) CSS-ML fit. Residuals + series mean at the original positions.

    With ``backcast=False`` (dplR behaviour) the first p residuals are NaN, as
    in R's ar() residuals. With ``backcast=True`` (ARSTAN) the p pre-sample
    values are synthesized by running the AR model in reverse (``bckcst``), so
    residuals are defined for every year; years t >= p are identical either way.
    A series with p or fewer measured values cannot be fitted and is returned
    all-NaN with a warning (audit B11)."""
    series = np.asarray(series, dtype=float)
    if p == 0:
        return series.copy()
    n_obs = int(np.count_nonzero(~np.isnan(series)))
    if n_obs <= p:
        if name is not None:
            _warn("chron_ars: series " + str(name) + " has " + str(n_obs)
                  + " value(s), not more than the pooled AR order " + str(p)
                  + "; it is left out of the residual chronology.", stacklevel=4)
        return np.full(series.shape, np.nan)
    if method == "arima.CSS-ML":
        return _prewhiten_arima(series, p, backcast=backcast)
    estimator = "burg" if method == "burg" else "yw"
    return _prewhiten(series, estimator=estimator, order=p, backcast=backcast)


def _prewhiten_arima(series, p, backcast=False):
    """Prewhiten one series with an ARIMA(p,0,0) CSS-ML fit. Matches dplR's
    arima(..., method="CSS-ML") within tolerance (different optimizer).

    ``backcast`` is accepted for a uniform call signature but has no effect here:
    the state-space (Kalman) filter already returns full-length residuals, so
    there are no leading-NaN years to backcast."""
    if p == 0:
        return series.copy()
    from statsmodels.tsa.arima.model import ARIMA

    mask = np.isnan(series)
    x2 = series[~mask]
    m = x2.mean()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ARIMA(x2, order=(p, 0, 0), trend="c").fit(method="statespace")
    out = series.copy()
    out[~mask] = fit.resid + m
    return out


def _post_ar(series, phi, backcast=True):
    """Re-redden a prewhitened (NaN-aware) series by reintroducing the pooled AR
    persistence: ``y[t] = e[t] + sum_j phi[j] y[t-j]`` on the non-NaN values,
    through `autoreg._redden()` (ARSTAN's ``commie``). With ``backcast=True``
    the p pre-sample values come from ARSTAN's ``bckcst`` run from the start of
    the series; with ``backcast=False`` they are zero. (dplR's postAR took them
    from the END of the reversed record -- audit A12 -- which dplPy no longer
    reproduces; only roughly the first 10 years of a series differ.)
    """
    series = np.asarray(series, dtype=float)
    phi = np.asarray(phi, dtype=float)
    out = series.copy()
    if phi.size == 0:
        return out
    mask = ~np.isnan(series)
    if mask.sum() == 0:
        return out
    out[mask] = _redden(series[mask], phi, backcast=backcast)
    return out


def _print_summary(out_ar, max_lag):
    labels = ["ar(" + str(k) + ")" for k in range(max_lag + 1)]
    print("Pooled AR Summary")
    print("ACF")
    print(pd.Series(out_ar["acf"], index=labels).to_string())
    print("AR Coefficients (dplR sign convention, row ar(p) = AR(p) fit)")
    coefs = pd.DataFrame(out_ar["arcoefs"], index=labels[1:],
                         columns=["lag" + str(k) for k in range(1, max_lag + 1)])
    print(coefs.to_string(float_format=lambda v: "%.4f" % v))
    print("AIC")
    print(pd.Series(out_ar["aic"], index=labels).to_string())
    print("Selected Order")
    print(out_ar["order"])
