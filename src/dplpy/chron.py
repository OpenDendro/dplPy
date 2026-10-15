from __future__ import print_function

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
# Title: chron.py
# Project: OpenDendro dplPy
# Description: Creates a mean value chronology for a dataset, typically the ring
#              width indices of a detrended series: the (biweight) mean of the
#              series in each year, optionally with a residual chronology from
#              AR-prewhitened series, a bootstrap confidence interval, and
#              variance-stabilized versions. Mirrors dplR's chron().


import matplotlib.pyplot as plt
import pandas as pd
from ._validate import (_require_dataframe, _alias_kwarg, _reject_kwargs,
                        _normalize_choice, _carry_attrs, _warn)
from ._plot_style import style_axes, finalize_font, ACCENT
import numpy as np
import scipy.stats
from .autoreg import _prewhiten
from .chron_ars import _aggregate

# Main function for creating chronology of series. Formats input, prewhitens if necessary
# and produces output mean value chronology in a dataframe.
def chron(rwi_data: pd.DataFrame, biweight=True, prewhiten=False, plot=True,
          ar_max=10, aic=True, ar_method="yw", first_aic_min=True, backcast=False,
          stabilize=None, stabilize_kwargs=None,
          ci=False, conf=0.95, R=100, seed=None, **kwargs):
    """Build a mean-value chronology from ring-width indices.

    Averages the series (detrended RWI) year by year into a standard chronology,
    optionally prewhitening each series with an AR model first to also produce a
    residual chronology. Mirrors dplR's chron().

    Parameters
    ----------
    rwi_data : pandas dataframe
        detrended ring-width indices (e.g. from dpl.detrend()).
    biweight : boolean, default True
        average with Tukey's biweight robust mean if True, else the arithmetic mean.
    prewhiten : boolean, default False
        prewhiten each series with an AR model to also build a residual chronology.
    plot : boolean, default True
        plot the chronology.
    ar_max : int, default 10
        maximum AR order considered when prewhiten is True. To reproduce dplR's
        chron() exactly, pass ar_max=floor(10*log10(n)) and first_aic_min=False.
        (``max_lag=`` is accepted as a deprecated alias.)
    aic : boolean, default True
        when prewhiten is True, select the AR order by AIC up to ar_max; if False,
        fit a fixed AR of order ar_max.
    ar_method : {"yw", "burg", "ols"}, default "yw"
        AR estimator for prewhitening (case-insensitive). "yw" (Yule-Walker)
        matches R's ar() default; "burg" is ARSTAN's estimator; "ols" matches R's
        ar(method="ols").
    first_aic_min : boolean, default True
        AR order-selection rule when aic is True. True takes the first local AIC
        minimum (ARSTAN's rule, dplPy's package default); False takes the global
        AIC minimum, which is what R's ar() and dplR's chron() do.
    backcast : boolean, default False
        when prewhiten is True, fill each series' first ``p`` residuals with
        ARSTAN-style backcast initials so no years are lost. False (default)
        leaves them NaN, as R's ar() residuals and dplR's chron() do.
    stabilize : {None, "rbar", "spline", "both"}, default None
        if set, also variance-stabilize each chronology and return the stabilized
        versions in extra *_vsc columns (see Notes). Method families are as in
        `stabilize_chron()`.
    stabilize_kwargs : dict, optional
        extra keyword arguments forwarded to `stabilize_chron()` when stabilize
        is set. The chronology, column, source RWI matrix, and sample depth are
        supplied automatically and may not be overridden here.
    ci : boolean, default False
        also compute a bootstrap confidence interval for the standard chronology,
        adding `lower_ci` and `upper_ci` columns. Each year's available series are
        resampled `R` times and the normal-approximation interval of the bootstrap
        means is taken. Years with fewer than two series get NaN limits (with a
        warning). The interval is only for the standard chronology.
    conf : float, default 0.95
        confidence level, used when `ci` is True.
    R : int, default 100
        number of bootstrap replicates per year, used when `ci` is True (dplR's
        chron.ci default).
    seed : int or None, default None
        seed for the `ci` bootstrap, for reproducible intervals.

    Returns
    -------
    data : pandas dataframe
        a mean-value chronology indexed by year (the full year index of
        ``rwi_data``), with columns std (standard chronology), res (residual
        chronology, only when prewhiten is True), and samp_depth. A year with
        no series present has NaN chronology values and samp_depth 0, as in
        dplR. With ci=True, lower_ci and upper_ci columns for the standard
        chronology. When stabilize is set, a stabilized *_vsc column is added
        next to each base chronology.

    Notes
    -----
    **Prewhitening.** Each series is whitened with `autoreg._prewhiten()`, the
    package-wide prewhitening (Yule-Walker, ``ar_max=10``, first local AIC
    minimum). This differs from R's ``ar()``/dplR, which take the global AIC
    minimum up to ``floor(10*log10(n))``; pass ``first_aic_min=False`` and
    ``ar_max`` accordingly to reproduce dplR.

    **Variance stabilization (``stabilize``).** The standard chronology is
    stabilized with the rbar of the standard RWI matrix and the residual chronology
    with the rbar of the residual matrix; both restandardize to the full period
    (see `stabilize_chron()`). For the re-reddened ARSTAN chronology, use
    `chron_ars()` with ``stabilize=``.

    Examples
    --------
    >>> import dplpy as dpl
    >>> rwi_data = dpl.detrend(dpl.readers("../tests/data/csv/file.csv"))
    >>> dpl.chron(rwi_data)
    >>> dpl.chron(rwi_data, prewhiten=True)

    References
    ----------
    .. [1] https://opendendro.github.io/dplPy/reference/chronology/#chron
    """
    _require_dataframe(rwi_data)
    ar_max = _alias_kwarg(kwargs, "max_lag", "ar_max", ar_max, "chron")
    _reject_kwargs(kwargs)
    ar_method = _normalize_choice(ar_method, ("yw", "burg", "ols"), "ar_method")
    from .stabilize import stabilize_chron, _prepare_stabilize_kwargs
    stabilize, skw = _prepare_stabilize_kwargs(stabilize, stabilize_kwargs)

    x = rwi_data.to_numpy(dtype=float)
    depths = rwi_data.notna().sum(axis=1).to_numpy()
    means = _aggregate(x, biweight)
    chron_res = pd.DataFrame({"std": means}, index=rwi_data.index)

    res_matrix = None
    whitened_means = None
    if prewhiten:
        res_matrix = prewhiten_matrix(rwi_data, ar_max=ar_max, aic=aic,
                                      method=ar_method, first_aic_min=first_aic_min,
                                      backcast=backcast)
        whitened_means = _aggregate(res_matrix.to_numpy(dtype=float), biweight)
        chron_res["res"] = whitened_means

    chron_res["samp_depth"] = depths

    if ci:
        lo, hi = _bootstrap_ci(rwi_data, biweight, conf, R, seed)
        chron_res["lower_ci"] = lo.to_numpy()
        chron_res["upper_ci"] = hi.to_numpy()

    if skw is not None:
        # std_vsc: stabilize the standard chronology with the standard matrix rbar
        chron_res["std_vsc"] = stabilize_chron(
            chron_res["std"], stabilize, rwi=rwi_data, **skw)["vsc"]
        if prewhiten:
            # res_vsc: stabilize the residual chronology with the residual matrix rbar
            chron_res["res_vsc"] = stabilize_chron(
                chron_res["res"], stabilize, rwi=res_matrix, **skw)["vsc"]

    # one final column order: each CI / stabilized column beside its base chronology
    order = [c for c in ("std", "lower_ci", "upper_ci", "std_vsc", "res", "res_vsc",
                         "samp_depth") if c in chron_res.columns]
    chron_res = chron_res[order]
    _carry_attrs(rwi_data, chron_res, kind="crn")

    if plot:
        plot_chron(chron_res.index.to_numpy(), depths, means, whitened_means)

    return chron_res


def prewhiten_matrix(rwi_data, ar_max=10, aic=True, method="yw", first_aic_min=True,
                     backcast=False):
    """Prewhiten every column of ``rwi_data`` with `autoreg._prewhiten()`.

    Returns the residual (prewhitened) RWI matrix: residuals + series mean at the
    original year positions, NaN where the input is NaN (and, with
    ``backcast=False``, in each series' first ``p`` years). ``aic=False`` fits a
    fixed AR of order ``ar_max``. The single home for "the prewhitened matrix"
    in the chronology functions (audit E18).
    """
    out = {}
    for col in rwi_data.columns:
        try:
            out[col] = _prewhiten(rwi_data[col].to_numpy(dtype=float),
                                  estimator=method, ar_max=ar_max,
                                  first_aic_min=first_aic_min,
                                  order=None if aic else ar_max, backcast=backcast)
        except ValueError as err:
            raise ValueError("prewhitening series " + str(col) + ": " + str(err)) from err
    return pd.DataFrame(out, index=rwi_data.index)


def _bootstrap_ci(rwi_data, biweight, conf, R, seed):
    """Per-year normal-approximation bootstrap CI for the standard chronology.

    Resamples each year's available series R times, takes the (biweight or
    arithmetic) mean of each replicate, and returns the normal-approximation
    interval (as dplR's chron.ci does). Returns two year-indexed Series
    (lower, upper); years with fewer than two series are NaN.
    """
    rng = np.random.default_rng(seed)
    z = scipy.stats.norm.ppf((1.0 + conf) / 2.0)
    M = rwi_data.to_numpy(dtype=float)
    lo = np.full(M.shape[0], np.nan)
    hi = np.full(M.shape[0], np.nan)
    n_thin = 0
    for i in range(M.shape[0]):
        avail = M[i][~np.isnan(M[i])]
        k = avail.size
        if k < 2:
            n_thin += int(k == 1)
            continue
        t0 = float(_aggregate(avail[None, :], biweight)[0])
        t_star = _aggregate(avail[rng.integers(0, k, size=(R, k))], biweight)
        centre = t0 - (float(np.mean(t_star)) - t0)          # boot.ci "norm"
        half = z * float(np.std(t_star, ddof=1))
        lo[i], hi[i] = centre - half, centre + half
    if n_thin:
        _warn("chron: ci=True -- %d year(s) had fewer than 2 series and "
              "were given NaN confidence limits." % n_thin, stacklevel=4)
    idx = pd.Index(np.asarray(rwi_data.index), name=rwi_data.index.name)
    return pd.Series(lo, index=idx), pd.Series(hi, index=idx)


# Plots the data created by the chronology
def plot_chron(years, depths, means, whitened_means, ax=None, show=True):
    """Plot a mean chronology with its sample depth on a twin axis.

    Pass an existing ``ax`` to draw into it, and ``show=False`` to keep the figure
    for saving; the figure and axes are returned either way.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
    else:
        fig = ax.figure

    # When prewhitening is on, chron() plots the residual (res) chronology rather
    # than the standard (std) one; make that explicit in the title and y-axis so
    # the figure is not mistaken for the standard chronology.
    if whitened_means is not None:
        y_val = whitened_means
        y_label = "res"
        title = "Residual (prewhitened) chronology"
        ref = 0.0                    # residual index oscillates about 0
    else:
        y_val = means
        y_label = "std"
        title = "Standard chronology"
        ref = 1.0                    # standard index oscillates about 1

    # sample depth first, on a twin axis, sitting BEHIND the chronology line
    ax2 = ax.twinx()
    ax2.fill_between(years, depths, color="0.6", alpha=0.28, linewidth=0,
                     zorder=1)
    ax2.set_ylabel("Sample depth")
    ax2.set_ylim(bottom=0)
    ax2.spines["top"].set_visible(False)
    # let the (transparent) chronology axis sit above the depth fill
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)

    # reference line at the chronology's expected level, then the chronology
    ax.axhline(ref, color="0.8", linewidth=0.8, zorder=2)
    ax.plot(years, y_val, color=ACCENT, linewidth=0.9, zorder=3)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Year")
    ax.set_ylabel(y_label)
    style_axes(ax, xgrid=True, hide_spines=("top",))

    fig.tight_layout()
    finalize_font(fig)
    if show:
        plt.show()
    return fig, ax
