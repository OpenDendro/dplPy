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
# Description: Creates a mean value chronology for a dataset, typically the ring width indices of a detrended series.
#              Takes three optional arguments 'biweight', 'prewhiten', and 'plot'. They determine whether to find means using tukey's
#              biweight robust mean (default True), whether to prewhiten data by fitting to an AR model (default False), and
#              whether to plot the results of the chronology (default True).


import matplotlib.pyplot as plt
import pandas as pd
from ._validate import _require_dataframe
from ._plot_style import style_axes, finalize_font, ACCENT
import numpy as np
from .tbrm import tbrm
from .autoreg import ar_func

# Main function for creating chronology of series. Formats input, prewhitens if necessary
# and produces output mean value chronology in a dataframe.
def chron(rwi_data: pd.DataFrame, biweight=True, prewhiten=False, plot=True,
          max_lag=10, aic=True, ar_method="yw", first_aic_min=False,
          stabilize=None, stabilize_kwargs=None):
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
    max_lag : int, default 10
        maximum AR order considered when prewhiten is True. To reproduce dplR's
        chron() exactly, pass max_lag=floor(10*log10(n)).
    aic : boolean, default True
        when prewhiten is True, select the AR order by AIC up to max_lag; if False,
        fit a fixed AR of order max_lag.
    ar_method : {"yw", "ols"}, default "yw"
        AR estimator for prewhitening. "yw" (Yule-Walker) matches R's ar() default;
        "ols" matches R's ar(method="ols").
    first_aic_min : boolean, default False
        AR order-selection rule when aic is True. False takes the global AIC minimum
        (reproduces dplR's chron()); True takes the first local AIC minimum (more
        parsimonious).
    stabilize : {None, "rbar", "spline", "both"}, default None
        if set, also variance-stabilize each chronology and return the stabilized
        versions in extra *_vsc columns (see Notes). Method families are as in
        :func:`stabilize_chron`.
    stabilize_kwargs : dict, optional
        extra keyword arguments forwarded to :func:`stabilize_chron` when stabilize
        is set. The chronology, column, source RWI matrix, and sample depth are
        supplied automatically and may not be overridden here.

    Returns
    -------
    data : pandas dataframe
        a mean-value chronology indexed by year, with columns std (standard
        chronology), res (residual chronology, only when prewhiten is True), and
        samp_depth. When stabilize is set, a stabilized *_vsc column is added next
        to each base chronology.

    Notes
    -----
    **Variance stabilization (``stabilize``).** The standard chronology is
    stabilized with the rbar of the standard RWI matrix and the residual chronology
    with the rbar of the residual matrix; both restandardize to the full period
    (see :func:`stabilize_chron`). For the re-reddened ARSTAN chronology, use
    :func:`chron_ars` with ``stabilize=``.

    Examples
    --------
    >>> import dplpy as dpl
    >>> rwi_data = dpl.detrend(dpl.readers("../tests/data/csv/file.csv"))
    >>> dpl.chron(rwi_data)
    >>> dpl.chron(rwi_data, prewhiten=True)

    References
    ----------
    .. [1] https:/opendendro.org/dplpy-man/#chron
    """
    _require_dataframe(rwi_data)
    from .stabilize import stabilize_chron, _prepare_stabilize_kwargs
    skw = _prepare_stabilize_kwargs(stabilize, stabilize_kwargs)

    chron_data = {}
    for series in rwi_data:
        series_data = rwi_data[series].dropna()
        for year, value in series_data.items():
            if year not in chron_data:
                chron_data[year] = [1, value]
            else:
                chron_data[year][0] += 1
                chron_data[year].append(value)
    
    years, means, depths = get_chron_info(chron_data, biweight)
    chron_res = pd.DataFrame(data={"Year":years})
    chron_res = pd.concat([chron_res, pd.Series(data=means, name="std")], axis=1)

    res_matrix = None
    if prewhiten:
        # capture the prewhitened (residual) matrix when stabilizing, so the
        # residual chronology can use the residual matrix's rbar (ARSTAN's
        # per-chronology rbar convention).
        if skw is not None:
            whitened_means, res_matrix = get_whitened_chron_info(
                rwi_data, chron_data, biweight, max_lag, aic, method=ar_method,
                first_aic_min=first_aic_min, return_matrix=True)
        else:
            whitened_means = get_whitened_chron_info(rwi_data, chron_data, biweight,
                                                     max_lag, aic, method=ar_method,
                                                     first_aic_min=first_aic_min)
        chron_res = pd.concat([chron_res, pd.Series(data=whitened_means, name="res")], axis=1)
    else:
        whitened_means = None

    chron_res = pd.concat([chron_res, pd.Series(data=depths, name="samp_depth")], axis=1)
    chron_res.set_index('Year', inplace = True, drop = True)

    if skw is not None:
        # std_vsc: stabilize the standard chronology with the standard matrix rbar
        chron_res["std_vsc"] = stabilize_chron(
            chron_res["std"], stabilize, rwi=rwi_data, **skw)["vsc"]
        if prewhiten:
            # res_vsc: stabilize the residual chronology with the residual matrix rbar
            chron_res["res_vsc"] = stabilize_chron(
                chron_res["res"], stabilize, rwi=res_matrix, **skw)["vsc"]
        # keep each stabilized column beside its base chronology
        order = [c for c in ("std", "std_vsc", "res", "res_vsc", "samp_depth")
                 if c in chron_res.columns]
        chron_res = chron_res[order]

    if plot:
        plot_chron(years, depths, means, whitened_means)

    return chron_res

# Does the work of creating the actual chronology by finding the mean RWI for each year
# in the dataset and keeping track of the sample size.
def get_chron_info(chron_data, biweight):
    years = []
    means = []
    depths = []
    
    for year in sorted(chron_data):
        years.append(year)
        if biweight:
            means.append(tbrm(chron_data[year][1:]))
        else:
            means.append(sum(chron_data[year][1:])/chron_data[year][0])
        depths.append(chron_data[year][0])

    return years, means, depths

# Does the work of creating a chronology when the data has to be fit to an AR model
# first (i.e. prewhitened).
def get_whitened_chron_info(rwi_data, chron_data, biweight, max_lag=10, aic=True,
                            method="yw", first_aic_min=False, return_matrix=False):
    whitened_data = {}
    ar_fit_data = {}

    for series in rwi_data:
        series_data = rwi_data[series].dropna()
        ar_fit_data[series] = ar_func(series_data, max_lag=max_lag, aic=aic,
                                      method=method, first_aic_min=first_aic_min)

        for year, value in ar_fit_data[series].items():
            if year not in whitened_data:
                whitened_data[year] = [1, value]
            else:
                whitened_data[year][0] += 1
                whitened_data[year].append(value)

    whitened_means = []
    for year in sorted(chron_data):
        if year not in whitened_data:
            whitened_means.append(np.nan)
        elif biweight:
            whitened_means.append(tbrm(whitened_data[year][1:]))
        else:
            whitened_means.append(sum(whitened_data[year][1:])/whitened_data[year][0])

    if return_matrix:
        # assemble the prewhitened series into a years x series matrix (the
        # residual RWI matrix), aligned to rwi_data's year index. Each column is
        # ar_func's residual+mean series; years a series does not cover are NaN.
        res_matrix = pd.DataFrame(ar_fit_data).reindex(rwi_data.index)
        return whitened_means, res_matrix
    return whitened_means

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
