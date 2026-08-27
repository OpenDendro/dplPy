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
import numpy as np
from .tbrm import tbrm
from .autoreg import ar_func

# Main function for creating chronology of series. Formats input, prewhitens if necessary
# and produces output mean value chronology in a dataframe.
def chron(rwi_data: pd.DataFrame, biweight=True, prewhiten=False, plot=True,
          max_lag=5, aic=True):
    """Creates a mean value chronology for a dataset of tree-ring widths.

    Extended Summary
    ----------------
    Creates a mean value chronology for a dataset, typically the ring width
    indices of a detrended series. Determines whether to find means using Tukey's
    bi-weight robust mean (default 'True'), whether to prewhiten data by fitting
    to an AR model (default 'False'), and whether to plot the results of the
    chronology (default 'True'). When prewhitening, the AR-model order can be
    controlled with ``max_lag`` and ``aic`` (mirroring dplR's chron(), which
    forwards these to ar() as order.max and aic).

    Parameters
    ----------
    data : pandas dataframe
        a pandas dataframe imported from dpl.readers()
    biweight : boolean, default True
        use Tukey's bi-weight robust mean
    prewhiten : boolean, default False
        run pre-whitening on the time series before building
        chronology
    plot : boolean, default True
        plot the results
    max_lag : int, default 5
        when ``prewhiten`` is True, the maximum AR order considered (dplR's
        ar() order.max). Note dplR's own default is floor(10*log10(n)); dplPy
        keeps 5 for backward compatibility.
    aic : boolean, default True
        when ``prewhiten`` is True, select the AR order by AIC up to ``max_lag``
        (dplR's ar() aic=TRUE). If False, fit a fixed AR of order ``max_lag``.

    Returns
    -------
    data: pandas dataframe
        a mean-value chronology indexed by year. Columns: ``std`` (the standard
        chronology), ``res`` (the residual/AR-prewhitened chronology; only when
        ``prewhiten`` is True), and ``samp_depth`` (the sample depth). These
        names match chron_ars() and mirror dplR's std/res/samp.depth.
    
    Examples
    --------
    >>> import dplpy as dpl 
    >>> data = dpl.readers("../tests/data/csv/file.csv")
    >>> rwi_data = dpl.detrend(data)
    >>> dpl.chron(rwi_data)
    >>> dpl.chron(rwi_data, prewhiten=True)
    >>> chron_data = dpl.chron(rwi_data, biweight=False, plot=False)
    
    References
    ----------
    .. [1] https:/opendendro.org/dplpy-man/#chron
    
    """
    _require_dataframe(rwi_data)
    
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

    if prewhiten:
        whitened_means = get_whitened_chron_info(rwi_data, chron_data, biweight,
                                                 max_lag, aic)
        chron_res = pd.concat([chron_res, pd.Series(data=whitened_means, name="res")], axis=1)
    else:
        whitened_means = None

    chron_res = pd.concat([chron_res, pd.Series(data=depths, name="samp_depth")], axis=1)
    chron_res.set_index('Year', inplace = True, drop = True)

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
def get_whitened_chron_info(rwi_data, chron_data, biweight, max_lag=5, aic=True):
    whitened_data = {}
    ar_fit_data = {}

    for series in rwi_data:
        series_data = rwi_data[series].dropna()
        ar_fit_data[series] = ar_func(series_data, max_lag=max_lag, aic=aic)

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
    return whitened_means

# Plots the data created by the chronology
def plot_chron(years, depths, means, whitened_means):
    # create figure and axis objects with subplots()
    fig,ax = plt.subplots()

    # When prewhitening is on, chron() plots the residual (res) chronology rather
    # than the standard (std) one; make that explicit in the title and y-axis so
    # the figure is not mistaken for the standard chronology.
    if whitened_means is not None:
        y_val = whitened_means
        y_label = "res"
        title = "Residual (prewhitened) chronology"
    else:
        y_val = means
        y_label = "std"
        title = "Standard chronology"

    # make plot of RWI means
    ax.plot(years, y_val, "k-")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Year", fontsize = 14)
    ax.set_ylabel(y_label, fontsize=14)

    # twin object for two different y-axis on the sample plot
    ax2=ax.twinx()
    # make plot of sample depths
    ax2.fill_between(years, depths, color=((0.2, 0.6, 0.9, 0.3)))
    ax2.set_ylabel("samp_depth",fontsize=14)
    fig.set_size_inches(14, 8)
    plt.show()
