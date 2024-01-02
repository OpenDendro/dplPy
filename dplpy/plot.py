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

# Date: 9/8/2022
# Author: Ifeoluwa Ale
# Title: plot.py
# Description: Generates plots of tree ring with data from dataframes. 
#              Currently capable of generating line (default), spag and seg plots.
#
# example usages: 
# >>> import dplpy as dpl 
# >>> data = dpl.readers("../tests/data/csv/file.csv")
# >>> data = dpl.plot(data)
# >>> data = dpl.plot(data[[SERIES_1, SERIES_2, SERIES_3]], type="seg")
# >>> data = dpl.plot("../tests/data/csv/file.csv")
# 

import pandas as pd
import matplotlib.pyplot as plt
from readers import readers
from stats import stats

def plot(inp: pd.DataFrame | str, type="seg"):
    """Plots a given dataframe
    
    Extended Description
    --------------------
    Plots a given dataframe or series of a specific dataframe in either 
    line ('default'), spaghetti ('spag') or segment ('seg') plots.
                 
    Parameters
    ----------
    data : str 
           a data frame loaded using dpl.readers()
    series : str
             a single time series within the data array
    type : str, default line
           type of plot to generate, e.g., line, spaghetti ('spag'), or segment ('seg') or 'line'.

    Returns
    -------
    plot : matplotlib.pyplot figure
           a plot of the data
    
    Examples
    --------
    >>> dpl.plot(<data>)
    # Plot series subset of dataframe with a specified plot type
    >>> dpl.plot(<data>["<series>"], type=<plot type>)
    
    References
    ----------
    .. [1] https:/opendendro.org/dplpy-man/#plot
    
    """
    if isinstance(inp, pd.DataFrame):
        series_data = inp
    elif isinstance(inp, str):
        series_data = readers(inp)
    else:
        return

    if type == "line":
        plt.plot(series_data)
    elif type == "spag":
        spag_plot(inp)
    elif type == "seg":
        seg_plot(inp)
    else:
        raise ValueError("Unsupported plot type.")

def spag_plot(data):
    # obtain a list of series names sorted by the start date
    data_stats = stats(data)
    series_by_start_date = data_stats.sort_values(by='first')['series']

    # Change the style of plot
    plt.style.use('seaborn-v0_8-darkgrid')

    years = data.index.to_numpy()

    # set width and height of the window based on the data
    dimensions = (max((years[-1] - years[0])//80, 1), max(len(data.columns)//3, 1))
    plt.figure(figsize=(dimensions))
 
    # separate plots for each series using the offset
    offset = (data.mean().mean() * 2)

    y_divisions = [] # needed to put series names on the y-axis
    num=0
    for column_name in series_by_start_date:
        num+=1
        plt.plot(years, data[column_name].to_numpy() + (offset * (num-1)), marker='', linewidth=1, alpha=0.9, color='k')
        y_divisions.append(offset*(num-1))

    # set y-axis to display series names at equal intervals, and x-axis to display years
    plt.yticks(y_divisions, series_by_start_date)
    plt.xlabel("Year")

    # Show the graph
    plt.show()

def seg_plot(data):
    # obtain a list of series names sorted by the start date
    data_stats = stats(data)
    series_by_start_date = data_stats.sort_values(by='first')['series']

    # Change the style of plot
    plt.style.use('seaborn-v0_8-darkgrid')

    years = data.index.to_numpy()

    # set width and height of the window based on the data
    dimensions = (max((years[-1] - years[0])//80, 1), max(len(data.columns)//3, 1))
    plt.figure(figsize=(dimensions))
 
    # separate plots for each series using the offset
    offset = (data.mean().mean() * 2)

    y_divisions = [] # needed to put series names on the y-axis
    num=0
    for column_name in series_by_start_date:
        num+=1
        plt.plot(years, (data[column_name].to_numpy() - data[column_name].to_numpy()) + (offset * (num-1)), marker='', linewidth=1, alpha=0.9, color='k')
        y_divisions.append(offset*(num-1))

    # set y-axis to display series names at equal intervals, and x-axis to display years
    plt.yticks(y_divisions, series_by_start_date)
    plt.xlabel("Year")

    # Show the graph
    plt.show()