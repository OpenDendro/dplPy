__copyright__ = """
   dplPy for tree ring width time series analyses
   Copyright (C) 2025  OpenDendro

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
from ..io.readers import readers
from .stats import stats

def plot(inp, type="line"):
    """
    Generate plots of tree ring width data using various visualization styles.
    
    This function creates different types of plots for tree ring width data,
    supporting both direct DataFrame input and file path input. It provides
    multiple visualization options tailored for dendrochronological analysis.
    
    Parameters
    ----------
    inp : pandas.DataFrame or str
        Input tree ring data. Can be either:
        - pandas.DataFrame: Tree ring data with year index and series columns
        - str: File path to CSV or RWL file that will be read automatically
    type : str, optional
        Type of plot to generate, by default "line". Supported options:
        - "line": Standard line plot showing all series overlaid
        - "spag": Spaghetti plot with series stacked vertically, sorted by start date
        - "seg": Segment plot showing horizontal lines for each series' time span
    
    Returns
    -------
    None
        Displays the plot but does not return data. Returns None if input type is invalid.
    
    Notes
    -----
    Plot types explanation:
    
    **Line plot**: Traditional overlay of all series on the same axes, useful for:
    - Comparing growth patterns across series
    - Identifying common signals and anomalous years
    - Quick visual assessment of data quality
    
    **Spaghetti plot**: Vertically stacked series plots, useful for:
    - Visualizing individual series without overlap
    - Comparing series start/end dates and lengths
    - Identifying missing data patterns
    - Series are sorted by start date for easier interpretation
    
    **Segment plot**: Horizontal line representation, useful for:
    - Visualizing temporal coverage of each series
    - Assessing dataset completeness through time
    - Planning sampling strategies or chronology development
    
    Examples
    --------
    >>> import dplpy as dpl
    >>> # Load data
    >>> data = dpl.readers('../tests/data/rwl/ca533.rwl')
    >>> 
    >>> # Create standard line plot
    >>> dpl.plot(data)
    >>> 
    >>> # Create spaghetti plot for selected series
    >>> selected_series = data[['CAM011', 'CAM021', 'CAM031']]
    >>> dpl.plot(selected_series, type='spag')
    >>> 
    >>> # Create segment plot to show temporal coverage
    >>> dpl.plot(data, type='seg')
    >>> 
    >>> # Plot directly from file
    >>> dpl.plot('../tests/data/rwl/ca533.rwl', type='line')
    
    See Also
    --------
    spag_plot : Generate spaghetti plot with stacked series
    seg_plot : Generate segment plot showing temporal coverage
    readers : Read tree ring data from files
    stats : Calculate series statistics used for plot ordering
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

def spag_plot(data):
    """
    Generate a spaghetti plot with vertically stacked tree ring series.
    
    This function creates a visualization where each tree ring series is plotted
    as a separate line with vertical offsets, allowing individual series patterns
    to be examined without overlap. Series are automatically sorted by start date
    and displayed with equal vertical spacing.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Tree ring width data with year index and series columns.
        Each column represents a different tree or core series.
    
    Returns
    -------
    None
        Displays the plot using matplotlib.pyplot.show().
    
    Notes
    -----
    Plot characteristics:
    - Series are sorted by first year (earliest to latest from bottom to top)
    - Each series is offset vertically by 2× the overall mean ring width
    - Y-axis labels show series names at their respective offset positions
    - X-axis shows calendar years
    - Uses seaborn dark grid style for enhanced readability
    - Figure size automatically adjusts based on data timespan and series count
    
    The spaghetti plot is particularly useful for:
    - Identifying crossdating errors or measurement problems in individual series
    - Comparing growth patterns without visual interference between series  
    - Assessing data quality and consistency across the dataset
    - Visualizing temporal coverage and missing data patterns
    
    Examples
    --------
    >>> import dplpy as dpl
    >>> data = dpl.readers('../tests/data/rwl/ca533.rwl')
    >>> # Create spaghetti plot 
    >>> dpl.spag_plot(data)
    >>> # Or use the main plot function
    >>> dpl.plot(data, type='spag')
    
    See Also
    --------
    plot : Main plotting function with multiple visualization options
    seg_plot : Alternative plot showing only temporal coverage
    stats : Function used to determine series start dates for sorting
    """
    # obtain a list of series names sorted by the start date
    data_stats = stats(data)
    series_by_start_date = data_stats.sort_values(by='first')['series']

    # Change the style of plot
    # Note: 'seaborn-darkgrid' style may be deprecated in newer matplotlib versions
    plt.style.use('seaborn-darkgrid')

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
    """
    Generate a segment plot showing temporal coverage of tree ring series.
    
    This function creates a visualization where each tree ring series is represented
    as a horizontal line showing only its temporal extent (start to end year).
    This type of plot is ideal for assessing dataset completeness and planning
    dendrochronological analyses.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Tree ring width data with year index and series columns.
        Each column represents a different tree or core series.
    
    Returns
    -------
    None
        Displays the plot using matplotlib.pyplot.show().
    
    Notes
    -----
    Plot characteristics:
    - Series are sorted by first year (earliest to latest from bottom to top)
    - Each series shows as a horizontal line spanning its date range
    - Y-axis labels show series names
    - X-axis shows calendar years
    - Uses seaborn dark grid style for enhanced readability
    - Figure size automatically adjusts based on data timespan and series count
    
    The segment plot is particularly useful for:
    - Visualizing sample depth through time
    - Identifying periods with good vs. poor replication
    - Planning additional sampling to fill temporal gaps
    - Assessing suitability for different types of analyses
    - Understanding dataset structure before chronology development
    
    **Note**: Current implementation may have a bug where series values are
    set to zero (data - data = 0), showing only temporal coverage without
    actual measurements.
    
    Examples
    --------
    >>> import dplpy as dpl
    >>> data = dpl.readers('../tests/data/rwl/ca533.rwl')
    >>> # Create segment plot
    >>> dpl.seg_plot(data) 
    >>> # Or use the main plot function
    >>> dpl.plot(data, type='seg')
    
    See Also
    --------
    plot : Main plotting function with multiple visualization options
    spag_plot : Alternative plot showing actual ring width values
    stats : Function used to determine series start dates for sorting
    """
    # obtain a list of series names sorted by the start date
    data_stats = stats(data)
    series_by_start_date = data_stats.sort_values(by='first')['series']

    # Change the style of plot
    # Note: 'seaborn-darkgrid' style may be deprecated in newer matplotlib versions
    plt.style.use('seaborn-darkgrid')

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