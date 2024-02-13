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

# Date: 5/27/2022
# Author: Ifeoluwa Ale
# Title: summary.py
# Description: Generates a summary of each series recorded in Tucson format and CSV format files
# example usage:
# >>> import dplpy as dpl 
# >>> data = dpl.readers("../tests/data/csv/file.csv")
# >>> dpl.summary(data)
#
# >>> dpl.summary("../tests/data/csv/file.csv")
# >>> Note: for file pathname inputs, only CSV and RWL file formats are accepted

# Create Summaries for Tucson (*rwl) files
import pandas as pd

from readers import readers
def summary(inp: pd.DataFrame | str):
    """ Summarizes a chronology
    
    Extended Summary
    ----------------
    This function summarizes a chronology from an array
    
    Parameters
    ----------
    data : str
           a data file (.CSV or .RWL) or a pandas dataframe imported from dpl.readers().
   
    Returns
    -------
    data : pandas dataframe
    
    Examples
    --------
    >>> import dplpy as dpl
    >>> data = dpl.readers("../tests/data/csv/file.csv")
    >>> dpl.summary(data)
    
    References
    ----------
    .. [1] https:/opendendro.org/dplpy-man/#summary
    
    """
    if isinstance(inp, pd.DataFrame):
        series_data = inp
    elif isinstance(inp, str):
        series_data = readers(inp)
    else:
        errorMsg = """
Unable to generate summary report. Input must be string path to file to be read
or Dataframe object.

Note: for file pathname inputs, only CSV and RWL file formats are accepted

Example usages:

>>> import dplpy as dpl
>>> data = dpl.readers('../tests/data/csv/file.csv')
>>> dpl.summary(data)
>>> dpl.summary('../tests/data/csv/file.csv')

"""
        raise TypeError(errorMsg)

    summary = series_data.describe()
    return summary