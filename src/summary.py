from __future__ import print_function

__copyright__ = """
   dplPy for tree ring width time series analyses
   Copyright (C) 2021  OpenDendro

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

# Date: 10/22/2021
# Author: Anushka Bande
# Title: summary.py
# Description: Generates Summary statistics for Tucson format and CSV format files
# usage: python dplpy summary --input <*.rwl> 

# Create Summaries for Tucson (*rwl) files
import pandas as pd
import numpy as np
import statistics

from readers import readers
def summary(inp):

    if isinstance(inp, dict):
        series_data = inp
    elif isinstance(inp, str):
        series_data = readers(inp)
    else:
        "Invalid command"
        return
    
    
    print("\nSummary:")

    print("series".rjust(10), "first", "last", "year", "mean".rjust(5), "median", "stdev", "gini")

    line = 1
    for key, value in series_data.items():
        print(str(line).ljust(4), end="")
        print(key, str(value[0]).rjust(5), str(value[0] + value[1] - 1).rjust(4), 
                str(value[1]).rjust(4), "{:.3f}".format(value[2]/value[1]).rjust(4),
                    "{:.3f}".format(statistics.median(value[3])).rjust(6),
                        "{:.3f}".format(statistics.stdev(value[3])),
                            get_auto_correlation(value[3]))
        line += 1
    print()

def get_auto_correlation(data_array):
    # might need to work on more efficient solution
    # Mean absolute difference
    mad = np.abs(np.subtract.outer(data_array, data_array)).mean()
    # Relative mean absolute difference
    rmad = mad/np.mean(data_array)
    # Gini coefficient
    g = 0.5 * rmad
    return "{:.3f}".format(g)