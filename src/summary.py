from __future__ import print_function
from xml.etree.ElementInclude import DEFAULT_MAX_INCLUSION_DEPTH

__copyright__ = """
   dplPy for tree ring width time series analyses
   Copyright (C) 2022  OpenDendro

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

# Date: 3/9/2021
# Author: Ifeoluwa Ale
# Title: summary.py
# Description: Generates Summary statistics for Tucson format and CSV format files
# example usage:
# >>> import dplpy as dpl 
# >>> data = dpl.readers("../tests/data/csv/file.csv")
# >>> dpl.summary(data)
#
# >>> dpl.summary("../tests/data/csv/file.csv")
# >>> Note: for file pathname inputs, only CSV and RWL file formats are accepted

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
        print("\nUnable to generate summary report. Invalid input")
        print("Note: for file pathname inputs, only CSV and RWL file formats are accepted\n")
        print("example usages:")
        print(">>> import dplpy as dpl")
        print(">>> data = dpl.readers('../tests/data/csv/file.csv')")
        print(">>> dpl.summary(data)\n")
        print(">>> dpl.summary('../tests/data/csv/file.csv')")        
        return
    
    # for potential better implementation, ignore
    #df = pd.DataFrame.from_dict(series_data, orient='index')
    #print(df)
    #print()

    print("\nSummary:")

    print("series".rjust(10), "first", "last", "year", "mean".rjust(5), 
            "median", "stdev", "skew".rjust(6), "gini".rjust(5))

    line = 1
    for key, value in series_data.items():
        print(str(line).ljust(4), end="")
        print(key, str(value[0]).rjust(5), str(value[0] + value[1] - 1).rjust(4), 
                str(value[1]).rjust(4), "{:.3f}".format(value[2]/value[1]).rjust(4),
                    "{:.2f}".format(statistics.median(value[3])).rjust(6),
                        "{:.3f}".format(statistics.stdev(value[3])),
                            get_skew(value[3]).rjust(6),
                                get_gini(value[3]).rjust(5))
                                
        line += 1
    print()

    if input("Would you like to see a report on the data?(yes/no) ") == "yes":
        print_report(series_data)

# gets gini coefficient for each series
def get_gini(data_array):
    # might need to work on more efficient solution
    # Mean absolute difference
    mad = np.abs(np.subtract.outer(data_array, data_array)).mean()
    # Relative mean absolute difference
    rmad = mad/np.mean(data_array)
    # Gini coefficient
    g = 0.5 * rmad
    return "{:.3f}".format(g)

# gets skew values for each series
def get_skew(data_array):
    # Should work, but produces values slightly different from those in dplR
    df = pd.DataFrame(data_array)

    return "{:.3f}".format(df.skew(skipna=False).pop(0))

# generates a report on the data collected
def print_report(series_data):
    print("Number of dated series:", len(series_data))

    first = 9999
    last = 0
    measurements = 0
    total = 0
    for values in series_data.values():
        if values[0] < first:
            first = values[0]
        last_year = values[0] + values[1] - 1
        if last_year > last:
            last = last_year
        measurements += len(values[3])
        total += sum(values[3])
    
    print("Number of measurements:", measurements)
    print("Avg series length:", measurements/len(series_data))
    print("Range:", last - first + 1)
    print("Span:", first, "-", last)
    print("-------------")
    print("Years with absent rings listed by series")
    print_absent_ring_data(series_data, measurements)

# gets information about absent ring data, marked as 0 in files
def print_absent_ring_data(series_data, data_count):
    years_absent = 0
    for series, data in series_data.items():
        if 0 in data[3]:
            print("Series", series, "--", end=" ")
            for i, val in enumerate(data[3]):
                if val == 0:
                    print(data[0] + i, end=" ")
                    years_absent += 1
            print("")
    print(str(years_absent) + " absent rings (" +
            "{:.3f}".format((years_absent/data_count) * 100) + "%)")

# Remaining tasks with this file:
# mean (std dev) series intercorrelation
# mean (std dev) ar1