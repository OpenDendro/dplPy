from __future__ import print_function

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

# Date: 9/8/2022
# Author: Ifeoluwa Ale
# Project: OpenDendro- Readers
# Description: Reads data from supported file types (*.CSV and *.RWL)
#              and stores them in a dataframe
#
# example usages: 
# >>> import dplpy as dpl 
# >>> data = dpl.readers("../tests/data/csv/file.csv")
# >>> data = dpl.readers("../tests/data/csv/file.rwl", header=True)
# 
# example command line application:
# $ python src/dplpy.py reader --input ./data/file.csv
#
# define `readers` module as a definition function
# input is expected to be a file path with file name and extension
import os
import sys
import pandas as pd
import numpy as np

def readers(filename: str, skip_lines=0, header=False):
    """
    This function imports common ring width data files into Python as arrays
    Accepted file types are CSV and RWL
    """
    FORMAT = "." + filename.split(".")[-1]
    print("\nAttempting to read input file: " + os.path.basename(filename) + " as " + FORMAT + " format\n")
    
    # open the input file and read its data into a pandas dataframe
    if filename.upper().endswith(".CSV"):
        series_data = pd.read_csv(filename, skiprows=skip_lines)
    elif filename.upper().endswith(".RWL"):
        series_data = process_rwl_pandas(filename, skip_lines, header)
    else:
        errorMsg = """

Unable to read file, please check that you're using a supported type
Accepted file types are .csv and .rwl

Example usages:
>>> import dplpy as dpl
>>> data = dpl.readers('../tests/data/csv/filename.csv')
>>> data = dpl.readers('../tests/data/rwl/filename.rwl'), header=True
"""
        
        raise ValueError(errorMsg)

    # If no data is returned, then an error was encountered when reading the file.
    if series_data is None:
        errorMsg = """
        Error reading file. Check that file exists and that file formatting is consistent with {format} format.
        If your file contains headers, run dpl.headers(file_path, header=True)
        """.format(format=FORMAT)
        raise ValueError(errorMsg)
    series_data.set_index('Year', inplace = True, drop = True)

    # Display message to show that reading was successful
    print("\nSUCCESS!\nFile read as:", FORMAT, "file\n")

    # Display names of all the series found
    print("Series names:")
    print(list(series_data.columns), "\n")
    return series_data

# Process data from .rwl file and store data in a pandas dataframe.
def process_rwl_pandas(filename, skip_lines, header):
    if header is True:
        skip_lines += 3 # working with the assumption that headers are 3 lines long

    with open(filename, "r") as rwl_file:
        file_lines = rwl_file.readlines()[skip_lines:]
    
    rwl_data, first_date, last_date = read_rwl(file_lines)
    if rwl_data is None:
        return None

    # create an array of indexes for the dataframe
    indexes = []
    for i in range(first_date, last_date):
        indexes.append(i)
    
    df = pd.DataFrame(data={"Year":indexes})

    # store raw data in pandas dataframe
    for series in rwl_data:
        series_data = []
        for i in range(first_date, last_date):
            if i in rwl_data[series]:
                series_data.append(rwl_data[series][i]/rwl_data[series]["div"])
            else:
                series_data.append(np.nan)
        df = pd.concat([df, pd.Series(data=series_data, name=series)], axis=1)
    return df

# Extract raw data from lines of .rwl file and store in a nested dictionary
def read_rwl(lines):
    rwl_data = {}
    first_date = sys.maxsize
    last_date = -sys.maxsize

    for line in lines:
        line = line.rstrip("\n").split()

        series_id = line[0]
        if series_id not in rwl_data:
            rwl_data[series_id] = {}
            
        # keep track of the first and last date in the dataset
        line_start = int(line[1])
        first_date = min(first_date, line_start)
        last_date = max(last_date, (line_start+len(line)-3))

        # will implement some standardization here so that all data read is consistent, and all data written in rwl
        # can be written to one of the two popular precisions.
        for i in range(2, len(line)):
            try:
                if line[i] == "999":
                    rwl_data[series_id]["div"] = 100
                    continue
                elif line[i] == "-9999":
                    rwl_data[series_id]["div"] = 1000
                    continue
                data = float(int(line[i]))
            except ValueError as valerr: # Stops reader, escalates to give the user an error when unexpected formatting is detected.
                return None, None, None
            rwl_data[series_id][line_start+i-2] = data
    return rwl_data, first_date, last_date
