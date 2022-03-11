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

# Date: 3/9/2021 
# Author: Ifeoluwa Ale
# Project: OpenDendro- Readers
# Description: Readers for supported file types (*.CSV and *.RWL)
# 
# example usages: 
# >>> import dplpy as dpl 
# >>> data = dpl.readers("../tests/data/csv/file.csv")
# >>> data = dpl.readers("../tests/data/csv/file.rwl")
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

def readers(filename):
    """
    This function imports common ring width
    data files into Python as arrays
    Accepted file types are CSV, RWL, TXT
    """

# open the input file and read its data into a pandas dataframe
    # tries to read file as .csv
    if filename.upper().endswith(".CSV"):
        try:
            series_data = pd.read_csv(filename)
        except:
            print("\nError reading file. Check that file exists and that data is consistent")
            print("with .CSV format")
            return
    # tries to read file as .rwl
    elif filename.upper().endswith(".RWL"):
        try:
            series_data = process_rwl_pandas(filename)
        except:
            print("\nError reading file. Check that file exists and that data is consistent")
            print("with .RWL format")
            return
    else:
        print("\nUnable to read file, please check that you're using a supported type\n")
        print("Accepted file types: .csv and .rwl")
        print("example usages:\n>>> import dplpy as dpl")
        print(">>> data = dpl.readers('../tests/data/csv/filename.csv')")
        return

    print("\nSUCCESS!\nFile read as:", filename[-4:], "file\n")
    series_data.set_index('Year', inplace = True, drop = True)
    return series_data

# function for reading files in rwl format
def process_rwl_pandas(filename):
    print("\nAttempting to read input file: " + os.path.basename(filename)
            + " as .rwl format\n")
    with open(filename, "r") as rwl_file:
        lines = rwl_file.readlines()
        rwl_data = {}
        first_date = sys.maxsize
        last_date = 0

        # read through lines of file and store raw data in a dictionary
        for line in lines:
            line = line.rstrip("\n").split()
            id = line[0]
            
            date = int(line[1])

            if id not in rwl_data:
                rwl_data[id] = [date, []]

            # keep track of the first and last date in the series
            if date < first_date:
                first_date = date
            if (date + len(line) - 3) > last_date:
                last_date = date + len(line) - 3
            
            for i in range(2, len(line)):
                try:
                    data = float(line[i])/100
                except ValueError:
                    data = np.nan
                rwl_data[id][1].append(data)

    # create an array of indexes for the dataframe
    indexes = []
    for i in range(first_date, last_date+1):
        indexes.append(i)

    # create a new dictionary to store the data in a way more suited for the
    # dataframe
    refined_data = {}
    for key, val in rwl_data.items():
        front_addition = [np.nan] * (val[0]-first_date)
        end_addition = [np.nan] * (last_date - (val[0] + len(val[1]) - 1))
        refined_data[key] = front_addition + val[1] + end_addition

    df = pd.DataFrame.from_dict(refined_data)

    df.insert(0, "Year", indexes)
    df.set_index("Year")
    return df