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

# Date: 11/17/2021 
# Author: Tyson Lee Swetnam
# Project: OpenDendro- Writers
# Description: Writers for all supported file types (*.CSV, *.RWL, and *.TXT)
# 
# example usages: 
# >>> import dplpy as dpl 
# >>> dpl.writers("./data/in_file.csv", "./data/out_file.rwl")
# >>> dpl.writers("./data/in_file.rwl", "./data/out_file.txt")
# >>> dpl.writers("./data/in_file.txt", "./data/out_file.csv")
# 
# example command line application:
# $ python src/dplpy.py writer --input ./data/in_file.csv --output ./data/out_file.rwl
#
# module uses two arguments: input file and output file 
# define `writer` module as a definition function
# input is expected to be a file path with file name and extension

import pandas as pd
import numpy as np
import os

def write(data, label, format):
    print("Entered function")
    """
    This function converts common ring width
    data files from one type to another
    It also allows you to append files that are missing metadata and write them back out
    Accepted file types are CSV, RWL, TXT
    """
    filename = label + "." + format
    print(filename)
    output = open(filename, "w")
    if format == "csv":
        write_csv(data, output)
    elif format == "rwl":
        write_rwl(data, output)

    output.close()

def conv_data(data):
    if np.isnan(data):
        return "NA"
    else:
        return str(data)

def write_csv(data, file):
    file.write('"Year","')
    file.write('","'.join(data.columns.tolist()))
    file.write('"\n')

    for year, row in data.iterrows():
        file.write(str(year))
        file.write(",")
        file.write(",".join(map(conv_data, row)))
        file.write('\n')

# Incomplete. Doesn't account yet for varying precision standards for RWL files (lines ending in 999 vs -9999)
def write_rwl(data, file):
    for series in data.columns:
        start = data[series].first_valid_index()
        end = data[series].last_valid_index()
        i = start
        done = False
        while i <= end:
            file.write(series + "\t")
            file.write(str(i) + "\t")
            line_end = i + 10
            while i < line_end:
                # write every series[i]
                try:
                    file.write(str(data[series][i]) + "\t")
                    i += 1
                except KeyError:
                    file.write(str(999))
                    file.write("\n")
                    done = True
                    break
            if done:
                break
            else:
                file.write("\n")
        
    