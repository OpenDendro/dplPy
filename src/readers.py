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
# Project: OpenDendro- Readers
# Description: Readers for all supported file types (*.CSV, *.RWL, and *.TXT)
# 
# example usages: 
# >>> import dplpy as dpl 
# >>> dpl.readers("./data/file.csv")
# >>> dpl.readers("./data/file.rwl")
# >>> dpl.readers("./data/file.txt")
# 
# example command line application:
# $ python src/dplpy.py reader --input ./data/file.csv
#
# define `reader` module as a definition function
# input is expected to be a file path with file name and extension
import os

def readers(filename):
    """
    This function imports common ring width
    data files into Python as arrays
    Accepted file types are CSV, RWL, TXT
    """

# open the input file and read its data into a dictionary
    # begin with CSV format
    if filename.upper().endswith((".CSV")):
        series_data = process_csv_file(filename)
    elif filename.upper().endswith((".RWL")):
        series_data = process_rwl_file(filename)
        
    else:
        print("Unable to read file, please check that you're using a supported type")
    # end of readers

    for key, value in series_data.items():
        print(str(key) + ":- " + str(value[:3]))

# read the files written in CSV format
def process_csv_file(filename):
    import csv
    print("\nAttempting to read input file: " + os.path.basename(filename)
            + " as .csv format\n")
    with open(filename, "r") as file:
        try:
            data = csv.reader(file)
            header = next(data)     # read the header of the file

            print("CSV header detail: \n")
            print(" ".join(header[1:]) + "\n")
            
            identifiers = header

            # assemble dictionary for data storage
            # format in dictionary is: 
            # key (series name):- first year of collection, 
            #                     no. of years data was collected for,
            #                     sum of collected data, 
            #                     individual data (measurements)
            series_data = get_series_data(identifiers[1:])

            line_no = 0

            for row in data:
                if line_no == 0:
                    year_1 = int(row[0])
                    
                for i, data in enumerate(row):
                    try:
                        num = float(data)
                        if i != 0:
                            record_data(series_data, identifiers, year_1, 
                                    line_no, i, num)
                    except ValueError:
                        num = 0
                line_no += 1
                
            return series_data
        except:
            print("Error in data format")
            return
    # End the CSV reader

def process_rwl_file(filename):
    print("\nAttempting to read input file: " + os.path.basename(filename)
            + " as .rwl format\n")
    with open(filename, "r") as rwl_file:
        lines = rwl_file.readlines()
        series_data = {}

        for line in lines:
            line = line.rstrip("\n").split()
            id = line[0]

            if id not in series_data:
                series_data[id] = [int(line[1]), 0, 0, []]
                
            for i in range(2, len(line)):
                series_data[id][1] += 1

                try:
                    data = float(line[i])/100
                    series_data[id][2] += data
                    series_data[id][3].append(data)
                except ValueError:
                    data = 0

    print("RWL header detail: \n")
    print(" ".join(list(sorted(series_data.keys()))) + "\n")
    return series_data

# This function arranges the series data into a dictionary where it is 
# stored in a format that makes processing easy
def get_series_data(id_array):
    dictionary = {}
    
    for id in id_array:
        dictionary[id] = ["first", 0, 0, []]
    
    return dictionary

# This function stores data from the time series in a dictionary
# as the program reads from the input file
def record_data(series_data, identifiers, year_1, line, col, data):

    if series_data[identifiers[col]][0] == "first":
        series_data[identifiers[col]][0] = year_1 + line

    series_data[identifiers[col]][1] += 1

    series_data[identifiers[col]][2] += float(data)

    series_data[identifiers[col]][3].append(data)