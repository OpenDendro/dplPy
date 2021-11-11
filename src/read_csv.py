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
__license__ = "GNU GPL3"

#!/usr/bin/python
# -*- coding: utf-8 -*-

# Date: 10/01/2021 
# Author: Sarah Jackson
# Project: OpenDendro- CSV Format 
# Description: Assigning variables to each value in a CSV format data set
# example usage: csv_var("./data/file.csv")

def read_csv(input):

with open("input", "r" ) as rings:
    data= rings.read()
    lines = data.split("\n")
    print (lines[0-1])
    #read every single line
    csvs = []
    for  rows  in lines:
        csvs.append(rows)
    
    #remove empty lines from the file.
    while "" in csvs:
        csvs.remove("")

    allvals = []
    csvtemp = csvs[0].split(",")
    startyear = csvtemp[0]

    for csv in csvs:
        temp = csv.split(",")
        for i in range(1, len(temp)):
            allvals.append(temp[i])

    #convert every element in each list to string- it is easier to manupilate the elements
