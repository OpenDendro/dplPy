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

def writers(input,output):
    """
    This function converts common ring width
    data files from one type to another
    It also allows you to append files that are missing metadata and write them back out
    Accepted file types are CSV, RWL, TXT
    """
    import csv
    import os