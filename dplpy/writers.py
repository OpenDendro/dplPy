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
from chron import chron
from detrend import detrend

"""
    This function converts common ring width
    data files from one type to another
    It also allows you to append files that are missing metadata and write them back out
    Accepted file types are CSV, RWL, CRN (in dev) and TXT (in dev)
"""
def writers(data, label, format):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Expected input data to be pandas dataframe, not " + str(type(data)))
    
    if not isinstance(label, str):
        raise TypeError("Expected label to be of type str, not " + str(type(label)))
    
    if not isinstance(format, str):
        raise TypeError("Expected format to be of type str, not " + str(type(format)))
    
    filename = label + "." + format
    print("Writing to " + filename)
    output = open(filename, "w")
    if format == "csv":
        write_csv(data, output)
    elif format == "rwl":
        write_rwl(data, output)
    elif format == "crn":
        write_crn(data, label, output)
    elif format == "txt":
        write_txt(data, output)
    else:
        output.close()
        raise ValueError("Invalid file format given as parameter. Accepted file formats are csv, rwl, crn and txt")

    output.close()
    print("Done.")


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

        if i < 0:
            file.write(series.ljust(7))
            file.write(str(i).rjust(5))
        else:
            file.write(series.ljust(8))
            file.write(str(i).rjust(4))
        while i <= end:
            if np.isnan(data[series][i]):
                file.write(str(-9999).rjust(6))
                file.write("\n")
                while i <= end and np.isnan(data[series][i]):
                    i += 1
                if i <= end:
                    if i < 0:
                        file.write(series.ljust(7))
                        file.write(str(i).rjust(5))
                    else:
                        file.write(series.ljust(8))
                        file.write(str(i).rjust(4))
                continue

            file.write((f"{data[series][i]:.3f}").lstrip('0').replace('.', '').rjust(4, '0').rjust(6))
            i += 1
            if i % 10 == 0:
                file.write("\n")
                if i < 0:
                    file.write(series.ljust(7))
                    file.write(str(i).rjust(5))
                else:
                    file.write(series.ljust(8))
                    file.write(str(i).rjust(4))

        file.write(str(-9999).rjust(6))
        file.write("\n")


def write_crn(data, site_id, file, site_name="Unnamed object", species_code="UNKN", location="Unknown", species="Plantae", elevation="", lat_long="", investigator="", comp_date=""):
    rwi_data = chron(detrend(data, fit="spline", method="residual", plot=False), prewhiten=True)
    first = rwi_data.first_valid_index()
    last = rwi_data.last_valid_index()

    # For header, to be improved upon eventually
    file.write(site_id.ljust(9) + site_name.ljust(61) + species_code.ljust(4) + "\n")
    file.write(site_id.ljust(9) + location.ljust(13) + species.ljust(8) + elevation.ljust(5) + lat_long.ljust(10) + str(first).ljust(4) + str(last).rjust(5) + "\n")
    file.write(site_id.ljust(9) + investigator.ljust(73) + comp_date.ljust(8) + "\n")
    
    file.write(site_id.rjust(6))
    file.write(str(rwi_data.first_valid_index()).rjust(4))

    for year in rwi_data.index.to_numpy():
        file.write(str(round(rwi_data["Mean RWI"][year], 2)).replace(".", "").replace("0", "").rjust(4))
        file.write(str(rwi_data["Sample depth"][year]).rjust(3))
        
        if year % 10 == 9:
            # write TRL ID#(optional) which takes columns 82-88 
            if year + 1 in rwi_data.index.to_numpy():
                file.write("\n")
                file.write(site_id.rjust(6))
                file.write(str(year + 1).rjust(4))
            else:
                break
    file.write("9990  0")


def write_txt(data, file):
    header = ["year", "num".rjust(7), "seg".rjust(7), "age".rjust(7), "raw".rjust(7), "std".rjust(7), "res".rjust(7), "ars".rjust(7)]
    file.write("    ".join(header))
    file.write("\n")
    rwi_data = detrend(data, fit="spline", method="residual", plot=False)
    rwi_chron = chron(rwi_data, prewhiten=False)
    mean_res = chron(rwi_data, biweight=False, prewhiten=False)
    ar_chron = chron(rwi_data, prewhiten=True)

    first = rwi_chron.first_valid_index()
    last = rwi_chron.last_valid_index()

    for year in range(first, last+1):
        samp_dep = rwi_chron["Sample depth"][year]
        
        # standard chronology of detrended data
        std = rwi_chron["Mean RWI"][year]

        # residuals of detrended data?
        res =  mean_res["Mean RWI"][year]

        # residuals of ar modeled data?
        ars = ar_chron["Mean Res"][year]
        
        year_data = data.loc[[year]].dropna(axis=1)
        column_names = year_data.columns
        
        seg = 0
        age = 0
        raw = 0
        
        for name in column_names:
            seg += len(data[name].dropna())
            age += year - data[name].first_valid_index() + 1
            raw += data[name][year]
        
        seg = seg/len(column_names)
        age = age/len(column_names)
        raw = raw/len(column_names)

        

        # double check what res and ars are supposed to be
        # work on other columns
        line = [str(year).rjust(4), (f"{samp_dep:.3f}").rjust(7), (f"{seg:.3f}").rjust(7), (f"{age:.3f}").rjust(7), (f"{raw:.3f}").rjust(7), (f"{std:.3f}").rjust(7), (f"{res:.3f}").rjust(7), (f"{ars:.3f}").rjust(7),]
        file.write("    ".join(line))
        file.write("\n")
