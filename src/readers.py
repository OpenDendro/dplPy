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

def readers(input):
    """
    This function imports common ring width
    data files into Python as arrays
    Accepted file types are CSV, RWL, TXT
    """
    import csv
    import os
# open the input file and parse it to see what exactly it is
    # begin with CSV format
    if input.endswith(('.csv','.CSV')):
        with open(input, "r") as rings:
            print("")
            print(
                "Attempting to read input file: "
                + os.path.basename(input)
                + " as .csv format"
            )
            print("")
            try:
                data = rings.read()
                lines = data.split("\n")
                print(
                    "CSV header detail: \n"
                + "\n" + lines[0] + "\n"
                )
                # creates blank array
                csvs = []
                for  rows  in lines:
                    csvs.append(rows)
                #remove empty lines from the file.
                while "" in csvs:
                    csvs.remove("")
                csv_temp = csvs[1].split(",")
                startyear = csv_temp[0]
                print(
                    "Start Year: "
                    + startyear
                )
                print("")
                # create blank array
                allvals = []
                for csv_data in csvs:
                    csv_data_split = csv_data.split(",")
                    for i in range(1, len(csv_data_split)):
                        allvals.append(csv_data_split[i]
                        )
                print(
                    "Successful -- loaded file name: "
                    + os.path.basename(input)
                )
                print("")       
            except Exception as e:
                print(e)
    # End the CSV reader
    # Next, test if the file is in a Tucson (RWL) format
        # Note, RWL may have many header variations
    elif input.endswith(('.rwl','.RWL')):
        with open(input, "r") as rings:
            print("")
            print(
                "Attempting to read input file: "
                + os.path.basename(input)
                + " as .rwl format"
            )
            print("")
            try:
                data= rings.read()
                lines = data.split("\n")
                print(
                    "RWL header detail:"
                )
                print("")
                print(
                    lines[0] + "\n" 
                    + lines[1] + "\n"
                    + lines[2]
                )
                print("")
                #read every line 
                #for rows in lines:
                #    print(rows)
                # convert every element in each list to string- it is easier to manupilate the elements
                for items in lines:
                    data_str=str(items)  
                    # state/province code
                    state_province = data_str[1,3]
                    print(
                        "State/Province: "
                        + state_province
                        )
                    # country code
                    country= data_str[0,2]
                    print(
                        "Country: "
                        + data_str[0,2]
                        )
                    #species
                    species = data_str[1,4]
                    print(
                        "Species: "
                        + species
                        )
                    #species code
                    species_code= data_str[1,5]
                    print(
                        "Species Code: "
                        + species_code
                        )
                    #start year- start of collection
                    start_year = int(data_str[1,6])
                    print(
                        " Start Year: "
                        + start_year
                    )
                    #end year- completion year
                    end_year = int(data_str[1,7])
                    print(
                        "End Year: "
                        + end_year
                    )
                    #latitude
                    lat = data_str[2,4]
                    print(
                        "Latitude: "
                        + lat
                    )
                    # longitude
                    lon = data_str[2,5]
                    print(
                        "Longitude: "
                        + lon
                    )
                    #lead investigater
                    leadinv= data_str[2,2]
                    print(
                        "Lead Investigator: "
                        + leadinv
                    ) 
                    #site_id
                    #Via indexing the first three digits of the site id are assigned to var named site_code and the rest to information.
                    for site_code in data_str:
                        current=0
                        information=0 
                        site_id=0 
                        while current >=0 and current<=len(data_str)-1:
                            if current <=3:   
                                site_code = data_str[current,0,0]       
                            if current>=4 and current<=len(data_str)-1:
                                if data_str[current,0,0] == data_str[4,0,0]:
                                    site_id = data_str[current,0,0]
                                    information = site_id[3::]  
                            site_id= site_code + information 
                            #data
                            data=[]
                            y=1
                            past= y-1
                            end=[current,-1]
                            if site_id == data_str[current,0] and start_year <= int(data_str[current,1]) and end_year>= int(data_str[current,1]):
                                for points in data_str[current,2:]:
                                        pts=[]
                                        pts.append(points)
                                        # End of data collection for that year
                                        if points== "999" or points== "-9999":
                                            pts[:-1]
                                            data.append(pts)
                                            pts=0
                                current += 1
                                pts=0
                            #end
                        print(
                            "Site ID: "
                            + side_id
                        )
            except Exception as e:
                print(e)
    # end RWL reader
    # Begin import  for .txt types
    elif input.endswith(('.txt','.TXT')):
        with open(input, "r") as rings:
            print("")
            print(
                "Attempting to read input file: "
                + os.path.basename(input)
                + " as .txt format"
            )
            print("")
            try:
                data = rings.read()
                lines = data.split("\n")
                print(
                    "TXT header detail: \n"
                    + "\n" + lines[0] + "\n"
                )
            except Exception as e:
                print(e)
    else: 
        print("Unable to read file, please check that you're using a supported type")