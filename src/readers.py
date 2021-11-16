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

# Date: 11/16/2021 
# Author: Tyson Lee Swetnam
# Project: OpenDendro- Readers
# Description: Readers for all supported file types (*.CSV, *.RWL, and *.TXT)
# 
# example usages: 
# >>> import dplpy as dpl 
# >>> dpl.readers("./data/file.csv")
# >>> dpl.readers("./data/file.rwl")
# >>> dpl.readers("./data/file.txt")

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
    if input.endswith(('.csv','.CSV')):
        with open(input, "r") as rings:
            print("")
            print(
                "Reading input as .CSV format"
            )
            print("")
            try:
                data = rings.read()
                lines = data.split("\n")
                print(
                    "CSV header detail:"
                    + lines[0]
                )
                #reads every single line
                csvs = []
                for  rows  in lines:
                    csvs.append(rows)
                #remove empty lines from the file.
                while "" in csvs:
                    csvs.remove("")
                allvals = []
                csvtemp = csvs[0].split(",")
                startyear = csvtemp[0]
                print(
                    "Start Year: "
                    + startyear
                )
                for csv in csvs:
                    temp = csv.split(",")
                    for i in range(1, len(temp)):
                        allvals.append(temp[i]
                        )
            except Exception as e:
                print(e)
    # End the CSV reader
    # Next, test if the file is a Tucson (RWL) format
        # Note, RWL may have many header variations
    elif input.endswith(('.rwl','.RWL')):
        with open(input, "r") as rings:
            print("")
            print(
                    "Reading as Tucson (RWL) format"
                )
            print("")
            try:
                data= rings.read()
                lines = data.split("\n")
                print(
                    "RWL header detail:"
                )
                print(
                    lines[0]  
                    + lines[1] 
                    + lines[2]
                )
                #read every line 
                #for rows in lines:
                #    print(rows)
                # convert every element in each list to string- it is easier to manupilate the elements
                for items in lines:
                    str(items)  
                    # state/province code
                    state_province = lines[1,3]
                    print(
                        "State/Province: "
                        + state_province
                        )
                    # country code
                    country= lines[0,2]
                    print(
                        "Country: "
                        + lines[0,2]
                        )
                    #species
                    species = lines[1,4]
                    print(
                        "Species: "
                        + species
                        )
                    #species code
                    species_code= lines[1,5]
                    print(
                        "Species Code: "
                        + species_code
                        )
                    #start year- start of collection
                    start_year = int(lines[1,6])
                    print(
                        " Start Year: "
                        + start_year
                    )
                    #end year- completion year
                    end_year = int(lines[1,7])
                    print(
                        "End Year: "
                        + end_year
                    )
                    #latitude
                    lat = lines[2,4]
                    print(
                        "Latitude: "
                        + lat
                    )
                    # longitude
                    lon = lines[2,5]
                    print(
                        "Longitude: "
                        + lon
                    )
                    #lead investigater
                    leadinv= lines[2,2]
                    print(
                        "Lead Investigator: "
                        + leadinv
                    ) 
                    #site_id
                    #Via indexing the first three digits of the site id are assigned to var named site_code and the rest to information.
                    for site_code in lines:
                        current=0
                        information=0 
                        site_id=0 
                        while current >=0 and current<=len(lines)-1:
                            if current <=3:   
                                site_code = lines[current,0,0]       
                            if current>=4 and current<=len(lines)-1:
                                if lines[current,0,0] == lines[4,0,0]:
                                    site_id = lines[current,0,0]
                                    information = site_id[3::]  
                            site_id= site_code + information 
                            #data
                            data=[]
                            y=1
                            past= y-1
                            end=[current,-1]
                            if site_id == lines[current,0] and start_year <= int(lines[current,1]) and end_year>= int(lines[current,1]):
                                for points in lines[current,2:]:
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
                "Reading as TXT format"
            )
            print("")
            try:
                data = rings.read()
                lines = data.split("\n")
                print(
                    "TXT header detail:"
                    + lines[0]
                )
            except Exception as e:
                print(e)
    else: 
        print("Unable to read file, please check that you're using a supported type")