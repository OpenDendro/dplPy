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

# Date:2021-08-13
# Author: Sarah Jackson
# Project: OpenDendro- Tucson Format 
# Description: Assigns variable names to each value in the tucson formated datasets using the emptyjson file.
# example usage:

import json

def create_json(input):

    with open("input", "r" ) as rings:
        data= rings.read()
        lines = data.split("\n")
        print (lines[0-2])
        # read every single line
        for  rows  in lines:
            print (rows)
        
        # remove empty lines from the file.

        # convert every element in each list to string- it is easier to manupilate the elements

        for items in lines:
            str(items)  

        # State/province
        state_province = lines[1,3]

        # country
        country= lines[0,2]

        # species
        species = lines[1,4]

        # species code
        species_code= lines[1,5]

        # start year- start of collection 
        start_date = int(lines[1,6])

        # end year- completion year
        end_year = int(lines[1,7])

    # latitude, longitude, elevation 
        latitude = lines[2,4]
        longitude = lines[2,5]

        # lead investigator
        lead_investigator= lines[2,2] 

        # site_id
            # Via indexing the first three digits of the site id are assigned to var named site_code and the rest to information.
            # Check if all the site codes are the same for the whole dataset 
        for site_code in lines:
            current=0
            information=0 
            site_id=0 
            unit = 0
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
                if site_id == lines[current,0] and start_date <= int(lines[current]):
                    for points in lines[current,2:]:
                        pts=[]
                        pts.append(points)
                        # End of data collection for that year
                        if points== "999" or points== "-9999":
                            pts[:-1]
                            data.append(pts)
                            pts=0
                            unit = points
                else:
                    current = current + 1
                    pts=0
    
    jsonfile = {}
    jsonfile["site_code"] = site_code
    jsonfile["site_name"] = site_id
    jsonfile["species_code"] = species_code
    jsonfile["species_name"] = species
    jsonfile["country"] = country
    jsonfile["state_province"] = state_province
    jsonfile["elevation"] = ""
    jsonfile["latitude"] = latitude
    jsonfile["longitude"] = longitude
    if unit == "999":
        jsonfile["unit"] = "0.01mm"
    elif unit == "-9999":
        jsonfile["unit"] = "0.001mm"
    else:
        print("ERROR - Stop code is not 999 or -9999. Cannot identify a unit.")
        exit
    jsonfile["lead_investigator"] = lead_investigator
    jsonfile["collection_date"] = ""
