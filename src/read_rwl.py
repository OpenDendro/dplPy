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

# Date: 10/01/2021
# Author: Anushka Bande  
# Project: OpenDendro- Tucson Format 
# Description: Assigning variables to each value in the Tucson format data sets
# example usage:

## This section needs to be fixed - the scripts must ask the users for an input file or directory and it should look for the .rwl formatted files within a folder.
## It must also echo out that the file is corrupt or there is no corresponding .rwl file.
################################

import sys
from os import listdir
from os.path import isfile, join

if len(sys.argv) == 2:
    rwldir = sys.argv[1]
else:
    print('')
    print('Missing Inputs:')
    print('Please include the path to the directory you want to load multiple files from,')
    print('Or include the path to the directory and file you want to load.')
    print('')
    print('Correct usage: python tucson_var.py ~/folder/dataset1.rwl')
    print('Correct usage: python tucson_var.py ~/folder/dataset1.rwl')
    print('Correct usage: python tucson_var.py ~/folder/dataset1.rwl dataset2.rwl')    
    exit(2)

def read_rwl(input):
    with open(input) as filename:

for filename in listdir(rwldir):
    if filename.endswith('.rwl'):
        try:
            rwl = open(rwldir+'/'+ filename) # open the rwl file
        except (IOError, SyntaxError) as e:
            print('Bad file:', filename) # print out the names of corrupt files
#################################

    with open(rwldir+'/'+ filename, 'r') as rings:
        data= rings.read()
        lines = data.split("\n")
        print (lines[0-2])
    #read every single line
    for  rows  in lines:
        print (rows)
    
    # remove empty lines from the file.

    # convert every element in each list to string- it is easier to manupilate the elements

    for items in lines:
        str(items)  
    
    # state/province code
    state_province = lines[1,3]
    # country code
    country= lines[0,2]

    # Use dictionary to check if the the place exist or not.  
    # If it does exist then it the program will continue running
    # If not then will tell user to change the location or stop the program completely 
    
    #species
    species = lines[1,4]

    #species code
    species_code= lines[1,5]

    #start year- start of collection
    start_year = int(lines[1,6])

    #end year- completion year
    end_year = int(lines[1,7])

    #latitude, longitude 
    latitude = lines[2,4]
    longitude = lines[2,5]

    #lead investigater
    lead_investigator= lines[2,2] 

    #site_id
        #Via indexing the first three digits of the site id are assigned to var named site_code and the rest to information.
        #Check if all the site codes are the same for the whole dataset 
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
   
