#!/usr/bin/python
# -*- coding: utf-8 -*-

# Date: 11/16/2021 
# Author: Tyson Lee Swetnam
# Project: OpenDendro- Read CSV Format 
# Description: Reads in a CSV format file into a numpy array
# example usage: 
# example usage: 
# >>> import dplpy as dpl 
# >>> dpl.read("./data/file.csv")

import csv
import sys
from os import listdir
from os.path import isfile, join

# define Read CSV module
def read_csv(input):
    with open(input) as rings:
        if rings.endswith(.'csv'):
            data = rings.read()
            lines = data.split("\n")
            print (lines[0-1])
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

            for csv in csvs:
                temp = csv.split(",")
                for i in range(1, len(temp)):
                    allvals.append(temp[i])

            #end
    # Echo error if file is corrupt  
    except Exception as e:
                print(e)