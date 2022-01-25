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

# Date: 10/22/2021
# Author: Anushka Bande
# Title: summary.py
# Description: Generates Summary statistics for Tucson format and CSV format files
# usage: python dplpy summary --input <*.rwl> 

# Create Summaries for Tucson (*rwl) files
def summary(input):
    import pandas as pd
    import numpy as np
    
    df = pd.read_rwl(input)
    
    #To see the dataframe 
    print(df)

    #First five rows 
    df.head()

    #Average / year
    df['Average'] = df.mean(axis=1)

    """
    Total amount of observation 
    Average 
    Standard Deviation 
    Minimum 
    Lower and Upper Quartile
    Interquartile
    Maximum 
    Data type 

    """
    df.describe()

# Create Summaries for CSV files
def summary_csv(input):    
    df = pd.read_csv(r'*.csv')

    #To see the dataframe 
    print(df)

    #First five rows 
    df.head()

    #Average / year
    df['Average'] = df.mean(axis=1)

    """
    Total amount of observation 
    Average 
    Standard Deviation 
    Minimum 
    Lower and Upper Quartile
    Interquartile
    Maximum 
    Data type 

    """
    df.describe()












