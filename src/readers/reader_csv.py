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


import pandas as pd

data_df = pd.DataFrame('/workspaces/dplPy/tests/data/csv/ca533.csv')

print(data_df.columns)