### Date: 10/22/2021
### Author: Anushka Bande
### Title: summary.tucson.py
### Description: 

import pandas as pd
import numpy as np

df = pd.read_csv(r'*')

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












