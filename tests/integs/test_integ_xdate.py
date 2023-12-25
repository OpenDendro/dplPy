import pytest
import dplpy as dpl
import pandas as pd
import os

def test_xdate_diff_bins():
    ca533 = dpl.readers("./tests/data/csv/ca533.csv")
    
    ca533_bindata_1 = dpl.xdate(ca533, bin_floor=0)
    ca533_bindata_2 = dpl.xdate(ca533, bin_floor=10)
    ca533_bindata_3 = dpl.xdate(ca533, bin_floor=100)

def test_xdate_diff_slide_periods():
    ca533 = dpl.readers("./tests/data/csv/ca533.csv")
    
    ca533_bindata_1 = dpl.xdate(ca533, slide_period=30)
    ca533_bindata_2 = dpl.xdate(ca533, slide_period=50)
    ca533_bindata_3 = dpl.xdate(ca533, slide_period=80)

def test_xdate_diff_corrs():
    ca667 = dpl.readers("./tests/data/rwl/ca667.rwl", header=True)

    ca667_bindata_1 = dpl.xdate(ca667, corr="Spearman")
    ca667_bindata_2 = dpl.xdate(ca667, corr="Pearson")

def test_xdate_not_prewhitened():
    ca667 = dpl.readers("./tests/data/rwl/ca667.rwl", header=True)

    ca667_bindata = dpl.xdate(ca667, prewhiten=False)

# Commented out because plots block execution in vscode. WIP
# def test_xdate_plot():
#     co021 = dpl.readers("./integs/data/rwl/co021.rwl")

#     dpl.xdate_plot(co021)