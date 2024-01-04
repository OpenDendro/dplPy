import pytest
import dplpy as dpl
import pandas as pd
import os

def test_summary_methods():
    data = dpl.readers("./tests/data/csv/ca533.csv")

    dpl.summary(data)
    dpl.report(data)
    dpl.stats(data)