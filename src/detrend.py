from tkinter import Y
import pandas as pd
import matplotlib.pyplot as plt
from readers import readers
from smoothingspline import get_spline

# In the future, detrend will probably only take a series as input
def detrend(series_data):
    detrend_type = input("residual or difference?")

    for series_name, data in series_data.items():
        nullremoved_data = data.dropna()
        yi = get_spline(nullremoved_data)
    
def residual(series, yi):
    x = series.index.to_numpy()
    y = series.to_numpy()
    res = yi/y

    plt.plot(x, y, "o", x, yi, "-")
    return yi

def difference(series, yi):
    x = series.index.to_numpy()
    y = series.to_numpy()
    res = yi - y

    plt.plot(x, y, "o", x, yi, "-")
    return yi