from tkinter import Y
import pandas as pd
import matplotlib.pyplot as plt
from readers import readers
from smoothingspline import spline

# In the future, detrend will probably only take a series as input
def detrend(series_data):
    for series_name, data in series_data.items():
        nullremoved_data = data.dropna()
        yi = spline(nullremoved_data)
        
        residual(nullremoved_data, yi)
        difference(nullremoved_data, yi)
    
def residual(series, yi):
    x = series.index.to_numpy()
    y = series.to_numpy()
    res = yi/y

    plt.plot(x, res, "o", x, res, "-")
    plt.show()
    return res

def difference(series, yi):
    x = series.index.to_numpy()
    y = series.to_numpy()
    res = yi - y

    plt.plot(x, res, "o", x, res, "-")
    plt.show()
    return res