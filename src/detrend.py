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
        print(nullremoved_data)
        print(yi)

        detrended = residual(nullremoved_data, yi)

        print(detrended)
        break
    
def residual(series, yi):
    x = series.index.to_numpy()
    y = series.to_numpy()
    res = yi/y

    plt.plot(x, y, "o", x, yi, "-")
    plt.show()
    return yi

def difference(series, yi):
    x = series.index.to_numpy()
    y = series.to_numpy()
    res = yi - y

    plt.plot(x, y, "o", x, yi, "-")
    plt.show()
    return yi