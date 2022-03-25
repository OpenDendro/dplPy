from math import cos
from math import pi
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from csaps import csaps
from mpl_toolkits.mplot3d import Axes3D

def get_param(amp, period):
    freq = 1/period
    spline_param = 1/(((cos(2 * pi * freq) + 2) * (1 - amp)/(12 * amp * (cos(2 * pi * freq) - 1) ** 2))+ 1)
    return spline_param

# Change function names and definitions
def univariate_data(series):
    series = series.dropna()
    x = series.index.to_numpy()
    
    y = series.to_numpy()

    p = get_param(0.5, 100)

    yi = csaps(x, y, x, smooth=p)

    plt.plot(x, y, "o", x, yi, "-")
    plt.savefig("spline.png")