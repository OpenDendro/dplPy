from math import cos
from math import pi
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from csaps import csaps
from mpl_toolkits.mplot3d import Axes3D

def smoothingspline(series, amp, series_len, period):
    freq = 1/period
    spline_param = 1/(((cos(2 * pi * freq) + 2) * (1 - amp)/(12 * amp * (cos(2 * pi * freq) - 1) ** 2))+ 1)


def univariate_data(series):
    x = series.index.to_numpy()
    y = series.to_numpy()

    xi = np.linspace(x[0], x[-1], 150)

    yi = csaps(x, y, xi, smooth=0.85)
    plt.plot(x, y, "o", xi, yi, "-")