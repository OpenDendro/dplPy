from math import cos
from math import pi
import numpy as np
import matplotlib.pyplot as plt
from csaps import csaps


def smoothingspline(series, amp, series_len, period):
    freq = 1/period
    spline_param = 1/(((cos(2 * pi * freq) + 2) * (1 - amp)/(12 * amp * (cos(2 * pi * freq) - 1) ** 2))+ 1)

def univariate_data(n=25, seed=1234):
    np.random.seed(seed)
    x = np.linspace(-5., 5., n)
    y = np.exp(-(x/2.5)**2) + (np.random.rand(n) - 0.2) * 0.3
    return x, y