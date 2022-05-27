import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Modified hugershoff function
def hugershoff_function(x, a, b, c, d):
    return a*(x**b)*np.exp(c*x) + d

# Attempt to fit a hugershoff curve to the series
def hugershoff(series):
    x = series.index.to_numpy()
    
    y = series.to_numpy()
    xi = np.arange(1, len(y)+1)
    pars, unk= curve_fit(hugershoff_function, xi, y, bounds=([0, -2, -np.inf, min(y)], [np.inf, 2, 0, max(y)]), 
                            p0=[max(y)-min(y), 0, 0, y[0]])
    a, b, c, d = pars

    yi = hugershoff_function(xi, a, b, c, d)

    plt.plot(x, y, "o", x, yi, "-")
    plt.show()
    return yi

# Modified negative exponential function
def negex_function(x, a, b, k):
    return a * np.exp(b * x) + k

# Attempt to fit a negative exponential curve to the series
def negex(series):
    x = series.index.to_numpy()
    
    y = series.to_numpy()
    xi = np.arange(1, len(y)+1)
    pars, unk= curve_fit(negex_function, xi, y, bounds=([0, -np.inf, 0], [np.inf, 0, np.inf]))
    a, b, k = pars

    yi = negex_function(xi, a, b, k)

    plt.plot(x, y, "o", x, yi, "-")
    plt.show()
    return yi

# Fit a horizontal line to the series
def horizontal(series):
    x = series.index.to_numpy()
    y = series.to_numpy()

    yi = np.asarray([np.mean(y)] * len(x))
    plt.plot(x, y, "o", x, yi, "-")
    plt.show()
    return yi
    
# Equation of a straight line
def line_function(x, m, c):
    return (m * x) + c

# Fit a line to the series
def linear(series, bounds=False):
    x = series.index.to_numpy()
    y = series.to_numpy()

    if bounds is False:
        pars, unk = curve_fit(line_function, x, y)
    else:
        pars, unk = curve_fit(line_function, x, y, bounds=([-np.inf, -np.inf], [0, np.inf]))
    m, c = pars
    yi = line_function(x, m, c)
    plt.plot(x, y, "o", x, yi, "-")
    plt.show()
    return yi