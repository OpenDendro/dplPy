import numpy as np
from scipy.optimize import curve_fit

def gaussian(x, a, b, c):
    return a*np.exp(-np.power(x - b, 2)/(2*np.power(c, 2)))

def main():
    # Generate dummy dataset
    x_dummy = np.linspace(start=-10, stop=10, num=100)
    y_dummy = gaussian(x_dummy, 8, -1, 3)
    # Add noise from a Gaussian distribution
    noise = 0.5*np.random.normal(size=y_dummy.size)
    y_dummy = y_dummy + noise

    pars, cov = curve_fit(f=gaussian, xdata=x_dummy, ydata=y_dummy, p0=[5, -1, 1], bounds=(-np.inf, np.inf))
    print(pars)
    print(cov)
    
