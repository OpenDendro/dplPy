__copyright__ = """
   dplPy for tree ring width time series analyses
   Copyright (C) 2022  OpenDendro

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

__license__ = "GNU GPLv3"

#!/usr/bin/python
# -*- coding: utf-8 -*-

# Date: 5/27/2022
# Author: Grigoriy Lozhkin
# Title: emdecomp.py
# Description: This contains the spline method which fits
#              a series to a spline curve.
# example usage (in other file):
# from emdecomp import emd
# yi = emd(series)


import numpy as np
from scipy.fft import fft
from PyEMD import EMD
from sktime.datasets import load_airline, load_shampoo_sales
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt

def emd(signal):
   emd = EMD()
   imfs = emd(signal.values)
   return imfs


def phase_spectrum(imfs):
   imfs_p = []
   for i, imf in enumerate(imfs):
        trans = fft(imf)
        imf_p = np.arctan(trans.imag / trans.real)   
        imfs_p.append(imf_p)
        
   return imfs_p

def phase_mi(phases):
   mis = []
   for i in range(len(phases)-1):
        mis.append(mutual_info_regression(phases[i].reshape(-1, 1), phases[i+1])[0])
        
   return np.array(mis)

def divide_signal(signal, imfs, mis, cutoff=0.05):

   x = signal.index.to_numpy()
   y = signal.to_numpy()

   cut_point = np.where(mis > cutoff)[0][0]
   stochastic_component = np.sum(imfs[:cut_point], axis=0)
   deterministic_component = np.sum(imfs[cut_point:], axis=0)
    
   t = [i for i in range(len(signal))]
    
   fig, axs = plt.subplots(3, 1, figsize=(15,8))
   axs[0].plot(t, signal.values)
   axs[0].set_title('Original Signal')
    
   axs[1].plot(t, stochastic_component)
   axs[1].set_title('Stochastic Component')
    
   axs[2].plot(t, deterministic_component)
   axs[2].set_title('Deterministic Component')
    
   plt.show()

   plt.plot(x, y, "o", x, deterministic_component, "-")
   plt.show()
   return stochastic_component, deterministic_component








