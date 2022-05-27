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
# Author: Ifeoluwa Ale
# Title: detrend.py
# Description: Contains methods that fit series to autoregressive
#              models and perform functions related to AR modeling
# example usage:
# >>> import dplpy as dpl 
# >>> data = dpl.readers("../tests/data/csv/file.csv")
# >>> dpl.autoreg(data['Name of series']) -> returns parameters of best fit AR model
#                                            with maxlag of 5 (default) or other 
#                                            specified number
# >>> dpl.ar_func(data['Name of series']) -> returns residuals plus mean of best fit 
#                                            from AR models with max lag of either 5 
#                                            (default) or specified number

from enum import auto
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.tsa.api import acf, graphics, pacf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# This function returns residuals plus mean of the best fit AR
# model of the data
def ar_func(data, max_lag=5):
    nullremoved_data = data.dropna()
    pars = autoreg(nullremoved_data, max_lag)
    
    if isinstance(data, pd.Series):
        y = nullremoved_data.to_numpy()
    else:
        y = nullremoved_data
    
    yi = fitted_values(y, pars)

    res = y[len(pars)-1:] - yi
    
    mean = np.mean(y)

    # Add mean to the residuals
    for i in range(len(res)):
        res[i] += mean

    return res

# This method selects the best AR model with a specified maximum order
# The best model is selected based on AIC value
def autoreg(data, max_lag=5):
    AR_data = ar_select_order(data.dropna(), max_lag, ic='aic', old_names=False)
    results = AR_data.model.fit()
    return results.params

# This function calculates the in-sample predicted values of a series,
# given an array containing the original data and the parameters for
# the AR model
def fitted_values(data_array, params):
    mean = np.mean(data_array)
    results = []
    
    for i in range((len(params)-1), len(data_array)):
        pred = params[0]
        for j in range(1, len(params)):
            pred += (params[j] * data_array[i-j])
        results.append(pred)
    return np.asarray(results)