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

# Date: 11/1/2022
# Author: Ifeoluwa Ale
# Title: autoreg.py
# Description: Contains methods that fit series to autoregressive models and perform functions 
#              related to AR modeling.
#              NOTE: This function only accepts pandas series and dataframes as parameters.
# example usage:
# >>> import dplpy as dpl 
# >>> data = dpl.readers("../tests/data/csv/file.csv")
# >>> dpl.autoreg(data['Name of series']) -> returns parameters of best fit AR model
#                                            with maxlag of 5 (default) or other 
#                                            specified number
# >>> dpl.ar_func(data['Name of series']) -> returns residuals plus mean of best fit 
#                                            from AR models with max lag of either 5 
#                                            (default) or specified number

from statsmodels.tsa.ar_model import ar_select_order
import pandas as pd
import numpy as np
import warnings

def ar_func(data, max_lag=5):
    warnings.filterwarnings("ignore")
    if isinstance(data, pd.DataFrame):
        res = {}
        for column in data.columns:
            res[column] = ar_func_series(data[column], max_lag).tolist()
        return res
    elif isinstance(data, pd.Series):
        res = ar_func_series(data, max_lag)
        return res
    else:
        return TypeError("argument should be either pandas dataframe or pandas series.")

# This function returns residuals plus mean of the best fit AR
# model of the data
def ar_func_series(data, max_lag):
    nullremoved_data = data.dropna()
    pars = autoreg(nullremoved_data, max_lag)
    
    y = nullremoved_data.to_numpy()
    
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
    ar_data = ar_select_order(data.dropna(), max_lag, ic='aic', old_names=False)
    results = ar_data.model.fit()
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