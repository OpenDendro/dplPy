from enum import auto
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.tsa.api import acf, graphics, pacf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# This function returns residuals plus mean of the best fit AR
# model of the data
def ar_func(data):
    pars = autoreg(data, 5)
    
    if isinstance(data, pd.Series):
        y = data.to_numpy()
    else:
        y = data
    
    yi = fitted_values(y, pars)

    res = y[len(pars)-1:] - yi
    
    mean = np.mean(y)

    # Add mean to the residuals
    for i in range(len(res)):
        res[i] += mean

    return res

# This method selects the best AR model with a specified maximum order
# The best model is selected based on AIC value
def autoreg(data, max_lag):
    AR_data = ar_select_order(data, max_lag, ic='aic', old_names=False)
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