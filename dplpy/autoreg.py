__copyright__ = """
   dplPy for tree ring width time series analyses
   Copyright (C) 2024  OpenDendro

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

from statsmodels.tsa.ar_model import ar_select_order
import pandas as pd
import numpy as np
import warnings

def ar_func(data, max_lag=5):
    """Auto Regressive (AR) functions 
      
    Extended Summary
    ---------------
    Contains methods that fit series to autoregressive models and perform functions 
    related to AR modeling.
  
    NOTE: This function only accepts pandas series and dataframes as parameters. 
   
    Parameters
    ----------
    data :  str
            a data file (.CSV or .RWL) or a pandas dataframe imported from dpl.readers().
    series: str
            an individual series within a chronology `data` file.
    lag:    int, default 5
            nuber of years.
   
    Returns
    -------
    data :  pandas dataframe
    
    Examples
    --------
    >>> import dplpy as dpl 
    >>> data = dpl.readers("../tests/data/csv/file.csv")
    >>> dpl.ar_func(data['series name']) -> returns residuals plus mean of best fit 
                                            from AR models with max lag of either 5 
                                            (default) or specified number
    
    References
    ----------
    .. [1] https:/opendendro.org/dplpy-man/#ar_func 
    
    """
    if isinstance(data, pd.DataFrame):
        start_df = pd.DataFrame(index=pd.Index(data.index))
        to_concat = [start_df]
        for column in data.columns:
            to_concat.append(ar_func_series(data[column], max_lag))
        res = pd.concat(to_concat, axis=1)
        return res
    elif isinstance(data, pd.Series):
        res = ar_func_series(data, max_lag)
        return res
    else:
        raise TypeError("Data argument should be either pandas dataframe or pandas series.")

# This function returns residuals plus mean of the best fit AR
# model of the data
def ar_func_series(data, max_lag):
    nullremoved_data = data.dropna()
    pars = autoreg(nullremoved_data, max_lag)
    
    y = nullremoved_data
    
    yi = fitted_values(y, pars)

    res = y[len(pars)-1:] - yi
    
    mean = np.mean(y)

    # Add mean to the residuals
    for i in range(len(res)):
        res.iloc[i] += mean

    return res

# This method selects the best AR model with a specified maximum order
# The best model is selected based on AIC value
def autoreg(data: pd.Series, max_lag=5):
    """ autoregressive (AR) models and perform functions related to AR modeling.
    
    Extended Summary
    ----------------
    Contains methods that fit series to autoregressive models and perform functions 
    related to AR modeling.
  
    NOTE: This function only accepts pandas series and dataframes as parameters. 

    Parameters
    ----------
    data : str
           a data file (.CSV or .RWL) or a pandas dataframe imported from dpl.readers().
    series : str
             an individual series within a chronology `data` file.
    lag : int, default 5
          nuber of years.
            
    Returns
    -------
    data :  pandas dataframe
        
    Examples
    --------
    >>> import dplpy as dpl 
    >>> data = dpl.readers("../tests/data/csv/file.csv")
    >>> dpl.autoreg(data['series name']) -> returns parameters of best fit AR model
                                            with maxlag of 5 (default) or other 
                                            specified number
    References
    ----------
    .. [1] https:/opendendro.org/dplpy-man/#autoreg
    
    """
    # validate data?
    if not isinstance(data, pd.Series):
        raise TypeError("Data argument should be pandas series. Received " + str(type(data)) + " instead.")

    
    max_allowable_lag = len(data.dropna())//2 - 1
    max_lag_used = max_lag if max_lag <= max_allowable_lag else max_allowable_lag

    # Need to change this to only ignore specific warnings instead of all
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        ar_data = ar_select_order(data.dropna(), max_lag_used, ic='aic', old_names=False)
    results = ar_data.model.fit()
    return results.params

# This function calculates the in-sample predicted values of a series,
# given an array containing the original data and the parameters for
# the AR model
def fitted_values(data_series, params):
    results = []
    
    for i in range((len(params)-1), len(data_series)):
        pred = params.iloc[0]
        for j in range(1, len(params)):
            pred += (params.iloc[j] * data_series.iloc[i-j])
        results.append(pred)
    return np.asarray(results)