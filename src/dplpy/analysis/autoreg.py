__copyright__ = """
   dplPy for tree ring width time series analyses
   Copyright (C) 2025  OpenDendro

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
    """
    Apply autoregressive modeling to tree ring data and return prewhitened residuals.
    
    This function fits optimal autoregressive (AR) models to tree ring width series
    to remove autocorrelation and return prewhitened residuals. The process removes
    biological persistence from growth series, emphasizing high-frequency climate
    signals and improving statistical properties for further analysis.
    
    Parameters
    ----------
    data : pandas.DataFrame or pandas.Series
        Input tree ring data. If DataFrame, applies AR modeling to each series
        independently. If Series, applies to single series.
    max_lag : int, optional
        Maximum lag (order) to consider for AR model selection, by default 5.
        The optimal lag is selected automatically using AIC criterion.
    
    Returns
    -------
    dict or pandas.Series
        If input is DataFrame, returns dict with series names as keys and
        prewhitened residuals as lists. If input is Series, returns Series
        of prewhitened residuals with mean added back.
    
    Raises
    ------
    TypeError
        If input data is neither pandas DataFrame nor pandas Series.
    
    Notes
    -----
    **Autoregressive modeling process**:
    
    1. **Model selection**: Tests AR models from order 0 to max_lag
    2. **Optimal order**: Selected using Akaike Information Criterion (AIC)
    3. **Model fitting**: Estimates AR parameters using maximum likelihood
    4. **Residual calculation**: Computes residuals = observed - predicted
    5. **Mean restoration**: Adds original series mean to residuals
    
    **Dendrochronological applications**:
    - **Prewhitening**: Removes biological autocorrelation before chronology development
    - **Climate analysis**: Enhances high-frequency climate signals
    - **Statistical modeling**: Improves assumptions for correlation analysis
    - **Disturbance detection**: Highlights non-climatic growth anomalies
    
    **Important considerations**:
    - AR modeling reduces series length by lag amount (data loss at beginning)
    - Higher max_lag allows more complex models but reduces sample size
    - Mean is restored to residuals to maintain interpretability
    - Missing values are automatically handled by dropping NaN observations
    
    Examples
    --------
    >>> import dplpy as dpl
    >>> # Load and detrend data first
    >>> data = dpl.readers('../tests/data/rwl/ca533.rwl')
    >>> rwi_data = dpl.detrend(data, plot=False)
    >>> 
    >>> # Apply AR modeling to single series
    >>> series_residuals = dpl.ar_func(rwi_data['CAM011'])
    >>> print(f"Original length: {len(rwi_data['CAM011'].dropna())}")
    >>> print(f"Residuals length: {len(series_residuals)}")
    >>> 
    >>> # Apply AR modeling to all series
    >>> all_residuals = dpl.ar_func(rwi_data)
    >>> print(f"Series processed: {len(all_residuals)}")
    >>> 
    >>> # Use different maximum lag
    >>> residuals_lag3 = dpl.ar_func(rwi_data['CAM011'], max_lag=3)
    
    See Also
    --------
    autoreg : Select optimal AR model parameters
    ar_func_series : Internal function for single series processing
    fitted_values : Calculate AR model predictions
    chron : Create prewhitened chronologies using AR residuals
    
    References
    ----------
    Cook, E. R. (1985). A time series analysis approach to tree ring standardization.
    Tree-Ring Bulletin, 45, 1-7.
    
    Box, G. E. P., Jenkins, G. M., & Reinsel, G. C. (2015). Time series analysis: 
    forecasting and control. John Wiley & Sons.
    """
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

def ar_func_series(data, max_lag):
    """
    Apply autoregressive modeling to a single tree ring series.
    
    This internal function processes individual tree ring series by fitting
    an optimal AR model and returning prewhitened residuals with the original
    mean added back for interpretability.
    
    Parameters
    ----------
    data : pandas.Series
        Tree ring width series with numeric values.
    max_lag : int
        Maximum lag to consider for AR model selection.
    
    Returns
    -------
    numpy.ndarray
        Prewhitened residuals with original series mean added back.
        Length is reduced by (optimal_lag - 1) observations due to
        lagged predictors requirement.
    
    Notes
    -----
    Processing steps:
    1. Remove missing values from input series
    2. Determine optimal AR model parameters
    3. Calculate fitted values using AR model
    4. Compute residuals (observed - fitted)
    5. Add original mean to residuals for interpretability
    
    The mean restoration ensures that residuals maintain the same
    scale as the original data while removing autocorrelation.
    """
    nullremoved_data = data.dropna()
    pars = autoreg(nullremoved_data, max_lag)
    
    y = nullremoved_data.to_numpy()
    
    yi = fitted_values(y, pars)

    res = y[len(pars)-1:] - yi
    
    mean = np.mean(y)

    # Add mean to the residuals to maintain interpretability
    for i in range(len(res)):
        res[i] += mean

    return res

def autoreg(data, max_lag=5):
    """
    Select optimal autoregressive model parameters using AIC criterion.
    
    This function determines the best-fitting AR model for tree ring data
    by testing models from order 0 to max_lag and selecting the one with
    the lowest Akaike Information Criterion (AIC) value.
    
    Parameters
    ----------
    data : pandas.Series
        Tree ring width series with missing values removed.
    max_lag : int, optional
        Maximum AR order to test, by default 5. Higher values allow more
        complex models but require more data points.
    
    Returns
    -------
    numpy.ndarray
        Array of AR model parameters [intercept, φ₁, φ₂, ..., φₚ]
        where p is the optimal lag order and φᵢ are the AR coefficients.
    
    Notes
    -----
    **Model selection process**:
    - Tests AR(0) through AR(max_lag) models
    - Calculates AIC for each model: AIC = 2k - 2ln(L)
      where k is number of parameters and L is likelihood
    - Selects model with minimum AIC (balances fit quality vs. complexity)
    
    **AIC advantages**:
    - Penalizes overfitting (more parameters increase AIC)
    - Provides objective model comparison
    - Widely accepted in time series analysis
    
    **Typical AR orders in dendrochronology**:
    - AR(1): Most common, captures year-to-year persistence
    - AR(2): Less common, may capture biennial patterns
    - AR(3+): Rare, may indicate complex biological processes
    
    Warnings are suppressed during model fitting to handle numerical
    optimization messages that don't affect results.
    
    Examples
    --------
    >>> import pandas as pd
    >>> # Example series
    >>> years = range(1900, 2000)
    >>> values = [100 + 50*np.sin(i/10) + np.random.normal(0,10) for i in range(100)]
    >>> series = pd.Series(values, index=years)
    >>> 
    >>> # Get AR parameters
    >>> params = autoreg(series, max_lag=3)
    >>> print(f"AR order selected: {len(params)-1}")
    >>> print(f"Intercept: {params[0]:.3f}")
    >>> print(f"AR coefficients: {params[1:][:3]}")
    
    See Also
    --------
    ar_func : Main function that uses these parameters for prewhitening
    fitted_values : Calculate predictions using these parameters
    statsmodels.tsa.ar_model.ar_select_order : Underlying model selection
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        ar_data = ar_select_order(data.dropna(), max_lag, ic='aic', old_names=False)
    results = ar_data.model.fit()
    return results.params

def fitted_values(data_array, params):
    """
    Calculate fitted values from autoregressive model parameters.
    
    This function computes in-sample predictions for tree ring data using
    AR model parameters. It applies the AR equation iteratively to generate
    fitted values for comparison with observed data.
    
    Parameters
    ----------
    data_array : numpy.ndarray
        1D array of original tree ring width observations.
    params : numpy.ndarray
        AR model parameters [intercept, φ₁, φ₂, ..., φₚ] where
        intercept is the constant term and φᵢ are AR coefficients.
    
    Returns
    -------
    numpy.ndarray
        Array of fitted values with length = len(data_array) - (p - 1)
        where p is the AR order (len(params) - 1).
    
    Notes
    -----
    **AR model equation**:
    ŷₜ = c + φ₁yₜ₋₁ + φ₂yₜ₋₂ + ... + φₚyₜ₋ₚ
    
    where:
    - ŷₜ is the fitted value at time t
    - c is the intercept (params[0])
    - φᵢ is the AR coefficient for lag i (params[i])
    - yₜ₋ᵢ is the observed value at lag i
    
    **Implementation details**:
    - Starts calculation at position (p-1) where p is AR order
    - Each fitted value uses previous p observed values as predictors
    - Fitted values are used to calculate residuals for prewhitening
    
    **Dendrochronological context**:
    - Fitted values represent the "expected" growth based on previous years
    - Residuals (observed - fitted) capture climate-related deviations
    - Process removes biological persistence from growth series
    
    Examples
    --------
    >>> import numpy as np
    >>> # Example data and AR(1) parameters
    >>> data = np.array([100, 105, 98, 102, 110, 95, 108])
    >>> params = np.array([20, 0.6])  # intercept=20, φ₁=0.6
    >>> 
    >>> # Calculate fitted values
    >>> fitted = fitted_values(data, params)
    >>> print(f"Fitted values: {fitted}")
    >>> print(f"Residuals: {data[1:] - fitted}")
    >>> 
    >>> # AR(2) example
    >>> params_ar2 = np.array([15, 0.5, 0.3])  # intercept=15, φ₁=0.5, φ₂=0.3
    >>> fitted_ar2 = fitted_values(data, params_ar2)
    >>> print(f"AR(2) fitted values: {fitted_ar2}")
    
    See Also
    --------
    autoreg : Function that estimates the AR parameters used here
    ar_func_series : Uses fitted values to calculate residuals
    """
    mean = np.mean(data_array)
    results = []
    
    for i in range((len(params)-1), len(data_array)):
        pred = params[0]
        for j in range(1, len(params)):
            pred += (params[j] * data_array[i-j])
        results.append(pred)
    return np.asarray(results)