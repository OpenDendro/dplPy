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

# Date: 5/27/2022
# Author: Ifeoluwa Ale
# Title: stats.py
# Description: Generates summary statistics for Tucson format and CSV format files
# example usage:
# >>> import dplpy as dpl 
# >>> data = dpl.readers("../tests/data/csv/file.csv")
# >>> dpl.stats(data)
#
# >>> dpl.stats("../tests/data/csv/file.csv")
# >>> Note: for file pathname inputs, only CSV and RWL file formats are accepted

# Create Summaries for Tucson (*rwl) files

# Ignore the following comments:
#Code to calculate ar1 when statsmodels can be imported
#from statsmodels.tsa import stattools
# x = 1-D array
# Yield normalized autocorrelation function of number lags
#autocorr = stattools.acf( x )

# Get autocorrelation coefficient at lag = 1
#autocorr_coeff = autocorr[1]

from typing import Union
import pandas as pd
import numpy as np
from ..io.readers import readers
from statsmodels.tsa.ar_model import AutoReg

def stats(inp: Union[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Calculate comprehensive statistics for tree ring width data series.
    
    This function computes a wide range of statistical measures for each series
    in the input data, providing essential information for dendrochronological
    analysis including temporal coverage, central tendencies, variability measures,
    and autocorrelation properties.
    
    Parameters
    ----------
    inp : pandas.DataFrame or str
        Input tree ring data. Can be either:
        - pandas.DataFrame: Tree ring data with year index and series columns
        - str: File path to CSV or RWL file that will be read automatically
    
    Returns
    -------
    pandas.DataFrame
        Statistics DataFrame with the following columns for each series:
        - 'series': Series identifier/name
        - 'first': First year with valid data
        - 'last': Last year with valid data 
        - 'year': Total span in years (last - first + 1)
        - 'mean': Arithmetic mean of ring widths (rounded to 3 decimals)
        - 'median': Median ring width (rounded to 2 decimals)
        - 'stdev': Standard deviation (rounded to 3 decimals)
        - 'skew': Skewness measure (rounded to 3 decimals)
        - 'gini': Gini coefficient of inequality (rounded to 3 decimals)
        - 'ar1': First-order autocorrelation coefficient (rounded to 3 decimals)
    
    Notes
    -----
    Statistical measures explanation:
    
    **Basic statistics**:
    - Mean and median provide measures of central tendency
    - Standard deviation quantifies variability
    - Temporal span shows series length and coverage
    
    **Advanced measures**:
    - **Skewness**: Measures asymmetry of the distribution
      - Positive values indicate right tail (few very large rings)
      - Negative values indicate left tail (few very small rings)
      - Values near 0 indicate symmetric distributions
    
    - **Gini coefficient**: Measures inequality in ring width distribution
      - Range: 0 to 1, where 0 = perfect equality, 1 = maximum inequality
      - Higher values indicate more variable growth patterns
    
    - **AR1 (autocorrelation)**: Measures year-to-year growth persistence
      - Range: -1 to +1
      - Positive values indicate growth persistence (good year followed by good year)
      - Values near 0 indicate little year-to-year dependence
      - Negative values indicate alternating growth patterns (rare)
    
    These statistics are essential for:
    - Data quality assessment
    - Site and species characterization
    - Chronology development planning
    - Statistical modeling preparation
    
    Examples
    --------
    >>> import dplpy as dpl
    >>> # Calculate statistics from DataFrame
    >>> data = dpl.readers('../tests/data/rwl/ca533.rwl')
    >>> statistics = dpl.stats(data)
    >>> print(statistics)
    >>> 
    >>> # Calculate statistics directly from file
    >>> stats_from_file = dpl.stats('../tests/data/rwl/ca533.rwl')
    >>> 
    >>> # Examine specific statistics
    >>> print(f"Mean AR1 across all series: {statistics['ar1'].mean():.3f}")
    >>> print(f"Series with highest variability: {statistics.loc[statistics['stdev'].idxmax(), 'series']}")
    
    See Also
    --------
    summary : Generate basic statistical summary using pandas describe()
    report : Generate detailed report including missing ring analysis
    get_gini : Calculate Gini coefficient for inequality measurement
    get_skew : Calculate skewness measure for distribution asymmetry
    """
    if isinstance(inp, pd.DataFrame):
        series_data = inp
    elif isinstance(inp, str):
        series_data = readers(inp)

        
    stats = {"series":[], "first":[], "last":[], "year": [], "mean": [], "median":[], "stdev":[], "skew":[], "gini":[], "ar1":[]}

    for series_name, data in series_data.items():
        stats["series"].append(series_name)
        stats["first"].append(data.first_valid_index())
        stats["last"].append(data.last_valid_index())
        stats["year"].append(stats["last"][-1] - stats["first"][-1] + 1)
        stats["mean"].append(round(data.mean(), 3))
        stats["median"].append(round(data.median(), 2))
        stats["stdev"].append(round(data.std(), 3))
        stats["skew"].append(round(get_skew(data), 3))
        stats["gini"].append(round(get_gini(data.dropna().to_numpy()), 3))
        stats["ar1"].append(round(AutoReg(data.dropna().to_numpy(), 1, old_names=False).fit().params[1], 3))


    statistics = pd.DataFrame(stats)
    statistics.index += 1
    return statistics

def get_gini(data_array):
    """
    Calculate the Gini coefficient of inequality for tree ring width data.
    
    The Gini coefficient measures the inequality of a distribution, originally
    developed for economic analysis but useful in dendrochronology to quantify
    the variability in ring width patterns within a series.
    
    Parameters
    ----------
    data_array : numpy.ndarray
        1D array of ring width measurements (NaN values should be removed).
    
    Returns
    -------
    float
        Gini coefficient ranging from 0 to 1, where:
        - 0 indicates perfect equality (all values identical)
        - 1 indicates maximum inequality (one value has everything)
        - Higher values indicate more unequal/variable growth patterns
    
    Notes
    -----
    The calculation follows these steps:
    1. **Mean Absolute Difference (MAD)**: Average absolute difference between all pairs
    2. **Relative MAD (RMAD)**: MAD divided by the mean
    3. **Gini coefficient**: Half of the RMAD
    
    Formula: G = (1/2n²μ) * Σᵢⱼ|xᵢ - xⱼ|
    where n is sample size, μ is mean, and the sum is over all pairs (i,j)
    
    In dendrochronological context:
    - Low Gini (~0.1-0.2): Consistent growth, stable conditions
    - Medium Gini (~0.2-0.4): Moderate variability, typical for many sites
    - High Gini (~0.4+): High variability, stressed conditions or disturbances
    
    **Performance note**: Current implementation has O(n²) complexity due to
    pairwise difference calculation. More efficient algorithms exist for very large datasets.
    
    Examples
    --------
    >>> import numpy as np
    >>> # Uniform values (low inequality)
    >>> uniform_data = np.array([100, 100, 100, 100])
    >>> print(get_gini(uniform_data))  # Close to 0
    >>> 
    >>> # Variable values (higher inequality) 
    >>> variable_data = np.array([50, 100, 150, 200])
    >>> print(get_gini(variable_data))  # Higher value
    
    References
    ----------
    Gini, C. (1912). Variabilità e mutabilità. Reprinted in Memorie di metodologia 
    statistica (Ed. Pizetti E, Salvemini, T). Rome: Libreria Eredi Virgilio Veschi.
    """
    # TODO: might need to work on more efficient solution for large datasets
    # Mean absolute difference
    mad = np.abs(np.subtract.outer(data_array, data_array)).mean()
    # Relative mean absolute difference
    rmad = mad/np.mean(data_array)
    # Gini coefficient
    g = 0.5 * rmad
    return g

def get_skew(data_series):
    """
    Calculate the skewness of tree ring width data series.
    
    Skewness measures the asymmetry of the probability distribution of ring widths
    within a series. This statistic helps characterize the growth pattern and
    identify potential issues with data quality or biological processes.
    
    Parameters
    ----------
    data_series : pandas.Series
        Tree ring width series with numeric values. Missing values (NaN) are 
        handled automatically by pandas operations.
    
    Returns
    -------
    float
        Skewness coefficient where:
        - 0 indicates a symmetric distribution
        - Positive values indicate right skewness (long right tail, few very large rings)
        - Negative values indicate left skewness (long left tail, few very small rings)
        - Typical range: -2 to +2 for most dendrochronological data
    
    Notes
    -----
    The calculation uses the third standardized moment:
    
    Skewness = E[((X - μ)/σ)³] = (1/n) * Σ[((xᵢ - μ)/σ)³]
    
    where μ is the mean, σ is the standard deviation, and n is sample size.
    
    **Interpretation in dendrochronology**:
    - **Positive skew (>0.5)**: Few exceptionally good growth years
      - Common in stressed environments or after disturbances
      - May indicate climate limiting factors or competition release
    
    - **Near zero skew (-0.5 to +0.5)**: Balanced growth distribution
      - Typical of sites with moderate, consistent growing conditions
      - Good for standard chronology development
    
    - **Negative skew (<-0.5)**: Few exceptionally poor growth years
      - Less common but may indicate specific disturbance patterns
      - Could suggest measurement or crossdating issues
    
    **Data quality implications**:
    - Extreme skewness may indicate measurement errors
    - Very high positive skew might suggest missing rings (many small values)
    - Unusual skewness patterns warrant further investigation
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> # Create sample series
    >>> years = range(1900, 2000)
    >>> # Normal distribution (low skew)
    >>> normal_rings = pd.Series(np.random.normal(100, 20, 100), index=years)
    >>> print(get_skew(normal_rings))  # Should be near 0
    >>> 
    >>> # Right-skewed distribution (few large rings)
    >>> skewed_rings = pd.Series(np.random.exponential(50, 100), index=years)
    >>> print(get_skew(skewed_rings))  # Should be positive
    
    See Also
    --------
    stats : Main function that calculates skewness along with other statistics
    scipy.stats.skew : Alternative implementation with more options
    """
    return (((data_series - data_series.mean()) / data_series.std()) ** 3).mean()