import pandas as pd
import numpy as np
from readers import readers

#Code to calculate ar1 when statsmodels can be imported
#from statsmodels.tsa import stattools
# x = 1-D array
# Yield normalized autocorrelation function of number lags
#autocorr = stattools.acf( x )

# Get autocorrelation coefficient at lag = 1
#autocorr_coeff = autocorr[1]

def stats(inp):
    if isinstance(inp, pd.DataFrame):
        series_data = inp
    elif isinstance(inp, str):
        series_data = readers(inp)

    series_data.plot()

    stats = {"series":[], "first":[], "last":[], "year": [], "mean": [], "median":[], "stdev":[], "skew":[], "gini":[]}

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

    statistics = pd.DataFrame(stats)
    statistics.index += 1
    return statistics

def get_gini(data_array):
    # might need to work on more efficient solution
    # Mean absolute difference
    mad = np.abs(np.subtract.outer(data_array, data_array)).mean()
    # Relative mean absolute difference
    rmad = mad/np.mean(data_array)
    # Gini coefficient
    g = 0.5 * rmad
    return g

# gets skew values for each series
def get_skew(data_series):
    return (((data_series - data_series.mean()) / data_series.std()) ** 3).mean()