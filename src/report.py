import pandas as pd
import numpy as np
from readers import readers
from stats import stats

def report(inp):
    if isinstance(inp, pd.DataFrame):
        series_data = inp
    elif isinstance(inp, str):
        series_data = readers(inp)
    else:
        return
    
    statistics = stats(series_data)

    no_of_series = series_data.shape[1]
    no_of_measurements = series_data.count().sum()
    first_year = statistics["first"].min()
    last_year = statistics["last"].max()


    print("Number of dated series:", no_of_series)
    print("Number of measurements:", no_of_measurements)
    print("Avg series length:", round(no_of_measurements/no_of_series, 4))
    print("Range:", (last_year - first_year + 1))
    print("Span:", first_year, "-", last_year)
    print("Mean (Std dev) series intercorrelation:")
    print("Mean (Std dev) AR1:")
    print("-------------")
    print("Years with absent rings listed by series")
    
