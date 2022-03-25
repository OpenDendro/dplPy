import pandas as pd
import matplotlib.pyplot as plt
from readers import readers
from smoothingspline import univariate_data

def plot(inp):
    if isinstance(inp, pd.DataFrame):
        series_data = inp
    elif isinstance(inp, str):
        series_data = readers(inp)
    else:
        return

    for series_name, data in series_data.items():
        univariate_data(data)
        break

    #if type == "line":
    #    print("Hello")
    #    plt.plot(series_data)
    #    plt.show(block=True)