import pandas as pd
import matplotlib.pyplot as plt
from readers import readers

def plot(inp, type="line"):
    if isinstance(inp, pd.DataFrame):
        series_data = inp
    elif isinstance(inp, str):
        series_data = readers(inp)
    else:
        return

    if type == "line":
        result = series_data.plot.line()
        plt.plot(series_data)
        plt.savefig("Line graph.png")
        print(result)