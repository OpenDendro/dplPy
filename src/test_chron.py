# For testing chron.py

from chron import chron
from readers import readers
from detrend import detrend

def main():
    data = readers("../tests/data/csv/ca533.csv")
    rwi_data = chron(detrend(data, fit="spline", method="residual", plot=False), prewhiten=True)
    print(rwi_data)

main()