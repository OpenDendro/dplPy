import dplpy as dpl
from series_corr import series_corr

def main():
    data = dpl.readers("../tests/data/rwl/ca533.rwl")
    series_corr(data, 'CAM031')

main()