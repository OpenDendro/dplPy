from xdate import xdate
import dplpy as dpl

def main():
    data = dpl.readers("../tests/data/rwl/ca533.rwl")
    xdate(data, slide_period=50, bin_floor=100)
    print("Done.")
    
main()