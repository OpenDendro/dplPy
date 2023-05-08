from xdate import xdate
import dplpy as dpl

def main():
    data = dpl.readers("../tests/data/rwl/ca533.rwl")
    print(xdate(data, slide_period=50, bin_floor=10, show_flags=False))
    print("Done.")
    
main()