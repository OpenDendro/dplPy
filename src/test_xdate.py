from xdate import xdate
import dplpy as dpl

def main():
    data = dpl.readers("../tests/data/rwl/xDateRTest.rwl")
    print(xdate(data, slide_period=50, bin_floor=10))
    print("Done.")
    
main()