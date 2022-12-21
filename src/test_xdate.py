from xdate import xdate
import dplpy as dpl

def main():
    data = dpl.readers("../tests/data/rwl/ca533.rwl")
    xdate(data)
    print("Done.")
    
main()