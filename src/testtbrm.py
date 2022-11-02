# For testing tbrm.py

import numpy as np
from tbrm import tbrm

def main():
    data = np.arange(1, 101)
    print(data)

    print(np.mean(data))
    print(tbrm(data))
    
main()
