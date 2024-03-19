import urllib.request
import pandas as pd
import numpy as np
from readers import read_rwl, readers

def readers_url(url, header=False, skip_lines=0):
    FORMAT = "." + url.split(".")[-1]
    file_read = urllib.request.urlopen(url).read().decode('utf-8').split('\n')

    if header is True:
        skip_lines += 3 # working with the assumption that headers are 3 lines long
    file_lines = file_read[skip_lines:]

    rwl_data, first_date, last_date = read_rwl(file_lines, skip_lines)
    if rwl_data is None:
        errorMsg = """
        Error reading file. Check that file exists and that file formatting is consistent with {format} format.
        If your file contains headers, run dpl.headers(file_path, header=True)
        """.format(format=FORMAT)
        raise ValueError(errorMsg)

    # create an array of indexes for the dataframe
    indexes = []
    for i in range(first_date, last_date):
        indexes.append(i)
    
    df = pd.DataFrame(data={"Year":indexes})

    # store raw data in pandas dataframe
    for series in rwl_data:
        series_data = []
        for i in range(first_date, last_date):
            if i in rwl_data[series]:
                series_data.append(rwl_data[series][i]/rwl_data[series]["div"])
            else:
                series_data.append(np.nan)
        df = pd.concat([df, pd.Series(data=series_data, name=series)], axis=1)
    
    df.set_index('Year', inplace = True, drop = True)

    # Display message to show that reading was successful
    print("\nSUCCESS!\nFile read as:", FORMAT, "file\n")

    # Display names of all the series found
    print("Series names:")
    print(list(df.columns), "\n")
    return df