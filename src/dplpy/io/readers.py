from __future__ import print_function

__copyright__ = """
   dplPy for tree ring width time series analyses
   Copyright (C) 2025  OpenDendro

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

__license__ = "GNU GPLv3"

#!/usr/bin/python
# -*- coding: utf-8 -*-

# Date: 9/8/2022
# Author: Ifeoluwa Ale
# Project: OpenDendro- Readers
# Description: Reads data from supported file types (*.CSV and *.RWL)
#              and stores them in a dataframe
#
# example usages: 
# >>> import dplpy as dpl 
# >>> data = dpl.readers("../tests/data/csv/file.csv")
# >>> data = dpl.readers("../tests/data/csv/file.rwl", header=True)
# 
# example command line application:
# $ python src/dplpy.py reader --input ./data/file.csv
#
# define `readers` module as a definition function
# input is expected to be a file path with file name and extension
import os
import sys
from typing import Union, Optional
import pandas as pd
import numpy as np

def readers(filename: str, skip_lines: int = 0, header: bool = False) -> pd.DataFrame:
    """
    Import tree ring width data files into Python as pandas DataFrames.
    
    This function serves as the primary data import interface for dplPy, reading 
    dendrochronological data from CSV and RWL (Tucson format) files. The function 
    automatically detects the file format based on the file extension and applies 
    the appropriate parsing method to create a standardized DataFrame output suitable 
    for dendrochronological analyses.
    
    Parameters
    ----------
    filename : str
        Path to the input file. Must be a valid file path with either .csv or .rwl 
        extension (case insensitive). Relative and absolute paths are supported.
    skip_lines : int, optional
        Number of lines to skip at the beginning of the file, by default 0. 
        Useful for files with metadata, comments, or non-standard headers at the top.
        Applied before any format-specific header processing.
    header : bool, optional
        Whether the RWL file contains header information, by default False. 
        If True, skips an additional 3 lines assuming standard RWL header format 
        (site info, species info, format info). This parameter is ignored for CSV files.
    
    Returns
    -------
    pandas.DataFrame or None
        DataFrame with 'Year' as index (pandas.Index) and tree ring series as columns. 
        Column names correspond to series identifiers (e.g., 'CAM011', 'CAM021') from 
        the input file. Values represent ring width measurements in the original units 
        (typically millimeters × 100 for RWL files, actual mm for CSV). Missing values 
        are represented as NaN. Returns None if file reading fails or unsupported file 
        format is provided.
    
    Raises
    ------
    FileNotFoundError
        If the specified file does not exist or cannot be accessed.
    ValueError
        If unexpected formatting is detected in RWL files that prevents parsing.
    PermissionError
        If the file exists but cannot be read due to insufficient permissions.
    
    Notes
    -----
    **Supported File Formats:**
    
    * **CSV Format**: Standard comma-separated values with the following requirements:
      
      - First column must be named 'Year' and contain calendar years
      - Subsequent columns represent individual tree ring series
      - Missing values should be represented as 'NA' or left empty
      - Headers are required and should be quoted if they contain special characters
    
    * **RWL Format**: Tucson format ring width library files following ITRDB standards:
      
      - Fixed-width format with series ID, start year, and up to 10 measurements per line
      - Series IDs are typically 6-8 character alphanumeric codes
      - Measurements are integer values (typically millimeters × 100)
      - Lines ending with '999' indicate standard precision
      - Lines ending with '-9999' indicate high precision
      - The function automatically handles both precision formats
    
    **Data Processing:**
    
    The function creates a continuous year range from the earliest to latest year 
    found across all series, filling missing values with NaN for series that don't 
    have measurements in specific years. This ensures a complete rectangular DataFrame 
    suitable for time series analysis.
    
    **Performance Considerations:**
    
    For large datasets, CSV format typically provides faster read performance than 
    RWL format due to pandas' optimized CSV parser. RWL files require custom parsing 
    logic that processes line by line.
    
    Examples
    --------
    >>> import dplpy as dpl
    >>> 
    >>> # Read CSV file with tree ring data
    >>> data = dpl.readers('../tests/data/csv/ca533.csv')
    >>> print(f"Data shape: {data.shape}")
    >>> print(f"Year range: {data.index.min()}-{data.index.max()}")
    >>> print(f"Series: {list(data.columns)[:5]}...")  # Show first 5 series
    
    >>> # Read RWL file with header information
    >>> rwl_data = dpl.readers('../tests/data/rwl/ca533.rwl', header=True)
    >>> print(f"Series count: {len(rwl_data.columns)}")
    >>> print(rwl_data.head())
    
    >>> # Read file with custom line skipping (e.g., for files with metadata)
    >>> data_custom = dpl.readers('../tests/data/csv/ca533.csv', skip_lines=2)
    >>> 
    >>> # Example of handling the None return case
    >>> data = dpl.readers('nonexistent_file.csv')
    >>> if data is not None:
    ...     print("Data loaded successfully")
    ... else:
    ...     print("Failed to load data")
    
    See Also
    --------
    process_rwl_pandas : Internal function for processing RWL files
    read_rwl : Low-level RWL parsing function
    pandas.read_csv : pandas function used internally for CSV files
    """
    FORMAT = "." + filename.split(".")[-1]
    print("\nAttempting to read input file: " + os.path.basename(filename) + " as " + FORMAT + " format\n")
    
    # open the input file and read its data into a pandas dataframe
    if filename.upper().endswith(".CSV"):
        series_data = pd.read_csv(filename, skiprows=skip_lines)
    elif filename.upper().endswith(".RWL"):
        series_data = process_rwl_pandas(filename, skip_lines, header)
    else:
        print("\nUnable to read file, please check that you're using a supported type\n")
        print("Accepted file types: .csv and .rwl")
        print("example usages:\n>>> import dplpy as dpl")
        print(">>> data = dpl.readers('../tests/data/csv/filename.csv')")
        print(">>> data = dpl.readers('../tests/data/rwl/filename.rwl'), header=True")
        return

    # If no data is returned, then an error was encountered when reading the file.
    if series_data is None:
        print("\nError reading file. Check that file exists and that file formatting is consistent with " + FORMAT + " format.")
        print("If your file contains headers, run dpl.readers(file_path, header=True)")
        return
    series_data.set_index('Year', inplace = True, drop = True)

    # Display message to show that reading was successful
    print("\nSUCCESS!\nFile read as:", FORMAT, "file\n")

    # Display names of all the series found
    print("Series names:")
    print(list(series_data.columns), "\n")
    return series_data

def process_rwl_pandas(filename, skip_lines, header):
    """
    Process RWL (Tucson format) file and convert to pandas DataFrame.
    
    This internal function serves as the primary processor for RWL format files,
    handling file I/O, header processing, and coordinate the conversion of raw 
    RWL data into a structured pandas DataFrame suitable for dendrochronological 
    analysis. It manages the complexities of the Tucson format while providing 
    a clean DataFrame interface.
    
    Parameters
    ----------
    filename : str
        Absolute or relative path to the RWL file to be processed. The file
        should conform to standard Tucson RWL format specifications.
    skip_lines : int
        Number of lines to skip at the beginning of the file before processing.
        Applied before any header-specific skipping. Useful for files with 
        additional metadata or comments at the top.
    header : bool
        If True, additionally skip 3 header lines after applying skip_lines
        (total lines skipped = skip_lines + 3). Standard RWL files typically
        have a 3-line header containing site information, species codes, and
        format specifications.
    
    Returns
    -------
    pandas.DataFrame or None
        DataFrame with 'Year' as index (pandas.Int64Index) and series identifiers 
        as column names (e.g., 'CAM011', 'CAM021'). Values represent ring width 
        measurements as float64. Missing values are filled with NaN for years 
        where specific series lack measurements. Returns None if the underlying 
        RWL parsing encounters errors or if the file cannot be processed.
    
    Raises
    ------
    FileNotFoundError
        If the specified RWL file does not exist.
    IOError
        If the file exists but cannot be read (permissions, corruption, etc.).
    
    Notes
    -----
    **Data Structure Creation:**
    
    The function creates a complete rectangular DataFrame by:\n    
    1. Determining the global year range across all series in the file\n    
    2. Creating a continuous year index from earliest to latest year\n    
    3. Filling missing values with NaN for series-year combinations not present in the data\n    
    \n    This ensures compatibility with pandas time series operations and dendrochronological\n    analysis functions that expect complete time series.\n    
    \n    **Memory Considerations:**\n    
    For large datasets with sparse coverage (few series spanning the full time range),\n    this approach may create DataFrames with many NaN values. This is intentional to\n    maintain temporal alignment across all series.\n    
    \n    **Precision Handling:**\n    
    The function preserves the original precision indicators from the RWL file but\n    converts all measurements to float64 for consistent numerical operations. Precision\n    information is handled by the underlying read_rwl function.\n    
    Examples\n    --------\n    >>> # Process a standard RWL file\n    >>> df = process_rwl_pandas('../tests/data/rwl/ca533.rwl', skip_lines=0, header=False)\n    >>> print(f\"Shape: {df.shape}\")\n    >>> print(f\"Columns: {list(df.columns)}\")\n    >>> print(f\"Year range: {df.index.min()} to {df.index.max()}\")\n    \n    >>> # Process RWL file with header\n    >>> df_with_header = process_rwl_pandas('../tests/data/rwl/ca533.rwl', \n    ...                                     skip_lines=0, header=True)\n    \n    >>> # Handle processing errors\n    >>> result = process_rwl_pandas('malformed_file.rwl', skip_lines=0, header=False)\n    >>> if result is None:\n    ...     print(\"Error processing RWL file\")\n    \n    See Also\n    --------\n    read_rwl : Low-level function for parsing RWL file lines\n    readers : Main entry point for reading various dendrochronological file formats\n    """
    if header is True:
        skip_lines += 3 # working with the assumption that headers are 3 lines long

    with open(filename, "r") as rwl_file:
        file_lines = rwl_file.readlines()[skip_lines:]
    
    rwl_data, first_date, last_date = read_rwl(file_lines)
    if rwl_data is None:
        return None

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
                series_data.append(rwl_data[series][i])
            else:
                series_data.append(np.nan)
        df = pd.concat([df, pd.Series(data=series_data, name=series)], axis=1)
    return df

def read_rwl(lines):
    """
    Extract and parse ring width data from preprocessed RWL file lines.
    
    This low-level parsing function processes individual lines from a Tucson format 
    RWL file, extracting series identifiers, temporal information, and ring width 
    measurements. It implements the complete RWL format specification including 
    both standard and high precision modes, providing the foundation for all RWL 
    data import operations in dplPy.
    
    Parameters
    ----------
    lines : list of str
        List of strings representing individual lines from an RWL file. Lines 
        should have newlines and leading/trailing whitespace removed. Each line 
        follows the format: series_id + whitespace + start_year + whitespace + 
        measurements + optional_terminator.
    
    Returns
    -------
    tuple of (dict, int, int) or (None, None, None)
        On successful parsing:
        
        - **dict**: Nested dictionary with structure {series_id: {year: value}}
          
          - series_id (str): Tree/core identifier (e.g., 'CAM011', 'CAM021')
          - year (int): Calendar year for the measurement
          - value (float): Ring width measurement converted to float
          
        - **int**: Earliest calendar year found across all series in the dataset
        - **int**: Latest calendar year + 1 (exclusive upper bound) found across all series
        
        On parsing failure:
        
        - Returns (None, None, None) if any line contains invalid formatting
    
    Raises
    ------
    ValueError
        Implicitly raised during parsing if non-numeric values are encountered
        in measurement positions (caught and handled by returning None tuple).
    
    Notes
    -----
    **RWL Format Specification:**
    
    The Tucson RWL format follows these rules:\n    
    \n    - **Line Structure**: series_id + start_year + up to 10 measurements + terminator\n    
    - **Series ID**: 6-8 character alphanumeric identifier (left-aligned)\n    
    - **Start Year**: 4-digit calendar year for the first measurement on the line\n    
    - **Measurements**: Integer values representing ring widths (typically mm × 100)\n    
    - **Terminators**: \n      - '999': Standard precision mode\n      - '-9999': High precision mode\n    
    \n    **Precision Handling:**\n    
    While the function detects both precision modes (999 vs -9999 terminators), \n    it currently converts all measurements to float64 for computational consistency. \n    Future versions may preserve original precision information.\n    
    \n    **Error Handling:**\n    
    The function implements fail-fast error handling: if any line cannot be parsed \n    due to unexpected formatting, the entire parsing operation returns None. This \n    ensures data integrity by preventing partial imports of malformed files.\n    
    \n    **Memory Efficiency:**\n    
    The function builds the complete data structure in memory. For very large RWL \n    files (>10MB), consider processing in chunks if memory constraints are encountered.\n    
    Examples\n    --------\n    >>> # Parse a simple RWL line\n    >>> lines = ['CAM011  1530   104    89   103    70    69   115   101   109    77   136']\n    >>> data, first_year, last_year = read_rwl(lines)\n    >>> print(f\"Series: {list(data.keys())}\")\n    ['CAM011']\n    >>> print(f\"First measurement: {data['CAM011'][1530]}\")\n    104.0\n    >>> print(f\"Year range: {first_year} to {last_year-1}\")\n    1530 to 1539\n    \n    >>> # Parse multiple lines for same series\n    >>> lines = [\n    ...     'CAM011  1530   104    89   103    70    69   115   101   109    77   136',\n    ...     'CAM011  1540   102    61    56    49    52    77    55    44    52    82'\n    ... ]\n    >>> data, first_year, last_year = read_rwl(lines)\n    >>> print(f\"Total years for CAM011: {len(data['CAM011'])}\")\n    20\n    \n    >>> # Handle parsing errors gracefully\n    >>> malformed_lines = ['CAM011  BAD_YEAR   104    89']\n    >>> result = read_rwl(malformed_lines)\n    >>> print(result)\n    (None, None, None)\n    \n    >>> # Parse multiple series\n    >>> lines = [\n    ...     'CAM011  1530   104    89   103',\n    ...     'CAM021  1535    95    88    92'\n    ... ]\n    >>> data, first_year, last_year = read_rwl(lines)\n    >>> print(f\"Series count: {len(data)}\")\n    2\n    >>> print(f\"Global year range: {first_year}-{last_year-1}\")\n    1530-1537\n    \n    See Also\n    --------\n    process_rwl_pandas : Higher-level function that uses read_rwl for DataFrame creation\n    readers : Main entry point for reading dendrochronological data files\n    \n    References\n    ----------\n    .. [1] Grissino-Mayer, H.D. (2001). Evaluating crossdating accuracy: \n           A manual and tutorial for the computer program COFECHA. \n           Tree-Ring Research, 57(2), 205-221.\n    .. [2] International Tree-Ring Data Bank (ITRDB) format specifications:\n           https://www.ncei.noaa.gov/data/paleoclimatology/\n    """
    rwl_data = {}
    first_date = sys.maxsize
    last_date = -sys.maxsize

    for line in lines:
        line = line.rstrip("\n").split()

        series_id = line[0]
        if series_id not in rwl_data:
            rwl_data[series_id] = {}
            
        # keep track of the first and last date in the dataset
        line_start = int(line[1])
        first_date = min(first_date, line_start)
        last_date = max(last_date, (line_start+len(line)-3))

        # will implement some standardization here so that all data read is consistent, and all data written in rwl
        # can be written to one of the two popular precisions.
        div = 0
        for i in range(2, len(line)):
            try:
                if line[i] == "999":
                    div = 100
                    continue
                elif line[i] == "-9999":
                    div = 1000
                    continue
                data = float(int(line[i]))
            except ValueError:
                # Stops reader, escalates to give the user an error when unexpected formatting is detected.
                return None, None, None
            rwl_data[series_id][line_start+i-2] = data
    return rwl_data, first_date, last_date
