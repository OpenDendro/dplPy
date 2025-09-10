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

# Date: 11/17/2021 
# Author: Tyson Lee Swetnam
# Project: OpenDendro- Writers
# Description: Writers for all supported file types (*.CSV, *.RWL, and *.TXT)
# 
# example usages: 
# >>> import dplpy as dpl 
# >>> dpl.writers("./data/in_file.csv", "./data/out_file.rwl")
# >>> dpl.writers("./data/in_file.rwl", "./data/out_file.txt")
# >>> dpl.writers("./data/in_file.txt", "./data/out_file.csv")
# 
# example command line application:
# $ python src/dplpy.py writer --input ./data/in_file.csv --output ./data/out_file.rwl
#
# module uses two arguments: input file and output file 
# define `writer` module as a definition function
# input is expected to be a file path with file name and extension

from typing import Union, Optional
import pandas as pd
import numpy as np
import os

def write(data: pd.DataFrame, label: str, format: str) -> None:
    """
    Export tree ring width data to file in specified format.
    
    This function serves as the primary data export interface for dplPy, converting 
    pandas DataFrames containing dendrochronological data into various standardized 
    file formats. It handles format-specific requirements and provides a unified 
    interface for data persistence and sharing with other dendrochronological software.
    
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing tree ring width data with the following requirements:
        
        - Index must be named 'Year' and contain calendar years (int)
        - Columns represent individual tree ring series with identifiers (str)
        - Values should be numeric ring width measurements (float or int)
        - Missing values should be represented as NaN
        
    label : str
        Base filename for the output file (without extension). The function 
        automatically appends the format extension. May include directory paths 
        (relative or absolute). Special characters in filenames should be avoided 
        for cross-platform compatibility.
    format : str
        Target output file format. Case-insensitive. Supported formats:
        
        - 'csv': Comma-separated values format compatible with spreadsheet software
        - 'rwl': Tucson format ring width library format for dendrochronological software
        
        Note: 'txt' format mentioned in legacy comments is not currently implemented.
    
    Returns
    -------
    None
        This function performs file I/O operations and does not return data. 
        Success is indicated by the creation of the output file; errors will 
        raise exceptions.
    
    Raises
    ------
    ValueError
        If an unsupported format is specified.
    IOError
        If the output file cannot be created or written (permissions, disk space, etc.).
    KeyError
        If the DataFrame index is not named 'Year' or has missing year information.
    
    Notes
    -----
    **File Output Behavior:**
    
    - Files are created in the current working directory unless a path is specified in `label`
    - Existing files with the same name will be overwritten without warning
    - The function creates files with names following the pattern: '{label}.{format}'
    
    **Format-Specific Handling:**
    
    * **CSV Format**: 
      - Uses quoted headers for compatibility with various CSV readers
      - Missing values are written as 'NA' (R/statistical software compatible)
      - Maintains full precision for floating-point values
      - Compatible with pandas.read_csv() and dplPy readers()
    
    * **RWL Format**:
      - **WARNING**: Current implementation has known limitations
      - Does not fully account for varying precision standards (999 vs -9999 terminators)
      - Uses simplified line termination (always 999)
      - May not be fully compatible with all RWL-reading software
      - Suitable for basic data exchange but should be validated before archival use
    
    **Performance Considerations:**
    
    CSV format generally provides faster write performance and smaller file sizes for 
    sparse datasets. RWL format is more compact for dense datasets but requires more 
    processing time due to format complexity.
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> import dplpy as dpl
    >>> 
    >>> # Create sample dendrochronological data
    >>> years = range(1530, 1540)
    >>> series1_data = [104, 89, 103, 70, 69, 115, 101, 109, 77, 136]
    >>> series2_data = [95, 88, 92, np.nan, 78, 105, 98, 102, 81, 125]
    >>> 
    >>> data = pd.DataFrame({
    ...     'CAM011': series1_data,
    ...     'CAM021': series2_data
    ... }, index=years)
    >>> data.index.name = 'Year'
    >>> 
    >>> # Export to CSV format
    >>> dpl.write(data, 'camels_ringwidth', 'csv')
    >>> # Creates file: camels_ringwidth.csv
    >>> 
    >>> # Export to RWL format  
    >>> dpl.write(data, 'camels_ringwidth', 'rwl')
    >>> # Creates file: camels_ringwidth.rwl
    >>> 
    >>> # Export with directory path
    >>> dpl.write(data, './output/processed_data', 'csv')
    >>> # Creates file: ./output/processed_data.csv
    >>> 
    >>> # Handle format validation
    >>> try:
    ...     dpl.write(data, 'test', 'xlsx')  # Unsupported format
    ... except ValueError as e:
    ...     print(f\"Format error: {e}\")
    
    See Also
    --------
    write_csv : Internal function implementing CSV format export
    write_rwl : Internal function implementing RWL format export  
    readers : Corresponding function for reading data files
    pandas.DataFrame.to_csv : Alternative pandas method for CSV export
    
    Warnings
    --------
    The RWL export functionality is incomplete and may produce files that are not 
    fully compatible with all dendrochronological software packages. Verify output 
    compatibility before using for archival purposes or sharing with other researchers.
    """
    print("Entered function")
    filename = label + "." + format
    print(filename)
    output = open(filename, "w")
    if format == "csv":
        write_csv(data, output)
    elif format == "rwl":
        write_rwl(data, output)

    output.close()

def conv_data(data):
    """
    Convert numeric data to string representation for CSV output.
    
    This utility function provides standardized conversion of numeric ring width 
    measurements to string format suitable for CSV export. It implements consistent 
    handling of missing values using the 'NA' convention commonly used in statistical 
    software and dendrochronological applications.
    
    Parameters
    ----------
    data : float or int or numpy.nan or pandas.NA
        Input data value to be converted. Typically represents a ring width 
        measurement or missing value indicator. Accepts any numeric type or 
        pandas/numpy missing value representations.
    
    Returns
    -------
    str
        String representation of the input data following these rules:
        
        - Numeric values: Converted to string using Python's default str() function
        - Missing values (NaN, pandas.NA): Converted to 'NA' string
        - Preserves full precision for floating-point numbers
        - Integer values are represented without decimal points
    
    Notes
    -----
    **Missing Value Handling:**
    
    The function uses 'NA' (Not Available) as the missing value indicator, which is:
    
    - Compatible with R and other statistical software
    - Recognized by pandas.read_csv() with na_values parameter
    - Distinguishable from numeric zero values
    - Human-readable in plain text
    
    **Precision Preservation:**
    
    The function relies on Python's str() conversion, which preserves full precision 
    for floating-point numbers. This ensures no loss of measurement precision during 
    the export process.
    
    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> 
    >>> # Convert various numeric types
    >>> conv_data(104)
    '104'
    >>> conv_data(104.5)
    '104.5'
    >>> conv_data(0.001234567)
    '0.001234567'
    >>> 
    >>> # Handle missing values
    >>> conv_data(np.nan)
    'NA'
    >>> conv_data(pd.NA)
    'NA'
    >>> 
    >>> # Typical usage in data processing
    >>> measurements = [104.0, 89.5, np.nan, 103.2]
    >>> csv_strings = [conv_data(val) for val in measurements]
    >>> print(csv_strings)
    ['104.0', '89.5', 'NA', '103.2']
    
    See Also
    --------
    write_csv : Function that uses conv_data for CSV row generation
    numpy.isnan : Function used internally for NaN detection
    pandas.isna : More general missing value detection function
    """
    if np.isnan(data):
        return "NA"
    else:
        return str(data)

def write_csv(data, file):
    """
    Write tree ring width data to CSV format with dendrochronological conventions.
    
    This internal function implements the CSV export logic for dendrochronological data,
    generating properly formatted CSV files with quoted headers, standardized missing 
    value representation, and year-indexed rows. The output follows conventions used 
    by statistical software and is compatible with dplPy's readers() function.
    
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing tree ring width measurements with the following structure:
        
        - Index: Named 'Year' containing calendar years (int)
        - Columns: Series identifiers (str) representing individual tree/core measurements
        - Values: Ring width measurements (numeric) with NaN for missing values
        
    file : file object
        Open file handle (from open() or similar) configured for text writing.
        The file should be opened in text mode ('w' or 'wt') with appropriate 
        encoding (typically UTF-8).
    
    Returns
    -------
    None
        This function performs direct file I/O operations and returns nothing. 
        The CSV data is written to the provided file handle.
    
    Raises
    ------
    IOError
        If writing to the file handle fails (disk full, permissions, etc.).
    AttributeError
        If the DataFrame lacks required structure (no 'Year' index, etc.).
    
    Notes
    -----
    **CSV Format Specification:**
    
    The generated CSV follows these conventions:
    
    - **Header Row**: \"Year\" followed by quoted column names (\"SeriesID1\", \"SeriesID2\", ...)
    - **Data Rows**: Unquoted year values followed by comma-separated measurements
    - **Missing Values**: Represented as 'NA' (compatible with R and statistical software)
    - **Encoding**: Text output using system default encoding
    - **Line Endings**: Platform-specific line endings (\\n on Unix, \\r\\n on Windows)
    
    **Data Processing:**
    
    The function iterates through the DataFrame row by row, converting each measurement 
    using the conv_data() utility function. This ensures consistent handling of numeric 
    precision and missing values across all series.
    
    **Memory Efficiency:**
    
    The function uses row-by-row iteration rather than building the entire CSV in memory,
    making it suitable for large datasets. Each row is written immediately to the file.
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> 
    >>> # Create sample dendrochronological data
    >>> data = pd.DataFrame({
    ...     'CAM011': [104, 89, np.nan, 103],
    ...     'CAM021': [95, np.nan, 92, 98]
    ... }, index=[1530, 1531, 1532, 1533])
    >>> data.index.name = 'Year'
    >>> 
    >>> # Write to CSV file
    >>> with open('ringwidth_data.csv', 'w') as f:
    ...     write_csv(data, f)
    >>> 
    >>> # The resulting file contains:
    >>> # \"Year\",\"CAM011\",\"CAM021\"
    >>> # 1530,104,95
    >>> # 1531,89,NA  
    >>> # 1532,NA,92
    >>> # 1533,103,98
    >>> 
    >>> # Usage within dplPy write() function
    >>> with open('output.csv', 'w') as output_file:
    ...     write_csv(processed_data, output_file)
    
    See Also
    --------
    conv_data : Utility function for converting individual values to CSV strings
    write : Main export function that uses write_csv for CSV format output
    pandas.DataFrame.to_csv : Alternative pandas method for CSV export
    readers : Corresponding function for reading CSV files back into dplPy
    """
    file.write('"Year","')
    file.write('","'.join(data.columns.tolist()))
    file.write('"\n')

    for year, row in data.iterrows():
        file.write(str(year))
        file.write(",")
        file.write(",".join(map(conv_data, row)))
        file.write('\n')

def write_rwl(data, file):
    """
    Write tree ring width data to RWL (Tucson format) file.
    
    This internal function implements export to the Tucson RWL format, the standard 
    format used by dendrochronological software including COFECHA, ARSTAN, and 
    various tree-ring databases. The function processes pandas DataFrames and 
    generates fixed-width format output compatible with legacy dendrochronological 
    software.
    
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing tree ring width measurements with the following structure:
        
        - Index: Named 'Year' with calendar years (int) representing measurement years
        - Columns: Series identifiers (str) typically 6-8 characters (e.g., 'CAM011')
        - Values: Ring width measurements (numeric, typically integers representing mm × 100)
        - Missing values should be represented as NaN
        
    file : file object
        Open file handle configured for text writing. The file should be opened 
        in text mode ('w' or 'wt') and will receive the formatted RWL output.
    
    Returns
    -------
    None
        This function performs direct file I/O operations and returns nothing. 
        The RWL data is written directly to the provided file handle.
    
    Raises
    ------
    IOError
        If writing to the file handle fails (disk space, permissions, etc.).
    KeyError
        If the DataFrame structure is incompatible (missing years, invalid series).
    
    Notes
    -----
    **CRITICAL LIMITATIONS:**
    
    ⚠️ **This implementation is incomplete and has significant limitations:**
    
    - Does not account for varying precision standards (999 vs -9999 terminators)
    - Uses simplified line termination (always 999)
    - May not handle all edge cases in series data spans correctly
    - Does not implement proper RWL header generation
    - Missing value handling within series may not conform to all RWL standards
    
    **Recommended Usage:**
    
    - Use for basic data exchange and prototyping only
    - Verify output compatibility with target software before archival use
    - Consider CSV format for more reliable data exchange
    - Always validate output with RWL-reading software (COFECHA, etc.)
    
    **RWL Format Structure:**
    
    The Tucson RWL format follows these specifications:
    
    - **Line Format**: series_id + tab + start_year + tab + measurements (up to 10) + tab
    - **Series ID**: Left-aligned alphanumeric identifier (typically 6-8 characters)
    - **Start Year**: 4-digit calendar year for first measurement on the line
    - **Measurements**: Integer values separated by tabs (typically mm × 100)
    - **Line Termination**: Current implementation uses '999' terminator
    - **Missing Values**: Gaps in series represented by omitting years (problematic in current implementation)
    
    **Processing Logic:**
    
    For each series in the DataFrame:
    1. Identify the first and last valid (non-NaN) measurements
    2. Process data in chunks of up to 10 measurements per line
    3. Write series_id, start_year, and measurements separated by tabs
    4. Terminate lines when 10 measurements are reached or series ends
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> 
    >>> # Create sample data (note: typical RWL values are integers)
    >>> data = pd.DataFrame({
    ...     'CAM011': [104, 89, 103, 70, 69, 115, 101, 109, 77, 136]
    ... }, index=range(1530, 1540))
    >>> data.index.name = 'Year'
    >>> 
    >>> # Write to RWL file 
    >>> with open('output.rwl', 'w') as f:
    ...     write_rwl(data, f)
    >>> 
    >>> # Expected output format:
    >>> # CAM011    1530    104     89      103     70      69      115     101     109     77      136
    >>> 
    >>> # Handle data with missing values (problematic in current implementation)
    >>> sparse_data = pd.DataFrame({
    ...     'CAM011': [104, np.nan, 103, 70]
    ... }, index=[1530, 1531, 1532, 1533])
    >>> sparse_data.index.name = 'Year'
    >>> 
    >>> # ⚠️  Missing value handling may produce incorrect output
    >>> with open('sparse_output.rwl', 'w') as f:
    ...     write_rwl(sparse_data, f)
    
    Warnings
    --------
    **This function should be used with extreme caution due to incomplete implementation.**
    
    - Output may not be fully compatible with all RWL-reading software
    - Missing value handling is incorrect and may cause data loss
    - No precision standard handling (999 vs -9999)
    - Not suitable for archival or publication use without extensive validation
    - Consider using CSV format for reliable data exchange
    
    See Also
    --------
    write : Main export function that calls write_rwl for RWL format
    write_csv : Alternative export function with more reliable implementation
    readers : Function for reading RWL files (more mature implementation)
    
    References
    ----------
    .. [1] Grissino-Mayer, H.D. (2001). Evaluating crossdating accuracy: 
           A manual and tutorial for the computer program COFECHA. 
           Tree-Ring Research, 57(2), 205-221.
    .. [2] Holmes, R.L. (1983). Computer-assisted quality control in tree-ring 
           dating and measurement. Tree-Ring Bulletin, 43, 69-78.
    """
    # Incomplete. Doesn't account yet for varying precision standards for RWL files (lines ending in 999 vs -9999)
    for series in data.columns:
        start = data[series].first_valid_index()
        end = data[series].last_valid_index()
        i = start
        done = False
        while i <= end:
            file.write(series + "\t")
            file.write(str(i) + "\t")
            line_end = i + 10
            while i < line_end:
                # write every series[i]
                try:
                    file.write(str(data[series][i]) + "\t")
                    i += 1
                except KeyError:
                    file.write(str(999))
                    file.write("\n")
                    done = True
                    break
            if done:
                break
            else:
                file.write("\n")
        
    