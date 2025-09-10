# Input/Output

Functions for reading and writing tree ring data files.

## readers

::: dplpy.io.readers.readers

## write

::: dplpy.io.writers.write

## Module Overview

This module provides functions for importing and exporting tree ring data in various formats commonly used in dendrochronology.

```python
dpl.readers(filename, skip_lines=0, header=False)
```

**Parameters:**

- **filename** (*str*): Path to the input file. Must have either .csv or .rwl extension (case insensitive).
- **skip_lines** (*int, optional*): Number of lines to skip at the beginning of the file. Default is 0.
- **header** (*bool, optional*): Whether RWL file contains header information. Default is False. Ignored for CSV files.

**Returns:**

- **pandas.DataFrame**: DataFrame with 'Year' as index and tree ring series as columns, or None if reading fails.

**Supported File Formats:**

#### CSV Format
- First column must be named 'Year' containing calendar years  
- Subsequent columns represent individual tree ring series
- Missing values as 'NA' or empty cells
- Headers required and should be quoted if containing special characters

#### RWL Format (Tucson Format)
- Standard dendrochronological format with fixed-width fields
- Each line contains: series ID, decade year, and up to 10 annual values
- Values typically in hundredths of millimeters (mm × 100)
- Missing values represented as 999 or -999

**Example Usage:**

```python
import dplpy as dpl

# Read CSV file
data_csv = dpl.readers("tests/data/csv/ca533.csv")

# Read RWL file with header
data_rwl = dpl.readers("tests/data/rwl/ca533.rwl", header=True)

# Read file skipping first 2 lines
data = dpl.readers("tests/data/csv/ca533.csv", skip_lines=2)

# Display basic information
print(f"Data shape: {data.shape}")
print(f"Year range: {data.index.min()} to {data.index.max()}")
print(f"Series names: {list(data.columns)}")
```

**Notes:**

- Function automatically detects file format based on extension
- Returns standardized DataFrame format suitable for all dplPy functions
- Handles various encoding issues and missing value representations
- Provides informative error messages for common formatting problems

---

### writers()

Export tree ring width data to various file formats.

```python
dpl.writers(input_file, output_file)
```

This is a high-level wrapper function that:
1. Reads data from the input file using `readers()`
2. Writes data to the output file using `write()` in the specified format

**Parameters:**

- **input_file** (*str*): Path to input data file (CSV or RWL format)
- **output_file** (*str*): Path for output file with desired extension

**Example Usage:**

```python
import dplpy as dpl

# Convert CSV to RWL format
dpl.writers("tests/data/csv/ca533.csv", "output/ca533.rwl")

# Convert RWL to CSV format  
dpl.writers("tests/data/rwl/ca533.rwl", "output/ca533.csv")
```

---

### write()

Low-level function to export DataFrame data to specific formats.

```python
write(data, label, format)
```

**Parameters:**

- **data** (*pandas.DataFrame*): DataFrame with tree ring data
  - Index must be named 'Year' with calendar years
  - Columns represent individual tree ring series
  - Values should be numeric ring width measurements
  - Missing values as NaN
- **label** (*str*): Base filename (without extension) 
- **format** (*str*): Output format ('csv' or 'rwl')

**Returns:**

- **None**: Creates output file; success indicated by file creation

**DataFrame Requirements:**

The input DataFrame must follow dplPy conventions:

```python
# Example DataFrame structure
import pandas as pd
import numpy as np

# Create sample data
years = range(1901, 2001)
data = pd.DataFrame({
    'CAM011': np.random.normal(100, 20, 100),  # Ring widths in hundredths mm
    'CAM021': np.random.normal(95, 25, 100),
    'CAM031': np.random.normal(110, 15, 100)
}, index=pd.Index(years, name='Year'))

# Export to different formats
write(data, "chronology_site1", "csv")  # Creates chronology_site1.csv
write(data, "chronology_site1", "rwl")  # Creates chronology_site1.rwl
```

**Output Format Details:**

#### CSV Output
- First column: 'Year' containing calendar years
- Subsequent columns: Tree ring series with original column names
- Missing values: Empty cells
- Headers: Column names from DataFrame
- Encoding: UTF-8

#### RWL Output  
- Fixed-width Tucson format
- Series IDs: Derived from DataFrame column names
- Decade format: Each line contains decade start year + up to 10 values
- Values: Original ring width measurements (assumed in hundredths mm)
- Missing values: Represented as 999

**Error Handling:**

```python
try:
    data = dpl.readers("input_file.csv")
    write(data, "output_file", "rwl")
except FileNotFoundError:
    print("Input file not found")
except ValueError as e:
    print(f"Data format error: {e}")
except IOError as e:
    print(f"Write error: {e}")
```

## Best Practices

1. **File Organization**: Keep input and output files in organized directory structures
2. **Naming Conventions**: Use descriptive filenames that include site codes or dates
3. **Data Validation**: Always check DataFrame structure before exporting
4. **Format Selection**: 
   - Use CSV for data sharing and spreadsheet compatibility
   - Use RWL for dendrochronological software compatibility
5. **Backup**: Keep original data files as backups before format conversion

## Common Issues

**Reading Problems:**
- Encoding issues with special characters
- Inconsistent missing value representations
- Malformed CSV files with irregular column counts
- RWL files with non-standard formatting

**Writing Problems:**
- Insufficient disk space or permissions
- Invalid characters in filenames
- DataFrame index not properly set as 'Year'
- Column names incompatible with target format

**Solutions:**
- Specify encoding explicitly when reading files
- Standardize missing value representations to NaN
- Validate DataFrame structure before writing
- Use cross-platform compatible filenames