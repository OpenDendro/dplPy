# Quick Start Guide

This guide will get you up and running with dplPy in minutes, walking through a basic dendrochronological analysis workflow.

## Prerequisites

Make sure you have dplPy installed and your environment activated:

```bash
conda activate dplpy
```

If you haven't installed dplPy yet, see the [Installation Guide](installation.md).

## Your First dplPy Analysis

### 1. Import dplPy

```python
import dplpy as dpl
import pandas as pd
import matplotlib.pyplot as plt
```

### 2. Load Sample Data

dplPy comes with test data in the `tests/data/` directory:

```python
# Load ring width data from CSV format
data = dpl.readers('tests/data/csv/ca533.csv')

# Or load from RWL (Tucson) format
# data = dpl.readers('tests/data/rwl/ca533.rwl', header=True)

print(f"Data loaded: {data.shape[0]} years, {data.shape[1]} series")
print(f"Year range: {data.index.min()} - {data.index.max()}")
```

### 3. Explore Your Data

Get a quick overview of the dataset:

```python
# Basic summary statistics
summary = dpl.summary(data)

# Comprehensive dendrochronological statistics  
stats = dpl.stats(data)

# Data quality report
report = dpl.report(data)
```

### 4. Visualize the Data

Create plots to understand your tree ring data:

```python
# Basic line plot of all series
dpl.plot(data)

# Spaghetti plot (series stacked by start date)
dpl.plot(data, type="spag")

# Segment plot showing temporal coverage
dpl.plot(data, type="seg")
```

### 5. Detrend the Series

Remove biological growth trends to create ring width indices:

```python
# Detrend using default smoothing splines
rwi = dpl.detrend(data)

# Try different detrending methods
# rwi = dpl.detrend(data, fit="linear", method="residual")
# rwi = dpl.detrend(data, fit="ModNegEx", method="difference")

print(f"Detrended {len(rwi)} series")
```

### 6. Build a Chronology

Create a master chronology from detrended series:

```python
# Build chronology with robust biweight mean
chronology = dpl.chron(rwi, biweight=True, plot=True)

print("Chronology created:")
print(chronology.head())
```

### 7. Cross-dating Analysis

Verify the quality of your tree ring dating:

```python
# Perform cross-dating analysis
xdate_results = dpl.xdate(data, prewhiten=True, corr="Spearman")

# Check for dating problems
print("Cross-dating analysis complete")
```

### 8. Export Results

Save your results in various formats:

```python
# Export chronology to CSV
chronology.to_csv('my_chronology.csv')

# Export detrended data to RWL format
# (Note: This requires converting the dict back to DataFrame first)
rwi_df = pd.DataFrame(rwi)
dpl.write(rwi_df, 'detrended_series', 'rwl')
```

## Complete Example Script

Here's a complete script that demonstrates the basic workflow:

```python
import dplpy as dpl
import pandas as pd

def basic_dendro_analysis(input_file):
    """
    Perform a basic dendrochronological analysis workflow.
    """
    print("=== dplPy Basic Dendrochronological Analysis ===")
    
    # 1. Load data
    print("\\n1. Loading data...")
    data = dpl.readers(input_file)
    print(f"   Loaded: {data.shape[0]} years, {data.shape[1]} series")
    
    # 2. Explore data
    print("\\n2. Generating statistics...")
    stats = dpl.stats(data)
    
    # 3. Generate plots
    print("\\n3. Creating visualizations...")
    dpl.plot(data, type="seg")  # Segment plot
    
    # 4. Detrend series
    print("\\n4. Detrending series...")
    rwi = dpl.detrend(data, plot=False)
    print(f"   Detrended {len(rwi)} series")
    
    # 5. Build chronology
    print("\\n5. Building chronology...")
    chronology = dpl.chron(rwi, plot=False)
    
    # 6. Cross-dating
    print("\\n6. Performing cross-dating analysis...")
    xdate_results = dpl.xdate(data, prewhiten=True)
    
    print("\\n=== Analysis Complete ===")
    return data, rwi, chronology

# Run the analysis
if __name__ == "__main__":
    data, rwi, chronology = basic_dendro_analysis('tests/data/csv/ca533.csv')
```

## Understanding the Output

### Ring Width Indices (RWI)

After detrending, you'll have ring width indices that:
- Are dimensionless (typically around 1.0)
- Remove age-related growth trends
- Emphasize climate-related variation
- Are suitable for chronology building

### Chronology

The master chronology represents:
- Common environmental signal across trees
- Climate-related growth variation
- Averaged individual tree variations
- Time series suitable for climate analysis

### Cross-dating Results

Cross-dating analysis helps identify:
- Potential dating errors
- Series with poor fit to the site chronology  
- Data quality issues
- Confidence in the dating

## Common Data Formats

### CSV Format
```csv
Year,CAM011,CAM021,CAM031
1901,104,95,110
1902,89,88,98
1903,103,92,115
```

### RWL Format (Tucson)
```
CAM011  1901  104   89  103   70   69  115  101  109   77  136
CAM011  1911   95   88   92  999
CAM021  1905   98  102   81  125  130   75   85   95  105  120
```

## Key Tips

1. **Always explore your data first** with `summary()` and `stats()`
2. **Visualize before processing** using `plot()` 
3. **Try different detrending methods** for your specific data
4. **Check cross-dating results** to validate data quality
5. **Save your work frequently** using appropriate export formats

## Next Steps

Now that you've completed your first analysis:

1. **Learn more about detrending**: [Detrending Methods Tutorial](tutorials/detrending.md)
2. **Explore visualization options**: [Visualization Tutorial](tutorials/visualization.md)  
3. **Understand cross-dating**: [Cross-dating Tutorial](tutorials/crossdating.md)
4. **Read the API documentation**: [API Reference](api/index.md)
5. **Try real-world examples**: [Example Workflows](examples/workflows.md)

## Getting Help

- Use `dpl.help()` for interactive help
- Use `help(dpl.function_name)` for specific function help
- Check the [API Reference](api/index.md) for detailed documentation
- Visit our [GitHub repository](https://github.com/opendendro/dplpy) for issues and discussions