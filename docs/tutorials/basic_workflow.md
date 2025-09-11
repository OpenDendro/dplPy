# Basic Dendrochronological Workflow

This tutorial provides a comprehensive walkthrough of a standard dendrochronological analysis using dplPy, from raw ring width measurements to final chronology development.

## Overview

A typical dendrochronological analysis follows these steps:

1. **Data Import** - Load ring width measurements
2. **Data Exploration** - Understand your dataset  
3. **Quality Assessment** - Check data quality and temporal coverage
4. **Visualization** - Create plots to examine patterns
5. **Detrending** - Remove biological growth trends
6. **Cross-dating** - Verify correct dating
7. **Chronology Building** - Create site master chronology
8. **Export Results** - Save processed data and chronologies

Let's work through each step with real data.

## Dataset

For this tutorial, we'll use the Campito Mountain bristlecone pine dataset (ca533) from the International Tree-Ring Data Bank. This dataset contains ring width measurements from *Pinus longaeva* trees in California's White Mountains.

## Step 1: Data Import

```python
import dplpy as dpl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the ring width data
print("Loading Campito Mountain bristlecone pine data...")
data = dpl.readers('tests/data/rwl/ca533.rwl', header=True)

# Display basic information
print(f"Dataset dimensions: {data.shape}")
print(f"Time span: {data.index.min()} - {data.index.max()} ({data.index.max() - data.index.min() + 1} years)")
print(f"Number of series: {len(data.columns)}")
print(f"Series names: {list(data.columns)[:5]}...")  # Show first 5
```

## Step 2: Data Exploration

Understanding your data is crucial before processing:

### Basic Summary Statistics

```python
# Generate comprehensive summary
print("\\n=== DATA SUMMARY ===")
summary = dpl.summary(data)

# More detailed dendrochronological statistics
print("\\n=== DENDROCHRONOLOGICAL STATISTICS ===")
stats = dpl.stats(data)
```

### Data Quality Assessment

```python
# Generate data quality report
print("\\n=== DATA QUALITY REPORT ===")
report = dpl.report(data)

# Check for missing values
missing_data = data.isnull().sum()
print(f"\\nSeries with missing values: {(missing_data > 0).sum()}")
print(f"Total missing values: {missing_data.sum()}")

# Series length statistics
series_lengths = data.count()
print(f"\\nSeries length statistics:")
print(f"  Mean length: {series_lengths.mean():.1f} years")
print(f"  Shortest series: {series_lengths.min()} years")
print(f"  Longest series: {series_lengths.max()} years")
```

## Step 3: Data Visualization

Visual inspection helps identify patterns and potential issues:

### Temporal Coverage

```python
# Create segment plot to show temporal coverage
plt.figure(figsize=(12, 8))
dpl.plot(data, type="seg")
plt.title("Temporal Coverage - Campito Mountain Bristlecone Pine")
plt.xlabel("Year")
plt.ylabel("Series")
plt.tight_layout()
plt.show()
```

### Ring Width Patterns

```python
# Plot raw ring width series
plt.figure(figsize=(14, 6))
dpl.plot(data.iloc[:, :10], type="line")  # Plot first 10 series
plt.title("Raw Ring Width Series (First 10 Series)")
plt.ylabel("Ring Width (mm × 100)")
plt.xlabel("Year")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Spaghetti plot for pattern recognition
plt.figure(figsize=(12, 10))
dpl.plot(data, type="spag")
plt.title("Ring Width Series - Spaghetti Plot")
plt.tight_layout()
plt.show()
```

## Step 4: Statistical Analysis

### Sample Depth Through Time

```python
# Calculate sample depth (number of series per year)
sample_depth = data.count(axis=1)

plt.figure(figsize=(12, 6))
plt.plot(sample_depth.index, sample_depth.values, 'b-', linewidth=2)
plt.title("Sample Depth Through Time")
plt.xlabel("Year")
plt.ylabel("Number of Series")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Mean sample depth: {sample_depth.mean():.1f} series")
print(f"Maximum sample depth: {sample_depth.max()} series in {sample_depth.idxmax()}")
print(f"Years with >10 series: {(sample_depth >= 10).sum()}")
```

### Ring Width Distribution

```python
# Examine ring width distribution
all_values = data.values.flatten()
all_values = all_values[~np.isnan(all_values)]  # Remove NaN values

plt.figure(figsize=(10, 6))
plt.hist(all_values, bins=50, alpha=0.7, edgecolor='black')
plt.title("Distribution of Ring Width Measurements")
plt.xlabel("Ring Width (mm × 100)")
plt.ylabel("Frequency")
plt.axvline(np.mean(all_values), color='red', linestyle='--', 
           label=f'Mean: {np.mean(all_values):.1f}')
plt.axvline(np.median(all_values), color='orange', linestyle='--', 
           label=f'Median: {np.median(all_values):.1f}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

## Step 5: Detrending

Remove biological age-related trends to create ring width indices:

### Standard Spline Detrending

```python
print("\\n=== DETRENDING ANALYSIS ===")

# Detrend using smoothing splines (default method)
print("Detrending with smoothing splines...")
rwi_spline = dpl.detrend(data, fit="spline", method="residual", plot=False)

print(f"Successfully detrended {len(rwi_spline)} series")

# Convert to DataFrame for easier handling
rwi_df = pd.DataFrame(rwi_spline)
rwi_df = rwi_df.dropna(how='all')  # Remove years with no data

print(f"Ring width indices span: {rwi_df.index.min()} - {rwi_df.index.max()}")
```

### Compare Detrending Methods

```python
# Try different detrending methods on a sample series
sample_series = data['CAM011'].dropna()

# Detrend with different methods
methods = ['spline', 'linear', 'ModNegEx']
detrended_comparison = {}

plt.figure(figsize=(15, 10))

for i, method in enumerate(methods):
    plt.subplot(3, 2, i*2 + 1)
    
    # Original data with fitted curve (this would show the fit)
    plt.plot(sample_series.index, sample_series.values, 'b-', alpha=0.7, label='Original')
    plt.title(f'{method} Fit - CAM011')
    plt.ylabel('Ring Width')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Detrended result
    plt.subplot(3, 2, i*2 + 2)
    detrended_single = dpl.detrend(sample_series, fit=method, method="residual", plot=False)
    plt.plot(detrended_single.index, detrended_single.values, 'r-', linewidth=1)
    plt.title(f'{method} - Ring Width Index')
    plt.ylabel('RWI')
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Step 6: Quality Control and Cross-dating

### Cross-dating Analysis

```python
print("\\n=== CROSS-DATING ANALYSIS ===")

# Perform cross-dating analysis
print("Running cross-dating analysis...")
xdate_results = dpl.xdate(data, prewhiten=True, corr="Spearman", 
                         slide_period=50, bin_floor=100)

print("Cross-dating analysis completed")
```

### Examine Detrended Series

```python
# Plot some detrended series
plt.figure(figsize=(14, 8))

# Select a few series for comparison
sample_indices = ['CAM011', 'CAM021', 'CAM031', 'CAM041']
available_indices = [col for col in sample_indices if col in rwi_df.columns]

for i, series_name in enumerate(available_indices[:4]):
    plt.subplot(2, 2, i+1)
    series_rwi = rwi_df[series_name].dropna()
    plt.plot(series_rwi.index, series_rwi.values, 'g-', linewidth=1)
    plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
    plt.title(f'Ring Width Index - {series_name}')
    plt.ylabel('RWI')
    plt.xlabel('Year')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Summary statistics for RWI
rwi_stats = rwi_df.describe()
print("\\nRing Width Index Summary Statistics:")
print(f"Mean RWI across all series: {rwi_df.mean().mean():.3f}")
print(f"Standard deviation: {rwi_df.std().mean():.3f}")
```

## Step 7: Chronology Development

Build the site master chronology:

### Create Master Chronology

```python
print("\\n=== CHRONOLOGY DEVELOPMENT ===")

# Build chronology using biweight robust mean
print("Building master chronology...")
chronology = dpl.chron(rwi_spline, biweight=True, prewhiten=False, plot=False)

print("Chronology statistics:")
print(chronology.describe())

# Plot chronology with sample depth
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Plot chronology
ax1.plot(chronology.index, chronology['Mean RWI'], 'b-', linewidth=2, label='Master Chronology')
ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
ax1.set_ylabel('Ring Width Index')
ax1.set_title('Campito Mountain Master Chronology')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot sample depth
sample_depth_chron = rwi_df.count(axis=1)
ax2.plot(sample_depth_chron.index, sample_depth_chron.values, 'g-', linewidth=2)
ax2.set_xlabel('Year')
ax2.set_ylabel('Sample Depth')
ax2.set_title('Sample Depth Through Time')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Chronology Quality Assessment

```python
# Calculate running correlations and other quality metrics
window_size = 50
running_corr = []
years = []

for i in range(len(chronology) - window_size + 1):
    window_data = rwi_df.iloc[i:i+window_size]
    if window_data.count(axis=1).min() >= 5:  # At least 5 series
        # Calculate mean inter-series correlation for this window
        correlations = window_data.corr()
        upper_triangle = correlations.where(
            np.triu(np.ones(correlations.shape), k=1).astype(bool)
        )
        mean_corr = upper_triangle.stack().mean()
        running_corr.append(mean_corr)
        years.append(chronology.index[i + window_size//2])

# Plot running correlation
plt.figure(figsize=(12, 6))
plt.plot(years, running_corr, 'purple', linewidth=2)
plt.title(f'{window_size}-Year Running Mean Inter-Series Correlation')
plt.xlabel('Year (Window Center)')
plt.ylabel('Mean Correlation')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Overall mean inter-series correlation: {np.mean(running_corr):.3f}")
```

## Step 8: Export Results

Save your processed data and results:

```python
print("\\n=== EXPORTING RESULTS ===")

# Export chronology
chronology.to_csv('campito_chronology.csv')
print("Chronology saved as: campito_chronology.csv")

# Export ring width indices
rwi_df.to_csv('campito_rwi.csv')
print("Ring width indices saved as: campito_rwi.csv")

# Export using dplPy writers for RWL format
try:
    dpl.write(rwi_df, 'campito_rwi', 'rwl')
    print("Ring width indices saved as: campito_rwi.rwl")
except Exception as e:
    print(f"RWL export note: {e}")

# Create summary report
with open('campito_analysis_summary.txt', 'w') as f:
    f.write("CAMPITO MOUNTAIN ANALYSIS SUMMARY\\n")
    f.write("="*50 + "\\n\\n")
    f.write(f"Dataset: Campito Mountain Bristlecone Pine (ca533)\\n")
    f.write(f"Analysis date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\\n\\n")
    f.write(f"Raw Data:\\n")
    f.write(f"  Time span: {data.index.min()} - {data.index.max()} ({data.index.max() - data.index.min() + 1} years)\\n")
    f.write(f"  Number of series: {len(data.columns)}\\n")
    f.write(f"  Total measurements: {data.count().sum()}\\n\\n")
    f.write(f"Chronology:\\n")
    f.write(f"  Time span: {chronology.index.min()} - {chronology.index.max()}\\n")
    f.write(f"  Mean RWI: {chronology['Mean RWI'].mean():.3f}\\n")
    f.write(f"  RWI standard deviation: {chronology['Mean RWI'].std():.3f}\\n")
    f.write(f"  Mean sample depth: {sample_depth_chron.mean():.1f}\\n")

print("Analysis summary saved as: campito_analysis_summary.txt")
```

## Interpretation and Next Steps

### Understanding Your Results

1. **Ring Width Indices (RWI)**: Values around 1.0 indicate average growth; >1.0 indicates above-average growth; <1.0 indicates below-average growth.

2. **Master Chronology**: Represents the common environmental signal shared across trees at the site.

3. **Sample Depth**: Higher sample depth (more trees) increases chronology reliability.

4. **Inter-series Correlation**: Higher correlations indicate better cross-dating and stronger common signal.

### Quality Indicators

- **Good chronology**: Mean RWI ≈ 1.0, reasonable variance, stable correlations
- **Adequate sample depth**: >10 series for robust chronology
- **Strong signal**: Mean inter-series correlation >0.4

### Further Analysis

Consider these additional analyses:

1. **Climate-growth relationships** using correlation analysis
2. **Pointer years** analysis for extreme climate events
3. **Comparison with other regional chronologies**
4. **Spectral analysis** for periodicities
5. **Moving correlations** with climate data

## Troubleshooting

### Common Issues

1. **Low correlations**: Check cross-dating, consider different detrending
2. **Unstable chronology**: May need more series or different time period
3. **Export problems**: Check file paths and permissions

### Best Practices

1. Always examine raw data before processing
2. Try multiple detrending methods
3. Validate cross-dating results
4. Maintain adequate sample depth
5. Document your analysis steps

## Summary

This workflow demonstrates the complete process from raw tree ring measurements to a final site chronology. The key steps involve careful data exploration, appropriate detrending, quality control through cross-dating, and robust chronology development.

The resulting chronology can be used for climate reconstruction, ecological studies, or comparison with other dendrochronological datasets.

## Additional Resources

- [Detrending Methods Tutorial](detrending.md) - Detailed guide to detrending options
- [Cross-dating Tutorial](crossdating.md) - In-depth cross-dating analysis
- [API Reference](../api/index.md) - Complete function documentation
- [Real Data Workflows](../examples/workflows.md) - More analysis examples