# Data Analysis Functions

This module contains core dendrochronological analysis functions for processing tree ring data, including detrending, statistical analysis, and chronology building.

## Functions

### detrend()

Remove biological growth trends from tree ring width series to create dimensionless ring width indices.

```python
dpl.detrend(data, fit="spline", method="residual", plot=True, period=None)
```

**Parameters:**

- **data** (*pandas.DataFrame or pandas.Series*): Input tree ring width data with years as index and series as columns (DataFrame) or single series (Series).
- **fit** (*str, optional*): Curve fitting method, by default "spline". Options:
  - "spline": Smoothing spline fit (flexible, default)
  - "ModNegEx": Modified negative exponential curve 
  - "Hugershoff": Hugershoff curve (biological growth model)
  - "linear": Linear regression trend
  - "horizontal": Horizontal line (mean value)
- **method** (*str, optional*): Detrending method, by default "residual". Options:
  - "residual": Calculate ratios (original/fitted), preserves variance structure
  - "difference": Calculate differences (original-fitted), preserves absolute values
- **plot** (*bool, optional*): Whether to display diagnostic plots, by default True.
- **period** (*int or None, optional*): Period parameter for spline fitting, by default None.

**Returns:**

- **dict or pandas.Series**: If input is DataFrame, returns dict with series names as keys and detrended Series as values. If input is Series, returns single detrended Series.

**Example Usage:**

```python
import dplpy as dpl

# Load data
data = dpl.readers('tests/data/rwl/ca533.rwl')

# Detrend all series with default spline method
rwi = dpl.detrend(data)

# Detrend single series with linear trend
single_rwi = dpl.detrend(data['CAM011'], fit='linear', plot=False)

# Use difference method instead of ratios
diff_rwi = dpl.detrend(data, method='difference')

# Custom spline parameters
custom_rwi = dpl.detrend(data, fit='spline', period=50)
```

**Notes:**

Detrending removes:
- Biological age-related trends in tree growth
- Stand dynamics and competition effects  
- Long-term environmental trends

The resulting ring width indices (RWI):
- Are dimensionless (typically centered around 1.0)
- Emphasize year-to-year climate variation
- Are suitable for chronology building and climate analysis

---

### chron()

Create a master chronology from ring width index data by calculating mean values across multiple tree ring series.

```python
dpl.chron(rwi_data, biweight=True, prewhiten=False, plot=True)
```

**Parameters:**

- **rwi_data** (*dict of pandas.Series*): Dictionary containing ring width index data from detrended series. Keys are series identifiers, values are Series with year indices.
- **biweight** (*bool, optional*): Whether to use Tukey's biweight robust mean, by default True. If False, uses arithmetic mean.
- **prewhiten** (*bool, optional*): Whether to prewhiten data using autoregressive modeling before chronology creation, by default False.
- **plot** (*bool, optional*): Whether to display diagnostic plots, by default True.

**Returns:**

- **pandas.DataFrame**: Chronology DataFrame with year index and columns:
  - 'Mean RWI': Mean ring width indices for each year
  - 'Mean Res': Prewhitened residuals (only if prewhiten=True)

**Example Usage:**

```python
import dplpy as dpl

# Load and detrend data
data = dpl.readers('tests/data/rwl/ca533.rwl')
rwi = dpl.detrend(data)

# Build chronology with robust biweight mean
chronology = dpl.chron(rwi, biweight=True, plot=True)

# Build chronology with prewhitening
prewhitened_chron = dpl.chron(rwi, prewhiten=True)

# Simple arithmetic mean chronology
simple_chron = dpl.chron(rwi, biweight=False, plot=False)

# Display chronology statistics
print(chronology.describe())
```

**Notes:**

The master chronology represents:
- Common environmental signal across trees at a site
- Climate-related growth variation after removing individual tree effects
- Time series suitable for climate reconstruction and comparison

Biweight robust mean:
- Reduces influence of outlier values
- More robust than arithmetic mean for dendrochronological data
- Recommended for most applications

---

### stats()

Calculate comprehensive dendrochronological statistics for tree ring width series.

```python
dpl.stats(inp)
```

**Parameters:**

- **inp** (*pandas.DataFrame or str*): Input data as DataFrame with tree ring series or file path to CSV/RWL file.

**Returns:**

- **None**: Prints comprehensive statistical summary to console.

**Calculated Statistics:**

For each series:
- **First Year**: Earliest year with data
- **Last Year**: Latest year with data  
- **Years**: Total number of years with measurements
- **Mean**: Average ring width value
- **Median**: Median ring width value
- **Standard Deviation**: Measure of variability
- **Mean Sensitivity**: Average relative change between consecutive years
- **Skewness**: Distribution asymmetry
- **Kurtosis**: Distribution tail behavior
- **AR1**: First-order autocorrelation coefficient

**Example Usage:**

```python
import dplpy as dpl

# Load data and calculate statistics
data = dpl.readers('tests/data/rwl/ca533.rwl')
dpl.stats(data)

# Calculate statistics directly from file
dpl.stats('tests/data/csv/ca533.csv')
```

**Notes:**

These statistics provide essential information for:
- Data quality assessment
- Series comparison and selection
- Understanding growth patterns
- Identifying potential dating issues

Mean sensitivity is particularly important in dendrochronology as it indicates year-to-year growth variability and climate sensitivity.

---

## Statistical Concepts

### Mean Sensitivity

Mean sensitivity measures the relative variability between consecutive ring measurements:

```
MS = (2/n-1) * Σ|2(Rt+1 - Rt)/(Rt+1 + Rt)|
```

Where:
- Rt is ring width at year t  
- n is number of years
- Higher values indicate greater year-to-year variability

**Interpretation:**
- MS > 0.3: High sensitivity (good for climate studies)
- MS 0.2-0.3: Moderate sensitivity  
- MS < 0.2: Low sensitivity (may indicate dating issues)

### Autocorrelation (AR1)

First-order autocorrelation measures the correlation between consecutive years:

```
AR1 = correlation(Rt, Rt+1)
```

**Interpretation:**
- High AR1 (>0.7): Strong biological persistence
- Moderate AR1 (0.3-0.7): Normal tree growth patterns
- Low AR1 (<0.3): May indicate dating errors or unusual growth

### Skewness and Kurtosis

- **Skewness**: Measures distribution asymmetry
  - Positive: Right-tailed (common in tree rings)
  - Negative: Left-tailed
  - Zero: Symmetric

- **Kurtosis**: Measures tail behavior
  - High: Heavy tails (outliers present)
  - Low: Light tails
  - Normal distribution has kurtosis = 3

## Quality Assessment

Use these statistics to assess data quality:

1. **Temporal Coverage**: Check first/last years and total years
2. **Central Tendencies**: Compare mean and median for distribution shape
3. **Variability**: Examine standard deviation and mean sensitivity
4. **Persistence**: Check AR1 values for biological realism
5. **Distribution**: Assess skewness and kurtosis for outliers

## Best Practices

1. **Calculate statistics before and after detrending** to understand processing effects
2. **Compare statistics across series** to identify potential issues
3. **Use statistics to guide detrending method selection**
4. **Document statistical summaries** for reproducible analysis
5. **Consider temporal stability** of statistics across different periods