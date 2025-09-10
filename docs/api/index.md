# API Reference

This section provides comprehensive documentation for all dplPy functions and modules. The API is organized by functionality to help you quickly find the tools you need for your dendrochronological analyses.

## Core Modules

### [Input/Output Operations](io.md)
Functions for reading and writing tree ring data in various formats:

- **`readers()`** - Import ring width data from CSV and RWL files
- **`write()`** - Export data to various formats including CSV, RWL, and text files

### [Data Analysis](analysis.md)  
Functions for core dendrochronological analyses:

- **`detrend()`** - Remove biological growth trends using splines, regression, or other methods
- **`chron()`** - Build site chronologies from multiple tree ring series
- **`stats()`** - Calculate comprehensive summary statistics for ring width series

### [Visualization](visualization.md)
Functions for creating dendrochronological visualizations:

- **`plot()`** - Generate line plots, spaghetti plots, and segment plots of ring width data

### [Crossdating](crossdating.md)
Functions for crossdating analysis and quality control:

- **`xdate()`** - Perform crossdating analysis to detect dating errors and assess data quality

### [Autoregression](autoregression.md)
Functions for time series modeling:

- **`autoreg()`** - Fit autoregressive models to tree ring series

### [Utilities](utilities.md)
Helper functions and general utilities:

- **`summary()`** - Generate detailed summaries of tree ring datasets
- **`report()`** - Create reports on data characteristics and absent rings
- **`help()`** - Display interactive help system
- **`readme()`** - Open online documentation

## Main Functions

::: dplpy
    options:
      show_root_heading: false
      show_source: false
      members:
        - readers
        - write
        - stats
        - summary
        - detrend
        - chron
        - xdate
        - plot
        - report
        - autoreg

## Quick Function Reference

| Function | Module | Purpose |
|----------|---------|---------|
| `dpl.readers()` | [io](io.md) | Import data files |
| `dpl.write()` | [io](io.md) | Export data files |
| `dpl.detrend()` | [analysis](analysis.md) | Remove growth trends |
| `dpl.chron()` | [analysis](analysis.md) | Build chronologies |
| `dpl.stats()` | [analysis](analysis.md) | Calculate statistics |
| `dpl.plot()` | [visualization](visualization.md) | Create visualizations |
| `dpl.xdate()` | [crossdating](crossdating.md) | Crossdating analysis |
| `dpl.autoreg()` | [autoregression](autoregression.md) | Autoregressive modeling |
| `dpl.summary()` | [utilities](utilities.md) | Data summaries |
| `dpl.report()` | [utilities](utilities.md) | Generate reports |

## Usage Patterns

Most dplPy workflows follow this general pattern:

1. **Import data** using `dpl.readers()`
2. **Explore and summarize** using `dpl.summary()` and `dpl.stats()`
3. **Visualize raw data** using `dpl.plot()`
4. **Detrend series** using `dpl.detrend()`
5. **Check crossdating** using `dpl.xdate()`
6. **Build chronology** using `dpl.chron()`
7. **Export results** using `dpl.write()`

## Data Format Requirements

dplPy works with standardized DataFrame formats:

- **Index**: Years as pandas Index
- **Columns**: Individual tree ring series (e.g., 'CAM011', 'CAM021')
- **Values**: Ring width measurements (typically mm × 100 for RWL format)
- **Missing values**: Represented as NaN

## Getting Started

For getting started with dplPy, see the [Quick Start Guide](../quickstart.md) and [Basic Workflow Tutorial](../tutorials/basic_workflow.md).

## Support

- Use `dpl.help()` for interactive help
- Check the [GitHub repository](https://github.com/OpenDendro/dplPy) for issues and updates
- See [Installation Guide](../installation.md) for setup instructions