# Coming from dplR

dplPy is a Python companion to the R library
[dplR](https://github.com/OpenDendro/dplR), and most workflows translate almost
one-to-one. The biggest differences are ergonomic rather than conceptual:

- **Data structure.** dplR uses a `data.frame` of class `"rwl"` (years as row
  names); dplPy uses a pandas `DataFrame` indexed by year. Same idea, same
  orientation (years down, series across).
- **Naming.** R's dotted names become Python underscores
  (`read.crn` → `read_crn`, `chron.ars` → `chron_ars`).
- **Calling style.** `library(dplR); chron(rwi)` becomes
  `import dplpy as dpl; dpl.chron(rwi)`.
- **Return values.** Some functions that print in R return a dict or DataFrame in
  dplPy (for example `xdate` returns a dict of segment correlations and flags).

## Function map

### Reading & writing

| dplR | dplPy | Notes |
|---|---|---|
| `read.rwl`, `read.tucson`, `read.compact` | `readers` | Auto-detects format; `readers_url` for URLs |
| `read.crn` | `read_crn` | Single, stacked, and combined multi-site files |
| `read.ids` | `read_ids` | Site–tree–core parsing |
| `write.rwl` / `write.tucson` | `writers(..., format="rwl")` | |
| `write.crn` | `writers(..., format="crn")` | |
| `combine.rwl` | `combine_rwl` | |

### Descriptive statistics

| dplR | dplPy | Notes |
|---|---|---|
| `rwl.stats` | `stats` | Per-series mean, stdev, skew, kurtosis, gini, AR1, … |
| `rwl.report` | `report` | |
| `rwl.check` | `check` | Findings-as-data QC checker (dplR 1.8.0 `rwl.check`) |
| `summary` | `summary` | |
| `sens1`, `sens2` | `sens1`, `sens2` | Mean sensitivity (kept out of `stats`, as in dplR) |
| `gini.coef` | (column in `stats`) | Reported in the `stats` table |

### Detrending & standardization

| dplR | dplPy | Notes |
|---|---|---|
| `detrend`, `detrend.series` | `detrend` | `fit=` selects Spline / ModNegExp / ModHugershoff / Mean / AgeDepSpline |
| `rcs` | `rcs` | Curve via `fit=`, arithmetic via `method=`, stiffness via `period=`; `preset="crust"` for the CRUST curve. **Pith offset differs — see below.** |
| `ssf` | `ssf` | Signal-free; `preset="crust"` variant |
| `ads` (age-dependent spline) | `ads` | Also `detrend(fit="AgeDepSpline")` |
| `powt` | `powt` | Power transformation |

### Chronology

| dplR | dplPy | Notes |
|---|---|---|
| `chron` | `chron` | `prewhiten=True` adds the residual chronology |
| `chron.ars` | `chron_ars` | ARSTAN-style |
| `chron.stabilized` | `chron_stabilized` | Variance stabilization |
| `ar` / AR modelling | `autoreg`, `ar_func` | Yule-Walker by default, matching R's `ar()` |

### Crossdating

| dplR | dplPy | Notes |
|---|---|---|
| `corr.rwl.seg` | `xdate` | dplR-faithful default; returns a dict |
| `corr.series.seg` | `series_corr` | Single-series diagnostics |
| `ccf.series.rwl` | `series_corr` | Cross-correlation view of one series |
| `interseries.cor` | `interseries_corr` | Overall interseries correlation |
| `xdate.floater` | `xdate_floater` | Extended with Wilson (2026) t-statistics |
| `xdate.report` | `xdate_report` | COFECHA-style batch report (`xdate.report` arrived in dplR 1.8.0) |
| *(COFECHA itself has no dplR equivalent)* | `xdate(preset="COFECHA")` | Full COFECHA emulation mode |

### Signal strength, agreement & indices

| dplR | dplPy | Notes |
|---|---|---|
| `rwi.stats`, `rwi.stats.running` | `rwi_stats`, `rwi_stats_running` | |
| `sss` | `sss` | Subsample signal strength |
| `glk`, `sgc` | `glk`, `sgc` | Gleichläufigkeit / synchronous growth changes |
| `treeMean` | `tree_mean` | Average cores to tree level |
| `common.interval` | `common_interval` | |
| `bai.out`, `bai.in` | `bai_out`, `bai_in` | Basal area increment |
| `po.to.wc`, `wc.to.po` | `po_to_wc`, `wc_to_po` | Pith-offset conversion. **dplPy's `pith_offset` is a true offset (`0` = pith present); dplR's is the cambial age of the first ring (`1` = pith present), so subtract 1 from a dplR `po` table.** |
| `fill.internal.NA` | `fill_internal` | Interior-gap infilling |

### Plotting

| dplR | dplPy | Notes |
|---|---|---|
| `plot.rwl`, `plot.crn`, `spag.plot`, `seg.plot` | `plot` | `type=` selects the plot style |
| crossdating plots | `xdate(plot=True)`, `xdate_plot`, `series_corr(plot=True)` | |

### Beyond dplR

dplPy adds LiPD interchange, which dplR does not provide:

| dplPy | Notes |
|---|---|
| `to_lipd`, `from_lipd`, `lipd_metadata` | Requires the `[lipd]` extra |

## Not yet ported

dplPy targets the core standardization, chronology, and crossdating workflow.
Some specialized dplR tools — for example superposed epoch analysis, spectral and
wavelet methods, skeleton-plot crossdating, and interactive detrending — are not
yet available. dplR 1.8.0 also added `insert.ring` (insert a missing or false
ring and refit the series length), `chron.ci` (bootstrap chronology confidence
intervals), and `read.sheet` / `write.sheet` (tidy/long and delimited spreadsheet
I/O); these are not yet in dplPy. Check the [API Reference](../reference/io.md) for
the current surface, and the [project repository](https://github.com/opendendro/dplpy)
to request or contribute a port.

## A note on fidelity

Where dplPy claims to reproduce a dplR (or COFECHA, or ARSTAN) method, the
behavior is checked against those reference implementations, and deliberate
departures are documented in the relevant function's docstring. The clearest
example is crossdating: `xdate`'s A flag reproduces dplR's `corr.rwl.seg`
exactly. dplR 1.8.0 added a lag search of its own (`corr.rwl.seg(lag.max=)`) but
reports it separately and still flags on significance alone; dplPy's B (lag-shift)
flag folds that lag result into an explicit COFECHA-style flag.
